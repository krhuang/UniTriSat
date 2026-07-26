#!/usr/bin/env python3
"""
tune_gpu.py — autotuner for the UniTriSat GPU intersection kernels.

The generated kernels have a handful of hardware-dependent shape parameters
(threads per block, tile size, unroll factor, AABB pre-filter, whether the face
table is staged in shared memory). The best combination depends on the SM count,
shared-memory budget, register file and integer throughput of the actual device,
so it is measured once and cached.

    python3 tune_gpu.py                     # tune every dimension, both types
    python3 tune_gpu.py --quick --dims 5    # short search, dimension 5 only
    python3 tune_gpu.py --verify            # correctness self-test vs a CPU reference
    python3 tune_gpu.py --force             # ignore cached entries and re-tune

Results land in the cache file the Julia module reads (see --out); a JSON
sidecar with every measurement is written next to it for the record.

The search itself is coordinate descent from a sensible seed: cheap, and the
objective is smooth enough in each knob that it does the job. Each candidate
costs one CUDA kernel compilation, which is why the search space is pruned in
Python before the worker is asked anything.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_project(start: Path) -> str:
    """Walk up for a Project.toml: the scripts may sit in src/ or
    src/Intersection_backends/, neither of which is the Julia environment."""
    d = start
    for _ in range(6):
        if (d / "Project.toml").is_file():
            return str(d)
        if d.parent == d:
            break
        d = d.parent
    return str(start)

# --------------------------------------------------------------------------- #
#  knob space
# --------------------------------------------------------------------------- #

FULL_GRID = {
    "threads": [64, 128, 256, 512],
    "tile": [8, 16, 32, 64],
    "unroll": [1, 2, 4],
    "aabb": [True, False],
    "stab": [True, False],
}

QUICK_GRID = {
    "threads": [128, 256],
    "tile": [16, 32],
    "unroll": [1, 2],
    "aabb": [True],
    "stab": [True],
}

SEED = {
    3: dict(threads=256, tile=32, unroll=4, aabb=True, stab=True),
    4: dict(threads=256, tile=32, unroll=2, aabb=True, stab=True),
    5: dict(threads=128, tile=16, unroll=2, aabb=True, stab=True),
    6: dict(threads=128, tile=16, unroll=1, aabb=True, stab=True),
    7: dict(threads=128, tile=16, unroll=1, aabb=True, stab=True),
}

# problem size per dimension: enough pairs that the kernel dominates the host
# call, small enough that a 6D search finishes in reasonable time
WORKLOAD = {3: dict(n=6000, w=4), 4: dict(n=4000, w=3),
            5: dict(n=2500, w=3), 6: dict(n=1200, w=2),
            7: dict(n=600, w=2)}


@dataclass(frozen=True)
class Cfg:
    threads: int
    tile: int
    unroll: int
    aabb: bool
    stab: bool

    def as_request(self) -> str:
        return (f"threads={self.threads} tile={self.tile} unroll={self.unroll} "
                f"aabb={1 if self.aabb else 0} stab={1 if self.stab else 0}")


# --------------------------------------------------------------------------- #
#  mirrors of the Julia layout, so bad configs are pruned without a round trip
# --------------------------------------------------------------------------- #

def comb(n: int, k: int) -> int:
    return math.comb(n, k)


def nfaces(D: int, k: int) -> int:
    return 1 if k == 0 else comb(D + 1, k + 1)


def reclen(D: int) -> int:
    return (D + 1) * D + 2 * D


def stride_of(D: int) -> int:
    return reclen(D) | 1


def table_len(D: int) -> int:
    W = D + 1
    return sum(nfaces(D, k) * W for k in range(D))


def shared_bytes(D: int, tsize: int, c: Cfg) -> int:
    b = 2 * c.tile * stride_of(D) * tsize
    if c.stab:
        b += table_len(D) * 4
    return b


def viable(D: int, tsize: int, c: Cfg, dev: dict) -> str | None:
    """Reason the config is unusable, or None."""
    if shared_bytes(D, tsize, c) > dev["shared"]:
        return "shared memory"
    if c.threads > dev["maxthreads"]:
        return "threads"
    if c.tile * c.tile < c.threads:
        return "tile too small for thread count"
    return None


# --------------------------------------------------------------------------- #
#  worker
# --------------------------------------------------------------------------- #

class Worker:
    def __init__(self, project: str, julia: str, verbose: bool):
        cmd = [julia]
        if project:
            cmd.append(f"--project={project}")
        cmd.append(str(HERE / "tune_worker.jl"))
        self.verbose = verbose
        self._log(f"starting worker: {shlex.join(cmd)}")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self._readline(timeout_hint="worker start-up (Julia + CUDA init)")
        if not line.startswith("ready"):
            raise SystemExit(f"worker did not come up: {line.strip()}")
        self.device = {}
        for tok in line.split()[1:]:
            k, _, v = tok.partition("=")
            self.device[k] = int(v) if v.isdigit() else v
        self._log(f"device: {self.device}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[tune] {msg}", file=sys.stderr, flush=True)

    def _readline(self, timeout_hint: str = "") -> str:
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit(f"worker died during {timeout_hint or 'a request'}")
        return line

    def ask(self, request: str) -> tuple[bool, dict]:
        self.p.stdin.write(request + "\n")
        self.p.stdin.flush()
        line = self._readline(request).strip()
        if line.startswith("ok"):
            out = {}
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                try:
                    out[k] = float(v) if "." in v else int(v)
                except ValueError:
                    out[k] = v
            return True, out
        return False, {"error": line[5:].strip() if line.startswith("fail") else line}

    def close(self) -> None:
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


# --------------------------------------------------------------------------- #
#  search
# --------------------------------------------------------------------------- #

def measure(w: Worker, D: int, tname: str, c: Cfg, reps: int,
            cache: dict, log: list) -> float:
    key = (D, tname, c)
    if key in cache:
        return cache[key]
    wl = WORKLOAD[D]
    req = (f"bench d={D} T={tname} {c.as_request()} "
           f"n={wl['n']} w={wl['w']} seed=1 reps={reps}")
    t0 = time.time()
    ok, res = w.ask(req)
    score = res.get("gpairs", 0.0) if ok else 0.0
    cache[key] = score
    log.append(dict(dim=D, type=tname, **asdict(c), ok=ok,
                    gpairs=score, ms=res.get("ms"), hits=res.get("hits"),
                    error=res.get("error"), wall_s=round(time.time() - t0, 1)))
    w._log(f"  d={D} {tname} {c} -> "
           + (f"{score:.3f} Gpair/s ({res.get('ms', 0):.1f} ms)" if ok
              else f"FAIL {res.get('error')}"))
    return score


def tune_one(w: Worker, D: int, tname: str, grid: dict, reps: int,
             rounds: int, log: list) -> tuple[Cfg | None, float]:
    tsize = 4 if tname == "Int32" else 8
    cache: dict = {}

    def ok(c: Cfg) -> bool:
        return viable(D, tsize, c, w.device) is None

    best = Cfg(**SEED.get(D, SEED[3]))
    if not ok(best):
        # shrink the seed until it fits, mirroring config_for
        while best.tile > 4 and not ok(best):
            best = replace(best, tile=best.tile // 2)
        if not ok(best) and best.stab:
            best = replace(best, stab=False)
    if not ok(best):
        print(f"  d={D} {tname}: no viable configuration "
              f"({viable(D, tsize, best, w.device)})", file=sys.stderr)
        return None, 0.0

    best_score = measure(w, D, tname, best, reps, cache, log)
    for rnd in range(rounds):
        improved = False
        for knob, values in grid.items():
            for v in values:
                cand = replace(best, **{knob: v})
                if cand == best or not ok(cand):
                    continue
                s = measure(w, D, tname, cand, reps, cache, log)
                if s > best_score * 1.01:      # 1% to ignore timing noise
                    best, best_score, improved = cand, s, True
        if not improved:
            break
    return best, best_score


# --------------------------------------------------------------------------- #
#  cache file
# --------------------------------------------------------------------------- #

def default_out(dev: dict) -> Path:
    base = os.environ.get("UNITRISAT_TUNING_DIR")
    if base is None:
        cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        base = str(Path(cache) / "UniTriSat")
    name = "".join(ch if ch.isalnum() else "_" for ch in str(dev["name"]))
    return Path(base) / f"gpu_tuning_{name}_sm{dev['sm']}.conf"


def read_existing(path: Path) -> dict:
    have = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
            if "d" in kv and "T" in kv:
                have[(int(kv["d"]), kv["T"])] = line
    return have


def write_out(path: Path, dev: dict, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# UniTriSat GPU intersection tuning — generated by tune_gpu.py",
        f"# device: {dev['name']}  sm={dev['sm']}  SMs={dev.get('sms')}  "
        f"shared={dev.get('shared')}",
        f"# generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
    ]
    for (d, t) in sorted(entries):
        lines.append(entries[(d, t)])
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="3,4,5,6,7",
                    help="comma-separated dimensions (default 3,4,5,6,7)")
    ap.add_argument("--types", default="Int32,Int64",
                    help="comma-separated element types (default Int32,Int64)")
    ap.add_argument("--quick", action="store_true",
                    help="small grid, one round — the mode used by autotune-on-first-run")
    ap.add_argument("--reps", type=int, default=3, help="timed repetitions per candidate")
    ap.add_argument("--rounds", type=int, default=2,
                    help="coordinate-descent rounds (default 2)")
    ap.add_argument("--out", default=None, help="cache file path")
    ap.add_argument("--project", default=find_project(HERE),
                    help="Julia --project path (default: nearest Project.toml above this script)")
    ap.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    ap.add_argument("--force", action="store_true", help="re-tune cached entries")
    ap.add_argument("--verify", action="store_true",
                    help="run the correctness self-test instead of tuning")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    types = [x.strip() for x in args.types.split(",") if x.strip()]
    grid = QUICK_GRID if args.quick else FULL_GRID
    rounds = 1 if args.quick else args.rounds

    w = Worker(args.project, args.julia, not args.quiet)
    try:
        if args.verify:
            bad = 0
            for D in dims:
                n = {3: 120, 4: 90, 5: 60, 6: 30, 7: 20}.get(D, 40)
                ok, res = w.ask(f"verify d={D} n={n} w={3 if D < 6 else 2} seed=7")
                if not ok:
                    print(f"  d={D}: FAILED — {res['error']}")
                    bad += 1
                    continue
                flag = "ok" if res["missing"] == 0 and res["extra"] == 0 else "MISMATCH"
                print(f"  d={D}: {res['simplices']} simplices, "
                      f"{res['expected']} intersecting pairs, "
                      f"missing={res['missing']} extra={res['extra']}  {flag}")
                bad += (res["missing"] != 0 or res["extra"] != 0)
            return 1 if bad else 0

        out = Path(args.out) if args.out else default_out(w.device)
        entries = {} if args.force else read_existing(out)
        log: list = []
        t_start = time.time()

        for D in dims:
            if D not in WORKLOAD:
                print(f"  skipping dimension {D} (no kernel)", file=sys.stderr)
                continue
            for tname in types:
                if (D, tname) in entries and not args.force:
                    if not args.quiet:
                        print(f"  d={D} {tname}: cached, skipping", file=sys.stderr)
                    continue
                cfg, score = tune_one(w, D, tname, grid, args.reps, rounds, log)
                if cfg is None:
                    continue
                entries[(D, tname)] = (
                    f"d={D} T={tname} threads={cfg.threads} tile={cfg.tile} "
                    f"unroll={cfg.unroll} aabb={1 if cfg.aabb else 0} "
                    f"stab={1 if cfg.stab else 0} gpairs={score:.6f}")
                print(f"  d={D} {tname}: best {cfg} -> {score:.3f} Gpair/s")

        write_out(out, w.device, entries)
        (out.with_suffix(".json")).write_text(json.dumps(
            dict(device=w.device, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 measurements=log), indent=1))
        print(f"\nwrote {out}  ({len(entries)} entries, "
              f"{time.time() - t_start:.0f}s, {len(log)} candidates measured)")
        return 0
    finally:
        w.close()


if __name__ == "__main__":
    sys.exit(main())
