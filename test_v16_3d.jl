# 1. Lade die Datei, die das Modul definiert
include("Triangulate.jl")

# 2. Bringe die exportierten Funktionen (wie 'triangulate') in den Geltungsbereich
using .Triangulate

# 3. Jetzt funktioniert dein Aufruf wie beabsichtigt
triangulate(
    "Polytopes/small-lattice-polytopes/data/3-polytopes/v16.txt",
    intersection_backend="cpu",
    only_unimodular=true,
    find_all=false,
    log_file="",
    terminal_output=true,
    validate=false,
    plotter="python plot_triangulation.py"
    )
