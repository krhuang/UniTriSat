module OpenCLBackend
using OpenCL
using LinearAlgebra
# --- Intersection logic: OpenCL GPU ---
# 4x3 Float64 per simplex, flattened to length 12
function prepare_simplices_for_opencl(P::Matrix{Int}, S_indices::Vector{NTuple{4,Int}})
    num_simplices = length(S_indices)
    flat = Vector{Float64}(undef, num_simplices * 12)
    @inbounds for i in 1:num_simplices
        inds = S_indices[i]
        off = (i-1)*12
        # rows are vertices (x,y,z)
        v = P[collect(inds), :]
        flat[off+1]  = v[1,1]; flat[off+2]  = v[1,2]; flat[off+3]  = v[1,3]
        flat[off+4]  = v[2,1]; flat[off+5]  = v[2,2]; flat[off+6]  = v[2,3]
        flat[off+7]  = v[3,1]; flat[off+8]  = v[3,2]; flat[off+9]  = v[3,3]
        flat[off+10] = v[4,1]; flat[off+11] = v[4,2]; flat[off+12] = v[4,3]
    end
    return flat
end

const OPENCL_SRC = raw"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline double3 my_cross(double3 a, double3 b){
    return (double3)(a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x);
}

inline double my_dot3(double3 a, double3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline double my_norm2(double3 a){
    return my_dot3(a,a);
}

inline void proj_minmax_tet(const double3 t[4], double3 axis, double *pmin, double *pmax){
    double p0 = my_dot3(t[0], axis);
    double p1 = my_dot3(t[1], axis);
    double p2 = my_dot3(t[2], axis);
    double p3 = my_dot3(t[3], axis);
    double mn = fmin(fmin(p0,p1), fmin(p2,p3));
    double mx = fmax(fmax(p0,p1), fmax(p2,p3));
    *pmin = mn; *pmax = mx;
}

inline int tetrahedra_intersect_gpu_cl(const double3 t1[4], const double3 t2[4]){
    const double eps = 1e-18;

    // Face normals (t1)
    {
        const int faces[4][4] = {{0,1,2,3},{0,1,3,2},{0,2,3,1},{1,2,3,0}};
        for(int i=0;i<4;i++){
            int a=faces[i][0], b=faces[i][1], c=faces[i][2], d=faces[i][3];
            double3 n = my_cross( (double3)(t1[b].x-t1[a].x, t1[b].y-t1[a].y, t1[b].z-t1[a].z),
                                  (double3)(t1[c].x-t1[a].x, t1[c].y-t1[a].y, t1[c].z-t1[a].z) );
            double3 vda = (double3)(t1[d].x-t1[a].x, t1[d].y-t1[a].y, t1[d].z-t1[a].z);
            if (my_dot3(n, vda) > 0.0) { n = (double3)(-n.x, -n.y, -n.z); }
            if (my_norm2(n) > eps){
                double min1,max1,min2,max2;
                proj_minmax_tet(t1, n, &min1, &max1);
                proj_minmax_tet(t2, n, &min2, &max2);
                if (max1 <= min2 || max2 <= min1) return 0;
            }
        }
    }

    // Face normals (t2)
    {
        const int faces[4][4] = {{0,1,2,3},{0,1,3,2},{0,2,3,1},{1,2,3,0}};
        for(int i=0;i<4;i++){
            int a=faces[i][0], b=faces[i][1], c=faces[i][2], d=faces[i][3];
            double3 n = my_cross( (double3)(t2[b].x-t2[a].x, t2[b].y-t2[a].y, t2[b].z-t2[a].z),
                                  (double3)(t2[c].x-t2[a].x, t2[c].y-t2[a].y, t2[c].z-t2[a].z) );
            double3 vda = (double3)(t2[d].x-t2[a].x, t2[d].y-t2[a].y, t2[d].z-t2[a].z);
            if (my_dot3(n, vda) > 0.0) { n = (double3)(-n.x, -n.y, -n.z); }
            if (my_norm2(n) > eps){
                double min1,max1,min2,max2;
                proj_minmax_tet(t1, n, &min1, &max1);
                proj_minmax_tet(t2, n, &min2, &max2);
                if (max1 <= min2 || max2 <= min1) return 0;
            }
        }
    }

    // 36 edge-edge axes
    for(int i1=0;i1<4;i1++){
        for(int j1=i1+1;j1<4;j1++){
            double3 e1 = (double3)(t1[j1].x-t1[i1].x, t1[j1].y-t1[i1].y, t1[j1].z-t1[i1].z);
            for(int i2=0;i2<4;i2++){
                for(int j2=i2+1;j2<4;j2++){
                    double3 e2 = (double3)(t2[j2].x-t2[i2].x, t2[j2].y-t2[i2].y, t2[j2].z-t2[i2].z);
                    double3 axis = my_cross(e1, e2);
                    if (my_norm2(axis) > eps){
                        double min1,max1,min2,max2;
                        proj_minmax_tet(t1, axis, &min1, &max1);
                        proj_minmax_tet(t2, axis, &min2, &max2);
                        if (max1 <= min2 || max2 <= min1) return 0;
                    }
                }
            }
        }
    }

    return 1;
}

__kernel void intersection_kernel_opencl(__global const double *simplices,  // num_simplices*12
                                         const int num_simplices,
                                         __global int *results)             // total_pairs*2
{
    long gid0 = get_global_id(0);
    long idx = gid0 + 1; // 1-based like Julia

    long total_pairs = ((long)num_simplices * (num_simplices - 1)) / 2;
    if (idx > total_pairs) return;

    // map idx -> (i1,i2) in upper triangular matrix
    long total_before = 0;
    int i1 = 1;
    while (total_before + (num_simplices - i1) < idx) {
        total_before += (num_simplices - i1);
        i1 += 1;
    }
    int i2 = i1 + (int)(idx - total_before);

    // load the two tetrahedra (4x3 each) from packed 12 doubles
    int base1 = (i1 - 1) * 12;
    int base2 = (i2 - 1) * 12;
    double3 t1[4];
    double3 t2[4];
    for (int r=0; r<4; r++) {
        t1[r].x = simplices[base1 + r*3 + 0];
        t1[r].y = simplices[base1 + r*3 + 1];
        t1[r].z = simplices[base1 + r*3 + 2];
        t2[r].x = simplices[base2 + r*3 + 0];
        t2[r].y = simplices[base2 + r*3 + 1];
        t2[r].z = simplices[base2 + r*3 + 2];
    }

    int hit = tetrahedra_intersect_gpu_cl(t1, t2);

    // write (i1,i2) if intersect, else (0,0)
    long out = (idx - 1) * 2;
    if (hit) {
        results[out + 0] = i1;
        results[out + 1] = i2;
    } else {
        results[out + 0] = 0;
        results[out + 1] = 0;
    }
}
"""

function get_intersecting_pairs_opencl(P::Matrix{Int}, S_indices::Vector{NTuple{4, Int}})
    num_simplices = length(S_indices)
    if num_simplices < 2
        return Vector{Vector{Int}}()
    end

    # Prepare data
    simplices_flat = prepare_simplices_for_opencl(P, S_indices)
    total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2

    # Select platform/device and create context/queue
    # (Default: first GPU device; falls back to CPU device if needed.)
    platforms = OpenCL.platforms()
    dev = nothing
    for plat in platforms
        devs = OpenCL.devices(plat)
        dev = findfirst(d -> OpenCL.device_type(d) == :GPU, devs)
        dev = isnothing(dev) ? findfirst(d -> OpenCL.device_type(d) == :ACCELERATOR, devs) : dev
        dev = isnothing(dev) ? findfirst(d -> OpenCL.device_type(d) == :CPU, devs) : dev
        if !isnothing(dev); dev = devs[dev]; break; end
    end
    if isnothing(dev)
        @warn "No OpenCL device found; falling back to CPU backend."
        return Vector{Vector{Int}}()
    end
    ctx = OpenCL.Context(dev)
    q   = OpenCL.Queue(ctx)

    # Build program
    prog = OpenCL.Program(ctx, source=OPENCL_SRC) |> p -> (OpenCL.build!(p); p)
    kern = OpenCL.Kernel(prog, "intersection_kernel_opencl")

    # Buffers
    buf_simplices = OpenCL.Buffer(Float64, ctx, (:r, :copy), hostbuf=simplices_flat)
    buf_results   = OpenCL.Buffer(Int32,  ctx, :w, length=2*total_pairs)

    # Kernel args
    OpenCL.set_args!(kern, buf_simplices, Int32(num_simplices), buf_results)

    # Launch: one work-item per pair
    gsize = (cl_size_t(total_pairs),)
    OpenCL.enqueue_ndrange_kernel(q, kern; global_size=gsize)
    OpenCL.finish(q)

    # Read-back
    results_vec = Vector{Int32}(undef, 2*total_pairs)
    OpenCL.enqueue_read_buffer(q, buf_results, results_vec)
    OpenCL.finish(q)

    # Compact nonzeros -> clauses [-i1, -i2]
    clauses = Vector{Vector{Int}}()
    @inbounds for k in 0:(total_pairs-1)
        i1 = results_vec[2k+1]
        i2 = results_vec[2k+2]
        if i1 != 0 && i2 != 0
            push!(clauses, [ -Int(i1), -Int(i2) ])
        end
    end

    # cleanup (buffers will be GC'd; explicit release if you like)
    return clauses
end
end # Ends the module

