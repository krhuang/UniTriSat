module CUDABackend

using CUDA
using StaticArrays
using LinearAlgebra

# --- Intersection logic: GPU ---

"""
Converts simplex data to a GPU-friendly format using double-precision (`Float64`)
to ensure numerical stability and consistency with the CPU implementation.
"""
function prepare_simplices_for_gpu(P::Matrix{Int}, S_indices::Vector{NTuple{4, Int}})
    num_simplices = length(S_indices)
    simplices_data = Vector{SMatrix{4, 3, Float64, 12}}(undef, num_simplices)

    for i in 1:num_simplices
        p_indices = S_indices[i]
        # FIX: Construct the SMatrix directly from the 4x3 slice without transposing.
        # This ensures vertex coordinates (rows) are preserved correctly.
        simplices_data[i] = SMatrix{4, 3, Float64, 12}(P[collect(p_indices), :])
    end
    
    return CuArray(simplices_data)
end

"""
GPU Device Function: Checks a single pair of tetrahedra (t1, t2) for intersection.
"""
function tetrahedra_intersect_gpu(t1, t2)
    # Statically define the mapping of face vertices to the fourth, off-face vertex.
    # This is required to correctly orient the face normals outwards.
    face_vertex_map = SMatrix{4, 4, Int, 16}(
        1, 2, 3, 4,
        1, 2, 4, 3,
        1, 3, 4, 2,
        2, 3, 4, 1
    )

    # Test Case 1: 4 outward face normals of tetrahedron 1
    for i in 1:4
        p0, p1, p2 = t1[face_vertex_map[i,1],:], t1[face_vertex_map[i,2],:], t1[face_vertex_map[i,3],:]
        p_fourth = t1[face_vertex_map[i,4],:]
        normal = cross(p1 - p0, p2 - p0)
        
        # Ensure the normal points outward from the tetrahedron
        if dot(normal, p_fourth - p0) > 0; normal = -normal; end

        # Project both tetrahedra onto the axis and check for separation
        if dot(normal, normal) > 1e-18 # Epsilon for Float64
            min1 = min(dot(t1[1,:], normal), dot(t1[2,:], normal), dot(t1[3,:], normal), dot(t1[4,:], normal))
            max1 = max(dot(t1[1,:], normal), dot(t1[2,:], normal), dot(t1[3,:], normal), dot(t1[4,:], normal))
            min2 = min(dot(t2[1,:], normal), dot(t2[2,:], normal), dot(t2[3,:], normal), dot(t2[4,:], normal))
            max2 = max(dot(t2[1,:], normal), dot(t2[2,:], normal), dot(t2[3,:], normal), dot(t2[4,:], normal))
            
            # This check correctly identifies separation, including touching cases.
            if max1 <= min2 || max2 <= min1; return false; end
        end
    end

    # Test Case 2: 4 outward face normals of tetrahedron 2
    for i in 1:4
        p0, p1, p2 = t2[face_vertex_map[i,1],:], t2[face_vertex_map[i,2],:], t2[face_vertex_map[i,3],:]
        p_fourth = t2[face_vertex_map[i,4],:]
        normal = cross(p1 - p0, p2 - p0)
        
        if dot(normal, p_fourth - p0) > 0; normal = -normal; end

        if dot(normal, normal) > 1e-18
            min1 = min(dot(t1[1,:], normal), dot(t1[2,:], normal), dot(t1[3,:], normal), dot(t1[4,:], normal))
            max1 = max(dot(t1[1,:], normal), dot(t1[2,:], normal), dot(t1[3,:], normal), dot(t1[4,:], normal))
            min2 = min(dot(t2[1,:], normal), dot(t2[2,:], normal), dot(t2[3,:], normal), dot(t2[4,:], normal))
            max2 = max(dot(t2[1,:], normal), dot(t2[2,:], normal), dot(t2[3,:], normal), dot(t2[4,:], normal))

            if max1 <= min2 || max2 <= min1; return false; end
        end
    end

    # Test Case 3: Cross product of all 36 edge pairs
    for i1 in 1:4, j1 in (i1+1):4
        edge1 = t1[j1,:] - t1[i1,:]
        for i2 in 1:4, j2 in (i2+1):4
            edge2 = t2[j2,:] - t2[i2,:]
            axis = cross(edge1, edge2)

            if dot(axis, axis) > 1e-18
                min1 = min(dot(t1[1,:], axis), dot(t1[2,:], axis), dot(t1[3,:], axis), dot(t1[4,:], axis))
                max1 = max(dot(t1[1,:], axis), dot(t1[2,:], axis), dot(t1[3,:], axis), dot(t1[4,:], axis))
                min2 = min(dot(t2[1,:], axis), dot(t2[2,:], axis), dot(t2[3,:], axis), dot(t2[4,:], axis))
                max2 = max(dot(t2[1,:], axis), dot(t2[2,:], axis), dot(t2[3,:], axis), dot(t2[4,:], axis))

                if max1 <= min2 || max2 <= min1; return false; end
            end
        end
    end
    
    # If no separating axis was found after all tests, the interiors must intersect.
    return true
end

"""
The main CUDA kernel. Each thread processes one pair of simplices.
"""
function intersection_kernel(simplices, num_simplices_arg, results_buffer, counter)
    num_simplices = Int64(num_simplices_arg)
    idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2

    if idx > total_pairs; return; end

    i1 = Int64(1)
    total_pairs_before_i1 = Int64(0)
    while total_pairs_before_i1 + (num_simplices - i1) < idx
        total_pairs_before_i1 += (num_simplices - i1)
        i1 += 1
    end
    i2 = i1 + (idx - total_pairs_before_i1)

    if i1 < i2 && i2 <= num_simplices
        t1 = simplices[i1]
        t2 = simplices[i2]
        
        if tetrahedra_intersect_gpu(t1, t2)
            # atomic_add! returns the *old* 0-based index. Add 1 to get a valid 1-based Julia index.
            old_idx = CUDA.@atomic counter[1] += Int32(1)
            res_idx = old_idx + 1
            
            # The check `res_idx <= size(...)` is a safeguard against buffer overflow,
            # in case more intersections are found than expected.
            if res_idx <= size(results_buffer, 1)
                results_buffer[res_idx, 1] = Int32(i1)
                results_buffer[res_idx, 2] = Int32(i2)
            end
        end
    end
    return
end

"""
Main function to orchestrate the GPU-based intersection testing.
"""
function get_intersecting_pairs_cuda(P::Matrix{Int}, S_indices::Vector{NTuple{4, Int}})
    num_simplices = length(S_indices)
    if num_simplices < 2
        return Vector{Vector{Int}}()
    end

    simplices_gpu = prepare_simplices_for_gpu(P, S_indices)

    total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2
    results_buffer_gpu = CUDA.zeros(Int32, total_pairs, 2) 
    counter_gpu = CUDA.zeros(Int32, 1)

    threads_per_block = 256
    blocks = cld(total_pairs, threads_per_block)

    @cuda threads=threads_per_block blocks=blocks intersection_kernel(simplices_gpu, num_simplices, results_buffer_gpu, counter_gpu)

    num_intersections = CUDA.@allowscalar counter_gpu[1]

    if num_intersections == 0
        results_buffer_cpu = Matrix{Int32}(undef, 0, 2)
    else
        results_view_gpu = view(results_buffer_gpu, 1:num_intersections, :)
        results_buffer_cpu = Array(results_view_gpu)
    end
    
    clauses = Vector{Vector{Int}}(undef, num_intersections)
    for i in 1:num_intersections
        i1 = results_buffer_cpu[i, 1]
        i2 = results_buffer_cpu[i, 2]
        # --- FIX 2: Correct the clause indexing ---
        # The indices from the GPU are already the correct 1-based SAT variables.
        clauses[i] = [-(i1), -(i2)]
    end
    
    CUDA.unsafe_free!(simplices_gpu)
    CUDA.unsafe_free!(results_buffer_gpu)
    CUDA.unsafe_free!(counter_gpu)
    CUDA.reclaim()

    return clauses
end
end # end of module
