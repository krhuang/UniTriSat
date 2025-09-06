module GPUIntersection3DFloats
    using ..LinearAlgebra, ..Printf
    using ..CUDA, ..StaticArrays, ..CUDA.Adapt
    
    # Data preparation function to convert Rational{BigInt} to Float64 for the GPU
    function prepare_simplices_for_gpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{4, Int}})
        num_simplices = length(S_indices)
        # Each 3D simplex has 4 vertices, each with 3 coordinates -> 12 Float64 values
        simplices_data = Vector{SMatrix{4, 3, Float64, 12}}(undef, num_simplices)
        for i in 1:num_simplices
            # Convert the BigInt rational matrix slice to a Float64 matrix
            cpu_matrix_f64 = Float64.(P[collect(S_indices[i]), :])
            simplices_data[i] = SMatrix{4, 3, Float64, 12}(cpu_matrix_f64)
        end
        return CuArray(simplices_data)
    end

    # Define vector operations for Float64
    @inline dot_gpu(v1, v2) = (v1[1] * v2[1]) + (v1[2] * v2[2]) + (v1[3] * v2[3])
    @inline cross_gpu(v1, v2) = SVector{3, Float64}(v1[2] * v2[3] - v1[3] * v2[2], v1[3] * v2[1] - v1[1] * v2[3], v1[1] * v2[2] - v1[2] * v2[1])
    @inline is_zero_gpu(v) = (v[1] == 0.0 && v[2] == 0.0 && v[3] == 0.0)

    # Core intersection logic using Separating Axis Theorem with Float64
    function tetrahedra_intersect_gpu(t1, t2)
        face_vertex_map = SMatrix{4, 4, Int, 16}(
            1,2,3,4,
            1,2,4,3,
            1,3,4,2,
            2,3,4,1)
        
        # Test axes normal to faces of the first tetrahedron
        for i in 1:4
            p0, p1, p2, p_fourth = t1[face_vertex_map[i,1],:], t1[face_vertex_map[i,2],:], t1[face_vertex_map[i,3],:], t1[face_vertex_map[i,4],:]
            normal = cross_gpu(p1 - p0, p2 - p0)
            if dot_gpu(normal, p_fourth - p0) > 0.0; normal = -normal; end
            if !is_zero_gpu(normal)
                proj1 = SVector(dot_gpu(t1[1,:], normal), dot_gpu(t1[2,:], normal), dot_gpu(t1[3,:], normal), dot_gpu(t1[4,:], normal))
                proj2 = SVector(dot_gpu(t2[1,:], normal), dot_gpu(t2[2,:], normal), dot_gpu(t2[3,:], normal), dot_gpu(t2[4,:], normal))
                min1, max1 = minimum(proj1), maximum(proj1)
                min2, max2 = minimum(proj2), maximum(proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end
        
        # Test axes normal to faces of the second tetrahedron
        for i in 1:4
            p0, p1, p2, p_fourth = t2[face_vertex_map[i,1],:], t2[face_vertex_map[i,2],:], t2[face_vertex_map[i,3],:], t2[face_vertex_map[i,4],:]
            normal = cross_gpu(p1 - p0, p2 - p0)
            if dot_gpu(normal, p_fourth - p0) > 0.0; normal = -normal; end
            if !is_zero_gpu(normal)
                proj1 = SVector(dot_gpu(t1[1,:], normal), dot_gpu(t1[2,:], normal), dot_gpu(t1[3,:], normal), dot_gpu(t1[4,:], normal))
                proj2 = SVector(dot_gpu(t2[1,:], normal), dot_gpu(t2[2,:], normal), dot_gpu(t2[3,:], normal), dot_gpu(t2[4,:], normal))
                min1, max1 = minimum(proj1), maximum(proj1)
                min2, max2 = minimum(proj2), maximum(proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end

        # Test axes from cross products of edges
        for i1 in 1:4, j1 in (i1+1):4
            edge1 = t1[j1,:] - t1[i1,:]
            for i2 in 1:4, j2 in (i2+1):4
                edge2 = t2[j2,:] - t2[i2,:]
                axis = cross_gpu(edge1, edge2)
                if !is_zero_gpu(axis)
                    proj1 = SVector(dot_gpu(t1[1,:], axis), dot_gpu(t1[2,:], axis), dot_gpu(t1[3,:], axis), dot_gpu(t1[4,:], axis))
                    proj2 = SVector(dot_gpu(t2[1,:], axis), dot_gpu(t2[2,:], axis), dot_gpu(t2[3,:], axis), dot_gpu(t2[4,:], axis))
                    min1, max1 = minimum(proj1), maximum(proj1)
                    min2, max2 = minimum(proj2), maximum(proj2)
                    if max1 <= min2 || max2 <= min1; return false; end
                end
            end
        end
        
        return true # No separating axis found, they intersect
    end

    # GPU kernel to check all pairs of simplices for intersection
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
            t1, t2 = simplices[i1], simplices[i2]
            if tetrahedra_intersect_gpu(t1, t2)
                old_idx = CUDA.@atomic counter[1] += Int32(1)
                res_idx = old_idx + 1
                if res_idx <= size(results_buffer, 1)
                    results_buffer[res_idx, 1] = Int32(i1)
                    results_buffer[res_idx, 2] = Int32(i2)
                end
            end
        end
        return
    end

    # Host function to manage the GPU intersection process
    function get_intersecting_pairs_gpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{4, Int}})
        num_simplices = length(S_indices)
        if num_simplices < 2; return Vector{Vector{Int}}(); end
        
        simplices_gpu = prepare_simplices_for_gpu(P, S_indices)
        total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2
        
        results_buffer_gpu = CUDA.zeros(Int32, total_pairs, 2) 
        counter_gpu = CUDA.zeros(Int32, 1)
        
        threads_per_block = 256
        blocks = cld(total_pairs, threads_per_block)
        
        @cuda threads=threads_per_block blocks=blocks intersection_kernel(simplices_gpu, num_simplices, results_buffer_gpu, counter_gpu)
        
        num_intersections = CUDA.@allowscalar counter_gpu[1]
        
        results_buffer_cpu = if num_intersections == 0
            Matrix{Int32}(undef, 0, 2)
        else
            Array(view(results_buffer_gpu, 1:num_intersections, :))
        end
        
        clauses = Vector{Vector{Int}}(undef, num_intersections)
        for i in 1:num_intersections
            clauses[i] = [-(Int(results_buffer_cpu[i, 1])), -(Int(results_buffer_cpu[i, 2]))]
        end
        
        CUDA.unsafe_free!(simplices_gpu); CUDA.unsafe_free!(results_buffer_gpu);
        CUDA.unsafe_free!(counter_gpu)
        CUDA.reclaim()
        
        return clauses
    end
end
