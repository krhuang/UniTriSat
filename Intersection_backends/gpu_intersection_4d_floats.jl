module GPUIntersection4DFloats
    using ..LinearAlgebra, ..Printf
    using ..CUDA, ..StaticArrays, ..CUDA.Adapt

    # Convert Rational{BigInt} data to Float64 for the GPU
    function prepare_simplices_for_gpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{5, Int}})
        num_simplices = length(S_indices)
        # 4D simplex: 5 vertices, 4 coordinates -> 20 Float64 values
        simplices_data = Vector{SMatrix{5, 4, Float64, 20}}(undef, num_simplices)
        for i in 1:num_simplices
            cpu_matrix_f64 = Float64.(P[collect(S_indices[i]), :])
            simplices_data[i] = SMatrix{5, 4, Float64, 20}(cpu_matrix_f64)
        end
        return CuArray(simplices_data)
    end

    # 4D vector operations for Float64
    @inline dot_gpu(v1, v2) = (v1[1] * v2[1]) + (v1[2] * v2[2]) + (v1[3] * v2[3]) + (v1[4] * v2[4])
    @inline is_zero_gpu(v) = all(x -> x == 0.0, v)
    
    @inline function det3x3_gpu(m11, m12, m13, m21, m22, m23, m31, m32, m33)
        return m11 * (m22 * m33 - m23 * m32) -
               m12 * (m21 * m33 - m23 * m31) +
               m13 * (m21 * m32 - m22 * m31)
    end
    
    @inline function normal_vector_4d(v1, v2, v3)
        return SVector{4, Float64}(
             det3x3_gpu(v1[2], v1[3], v1[4], v2[2], v2[3], v2[4], v3[2], v3[3], v3[4]),
            -det3x3_gpu(v1[1], v1[3], v1[4], v2[1], v2[3], v2[4], v3[1], v3[3], v3[4]),
             det3x3_gpu(v1[1], v1[2], v1[4], v2[1], v2[2], v2[4], v3[1], v3[2], v3[4]),
            -det3x3_gpu(v1[1], v1[2], v1[3], v2[1], v2[2], v2[3], v3[1], v3[2], v3[3])
        )
    end

    # Core 4D intersection logic using Separating Axis Theorem with Float64
    function simplices_intersect_gpu(s1, s2)
        face_map = SMatrix{5, 5, Int, 25}(
            1, 2, 3, 4, 5, 1, 2, 3, 5, 4, 1, 2, 4, 5, 3, 1, 3, 4, 5, 2, 2, 3, 4, 5, 1
        )

        # Test axes normal to the faces of the first simplex
        for i in 1:5
            p0_idx, p1_idx, p2_idx, p3_idx = face_map[i,1], face_map[i,2], face_map[i,3], face_map[i,4]
            p0, p1, p2, p3 = s1[p0_idx,:], s1[p1_idx,:], s1[p2_idx,:], s1[p3_idx,:]
            p_off_face = s1[face_map[i,5],:]
            normal = normal_vector_4d(p1 - p0, p2 - p0, p3 - p0)
            if dot_gpu(normal, p_off_face - p0) > 0.0; normal = -normal; end
            if !is_zero_gpu(normal)
                proj1 = SVector(dot_gpu(s1[1,:], normal), dot_gpu(s1[2,:], normal), dot_gpu(s1[3,:], normal), dot_gpu(s1[4,:], normal), dot_gpu(s1[5,:], normal))
                proj2 = SVector(dot_gpu(s2[1,:], normal), dot_gpu(s2[2,:], normal), dot_gpu(s2[3,:], normal), dot_gpu(s2[4,:], normal), dot_gpu(s2[5,:], normal))
                min1, max1 = minimum(proj1), maximum(proj1)
                min2, max2 = minimum(proj2), maximum(proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end

        # Test axes normal to the faces of the second simplex
        for i in 1:5
            p0_idx, p1_idx, p2_idx, p3_idx = face_map[i,1], face_map[i,2], face_map[i,3], face_map[i,4]
            p0, p1, p2, p3 = s2[p0_idx,:], s2[p1_idx,:], s2[p2_idx,:], s2[p3_idx,:]
            p_off_face = s2[face_map[i,5],:]
            normal = normal_vector_4d(p1 - p0, p2 - p0, p3 - p0)
            if dot_gpu(normal, p_off_face - p0) > 0.0; normal = -normal; end
            if !is_zero_gpu(normal)
                proj1 = SVector(dot_gpu(s1[1,:], normal), dot_gpu(s1[2,:], normal), dot_gpu(s1[3,:], normal), dot_gpu(s1[4,:], normal), dot_gpu(s1[5,:], normal))
                proj2 = SVector(dot_gpu(s2[1,:], normal), dot_gpu(s2[2,:], normal), dot_gpu(s2[3,:], normal), dot_gpu(s2[4,:], normal), dot_gpu(s2[5,:], normal))
                min1, max1 = minimum(proj1), maximum(proj1)
                min2, max2 = minimum(proj2), maximum(proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end

        edge_indices = SMatrix{10, 2, Int, 20}(1,2, 1,3, 1,4, 1,5, 2,3, 2,4, 2,5, 3,4, 3,5, 4,5)
        triangle_indices = SMatrix{10, 3, Int, 30}(1,2,3, 1,2,4, 1,2,5, 1,3,4, 1,3,5, 1,4,5, 2,3,4, 2,3,5, 2,4,5, 3,4,5)

        # Test axes: perp(edge from s1, triangle from s2)
        for i in 1:10, j in 1:10
            edge_vec = s1[edge_indices[i,2],:] - s1[edge_indices[i,1],:]
            tri_p1 = s2[triangle_indices[j,1],:]
            tri_vec1 = s2[triangle_indices[j,2],:] - tri_p1
            tri_vec2 = s2[triangle_indices[j,3],:] - tri_p1
            axis = normal_vector_4d(edge_vec, tri_vec1, tri_vec2)
            if !is_zero_gpu(axis)
                proj1 = SVector(dot_gpu(s1[1,:], axis), dot_gpu(s1[2,:], axis), dot_gpu(s1[3,:], axis), dot_gpu(s1[4,:], axis), dot_gpu(s1[5,:], axis))
                proj2 = SVector(dot_gpu(s2[1,:], axis), dot_gpu(s2[2,:], axis), dot_gpu(s2[3,:], axis), dot_gpu(s2[4,:], axis), dot_gpu(s2[5,:], axis))
                min1, max1 = minimum(proj1), maximum(proj1)
                min2, max2 = minimum(proj2), maximum(proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end

        # Test axes: perp(triangle from s1, edge from s2)
        for i in 1:10, j in 1:10
            tri_p1 = s1[triangle_indices[i,1],:]
            tri_vec1 = s1[triangle_indices[i,2],:] - tri_p1
            tri_vec2 = s1[triangle_indices[i,3],:] - tri_p1
            edge_vec = s2[edge_indices[j,2],:] - s2[edge_indices[j,1],:]
            axis = normal_vector_4d(tri_vec1, tri_vec2, edge_vec)
            if !is_zero_gpu(axis)
                proj1 = SVector(dot_gpu(s1[1,:], axis), dot_gpu(s1[2,:], axis), dot_gpu(s1[3,:], axis), dot_gpu(s1[4,:], axis), dot_gpu(s1[5,:], axis))
                proj2 = SVector(dot_gpu(s2[1,:], axis), dot_gpu(s2[2,:], axis), dot_gpu(s2[3,:], axis), dot_gpu(s2[4,:], axis), dot_gpu(s2[5,:], axis))
                min1, max1 = minimum(proj1), maximum(proj1)
                min2, max2 = minimum(proj2), maximum(proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end

        return true
    end

    # GPU kernel (identical logic to rational version, just calls the float function)
    function intersection_kernel(simplices, num_simplices_arg, results_buffer, counter)
        num_simplices = Int64(num_simplices_arg)
        idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2
        if idx > total_pairs; return; end
        
        i1 = Int64(1);
        total_pairs_before_i1 = Int64(0)
        while total_pairs_before_i1 + (num_simplices - i1) < idx
            total_pairs_before_i1 += (num_simplices - i1);
            i1 += 1
        end
        i2 = i1 + (idx - total_pairs_before_i1)
        
        if i1 < i2 && i2 <= num_simplices
            s1, s2 = simplices[i1], simplices[i2]
            if simplices_intersect_gpu(s1, s2)
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
    
    # Host function (identical logic to rational version)
    function get_intersecting_pairs_gpu_4d(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{5, Int}})
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

        CUDA.unsafe_free!(simplices_gpu); CUDA.unsafe_free!(results_buffer_gpu); CUDA.unsafe_free!(counter_gpu)
        CUDA.reclaim()

        return clauses
    end
end
