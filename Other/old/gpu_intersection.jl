module GPUIntersection
    # This module is only loaded if CUDA, StaticArrays, and CUDA.Adapt are available.
    # The `..` prefix indicates that these are from the parent module (Main).
    using ..Combinatorics, ..LinearAlgebra, ..Printf
    using ..CUDA, ..StaticArrays, ..CUDA.Adapt
    
    struct RationalGPU
        num::Int64
        den::Int64
    end
    @inline function gcd_gpu(a::Int64, b::Int64)::Int64
        while b != 0; a, b = b, a % b; end
        return a > 0 ? a : -a
    end
    @inline function create_rational_gpu(num::Int64, den::Int64)::RationalGPU
        if den == 0; return RationalGPU(num > 0 ? 1 : (num < 0 ? -1 : 0), 0); end
        common = gcd_gpu(num, den)
        d = den > 0 ? den : -den
        n = den > 0 ? num : -num
        return RationalGPU(n ÷ common, d ÷ common)
    end
    @inline Base.:+(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.den + r2.num * r1.den, r1.den * r2.den)
    @inline Base.:-(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.den - r2.num * r1.den, r1.den * r2.den)
    @inline Base.:*(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.num, r1.den * r2.den)
    @inline Base.:-(r::RationalGPU) = RationalGPU(-r.num, r.den)
    @inline Base.:(==)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den == r2.num * r1.den
    @inline Base.:(<)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den < r2.num * r1.den
    @inline Base.:(<=)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den <= r2.num * r1.den
    @inline dot_gpu(v1, v2) = (v1[1] * v2[1]) + (v1[2] * v2[2]) + (v1[3] * v2[3])
    @inline cross_gpu(v1, v2) = SVector{3, RationalGPU}(v1[2] * v2[3] - v1[3] * v2[2], v1[3] * v2[1] - v1[1] * v2[3], v1[1] * v2[2] - v1[2] * v2[1])
    @inline is_zero_gpu(v) = (v[1] == RationalGPU(0,1) && v[2] == RationalGPU(0,1) && v[3] == RationalGPU(0,1))
    @inline min_rat(a::RationalGPU, b::RationalGPU) = a <= b ? a : b
    @inline max_rat(a::RationalGPU, b::RationalGPU) = a <= b ? b : a

    function prepare_simplices_for_gpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{4, Int}})
        num_simplices = length(S_indices)
        simplices_data = Vector{SMatrix{4, 3, RationalGPU, 12}}(undef, num_simplices)
        for i in 1:num_simplices
            cpu_matrix = P[collect(S_indices[i]), :]
            gpu_matrix_data = MVector{12, RationalGPU}(undef)
            for k in 1:12
                r_big = cpu_matrix[k]
                num_big, den_big = r_big.num, r_big.den
                if num_big > typemax(Int64) || num_big < typemin(Int64) || den_big > typemax(Int64)
                    @warn "Rational number component exceeds Int64 range. GPU results may be incorrect."
                end
                gpu_matrix_data[k] = RationalGPU(Int64(num_big), Int64(den_big))
            end
            simplices_data[i] = SMatrix{4, 3, RationalGPU, 12}(gpu_matrix_data)
        end
        return CuArray(simplices_data)
    end

    function tetrahedra_intersect_gpu(t1, t2)
        face_vertex_map = SMatrix{4, 4, Int, 16}(1, 2, 3, 4, 1, 2, 4, 3, 1, 3, 4, 2, 2, 3, 4, 1)
        ZERO = RationalGPU(0, 1)
        for i in 1:4
            p0, p1, p2, p_fourth = t1[face_vertex_map[i,1],:], t1[face_vertex_map[i,2],:], t1[face_vertex_map[i,3],:], t1[face_vertex_map[i,4],:]
            normal = cross_gpu(p1 - p0, p2 - p0)
            if dot_gpu(normal, p_fourth - p0) > ZERO; normal = -normal; end
            if !is_zero_gpu(normal)
                proj1 = SVector(dot_gpu(t1[1,:], normal), dot_gpu(t1[2,:], normal), dot_gpu(t1[3,:], normal), dot_gpu(t1[4,:], normal))
                proj2 = SVector(dot_gpu(t2[1,:], normal), dot_gpu(t2[2,:], normal), dot_gpu(t2[3,:], normal), dot_gpu(t2[4,:], normal))
                min1, max1 = min_rat(min_rat(proj1[1], proj1[2]), min_rat(proj1[3], proj1[4])), max_rat(max_rat(proj1[1], proj1[2]), max_rat(proj1[3], proj1[4]))
                min2, max2 = min_rat(min_rat(proj2[1], proj2[2]), min_rat(proj2[3], proj2[4])), max_rat(max_rat(proj2[1], proj2[2]), max_rat(proj2[3], proj2[4]))
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end
        for i in 1:4
            p0, p1, p2, p_fourth = t2[face_vertex_map[i,1],:], t2[face_vertex_map[i,2],:], t2[face_vertex_map[i,3],:], t2[face_vertex_map[i,4],:]
            normal = cross_gpu(p1 - p0, p2 - p0)
            if dot_gpu(normal, p_fourth - p0) > ZERO; normal = -normal; end
            if !is_zero_gpu(normal)
                proj1 = SVector(dot_gpu(t1[1,:], normal), dot_gpu(t1[2,:], normal), dot_gpu(t1[3,:], normal), dot_gpu(t1[4,:], normal))
                proj2 = SVector(dot_gpu(t2[1,:], normal), dot_gpu(t2[2,:], normal), dot_gpu(t2[3,:], normal), dot_gpu(t2[4,:], normal))
                min1, max1 = min_rat(min_rat(proj1[1], proj1[2]), min_rat(proj1[3], proj1[4])), max_rat(max_rat(proj1[1], proj1[2]), max_rat(proj1[3], proj1[4]))
                min2, max2 = min_rat(min_rat(proj2[1], proj2[2]), min_rat(proj2[3], proj2[4])), max_rat(max_rat(proj2[1], proj2[2]), max_rat(proj2[3], proj2[4]))
                if max1 <= min2 || max2 <= min1; return false; end
            end
        end
        for i1 in 1:4, j1 in (i1+1):4
            edge1 = t1[j1,:] - t1[i1,:]
            for i2 in 1:4, j2 in (i2+1):4
                edge2 = t2[j2,:] - t2[i2,:]
                axis = cross_gpu(edge1, edge2)
                if !is_zero_gpu(axis)
                    proj1 = SVector(dot_gpu(t1[1,:], axis), dot_gpu(t1[2,:], axis), dot_gpu(t1[3,:], axis), dot_gpu(t1[4,:], axis))
                    proj2 = SVector(dot_gpu(t2[1,:], axis), dot_gpu(t2[2,:], axis), dot_gpu(t2[3,:], axis), dot_gpu(t2[4,:], axis))
                    min1, max1 = min_rat(min_rat(proj1[1], proj1[2]), min_rat(proj1[3], proj1[4])), max_rat(max_rat(proj1[1], proj1[2]), max_rat(proj1[3], proj1[4]))
                    min2, max2 = min_rat(min_rat(proj2[1], proj2[2]), min_rat(proj2[3], proj2[4])), max_rat(max_rat(proj2[1], proj2[2]), max_rat(proj2[3], proj2[4]))
                    if max1 <= min2 || max2 <= min1; return false; end
                end
            end
        end
        return true
    end

    function intersection_kernel(simplices, num_simplices_arg, results_buffer, counter)
        num_simplices = Int64(num_simplices_arg)
        idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2
        if idx > total_pairs; return; end
        i1 = Int64(1); total_pairs_before_i1 = Int64(0)
        while total_pairs_before_i1 + (num_simplices - i1) < idx
            total_pairs_before_i1 += (num_simplices - i1); i1 += 1
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
        CUDA.unsafe_free!(simplices_gpu); CUDA.unsafe_free!(results_buffer_gpu); CUDA.unsafe_free!(counter_gpu)
        CUDA.reclaim()
        return clauses
    end
end
