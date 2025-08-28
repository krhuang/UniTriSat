# Intersection_backends/gpu_intersection_5d.jl

module GPUIntersection5D
    using ..Combinatorics, ..LinearAlgebra, ..Printf
    using ..CUDA, ..StaticArrays, ..CUDA.Adapt

    # --- Rationale Arithmetik auf der GPU ---
    struct RationalGPU; num::Int64; den::Int64; end
    @inline function gcd_gpu(a::Int64, b::Int64)::Int64; while b != 0; a, b = b, a % b; end; return a > 0 ? a : -a; end
    @inline function create_rational_gpu(num::Int64, den::Int64)::RationalGPU
        if den == 0; return RationalGPU(num > 0 ? 1 : (num < 0 ? -1 : 0), 0); end
        common = gcd_gpu(num, den); d = den > 0 ? den : -den; n = den > 0 ? num : -num; return RationalGPU(n ÷ common, d ÷ common)
    end
    @inline Base.:+(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.den + r2.num * r1.den, r1.den * r2.den)
    @inline Base.:-(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.den - r2.num * r1.den, r1.den * r2.den)
    @inline Base.:*(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.num, r1.den * r2.den)
    @inline Base.:-(r::RationalGPU) = RationalGPU(-r.num, r.den)
    @inline Base.:(==)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den == r2.num * r1.den
    @inline Base.:(<=)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den <= r2.num * r1.den
    @inline min_rat(a::RationalGPU, b::RationalGPU) = a <= b ? a : b
    @inline max_rat(a::RationalGPU, b::RationalGPU) = a <= b ? b : a

    # --- 5D-Vektoroperationen (RationalGPU) ---
    @inline dot_gpu(v1, v2) = (v1[1]*v2[1]) + (v1[2]*v2[2]) + (v1[3]*v2[3]) + (v1[4]*v2[4]) + (v1[5]*v2[5])
    
    @inline function det3x3_gpu(m11, m12, m13, m21, m22, m23, m31, m32, m33)
        return m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31)
    end
    
    @inline function det4x4_gpu(m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44)
        d1 = det3x3_gpu(m22, m23, m24, m32, m33, m34, m42, m43, m44)
        d2 = det3x3_gpu(m21, m23, m24, m31, m33, m34, m41, m43, m44)
        d3 = det3x3_gpu(m21, m22, m24, m31, m32, m34, m41, m42, m44)
        d4 = det3x3_gpu(m21, m22, m23, m31, m32, m33, m41, m42, m43)
        return m11 * d1 - m12 * d2 + m13 * d3 - m14 * d4
    end

    @inline function normal_vector_5d(v1, v2, v3, v4)
        M = SMatrix{4, 5, RationalGPU, 20}(
            v1[1], v2[1], v3[1], v4[1], v1[2], v2[2], v3[2], v4[2], v1[3], v2[3], v3[3], v4[3],
            v1[4], v2[4], v3[4], v4[4], v1[5], v2[5], v3[5], v4[5]
        )
        c1 =  det4x4_gpu(M[1,2],M[1,3],M[1,4],M[1,5], M[2,2],M[2,3],M[2,4],M[2,5], M[3,2],M[3,3],M[3,4],M[3,5], M[4,2],M[4,3],M[4,4],M[4,5])
        c2 = -det4x4_gpu(M[1,1],M[1,3],M[1,4],M[1,5], M[2,1],M[2,3],M[2,4],M[2,5], M[3,1],M[3,3],M[3,4],M[3,5], M[4,1],M[4,3],M[4,4],M[4,5])
        c3 =  det4x4_gpu(M[1,1],M[1,2],M[1,4],M[1,5], M[2,1],M[2,2],M[2,4],M[2,5], M[3,1],M[3,2],M[3,4],M[3,5], M[4,1],M[4,2],M[4,4],M[4,5])
        c4 = -det4x4_gpu(M[1,1],M[1,2],M[1,3],M[1,5], M[2,1],M[2,2],M[2,3],M[2,5], M[3,1],M[3,2],M[3,3],M[3,5], M[4,1],M[4,2],M[4,3],M[4,5])
        c5 =  det4x4_gpu(M[1,1],M[1,2],M[1,3],M[1,4], M[2,1],M[2,2],M[2,3],M[2,4], M[3,1],M[3,2],M[3,3],M[3,4], M[4,1],M[4,2],M[4,3],M[4,4])
        return SVector{5, RationalGPU}(c1, c2, c3, c4, c5)
    end

    # --- Datenvorbereitung ---
    function prepare_simplices_for_gpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{6, Int}})
        num_simplices = length(S_indices)
        simplices_data = Vector{SMatrix{6, 5, RationalGPU, 30}}(undef, num_simplices)
        for i in 1:num_simplices
            cpu_matrix = P[collect(S_indices[i]), :]
            gpu_matrix_data = MVector{30, RationalGPU}(undef)
            for k in 1:30
                r_big = cpu_matrix[k]
                num_big, den_big = r_big.num, r_big.den
                if num_big > typemax(Int64) || num_big < typemin(Int64) || den_big > typemax(Int64)
                    @warn "Rational number component exceeds Int64 range. GPU results may be incorrect."
                end
                gpu_matrix_data[k] = RationalGPU(Int64(num_big), Int64(den_big))
            end
            simplices_data[i] = SMatrix{6, 5, RationalGPU, 30}(gpu_matrix_data)
        end
        return CuArray(simplices_data)
    end
    
    # --- Kernlogik: 5D Separating Axis Theorem ---
    function simplices_intersect_gpu(s1, s2)
        face_map = SMatrix{6, 6, Int, 36}(1,2,3,4,5,6, 1,2,3,4,6,5, 1,2,3,5,6,4, 1,2,4,5,6,3, 1,3,4,5,6,2, 2,3,4,5,6,1)
        for simplex_idx in 1:2
            current_s = (simplex_idx == 1) ? s1 : s2
            for i in 1:6
                p0_idx, p1_idx, p2_idx, p3_idx, p4_idx = face_map[i,1], face_map[i,2], face_map[i,3], face_map[i,4], face_map[i,5]
                p_off_face = current_s[face_map[i,6],:]
                p0, p1, p2, p3, p4 = current_s[p0_idx,:], current_s[p1_idx,:], current_s[p2_idx,:], current_s[p3_idx,:], current_s[p4_idx,:]
                normal = normal_vector_5d(p1 - p0, p2 - p0, p3 - p0, p4 - p0)
                if (dot_gpu(normal, p_off_face - p0).num > 0) != (dot_gpu(normal, p_off_face - p0).den < 0); normal = -normal; end
                
                # KORRIGIERT: is_zero_gpu() entfernt und direkt die Numeratoren überprüft.
                is_normal_zero = normal[1].num == 0 && normal[2].num == 0 && normal[3].num == 0 && normal[4].num == 0 && normal[5].num == 0
                if !is_normal_zero
                    proj1 = SVector(dot_gpu(s1[1,:], normal), dot_gpu(s1[2,:], normal), dot_gpu(s1[3,:], normal), dot_gpu(s1[4,:], normal), dot_gpu(s1[5,:], normal), dot_gpu(s1[6,:], normal))
                    proj2 = SVector(dot_gpu(s2[1,:], normal), dot_gpu(s2[2,:], normal), dot_gpu(s2[3,:], normal), dot_gpu(s2[4,:], normal), dot_gpu(s2[5,:], normal), dot_gpu(s2[6,:], normal))
                    min1 = reduce(min_rat, proj1); max1 = reduce(max_rat, proj1)
                    min2 = reduce(min_rat, proj2); max2 = reduce(max_rat, proj2)
                    if max1 <= min2 || max2 <= min1; return false; end
                end
            end
        end

        edge_indices = SMatrix{15, 2, Int, 30}(1,2, 1,3, 1,4, 1,5, 1,6, 2,3, 2,4, 2,5, 2,6, 3,4, 3,5, 3,6, 4,5, 4,6, 5,6)
        triangle_indices = SMatrix{20, 3, Int, 60}(1,2,3, 1,2,4, 1,2,5, 1,2,6, 1,3,4, 1,3,5, 1,3,6, 1,4,5, 1,4,6, 1,5,6, 2,3,4, 2,3,5, 2,3,6, 2,4,5, 2,4,6, 2,5,6, 3,4,5, 3,4,6, 3,5,6, 4,5,6)
        tetra_indices = SMatrix{15, 4, Int, 60}(1,2,3,4, 1,2,3,5, 1,2,3,6, 1,2,4,5, 1,2,4,6, 1,2,5,6, 1,3,4,5, 1,3,4,6, 1,3,5,6, 1,4,5,6, 2,3,4,5, 2,3,4,6, 2,3,5,6, 2,4,5,6, 3,4,5,6)
        
        function test_axis(axis, s1, s2)
            # KORRIGIERT: is_zero_gpu() entfernt und direkt die Numeratoren überprüft.
            is_axis_zero = axis[1].num == 0 && axis[2].num == 0 && axis[3].num == 0 && axis[4].num == 0 && axis[5].num == 0
            if !is_axis_zero
                proj1 = SVector(dot_gpu(s1[1,:], axis), dot_gpu(s1[2,:], axis), dot_gpu(s1[3,:], axis), dot_gpu(s1[4,:], axis), dot_gpu(s1[5,:], axis), dot_gpu(s1[6,:], axis))
                proj2 = SVector(dot_gpu(s2[1,:], axis), dot_gpu(s2[2,:], axis), dot_gpu(s2[3,:], axis), dot_gpu(s2[4,:], axis), dot_gpu(s2[5,:], axis), dot_gpu(s2[6,:], axis))
                min1 = reduce(min_rat, proj1); max1 = reduce(max_rat, proj1)
                min2 = reduce(min_rat, proj2); max2 = reduce(max_rat, proj2)
                if max1 <= min2 || max2 <= min1; return false; end
            end
            return true
        end

        for i in 1:15, j in 1:15
            edge_vec = s1[edge_indices[i,2],:] - s1[edge_indices[i,1],:]
            tetra_p1 = s2[tetra_indices[j,1],:]; tetra_vec1 = s2[tetra_indices[j,2],:] - tetra_p1; tetra_vec2 = s2[tetra_indices[j,3],:] - tetra_p1; tetra_vec3 = s2[tetra_indices[j,4],:] - tetra_p1
            if !test_axis(normal_vector_5d(edge_vec, tetra_vec1, tetra_vec2, tetra_vec3), s1, s2); return false; end
        end

        for i in 1:20, j in 1:20
            tri1_p1 = s1[triangle_indices[i,1],:]; tri1_vec1 = s1[triangle_indices[i,2],:] - tri1_p1; tri1_vec2 = s1[triangle_indices[i,3],:] - tri1_p1
            tri2_p1 = s2[triangle_indices[j,1],:]; tri2_vec1 = s2[triangle_indices[j,2],:] - tri2_p1; tri2_vec2 = s2[triangle_indices[j,3],:] - tri2_p1
            if !test_axis(normal_vector_5d(tri1_vec1, tri1_vec2, tri2_vec1, tri2_vec2), s1, s2); return false; end
        end

        return true
    end

    # --- GPU Kernel und Host-Funktion ---
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
            if simplices_intersect_gpu(simplices[i1], simplices[i2])
                res_idx = (CUDA.@atomic counter[1] += Int32(1)) + 1
                if res_idx <= size(results_buffer, 1)
                    results_buffer[res_idx, 1] = Int32(i1)
                    results_buffer[res_idx, 2] = Int32(i2)
                end
            end
        end
        return
    end
    
    function get_intersecting_pairs_gpu_5d(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{6, Int}})
        num_simplices = length(S_indices)
        if num_simplices < 2; return Vector{Vector{Int}}(); end
        simplices_gpu = prepare_simplices_for_gpu(P, S_indices)
        total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2
        results_buffer_gpu = CUDA.zeros(Int32, total_pairs, 2); counter_gpu = CUDA.zeros(Int32, 1)
        threads = 256; blocks = cld(total_pairs, threads)
        @cuda threads=threads blocks=blocks intersection_kernel(simplices_gpu, num_simplices, results_buffer_gpu, counter_gpu)
        num_intersections = CUDA.@allowscalar counter_gpu[1]
        results_cpu = if num_intersections == 0; Matrix{Int32}(undef, 0, 2); else; Array(view(results_buffer_gpu, 1:num_intersections, :)); end
        clauses = [ [-(Int(results_cpu[i, 1])), -(Int(results_cpu[i, 2]))] for i in 1:num_intersections ]
        CUDA.unsafe_free!(simplices_gpu); CUDA.unsafe_free!(results_buffer_gpu); CUDA.unsafe_free!(counter_gpu); CUDA.reclaim()
        return clauses
    end
end
