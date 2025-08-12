module GPUIntersection4D
# This module is only loaded if CUDA, StaticArrays, and CUDA.Adapt are available.
# The `..` prefix indicates that these are from the parent module (Main).
using ..Combinatorics, ..LinearAlgebra, ..Printf
using ..CUDA, ..StaticArrays, ..CUDA.Adapt

#=
RATIONAL ARITHMETIC ON GPU
A custom Rational type to be used on the GPU, as BigInt rationals are not supported.
=#
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
    # Ensure denominator is always positive for a canonical representation.
    d = den > 0 ? den : -den
    n = den > 0 ? num : -num
    return RationalGPU(n ÷ common, d ÷ common)
end

# Overload basic arithmetic and comparison operators for our RationalGPU struct.
@inline Base.:+(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.den + r2.num * r1.den, r1.den * r2.den)
@inline Base.:-(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.den - r2.num * r1.den, r1.den * r2.den)
@inline Base.:*(r1::RationalGPU, r2::RationalGPU) = create_rational_gpu(r1.num * r2.num, r1.den * r2.den)
@inline Base.:-(r::RationalGPU) = RationalGPU(-r.num, r.den)
@inline Base.:(==)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den == r2.num * r1.den
@inline Base.:(<)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den < r2.num * r1.den
@inline Base.:(<=)(r1::RationalGPU, r2::RationalGPU) = r1.num * r2.den <= r2.num * r1.den
@inline min_rat(a::RationalGPU, b::RationalGPU) = a <= b ? a : b
@inline max_rat(a::RationalGPU, b::RationalGPU) = a <= b ? b : a

#=
4D VECTOR OPERATIONS
These functions are generalized to work with 4D vectors (SVector{4, RationalGPU}).
=#
@inline dot_gpu(v1, v2) = (v1[1] * v2[1]) + (v1[2] * v2[2]) + (v1[3] * v2[3]) + (v1[4] * v2[4])
@inline is_zero_gpu(v) = (v[1] == RationalGPU(0,1) && v[2] == RationalGPU(0,1) && v[3] == RationalGPU(0,1) && v[4] == RationalGPU(0,1))

# Helper for 3x3 determinant, needed for 4D normal vector calculation.
@inline function det3x3_gpu(m11, m12, m13, m21, m22, m23, m31, m32, m33)
    return m11 * (m22 * m33 - m23 * m32) -
        m12 * (m21 * m33 - m23 * m31) +
            m13 * (m21 * m32 - m22 * m31)
end

# Computes the normal vector to three 4D vectors (defines a hyperplane).
# This is the 4D equivalent of the 3D cross product.
@inline function normal_vector_4d(v1, v2, v3)
    return SVector{4, RationalGPU}(
        det3x3_gpu(v1[2], v1[3], v1[4], v2[2], v2[3], v2[4], v3[2], v3[3], v3[4]),
        -det3x3_gpu(v1[1], v1[3], v1[4], v2[1], v2[3], v2[4], v3[1], v3[3], v3[4]),
        det3x3_gpu(v1[1], v1[2], v1[4], v2[1], v2[2], v2[4], v3[1], v3[2], v3[4]),
        -det3x3_gpu(v1[1], v1[2], v1[3], v2[1], v2[2], v2[3], v3[1], v3[2], v3[3])
        )
end

#=
DATA PREPARATION
Converts CPU data to GPU-compatible format. Now handles 4D simplices (5 vertices, 4 coords).
=#
function prepare_simplices_for_gpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{5, Int}})
    num_simplices = length(S_indices)
    # A 4D simplex has 5 vertices, each with 4 coordinates.
    simplices_data = Vector{SMatrix{5, 4, RationalGPU, 20}}(undef, num_simplices)
    for i in 1:num_simplices
        cpu_matrix = P[collect(S_indices[i]), :]
        gpu_matrix_data = MVector{20, RationalGPU}(undef)
        for k in 1:20 # 5 vertices * 4 coordinates
            r_big = cpu_matrix[k]
            num_big, den_big = r_big.num, r_big.den
            if num_big > typemax(Int64) || num_big < typemin(Int64) || den_big > typemax(Int64)
                @warn "Rational number component exceeds Int64 range. GPU results may be incorrect."
            end
            gpu_matrix_data[k] = RationalGPU(Int64(num_big), Int64(den_big))
        end
        simplices_data[i] = SMatrix{5, 4, RationalGPU, 20}(gpu_matrix_data)
    end
    return CuArray(simplices_data)
end

#=
4D SEPARATING AXIS THEOREM (SAT) IMPLEMENTATION
This is the core intersection logic for two 4D simplices (pentatopes).
=#
function simplices_intersect_gpu(s1, s2)
    ZERO = RationalGPU(0, 1)

    # A 4D simplex has 5 vertices. The faces are tetrahedra formed by 4 vertices.
    # The 5th vertex is used to orient the normal.
    face_map = SMatrix{5, 5, Int, 25}(
        1, 2, 3, 4, 5,
        1, 2, 3, 5, 4,
        1, 2, 4, 5, 3,
        1, 3, 4, 5, 2,
        2, 3, 4, 5, 1
        )

    # Test axes normal to the faces of the first simplex
    for i in 1:5
        p0_idx, p1_idx, p2_idx, p3_idx = face_map[i,1], face_map[i,2], face_map[i,3], face_map[i,4]
        p_off_face_idx = face_map[i,5]

        p0, p1, p2 = s1[p0_idx,:], s1[p1_idx,:], s1[p2_idx,:]
        p3 = s1[p3_idx,:]
        p_off_face = s1[p_off_face_idx,:]

        normal = normal_vector_4d(p1 - p0, p2 - p0, p3 - p0)

        # Ensure normal points inwards
        if dot_gpu(normal, p_off_face - p0) > ZERO; normal = -normal; end

        if !is_zero_gpu(normal)
            proj1_vals = SVector(dot_gpu(s1[1,:], normal), dot_gpu(s1[2,:], normal), dot_gpu(s1[3,:], normal), dot_gpu(s1[4,:], normal), dot_gpu(s1[5,:], normal))
            proj2_vals = SVector(dot_gpu(s2[1,:], normal), dot_gpu(s2[2,:], normal), dot_gpu(s2[3,:], normal), dot_gpu(s2[4,:], normal), dot_gpu(s2[5,:], normal))

            min1 = min_rat(min_rat(proj1_vals[1], proj1_vals[2]), min_rat(min_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
            max1 = max_rat(max_rat(proj1_vals[1], proj1_vals[2]), max_rat(max_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
            min2 = min_rat(min_rat(proj2_vals[1], proj2_vals[2]), min_rat(min_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))
            max2 = max_rat(max_rat(proj2_vals[1], proj2_vals[2]), max_rat(max_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))

            if max1 <= min2 || max2 <= min1; return false; end
        end
    end

    # Test axes normal to the faces of the second simplex
    for i in 1:5
        p0_idx, p1_idx, p2_idx, p3_idx = face_map[i,1], face_map[i,2], face_map[i,3], face_map[i,4]
        p_off_face_idx = face_map[i,5]

        p0, p1, p2 = s2[p0_idx,:], s2[p1_idx,:], s2[p2_idx,:]
        p3 = s2[p3_idx,:]
        p_off_face = s2[p_off_face_idx,:]

        normal = normal_vector_4d(p1 - p0, p2 - p0, p3 - p0)
        if dot_gpu(normal, p_off_face - p0) > ZERO; normal = -normal; end

        if !is_zero_gpu(normal)
            proj1_vals = SVector(dot_gpu(s1[1,:], normal), dot_gpu(s1[2,:], normal), dot_gpu(s1[3,:], normal), dot_gpu(s1[4,:], normal), dot_gpu(s1[5,:], normal))
            proj2_vals = SVector(dot_gpu(s2[1,:], normal), dot_gpu(s2[2,:], normal), dot_gpu(s2[3,:], normal), dot_gpu(s2[4,:], normal), dot_gpu(s2[5,:], normal))

            min1 = min_rat(min_rat(proj1_vals[1], proj1_vals[2]), min_rat(min_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
            max1 = max_rat(max_rat(proj1_vals[1], proj1_vals[2]), max_rat(max_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
            min2 = min_rat(min_rat(proj2_vals[1], proj2_vals[2]), min_rat(min_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))
            max2 = max_rat(max_rat(proj2_vals[1], proj2_vals[2]), max_rat(max_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))

            if max1 <= min2 || max2 <= min1; return false; end
        end
    end

    # In 4D, we must test axes perpendicular to an edge from one simplex and a 2-face (triangle) from the other.
    # There are 10 edges and 10 triangles in a 4D simplex.
    edge_indices = SMatrix{10, 2, Int, 20}(1,2, 1,3, 1,4, 1,5, 2,3, 2,4, 2,5, 3,4, 3,5, 4,5)
    triangle_indices = SMatrix{10, 3, Int, 30}(1,2,3, 1,2,4, 1,2,5, 1,3,4, 1,3,5, 1,4,5, 2,3,4, 2,3,5, 2,4,5, 3,4,5)

    # Test axes: perp(edge from s1, triangle from s2)
    for i in 1:10 # Edges of s1
        edge_vec = s1[edge_indices[i,2],:] - s1[edge_indices[i,1],:]
        for j in 1:10 # Triangles of s2
            tri_p1 = s2[triangle_indices[j,1],:]
            tri_vec1 = s2[triangle_indices[j,2],:] - tri_p1
            tri_vec2 = s2[triangle_indices[j,3],:] - tri_p1

            axis = normal_vector_4d(edge_vec, tri_vec1, tri_vec2)
            if !is_zero_gpu(axis)
                proj1_vals = SVector(dot_gpu(s1[1,:], axis), dot_gpu(s1[2,:], axis), dot_gpu(s1[3,:], axis), dot_gpu(s1[4,:], axis), dot_gpu(s1[5,:], axis))
                proj2_vals = SVector(dot_gpu(s2[1,:], axis), dot_gpu(s2[2,:], axis), dot_gpu(s2[3,:], axis), dot_gpu(s2[4,:], axis), dot_gpu(s2[5,:], axis))

                min1 = min_rat(min_rat(proj1_vals[1], proj1_vals[2]), min_rat(min_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
                max1 = max_rat(max_rat(proj1_vals[1], proj1_vals[2]), max_rat(max_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
                min2 = min_rat(min_rat(proj2_vals[1], proj2_vals[2]), min_rat(min_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))
                max2 = max_rat(max_rat(proj2_vals[1], proj2_vals[2]), max_rat(max_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))

                if max1 <= min2 || max2 <= min1; return false; end
            end
        end
    end

    # Test axes: perp(triangle from s1, edge from s2)
    for i in 1:10 # Triangles of s1
        tri_p1 = s1[triangle_indices[i,1],:]
        tri_vec1 = s1[triangle_indices[i,2],:] - tri_p1
        tri_vec2 = s1[triangle_indices[i,3],:] - tri_p1
        for j in 1:10 # Edges of s2
            edge_vec = s2[edge_indices[j,2],:] - s2[edge_indices[j,1],:]

            axis = normal_vector_4d(tri_vec1, tri_vec2, edge_vec)
            if !is_zero_gpu(axis)
                proj1_vals = SVector(dot_gpu(s1[1,:], axis), dot_gpu(s1[2,:], axis), dot_gpu(s1[3,:], axis), dot_gpu(s1[4,:], axis), dot_gpu(s1[5,:], axis))
                proj2_vals = SVector(dot_gpu(s2[1,:], axis), dot_gpu(s2[2,:], axis), dot_gpu(s2[3,:], axis), dot_gpu(s2[4,:], axis), dot_gpu(s2[5,:], axis))

                min1 = min_rat(min_rat(proj1_vals[1], proj1_vals[2]), min_rat(min_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
                max1 = max_rat(max_rat(proj1_vals[1], proj1_vals[2]), max_rat(max_rat(proj1_vals[3], proj1_vals[4]), proj1_vals[5]))
                min2 = min_rat(min_rat(proj2_vals[1], proj2_vals[2]), min_rat(min_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))
                max2 = max_rat(max_rat(proj2_vals[1], proj2_vals[2]), max_rat(max_rat(proj2_vals[3], proj2_vals[4]), proj2_vals[5]))

                if max1 <= min2 || max2 <= min1; return false; end
            end
        end
    end

    # If no separating axis was found after all tests, the simplices intersect.
    return true
end

#=
GPU KERNEL AND HOST FUNCTION
The kernel logic remains largely the same, just calling the new 4D intersection function.
=#
function intersection_kernel(simplices, num_simplices_arg, results_buffer, counter)
    num_simplices = Int64(num_simplices_arg)
    idx = Int64((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2

    if idx > total_pairs; return; end

    # Decode the pair index (i1, i2) from the linear thread index (idx)
    i1 = Int64(1); total_pairs_before_i1 = Int64(0)
    while total_pairs_before_i1 + (num_simplices - i1) < idx
        total_pairs_before_i1 += (num_simplices - i1); i1 += 1
    end
    i2 = i1 + (idx - total_pairs_before_i1)

    if i1 < i2 && i2 <= num_simplices
        s1, s2 = simplices[i1], simplices[i2]
        # Call the new 4D intersection function
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

# Main host function to be called from CPU code.
function get_intersecting_pairs_gpu_4d(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{5, Int}})
    num_simplices = length(S_indices)
    if num_simplices < 2; return Vector{Vector{Int}}(); end

    simplices_gpu = prepare_simplices_for_gpu(P, S_indices)
    total_pairs = (num_simplices * (num_simplices - 1)) ÷ 2

    # Allocate GPU memory for results and a counter.
    results_buffer_gpu = CUDA.zeros(Int32, total_pairs, 2)
    counter_gpu = CUDA.zeros(Int32, 1)

    # Launch the kernel.
    threads_per_block = 256
    blocks = cld(total_pairs, threads_per_block)
    @cuda threads=threads_per_block blocks=blocks intersection_kernel(simplices_gpu, num_simplices, results_buffer_gpu, counter_gpu)

    # Copy results back to CPU.
    num_intersections = CUDA.@allowscalar counter_gpu[1]
    results_buffer_cpu = if num_intersections == 0
        Matrix{Int32}(undef, 0, 2)
    else
        # Copy only the valid results.
        Array(view(results_buffer_gpu, 1:num_intersections, :))
    end

    # Format results as requested.
    clauses = Vector{Vector{Int}}(undef, num_intersections)
    for i in 1:num_intersections
        clauses[i] = [-(Int(results_buffer_cpu[i, 1])), -(Int(results_buffer_cpu[i, 2]))]
    end

    # Free GPU memory.
    CUDA.unsafe_free!(simplices_gpu); CUDA.unsafe_free!(results_buffer_gpu); CUDA.unsafe_free!(counter_gpu)
    CUDA.reclaim()

    return clauses
end
end
