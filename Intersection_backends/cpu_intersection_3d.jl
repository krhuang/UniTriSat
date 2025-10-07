module CPUIntersection3D

using StaticArrays
using Base.Threads

export tetrahedra_intersect_cpu, get_intersecting_pairs_cpu, prepare_tetrahedra_cpu

# ----------------------------
# Vector operations
# ----------------------------
@inline dot(v1::SVector{3,Int64}, v2::SVector{3,Int64}) =
    v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3]

@inline cross(v1::SVector{3,Int64}, v2::SVector{3,Int64}) = SVector{3,Int64}(
    v1[2]*v2[3] - v1[3]*v2[2],
    v1[3]*v2[1] - v1[1]*v2[3],
    v1[1]*v2[2] - v1[2]*v2[1]
)

@inline is_zero_3d(v::SVector{3,Int64}) = v[1] == Int64(0) && v[2] == Int64(0) && v[3] == Int64(0)

@inline min_rat(a::Int64, b::Int64) = a <= b ? a : b
@inline max_rat(a::Int64, b::Int64) = a <= b ? b : a

@inline function project_tetrahedron(t::SMatrix{4,3,Int64}, axis::SVector{3,Int64})
    proj = SVector(dot(t[1,:], axis), dot(t[2,:], axis), dot(t[3,:], axis), dot(t[4,:], axis))
    min_proj = min_rat(min_rat(proj[1], proj[2]), min_rat(proj[3], proj[4]))
    max_proj = max_rat(max_rat(proj[1], proj[2]), max_rat(proj[3], proj[4]))
    return min_proj, max_proj
end

# ----------------------------
# Tetrahedron intersection test
# ----------------------------
function tetrahedra_intersect_cpu(t1::SMatrix{4,3,Int64}, t2::SMatrix{4,3,Int64})::Bool
    face_vertex_map = SMatrix{4,4,Int,16}(1,2,3,4, 1,2,4,3, 1,3,4,2, 2,3,4,1)

    for t in (t1, t2)
        for i in 1:4
            p0, p1, p2, p3 = t[face_vertex_map[i,1],:], t[face_vertex_map[i,2],:], t[face_vertex_map[i,3],:], t[face_vertex_map[i,4],:]
            normal = cross(p1 - p0, p2 - p0)
            if dot(normal, p3 - p0) > 0
                normal = -normal
            end
            if !is_zero_3d(normal)
                min1, max1 = project_tetrahedron(t1, normal)
                min2, max2 = project_tetrahedron(t2, normal)
                if max1 <= min2 || max2 <= min1
                    return false
                end
            end
        end
    end

    for i1 in 1:4, j1 in (i1+1):4
        edge1 = t1[j1,:] - t1[i1,:]
        for i2 in 1:4, j2 in (i2+1):4
            edge2 = t2[j2,:] - t2[i2,:]
            axis = cross(edge1, edge2)
            if !is_zero_3d(axis)
                min1, max1 = project_tetrahedron(t1, axis)
                min2, max2 = project_tetrahedron(t2, axis)
                if max1 <= min2 || max2 <= min1
                    return false
                end
            end
        end
    end
    return true
end

# ----------------------------
# Prepare tetrahedra from Rational{BigInt} CPU matrix
# ----------------------------
function prepare_tetrahedra_cpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{4,Int}})
    num_simplices = length(S_indices)
    simplices = Vector{SMatrix{4,3,Int64}}(undef, num_simplices)
    for i in 1:num_simplices
        idxs = S_indices[i]
        mat = MMatrix{4,3,Int64}(undef)
        for r in 1:4, c in 1:3
            r_big = P[idxs[r], c]
            n, d = Int(r_big.num), Int(r_big.den)
            @assert d == 1
            mat[r,c] = n
        end
        simplices[i] = SMatrix{4,3,Int64}(mat)
    end
    return simplices
end

# ----------------------------
# Multithreaded intersection
# ----------------------------
function get_intersecting_pairs_cpu(P::Matrix{Rational{BigInt}}, S_indices::Vector{NTuple{4, Int}})
    num_simplices = length(S_indices)
    if num_simplices < 2
        return Vector{Tuple{Int,Int}}()
    end

    simplices = prepare_tetrahedra_cpu(P, S_indices)
    total_pairs = div(num_simplices * (num_simplices - 1), 2)
    num_threads = Threads.nthreads()

    # Preallocate per-thread buffers
    thread_buffers = [Vector{Tuple{Int,Int}}() for _ in 1:num_threads]

    # Split work evenly among threads
    pairs_per_thread = div(total_pairs + num_threads - 1, num_threads)

    Threads.@threads for thread_id in 1:num_threads
        start_idx = (thread_id - 1) * pairs_per_thread + 1
        end_idx = min(thread_id * pairs_per_thread, total_pairs)
        buf = thread_buffers[thread_id]

        for idx in start_idx:end_idx
            # Compute (i,j) from linear index
            i = Int(floor((1 + sqrt(1 + 8*(idx-1))) / 2))
            acc = div(i*(i-1), 2)
            j = idx - acc + i
            if j > num_simplices
                continue
            end

            t1, t2 = simplices[i], simplices[j]
            if tetrahedra_intersect_cpu(t1, t2)
                push!(buf, (i,j))
            end
        end
    end

    # Merge thread-local buffers
    results = Vector{Tuple{Int,Int}}()
    for buf in thread_buffers
        append!(results, buf)
    end

    return [ [-i,-j] for (i,j) in results ]
end

end
