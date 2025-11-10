module Plot

using Polyhedra
using LinearAlgebra
using Combinatorics

export plot

    function format_simplices_for_plotter(simplices::Vector{Matrix{Int}})
        return "[" * join(["[" * join(["[$(join(v, ","))]" for v in eachrow(s)], ",") * "]" for s in simplices], ",") * "]"
    end

    function generalized_cross_product_4d(v1::Vector{T}, v2::Vector{T}, v3::Vector{T}) where T
        M = hcat(v1, v2, v3)
        return [ det(M[[2,3,4], :]), -det(M[[1,3,4], :]), det(M[[1,2,4], :]), -det(M[[1,2,3], :]) ]
    end

    function get_orthonormal_basis(normal::Vector{Rational{BigInt}})
        normal_f64 = Float64.(normal)
        if iszero(norm(normal_f64))
            return [ [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0] ]
        end
        normal_f64 ./= norm(normal_f64)

        basis = [zeros(Float64, 4) for _ in 1:3]

        j = argmax(abs.(normal_f64))
        v = zeros(Float64, 4)
        v[mod1(j + 1, 4)] = 1.0

        basis[1] = v - dot(v, normal_f64) * normal_f64
        basis[1] ./= norm(basis[1])

        k = zeros(Float64, 4)
        k_idx = mod1(j + 2, 4)
        if k_idx == argmax(v)
            k_idx = mod1(j + 3, 4)
        end
        k[k_idx] = 1.0

        basis[2] = k - dot(k, normal_f64) * normal_f64 - dot(k, basis[1]) * basis[1]
        basis[2] ./= norm(basis[2])

        basis[3] = generalized_cross_product_4d(basis[1], basis[2], normal_f64)
        basis[3] ./= norm(basis[3])

        return basis
    end

    function get_orthonormal_basis_for_subspace_3d(n1_rat::Vector{Rational{BigInt}}, n2_rat::Vector{Rational{BigInt}})
        n1 = normalize(Float64.(n1_rat))
        n2_ortho = normalize(Float64.(n2_rat) - dot(Float64.(n2_rat), n1) * n1)

        b_n1 = n1
        b_n2 = n2_ortho

        basis = Vector{Vector{Float64}}()

        for i in 1:5
            e = zeros(Float64, 5)
            e[i] = 1.0

            v = e - dot(e, b_n1) * b_n1 - dot(e, b_n2) * b_n2
            for b_found in basis
                v -= dot(v, b_found) * b_found
            end

            if norm(v) > 1e-9
                push!(basis, normalize(v))
            end
            if length(basis) == 3; break; end
        end

        return basis
    end

    function get_orthonormal_basis_for_subspace_3d_from_6d(n1_rat::Vector{Rational{BigInt}}, n2_rat::Vector{Rational{BigInt}}, n3_rat::Vector{Rational{BigInt}})
        b1 = normalize(Float64.(n1_rat))
        b2 = normalize(Float64.(n2_rat) - dot(Float64.(n2_rat), b1) * b1)
        b3 = normalize(Float64.(n3_rat) - dot(Float64.(n3_rat), b1) * b1 - dot(Float64.(n3_rat), b2) * b2)

        basis = Vector{Vector{Float64}}()
        for i in 1:6
            e = zeros(Float64, 6); e[i] = 1.0
            v = e - dot(e, b1)*b1 - dot(e, b2)*b2 - dot(e, b3)*b3
            for b_found in basis
                v -= dot(v, b_found) * b_found
            end
            if norm(v) > 1e-9
                push!(basis, normalize(v))
            end
            if length(basis) == 3; break; end
        end
        return basis
    end

    function _normalize_axis(axis::Vector{Rational{BigInt}})
        if all(iszero, axis); return axis; end
        denominators = [v.den for v in axis]; common_mult = lcm(denominators)
        int_axis = [v.num * (common_mult ÷ v.den) for v in axis]
        common_divisor = gcd(int_axis)
        if common_divisor != 0; int_axis .÷= common_divisor; end
        first_nonzero_idx = findfirst(!iszero, int_axis)
        if first_nonzero_idx !== nothing && int_axis[first_nonzero_idx] < 0; int_axis .*= -1; end
        return Rational{BigInt}.(int_axis)
    end

    function plot(initial_vertices::Matrix{Int}, dim::Int, first_solution_simplices::Vector{Matrix{Int}})
        if dim == 3
            temp_path, temp_io = mktemp()
            try write(temp_io, format_simplices_for_plotter(first_solution_simplices))
                close(temp_io)
                run(`python src/plot_triangulation.py $(temp_path)`)
            finally
                rm(temp_path, force=true)
            end
        elseif dim == 4
            initial_poly = Polyhedra.polyhedron(vrep(initial_vertices))
            boundary_planes = collect(halfspaces(hrep(initial_poly)))
            for (plane_idx, plane) in enumerate(boundary_planes)
                facet_triangulation_4D = [s for s in first_solution_simplices if count(v -> iszero(dot(plane.a, v) - plane.β), eachrow(s)) == 4]
                if isempty(facet_triangulation_4D); continue; end
                origin_4d = facet_triangulation_4D[1][1,:]; basis_3d = get_orthonormal_basis(plane.a)
                projected_simplices = Vector{Matrix{Int}}()
                for s in facet_triangulation_4D
                    face_vertices_on_plane = filter(v -> iszero(dot(plane.a, v) - plane.β), eachrow(s))
                    if length(face_vertices_on_plane) == 4
                        projected_verts_3d = [round.(Int, [dot(v - origin_4d, b) for b in basis_3d]) for v in face_vertices_on_plane]
                        push!(projected_simplices, vcat(projected_verts_3d'...))
                    end
                end
                temp_path, temp_io = mktemp()
                try write(temp_io, format_simplices_for_plotter(projected_simplices))
                    close(temp_io); run(`python src/plot_triangulation.py $(temp_path)`)
                finally
                    rm(temp_path, force=true)
                end
            end
        elseif dim == 5
            initial_poly = Polyhedra.polyhedron(vrep(initial_vertices))
            boundary_planes = collect(halfspaces(hrep(initial_poly)))
            for i in 1:length(boundary_planes)
                for j in (i + 1):length(boundary_planes)
                    plane1 = boundary_planes[i]
                    plane2 = boundary_planes[j]
                    face_simplices_5D = [s for s in first_solution_simplices if count(v -> iszero(dot(plane1.a, v) - plane1.β) && iszero(dot(plane2.a, v) - plane2.β), eachrow(s)) >= 4]
                    if isempty(face_simplices_5D); continue; end
                    origin_5d = first(filter(v -> iszero(dot(plane1.a, v) - plane1.β) && iszero(dot(plane2.a, v) - plane2.β), eachrow(face_simplices_5D[1])))
                    basis_3d = get_orthonormal_basis_for_subspace_3d(plane1.a, plane2.a)
                    projected_simplices = Vector{Matrix{Int}}()
                    for s in face_simplices_5D
                        verts_on_face = filter(v -> iszero(dot(plane1.a, v) - plane1.β) && iszero(dot(plane2.a, v) - plane2.β), eachrow(s))
                        for tetra_verts in combinations(verts_on_face, 4)
                            projected_verts_3d = [round.(Int, [dot(v - origin_5d, b) for b in basis_3d]) for v in tetra_verts]
                            push!(projected_simplices, vcat(projected_verts_3d'...))
                        end
                    end
                    if !isempty(projected_simplices)
                        unique_simplices = unique(s -> Tuple(sortslices(s, dims=1)), projected_simplices)
                        temp_path, temp_io = mktemp()
                        try write(temp_io, format_simplices_for_plotter(unique_simplices))
                            close(temp_io); run(`python src/plot_triangulation.py $(temp_path)`)
                        finally
                            rm(temp_path, force=true)
                        end
                    end
                end
            end
        elseif dim == 6
            initial_poly = Polyhedra.polyhedron(vrep(initial_vertices))
            boundary_planes = collect(halfspaces(hrep(initial_poly)))
            for i in 1:length(boundary_planes), j in (i+1):length(boundary_planes), k in (j+1):length(boundary_planes)
                p1, p2, p3 = boundary_planes[i], boundary_planes[j], boundary_planes[k]
                face_simplices_6D = [s for s in first_solution_simplices if count(v -> iszero(dot(p1.a, v) - p1.β) && iszero(dot(p2.a, v) - p2.β) && iszero(dot(p3.a, v) - p3.β), eachrow(s)) >= 4]
                if isempty(face_simplices_6D); continue; end
                origin_6d = first(filter(v -> iszero(dot(p1.a, v) - p1.β) && iszero(dot(p2.a, v) - p2.β) && iszero(dot(p3.a, v) - p3.β), eachrow(face_simplices_6D[1])))
                basis_3d = get_orthonormal_basis_for_subspace_3d_from_6d(p1.a, p2.a, p3.a)
                projected_simplices = Vector{Matrix{Int}}()
                for s in face_simplices_6D
                    verts_on_face = filter(v -> iszero(dot(p1.a, v) - p1.β) && iszero(dot(p2.a, v) - p2.β) && iszero(dot(p3.a, v) - p3.β), eachrow(s))
                    for tetra_verts in combinations(verts_on_face, 4)
                        projected_verts_3d = [round.(Int, [dot(v - origin_6d, b) for b in basis_3d]) for v in tetra_verts]
                        push!(projected_simplices, vcat(projected_verts_3d'...))
                    end
                end
                if !isempty(projected_simplices)
                    unique_simplices = unique(s -> Tuple(sortslices(s, dims=1)), projected_simplices)
                    temp_path, temp_io = mktemp(); try write(temp_io, format_simplices_for_plotter(unique_simplices)); close(temp_io); run(`python src/plot_triangulation.py $(temp_path)`); finally rm(temp_path, force=true); end
                end
            end
        end
    end


end
