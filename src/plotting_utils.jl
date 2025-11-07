# TODO: make this a module rather than inclusion as a script? x

function format_simplices_for_plotter(simplices::Vector{Matrix{Int}})
    return "[" * join(["[" * join(["[$(join(v, ","))]" for v in eachrow(s)], ",") * "]" for s in simplices], ",") * "]"
end

# --- Geometric Helper Functions ---

function generalized_cross_product_4d(v1::Vector{T}, v2::Vector{T}, v3::Vector{T}) where T
    M = hcat(v1, v2, v3)
    return [ det(M[[2,3,4], :]), -det(M[[1,3,4], :]), det(M[[1,2,4], :]), -det(M[[1,2,3], :]) ]
end

# --- Seems to only be for plotting? ---

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
