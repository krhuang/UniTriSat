#Test cases for SATUNITRI functions

using Combinatorics
using LinearAlgebra
using Polyhedra
using PicoSAT
using Dates
using Printf
using Base.Threads
using TOML
using Random
using Test

include("main.jl")
include("Intersection_backends/cpu_intersection.jl")

test_config = Config("", "logs/test_cases_log", "-", "normal", "P", "first", false, "cpu", false, "single-line", "verbose", true, true, true, "PicoSAT", "plot_triangulation.py", "")

# ===============================================
# Assert that simplices intersect with themselves
# ===============================================
simplices_01 = [ # Simplices of the 01 cube
	[0 0 0; 1 0 0; 0 1 0; 0 0 1], 
	[0 0 0; 1 1 0; 0 1 0; 0 0 1]
] 
#=
for simplex in simplices_01
	bool = CPUIntersection.simplices_intersect_sat_cpu(simplex, simplex)
	@test bool == true 
end
=#
# ===============================================
# Assert that the Reeve tetrahedra has no unimodular triangulation
# ===============================================
# TODO: make this work
for t in 2:15
	reeve_simplex = [0 0 0; 1 0 0; 0 1 0; 1 1 t]
	@test process_polytope(reeve_simplex, 1, 1, 1, test_config).num_solutions_found == 0
end

# ===============================================
# Assert that the White tetrahedra has no unimodular triangulations
# ===============================================
# See the survey here: https://arxiv.org/pdf/1610.01981v1
# 
upper_bound = 10
for a in 1:upper_bound
    for b in 1:upper_bound
        for c in 1:upper_bound
            d = mod((1 - a - b), c)
            if gcd(a, c) == 1 && gcd(b, c) == 1 && gcd(d, c) == 1
                if a == 1 || b == 1 || c == 1 || d == 1
                    white_tetrahedra = [0 0 0; 1 0 0; 0 1 0; a b c] # Construct the White tetrahedra
                    # Test that this tetrahedra is indeed empty
                    @test size(lattice_points_via_Oscar(white_tetrahedra), 1) == 4
                    #@test process_polytope(white_tetrahedra, 1, 1, 1, test_config).num_solutions_found == 0
                end
            end
        end
    end
end

# ===============================================
# Assert that every smooth 3-polytope in the dataset has a unimodular triangulation
# TODO
# @test find_unimodular_triangulation(file = ??) == true

# ===============================================
# Every matroid base polytope has a unimodular triangultaion
# https://arxiv.org/pdf/2309.10229


# ===============================================
# Hypersimplices? Delta(k,n) 
# https://arxiv.org/pdf/math/0501246