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

# ===============================================
# Assert that simplices intersect with themselves
# ===============================================
simplices_01 = [ # Simplices of the 01 cube
	[0 0 0; 1 0 0; 0 1 0; 0 0 1], 
	[0 0 0; 1 0 0; 0 1 0; 0 0 1]
] 

for simplex in simplices_01
	bool = simplices_intersect_sat_cpu(simplex, simplex)
	@test bool == true 
end

# ===============================================
# Assert that the Reeve tetrahedra has no unimodular triangulation
# ===============================================
# TODO: make this work
for t in 2:15
	reeve_simplex = [[0 0 0; 1 0 0; 0 1 0; 1 1 t]]
	# @test find_unimodular_triangulation(reeve_simplex) == false
end

# ===============================================
# Assert that the White tetrahedra has no unimodular triangulations
# ===============================================
# TODO
#= gcd computation
for ???
	white_tetrahedra = ??
	@test find_unimodular_triangulation(white_tetrahedra) == false
end
=#

# ===============================================
# Assert that every smooth 3-polytope in the dataset has a unimodular triangulation
# TODO
# @test find_unimodular_triangulation(file = ??) == true
