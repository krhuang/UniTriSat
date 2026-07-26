#!/usr/bin/env julia
#
# Converts the H-representation (facet inequalities) of lattice polytopes from
# a file into their V-representation (list of vertices) using exact rational
# arithmetic provided by CDDLib.jl.
#
#     julia scripts/polytope_representation_conversion.jl [infile] [dim] [outfile]
#
# Paths given on the command line are used as-is; the defaults are anchored to
# <root>/Polytopes rather than to the current working directory.

include(joinpath(@__DIR__, "_setup.jl"))

using Polyhedra
using CDDLib
using LinearAlgebra: I

const R = Rational{Int} # Use rational numbers for guaranteed exactness

# Initialize the CDDLib library instance with explicit exact arithmetic.
const LIB_INSTANCE = CDDLib.Library(:exact)

# --- 1. Core conversion logic ---

"""
Converts a single polytope defined by a list of facet lines into its vertices.
It uses exact rational arithmetic (Rational{Int}) via CDDLib.
"""
function convert_polytope_data(lines::Vector{String}, dim::Int, polytope_index::Int)
    A_rows = R[]
    b_vec = R[]

    # Parse each line into the inequality components
    for line in lines
        # Split the line into individual coefficients and parse as integers
        coeffs = try
            parse.(Int, split(line))
        catch e
            # Handle lines that cannot be parsed (e.g. empty or non-numeric)
            @warn "Skipping malformed line in Polytope $polytope_index: $line. Error: $e"
            continue
        end

        # Expected format: b a1 a2 ... aN (1 + dim coefficients)
        if length(coeffs) < dim + 1
            @warn "Skipping line in Polytope $polytope_index with insufficient coefficients ($line). Expected at least $(dim + 1)."
            continue
        end

        b = R(coeffs[1])
        # The remaining coefficients define the vector 'a'
        a_coeffs = R.(coeffs[2:dim+1])

        # The input format is: 0 <= b + sum(a_i * x_i)
        # The standard H-representation is: A*x <= b_std
        # Rearranging: -sum(a_i * x_i) <= b
        # This means A = -a_coeffs and b_std = b.
        A_row = -a_coeffs

        append!(A_rows, A_row)
        push!(b_vec, b)
    end

    num_facets = length(b_vec)
    if num_facets == 0
        @warn "Polytope $polytope_index has no valid facets. Skipping."
        return nothing
    end

    # Reshape the flattened A_rows vector into a matrix (num_facets x dim).
    # permutedims rather than ' so that a materialised Matrix reaches Polyhedra.
    A = permutedims(reshape(A_rows, dim, num_facets))
    b = b_vec

    try
        # 1. Create the H-representation with Rational{Int} data
        h_rep = hrep(A, b)

        # 2. Create the Polyhedron object.
        poly = polyhedron(h_rep, LIB_INSTANCE)

        # 3. Compute the V-representation (vertex enumeration)
        v_rep = vrep(poly)

        # 4. Extract the vertices using the standard Polyhedra API
        verts_iter = points(v_rep)

        if isempty(verts_iter)
            vertices = Matrix{R}(undef, dim, 0)
        else
            # Convert the iterator of vectors into a Matrix whose columns are vertices
            vertices = reduce(hcat, verts_iter)
        end

        return (
            polytope_index = polytope_index,
            num_facets = num_facets,
            num_vertices = size(vertices, 2),
            vertices = vertices,
        )
    catch e
        @error "An error occurred during vertex enumeration for Polytope $polytope_index: $e"
        return nothing
    end
end

# --- 2. Output writing logic ---

"""
Writes the vertices of a single polytope to the output file stream.
Includes the lattice point assertion.
"""
function write_polytope_result(io::IO, result::NamedTuple, add_leading_newline::Bool)
    if add_leading_newline
        println(io, "") # Separate polytopes with an empty line
    end

    # Vertices are stored as columns in result.vertices
    for vertex in eachcol(result.vertices)

        # LATTICE POINT ASSERTION: check that all coordinates are integers
        for (j, v) in enumerate(vertex)
            if denominator(v) != 1
                error("Lattice point assertion failed for Polytope $(result.polytope_index), Vertex: $(vertex). Coordinate #$j is not an integer (Denominator is $(denominator(v))).")
            end
        end

        # If the assertion passed, write the integer coordinates (numerator)
        compact_vertex = [string(numerator(v)) for v in vertex]

        # Write the vertex coordinates as a space-separated row
        println(io, join(compact_vertex, " "))
    end
end

# --- 3. Memoryless file reading and conversion logic ---

"""
Reads polytope H-representations from a file, converts them to V-representations,
and writes the results immediately to the output file.
This prevents storing all polytope data in memory simultaneously.
"""
function convert_polytopes_file_hrep_to_vrep(filepath::String, dimension::Int,
                                             output_file::String)
    polytope_index = 0
    converted_count = 0

    println("Starting memoryless conversion from $filepath to $output_file...")

    if !isfile(filepath)
        error("File not found: $filepath. Please ensure the file is accessible.")
    end
    mkpath(dirname(output_file))

    # Open both files before the main loop
    open(filepath, "r") do input_io
        open(output_file, "w") do output_io
            current_polytope_lines = String[]

            for line in eachline(input_io)
                stripped_line = strip(line)

                # Check for the start of a new polytope block
                if startswith(stripped_line, "FACETS")
                    # Process the previous polytope if lines were collected
                    if !isempty(current_polytope_lines)
                        polytope_index += 1
                        print("Processing Polytope #$polytope_index... ")

                        result = convert_polytope_data(current_polytope_lines,
                                                       dimension, polytope_index)

                        if result !== nothing
                            # Immediate output to file
                            write_polytope_result(output_io, result, converted_count > 0)
                            converted_count += 1
                            println("Done. Wrote $(result.num_vertices) vertices.")
                        else
                            println("Skipped (invalid facets).")
                        end
                    end

                    # Reset and start collecting lines for the new polytope
                    current_polytope_lines = String[]
                elseif !isempty(stripped_line) && !startswith(stripped_line, "#")
                    # Collect the facet lines, ignoring comments and empty lines
                    push!(current_polytope_lines, stripped_line)
                end
            end

            # Process the very last polytope block after the loop finishes
            if !isempty(current_polytope_lines)
                polytope_index += 1
                print("Processing Polytope #$polytope_index (End of file)... ")

                result = convert_polytope_data(current_polytope_lines,
                                               dimension, polytope_index)

                if result !== nothing
                    # Immediate output to file
                    write_polytope_result(output_io, result, converted_count > 0)
                    converted_count += 1
                    println("Done. Wrote $(result.num_vertices) vertices.")
                else
                    println("Skipped (invalid facets).")
                end
            end
        end # output_io is closed here
    end # input_io is closed here

    println("\nFinished conversion. Total Polytopes Converted and Written: $(converted_count)")
end

# -------------
# Function call
# -------------

const INFILE  = argordefault(1, polytope_path("smooth_fano_7d"))
const DIM     = parse(Int, argordefault(2, "7"))
const OUTFILE = argordefault(3, polytope_path("smooth_fano_7d_vrep.txt"))

convert_polytopes_file_hrep_to_vrep(INFILE, DIM, OUTFILE)
