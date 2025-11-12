using Printf

function split_dataset(input_filename::String, n::Int)
        infile = open(input_filename, "r")
        polytope_counter = 0
        file_counter = 1
        polytopes_buffer = String[]

        for line in eachline(infile, keep=true)
            if strip(line) == ""
                if !isempty(polytopes_buffer)
                    polytope_counter += 1
                end
                push!(polytopes_buffer, line)
                if polytope_counter == n
                    write_polytopes_to_file(file_counter, polytopes_buffer)
                    polytope_counter = 0
                    file_counter += 1
                    polytopes_buffer = []
                end
            else
                push!(polytopes_buffer, line)
            end
        end
        if polytope_counter > 0
            if !isempty(polytopes_buffer) && strip(polytopes_buffer[end]) != ""
                 push!(polytopes_buffer, "\n")
            end
            write_polytopes_to_file(file_counter, polytopes_buffer)
        end
        close(infile)
end

function write_polytopes_to_file(file_index::Int, data::Vector)
    output_filename = @sprintf("smooth_3d_chunks/smooth_lattice_polytopes_3d_%03d", file_index)
    open(output_filename, "w") do outfile
        for line in data
            write(outfile, line)
        end
    end
end

function data_lines_as_polytopes(data::Vector{String})::Int
    return sum(strip(line) == "" for line in data)
end

input_file = ARGS[1]
n = parse(Int, ARGS[2])

split_dataset(input_file, n)
