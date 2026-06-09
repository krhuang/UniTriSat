
module Helpers

using Printf
using Polyhedra

export read_polytopes_from_file, _convert_polyhedron_to_vmatrix, format_bytes, format_duration, ghost_print



function format_duration(total_seconds::Float64)
    total_seconds_int = floor(Int, total_seconds)
    h = total_seconds_int ÷ 3600
    rem_seconds = total_seconds_int % 3600
    m = rem_seconds ÷ 60
    s = rem_seconds % 60
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function format_bytes(b::Real)
    if b > 1024^3
        return @sprintf("%.2f GiB", b / 1024^3)
    elseif b > 1024^2
        return @sprintf("%.2f MiB", b / 1024^2)
    elseif b > 1024
        return @sprintf("%.2f KiB", b / 1024)
    else
        return @sprintf("%d B", b)
    end
end

function read_polytopes_from_file(filepath::String)
    polytopes = Vector{Matrix{Int}}()
    current_vertices = Vector{Vector{Int}}()
    function process_buffered_vertices()
        if !isempty(current_vertices)
            push!(polytopes, vcat(current_vertices'...))
            empty!(current_vertices)
        end
    end
    for line in eachline(filepath)
        line = strip(line)
        if isempty(line) || startswith(line, "#"); process_buffered_vertices(); continue; end
        vertex_pattern = r"\[([^\[\]]+)\]"
        if startswith(line, "[[")
            process_buffered_vertices()
            vertices_new_format = Vector{Vector{Int}}()
            for m in eachmatch(vertex_pattern, line)
                push!(vertices_new_format, parse.(Int, split(m.captures[1], ",")))
            end
            if !isempty(vertices_new_format); push!(polytopes, vcat(vertices_new_format'...)); end
        else
            try; push!(current_vertices, parse.(Int, split(line))); catch e; @warn "Skipping malformed line: $line. Error: $e"; end
        end
    end
    process_buffered_vertices()
    return polytopes
end

function _convert_polyhedron_to_vmatrix(p::Polyhedron)
    try
        return vcat([Int.(v)' for v in points(p)]...)
    catch e
        @error("Error converting Polyhedron object to Matrix{Int}: $e")
        return Matrix{Int}(undef, 0, 0) # Leere Matrix zurückgeben
    end
end

# prints and moves the cursor up so the next print will overwrite it
# if the string contains newlines, then the cursor must be moved up more
function ghost_print(msg::String)
    
    #moves cursor to the front of the line
    print("\r\u001b[K" * msg)
    
    # Count the number of newlines in the message to determine how many lines to move the cursor up
    num_newlines = count(==('\n'), msg)
    
    if num_newlines > 0
        print("\u001b[$(num_newlines)A") 
    end
end

end
