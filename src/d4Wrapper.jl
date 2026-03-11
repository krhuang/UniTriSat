module D4AllSat

export all_sat, d4_itersolve

# ---------------------------------------------------------
# 1. d-DNNF Parsing and Structs
# ---------------------------------------------------------

struct Arc
    target::Int
    literals::Vector{Int}
end

function parse_nnf(nnf_file::String)
    nodes = Dict{Int, Symbol}()
    arcs = Dict{Int, Vector{Arc}}()
    
    for line in eachline(nnf_file)
        tokens = split(line)
        isempty(tokens) && continue
        
        type_str = tokens[1]
        if type_str in ("o", "a", "t", "f")
            idx = parse(Int, tokens[2])
            if type_str == "o" nodes[idx] = :OR
            elseif type_str == "a" nodes[idx] = :AND
            elseif type_str == "t" nodes[idx] = :TRUE
            elseif type_str == "f" nodes[idx] = :FALSE
            end
        else
            src = parse(Int, tokens[1])
            tgt = parse(Int, tokens[2])
            lits = Int[]
            for i in 3:(length(tokens)-1)
                push!(lits, parse(Int, tokens[i]))
            end
            if !haskey(arcs, src) arcs[src] = Arc[] end
            push!(arcs[src], Arc(tgt, lits))
        end
    end
    return nodes, arcs
end

# ---------------------------------------------------------
# 2. O(1) Memory Graph Traversal
# ---------------------------------------------------------

function stream_expanded_models!(path::Vector{Int}, unassigned::Vector{Int}, idx::Int, out_channel::Channel)
    if idx > length(unassigned)
        put!(out_channel, sort!(copy(path), by=abs))
        return
    end
    
    v = unassigned[idx]
    
    push!(path, v)
    stream_expanded_models!(path, unassigned, idx + 1, out_channel)
    pop!(path)
    
    push!(path, -v)
    stream_expanded_models!(path, unassigned, idx + 1, out_channel)
    pop!(path)
end

function dfs_and_edges!(edges::Vector{Arc}, idx::Int, path::Vector{Int}, nodes::Dict{Int, Symbol}, arcs::Dict{Int, Vector{Arc}}, cont::Function)
    if idx > length(edges)
        cont(path)
    else
        edge = edges[idx]
        l = length(path)
        append!(path, edge.literals)
        
        next_cont = (p) -> dfs_and_edges!(edges, idx + 1, p, nodes, arcs, cont)
        stream_paths!(edge.target, path, nodes, arcs, next_cont)
        
        resize!(path, l)
    end
end

function stream_paths!(node::Int, path::Vector{Int}, nodes::Dict{Int, Symbol}, arcs::Dict{Int, Vector{Arc}}, cont::Function)
    ntype = nodes[node]
    if ntype == :TRUE
        cont(path)
    elseif ntype == :FALSE
        return
    elseif ntype == :OR
        edges = get(arcs, node, Arc[])
        for edge in edges
            l = length(path)
            append!(path, edge.literals)
            stream_paths!(edge.target, path, nodes, arcs, cont)
            resize!(path, l)
        end
    elseif ntype == :AND
        edges = get(arcs, node, Arc[])
        dfs_and_edges!(edges, 1, path, nodes, arcs, cont)
    end
end

# ---------------------------------------------------------
# 3. Domain-Specific Parallel Iterator
# ---------------------------------------------------------

"""
Takes the CNF and the geometric partitioning groups, running d4 in parallel.
"""
function d4_itersolve(cnf::Vector{Vector{Int}}, groups::Vector{Vector{Int}}; d4_path::String="d4")
    num_vars = isempty(cnf) ? 0 : maximum(maximum(abs.(clause)) for clause in cnf if !isempty(clause))
    out_channel = Channel{Vector{Int}}(10000) 
    
    # Run the worker pool in the background
    errormonitor(Threads.@spawn begin
        try
            # Spawn one thread per geometric group
            Threads.@threads for group in groups
                if isempty(group)
                    continue
                end
                
                # Apply the geometric partitioning constraint
                local_cnf = copy(cnf)
                push!(local_cnf, group)
                
                cnf_file = tempname() * ".cnf"
                nnf_file = tempname() * ".nnf"
                
                try
                    open(cnf_file, "w") do io
                        println(io, "p cnf $num_vars $(length(local_cnf))")
                        for clause in local_cnf
                            println(io, join(clause, " "), " 0")
                        end
                    end
                    
                    run(pipeline(`$d4_path -dDNNF $cnf_file -out=$nnf_file`, stdout=devnull, stderr=devnull))
                    
                    if isfile(nnf_file) && filesize(nnf_file) > 0 
                        nodes, arcs = parse_nnf(nnf_file)
                        
                        in_degrees = Dict{Int, Int}(n => 0 for n in keys(nodes))
                        for edge_list in values(arcs), edge in edge_list
                            in_degrees[edge.target] += 1
                        end
                        
                        roots = [n for (n, d) in in_degrees if d == 0]
                        if !isempty(roots)
                            root_cont = (p) -> begin
                                is_assigned = fill(false, num_vars)
                                for lit in p
                                    is_assigned[abs(lit)] = true
                                end
                                unassigned = Int[]
                                for v in 1:num_vars
                                    if !is_assigned[v]
                                        push!(unassigned, v)
                                    end
                                end
                                stream_expanded_models!(p, unassigned, 1, out_channel)
                            end
                            
                            initial_path = Int[]
                            stream_paths!(roots[1], initial_path, nodes, arcs, root_cont)
                        end
                    end
                catch e
                    println(stderr, "Worker error on a group: ", e)
                finally
                    rm(cnf_file, force=true)
                    rm(nnf_file, force=true)
                end
            end
        finally
            close(out_channel) # Gracefully end the iteration when all threads finish
        end
    end)
    
    return out_channel
end

end # module