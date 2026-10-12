"""Exact bit-mask type enumeration, with complete-output and work limits.

Uses the same direct affordability graph as enumerate_types. For RUM an
incremental transitive closure detects a cycle created by the new vertex.
Closure is never used for the WARP check, which tests direct two-cycles only.
"""
function enumerate_types_fast(X::AbstractMatrix; restriction::Symbol=:rum,
        max_types::Int=200_000, max_nodes::Int=20_000_000)
    _check_type_restriction(restriction)
    max_types > 0 && max_nodes > 0 || throw(ArgumentError("positive enumeration limits required"))
    groups = _patch_groups(X)
    I, J = size(X)
    J <= 63 || throw(ArgumentError("bit-mask enumeration supports at most 63 budgets"))
    below = zeros(UInt64, I)
    for row in 1:I, j in 1:J
        X[row,j] < 0 && (below[row] |= UInt64(1) << (j-1))
    end
    selected = zeros(Int, J)
    closures = zeros(UInt64, J, J+1)
    assignments = Vector{Vector{Int}}()
    nodes = Ref(0)
    function descend(level)
        active = (UInt64(1) << (level-1))-1
        newbit = UInt64(1) << (level-1)
        for row in groups[level]
            nodes[] += 1
            nodes[] <= max_nodes || throw(ArgumentError("type search exceeds max_nodes=$max_nodes; no truncated model returned"))
            incoming = below[row] & active
            outgoing = zero(UInt64)
            for old in 1:level-1
                below[selected[old]] & newbit != 0 && (outgoing |= UInt64(1) << (old-1))
            end
            if restriction == :warp
                incoming & outgoing != 0 && continue
            elseif restriction == :rum
                reached = outgoing
                for old in 1:level-1
                    outgoing & (UInt64(1) << (old-1)) != 0 &&
                        (reached |= closures[old,level])
                end
                reached & incoming != 0 && continue
                for old in 1:level-1
                    r = closures[old,level]
                    pred = incoming & (UInt64(1) << (old-1)) != 0 || r & incoming != 0
                    closures[old,level+1] = pred ? r | reached | newbit : r
                end
                closures[level,level+1] = reached
            end
            selected[level] = row
            if level == J
                length(assignments) < max_types || throw(ArgumentError("type count exceeds max_types=$max_types; no truncated model returned"))
                push!(assignments, copy(selected))
            else
                descend(level+1)
            end
        end
    end
    descend(1)
    A = zeros(Int8, I, length(assignments))
    for (column, rows) in enumerate(assignments), row in rows
        A[row,column] = 1
    end
    return A
end
