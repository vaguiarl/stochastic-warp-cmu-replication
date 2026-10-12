"""Versioned convention used by the deterministic type-cache identity."""
const TYPE_ENUMERATOR_VERSION = "ks-warp-julia-types-v1"
const TYPE_SIGN_CONVENTION =
    "one-own-zero;off-budget-pm1;source-to-target-X[target,source]<0;" *
    "no-self-edges;budget-then-row-order;complete-enumeration"

function _check_type_restriction(restriction::Symbol)
    restriction in (:warp, :rum, :unrestricted) ||
        throw(ArgumentError("restriction must be :warp, :rum, or :unrestricted"))
    return restriction
end

function _patch_groups(X::AbstractMatrix)
    Base.require_one_based_indexing(X)
    npatches, nbudgets = size(X)
    npatches > 0 && nbudgets > 0 ||
        throw(ArgumentError("the sign matrix must have patches and budgets"))
    groups = [Int[] for _ in 1:nbudgets]
    for row in 1:npatches
        owner = 0
        for budget in 1:nbudgets
            sign = X[row, budget]
            if sign == 0
                owner == 0 || throw(ArgumentError("each patch must have exactly one own-budget zero"))
                owner = budget
            elseif sign != -1 && sign != 1
                throw(ArgumentError("off-budget patch signs must be -1 or +1"))
            end
        end
        owner != 0 || throw(ArgumentError("each patch must have exactly one own-budget zero"))
        push!(groups[owner], row)
    end
    all(group -> !isempty(group), groups) ||
        throw(ArgumentError("every budget must have at least one patch"))
    return groups
end

function _check_square_graph(graph::AbstractMatrix{Bool})
    Base.require_one_based_indexing(graph)
    size(graph, 1) == size(graph, 2) || throw(ArgumentError("the graph must be square"))
    return size(graph, 1)
end

"""Return whether a Boolean adjacency matrix has a cycle between two distinct vertices.

Diagonal entries do not constitute two-cycles. Use `has_directed_cycle` to detect
self-loops as well as cycles of arbitrary length.
"""
function has_two_cycle(graph::AbstractMatrix{Bool})
    n = _check_square_graph(graph)
    for source in 1:n, target in (source + 1):n
        graph[source, target] && graph[target, source] && return true
    end
    return false
end

"""Return whether a Boolean adjacency matrix has any directed cycle, including a self-loop.

This depth-first search is independent of the Cartesian/Kahn oracle in the tests.
"""
function has_directed_cycle(graph::AbstractMatrix{Bool})
    n = _check_square_graph(graph)
    color = zeros(Int8, n) # 0: unseen; 1: current DFS stack; 2: finished.
    function visit(source::Int)
        color[source] = 1
        for target in 1:n
            graph[source, target] || continue
            color[target] == 1 && return true
            color[target] == 0 && visit(target) && return true
        end
        color[source] = 2
        return false
    end
    for source in 1:n
        color[source] == 0 && visit(source) && return true
    end
    return false
end

# RP[source budget, target budget] follows the original KS construction:
# the target's chosen patch lies weakly below the source budget. Off-budget
# comparisons are strict on the validated open patches, so <= 0 equals < 0.
function _selected_patch_graph(X::AbstractMatrix, selected::Vector{Int}, level::Int)
    graph = falses(level, level)
    for source in 1:level, target in 1:level
        source == target && continue
        graph[source, target] = X[selected[target], source] < 0
    end
    return graph
end

"""Enumerate binary deterministic patch-choice columns for an open-patch sign matrix.

Rows are patches and columns are budgets. Every row has exactly one zero,
identifying its own budget; all other entries must be -1 or +1. Rows may be
interleaved across budgets. Each output column chooses one patch per budget.

`:warp` rules out mutual revealed preference (two-cycles), `:rum` rules out
all directed cycles, and `:unrestricted` supplies the Cartesian baseline.
Enumeration is deterministic in budget order and then input-row order.
The type limit aborts with an error instead of silently truncating the model.

These are finite deterministic restrictions. This function does not estimate
patch probabilities, reproduce an empirical bootstrap, or handle atoms on
budget intersections.
"""
function enumerate_types(X::AbstractMatrix; restriction::Symbol=:rum, max_types::Int=10^7)
    _check_type_restriction(restriction)
    max_types > 0 || throw(ArgumentError("max_types must be positive"))
    groups = _patch_groups(X)
    npatches, nbudgets = size(X)
    selected = zeros(Int, nbudgets)
    assignments = Vector{Vector{Int}}()

    function descend(level::Int)
        for row in groups[level]
            selected[level] = row
            admissible = true
            if restriction == :warp
                # Earlier selections have already passed; only new pairs need checking.
                for earlier in 1:(level - 1)
                    if X[row, earlier] < 0 && X[selected[earlier], level] < 0
                        admissible = false
                        break
                    end
                end
            elseif restriction == :rum
                admissible = !has_directed_cycle(_selected_patch_graph(X, selected, level))
            end
            admissible || continue
            if level == nbudgets
                length(assignments) < max_types || throw(ArgumentError(
                    "type count exceeds max_types=$max_types; enumeration aborted without truncation"))
                push!(assignments, copy(selected))
            else
                descend(level + 1)
            end
        end
    end

    descend(1)
    A = zeros(Int8, npatches, length(assignments))
    for (column, rows) in enumerate(assignments), row in rows
        A[row, column] = 1
    end
    return A
end

"""Hash ordered signs, dimensions, model restriction, and the versioned open-patch convention.

Numerically identical integral and floating-point sign matrices intentionally
share a key. Row or budget permutations change the key because the binary
matrix uses input patch-row coordinates.
"""
function type_cache_key(X::AbstractMatrix; restriction::Symbol=:rum)
    _check_type_restriction(restriction)
    _patch_groups(X) # Never assign a reusable identity to an invalid geometry.
    buffer = IOBuffer()
    print(buffer, TYPE_ENUMERATOR_VERSION, '\n', TYPE_SIGN_CONVENTION, '\n',
          restriction, '\n', size(X, 1), ',', size(X, 2), '\n')
    for row in axes(X, 1), budget in axes(X, 2)
        write(buffer, Int8(X[row, budget]))
    end
    return bytes2hex(SHA.sha256(take!(buffer)))
end
