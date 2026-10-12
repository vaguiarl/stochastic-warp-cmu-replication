"""Build normalized-budget patch signs using exhaustive linear feasibility.

`mode=:source_weak` reproduces the C++ computational-replication convention:
nonnegative quantities, weak cross-budget signs, minimum relaxation <= tol.
`:strict` retains only cells with a strictly positive common cross-budget
margin. Comparing both is a geometry audit, not a change in observed data.
Prices are already normalized to expenditure one and must not be normalized
again. Sign -1 means below a budget, +1 above, and 0 marks the owner.
"""
function build_patches(P::AbstractMatrix; mode::Symbol=:source_weak,
                       tolerance::Float64=1e-7, minimum_quantity::Float64=0.0)
    mode in (:source_weak, :strict) || throw(ArgumentError("unknown geometry mode"))
    all(isfinite, P) && all(>(0), P) || throw(ArgumentError("prices must be finite and positive"))
    J, K = size(P)
    1 <= J <= 16 && K >= 2 || throw(ArgumentError("unsupported budget/commodity dimensions"))
    tolerance > 0 && minimum_quantity >= 0 || throw(ArgumentError("invalid feasibility settings"))
    rows = Vector{Vector{Int8}}()
    margins = Float64[]
    for own in 1:J
        other = filter(!=(own), 1:J)
        # Expenditure coordinates improve conditioning without changing q>=0.
        model = Model(HiGHS.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "threads", 1)
        set_optimizer_attribute(model, "primal_feasibility_tolerance", 1e-9)
        set_optimizer_attribute(model, "dual_feasibility_tolerance", 1e-9)
        @variable(model, y[1:K] >= 0)
        for k in 1:K
            set_lower_bound(y[k], minimum_quantity * P[own,k])
        end
        @constraint(model, sum(y) == 1)
        @variable(model, 0 <= delta <= 2)
        if mode == :strict
            @objective(model, Max, delta)
        else
            @objective(model, Min, delta)
        end
        for pattern in 0:(2^length(other)-1)
            signs = zeros(Int8, J)
            constraints = ConstraintRef[]
            for (bit, j) in enumerate(other)
                below = ((pattern >> (bit-1)) & 1) == 0
                signs[j] = below ? -1 : 1
                price_expr = sum(P[j,k]/P[own,k] * y[k] for k in 1:K)
                if mode == :strict
                    push!(constraints, below ? @constraint(model, price_expr + delta <= 1) :
                                               @constraint(model, price_expr - delta >= 1))
                else
                    push!(constraints, below ? @constraint(model, price_expr - delta <= 1) :
                                               @constraint(model, price_expr + delta >= 1))
                end
            end
            optimize!(model)
            status = termination_status(model)
            if status == JuMP.MOI.OPTIMAL
                margin = value(delta)
                accepted = mode == :strict ? margin > tolerance : margin <= tolerance
                if accepted
                    push!(rows, signs)
                    push!(margins, margin)
                end
            elseif status != JuMP.MOI.INFEASIBLE
                error("geometry LP failed with status $status")
            end
            for c in constraints
                delete(model, c)
            end
        end
    end
    isempty(rows) && error("empty budget geometry")
    X = reduce(vcat, permutedims.(rows))
    groups = _patch_groups(X)
    return X, Dict("mode"=>string(mode), "tolerance"=>tolerance,
        "minimum_quantity"=>minimum_quantity, "patch_counts"=>length.(groups),
        "total_patches"=>size(X,1), "normalization"=>"source prices; wealth one",
        "quantity_coordinate"=>"nonnegative; expenditure shares in LP",
        "accepted_feasibility_objectives"=>margins)
end
