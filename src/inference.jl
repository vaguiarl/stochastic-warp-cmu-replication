"""Reusable full-coordinate identity-weighted cone QP, no simplex constraint.

Matches the mathematical criterion in RUMCG_54_BasisTeststat.m, using all
enumerated columns rather than column generation. OSQP replaces CPLEX.
The objective here is the squared distance (no half-factor).
"""
struct ConeProjector
    model::Model
    residual::Vector{VariableRef}
    mass::Vector{VariableRef}
    equation::Vector{ConstraintRef}
    A::Matrix{Float64}
    residual_floor::Union{Nothing,Float64}
end

function ConeProjector(A::AbstractMatrix; residual_floor::Union{Nothing,Float64}=nothing)
    I,H = size(A)
    I > 0 && H > 0 || throw(ArgumentError("empty type matrix"))
    all(x->x==0 || x==1, A) || throw(ArgumentError("nonbinary type matrix"))
    Af = Float64.(A)
    model = Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "eps_abs", 1e-9)
    set_optimizer_attribute(model, "eps_rel", 1e-9)
    set_optimizer_attribute(model, "max_iter", 200000)
    set_optimizer_attribute(model, "polish", true)
    @variable(model, e[1:I])
    if residual_floor !== nothing
        isfinite(residual_floor) || throw(ArgumentError("nonfinite residual floor"))
        for variable in e
            set_lower_bound(variable,residual_floor)
        end
    end
    @variable(model, v[1:H] >= 0)
    equations = @constraint(model, [i=1:I], e[i] + sum(Af[i,h]*v[h] for h=1:H) == 0)
    @objective(model, Min, sum(e[i]^2 for i=1:I))
    return ConeProjector(model, collect(e), collect(v), collect(equations), Af, residual_floor)
end

function project!(projector::ConeProjector, pi::AbstractVector;
                  origin::AbstractVector=zeros(length(pi)))
    I = size(projector.A,1)
    length(pi) == I == length(origin) || throw(DimensionMismatch("projection input"))
    all(isfinite,pi) && all(isfinite,origin) || throw(ArgumentError("nonfinite projection target"))
    for i in 1:I
        set_normalized_rhs(projector.equation[i], pi[i]-origin[i])
    end
    optimize!(projector.model)
    status = termination_status(projector.model)
    status == JuMP.MOI.OPTIMAL || error("cone QP failed: $status")
    mass = value.(projector.mass)
    residual = value.(projector.residual)
    fitted = projector.A * mass + origin
    mismatch = maximum(abs.(fitted + residual - pi))
    mismatch <= 1e-7 || error("QP equality residual $mismatch exceeds tolerance")
    minimum(mass) >= -1e-8 || error("negative type mass from QP")
    # Global cone optimality is independently checked against every column.
    effective_residual = dual.(projector.equation) ./ 2
    if projector.residual_floor === nothing
        maximum(abs.(effective_residual-residual)) <= 1e-6 ||
            error("QP residual stationarity failed")
    else
        floor = projector.residual_floor
        minimum(residual) >= floor-1e-7 || error("QP residual floor violated")
        floor_multiplier = residual-effective_residual
        minimum(floor_multiplier) >= -1e-6 || error("QP floor dual infeasibility")
        maximum(abs.(floor_multiplier .* (residual .- floor))) <= 1e-6 ||
            error("QP floor complementarity failed")
    end
    gradient = projector.A' * effective_residual
    maximum(gradient) <= 1e-6 || error("QP dual violation $(maximum(gradient))")
    abs(dot(mass,gradient)) <= 1e-5 || error("QP complementarity check failed")
    return (fitted=fitted, squared_distance=sum(abs2,residual),
            mass_sum=sum(mass), equality_error=mismatch,
            dual_violation=max(0.0,maximum(gradient)), minimum_residual=minimum(residual),
            active_residual_floors=projector.residual_floor===nothing ? 0 :
                count(x->x<=projector.residual_floor+1e-7,residual))
end

"""KS-style variance-based tuning, with denominator B as in MATLAB var(...,1)."""
function tuning_parameter(pi_bootstrap::AbstractMatrix, X::AbstractMatrix)
    I,B = size(pi_bootstrap)
    B >= 2 && I == size(X,1) || throw(ArgumentError("invalid variance bootstrap"))
    all(isfinite,pi_bootstrap) || throw(ArgumentError("nonfinite variance bootstrap"))
    variances = vec(var(pi_bootstrap; dims=2, corrected=false))
    groups = _patch_groups(X)
    nK = [sum(variances[g]) == 0 ? Inf : length(g)/sum(variances[g]) for g in groups]
    effective_n = minimum(nK)
    effective_n > 1 || error("KS tuning requires effective_n > 1; got $effective_n")
    tau = isinf(effective_n) ? 0.0 : sqrt(log(effective_n)/effective_n)
    return tau, Dict("effective_n"=>isfinite(effective_n) ? effective_n : nothing,
        "effective_n_by_budget"=>[isfinite(n) ? n : nothing for n in nK],
        "variance_denominator"=>B, "coordinate_variances"=>variances)
end

"""Paired RUM/WARP test with separately recomputed full-type tightening origins.

The two B-column bootstrap matrices are fresh stages: one for tau, one for
inference. They must have been estimated by refitting both CF stages with
fixed budgets and fixed evaluation median. No household data are returned.
"""
function compare_models(pi::AbstractVector, X::AbstractMatrix,
                        variance_bootstrap::AbstractMatrix, test_bootstrap::AbstractMatrix,
                        annual_n::AbstractVector; degree::Int=3,
                        zero_tolerance::Float64=1e-8, progress=nothing)
    degree >= 1 || throw(ArgumentError("degree must be positive"))
    isfinite(zero_tolerance) && zero_tolerance >= 0 || throw(ArgumentError("invalid zero tolerance"))
    B = size(test_bootstrap,2)
    B >= 2 && size(test_bootstrap,1) == length(pi) == size(X,1) ||
        throw(DimensionMismatch("test bootstrap dimensions"))
    all(isfinite,test_bootstrap) || throw(ArgumentError("nonfinite test bootstrap"))
    size(variance_bootstrap,2) == B || throw(ArgumentError("two stages must use the same B"))
    length(annual_n) == size(X,2) && all(isfinite,annual_n) && all(>(0), annual_n) ||
        throw(ArgumentError("annual sample sizes"))
    tau, tuning = tuning_parameter(variance_bootstrap,X)
    scale = sum(annual_n)/degree # MATLAB source; C++ used sum N without /degree.
    report = Dict{String,Any}()
    distances = Dict{Symbol,Float64}()
    fitted_bootstrap = Dict{Symbol,Vector{Float64}}()
    for model in (:rum,:warp)
        A = enumerate_types(X;restriction=model)
        size(A,2) > 0 || error("empty $model model")
        # Literal MATLAB master bound: e >= -1. Audit whether it ever binds.
        projector = DualConeProjector(A;residual_floor=-1.0)
        origin = tau .* vec(mean(A;dims=2))
        observed = project!(projector,pi)
        tightened = project!(projector,pi;origin)
        J = scale * observed.squared_distance
        J_bootstrap = zeros(B)
        active_floor_bootstraps = 0
        minimum_bootstrap_residual = Inf
        maximum_bootstrap_gap = 0.0
        almost_optimal_bootstraps = 0
        for b in 1:B
            recentered = test_bootstrap[:,b] - pi + tightened.fitted
            fit = project!(projector,recentered;origin)
            J_bootstrap[b] = scale * fit.squared_distance
            active_floor_bootstraps += fit.active_residual_floors > 0
            minimum_bootstrap_residual = min(minimum_bootstrap_residual,fit.minimum_residual)
            maximum_bootstrap_gap = max(maximum_bootstrap_gap,fit.primal_dual_gap)
            almost_optimal_bootstraps += fit.solver_status=="ALMOST_OPTIMAL"
            b % 100 == 0 && progress !== nothing && progress("$model QP $b/$B")
        end
        distances[model] = observed.squared_distance
        fitted_bootstrap[model] = J_bootstrap
        # Correct upper tail, explicit ties; do not port C++'s integer-division bug.
        p = J <= zero_tolerance ? 1.0 : count(x->x>=J-zero_tolerance,J_bootstrap)/B
        sorted = sort(J_bootstrap)
        report[string(model)] = Dict("type_count"=>size(A,2),
            "type_cache_key"=>type_cache_key(X;restriction=model),
            "squared_distance"=>observed.squared_distance, "J"=>J,
            "p_value"=>p, "tail_count"=>J<=zero_tolerance ? B : count(x->x>=J-zero_tolerance,J_bootstrap),
            "critical_95"=>sorted[ceil(Int,.95B)], "critical_99"=>sorted[ceil(Int,.99B)],
            "fitted_mass_sum"=>observed.mass_sum,
            "fitted_probabilities"=>observed.fitted,
            "tightened_fitted_probabilities"=>tightened.fitted,
            "tightening_origin"=>origin, "bootstrap_statistics"=>J_bootstrap,
            "p_value_mc_se"=>sqrt(p*(1-p)/B),
            "qp_equality_error"=>observed.equality_error,
            "qp_dual_violation"=>observed.dual_violation,
            "minimum_observed_residual"=>observed.minimum_residual,
            "observed_active_residual_floors"=>observed.active_residual_floors,
            "bootstraps_with_active_residual_floor"=>active_floor_bootstraps,
            "minimum_bootstrap_residual"=>minimum_bootstrap_residual,
            "qp_primal_dual_gap"=>observed.primal_dual_gap,
            "maximum_bootstrap_primal_dual_gap"=>maximum_bootstrap_gap,
            "almost_optimal_bootstrap_statuses"=>almost_optimal_bootstraps,
            "observed_solver_status"=>observed.solver_status)
    end
    # Larger WARP cone cannot be farther in the identical observed metric.
    distances[:warp] <= distances[:rum] + 1e-8 || error("nested-model distance check failed")
    return Dict("models"=>report, "tau"=>tau, "tuning"=>tuning,
        "sample_scale"=>scale, "sample_scale_convention"=>"sum(annual_n)/degree; MATLAB CG source",
        "bootstrap_repetitions"=>B, "zero_tolerance"=>zero_tolerance,
        "weighting"=>"full patch coordinates; identity",
        "mixture_constraint"=>"nonnegative cone; no sum-to-one constraint",
        "source_residual_floor"=>-1.0,
        "qp_solver"=>"OSQP low-dimensional exact master dual; absolute/relative tolerances 1e-9; all type constraints; polishing; primal/dual certificate",
        "tightening"=>"tau times all-type centroid; separately regenerated for each model",
        "bootstrap_pairing"=>"identical refits shared by models; distinct variance/test stages",
        "nested_observed_distance_verified"=>true)
end
