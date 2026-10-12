"""Worker-local exact QP projection of paired bootstrap targets.

Each task owns an OSQP/JuMP model and processes a deterministic chunk. No
solver workspace is shared between tasks. Both models reuse the same fitted
probability draws; tightening origins remain model-specific.
"""
function _project_bootstrap_chunks(A, origin, targets, scale; workers=Threads.nthreads())
    B = size(targets,2)
    workers = min(max(1,workers), B, Threads.nthreads())
    # Dense certificate matrices and solver workspaces are replicated per task.
    workers = min(workers, max(1, fld(800_000,max(1,size(A,2)))))
    statistics = zeros(Float64, B)
    chunks = [collect(w:workers:B) for w in 1:workers]
    audits = Vector{Any}(undef, workers)
    function run_chunk(w)
        projector = DualConeProjector(A;residual_floor=-1.0)
        gaps, errors, duals = 0.0, 0.0, 0.0
        floors, almost = 0, 0
        minimum_residual = Inf
        for b in chunks[w]
            fit = project!(projector,view(targets,:,b);origin)
            statistics[b] = scale*fit.squared_distance
            gaps = max(gaps,fit.primal_dual_gap)
            errors = max(errors,fit.equality_error)
            duals = max(duals,fit.dual_violation)
            floors += fit.active_residual_floors > 0
            almost += fit.solver_status == "ALMOST_OPTIMAL"
            minimum_residual = min(minimum_residual,fit.minimum_residual)
        end
        audits[w] = (;gaps,errors,duals,floors,almost,minimum_residual)
    end
    if workers == 1
        run_chunk(1)
    else
        @sync for w in 1:workers
            Threads.@spawn run_chunk(w)
        end
    end
    return statistics, Dict("workers"=>workers,
        "maximum_primal_dual_gap"=>maximum(a.gaps for a in audits),
        "maximum_equality_error"=>maximum(a.errors for a in audits),
        "maximum_dual_violation"=>maximum(a.duals for a in audits),
        "floor_active_draws"=>sum(a.floors for a in audits),
        "almost_optimal_draws"=>sum(a.almost for a in audits),
        "minimum_residual"=>minimum(a.minimum_residual for a in audits))
end

"""Clopper–Pearson interval for finite-bootstrap tail frequency (not data CI)."""
function bootstrap_tail_interval(k::Integer, B::Integer; alpha=0.05)
    0 <= k <= B && B >= 1 && 0 < alpha < 1 || throw(ArgumentError("invalid binomial interval"))
    lower = k==0 ? 0.0 : quantile(Beta(k,B-k+1),alpha/2)
    upper = k==B ? 1.0 : quantile(Beta(k+1,B-k),1-alpha/2)
    return [lower,upper]
end

function compare_models_fast(pi::AbstractVector, X::AbstractMatrix,
        variance_bootstrap::AbstractMatrix, test_bootstrap::AbstractMatrix,
        annual_n::AbstractVector; degree::Int=3, zero_tolerance=1e-8,
        A_rum=nothing,A_warp=nothing,workers=Threads.nthreads(),
        mc_alpha=0.05, max_types=200_000,max_nodes=20_000_000)
    degree >= 1 || throw(ArgumentError("positive degree required"))
    B = size(test_bootstrap,2)
    B >= 2 && size(test_bootstrap,1)==length(pi)==size(X,1) ||
        throw(DimensionMismatch("test bootstrap dimensions"))
    all(isfinite,test_bootstrap) && all(isfinite,pi) || throw(ArgumentError("nonfinite probabilities"))
    isfinite(zero_tolerance) && zero_tolerance >= 0 || throw(ArgumentError("invalid tie tolerance"))
    length(annual_n)==size(X,2) && all(isfinite,annual_n) && all(>(0),annual_n) ||
        throw(ArgumentError("invalid annual sample sizes"))
    A_rum === nothing && (A_rum=enumerate_types_fast(X;restriction=:rum,max_types,max_nodes))
    A_warp === nothing && (A_warp=enumerate_types_fast(X;restriction=:warp,max_types,max_nodes))
    identical = A_rum == A_warp
    tau,tuning = tuning_parameter(variance_bootstrap,X)
    scale = sum(annual_n)/degree
    models = Dict{String,Any}()
    for (name,A) in (("rum",A_rum),("warp",A_warp))
        if name=="warp" && identical
            models[name] = deepcopy(models["rum"])
            models[name]["identical_cone_reused"] = true
            continue
        end
        projector = DualConeProjector(A;residual_floor=-1.0)
        origin = tau .* vec(mean(A;dims=2))
        observed = project!(projector,pi)
        tightened = project!(projector,pi;origin)
        targets = test_bootstrap .- pi .+ tightened.fitted
        boot,audit = _project_bootstrap_chunks(A,origin,targets,scale;workers)
        J = scale*observed.squared_distance
        tail = J<=zero_tolerance ? B : count(x->x>=J-zero_tolerance,boot)
        p = tail/B
        sorted = sort(boot)
        models[name] = Dict("type_count"=>size(A,2),
            "squared_distance"=>observed.squared_distance,"J"=>J,
            "p_value"=>p,"tail_count"=>tail,"bootstrap_repetitions"=>B,
            "p_value_mc_se"=>sqrt(p*(1-p)/B),
            "p_value_mc_interval"=>bootstrap_tail_interval(tail,B;alpha=mc_alpha),
            "mc_interval_alpha"=>mc_alpha,
            "critical_95"=>sorted[ceil(Int,.95B)],
            "critical_99"=>sorted[ceil(Int,.99B)],
            "fitted_probabilities"=>observed.fitted,
            "tightened_fitted_probabilities"=>tightened.fitted,
            "tightening_origin"=>origin,"bootstrap_statistics"=>boot,
            "qp_primal_dual_gap"=>observed.primal_dual_gap,
            "qp_equality_error"=>observed.equality_error,
            "observed_solver_status"=>observed.solver_status,
            "observed_active_residual_floors"=>observed.active_residual_floors,
            "bootstrap_qp_audit"=>audit,"identical_cone_reused"=>false)
    end
    gap = models["rum"]["J"]-models["warp"]["J"]
    gap >= -scale*1e-8 || error("nested observed distance check failed")
    return Dict("models"=>models,"tau"=>tau,"tuning"=>tuning,
        "sample_scale"=>scale,"paired_J_gap"=>gap,
        "variance_repetitions"=>size(variance_bootstrap,2),
        "bootstrap_repetitions"=>B,"identical_type_matrices"=>identical,
        "zero_tolerance"=>zero_tolerance,"weighting"=>"full patch coordinates; identity",
        "mixture_constraint"=>"source nonnegative cone; no simplex constraint",
        "tightening"=>"model-specific full-type centroid",
        "nested_observed_distance_verified"=>true,
        "parallelism"=>"worker-local independent OSQP models; paired shared probability draws")
end

"""Holm adjustments including unavailable tests as p=1, retaining full family size."""
function holm_adjust(p::AbstractVector)
    all(x->isfinite(x)&&0<=x<=1,p) || throw(ArgumentError("invalid p values"))
    M = length(p)
    order = sortperm(p)
    adjusted = ones(Float64,M)
    previous = 0.0
    for (rank,index) in enumerate(order)
        previous = max(previous,min(1.0,(M-rank+1)*p[index]))
        adjusted[index] = previous
    end
    return adjusted
end
