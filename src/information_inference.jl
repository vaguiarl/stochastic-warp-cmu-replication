"""
    information_distance_gain(z, covariance, rum_projector, cmu_projector)

Squared Mahalanobis distance to the smaller normalized probability polytope,
minus distance to the larger one. The projectors must describe nested sets.
No type-count degrees of freedom or diagonal-metric KS calibration is used.
Callable projectors are supported for independent analytic validation.
"""
project_information!(projector::Function, z; covariance=nothing) = projector(z, covariance)

function information_distance_gain(z, covariance, rum_projector, cmu_projector;
                                   absolute_tolerance=1e-10)
    r = project_information!(rum_projector, z; covariance)
    c = project_information!(cmu_projector, z; covariance)
    all(isfinite, (r.squared_distance, c.squared_distance)) ||
        error("nonfinite information distance")
    min(r.squared_distance, c.squared_distance) >= -absolute_tolerance ||
        error("negative squared information distance")
    er = hasproperty(r, :objective_error_bound) ? r.objective_error_bound : 0.0
    ec = hasproperty(c, :objective_error_bound) ? c.objective_error_bound : 0.0
    err = Float64(er + ec + absolute_tolerance)
    raw = r.squared_distance - c.squared_distance
    raw >= -err || error("nested distance ordering failed beyond certificate tolerance")
    # This only removes solver roundoff certified by the projection bounds.
    gap = abs(raw) <= err ? 0.0 : raw
    return (; gap, raw_gap=raw, rum=r, cmu=c, error_bound=err)
end

function _information_projection_summary(p)
    out = Dict{String,Any}("squared_distance"=>p.squared_distance,
        "fitted_contrast"=>p.fitted_contrast)
    for k in (:fitted_probability, :status, :mass_sum, :equality_error,
              :nonnegativity_error, :support_violation, :add_up_error,
              :primal_dual_gap, :objective_error_bound, :mass_correction_l1,
              :budget_adding_up_error, :probability_nonnegative_violation,
              :stationarity_error, :support_variational_gap,
              :solver_status, :certified, :covariance_condition)
        hasproperty(p, k) && (out[string(k)] = getproperty(p, k))
    end
    return out
end

"""
    numerical_information_test(zhat, Sigmahat, zstars, Sigmastars; rate,
        rum_projector, cmu_projector, epsilons=[.5,1,2]/sqrt(rate), alpha=.05)

Joint second-order numerical-delta bootstrap of the nested distance gain.
`Sigmahat` and each bootstrap covariance estimate are covariances of the
*rate-scaled* probability estimator. Bootstrap columns are unscaled estimates.
The observed statistic is `rate^2 * (Q_R - Q_C)`.

Joint perturbations use (z, log(Sigma)) coordinates so every perturbed metric
remains positive definite. No ridge, singular-direction deletion, failed-draw
replacement, or clipping of finite signed derivative estimates is performed.
Pointwise asymptotic validity requires a joint estimator bootstrap, negligible
bias, and epsilon -> 0 with rate*epsilon -> infinity. This routine checks the
numerics, not those statistical assumptions. Epsilon sensitivity is returned,
not used to choose a favorable reported result.
"""
function numerical_information_test(zhat, Sigmahat, zstars, Sigmastars;
        rate, rum_projector, cmu_projector,
        epsilons=[0.5, 1.0, 2.0] ./ sqrt(rate), alpha=0.05,
        absolute_tolerance=1e-10, include_draws=true)
    d = length(zhat)
    size(Sigmahat) == (d,d) || throw(DimensionMismatch("base covariance"))
    size(zstars,1) == d || throw(DimensionMismatch("bootstrap probability coordinates"))
    draws = size(zstars,2)
    draws >= 2 || throw(ArgumentError("at least two joint bootstrap draws required"))
    size(Sigmastars) == (d,d,draws) || throw(DimensionMismatch("joint covariance draws"))
    isfinite(rate) && rate > 0 || throw(ArgumentError("rate must be positive finite"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0,1)"))
    all(isfinite, zhat) && all(isfinite,zstars) || error("nonfinite probability estimates")
    eps = Float64.(collect(epsilons))
    !isempty(eps) && all(x->isfinite(x) && x>0,eps) ||
        throw(ArgumentError("epsilon grid must be finite and positive"))
    Lhat = matrix_log_spd(Sigmahat)
    Lstars = Array{Float64}(undef,d,d,draws)
    for b in 1:draws
        # Validates every covariance draw. No draws are silently discarded.
        Lstars[:,:,b] = matrix_log_spd(view(Sigmastars,:,:,b))
    end
    point = information_distance_gain(zhat, Sigmahat, rum_projector, cmu_projector;
                                     absolute_tolerance)
    observed = rate^2 * point.gap
    summaries = Dict{String,Any}[]
    for e in eps
        stats = zeros(draws)
        uncertainty = zeros(draws)
        max_mass_correction = 0.0
        max_adding_up_error = 0.0
        solve_status = Dict{String,Int}()
        for b in 1:draws
            z = zhat .+ (e*rate) .* (view(zstars,:,b) .- zhat)
            L = Lhat .+ (e*rate) .* (view(Lstars,:,:,b) .- Lhat)
            S = matrix_exp_symmetric(L)
            perturbed = information_distance_gain(z,S,rum_projector,cmu_projector;
                                                absolute_tolerance)
            # The finite numerical derivative can be negative. Preserve it.
            stats[b] = (perturbed.gap - point.gap) / e^2
            uncertainty[b] = (perturbed.error_bound + point.error_bound) / e^2
            for fit in (perturbed.rum,perturbed.cmu)
                if hasproperty(fit,:mass_correction_l1)
                    max_mass_correction = max(max_mass_correction,fit.mass_correction_l1)
                    max_adding_up_error = max(max_adding_up_error,fit.budget_adding_up_error)
                end
                status = hasproperty(fit,:solver_status) ? fit.solver_status : "analytic_projection"
                solve_status[status] = get(solve_status,status,0)+1
            end
        end
        # Roundoff-only ties use a conservative upper-tail convention. A zero
        # observed statistic cannot provide evidence of strictly smaller fit.
        comparison_tolerance = max(rate^2*point.error_bound, maximum(uncertainty))
        exceedances = observed == 0 ? draws :
            count(i->stats[i] + comparison_tolerance >= observed,eachindex(stats))
        p = (exceedances+1)/(draws+1)
        critical = max(0.0, quantile(stats,1-alpha))
        row = Dict{String,Any}("epsilon"=>e,"rate_times_epsilon"=>rate*e,
            "p_value_mc"=>p,"exceedances"=>exceedances,
            "p_value_mc_interval"=>bootstrap_tail_interval(exceedances,draws),
            "bootstrap_repetitions"=>draws,"critical_value"=>critical,
            "reject"=>(observed>comparison_tolerance && p<=alpha),
            "quantile_reject_diagnostic"=>(observed>critical+comparison_tolerance),
            "negative_numerical_derivative_draws"=>count(x->x<0,stats),
            "minimum_numerical_derivative"=>minimum(stats),
            "maximum_projection_uncertainty"=>maximum(uncertainty),
            "maximum_mass_certificate_correction_l1"=>max_mass_correction,
            "maximum_budget_adding_up_error"=>max_adding_up_error,
            "solve_status_counts"=>solve_status,
            "comparison_tolerance"=>comparison_tolerance)
        include_draws && (row["bootstrap_statistics"] = stats)
        push!(summaries,row)
    end
    return Dict{String,Any}("status"=>"computed_conditional_experimental",
        "statistic"=>observed,"point_gap"=>point.gap,"raw_point_gap"=>point.raw_gap,
        "point_gap_numerical_error_bound"=>point.error_bound,
        "point_projection"=>Dict("rum"=>_information_projection_summary(point.rum),
                                  "cmu"=>_information_projection_summary(point.cmu)),
        "epsilon_results"=>summaries,
        "protocol"=>Dict("null"=>"equal population minimum full-covariance distance",
            "alternative"=>"CMU/WARP has strictly smaller population distance",
            "rate"=>rate,"alpha"=>alpha,"bootstrap_repetitions"=>draws,
            "covariance_chart"=>"symmetric matrix logarithm; joint perturbation",
            "calibration"=>"second-order numerical-delta bootstrap",
            "tail_convention"=>"(1+upper-tail exceedances)/(B+1); conservative numerical ties",
            "epsilon_selection"=>"prespecified sensitivity grid; no outcome-based selection",
            "asymptotic_validity_verified_by_this_routine"=>false,
            "failed_or_replaced_draws"=>0))
end
