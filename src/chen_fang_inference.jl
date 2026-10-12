"""Numerical distance envelopes; Float64 primal/dual checks are not rigorous intervals."""
function _chen_fang_distance_envelope(fit, tolerance)
    hasproperty(fit, :certified) && !fit.certified && error("uncertified projection")
    q = Float64(fit.squared_distance)
    isfinite(q) && q >= 0 || throw(ArgumentError("invalid squared distance"))
    err = hasproperty(fit, :objective_error_bound) ? Float64(fit.objective_error_bound) : 0.0
    isfinite(err) && err >= 0 || throw(ArgumentError("invalid objective error bound"))
    lo = hasproperty(fit, :objective_lower_bound) ? Float64(fit.objective_lower_bound) : q-err
    hi = hasproperty(fit, :objective_upper_bound) ? Float64(fit.objective_upper_bound) : q+err
    all(isfinite, (lo,hi)) && lo <= hi+tolerance && lo <= q+tolerance &&
        q <= hi+tolerance || throw(ArgumentError("inconsistent objective envelope"))
    return (;value=q, lower=max(0.0,lo-tolerance), upper=max(0.0,hi+tolerance))
end

function _chen_fang_gap_envelope(z, S, rp, cp, tolerance)
    r = project_information!(rp,z;covariance=S)
    c = project_information!(cp,z;covariance=S)
    re = _chen_fang_distance_envelope(r,tolerance)
    ce = _chen_fang_distance_envelope(c,tolerance)
    raw = re.value-ce.value
    raw_lower,raw_upper = re.lower-ce.upper,re.upper-ce.lower
    raw_upper >= -tolerance || error("projectors violate nested distance ordering")
    lower = max(0.0,raw_lower)
    upper = max(0.0,raw_upper)
    lower <= upper+tolerance || error("inconsistent nested gap envelope")
    # Keep the raw central gap: snapping small gaps to zero corrupts the
    # uncertainty of differences divided by epsilon^2. Nonnegative envelope
    # clipping uses ONLY the maintained nesting of the exact mathematical sets.
    return (;raw,lower,upper,raw_lower,raw_upper,rum=r,cmu=c)
end

"""
    chen_fang_fixed_metric_test(zhat, zstars; rate, rum_projector, cmu_projector,
        metric_covariance=I, epsilons=nothing, alpha=.05, include_draws=true,
        workers=1, projector_factory=nothing)

Chen--Fang (2019), Theorem 3.3/Proposition 3.1, numerical derivative (38),
applied to phi(z)=d_R(z)^2-d_C(z)^2 for supplied nonempty nested convex
probability polytopes under ONE fixed positive-definite metric. The default
metric is identity, NOT an inverse of an estimated sampling covariance.

The observed statistic is rate^2*phi(zhat). For G_b=rate*(zstar_b-zhat),
the signed derivative is [phi(zhat+epsilon*G_b)-phi(zhat)]/epsilon^2.
There is no extra factor 1/2 in Chen--Fang's convention. Default epsilon
multipliers [.5,1,2]/sqrt(rate) have epsilon->0 and rate*epsilon->infinity.
Singular sampling covariance is allowed; no coordinate/rank truncation occurs.

Every comparison uses the observed LOWER numerical gap versus the per-draw
UPPER derivative envelope, accounting for error in BOTH subtracted gaps.
Signed central derivatives, negative draws and ties remain visible. Envelopes
are floating-point numerical checks, not outward-rounded interval certificates.

Statistical validity remains conditional on a valid joint estimator bootstrap,
negligible bias, a suitable rate, second-order directional differentiability,
and the critical-value regularity in the paper. A nonpositive or numerically
unresolved central critical value is flagged, with NO inferential rejection;
the raw Monte Carlo tail is then a diagnostic, not an established p-value.
MC+1 and ties do not solve the degenerate-limit inference problem.

Parallel callers must supply a factory returning an independent projector pair
for each worker. Mutable QP models are never shared. Any failed draw/solve
aborts the batch with its epsilon/draw context; none are replaced or omitted.
"""
function chen_fang_fixed_metric_test(zhat::AbstractVector,zstars::AbstractMatrix;
        rate,rum_projector,cmu_projector,metric_covariance=I,epsilons=nothing,
        alpha=.05,absolute_tolerance=1e-10,include_draws=true,
        workers::Integer=1,projector_factory=nothing)
    r = _weighted_gain_rate(rate)
    tolerance = _weighted_gain_absolute_tolerance(absolute_tolerance)
    d,draws = length(zhat),size(zstars,2)
    d>0 && size(zstars,1)==d || throw(DimensionMismatch("joint estimator coordinates"))
    draws>=2 || throw(ArgumentError("at least two joint bootstrap draws required"))
    all(isfinite,zhat) && all(isfinite,zstars) || throw(ArgumentError("nonfinite estimator/bootstrap"))
    0<alpha<1 || throw(ArgumentError("alpha must be in (0,1)"))
    1<=workers<=Threads.nthreads() || throw(ArgumentError("workers must fit available Julia threads"))
    workers==1 || projector_factory isa Function || throw(ArgumentError("parallel projector factory required"))
    S = _fixed_gain_metric(metric_covariance,d)
    eps = Float64.(collect(_weighted_gain_epsilons(epsilons,r)))
    !isempty(eps) && all(e->isfinite(e) && e>0 && isfinite(e^2) && e^2>0 &&
        isfinite(e*r),eps) || throw(ArgumentError("finite representable positive epsilon grid required"))
    point = _chen_fang_gap_envelope(zhat,S,rum_projector,cmu_projector,tolerance)
    observed = r^2*max(0.0,point.raw)
    observed_lower,observed_upper = r^2*point.lower,r^2*point.upper
    all(isfinite,(observed,observed_lower,observed_upper)) || error("nonfinite observed statistic")
    noise_energy = [r^2*sum(abs2,view(zstars,:,b).-zhat) for b in 1:draws]
    all(isfinite,noise_energy) || error("nonfinite centered joint noise norm")
    W = S\Matrix{Float64}(I,d,d)
    weighted_noise_energy = [begin
        increment = view(zstars,:,b).-zhat
        r^2*dot(increment,W*increment)
    end for b in 1:draws]
    all(isfinite,weighted_noise_energy) || error("nonfinite centered weighted joint noise norm")
    pairs = workers==1 ? [(rum_projector,cmu_projector)] :
        [projector_factory(w) for w in 1:workers]
    all(pair->length(pair)==2,pairs) || throw(ArgumentError("factory must return projector pairs"))
    models = [projector.model for pair in pairs for projector in pair
        if projector isa InformationProjector]
    length(unique(objectid.(models)))==length(models) ||
        throw(ArgumentError("parallel workers cannot share mutable QP models"))
    order = ceil(Int,(1-alpha)*(draws+1))
    rows = Dict{String,Any}[]
    for e in eps
        central,lower,upper = zeros(draws),zeros(draws),zeros(draws)
        corrections,adding_up,objective_width = zeros(draws),zeros(draws),zeros(draws)
        statuses = Vector{Tuple{String,String}}(undef,draws)
        function run_worker(w)
            rp,cp = pairs[w]
            for b in w:workers:draws
                try
                    z = zhat .+ (e*r).*(view(zstars,:,b).-zhat)
                    all(isfinite,z) || error("nonfinite numerical perturbation")
                    perturbed = _chen_fang_gap_envelope(z,S,rp,cp,tolerance)
                    central[b] = (perturbed.raw-point.raw)/e^2
                    lower[b] = (perturbed.lower-point.upper)/e^2
                    upper[b] = (perturbed.upper-point.lower)/e^2
                    all(isfinite,(central[b],lower[b],upper[b])) || error("nonfinite derivative")
                    # The raw numerical gap can be slightly negative even
                    # when the certified exact gap is zero by nesting. Check
                    # this diagnostic against UNCLIPPED objective intervals;
                    # inference still uses the nesting-clipped exact bounds.
                    raw_lower = (perturbed.raw_lower-point.raw_upper)/e^2
                    raw_upper = (perturbed.raw_upper-point.raw_lower)/e^2
                    raw_lower<=central[b]+tolerance/e^2 && central[b]<=raw_upper+tolerance/e^2 ||
                        error("central derivative outside raw numerical envelope")
                    objective_width[b] = upper[b]-lower[b]
                    isfinite(objective_width[b]) && objective_width[b]>=0 ||
                        error("nonfinite or reversed derivative envelope width")
                    for fit in (perturbed.rum,perturbed.cmu)
                        hasproperty(fit,:mass_correction_l1) &&
                            (corrections[b]=max(corrections[b],fit.mass_correction_l1))
                        hasproperty(fit,:budget_adding_up_error) &&
                            (adding_up[b]=max(adding_up[b],fit.budget_adding_up_error))
                    end
                    statuses[b] = tuple((hasproperty(fit,:solver_status) ?
                        string(fit.solver_status) : "analytic_projection" for fit in
                        (perturbed.rum,perturbed.cmu))...)
                catch err
                    throw(ErrorException("Chen--Fang epsilon=$e draw=$b failed without replacement: $(sprint(showerror,err))"))
                end
            end
        end
        if workers==1
            run_worker(1)
        else
            @sync for w in 1:workers
                Threads.@spawn run_worker(w)
            end
        end
        central_critical = order>draws ? Inf : sort(central)[order]
        upper_critical = order>draws ? Inf : sort(upper)[order]
        maximum_width = maximum(objective_width)
        resolved = isfinite(central_critical) && central_critical>maximum_width && central_critical>0
        tail = observed_lower<=0 ? draws : count(x->x>=observed_lower,upper)
        p = (tail+1)/(draws+1)
        rejection = resolved && observed_lower>upper_critical && p<=alpha
        status_counts = Dict{String,Int}()
        for pair in statuses,status in pair
            status_counts[status] = get(status_counts,status,0)+1
        end
        row = Dict{String,Any}(
            "epsilon"=>e,"rate_times_epsilon"=>r*e,"epsilon_multiplier"=>e*sqrt(r),
            "p_value_mc"=>p,"p_value_mc_interval"=>bootstrap_tail_interval(tail,draws),
            "exceedances"=>tail,"bootstrap_repetitions"=>draws,"reject"=>rejection,
            "critical_value"=>isfinite(upper_critical) ? upper_critical : nothing,
            "central_critical_value"=>isfinite(central_critical) ? central_critical : nothing,
            "critical_value_positive_resolved"=>resolved,
            "calibration_status"=>resolved ? "computed_conditional_positive_critical" :
                "degenerate_or_unresolved_critical_no_rejection",
            "inferential_p_value_conditional"=>resolved ? p : nothing,
            "raw_mc_tail_not_inferential_when_critical_unresolved"=>!resolved,
            "negative_numerical_derivative_draws"=>count(<(0),central),
            "central_diagnostic_outside_nesting_clipped_envelope_draws"=>
                count(b->central[b]<lower[b] || central[b]>upper[b],1:draws),
            "numerical_envelope_contains_zero_draws"=>count(b->lower[b]<=0<=upper[b],1:draws),
            "minimum_numerical_derivative"=>minimum(central),
            "maximum_numerical_derivative"=>maximum(central),
            "maximum_projection_uncertainty"=>maximum_width,
            "maximum_mass_certificate_correction_l1"=>maximum(corrections),
            "maximum_budget_adding_up_error"=>maximum(adding_up),
            "solve_status_counts"=>status_counts,"failed_or_replaced_draws"=>0,
            "quantile_reject_diagnostic"=>rejection,
            "perturbation_norm_summary"=>Dict("median"=>e*sqrt(median(noise_energy)),
                "percentile_95"=>e*sqrt(quantile(noise_energy,.95)),
                "maximum"=>e*sqrt(maximum(noise_energy))),
            "central_derivative_above_joint_norm_diagnostic_draws"=>
                count(b->lower[b]>weighted_noise_energy[b],1:draws),
            "joint_norm_bound_diagnostic_uses_fixed_metric"=>true,
            "central_tail_diagnostic"=>(count(x->x>=observed,central)+1)/(draws+1))
        if include_draws
            row["bootstrap_statistics"] = central
            row["bootstrap_statistics_lower"] = lower
            row["bootstrap_statistics_upper"] = upper
        end
        push!(rows,row)
    end
    return Dict{String,Any}("status"=>"computed_conditional_chen_fang",
        "method"=>"chen_fang_fixed_metric_numerical_second_order",
        "statistic"=>observed,"statistic_lower_numerical"=>observed_lower,
        "statistic_upper_numerical"=>observed_upper,"point_gap"=>max(0.0,point.raw),
        "raw_point_gap"=>point.raw,"point_gap_lower_numerical"=>point.lower,
        "point_gap_upper_numerical"=>point.upper,
        "point_projection"=>Dict("rum"=>_information_projection_summary(point.rum),
            "cmu"=>_information_projection_summary(point.cmu)),"epsilon_results"=>rows,
        "metric"=>Dict("metric_covariance"=>S,"fixed_within_test"=>true,
            "sampling_covariance_inverse_used"=>false,"ridge_applied"=>false,
            "coordinates_retained"=>d,"nullspace_coordinates_deleted"=>false),
        "protocol"=>Dict("null"=>"equal population minimum distance under fixed metric",
            "alternative"=>"CMU has strictly smaller population fixed-metric distance",
            "reference"=>"Chen and Fang (2019), Theorem 3.3, Proposition 3.1, equation (38)",
            "reference_url"=>"https://arxiv.org/abs/1901.04861",
            "rate"=>r,"alpha"=>alpha,"bootstrap_repetitions"=>draws,
            "epsilon_condition"=>"epsilon -> 0 and rate*epsilon -> infinity",
            "signed_derivative_no_extra_half_factor"=>true,"gap_snapping_applied"=>false,
            "tail"=>"MC+1; observed lower envelope versus per-draw derivative upper envelope",
            "critical_value_positive_resolution_gate"=>true,
            "critical_value_continuity_established"=>false,
            "critical_limit_positive_continuous_quantile_assumed_not_verified"=>true,
            "epsilon_selection"=>"all prespecified grid results retained; no favorable selection",
            "floating_point_envelopes_not_interval_arithmetic"=>true,
            "sampling_bootstrap_consistency_verified"=>false,"uniform_validity_claimed"=>false,
            "empirical_clt_or_bias_verified"=>false,"workers"=>workers,
            "failed_or_replaced_draws"=>0,"mutable_projectors_shared"=>false,
            "projection_error_must_be_negligible_at_rate_squared_scale"=>true,
            "derivative_error_must_be_negligible_relative_to_epsilon_squared"=>true,
            "local_uniform_validity_not_inherited"=>true))
end
