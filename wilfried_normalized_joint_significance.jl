# New inference helpers; all earlier estimators and result artifacts are read-only.
include(joinpath(@__DIR__, "wilfried_joint_calendar.jl"))
include(joinpath(@__DIR__, "run_wilfried_normalized_fit_robustness.jl"))
include(joinpath(@__DIR__, "wilfried_normalized_working_set.jl"))

const WNJ_MULTIPLIERS = [0.5, 1.0, 2.0]
const WNJ_PRIMARY_MULTIPLIER = 1.0
const WNJ_NORMALIZED_SHA = "05c97e6b5a991e37878bcc50ee8bd6b5da3f0c87f2fb440d53180ba381e2a61e"

"""Restore and verify the saved paired, column-major raw-affine bootstrap."""
function wnj_saved_matrix(record; expected_B=1000)
    coordinates, repetitions = Int(record["coordinates"]), Int(record["B"])
    coordinates > 0 && repetitions == expected_B || error("saved bootstrap dimensions changed")
    flat = Float64.(collect(record["raw_affine_bootstrap"]))
    length(flat) == coordinates*repetitions || error("saved bootstrap is incomplete")
    all(isfinite, flat) || error("nonfinite saved bootstrap")
    stars = reshape(flat, coordinates, repetitions)
    bytes2hex(sha256(JSON3.write(stars))) == String(record["raw_affine_bootstrap_sha256"]) ||
        error("saved bootstrap matrix/draw ordering changed")
    point = Float64.(record["raw_affine_point"])
    length(point) == coordinates && all(isfinite, point) || error("saved point mismatch")
    record["refit_diagnostics"]["both_CF_stages_refit"] === true || error("both CF stages must be re-fit")
    Int(record["refit_diagnostics"]["failed_or_replaced_draws"]) == 0 || error("replaced or failed saved draw")
    (; point, stars)
end

function wnj_gap!(pair, target)
    r, w = nh_fit!(pair[1], target), nh_fit!(pair[2], target)
    raw = r.upper-w.upper
    lo, hi = max(0.0, r.lower-w.upper), max(0.0, r.upper-w.lower)
    raw >= -(r.upper-r.lower)-(w.upper-w.lower)-1e-11 || error("nested hull ordering failed")
    lo <= hi || error("reversed hull gap bounds")
    (; raw=max(0.0, raw), lower=lo, upper=hi, rum=r, warp=w)
end

"""Actual complete normalized hulls, with raw affine perturbations, never clamp.

Only complete column-set equality permits the analytic-zero shortcut. A zero
observed gap in distinct models never removes its bootstrap contribution.
Each worker owns its own mutable pair, and every solve prices all columns.
"""
function wnj_window_envelopes(cell, point, stars, Ar, Aw, rate, epsilons;
        workers=6, tolerance=1e-8)
    B = size(stars, 2)
    B>=2 && size(Ar,2)>0 && size(Aw,2)>0 && !isempty(epsilons) || error("nonempty models and at least two draws required")
    size(stars, 1) == length(point) == size(Ar, 1) == size(Aw, 1) || error("coordinate mismatch")
    all(isfinite, point) && all(isfinite, stars) || error("nonfinite estimator")
    for A in (Ar,Aw)
        all(x->x==0||x==1,A) || error("binary complete types required")
        sums=vec(sum(A;dims=1))
        first(sums)>0 && length(unique(sums))==1 || error("positive equal type column sums required")
    end
    rate > 0 && isfinite(rate) && all(e->isfinite(e)&&e>0, epsilons) || error("invalid derivative rates")
    1 <= workers <= Threads.nthreads() || error("invalid worker count")
    rset, wset = wnr_columns(Ar), wnr_columns(Aw)
    issubset(rset, wset) || error("complete normalized hulls must be nested")
    if rset == wset
        return Dict("cell"=>cell, "analytic_identical_models"=>true,
            "point_raw"=>0.0, "point_lower"=>0.0, "point_upper"=>0.0,
            "derivative_central"=>zeros(B,length(epsilons)),
            "derivative_lower"=>zeros(B,length(epsilons)),
            "derivative_upper"=>zeros(B,length(epsilons)),
            "point_projection"=>nothing, "point_exact_certificates"=>nothing,
            "projection_audit"=>Dict("fits"=>0,"all_columns_priced"=>true))
    end
    make_pair() = (NormalizedHullProjector(Ar;certificate_tolerance=tolerance),
        NormalizedHullProjector(Aw;certificate_tolerance=tolerance))
    basepair, pairs = make_pair(), [make_pair() for _ in 1:workers]
    all(pairs[i][j].model !== basepair[j].model for i in 1:workers for j in 1:2) || error("shared base solver")
    all(pairs[i][j].model !== pairs[k][j].model for i in 1:workers for k in 1:workers if i!=k for j in 1:2) ||
        error("mutable solvers shared across workers")
    base = wnj_gap!(basepair, point)
    rexact = wnr_exact_certificate(Ar,point,base.rum.mass)
    wexact = wnr_exact_certificate(Aw,point,base.warp.mass)
    # Base objectives are exact-rational certified independently of QP optimality.
    blo, bhi = max(WNR_Q(0),rexact.lower-wexact.upper), max(WNR_Q(0),rexact.upper-wexact.lower)
    central, lower, upper = zeros(B,length(epsilons)), zeros(B,length(epsilons)), zeros(B,length(epsilons))
    fit_counts, maximum_width, maximum_cuts = zeros(Int,workers), zeros(workers), zeros(Int,workers)
    function run_worker(worker)
        pair = pairs[worker]
        for (k,e) in enumerate(epsilons), b in worker:workers:B
            try
                target = point .+ (e*rate).*(view(stars,:,b).-point)
                fit = wnj_gap!(pair,target)
                central[b,k] = (fit.raw-base.raw)/e^2
                lower[b,k] = (fit.lower-Float64(bhi))/e^2
                upper[b,k] = (fit.upper-Float64(blo))/e^2
                all(isfinite,(central[b,k],lower[b,k],upper[b,k])) && lower[b,k]<=upper[b,k] ||
                    error("invalid derivative envelope")
                lower[b,k]-1e-9 <= central[b,k] <= upper[b,k]+1e-9 || error("central derivative outside envelope")
                fit_counts[worker] += 2
                maximum_width[worker] = max(maximum_width[worker],fit.rum.upper-fit.rum.lower,fit.warp.upper-fit.warp.lower)
                maximum_cuts[worker] = max(maximum_cuts[worker],fit.rum.diagnostic.working_cuts,fit.warp.diagnostic.working_cuts)
            catch err
                error("normalized window=$cell epsilon=$e draw=$b failed, no replacement: $(sprint(showerror,err))")
            end
        end
    end
    if workers == 1
        run_worker(1)
    else
        @sync for worker in 1:workers
            Threads.@spawn run_worker(worker)
        end
    end
    Dict("cell"=>cell,"analytic_identical_models"=>false,
        "point_raw"=>base.raw,"point_lower"=>Float64(blo),"point_upper"=>Float64(bhi),
        "derivative_central"=>central,"derivative_lower"=>lower,"derivative_upper"=>upper,
        "point_projection"=>Dict("rum_squared_distance"=>base.rum.upper,"warp_squared_distance"=>base.warp.upper,
            "rum_mass_sum"=>base.rum.mass_sum,"warp_mass_sum"=>base.warp.mass_sum),
        "point_exact_certificates"=>Dict("rum"=>wnr_exact_summary(rexact),"warp"=>wnr_exact_summary(wexact)),
        "projection_audit"=>Dict("fits"=>2+sum(fit_counts),"all_columns_priced"=>true,
            "maximum_bootstrap_objective_envelope_width"=>maximum(maximum_width),
            "maximum_working_cuts"=>maximum(maximum_cuts),
            "point_bounds_exact_rational"=>true,"bootstrap_bounds_exact_rational"=>false))
end

function wnj_aggregate(envelopes, weights, rate, epsilons; alpha=.05, require_full_family=true)
    require_full_family && Set(Int(e["cell"]) for e in envelopes)!=Set(1:19) && error("all19 windows required")
    isfinite(rate) && rate>0 && 0<alpha<1 || error("invalid aggregate contract")
    length(epsilons)==3 && maximum(abs.(epsilons.*sqrt(rate).-WNJ_MULTIPLIERS))<=1e-12 ||
        error("frozen epsilon multipliers changed")
    result = wjc_aggregate(envelopes, weights, rate, epsilons;alpha)
    result["method"] = "joint_calendar_year_chen_fang_raw_affine_normalized_hull_gain"
    result["normalized_type_mass_sum_one"] = true
    result["raw_affine_targets_clipped_or_renormalized"] = false
    result["alpha"] = alpha
    result["primary_result_index_one_based"] = 2
    result["primary_epsilon_multiplier"] = WNJ_PRIMARY_MULTIPLIER
    result["tuning_selected_by_p_value"] = false
    for (k,r) in enumerate(result["epsilon_results"])
        r["role"] = k==2 ? "frozen_primary" : "sensitivity_only_not_selected_for_significance"
    end
    result
end

"""C-free conservative companion: under zero population gain, Phi(error)<=||error||_w².

This is not an ordinary centered gap bootstrap. It bootstraps the regular
stacked raw-affine estimator error itself. Its validity is still conditional
on the maintained joint estimator bootstrap, not a finite-sample UK theorem.
"""
function wnj_null_error_envelope(records, weights, rate, observed_lower;alpha=.05, require_full_family=true)
    !isempty(records) && isfinite(rate)&&rate>0 && isfinite(alpha)&&0<alpha<1 || error("invalid norm calibration contract")
    isfinite(observed_lower)&&observed_lower>=0 || error("finite nonnegative observed bound required")
    require_full_family && Set(keys(records))!=Set(1:19) && error("all19 windows required")
    Set(keys(records))==Set(keys(weights)) || error("error envelope family mismatch")
    B = size(first(values(records)).stars,2)
    B>=2 || error("at least two joint draws required")
    norm_draws = zeros(B)
    for cell in sort(collect(keys(records)))
        record, weight = records[cell], weights[cell]
        isfinite(weight)&&weight>0 && size(record.stars,2)==B || error("invalid joint norm input")
        size(record.stars,1)==length(record.point) && !isempty(record.point) || error("norm coordinate mismatch")
        all(isfinite,record.point)&&all(isfinite,record.stars) || error("nonfinite joint norm input")
        for b in 1:B
            norm_draws[b] += weight*sum(abs2,rate.*(view(record.stars,:,b).-record.point))
        end
    end
    order = ceil(Int,(1-alpha)*(B+1))
    critical = order<=B ? sort(norm_draws)[order] : Inf
    tail = count(>=(observed_lower),norm_draws)
    Dict("method"=>"c_free_global_null_gain_error_norm_envelope",
        "alpha"=>alpha,"observed_statistic_lower"=>observed_lower,
        "critical_value"=>isfinite(critical) ? critical : nothing,
        "exceedances"=>tail,"B"=>B,"p_value_mc"=>(tail+1)/(B+1),
        "reject_conditional"=>isfinite(critical)&&critical>0&&observed_lower>critical,
        "norm_draws"=>norm_draws,"covariance_inverse_or_eigenvalue_selection_used"=>false,
        "epsilon_tuning_used"=>false,"all_probability_coordinates_retained"=>true,
        "joint_estimator_bootstrap_validity_verified"=>false)
end
