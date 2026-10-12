#!/usr/bin/env julia
# New descriptive probability-normalization robustness; no revised inference.
using KSReplication, LinearAlgebra, JSON3, SHA, Statistics

const WNR_ROOT = @__DIR__
const WNR_PRIOR = joinpath(WNR_ROOT, "results/wilfried_windows_20261009_v2/summary.json")
const WNR_JOINT = joinpath(WNR_ROOT, "results/wilfried_joint_gap_20261010_v1")
const WNR_OUTPUT = joinpath(WNR_ROOT, "results/wilfried_normalized_fit_robustness_20261010_v1.json")
const WNR_PRIOR_SHA = "12141fd6314b9680ec7ef1aa19545c785aa4c42a28bcfb028bde6cbca986e4b1"
const WNR_JOINT_SHA = "9adae2a8bf5dd6e6b603f4079d21c9d171b57455a82ff02cfe4c745528b6dac3"

wnr_sha(path) = bytes2hex(sha256(read(path)))
wnr_columns(A) = Set(Tuple(view(A, :, h)) for h in axes(A, 2))
const WNR_Q = Rational{BigInt}

"""Exact full-Euclidean certificate from a numerical simplex proposal.

Every finite Float64 target and proposal weight is converted to its EXACT
binary rational value. Negative proposal roundoff is clipped and nonnegative
weights are normalized by their exact rational sum. All 0/1 columns are priced.
For residual r=target-A*nu, U=||r||^2 is a feasible upper bound and
L=max(0,2r'target-||r||^2-2max_h r'A_h) is a globally valid lower bound.
No QP optimality, floating contrast matrix, or solver error assertion enters
these bounds. They concern the frozen binary-rational target, not its unknown
population law or control-function estimation accuracy.
"""
function wnr_exact_certificate(A, target, proposal)
    size(A, 1) == length(target) && size(A, 2) == length(proposal) ||
        throw(DimensionMismatch("exact hull certificate dimensions"))
    all(a -> a == 0 || a == 1, A) || throw(ArgumentError("exact binary columns required"))
    all(isfinite, target) && all(isfinite, proposal) || throw(ArgumentError("finite exact proposals required"))
    pi = WNR_Q.(target)
    nu = WNR_Q.(max.(proposal, 0))
    mass = sum(nu)
    mass > 0 || throw(ArgumentError("positive proposed simplex mass required"))
    nu ./= mass
    sum(nu) == 1 && all(>=(0), nu) || error("exact simplex repair failed")
    column_rows = [findall(!iszero, view(A, :, h)) for h in axes(A, 2)]
    fitted = zeros(WNR_Q, length(target))
    for h in findall(>(0), nu), i in column_rows[h]
        fitted[i] += nu[h]
    end
    residual = pi-fitted
    upper = sum(abs2, residual)
    support = maximum(sum(residual[i] for i in rows; init=WNR_Q(0)) for rows in column_rows)
    lower = max(WNR_Q(0), 2dot(residual, pi)-upper-2support)
    lower <= upper || error("exact primal/support-dual ordering failed")
    return (; lower, upper, support, fitted, mass_sum=sum(nu),
        active_proposal_columns=count(>(0),nu), full_priced_columns=size(A,2))
end

function wnr_exact_summary(certificate)
    Dict{String,Any}("lower_bound_exact" => string(certificate.lower),
        "upper_bound_exact" => string(certificate.upper),
        "width_exact" => string(certificate.upper-certificate.lower),
        "lower_bound_float_diagnostic" => Float64(certificate.lower),
        "upper_bound_float_diagnostic" => Float64(certificate.upper),
        "width_float_diagnostic" => Float64(certificate.upper-certificate.lower),
        "support_maximum_exact" => string(certificate.support),
        "mass_sum_exact" => string(certificate.mass_sum),
        "active_proposal_columns" => certificate.active_proposal_columns,
        "full_priced_columns" => certificate.full_priced_columns,
        "QP_optimality_or_floating_contrast_certificate_assumed" => false,
        "fixed_Float64_target_interpreted_as_exact_binary_rationals" => true,
        "population_probability_estimation_error_certified" => false)
end

"""Repair an arbitrary finite solver mass proposal, WITHOUT optimality credit.

This is used only when the preexisting numerical projection acceptance gate
throws. The unchanged exact rational full-column certificate is still computed
afterwards. A poor proposal widens bounds; it cannot authorize a false claim.
"""
function wnr_uncertified_proposal(projector, z, rawmass, solver_status, failure)
    length(rawmass) == size(projector.A,2) && all(isfinite,rawmass) ||
        error("unavailable or nonfinite numerical solver proposal")
    mass = max.(Float64.(rawmass),0.0)
    sum(mass)>0 || error("failed numerical solver supplied no positive proposal mass")
    mass ./= sum(mass)
    fitted_probability = projector.A*mass
    fitted = projector.B*fitted_probability
    residual = z-fitted
    upper = sum(abs2,residual)
    support = maximum(transpose(projector.contrast_types)*residual)
    lower = max(0.0,2dot(residual,z)-upper-2support)
    lower <= upper+1e-12 || error("uncertified proposal has inconsistent diagnostic bounds")
    budget_error = maximum(abs(sum(fitted_probability[g])-1) for g in projector.blocks)
    (; fitted_probability, squared_distance=upper, mass, mass_sum=sum(mass),
        raw_mass_sum=sum(rawmass), mass_correction_l1=sum(abs,mass-rawmass),
        raw_mass_error=abs(sum(rawmass)-1), raw_nonnegative_violation=max(0.0,-minimum(rawmass)),
        budget_adding_up_error=budget_error,
        probability_nonnegative_violation=max(0.0,-minimum(fitted_probability)),
        contrast_consistency_error=maximum(abs,projector.B*fitted_probability-fitted),
        stationarity_error=nothing, support_variational_gap=upper-lower,
        dual_constraint_violation=nothing, complementarity_error=nothing,
        primal_dual_gap=(upper-lower)/2, certificate_tolerance=projector.tolerance,
        objective_lower_bound=lower, objective_upper_bound=upper,
        objective_error_bound=upper-lower, certified=false,
        solver_status=String(solver_status), numerical_acceptance_failure=String(failure))
end

function wnr_propose!(projector,z)
    try
        return project_information!(projector,z)
    catch err
        # Recover only finite proposals from the SAME full model; no retry,
        # target change, type omission, draw replacement, or zero filling.
        rawmass = -KSReplication.dual.(projector.type_constraints)
        return wnr_uncertified_proposal(projector,z,rawmass,
            string(KSReplication.termination_status(projector.model)),sprint(showerror,err))
    end
end

"""The budget-normal component common to all normalized type hulls.

With orthonormal Helmert contrasts, full-coordinate Euclidean distance equals
contrast distance plus this constant. Normalization error is reported, not
silently discarded or corrected in the original estimator.
"""
function wnr_normal_component(target, blocks)
    all(isfinite, target) || throw(ArgumentError("finite target required"))
    return sum((sum(target[g])-1)^2/length(g) for g in blocks)
end

function wnr_fit_summary(fit, target, blocks)
    normal = wnr_normal_component(target, blocks)
    full_distance = sum(abs2, target-fit.fitted_probability)
    abs(full_distance-(fit.squared_distance+normal)) <= 2e-10*max(1, full_distance) ||
        error("contrast/full Euclidean identity failed")
    return Dict{String,Any}(
        "contrast_squared_distance" => fit.squared_distance,
        "common_normal_squared_distance" => normal,
        "full_squared_distance" => full_distance,
        "full_objective_lower_bound" => fit.objective_lower_bound+normal,
        "full_objective_upper_bound" => fit.objective_upper_bound+normal,
        "objective_error_bound" => fit.objective_error_bound,
        "fitted_probability" => fit.fitted_probability,
        "mass_sum" => fit.mass_sum, "raw_mass_sum" => fit.raw_mass_sum,
        "mass_correction_l1" => fit.mass_correction_l1,
        "raw_mass_error" => fit.raw_mass_error,
        "raw_nonnegative_violation" => fit.raw_nonnegative_violation,
        "budget_adding_up_error" => fit.budget_adding_up_error,
        "probability_nonnegative_violation" => fit.probability_nonnegative_violation,
        "contrast_consistency_error" => fit.contrast_consistency_error,
        "stationarity_error" => fit.stationarity_error,
        "support_variational_gap" => fit.support_variational_gap,
        "dual_constraint_violation" => fit.dual_constraint_violation,
        "complementarity_error" => fit.complementarity_error,
        "primal_dual_gap" => fit.primal_dual_gap,
        "certificate_tolerance" => fit.certificate_tolerance,
        "certified" => fit.certified, "solver_status" => fit.solver_status,
        "numerical_acceptance_failure" => hasproperty(fit,:numerical_acceptance_failure) ?
            fit.numerical_acceptance_failure : nothing,
        "uncertified_solver_proposal_fallback_used" => !fit.certified,
        "certificate_is_exact_interval_arithmetic" => false)
end

"""Fixed identity-metric normalized RUM/WARP full-hull comparison.

No sample rows, types, covariance directions, or probability coordinates are
selected. The two targets are frozen raw affine and the original source clamp.
It is descriptive fit robustness, not a test or a population-share estimate.
"""
function wnr_compare(X, Ar, Aw, raw, clipped; tolerance=1e-7)
    I = size(X, 1)
    length(raw) == length(clipped) == I == size(Ar, 1) == size(Aw, 1) ||
        throw(DimensionMismatch("normalized comparison dimensions"))
    all(isfinite, raw) && all(isfinite, clipped) || throw(ArgumentError("finite probabilities required"))
    maximum(abs.(clamp.(raw, 0, 1).-clipped)) <= 1e-12 ||
        error("source-clipped policy changed")
    blocks = KSReplication._patch_groups(X)
    maximum(abs(sum(raw[g])-1) for g in blocks) <= 1e-10 ||
        error("raw affine source probabilities do not add up")
    rcolumns, wcolumns = wnr_columns(Ar), wnr_columns(Aw)
    issubset(rcolumns, wcolumns) || error("complete hulls are not nested")
    identical = rcolumns == wcolumns
    B = adding_up_contrasts(X)
    covariance = Matrix{Float64}(LinearAlgebra.I, size(B, 1), size(B, 1))
    rp = InformationProjector(Ar, B; covariance, blocks, tolerance)
    wp = identical ? nothing : InformationProjector(Aw, B; covariance, blocks, tolerance)
    # Uniform proposal-engine policy across ALL windows and BOTH targets.
    # A failed first attempt at the API's 1e-10 tolerance hit the iteration
    # limit on window16. Exact full-rational bounds below remain decisive,
    # independently of solver tolerances and optimality assumptions.
    for projector in (rp, wp)
        projector === nothing && continue
        KSReplication.set_optimizer_attribute(projector.model, "eps_abs", 1e-8)
        KSReplication.set_optimizer_attribute(projector.model, "eps_rel", 1e-8)
    end
    policies = Dict{String,Any}()
    for (policy, target) in (("raw_affine", Float64.(raw)), ("source_clipped", Float64.(clipped)))
        rfit = wnr_propose!(rp, B*target)
        wfit = identical ? rfit : wnr_propose!(wp, B*target)
        r, w = wnr_fit_summary(rfit, target, blocks), wnr_fit_summary(wfit, target, blocks)
        rexact = wnr_exact_certificate(Ar, target, rfit.mass)
        wexact = identical ? rexact : wnr_exact_certificate(Aw, target, wfit.mass)
        r["exact_full_projection_certificate"] = wnr_exact_summary(rexact)
        w["exact_full_projection_certificate"] = wnr_exact_summary(wexact)
        exactlower = identical ? WNR_Q(0) : max(WNR_Q(0), rexact.lower-wexact.upper)
        exactupper = identical ? WNR_Q(0) : max(WNR_Q(0), rexact.upper-wexact.lower)
        rawgap = r["full_squared_distance"]-w["full_squared_distance"]
        lower = identical ? 0.0 : max(0.0, r["full_objective_lower_bound"]-w["full_objective_upper_bound"])
        upper = identical ? 0.0 : max(0.0, r["full_objective_upper_bound"]-w["full_objective_lower_bound"])
        errorbound = r["objective_error_bound"]+w["objective_error_bound"]
        rawgap >= -errorbound-1e-12 || error("nested projection violates certified distance ordering")
        gap = identical ? 0.0 : max(0.0, rawgap)
        lower-1e-12 <= gap <= upper+1e-12 || error("gap outside numerical certificate")
        policies[policy] = Dict{String,Any}(
            "target_budget_sums" => [sum(target[g]) for g in blocks],
            "maximum_target_adding_up_error" => maximum(abs(sum(target[g])-1) for g in blocks),
            "minimum_target_probability" => minimum(target),
            "maximum_target_probability" => maximum(target),
            "negative_target_coordinate_count" => count(<(0), target),
            "above_one_target_coordinate_count" => count(>(1), target),
            "models" => Dict("RUM" => r, "WARP" => w),
            "distance_gain" => gap, "signed_distance_gain" => rawgap,
            "distance_gain_lower_numerical" => lower,
            "distance_gain_upper_numerical" => upper,
            "distance_gain_lower_exact" => string(exactlower),
            "distance_gain_upper_exact" => string(exactupper),
            "gain_strictly_positive_exact_fixed_target_certificate" => exactlower > 0,
            "gain_exceeds_numerical_error" => exactlower > 0,
            "relative_distance_reduction" => r["full_squared_distance"] > errorbound ?
                gap/r["full_squared_distance"] : nothing)
    end
    return Dict{String,Any}("policies" => policies, "identical_complete_column_sets" => identical,
        "complete_types" => Dict("RUM" => size(Ar, 2), "WARP" => size(Aw, 2)),
        "budget_coordinate_counts" => length.(blocks), "probability_coordinates" => I,
        "contrast_coordinates" => size(B, 1), "projection_count" => identical ? 2 : 4)
end

function wnr_protected_hashes()
    paths = [WNR_PRIOR, joinpath(WNR_JOINT, "summary.json"),
        joinpath(WNR_JOINT, "frozen_protocol_and_calendar_digests.json"),
        joinpath(WNR_ROOT, "Project.toml"), joinpath(WNR_ROOT, "Manifest.toml"),
        normpath(joinpath(WNR_ROOT, "../../TheLawsOfDemand.tex")),
        normpath(joinpath(WNR_ROOT, "../../TheLawsOfDemand.bib"))]
    append!(paths, [joinpath(WNR_ROOT, "src", n) for n in readdir(joinpath(WNR_ROOT, "src")) if endswith(n, ".jl")])
    for cell in 1:19
        suffix = lpad(cell, 2, '0')*".json"
        push!(paths, joinpath(WNR_ROOT, "results/wilfried_windows_20261009_v2", "cell_"*suffix))
        push!(paths, joinpath(WNR_JOINT, "cell_"*suffix))
        push!(paths, joinpath(WNR_JOINT, "raw_bootstrap_cell_"*suffix))
    end
    for n in ("run_wilfried_joint_gap.jl", "wilfried_joint_calendar.jl", "run_wilfried_windows.jl",
            "run_wilfried_chen_fang.jl", "wilfried_nnls_projection.jl", "wilfried_working_set_projection.jl",
            "run_wilfried_replication.jl", "run_wilfried_rank_control_audit.jl", "run_wilfried_rank_local_audit.jl")
        push!(paths, joinpath(WNR_ROOT, n))
    end
    for n in ("wilfried_rank_control_audit_20261010_v1.json", "wilfried_rank_local_audit_20261010_v1.json")
        push!(paths, joinpath(WNR_ROOT, "results", n))
    end
    Dict(relpath(p, WNR_ROOT) => wnr_sha(p) for p in paths)
end

function wnr_aggregate(windows, policy, unique_n)
    r, w, gain, lower, upper = 0.0, 0.0, 0.0, 0.0, 0.0
    exactlower, exactupper = WNR_Q(0), WNR_Q(0)
    for window in windows
        d = window["comparison"]["policies"][policy]
        weight = window["annual_n_sum"]/unique_n
        r += weight*d["models"]["RUM"]["full_squared_distance"]
        w += weight*d["models"]["WARP"]["full_squared_distance"]
        gain += weight*d["distance_gain"]
        lower += weight*d["distance_gain_lower_numerical"]
        upper += weight*d["distance_gain_upper_numerical"]
        exactweight = WNR_Q(window["annual_n_sum"],unique_n)
        exactlower += exactweight*parse(WNR_Q,d["distance_gain_lower_exact"])
        exactupper += exactweight*parse(WNR_Q,d["distance_gain_upper_exact"])
    end
    Dict{String,Any}("weighted_RUM_squared_distance" => r,
        "weighted_WARP_squared_distance" => w, "weighted_distance_gain" => gain,
        "weighted_gain_lower_numerical" => lower, "weighted_gain_upper_numerical" => upper,
        "relative_weighted_distance_reduction" => r > 0 ? gain/r : nothing,
        "source_scaled_descriptive_gain" => (unique_n/3)*gain,
        "source_scaled_descriptive_gain_lower_numerical" => (unique_n/3)*lower,
        "source_scaled_descriptive_gain_upper_numerical" => (unique_n/3)*upper,
        "weighted_gain_lower_exact" => string(exactlower),
        "weighted_gain_upper_exact" => string(exactupper),
        "source_scaled_descriptive_gain_lower_exact" => string((WNR_Q(unique_n,3))*exactlower),
        "source_scaled_descriptive_gain_upper_exact" => string((WNR_Q(unique_n,3))*exactupper),
        "gain_strictly_positive_exact_fixed_target_certificate" => exactlower > 0,
        "gain_exceeds_numerical_error" => exactlower > 0,
        "positive_resolved_windows" => [v["cell"] for v in windows if
            v["comparison"]["policies"][policy]["gain_exceeds_numerical_error"]],
        "source_scaled_descriptive_gain_is_NOT_a_calibrated_test_statistic" => true)
end

function wnr_main(args)
    args == ["--write-approved-report"] || error("Final artifact requires the explicit approved run flag; no profile overrides")
    ispath(WNR_OUTPUT) && error("refusing existing normalized robustness output")
    wnr_sha(WNR_PRIOR) == WNR_PRIOR_SHA || error("frozen geometry summary changed")
    wnr_sha(joinpath(WNR_JOINT, "summary.json")) == WNR_JOINT_SHA || error("frozen joint comparison changed")
    before = wnr_protected_hashes()
    prior = JSON3.read(read(WNR_PRIOR, String))
    joint = JSON3.read(read(joinpath(WNR_JOINT, "summary.json"), String))
    prior["status"] == "completed" && joint["source_integrity_unchanged"] || error("frozen inputs incomplete")
    cases = sort(collect(prior["cases"]); by = c -> Int(c["cell"]))
    Int.([c["cell"] for c in cases]) == collect(1:19) || error("all 19 windows required")
    unique_n = Int(joint["protocol"]["N_unique"])
    unique_n == 46391 || error("frozen annual sample size differs")
    BLAS.set_num_threads(1)
    windows = Dict{String,Any}[]
    start = time()
    for old in cases
        cell = Int(old["cell"])
        rawpath = joinpath(WNR_JOINT, "raw_bootstrap_cell_$(lpad(cell, 2, '0')).json")
        rawcheckpoint = JSON3.read(read(rawpath, String))
        rawcheckpoint["cell"] == cell && rawcheckpoint["years"] == old["years"] || error("raw point alignment changed")
        X = reduce(vcat, permutedims(Int8.(row)) for row in old["sign_matrix"])
        raw = Float64.(rawcheckpoint["raw_affine_point"])
        clipped = Float64.(old["pi_hat_regenerated"])
        Ar = enumerate_types_fast(X; restriction=:rum, max_types=2_000_000, max_nodes=50_000_000)
        Aw = enumerate_types_fast(X; restriction=:warp, max_types=2_000_000, max_nodes=50_000_000)
        size(Ar, 2) == old["models"]["RUM"]["types"] &&
            size(Aw, 2) == old["models"]["correct_WARP"]["types"] || error("frozen complete type counts differ")
        timed = @timed wnr_compare(X, Ar, Aw, raw, clipped)
        comparison = timed.value
        push!(windows, Dict{String,Any}("cell" => cell, "years" => Int.(old["years"]),
            "annual_n" => Int.(old["annual_n"]), "annual_n_sum" => sum(Int.(old["annual_n"])),
            "comparison" => comparison, "point_projection_seconds" => timed.time,
            "source_cone_RUM_mass_sum" => old["models"]["RUM"]["projection_mass_sum"],
            "source_cone_WARP_mass_sum" => old["models"]["correct_WARP"]["projection_mass_sum"],
            "source_cone_scaled_gain" => old["models"]["RUM"]["J_unrounded"]-
                old["models"]["correct_WARP"]["J_unrounded"]))
        println("Normalized robustness cell $cell; types=$(size(Ar,2))/$(size(Aw,2)); raw gain=",
            comparison["policies"]["raw_affine"]["distance_gain"])
        flush(stdout)
    end
    after = wnr_protected_hashes()
    before == after || error("protected source/report/manuscript changed; no artifact emitted")
    report = Dict{String,Any}("status" => "completed_descriptive_normalized_probability_fit_robustness",
        "protocol" => Dict("windows" => collect(1:19), "primary_probability_policy" => "raw_affine",
            "sensitivity_probability_policy" => "source_clipped", "metric" => "identity full-coordinate Euclidean normalized type hull",
            "contrast_covariance" => "identity; no estimated covariance, ridge, eigenvalue selection or tuning",
            "complete_column_sets_regenerated_and_count_verified" => true,
            "mass_constraint" => "nonnegative type masses with sum exactly one in model",
            "projection_certificate" => "exact binary-rational target, exactly renormalized proposed simplex mass, full 0/1-column residual support dual; independent of solver optimality",
            "uniform_OSQP_proposal_settings" => Dict("eps_abs" => 1e-8, "eps_rel" => 1e-8,
                "max_iter" => 200000, "polish" => true,
                "uniform_across_all19_windows_and_both_policies" => true,
                "exact_rational_certificate_depends_on_proposal_solver_optimality" => false),
            "numerical_gate_failure_policy" => "Recover finite masses from the SAME complete solver model; disclose numerical optimality uncertified; clip and exactly renormalize the rational proposal; price EVERY column for exact global bounds. No substituted target, retry, missing-cell zero filling, or type truncation.",
            "failed_first_attempt" => Dict("proposal_eps_abs" => 1e-10, "proposal_eps_rel" => 1e-10,
                "failed_window" => 16, "solver_status" => "ITERATION_LIMIT",
                "final_result_artifact_written" => false,
                "metric_targets_types_or_statistical_profile_changed" => false),
            "all_probability_rows_retained" => true, "original_targets_renormalized" => false,
            "common_normal_component_retained_in_distance_levels" => true,
            "joint_window_weights" => "N_window/N_unique; fixed full19 family, not selected positive windows",
            "N_unique" => unique_n, "bootstrap_recalibrated" => false, "new_p_value_produced" => false,
            "population_nontransitive_share_or_confidence_bound_produced" => false,
            "minimum_purchase_or_price_extrapolation_or_household_filters_changed" => false),
        "aggregate" => Dict(policy => wnr_aggregate(windows, policy, unique_n) for policy in ("raw_affine", "source_clipped")),
        "windows" => windows, "projection_count" => sum(v["comparison"]["projection_count"] for v in windows),
        "elapsed_seconds" => time()-start, "protected_sha256_before" => before, "protected_sha256_after" => after,
        "source_integrity_unchanged" => true,
        "implementation_sha256" => Dict("run_wilfried_normalized_fit_robustness.jl" => wnr_sha(@__FILE__),
            "test/test_wilfried_normalized_fit_robustness.jl" => wnr_sha(joinpath(WNR_ROOT, "test/test_wilfried_normalized_fit_robustness.jl"))),
        "interpretation" => "New normalized full-polytope point-fit robustness on frozen UK/FES probabilities, not a new hypothesis test. Exact rational primal/support-dual certificates concern optimization of the fixed binary-rational Float64 target, not population estimation error or statistical coverage. Solver proposals need not be optimal for these bounds to be valid. Positive relative-fit gains do not establish WARP adequacy, a WARP-pass/RUM-fail contrast, population nontransitive shares, or fixed-budget finite-sample theorem applicability to generated control-function probabilities. Both policies and all19 windows remain visible; no favorable policy or cell is selected.",
        "raw_household_records_identifiers_or_bootstrap_frequency_vectors_exported" => false)
    ispath(WNR_OUTPUT) && error("refusing an output created concurrently")
    open(WNR_OUTPUT, "w") do io
        JSON3.pretty(io, report)
    end
    println("Saved $WNR_OUTPUT")
    return report
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    wnr_main(ARGS)
end
