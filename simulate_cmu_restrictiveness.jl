# New, descriptive geometric benchmark. Existing empirical inputs/results are
# read-only. This is NOT a power experiment, p-value or full-CMU characterization.
using Random, Distributions, LinearAlgebra, Statistics, JSON3, SHA
include(joinpath(@__DIR__, "cmu_superiority_floor_inference.jl"))

const CRB_VERSION = "cmu-restrictiveness-ray-benchmark-v1"
const CRB_Q = Rational{BigInt}
const CRB_M = 100
const CRB_PRIORS = (1., .25)
const CRB_SEED = 202610125
const CRB_WORKERS = 6
const CRB_TOLERANCE = 1e-8
const CRB_SPARSE_BITS = 40
const CRB_CAPS = copy(CEIP_CAPS)

crb_sha(path) = bytes2hex(sha256(read(path)))
crb_read(path) = JSON3.read(read(path, String))
function crb_new(path, object)
    ispath(path) && error("refusing overwrite: $path")
    open(path, "w") do io
        JSON3.pretty(io, object)
    end
end

"""Fixed largest-remainder normalization of a proposed budget distribution.

Binary-rational proposed coordinates are first normalized EXACTLY, then rounded
to counts with denominator 2^bits. Ties are resolved by original row order.
No targets are replaced when the sparse prior rounds a coordinate to zero.
"""
function crb_largest_remainder(values; bits=CRB_SPARSE_BITS)
    8 <= bits <= 52 || throw(ArgumentError("dyadic precision must lie between 8 and 52 bits"))
    all(x -> isfinite(x) && x >= 0, values) || throw(ArgumentError("nonnegative finite proposal required"))
    source = CRB_Q.(values)
    total = sum(source)
    total > 0 || throw(ArgumentError("positive proposal total required"))
    normalized = source ./ total
    denominator = big(1) << bits
    scaled = normalized .* denominator
    counts = floor.(BigInt, scaled)
    remainder = Int(denominator - sum(counts))
    0 <= remainder < length(counts) || error("largest-remainder count invariant failed")
    fractions = scaled .- counts
    order = sortperm(collect(eachindex(counts)); by=i -> (-fractions[i], i))
    for i in order[1:remainder]
        counts[i] += 1
    end
    exact = counts .// denominator
    point = Float64.(exact)
    all(>=(0), exact) && sum(exact) == 1 && CRB_Q.(point) == exact ||
        error("dyadic normalization is not exactly valid")
    maximum_error = maximum(abs.(exact .- normalized))
    maximum_error <= inv(CRB_Q(denominator)) || error("largest-remainder error bound failed")
    (; point, exact, counts, denominator, maximum_error,
       source_normalization_change=maximum(abs.(normalized .- source)),
       rounded_zero_coordinates=count(iszero, exact))
end

"""Valid product-simplex reference probabilities; original patch groups retained.

Dirichlet(1) uses ordered uniform spacings, whose Float64 cutpoints and spacings
are dyadic and add to one exactly as binary rationals. Dirichlet(.25) is the
prespecified sparse sensitivity prior, normalized by the fixed 2^40 rule.
"""
function crb_probability(rng, groups, alpha)
    alpha in CRB_PRIORS || throw(ArgumentError("approved priors are alpha=1 and alpha=.25"))
    I = maximum(reduce(vcat, groups))
    sort(reduce(vcat, groups)) == collect(1:I) || error("budget groups must partition coordinates")
    point = zeros(Float64, I)
    details = Dict{String,Any}[]
    for (j, group) in enumerate(groups)
        k = length(group)
        k > 0 || error("empty budget group")
        if alpha == 1.
            cuts = vcat(0., sort(rand(rng, k - 1)), 1.)
            p = diff(cuts)
            exact = CRB_Q.(p)
            all(>=(0), exact) && sum(exact) == 1 || error("uniform spacing dyadic adding-up failed")
            point[group] = p
            push!(details, Dict("budget"=>j, "generator"=>"ordered_uniform_spacings_Dirichlet_1",
                "normalization_max_error_exact"=>"0//1", "rounded_zero_coordinates"=>count(iszero, exact)))
        else
            proposed = k == 1 ? [1.] : rand(rng, Dirichlet(k, alpha))
            normalized = crb_largest_remainder(proposed)
            point[group] = normalized.point
            push!(details, Dict("budget"=>j, "generator"=>"Dirichlet_0.25_then_fixed_largest_remainder",
                "denominator"=>string(normalized.denominator),
                "counts"=>string.(normalized.counts),
                "normalization_max_error_exact"=>string(normalized.maximum_error),
                "source_normalization_change_exact"=>string(normalized.source_normalization_change),
                "rounded_zero_coordinates"=>normalized.rounded_zero_coordinates))
        end
    end
    exact = CRB_Q.(point)
    all(g -> all(>=(0), exact[g]) && sum(exact[g]) == 1, groups) || error("reference point is outside S")
    (; point, exact, details)
end

function crb_support(family, y)
    denominator = foldl(lcm, Base.denominator.(y); init=big(1))
    numerators = [numerator(v) * div(denominator, Base.denominator(v)) for v in y]
    priced = csfi_integer_support(family, numerators)
    (; value=priced.maximum // denominator, arithmetic=priced.arithmetic,
       all_columns=family.n_complete, denominator, numerators)
end

"""Recover and check the feasible complete-type primal from an existing fit.

The complete Cartesian identity has an analytic product-distribution witness.
Otherwise the saved nonnegative complete masses sum exactly to one and every
original budget selects one patch. No arbitrary budget deletion is admitted.
"""
function crb_reconstruct_primal(family, fit, p)
    certificate = fit.certificate
    certificate === nothing && error("full primal certificate required")
    if get(certificate, "analytic_RUM_equals_product_simplex", false)
        csfi_cartesian_identity(family) || error("unverified Cartesian shortcut")
        q = parse.(CRB_Q, String.(certificate["simplex_target_exact"]))
        q == p || error("analytic complete-S fit changed valid reference point")
        return (; q, witness="verified_full_Cartesian_product_distribution")
    end
    indices = Int.(certificate["complete_column_indices"])
    denominator = parse(BigInt, String(certificate["population_mass_common_denominator"]))
    numerators = parse.(BigInt, String.(certificate["population_mass_numerators"]))
    length(indices) == length(numerators) && denominator > 0 || error("complete primal dimensions")
    all(i -> 1 <= i <= family.n_complete, indices) && all(>=(0), numerators) || error("invalid complete primal")
    sum(numerators) == denominator || error("complete masses do not sum to one")
    qnum = zeros(BigInt, family.n_patches)
    for (h, mass) in zip(indices, numerators), j in 1:family.n_budgets
        qnum[family.complete_rows[j, h]] += mass
    end
    q = qnum .// denominator
    all(g -> all(>=(0), q[g]) && sum(q[g]) == 1, family.budget_groups) || error("primal adding-up failed")
    sum(abs2, p - q) == fit.upper || error("reconstructed complete primal objective mismatch")
    (; q, witness="checked_saved_complete_column_mass_certificate")
end

"""Exact geometric ray bounds for the maintained actual-empty INNER family.

B_beta=(1-beta)W+beta*S; beta=min(1,delta/((J-1)*(1-delta))).
The primal blend is (1-beta)qW+beta*p. The dual ray is lambda=t*(p-qW),
t=max(0,p'v-sigma_B(v))/||v||^2. Interval width here is GEOMETRIC slack,
not a solver tolerance or an inferential ambiguity. WARP feasibility of qW is
supplied and checked by `crb_reconstruct_primal` in the calling pipeline.
"""
function crb_ray_bounds(p, qW, family, cap; warp_lower, warp_upper,
                        support_certificate=nothing)
    p = CRB_Q.(p); qW = CRB_Q.(qW)
    length(p) == length(qW) == family.n_patches || error("ray dimensions")
    all(g -> sum(p[g]) == 1 && all(>=(0), p[g]) && sum(qW[g]) == 1 && all(>=(0), qW[g]),
        family.budget_groups) || error("ray points must be valid product probabilities")
    0 <= cap < 1 || throw(ArgumentError("empty allowance must lie in [0,1)"))
    tau = 1 - CRB_Q(Float64(cap))
    beta = min(CRB_Q(1), (1 - tau) / ((family.n_budgets - 1) * tau))
    v = p - qW
    norm2 = sum(abs2, v)
    if support_certificate === nothing
        priced = crb_support(family, v)
    else
        support_certificate.v == v && support_certificate.priced.all_columns == family.n_complete ||
            error("cached support certificate has a different residual or family size")
        priced = support_certificate.priced
    end
    supportW = priced.value
    supportS = sum(maximum(v[g]) for g in family.budget_groups)
    supportW <= supportS || error("complete WARP support exceeds product-simplex support")
    dotpv = dot(p, v)
    supportB = (1 - beta) * supportW + beta * supportS
    advantage = dotpv - supportB
    positive_advantage = max(CRB_Q(0), advantage)
    ray_t = iszero(norm2) ? CRB_Q(0) : positive_advantage / norm2
    ray_lower = iszero(norm2) ? CRB_Q(0) : positive_advantage^2 / norm2
    blend_upper = (1 - beta)^2 * norm2
    if iszero(beta)
        lower, upper = CRB_Q(warp_lower), CRB_Q(warp_upper)
        ray_lower <= upper && lower <= norm2 || error("cap-zero ray is inconsistent with WARP certificate")
        method = "cap_zero_existing_certified_complete_WARP_fit"
    elseif beta == 1
        lower = upper = CRB_Q(0)
        method = "beta_one_exact_product_simplex_identity"
    else
        lower, upper = ray_lower, blend_upper
        method = "exact_feasible_blend_and_optimized_dual_ray"
    end
    0 <= lower <= upper || error("geometric inner loss bounds inverted")
    complete_mass = tau * (1 - beta)
    singleton_budget_mass = tau * beta
    null_mass = 1 - family.n_budgets * tau + (family.n_budgets - 1) * complete_mass
    null_mass >= 0 && complete_mass + family.n_budgets * singleton_budget_mass + null_mass == 1 ||
        error("common-response population blend invalid")
    complete_mass + singleton_budget_mass == tau || error("common response identity failed")
    (; lower, upper, tau, beta, v, norm2, dotpv, supportW, supportS, supportB,
       advantage, ray_t, ray_lower, blend_upper, complete_mass, singleton_budget_mass,
       null_mass, method, support_arithmetic=priced.arithmetic,
       all_WARP_columns_priced=priced.all_columns)
end

function crb_interval_record(lower, upper)
    0 <= lower <= upper || error("invalid nonnegative loss/gain interval")
    Dict("lower_exact"=>string(lower), "upper_exact"=>string(upper),
         "lower"=>ceip_float_lower(lower), "upper"=>ceip_float_upper(upper),
         "midpoint"=>Float64((lower + upper) / 2),
         "width"=>ceip_float_upper(upper - lower))
end

function crb_target_record(cell, prior_index, repetition, probability, R, W, sourceW; seed)
    p = probability.exact
    floor = ceip_simplex_floor(probability.point, sourceW.family.budget_groups)
    iszero(floor.floor) && floor.q == p || error("reference point has a nonzero simplex floor")
    rfit = csfi_check_fit(R); wfit = csfi_check_fit(W)
    primal = crb_reconstruct_primal(sourceW.family, wfit, p)
    R_lower, R_upper = rfit.lower, rfit.upper
    W_lower, W_upper = wfit.lower, wfit.upper
    W_lower <= R_upper || error("certified normalized models violate RUM subset WARP")
    v = p - primal.q
    support = crb_support(sourceW.family, v)
    support_certificate = (; v, priced=support)
    caps = Dict{String,Any}[]
    previous_upper = nothing
    for cap in CRB_CAPS
        b = crb_ray_bounds(p, primal.q, sourceW.family, cap;
                          warp_lower=W_lower, warp_upper=W_upper, support_certificate)
        # True losses and these blend upper bounds decrease with the cap. An
        # independently sharper lower certificate need not itself be monotone.
        previous_upper !== nothing && b.upper > previous_upper &&
            error("nested-cap geometric upper bounds are not monotone")
        previous_upper = b.upper
        gainR_lower = max(CRB_Q(0), R_lower - b.upper)
        gainR_upper = R_upper - b.lower
        gainW_lower = max(CRB_Q(0), W_lower - b.upper)
        gainW_upper = W_upper - b.lower
        push!(caps, Dict("cap"=>cap, "tau_exact"=>string(b.tau), "beta_exact"=>string(b.beta),
            "loss"=>crb_interval_record(b.lower, b.upper),
            "RUM_minus_inner_gain"=>crb_interval_record(gainR_lower, gainR_upper),
            "WARP_minus_inner_gain"=>crb_interval_record(gainW_lower, gainW_upper),
            "bound_method"=>b.method, "geometric_not_solver_width"=>0 < b.beta < 1,
            "width_kind"=>iszero(b.beta) ? "certified_WARP_solver_objective_interval" :
                          b.beta==1 ? "analytic_zero" : "geometric_ray_blend_slack",
            "support_B_exact"=>string(b.supportB), "ray_advantage_exact"=>string(b.advantage),
            "optimized_dual_ray_multiplier_exact"=>string(b.ray_t),
            "dual_ray_loss_lower_exact"=>string(b.ray_lower),
            "feasible_blend_loss_upper_exact"=>string(b.blend_upper),
            "complete_population_mass_exact"=>string(b.complete_mass),
            "singleton_population_mass_per_budget_exact"=>string(b.singleton_budget_mass),
            "all_null_population_mass_exact"=>string(b.null_mass)))
    end
    Dict("cell"=>cell, "prior_index"=>prior_index, "alpha"=>CRB_PRIORS[prior_index],
        "repetition"=>repetition, "seed"=>seed,
        "point"=>probability.point, "point_exact"=>string.(p),
        "probability_generation"=>probability.details, "simplex_floor_exact"=>"0//1",
        "RUM_loss"=>crb_interval_record(R_lower, R_upper),
        "WARP_loss"=>crb_interval_record(W_lower, W_upper),
        "Psi_RUM_minus_simplex_floor"=>crb_interval_record(R_lower, R_upper),
        "RUM_certificate"=>rfit.certificate, "WARP_certificate"=>wfit.certificate,
        "RUM_diagnostic"=>rfit.diagnostic, "WARP_diagnostic"=>wfit.diagnostic,
        "WARP_primal_witness"=>primal.witness, "qW_exact"=>string.(primal.q),
        "ray_residual_v_exact"=>string.(v), "ray_norm_squared_exact"=>string(sum(abs2, v)),
        "ray_p_dot_v_exact"=>string(dot(p, v)),
        "ray_WARP_support_exact"=>string(support.value),
        "ray_simplex_support_exact"=>string(sum(maximum(v[g]) for g in sourceW.family.budget_groups)),
        "all_WARP_columns_priced"=>support.all_columns,
        "support_arithmetic"=>support.arithmetic, "caps"=>caps)
end

crb_parse_interval(record) = (parse(CRB_Q, String(record["lower_exact"])),
                              parse(CRB_Q, String(record["upper_exact"])))
function crb_mean_interval(records)
    isempty(records) && error("empty descriptive sample")
    pairs = crb_parse_interval.(records)
    lower = sum(first, pairs) / length(pairs)
    upper = sum(last, pairs) / length(pairs)
    midpoints = [Float64((lo + hi) / 2) for (lo, hi) in pairs]
    record = crb_interval_record(lower, upper)
    record["Monte_Carlo_SE_midpoint"] = length(midpoints) > 1 ? std(midpoints) / sqrt(length(midpoints)) : nothing
    record["Monte_Carlo_SE_scope"] = "sampling variation of geometric interval midpoints under the disclosed pseudo-probability reference; not econometric sampling uncertainty or exact B loss SE"
    record
end

function crb_observed_descriptive(actual)
    windows = Dict{String,Any}[]
    for old in actual.windows
        floor = parse(CRB_Q, String(old.probability_domain_floor.squared_distance_exact))
        function excess(fit)
            certificate = fit.certificate
            lo = parse(CRB_Q, String(certificate.squared_distance_lower_exact)) - floor
            hi = parse(CRB_Q, String(certificate.squared_distance_upper_exact)) - floor
            crb_interval_record(lo, hi)
        end
        push!(windows, Dict("cell"=>Int(old.cell), "years"=>Int.(old.years),
            "unchanged_raw_target_simplex_floor_exact"=>string(floor),
            "fixed_RUM_excess_loss"=>excess(old.fixed_RUM),
            "inner_profiles"=>[Dict("cap"=>Float64(profile.empty_allowance_cap),
                "excess_loss"=>excess(profile.CMU_actual_empty_inner)) for profile in old.profiles]))
    end
    Dict("scope"=>"Descriptive existing raw-affine fits with their saved simplex floors subtracted exactly. These raw estimands are NOT the pseudo-probability DGP and are not directly comparable to uniformly sampled loss levels.",
         "windows"=>windows)
end

function crb_summarize(cell_records, geometry_records; targets_per_prior)
    per_window = Dict{String,Any}[]
    for geometry in geometry_records, prior_index in eachindex(CRB_PRIORS)
        records = [r for r in cell_records[geometry["cell"]] if r["prior_index"] == prior_index]
        length(records) == targets_per_prior || error("incomplete reference sample")
        push!(per_window, Dict("cell"=>geometry["cell"], "years"=>geometry["years"],
            "alpha"=>CRB_PRIORS[prior_index], "targets"=>targets_per_prior,
            "RUM_MSE"=>crb_mean_interval([r["RUM_loss"] for r in records]),
            "WARP_MSE"=>crb_mean_interval([r["WARP_loss"] for r in records]),
            "inner_caps"=>[Dict("cap"=>cap,
                "MSE"=>crb_mean_interval([r["caps"][k]["loss"] for r in records]),
                "RUM_minus_inner_mean_gain"=>crb_mean_interval([r["caps"][k]["RUM_minus_inner_gain"] for r in records]),
                "WARP_minus_inner_mean_gain"=>crb_mean_interval([r["caps"][k]["WARP_minus_inner_gain"] for r in records]))
                for (k, cap) in enumerate(CRB_CAPS)]))
    end
    counts = [Int(g["frozen_annual_n_sum"]) for g in geometry_records]
    total = sum(counts)
    weights = CRB_Q.(counts) ./ total
    pooled = Dict{String,Any}[]
    for prior_index in eachindex(CRB_PRIORS)
        function aggregate(getter)
            draws = Dict{String,Any}[]
            for repetition in 1:targets_per_prior
                lo = hi = CRB_Q(0)
                for (g, weight) in zip(geometry_records, weights)
                    target = only(r for r in cell_records[g["cell"]] if
                                  r["prior_index"] == prior_index && r["repetition"] == repetition)
                    l, u = crb_parse_interval(getter(target))
                    lo += weight * l; hi += weight * u
                end
                push!(draws, crb_interval_record(lo, hi))
            end
            crb_mean_interval(draws)
        end
        rum = aggregate(r -> r["RUM_loss"])
        rlo, rhi = crb_parse_interval(rum)
        caps = [begin
            mse = aggregate(r -> r["caps"][k]["loss"])
            blo, bhi = crb_parse_interval(mse)
            ratio = rlo > 0 ? Dict("lower_exact"=>string(blo / rhi), "upper_exact"=>string(min(CRB_Q(1), bhi / rlo)),
                "lower"=>ceip_float_lower(blo / rhi), "upper"=>ceip_float_upper(min(CRB_Q(1), bhi / rlo))) : nothing
            Dict("cap"=>cap, "MSE"=>mse, "inner_to_RUM_ratio_of_mean_losses"=>ratio,
                "RUM_minus_inner_mean_gain"=>aggregate(r -> r["caps"][k]["RUM_minus_inner_gain"]),
                "WARP_minus_inner_mean_gain"=>aggregate(r -> r["caps"][k]["WARP_minus_inner_gain"]))
        end for (k, cap) in enumerate(CRB_CAPS)]
        push!(pooled, Dict("alpha"=>CRB_PRIORS[prior_index], "targets_per_window"=>targets_per_prior,
            "RUM_MSE"=>rum, "WARP_MSE"=>aggregate(r -> r["WARP_loss"]), "inner_caps"=>caps))
    end
    Dict("per_window"=>per_window, "pooled"=>pooled,
        "normalized_frozen_window_weights_exact"=>string.(weights),
        "normalized_frozen_window_weights"=>Float64.(weights),
        "unnormalized_original_window_weights_exact"=>string.(CRB_Q.(counts) ./ 46391),
        "reference_draws_are_independent_across_windows"=>true,
        "single_25_budget_probability_coupling_claimed"=>false,
        "p_values_power_membership_counts_or_model_volume_estimates"=>false,
        "bounds_scope"=>"RUM/WARP intervals are exact solver objective certificates. Positive-cap inner intervals are exact geometric ray/blend bounds, not claims of numerical optimization to their midpoint.")
end

function crb_crosscheck(records, Aw, groups; cells=(7, 12), caps=(.01, .10))
    isempty(records) && return Dict{String,Any}[]
    Int(records[1]["cell"]) in cells || return Dict{String,Any}[]
    family = ceip_build_family(Aw, groups; label="fixed_complete_WARP_for_ray_crosscheck")
    projector = CMUEmptyInnerProjector(family; tolerance=CRB_TOLERANCE)
    checks = Dict{String,Any}[]
    for prior_index in eachindex(CRB_PRIORS)
        target = only(r for r in records if r["prior_index"] == prior_index && r["repetition"] == 1)
        for cap in caps
            fit = ceip_fit!(projector, Float64.(target["point"]), cap)
            lo = parse(CRB_Q, String(fit["certificate"]["squared_distance_lower_exact"]))
            hi = parse(CRB_Q, String(fit["certificate"]["squared_distance_upper_exact"]))
            ray = only(b for b in target["caps"] if b["cap"] == cap)
            raylo, rayhi = crb_parse_interval(ray["loss"])
            raylo <= hi && lo <= rayhi || error("exact ray bracket and full-QP bracket are disjoint")
            push!(checks, Dict("cell"=>target["cell"], "alpha"=>target["alpha"],
                "repetition"=>1, "cap"=>cap, "exact_ray_interval"=>ray["loss"],
                "independent_full_Clarabel_interval"=>crb_interval_record(lo, hi),
                "intervals_overlap"=>true, "full_QP_certificate"=>fit["certificate"]))
        end
    end
    checks
end

function crb_protected_paths(root)
    paths = [joinpath(root, f) for f in (
        "simulate_cmu_restrictiveness.jl", "test/test_cmu_restrictiveness.jl",
        "cmu_superiority_floor_inference.jl", "cmu_empty_inner_projection.jl",
        "wilfried_normalized_working_set.jl", "src/fast_types.jl", "src/types.jl",
        "Project.toml", "Manifest.toml", "selection_solver_environment/Project.toml",
        "selection_solver_environment/Manifest.toml",
        "results/wilfried_windows_20261009_v2/summary.json",
        "results/cmu_actual_empty_comparison_20261011_v1/summary.json",
        "results/common_cmu_empty_witnesses_20261011_v1/summary.json",
        "results/common_cmu_empty_witnesses_20261011_v1/independent_audit.json")]
    append!(paths, [joinpath(root, "results/common_cmu_empty_witnesses_20261011_v1",
                             "cell_$(lpad(cell, 2, '0')).json") for cell in 1:19])
    # Preserve historical probes with their original source hashes. Their
    # metadata predated the main-run wording clarification, not the economics.
    for version in (1, 2)
        directory = joinpath(root, "results", "cmu_restrictiveness_probe_20261012_v$version")
        isdir(directory) || continue
        append!(paths, [joinpath(directory, name) for name in readdir(directory)
                        if isfile(joinpath(directory, name))])
    end
    paths
end

function crb_historical_probes(root)
    records = Dict{String,Any}[]
    for version in (1, 2)
        relative = "results/cmu_restrictiveness_probe_20261012_v$version"
        path = joinpath(root, relative, "protocol_frozen_before_outcomes.json")
        isfile(path) || continue
        protocol = crb_read(path)
        push!(records, Dict("directory"=>relative, "protocol_sha256"=>crb_sha(path),
            "historical_benchmark_source_sha256"=>String(protocol.source_and_input_sha256_before["simulate_cmu_restrictiveness.jl"]),
            "historical_records_preserved_not_rewritten"=>true))
    end
    records
end

function crb_run(; probe=false, root=@__DIR__)
    Threads.nthreads() >= CRB_WORKERS || error("six Julia threads required")
    BLAS.set_num_threads(1)
    output = joinpath(root, "results", probe ? "cmu_restrictiveness_probe_20261012_v2" : "cmu_restrictiveness_20261012_v1")
    ispath(output) && error("refusing existing output directory: $output")
    paths = crb_protected_paths(root)
    before = Dict(relpath(path, root)=>crb_sha(path) for path in paths)
    actual = crb_read(joinpath(root, "results/cmu_actual_empty_comparison_20261011_v1/summary.json"))
    targets_per_prior = probe ? 1 : CRB_M
    protocol = Dict("version"=>CRB_VERSION, "probe"=>probe,
        "cells"=>collect(1:19), "targets_per_prior_per_window"=>targets_per_prior,
        "priors"=>collect(CRB_PRIORS), "seed"=>CRB_SEED, "workers"=>CRB_WORKERS,
        "seed_rule"=>"seed + 1000003*cell + 100003*prior_index + 10007*repetition",
        "empty_allowance_caps"=>CRB_CAPS, "cap_arithmetic"=>"exact binary-rational Float64 cap, matching the existing frozen inner-family implementation",
        "models"=>["unchanged_normalized_complete_RUM", "unchanged_normalized_complete_WARP",
                     "certified_actual_empty_common_response_CMU_INNER_B_beta"],
        "inner_formula"=>"B_beta=(1-beta)*W+beta*S; beta=min(1,delta/((7-1)*(1-delta)))",
        "metric"=>"original full-coordinate squared Euclidean loss; no inverse, ridge, whitening or coordinate deletion",
        "loss_normalization"=>"mean over pseudo targets of SUM squared residuals across original patch coordinates; not per-coordinate MSE",
        "window_weighting"=>"normalize the original frozen annual_n_sum/46391 weights to sum one for this descriptive MSE benchmark; original statistic/weights unchanged",
        "alpha_1_generator"=>"independent ordered uniform spacings per original budget patch group",
        "alpha_025_generator"=>"Dirichlet(.25), exact normalization of binary-rational proposal and fixed largest remainder on denominator 2^40, stable original-row tie breaking",
        "no_draw_replacements"=>true, "no_arbitrary_budget_deletion"=>true,
        "frozen_before_main_campaign_run"=>true,
        "pilot_uses_first_target_per_prior"=>true,
        "no_outcome_based_design_changes"=>true,
        "prospective_preregistration_claimed"=>false,
        "metadata_only_changes_after_probe"=>["main-run freeze wording acknowledges pilot overlap", "explicit sum-of-coordinates loss normalization", "historical probe hashes preserved"],
        "historical_probes"=>crb_historical_probes(root),
        "B_bounds"=>"feasible blend upper plus optimized exact residual-ray dual lower; cap zero certified WARP projection; beta one analytic S identity",
        "inner_geometric_width_is_not_solver_width"=>true,
        "RUM_WARP_objective_tolerance"=>CRB_TOLERANCE,
        "full_Clarabel_crosschecks"=>"first target of each prior in cells7/12 at caps .01 and .10; exact interval-overlap check",
        "claim_scope"=>"reference-distribution geometric restrictiveness only; not statistical power, p-values, exact membership/model volume, all-CMU characterization or a single 25-budget population",
        "observed_raw_fits"=>"descriptive appendix only; unchanged target, exact saved floor subtraction; not used to generate pseudo probabilities",
        "source_and_input_sha256_before"=>before)
    mkpath(output)
    crb_new(joinpath(output, "protocol_frozen_before_outcomes.json"), protocol)
    started = time()
    cell_records = Dict{Int,Vector{Dict{String,Any}}}()
    geometry_records = Dict{String,Any}[]
    crosschecks = Dict{String,Any}[]
    for cell in 1:19
        try
            old = only(w for w in actual.windows if Int(w.cell) == cell)
            X = reduce(vcat, permutedims(Int8.(row)) for row in old.sign_matrix)
            groups = KSReplication._patch_groups(X)
            Ar = enumerate_types_fast(X; restriction=:rum, max_types=2_000_000, max_nodes=50_000_000)
            Aw = enumerate_types_fast(X; restriction=:warp, max_types=2_000_000, max_nodes=50_000_000)
            size(Ar, 2) == old.types.complete_RUM && size(Aw, 2) == old.types.complete_WARP || error("complete families changed")
            warp_set = Set(Tuple(view(Aw, :, h)) for h in axes(Aw, 2))
            all(h -> Tuple(view(Ar, :, h)) in warp_set, axes(Ar, 2)) || error("RUM subset WARP failed")
            witnesses = old.actual_empty_single_budget_witnesses
            old.all_singleton_columns_have_same_preference_across_budgets &&
                length(witnesses) == size(X, 1) &&
                sort([Int(witness.row) for witness in witnesses]) == collect(1:size(X, 1)) &&
                all(witness -> witness.arithmetic_whole_budget_certificate_recomputed, witnesses) ||
                error("external actual-empty witness registry is incomplete")
            geometry = Dict("cell"=>cell, "years"=>Int.(old.years), "sign_matrix"=>[Int.(row) for row in eachrow(X)],
                "budget_groups"=>groups, "RUM_types"=>size(Ar, 2), "WARP_types"=>size(Aw, 2),
                "RUM_matrix_sha256"=>bytes2hex(sha256(reinterpret(UInt8, vec(Ar)))),
                "WARP_matrix_sha256"=>bytes2hex(sha256(reinterpret(UInt8, vec(Aw)))),
                "frozen_annual_n_sum"=>Int(old.annual_n_sum),
                "actual_empty_microfoundation_scope"=>"unchanged certified complete WARP types, one-budget responders and all-null score; additional common-response restriction")
            push!(geometry_records, geometry)
            crb_new(joinpath(output, "geometry_$(lpad(cell, 2, '0')).json"), geometry)
            jobs = [(prior_index, repetition) for prior_index in eachindex(CRB_PRIORS) for repetition in 1:targets_per_prior]
            results = Vector{Dict{String,Any}}(undef, length(jobs))
            active_workers = min(CRB_WORKERS, length(jobs))
            @sync for worker in 1:active_workers
                Threads.@spawn begin
                    sourceR = csfi_make_fit(Ar, groups; tolerance=CRB_TOLERANCE, store_certificate=true)
                    sourceW = csfi_make_fit(Aw, groups; tolerance=CRB_TOLERANCE, store_certificate=true)
                    for job in worker:active_workers:length(jobs)
                        prior_index, repetition = jobs[job]
                        seed = CRB_SEED + 1_000_003cell + 100_003prior_index + 10_007repetition
                        probability = crb_probability(MersenneTwister(seed), groups, CRB_PRIORS[prior_index])
                        R = sourceR.fit(probability.point)
                        W = sourceW.fit(probability.point)
                        results[job] = crb_target_record(cell, prior_index, repetition, probability, R, W, sourceW; seed)
                    end
                end
            end
            cell_records[cell] = results
            append!(crosschecks, crb_crosscheck(results, Aw, groups))
            crb_new(joinpath(output, "cell_$(lpad(cell, 2, '0'))_targets.json"), results)
            println("restrictiveness cell=", cell, " targets=", length(results), " RUM=", size(Ar, 2),
                    " WARP=", size(Aw, 2), " elapsed_seconds=", round(time()-started; digits=2))
            flush(stdout)
        catch err
            crb_new(joinpath(output, "failed_checkpoint.json"), Dict("failed_cell"=>cell,
                "error"=>sprint(showerror, err, catch_backtrace()), "completed_cells"=>sort(collect(keys(cell_records))),
                "failed_or_replaced_targets_permitted"=>false))
            rethrow()
        end
    end
    summary = crb_summarize(cell_records, geometry_records; targets_per_prior)
    summary["status"] = probe ? "completed_all19_restrictiveness_probe_not_main_campaign" : "completed_conditional_reference_distribution_restrictiveness_benchmark"
    summary["protocol"] = protocol
    summary["elapsed_seconds"] = time() - started
    summary["RUM_WARP_projection_calls"] = 19 * length(CRB_PRIORS) * targets_per_prior * 2
    summary["full_Clarabel_crosschecks"] = crosschecks
    summary["observed_raw_descriptive_only"] = crb_observed_descriptive(actual)
    after = Dict(relpath(path, root)=>crb_sha(path) for path in paths)
    before == after || error("protected source/input changed during benchmark")
    summary["source_and_input_sha256_after"] = after
    summary["source_integrity_unchanged"] = true
    crb_new(joinpath(output, "summary.json"), summary)
    println("completed ", output, " elapsed_seconds=", summary["elapsed_seconds"])
    summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    ARGS == ["--probe"] ? crb_run(; probe=true) :
    ARGS == ["--frozen-all19"] ? crb_run(; probe=false) :
    error("requires --probe or explicitly approved --frozen-all19; no tuning/target overrides")
end
