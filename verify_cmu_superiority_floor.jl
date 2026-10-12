#!/usr/bin/env julia
"""Independent read-only audit of the frozen RUM-minus-simplex-floor campaign.

No inference engine, projection solver, or certificate helper is imported.
Complete RUM types are regenerated from the original sign geometry. Every
stored primal is reconstructed, every complete column is priced, and simplex
KKT equations and signed numerical-derivative endpoints are checked exactly.
The joint aggregation, MC tails, primary/sensitivity registry, critical-value
gates, canonical year pairing, and source/checkpoint hashes are recomputed.
This verifies arithmetic and lineage, not the CF-bootstrap CLT or bias claims.
"""
module IndependentSuperiorityFloorAudit

using KSReplication, JSON3, SHA, Random, Statistics, LinearAlgebra, Distributions

const Q = Rational{BigInt}
const ROOT = @__DIR__
const DEFAULT_DIRECTORY = joinpath(ROOT, "results/cmu_superiority_floor_20261012_v1")
const GRID = [.5, 1., 2.]
const B_EXPECTED = 1000
const N_EXPECTED = 46391
const RATE = sqrt(N_EXPECTED / 3)

check(condition, message) = condition || throw(ArgumentError(message))
readjson(path) = JSON3.read(read(path, String))
parseq(x) = parse(Q, String(x))
filehash(path) = bytes2hex(sha256(read(path)))
cellname(cell) = "cell_" * lpad(string(cell), 2, "0") * ".json"

function lower_float(x::Q)
    f = Float64(x)
    Q(f) <= x ? f : prevfloat(f)
end
function upper_float(x::Q)
    f = Float64(x)
    Q(f) >= x ? f : nextfloat(f)
end

function same(a, b)
    if a isa AbstractDict && b isa AbstractDict
        return Set(keys(a)) == Set(keys(b)) && all(same(a[k], b[k]) for k in keys(a))
    elseif a isa AbstractArray && b isa AbstractArray
        return size(a) == size(b) && all(same(a[i], b[i]) for i in eachindex(a))
    end
    a == b
end

function patch_groups(X)
    check(all(s -> s in (-1, 0, 1), X), "invalid patch sign alphabet")
    check(all(i -> count(==(0), view(X,i,:)) == 1, axes(X,1)), "patch has nonunique budget")
    groups = [findall(==(0), view(X,:,j)) for j in axes(X,2)]
    check(all(g -> !isempty(g), groups), "empty observed patch block")
    groups
end

function complete_context(A, groups)
    I,H = size(A); J = length(groups)
    check(I > 0 && H > 0 && all(v -> v in (0,1), A), "invalid complete incidence")
    rows = Matrix{Int}(undef,J,H)
    for h in 1:H
        check(sum(view(A,:,h)) == J, "incomplete rational type")
        for (j,g) in enumerate(groups)
            selected = [i for i in g if A[i,h] == 1]
            check(length(selected) == 1, "type not single-valued in a budget")
            rows[j,h] = only(selected)
        end
    end
    unique_types = Set(Tuple(view(rows,:,h)) for h in 1:H)
    check(length(unique_types) == H, "duplicate complete type")
    cartesian = big(H) == prod(big(length(g)) for g in groups)
    (; I,H,J,rows,groups,cartesian)
end

function _scan_support(rows, values)
    J,H = size(rows)
    best = sum(values[rows[j,1]] for j in 1:J)
    @inbounds for h in 2:H
        total = zero(eltype(values))
        for j in 1:J
            total += values[rows[j,h]]
        end
        best = max(best,total)
    end
    BigInt(best)
end

"""Independent integer support pricing; the bound controls EVERY partial sum."""
function support_integer(rows, numerators)
    bound = sum(abs, numerators)
    if bound <= big(typemax(Int128))
        return (; value=_scan_support(rows,Int128.(numerators)), mode="checked_Int128")
    end
    (; value=_scan_support(rows,numerators), mode="BigInt")
end

mutable struct Counters
    certificates::Int
    analytic_certificates::Int
    int128_supports::Int
    bigint_supports::Int
    slow_rational_crosschecks::Int
    maximum_objective_width::Q
end
Counters() = Counters(0,0,0,0,0,Q(0))

function simplex_floor_check(p, cert, groups)
    z = parseq.(cert["simplex_target_exact"])
    check(length(z) == length(p) && all(>=(0),z), "invalid simplex point")
    for g in groups
        check(sum(z[g]) == 1, "simplex block not normalized")
        active = [i for i in g if z[i] > 0]
        check(!isempty(active), "simplex block has no positive coordinate")
        theta = p[first(active)] - z[first(active)]
        check(all(i -> z[i] > 0 ? p[i]-z[i] == theta : p[i] <= theta, g),
            "independent exact simplex KKT fails")
    end
    floor = sum((p[i]-z[i])^2 for i in eachindex(p))
    check(floor == parseq(cert["probability_domain_floor_exact"]), "simplex floor value differs")
    floor
end

function verify_compact_certificate(context, target, cert, counters)
    check(Float64.(cert["target_float"]) == target, "certificate target changed")
    check(all(isfinite,target) && length(target) == context.I, "invalid target dimensions")
    p = Q.(target)
    floor = simplex_floor_check(p,cert,context.groups)
    counters.certificates += 1
    check(Bool(get(cert,"analytic_RUM_equals_product_simplex",false)) == context.cartesian,
        "certificate shortcut differs from exhaustive Cartesian type coverage")
    if get(cert,"analytic_RUM_equals_product_simplex",false)
        check(context.cartesian && Int(cert["verified_complete_cartesian_columns"]) == context.H,
            "invalid complete Cartesian identity shortcut")
        check(parseq(cert["Psi_lower_exact"]) == parseq(cert["Psi_upper_exact"]) == 0,
            "analytic identity has nonzero Psi")
        counters.analytic_certificates += 1
        return (; lower=Q(0),upper=Q(0),floor)
    end
    indices = Int.(cert["complete_column_indices"])
    mass_num = parse.(BigInt,String.(cert["population_mass_numerators"]))
    M = parse(BigInt,String(cert["population_mass_common_denominator"]))
    check(M > 0 && length(indices) == length(mass_num) && allunique(indices) &&
        all(i -> 1 <= i <= context.H,indices), "invalid complete primal registry")
    check(all(>=(0),mass_num) && sum(mass_num) == M, "primal masses not nonnegative normalized")
    fitted_num = zeros(BigInt,context.I)
    for (column,mass) in zip(indices,mass_num), j in 1:context.J
        fitted_num[context.rows[j,column]] += mass
    end
    check(all(sum(fitted_num[g]) == M for g in context.groups), "primal block response differs from one")
    D = foldl(lcm,denominator.(p);init=big(1))
    p_num = [numerator(x)*div(D,denominator(x)) for x in p]
    U = M*D
    residual_num = p_num.*M - fitted_num.*D
    pricing = support_integer(context.rows,residual_num)
    pricing.mode == "checked_Int128" ? (counters.int128_supports += 1) :
        (counters.bigint_supports += 1)
    norm = sum(x*x for x in residual_num)//(U*U)
    targetdot = sum(p_num[i]*residual_num[i] for i in eachindex(p_num))//(D*U)
    dual = 2targetdot - norm - 2*(pricing.value//U)
    lower = max(floor,dual); upper = norm
    check(0 <= floor <= lower <= upper, "independent primal/dual interval inverted")
    check(lower == parseq(cert["RUM_squared_distance_lower_exact"]) &&
        upper == parseq(cert["RUM_squared_distance_upper_exact"]), "exhaustive exact RUM distance interval differs")
    check(Int(cert["all_complete_columns_priced"]) == context.H &&
        cert["fixed_target_rational_certificate"], "complete support registry/scope differs")
    check(lower-floor == parseq(cert["Psi_lower_exact"]) &&
        upper-floor == parseq(cert["Psi_upper_exact"]), "exact floor-subtracted interval differs")
    check(upper-lower <= Q(1e-8), "unresolved exact projection interval")
    counters.maximum_objective_width = max(counters.maximum_objective_width,upper-lower)
    if context.H <= 2000 && counters.slow_rational_crosschecks < 6
        residual = residual_num.//U
        slow = maximum(sum(residual[context.rows[j,h]] for j in 1:context.J)
            for h in 1:context.H)
        check(slow == pricing.value//U, "integer support disagrees with slow rational pricing")
        check(sum((p[i]-fitted_num[i]//M)^2 for i in eachindex(p)) == upper,
            "integer objective disagrees with slow rational norm")
        counters.slow_rational_crosschecks += 1
    end
    (; lower=lower-floor,upper=upper-floor,floor)
end

function verify_saved_bridge(saved, old)
    check(Int(saved["cell"]) == Int(old["cell"]), "point bridge cell differs")
    r = old["fixed_RUM"]["certificate"]
    rlo,rhi = parseq(r["squared_distance_lower_exact"]),parseq(r["squared_distance_upper_exact"])
    qr = parseq.(r["conditional_fitted_exact"])
    floor = parseq(old["probability_domain_floor"]["squared_distance_exact"])
    qs = parseq.(old["probability_domain_floor"]["simplex_target_exact"])
    profiles = [p for p in old["profiles"] if p["empty_allowance_cap"] != 0]
    check(length(saved["positive_cap_checks"]) == length(profiles), "positive-cap bridge registry incomplete")
    for (s,p) in zip(saved["positive_cap_checks"],profiles)
        c = p["CMU_actual_empty_inner"]["certificate"]
        beta = parseq(c["unary_mix_fraction_exact"])
        clo,chi = parseq(c["squared_distance_lower_exact"]),parseq(c["squared_distance_upper_exact"])
        gainlo,gainhi = max(Q(0),rlo-chi),max(Q(0),rhi-clo)
        stronger = rlo-((1-beta)*rhi+beta*floor-beta*(1-beta)*sum(abs2,qr-qs))
        check(beta*(rlo-floor) <= gainhi && gainlo <= rhi-floor && stronger <= gainhi,
            "saved finite candidate contradicts superiority sandwich")
        check(s["cap"] == p["empty_allowance_cap"] && parseq(s["beta_exact"]) == beta &&
            parseq(s["beta_Psi_lower_exact"]) == beta*(rlo-floor) &&
            parseq(s["Psi_upper_exact"]) == rhi-floor &&
            parseq(s["feasible_candidate_gain_lower_exact"]) == stronger && s["interval_bridge_verified"],
            "reported positive-cap bridge values differ")
    end
end

function verify_window(envelope, old, source, cache, counters)
    cell = Int(envelope["cell"])
    check(cell == old["cell"] == source["cell"] == cache["cell"], "window source registry differs")
    check(old["years"] == source["years"] == cache["years"], "window calendar differs")
    X = reduce(vcat,permutedims(Int8.(r)) for r in source["sign_matrix"])
    check(same(old["sign_matrix"],source["sign_matrix"]), "original sign matrix changed")
    groups = patch_groups(X)
    A = KSReplication.enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
    context = complete_context(A,groups)
    check(context.H == old["types"]["complete_RUM"] &&
        context.H == envelope["projection_audit"]["complete_RUM_columns"], "complete RUM family differs")
    check(B_EXPECTED == envelope["B"] == cache["B"] && envelope["c_dimension"] == 3 &&
        envelope["coordinates"] == context.I, "draw/coordinate dimensions differ")
    check(envelope["matrix_storage"] == "column_major_B_by_c" &&
        !envelope["analytic_validation_fit_factory_used"], "real-data certificate storage scope differs")
    check(Bool(envelope["analytic_RUM_equals_product_simplex"]) == context.cartesian,
        "false analytic RUM identity shortcut")
    point = Float64.(cache["raw_affine_point"])
    check(point == Float64.(old["raw_affine_point_unchanged"]), "raw point target changed")
    stars = reshape(Float64.(cache["raw_affine_bootstrap"]),context.I,B_EXPECTED)
    check(all(isfinite,stars), "nonfinite original bootstrap target")
    base = verify_compact_certificate(context,point,envelope["point_exact_certificate"],counters)
    check(parseq(envelope["point_lower_exact"]) == base.lower &&
        parseq(envelope["point_upper_exact"]) == base.upper &&
        Float64(envelope["point_raw"]) == Float64((base.lower+base.upper)/2) &&
        Float64(envelope["point_lower"]) == lower_float(base.lower) &&
        Float64(envelope["point_upper"]) == upper_float(base.upper), "point interval fields differ")
    h = GRID./sqrt(RATE)
    check(Float64.(envelope["epsilons"]) == h &&
        Float64.(envelope["protocol"]["c_grid"]) == GRID &&
        envelope["protocol"]["c_primary"] == 1 && !envelope["protocol"]["c_selected_by_p_value"],
        "frozen numerical step grid changed")
    check(!envelope["protocol"]["raw_target_preprocessed_or_null_recentered"] &&
        !envelope["protocol"]["RUM_equals_WARP_shortcut_used"] &&
        !envelope["protocol"]["sampling_covariance_inverse_ridge_coordinate_deletion"] &&
        envelope["protocol"]["no_extra_half_factor"], "functional or sampling policy changed")
    central = reshape(Float64.(envelope["derivative_central"]),B_EXPECTED,3)
    lower = reshape(Float64.(envelope["derivative_lower"]),B_EXPECTED,3)
    upper = reshape(Float64.(envelope["derivative_upper"]),B_EXPECTED,3)
    lo_saved = reshape(collect(envelope["derivative_lower_exact"]),B_EXPECTED,3)
    hi_saved = reshape(collect(envelope["derivative_upper_exact"]),B_EXPECTED,3)
    draws = reshape(collect(envelope["per_draw_exact_certificates"]),B_EXPECTED,3)
    lo_exact,hi_exact = Matrix{Q}(undef,B_EXPECTED,3),Matrix{Q}(undef,B_EXPECTED,3)
    for k in 1:3, b in 1:B_EXPECTED
        target = point .+ (h[k]*RATE).*(view(stars,:,b).-point)
        draw = draws[b,k]
        fit = verify_compact_certificate(context,target,draw["RUM_floor_fit"],counters)
        hq = Q(h[k]); denominator = hq^2
        lo,hi = (fit.lower-base.upper)/denominator,(fit.upper-base.lower)/denominator
        signed = ((fit.lower+fit.upper)-(base.lower+base.upper))/(2denominator)
        check(parseq(draw["h_exact"]) == hq &&
            parseq(draw["derivative_lower_exact"]) == lo &&
            parseq(draw["derivative_upper_exact"]) == hi &&
            parseq(draw["signed_midpoint_derivative_exact"]) == signed &&
            parseq(lo_saved[b,k]) == lo && parseq(hi_saved[b,k]) == hi,
            "signed derivative scalar certificate differs at cell=$cell c=$(GRID[k]) draw=$b")
        check(lo <= signed <= hi && central[b,k] == Float64(signed) &&
            lower[b,k] == lower_float(lo) && upper[b,k] == upper_float(hi),
            "outward derivative endpoints or signed central value differ")
        lo_exact[b,k],hi_exact[b,k] = lo,hi
    end
    audit = envelope["projection_audit"]
    check(audit["all_complete_columns_priced"] && audit["exact_rational_point_and_perturbed_intervals"] &&
        audit["failed_or_replaced_draws"] == 0 && audit["private_mutable_QP_models"],
        "projection audit scope or failed-draw policy changed")
    check(audit["maximum_derivative_interval_width"] == maximum(upper-lower) &&
        sum(Int(v) for v in values(audit["solve_status_counts"])) == B_EXPECTED*3,
        "draw status registry or uncertainty diagnostic incomplete")
    verify_saved_bridge(envelope["saved_positive_cap_bridge_check"],old)
    (; cell,base,central,lo_exact,hi_exact,
        audit=Dict("cell"=>cell,"complete_RUM_types"=>context.H,"coordinates"=>context.I,
            "point_certificates_verified"=>1,"perturbed_certificates_verified"=>B_EXPECTED*3,
            "analytic_Cartesian_identity"=>context.cartesian))
end

function binomial_interval(k,B)
    [k == 0 ? 0. : quantile(Beta(k,B-k+1),.025),
     k == B ? 1. : quantile(Beta(k+1,B-k),.975)]
end

function verify_aggregate(saved, windows, weights, alpha)
    check(saved["alpha"] == alpha && saved["rate"] == RATE && saved["B"] == B_EXPECTED &&
        Int.(saved["windows"]) == collect(1:19), "aggregate registry changed")
    check(saved["primary_result_index_one_based"] == 2 && saved["primary_epsilon_multiplier"] == 1,
        "primary step changed")
    central = zeros(B_EXPECTED,3)
    lo_exact,hi_exact = zeros(Q,B_EXPECTED,3),zeros(Q,B_EXPECTED,3)
    pointlo,pointhi = Q(0),Q(0)
    for cell in 1:19
        w = Q(weights[cell]); check(w > 0,"nonpositive window weight")
        check(parseq(saved["weights_exact"][string(cell)]) == w,"exact binary window weight changed")
        pointlo += w*windows[cell].base.lower; pointhi += w*windows[cell].base.upper
        for k in 1:3, b in 1:B_EXPECTED
            central[b,k] += Float64(w)*windows[cell].central[b,k]
            lo_exact[b,k] += w*windows[cell].lo_exact[b,k]
            hi_exact[b,k] += w*windows[cell].hi_exact[b,k]
        end
    end
    observedlo,observedhi = Q(RATE)^2*pointlo,Q(RATE)^2*pointhi
    observed_lower,observed_upper = lower_float(observedlo),upper_float(observedhi)
    check(parseq(saved["point_Psi_lower_exact"]) == pointlo &&
        parseq(saved["point_Psi_upper_exact"]) == pointhi &&
        parseq(saved["observed_statistic_lower_exact"]) == observedlo &&
        parseq(saved["observed_statistic_upper_exact"]) == observedhi &&
        saved["observed_statistic_lower"] == observed_lower &&
        saved["observed_statistic_upper"] == observed_upper, "weighted point/statistic differs")
    check(length(saved["epsilon_results"]) == 3,"aggregate c grid incomplete")
    order = ceil(Int,(1-alpha)*(B_EXPECTED+1))
    rows = Dict{String,Any}[]
    for k in 1:3
        row = saved["epsilon_results"][k]
        lo,hi = lower_float.(lo_exact[:,k]),upper_float.(hi_exact[:,k])
        c = central[:,k]; width = maximum(hi-lo)
        critical,centralcritical = sort(hi)[order],sort(c)[order]
        resolved = isfinite(centralcritical) && centralcritical > width && centralcritical > 0
        tail = count(x -> x >= observed_lower,hi)
        p = (tail+1)/(B_EXPECTED+1)
        reject = resolved && observed_lower > critical && p <= alpha
        check(row["c_multiplier"] == GRID[k] && row["epsilon_multiplier"] == GRID[k] &&
            row["h"] == GRID[k]/sqrt(RATE) && row["epsilon"] == GRID[k]/sqrt(RATE) &&
            row["role"] == (k == 2 ? "frozen_primary" : "retained_sensitivity_NOT_selected"),
            "c registry/order or role changed")
        check(Float64.(row["bootstrap_statistics"]) == c &&
            Float64.(row["bootstrap_statistics_lower"]) == lo &&
            Float64.(row["bootstrap_statistics_upper"]) == hi &&
            parseq.(row["bootstrap_statistics_lower_exact"]) == lo_exact[:,k] &&
            parseq.(row["bootstrap_statistics_upper_exact"]) == hi_exact[:,k],
            "joint weighted bootstrap vectors differ")
        check(row["central_critical_value"] == centralcritical && row["critical_value_upper"] == critical &&
            row["maximum_projection_uncertainty"] == width && row["quantile_order_statistic_index"] == order &&
            row["critical_value_positive_resolved"] == resolved && row["exceedances"] == tail &&
            row["B"] == B_EXPECTED && row["p_value_mc"] == p && row["reject_conditional"] == reject &&
            row["negative_signed_derivative_draws"] == count(<(0),c), "joint critical/tail/gate differs")
        check(row["inferential_p_value_conditional"] === (resolved ? p : nothing) &&
            row["calibration_status"] == (resolved ? "conditional_finite_step_positive_critical" :
                "degenerate_or_unresolved_NO_inferential_rejection"), "unresolved critical promoted to inference")
        check(maximum(abs.(Float64.(row["p_value_mc_interval"])-binomial_interval(tail,B_EXPECTED))) <= 1e-12,
            "MC tail interval differs")
        exact_critical = sort(hi_exact[:,k])[order]
        exact_exceedances = count(x -> x >= observedlo,hi_exact[:,k])
        minimum_tail = count(x -> x >= observed_upper,lo)
        minimum_exact_tail = count(x -> x >= observedhi,lo_exact[:,k])
        check(minimum_tail <= minimum_exact_tail <= exact_exceedances <= tail,
            "solver-uncertainty tail interval or outward rounding differs")
        check(Q(critical) >= exact_critical && tail >= exact_exceedances,
            "outward rounding is not conservative for exact critical/tail")
        push!(rows,Dict("alpha"=>alpha,"c"=>GRID[k],"exceedances"=>tail,"p_value_mc"=>p,
            "critical_positive_resolved"=>resolved,"reject_conditional"=>reject,
            "exact_upper_critical"=>string(exact_critical),"exact_interval_exceedances"=>exact_exceedances,
            "solver_uncertainty_tail_range"=>[minimum_tail,tail],
            "exact_solver_uncertainty_tail_range"=>[minimum_exact_tail,exact_exceedances],
            "solver_uncertainty_p_range"=>[(minimum_tail+1)/(B_EXPECTED+1),p],
            "solver_uncertainty_has_no_effect_on_tail"=>minimum_tail == tail))
    end
    protocol = saved["protocol"]
    check(protocol["all19_required"] && protocol["no_Bonferroni_or_Holm"] &&
        !protocol["nearest_RUM_null_recenter_used"] && protocol["normal_residual_not_removed"] &&
        !protocol["ordinary_centered_gain_bootstrap_used"] && !protocol["c_selected_by_smallest_real_data_p"] &&
        !protocol["bootstrap_CLT_bias_survey_and_geometry_assumptions_verified"] &&
        !protocol["uniform_finite_sample_validity_claimed"] && protocol["sampling_significance_is_conditional"],
        "aggregate statistical scope changed")
    rows
end

function verify_calendar(windows, actual)
    folder = joinpath(ROOT,"results/wilfried_joint_gap_20261010_v1")
    frozen_text = read(joinpath(folder,"frozen_protocol_and_calendar_digests.json"),String)
    frozen = JSON3.read(frozen_text)
    prior = readjson(joinpath(folder,"summary.json"))
    protocol = prior["protocol"]
    check(same(protocol,frozen["protocol"]), "prior joint frozen protocol differs")
    years,annual = Int.(protocol["calendar_years"]),Int.(protocol["annual_n"])
    check(years == collect(1975:1999) && sum(annual) == N_EXPECTED &&
        protocol["N_unique"] == N_EXPECTED && protocol["B"] == B_EXPECTED &&
        protocol["seed"] == 13732, "canonical calendar plan changed")
    seeds = rand(MersenneTwister(13732),UInt64,length(years),B_EXPECTED)
    # The legacy seed_grid JSON is a lossy Float64 DISPLAY (scientific notation),
    # not the RNG input. Regenerate exact UInt64 seeds from the master seed and
    # verify their recorded SHA; then verify every household count digest below.
    # We never turn the rounded display values back into RNG seeds.
    check(reshape(Float64.(frozen["seed_grid"]),length(years),B_EXPECTED) == Float64.(seeds) &&
        bytes2hex(sha256(JSON3.write(seeds))) == protocol["seed_grid_sha256"], "canonical seed grid differs")
    digests = Matrix{String}(undef,length(years),B_EXPECTED)
    for j in eachindex(years), b in 1:B_EXPECTED
        rng = MersenneTwister(seeds[j,b]); counts = zeros(Int,annual[j])
        for draw in 1:annual[j]
            counts[rand(rng,1:annual[j])] += 1
        end
        check(sum(counts) == annual[j],"canonical frequency draw not normalized")
        digests[j,b] = bytes2hex(sha256(JSON3.write(counts)))
    end
    check(reshape(String.(frozen["calendar_count_digests"]),length(years),B_EXPECTED) == digests &&
        bytes2hex(sha256(JSON3.write(digests))) == protocol["count_digest_matrix_sha256"],
        "canonical count digests differ")
    maximum_adding_up = 0.
    for cell in 1:19
        old = actual["windows"][cell]
        cache = readjson(joinpath(folder,"raw_bootstrap_"*cellname(cell)))
        positions = [findfirst(==(y),years) for y in Int.(old["years"])]
        check(Int.(cache["canonical_year_seed_rows"]) == positions &&
            cache["year_count_digests_sha256"] == bytes2hex(sha256(JSON3.write(digests[positions,:]))) &&
            Int.(old["annual_n"]) == annual[positions], "cached windows no longer share calendar draws")
        I = Int(cache["coordinates"])
        stars = reshape(Float64.(cache["raw_affine_bootstrap"]),I,B_EXPECTED)
        check(bytes2hex(sha256(JSON3.write(stars))) == cache["raw_affine_bootstrap_sha256"],
            "raw bootstrap matrix hash differs")
        diagnostics = cache["refit_diagnostics"]
        check(diagnostics["both_CF_stages_refit"] && diagnostics["failed_or_replaced_draws"] == 0 &&
            diagnostics["raw_affine_probability_policy"] && diagnostics["all_original_annual_n_retained"],
            "cached CF refit scope differs")
        X = reduce(vcat,permutedims(Int8.(r)) for r in old["sign_matrix"])
        for g in patch_groups(X)
            maximum_adding_up = max(maximum_adding_up,maximum(abs.(vec(sum(stars[g,:];dims=1)).-1)))
        end
    end
    check(maximum_adding_up <= 1e-10,"raw bootstrap adding-up failed")
    Dict("canonical_years"=>25,"canonical_draws"=>B_EXPECTED,
        "seed_grid_and_25000_count_digests_regenerated"=>true,
        "legacy_seed_grid_is_lossy_display_not_RNG_source"=>true,
        "exact_UInt64_seeds_regenerated_and_SHA_verified"=>true,
        "maximum_raw_bootstrap_blocksum_error"=>maximum_adding_up)
end

function verify_superiority_floor(directory=DEFAULT_DIRECTORY; calendar=true)
    directory = abspath(directory)
    summary = readjson(joinpath(directory,"summary.json"))
    frozen = readjson(joinpath(directory,"frozen_protocol.json"))
    check(summary["status"] == "completed_conditional_null_equivalent_superiority_test" &&
        same(summary["protocol"],frozen["protocol"]), "completed frozen campaign missing or changed")
    check(!isfile(joinpath(directory,"failed_checkpoint.json")), "failed campaign cannot be certified")
    protocol = summary["protocol"]
    check(Int.(protocol["windows"]) == collect(1:19) && protocol["N_unique"] == N_EXPECTED &&
        protocol["B"] == B_EXPECTED && protocol["rate"] == RATE &&
        Float64.(protocol["epsilon_multipliers"]) == GRID && protocol["primary_epsilon_multiplier"] == 1 &&
        !protocol["c_selected_by_real_data_tail"] && protocol["no_Bonferroni_or_Holm"] &&
        !protocol["failed_or_replaced_draws_permitted"] && protocol["no_old_model_p_values_reused"],
        "frozen real-data test registry changed")
    check(same(summary["source_and_implementation_sha256_before"],summary["source_and_implementation_sha256_after"]) &&
        same(summary["source_and_implementation_sha256_before"],frozen["source_and_implementation_sha256_before"]),
        "before/after source hashes differ")
    for (relative,digest) in pairs(summary["source_and_implementation_sha256_after"])
        check(filehash(normpath(joinpath(ROOT,String(relative)))) == String(digest),
            "current frozen source changed: $relative")
    end
    actual = readjson(joinpath(ROOT,"results/cmu_actual_empty_comparison_20261011_v1/summary.json"))
    check(length(actual["windows"]) == 19,"actual-CMU source windows incomplete")
    weights = [actual["windows"][c]["annual_n_sum"]/N_EXPECTED for c in 1:19]
    check(Float64.(protocol["window_weights"]) == weights,"frozen window weights changed")
    names = cellname.(1:19)
    check(String.(summary["checkpoint_files"]) == names && length(summary["checkpoint_sha256"]) == 19,
        "checkpoint registry incomplete")
    windows = Vector{Any}(undef,19)
    # Each window has a private exact-arithmetic counter and parser buffer.
    # Static threading preserves deterministic window/aggregate ordering and
    # allows a bounded JULIA_NUM_THREADS=4 independent audit on local resources.
    window_counters = [Counters() for cell in 1:19]
    Threads.@threads :static for cell in 1:19
        name = names[cell]; path = joinpath(directory,name)
        check(filehash(path) == String(summary["checkpoint_sha256"][cell]),"checkpoint hash differs")
        source = readjson(joinpath(ROOT,"results/wilfried_windows_20261009_v2",name))
        cache = readjson(joinpath(ROOT,"results/wilfried_joint_gap_20261010_v1","raw_bootstrap_"*name))
        envelope = readjson(path)
        try
            windows[cell] = verify_window(envelope,actual["windows"][cell],source,cache,window_counters[cell])
        catch err
            error("independent audit failed in window $cell: "*sprint(showerror,err))
        end
        println(stderr,"Independent superiority audit: window $cell; ",window_counters[cell].certificates," certificates")
    end
    counters = Counters(sum(c.certificates for c in window_counters),
        sum(c.analytic_certificates for c in window_counters),
        sum(c.int128_supports for c in window_counters),
        sum(c.bigint_supports for c in window_counters),
        sum(c.slow_rational_crosschecks for c in window_counters),
        maximum(c.maximum_objective_width for c in window_counters))
    check(counters.certificates == 19*(B_EXPECTED*3+1),"point or perturbed certificate omitted")
    joint = Dict{String,Any}()
    for (key,alpha) in (("0.05",.05),("0.10",.10))
        joint[key] = verify_aggregate(summary["joint_tests"][key],windows,weights,alpha)
    end
    calendar_audit = calendar ? verify_calendar(windows,actual) : nothing
    Dict("status"=>"independent_all19_superiority_floor_audit_passed","directory"=>directory,
        "verifier_sha256"=>filehash(@__FILE__),"summary_sha256"=>filehash(joinpath(directory,"summary.json")),
        "point_certificates_verified"=>19,"perturbed_certificates_verified"=>19*B_EXPECTED*3,
        "total_certificates_verified"=>counters.certificates,"analytic_Cartesian_certificates"=>counters.analytic_certificates,
        "checked_Int128_supports"=>counters.int128_supports,"BigInt_supports"=>counters.bigint_supports,
        "slow_rational_support_crosschecks"=>counters.slow_rational_crosschecks,
        "private_window_arithmetic_workers"=>Threads.nthreads(),
        "maximum_exact_projection_width"=>string(counters.maximum_objective_width),
        "all_complete_columns_priced_independently"=>true,"signed_derivative_intervals_verified"=>true,
        "joint_aggregation_critical_values_tails_and_gates_verified"=>true,"no_omitted_or_replaced_draws"=>true,
        "frozen_source_and_checkpoint_hashes_verified"=>true,"calendar_audit"=>calendar_audit,
        "joint_results"=>joint,"statistical_sampling_assumptions_verified"=>false,
        "scope"=>"Exact arithmetic and source-lineage audit, conditional on the maintained geometric null equivalence and valid joint raw-CF bootstrap/rate/bias/critical-quantile assumptions. Not a proof of unconditional or uniform test size.",
        "new_superiority_inference_engine_or_certificate_helper_imported"=>false)
end

function self_test()
    A = Int8[1 0 0;0 1 1;0 1 0;1 0 1]
    context = complete_context(A,[[1,2],[3,4]])
    for nums in (BigInt[1,-2,3,4],big(10)^50*BigInt[1,-2,3,4])
        priced = support_integer(context.rows,nums)
        slow = maximum(sum(Q(nums[context.rows[j,h]]) for j in 1:2) for h in 1:3)
        check(Q(priced.value) == slow,"integer support self-test differs")
    end
    target = [.75,.25,.75,.25]
    cert = Dict("target_float"=>target,"complete_column_indices"=>[1,2],
        "population_mass_common_denominator"=>"2","population_mass_numerators"=>["1","1"],
        "simplex_target_exact"=>string.(Q.(target)),"probability_domain_floor_exact"=>"0//1",
        "RUM_squared_distance_lower_exact"=>"1//4","RUM_squared_distance_upper_exact"=>"1//4",
        "Psi_lower_exact"=>"1//4","Psi_upper_exact"=>"1//4","all_complete_columns_priced"=>3,
        "fixed_target_rational_certificate"=>true)
    result = verify_compact_certificate(context,target,cert,Counters())
    check(result.lower == result.upper == 1//4,"exact compact certificate self-test differs")
    tampered = deepcopy(cert); tampered["population_mass_numerators"] = ["-1","3"]
    rejected = false
    try
        verify_compact_certificate(context,target,tampered,Counters())
    catch err
        rejected = err isa ArgumentError
    end
    check(rejected,"tampered negative population mass accepted")
    true
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    if ARGS == ["--self-test"]
        println("independent_superiority_verifier_self_test=",IndependentSuperiorityFloorAudit.self_test())
    else
        length(ARGS) <= 1 || error("Usage: julia --project=. verify_cmu_superiority_floor.jl [REPORT_DIRECTORY]")
        directory = isempty(ARGS) ? IndependentSuperiorityFloorAudit.DEFAULT_DIRECTORY : only(ARGS)
        println(IndependentSuperiorityFloorAudit.JSON3.write(
            IndependentSuperiorityFloorAudit.verify_superiority_floor(directory)))
    end
end
