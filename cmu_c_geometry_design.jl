# Geometry-only inputs for a bounded, per-window Gaussian c-calibration study.
# All existing geometries, data, estimates, packages and reports are read-only.
using KSReplication, JSON3, LinearAlgebra, Statistics, SHA

const CSDG_VERSION = "cmu-c-geometry-design-v1"
const CSDG_CELLS = (7, 12)
const CSDG_Q = Rational{BigInt}
const CSDG_DEFAULT_RATE = sqrt(46391 / 3)

csdg_sha(path) = bytes2hex(sha256(read(path)))

"""Strictly positive, nearly uniform DYADIC weights on every listed column.

The power-of-two denominator is the smallest such denominator not below n.
The first remainder columns receive one additional count. This is a fixed
enumeration-order construction, not a selection using observed probabilities.
"""
function csdg_dyadic_weights(n::Integer)
    n > 0 || throw(ArgumentError("a nonempty column set is required"))
    denominator = big(1)
    while denominator < n
        denominator <<= 1
    end
    base, remainder = divrem(denominator, big(n))
    counts = fill(base, Int(n))
    for i in 1:Int(remainder)
        counts[i] += 1
    end
    all(>(0), counts) && sum(counts) == denominator || error("dyadic weight construction failed")
    (; counts, denominator, weights=counts .// denominator)
end

function csdg_column_mixture(A, indices)
    weights = csdg_dyadic_weights(length(indices))
    numerators = zeros(BigInt, size(A, 1))
    for (column, count) in zip(indices, weights.counts), row in axes(A, 1)
        A[row, column] != 0 && (numerators[row] += count)
    end
    point_exact = numerators .// weights.denominator
    point = Float64.(point_exact)
    CSDG_Q.(point) == point_exact || error("null mixture is not exactly representable in Float64")
    (; point, point_exact, numerators, weights)
end

"""No rank truncation: F' is B-by-I; the full thin R is I-by-I.

All I Gaussian factor columns are retained, including singular directions.
The estimated covariance is F*F' and the returned L=R' has L*L'=F*F'
up to floating arithmetic. There is no covariance inverse or ridge.
"""
function csdg_covariance_factor(stars, rate)
    I, B = size(stars)
    B >= I && B > 1 || throw(ArgumentError("full thin QR requires B >= I and B > 1"))
    rate > 0 && isfinite(rate) || throw(ArgumentError("positive finite rate required"))
    bootstrap_mean = vec(mean(stars; dims=2))
    F = (stars .- bootstrap_mean) .* (rate / sqrt(B - 1))
    R = Matrix(qr(Matrix(transpose(F))).R)
    size(R) == (I, I) || error("thin QR unexpectedly changed the coordinate dimension")
    L = Matrix(transpose(R))
    covariance = F * transpose(F)
    reconstructed = L * transpose(L)
    discrepancy = maximum(abs.(reconstructed .- covariance))
    scale = max(1., maximum(abs.(covariance)))
    discrepancy <= 1e-12 * scale || error("full QR covariance check failed")
    (; L, F, covariance, bootstrap_mean, discrepancy, B)
end

"""Construct, without fitting outcomes, the two approved local geometries.

The first enumerated complete WARP column absent from the complete RUM set is
w. The exact exposed RUM face maximizes w'A; it need not be a facet. Every
maximizing column receives a positive dyadic weight. The null centroid similarly
uses all complete RUM columns with positive dyadic weights.

Gaussian covariance is an estimated nuisance from the unchanged local raw
bootstrap cache. These are per-window Gaussian functional stress designs,
NOT a claim of one joint utility distribution over all 25 annual budgets.
"""
function csdg_load_design(cell::Integer; root=@__DIR__, rate=CSDG_DEFAULT_RATE,
        signal_levels=(1., 2.), mixture_denominator_power=32)
    cell in CSDG_CELLS || throw(ArgumentError("approved geometry cells are 7 and 12 only"))
    16 <= mixture_denominator_power <= 40 ||
        throw(ArgumentError("dyadic alternative precision must lie between 16 and 40 bits"))
    all(level -> isfinite(level) && level > 0, signal_levels) ||
        throw(ArgumentError("positive finite signal levels required"))
    source_path = joinpath(root, "results/wilfried_windows_20261009_v2/summary.json")
    cache_path = joinpath(root, "results/wilfried_joint_gap_20261010_v1",
                          "raw_bootstrap_cell_$(lpad(cell, 2, '0')).json")
    protected = [source_path, cache_path, joinpath(root, "Project.toml"),
                 joinpath(root, "Manifest.toml"), joinpath(root, "src/fast_types.jl"),
                 joinpath(root, "src/types.jl")]
    hashes_before = Dict(relpath(path, root) => csdg_sha(path) for path in protected)
    source = JSON3.read(read(source_path, String))
    cases = [case for case in source.cases if Int(case.cell) == cell]
    length(cases) == 1 || error("source must contain exactly one requested geometry")
    case = only(cases)
    X = reduce(vcat, permutedims(Int8.(row)) for row in case.sign_matrix)
    groups = KSReplication._patch_groups(X)
    length(groups) == 7 || error("the approved design requires the original seven budgets")
    Ar = enumerate_types_fast(X; restriction=:rum, max_types=2_000_000, max_nodes=50_000_000)
    Aw = enumerate_types_fast(X; restriction=:warp, max_types=2_000_000, max_nodes=50_000_000)
    size(Ar, 2) < size(Aw, 2) || error("selected geometry must have strict RUM inclusion in WARP")
    rum_columns = Set(Tuple(view(Ar, :, h)) for h in axes(Ar, 2))
    warp_columns = Set(Tuple(view(Aw, :, h)) for h in axes(Aw, 2))
    all(column -> column in warp_columns, rum_columns) || error("regenerated RUM is not a subset of WARP")
    excluded_index = findfirst(h -> !(Tuple(view(Aw, :, h)) in rum_columns), axes(Aw, 2))
    excluded_index === nothing && error("no excluded WARP vertex found")
    w = Vector{Int8}(view(Aw, :, excluded_index))
    all(sum(w[g]) == 1 for g in groups) || error("excluded vertex violates budget adding-up")
    scores = [sum(Int(w[row]) * Int(Ar[row, h]) for row in axes(Ar, 1)) for h in axes(Ar, 2)]
    support = maximum(scores)
    face_indices = findall(==(support), scores)
    excluded_score = sum(Int, w)
    support < excluded_score || error("the chosen vertex is not strictly separated by its own selector")
    face = csdg_column_mixture(Ar, face_indices)
    centroid = csdg_column_mixture(Ar, collect(axes(Ar, 2)))
    all(sum(face.point_exact[g]) == 1 && sum(centroid.point_exact[g]) == 1 for g in groups) ||
        error("exact null adding-up failed")
    sum(CSDG_Q.(w) .* face.point_exact) == support || error("exposed-face equality failed")
    all(score <= support for score in scores) || error("exact complete support audit failed")
    # Removing block constants is an affine-space equivalent normal, not a
    # deletion of any observed coordinate or a change to the projection metric.
    normal_exact = CSDG_Q.(w)
    for group in groups
        normal_exact[group] .-= sum(normal_exact[group]) / length(group)
    end
    normal = Float64.(normal_exact)
    face_matrix = Float64.(Ar[:, face_indices])
    face_dimension = rank(face_matrix .- face_matrix[:, 1])
    rum_dimension = rank(Float64.(Ar) .- Float64.(Ar[:, 1]))
    cache = JSON3.read(read(cache_path, String))
    Int(cache.cell) == cell && Int.(cache.years) == Int.(case.years) || error("cache geometry identifier mismatch")
    Int(cache.coordinates) == size(X, 1) || error("cache coordinate dimension mismatch")
    stars = reshape(Float64.(cache.raw_affine_bootstrap), size(X, 1), Int(cache.B))
    all(isfinite, stars) || error("raw bootstrap contains nonfinite values")
    noise = csdg_covariance_factor(stars, Float64(rate))
    adding_up_error = maximum(abs.(reduce(vcat,
        permutedims(vec(sum(noise.F[g, :]; dims=1))) for g in groups)))
    adding_up_error <= 1e-9 || error("raw cache factor does not preserve budget adding-up")
    score_sd = norm(transpose(noise.L) * Float64.(w))
    score_sd > 0 && isfinite(score_sd) || error("exposed score has zero or invalid estimated noise")
    separation = excluded_score - support
    mixing_denominator = big(1) << mixture_denominator_power
    alternatives = map(collect(signal_levels)) do requested_level
        requested_t = Float64(requested_level) * score_sd / (Float64(rate) * separation)
        capped_t = min(1., requested_t)
        mixing_count = clamp(round(BigInt, capped_t * Float64(mixing_denominator)),
                             big(0), mixing_denominator)
        mixing_count > 0 || error("requested alternative rounded to the null; increase dyadic precision")
        t_exact = mixing_count // mixing_denominator
        point_exact = (1 - t_exact) .* face.point_exact .+ t_exact .* CSDG_Q.(w)
        all(sum(point_exact[g]) == 1 && all(>=(0), point_exact[g]) for g in groups) ||
            error("alternative is not exactly in the product simplex")
        actual_score_shift = sum(CSDG_Q.(w) .* point_exact) - support
        actual_score_shift > 0 || error("alternative is not strictly beyond the RUM supporting plane")
        point = Float64.(point_exact)
        CSDG_Q.(point) == point_exact || error("alternative lost exact dyadic representability")
        (; requested_level=Float64(requested_level), point, point_exact, t_exact,
           requested_t, mixture_capped=requested_t > 1., actual_score_shift,
           achieved_level=Float64(rate) * Float64(actual_score_shift) / score_sd)
    end
    hashes_after = Dict(relpath(path, root) => csdg_sha(path) for path in protected)
    hashes_before == hashes_after || error("protected source changed during construction")
    metadata = Dict(
        "version" => CSDG_VERSION, "cell" => Int(cell), "years" => Int.(case.years),
        "selection_rule" => "cells 7 and 12: geometry-only small strict complete RUM subset WARP cases; no empirical fit gain or tail used",
        "geometry_dimensions" => [size(X, 1), size(X, 2)],
        "RUM_types" => size(Ar, 2), "WARP_types" => size(Aw, 2),
        "excluded_WARP_column_index" => excluded_index, "w" => Int.(w),
        "support_exact" => string(CSDG_Q(support)), "excluded_score_exact" => string(CSDG_Q(excluded_score)),
        "all_RUM_column_scores_exact" => scores, "face_column_indices" => face_indices,
        "face_positive_weight_numerators" => string.(face.weights.counts),
        "face_weight_denominator" => string(face.weights.denominator),
        "q0_exact" => string.(face.point_exact), "centroid_exact" => string.(centroid.point_exact),
        "centroid_positive_weight_numerators" => string.(centroid.weights.counts),
        "centroid_weight_denominator" => string(centroid.weights.denominator),
        "affine_centered_normal_exact" => string.(normal_exact),
        "face_dimension_numeric_diagnostic_only" => face_dimension,
        "RUM_dimension_numeric_diagnostic_only" => rum_dimension,
        "face_claim" => "exposed face, not asserted to be a facet",
        "rate" => Float64(rate), "cache_B" => noise.B,
        "factor_shape" => collect(size(noise.L)),
        "covariance_construction" => "center raw bootstrap; F=rate*(stars-rowmean)/sqrt(B-1); L=R' from full thin QR of F'; retain all I coordinates and I factor columns",
        "covariance_inverse_ridge_or_rank_truncation" => false,
        "QR_covariance_max_abs_error" => noise.discrepancy,
        "noise_budget_adding_up_max_abs_error" => adding_up_error,
        "exposed_score_noise_sd" => score_sd,
        "alternatives" => [Dict("requested_SE_level" => alt.requested_level,
             "achieved_SE_level" => alt.achieved_level,
             "requested_mixture" => alt.requested_t, "mixture_exact" => string(alt.t_exact),
             "simplex_cap_applied" => alt.mixture_capped,
             "score_shift_exact" => string(alt.actual_score_shift),
             "point_exact" => string.(alt.point_exact)) for alt in alternatives],
        "scope" => "per-window full-coordinate Gaussian functional stress under frozen estimated nuisance; not all-19 joint c selection, household CF resimulation, or a single 25-budget utility coupling",
        "observed_fit_gains_or_empirical_pvalues_used" => false,
        "protected_sha256_before" => hashes_before, "protected_sha256_after" => hashes_after,
        "helper_source_sha256" => csdg_sha(@__FILE__))
    (; cell=Int(cell), years=Int.(case.years), X, Ar, Aw, groups,
       centroid=centroid.point, centroid_exact=centroid.point_exact,
       q0=face.point, q0_exact=face.point_exact,
       w=Float64.(w), w_exact=CSDG_Q.(w), normal, normal_exact,
       support, face_indices, face_weights_exact=face.weights.weights,
       factor=noise.L, covariance=noise.covariance, rate=Float64(rate),
       score_sd, alternatives, metadata)
end

"""Adapter consumed by the isolated local Gaussian campaign driver.

The four scenario means are constructed from geometry only. `signal_se` is
the achieved exposed-score shift, not an observed empirical fit improvement.
"""
function ccgd_design(cell::Integer; rate=CSDG_DEFAULT_RATE, kwargs...)
    d = csdg_load_design(cell; rate, kwargs...)
    length(d.alternatives) == 2 || error("campaign adapter requires exactly one- and two-SE alternatives")
    requested = [alt.requested_level for alt in d.alternatives]
    requested == [1., 2.] || error("campaign adapter fixes requested signals at one and two SE")
    scenarios = [
        (; id="centroid_null", null=true, mean=d.centroid, signal_se=0.),
        (; id="exposed_face_null", null=true, mean=d.q0, signal_se=0.),
        (; id="oneSE", null=false, mean=d.alternatives[1].point,
           signal_se=d.alternatives[1].achieved_level),
        (; id="twoSE", null=false, mean=d.alternatives[2].point,
           signal_se=d.alternatives[2].achieved_level)
    ]
    record = copy(d.metadata)
    record["scenarios"] = [Dict("id"=>s.id, "null"=>s.null,
        "mean_exact"=>string.(CSDG_Q.(s.mean)), "signal_se"=>s.signal_se) for s in scenarios]
    (; cell=d.cell, years=d.years, A=d.Ar, groups=d.groups, L=d.factor,
       scenarios, record, design=d)
end
