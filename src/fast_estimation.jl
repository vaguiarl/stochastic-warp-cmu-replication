"""Immutable-by-contract per-year preparation; contains no resampled household matrix."""
struct PreparedKSYear
    year::Int
    n::Int
    expenditure::Vector{Float64}
    log_expenditure::Vector{Float64}
    instrument_basis::Matrix{Float64}
    expenditure_order::Vector{Int}
    patch_positions::Vector{Int}
    patch_rows::Vector{Int}
end

"""Read-only preparation shared across worker-local exact-frequency fits.

Treat these arrays as immutable after preparation. They are copies or derived
arrays, independent of the caller's original annual household arrays.
"""
struct PreparedKSEstimator
    years::Vector{PreparedKSYear}
    degree::Int
    evaluation_log_expenditure::Vector{Float64}
    coordinates::Int
end

"""Prepare fixed logs, raw-Z basis, expenditure order, and open-patch IDs once.

This performs all original input/patch validation before parallel fitting.
The default evaluation remains the full-sample median log E, not a bootstrap
median. It does not change prices, patches, degree, smoothing, or clipping.
"""
function prepare_estimator(annual::Vector{KSAnnual}, X::AbstractMatrix;
        degree=3, evaluation_log_expenditure=nothing)
    degree = _checked_degree(degree)
    isempty(annual) && throw(ArgumentError("annual inputs must be nonempty"))
    groups = _patch_groups(X)
    length(groups) == length(annual) || throw(ArgumentError("annual/budget counts disagree"))
    sizes = [_validate_annual(a) for a in annual]
    length(unique(last.(sizes))) == 1 || throw(ArgumentError("annual goods counts disagree"))
    length(unique(a.year for a in annual)) == length(annual) ||
        throw(ArgumentError("annual years must be distinct"))
    evaluation = evaluation_log_expenditure === nothing ?
        [median(log.(a.expenditure)) for a in annual] : Float64.(collect(evaluation_log_expenditure))
    length(evaluation) == length(annual) && all(isfinite, evaluation) ||
        throw(ArgumentError("one finite log-expenditure evaluation per year is required"))
    prices = reduce(vcat, transpose(a.normalized_prices) for a in annual)
    prepared = PreparedKSYear[]
    for (budget, a) in enumerate(annual)
        n = sizes[budget][1]
        n >= 8 || throw(ArgumentError("smooth rank clamp requires n >= 8"))
        rows = _classify_patch_rows(a.shares, a.normalized_prices, prices, budget, groups, X)
        positions = Dict(row => column for (column, row) in enumerate(groups[budget]))
        expenditure = copy(a.expenditure)
        push!(prepared, PreparedKSYear(a.year, n, expenditure, log.(expenditure),
            _instrument_basis(a.instrument, degree), sortperm(expenditure; alg=MergeSort),
            [positions[row] for row in rows], copy(groups[budget])))
    end
    return PreparedKSEstimator(prepared, degree, copy(evaluation), size(X, 1))
end

function _frequency_counts(prepared::PreparedKSEstimator; indices=nothing, counts=nothing)
    indices === nothing || counts === nothing ||
        throw(ArgumentError("supply either indices or counts, not both"))
    if indices === nothing && counts === nothing
        return [ones(Int, year.n) for year in prepared.years]
    end
    inputs = indices === nothing ? counts : indices
    length(inputs) == length(prepared.years) ||
        throw(ArgumentError("one draw vector per year is required"))
    output = Vector{Vector{Int}}(undef, length(prepared.years))
    for (budget, year) in enumerate(prepared.years)
        input = inputs[budget]
        if indices !== nothing
            length(input) == year.n && all(x -> x isa Integer && 1 <= x <= year.n, input) ||
                throw(ArgumentError("bootstrap indices must retain annual sample size and valid integer rows"))
            count = zeros(Int, year.n)
            for id in input
                count[id] += 1
            end
            output[budget] = count
        else
            length(input) == year.n && all(x -> x isa Integer && 0 <= x <= year.n, input) ||
                throw(ArgumentError("frequency counts must be nonnegative integers of annual length"))
            count = Int.(input)
            sum(count) == year.n || throw(ArgumentError("frequency counts must sum to original annual n"))
            output[budget] = count
        end
    end
    return output
end

"""Weighted column-scaled QR with the reference's original-n rank threshold."""
function _frequency_qr(basis::AbstractMatrix, sqrt_counts::AbstractVector,
                       original_n::Int; label="weighted basis")
    size(basis, 1) == length(sqrt_counts) || throw(DimensionMismatch("frequency basis rows"))
    weighted = Matrix{Float64}(basis) .* sqrt_counts
    fit = _scaled_qr(weighted; label)
    diagonals = abs.(diag(fit.factor.R))
    tolerance = max(original_n, size(basis, 2)) * eps(Float64) * maximum(diagonals)
    count(>(tolerance), diagonals) == size(basis, 2) ||
        throw(ArgumentError("$label is numerically rank deficient"))
    return fit
end

"""Exact repeated-row conditional-CDF projection, compressed by integer frequency.

If A=sqrt(W)R=QR, a selected original row's fitted CDF is
Q_h/sqrt(w_h) dotted with sum_{E_k<=E_h} sqrt(w_k)Q_k. This is the same
R_h(R'WR)^(-1)R'W 1{E_h>=E_k} as fitting the duplicated bootstrap sample.
All expenditure ties are completed before their CDF is evaluated.
"""
function _frequency_conditional_ranks(year::PreparedKSYear, counts::Vector{Int})
    active = findall(>(0), counts)
    sqrt_counts = sqrt.(Float64.(counts[active]))
    fit = _frequency_qr(view(year.instrument_basis, active, :), sqrt_counts, year.n;
                        label="frequency instrument basis")
    m, p = length(active), size(year.instrument_basis, 2)
    Q = fit.factor.Q * Matrix{Float64}(I, m, p)
    positions = zeros(Int, year.n)
    positions[active] = 1:m
    cumulative = zeros(Float64, p)
    ranks = zeros(Float64, m)
    order = year.expenditure_order
    first = 1
    while first <= year.n
        last = first
        while last < year.n && year.expenditure[order[last + 1]] == year.expenditure[order[first]]
            last += 1
        end
        for sorted_position in first:last
            row = positions[order[sorted_position]]
            row == 0 && continue
            for column in 1:p
                cumulative[column] += sqrt_counts[row] * Q[row, column]
            end
        end
        for sorted_position in first:last
            row = positions[order[sorted_position]]
            row == 0 && continue
            ranks[row] = dot(view(Q, row, :), cumulative) / sqrt_counts[row]
        end
        first = last + 1
    end
    all(isfinite, ranks) || throw(ArgumentError("frequency conditional ranks are nonfinite"))
    return (; ranks, active, sqrt_counts, fit)
end

"""Exact QR evaluation adjoint; no one-hot regression response is constructed.

For pivoted A[:,p]=QR, d' LS(A,Y) = (Q solve(R',d[p]))'Y.
The returned observation weights include sqrt(frequency), so accumulating
them by patch equals weighted least-squares prediction for every indicator.
"""
function _qr_evaluation_weights(fit, integrated::AbstractVector,
                               sqrt_counts::AbstractVector)
    p = fit.rank
    length(integrated) == p || throw(DimensionMismatch("integrated basis"))
    dscaled = integrated ./ fit.scales
    v = LowerTriangular(transpose(fit.factor.R)) \ dscaled[fit.factor.p]
    weights = zeros(Float64, length(sqrt_counts))
    weights[1:p] .= v
    lmul!(fit.factor.Q, weights)
    weights .*= sqrt_counts
    all(isfinite, weights) || throw(ArgumentError("evaluation weights are nonfinite"))
    return weights
end

"""Fast algebraically exact point/bootstrap estimator in unchanged patch coordinates.

Each bootstrap draw is represented by integer frequencies; both least-squares
objectives, inclusive ties, fixed original-n smoothing, and frozen evaluation
are unchanged. The default `probability_policy=:source_clip` retains the
source's coordinatewise clipping without renormalization. The explicit
`:unclipped_affine` policy instead returns the finite unrestricted regression
estimates: they add to one within each budget, but may be negative or exceed
one. It does not project, clip, or renormalize those estimates.
Weighted QR can differ from the
duplicated-row reference in floating-point roundoff. Consequently diagnostic
integer counters at exactly zero/one may differ when a value is within machine
roundoff of that boundary; raw sums and probability coordinates remain audited.
The reported QR diagonal ratios are each fit's actual numerical diagnostics,
not invariants of the weighted objective: unit-column-norm pivot ties can
change pivot order and those ratios under row compression.
"""
function estimate_prepared_probabilities(prepared::PreparedKSEstimator;
        indices=nothing, counts=nothing, probability_policy::Symbol=:source_clip)
    probability_policy in (:source_clip, :unclipped_affine) ||
        throw(ArgumentError("probability_policy must be :source_clip or :unclipped_affine"))
    frequencies = _frequency_counts(prepared; indices, counts)
    pi = zeros(Float64, prepared.coordinates)
    by_year = Dict{String, Any}[]
    for (budget, year) in enumerate(prepared.years)
        frequency = frequencies[budget]
        rank_fit = _frequency_conditional_ranks(year, frequency)
        active, sqrt_counts = rank_fit.active, rank_fit.sqrt_counts
        ranks = smooth_clamp_ranks(rank_fit.ranks, year.n)
        basis = _demand_basis(view(year.log_expenditure, active), ranks, prepared.degree)
        fit = _frequency_qr(basis, sqrt_counts, year.n;
                            label="frequency demand basis for year $(year.year)")
        integrated = _integrated_demand_basis(prepared.evaluation_log_expenditure[budget], prepared.degree)
        weights = _qr_evaluation_weights(fit, integrated, sqrt_counts)
        raw_pi = zeros(Float64, length(year.patch_rows))
        patch_counts = zeros(Float64, length(year.patch_rows))
        below_zero, above_one = 0, 0
        for (row, original_row) in enumerate(active)
            patch = year.patch_positions[original_row]
            raw_pi[patch] += weights[row]
            patch_counts[patch] += frequency[original_row]
            below_zero += rank_fit.ranks[row] < 0 ? frequency[original_row] : 0
            above_one += rank_fit.ranks[row] > 1 ? frequency[original_row] : 0
        end
        all(isfinite, raw_pi) || throw(ArgumentError("estimated probabilities are nonfinite"))
        clipped_pi = clamp.(raw_pi, 0, 1)
        adding_up_error = abs(sum(raw_pi) - 1)
        if probability_policy == :unclipped_affine && adding_up_error > 1e-7
            throw(ArgumentError("unrestricted estimates fail budget adding-up beyond numerical tolerance"))
        end
        selected_pi = probability_policy == :source_clip ? clipped_pi : raw_pi
        pi[year.patch_rows] = selected_pi
        push!(by_year, Dict("year" => year.year, "n" => year.n,
            "evaluation_log_expenditure" => prepared.evaluation_log_expenditure[budget],
            "instrument_basis_rank" => rank_fit.fit.rank,
            "instrument_qr_diagonal_ratio" => rank_fit.fit.qr_diagonal_ratio,
            "demand_basis_rank" => fit.rank, "demand_qr_diagonal_ratio" => fit.qr_diagonal_ratio,
            "raw_probability_sum" => sum(raw_pi), "clipped_probability_sum" => sum(clipped_pi),
            "selected_probability_sum" => sum(selected_pi),
            "raw_probability_adding_up_error" => adding_up_error,
            "minimum_raw_probability" => minimum(raw_pi),
            "maximum_raw_probability" => maximum(raw_pi),
            "coordinates_clipped_low" => count(<(0), raw_pi),
            "coordinates_clipped_high" => count(>(1), raw_pi),
            "rank_below_zero_before_smoothing" => below_zero,
            "rank_above_one_before_smoothing" => above_one, "patch_counts" => patch_counts,
            "unique_selected_rows" => length(active),
            "raw_probability_coordinates_within_roundoff_boundary" =>
                count(x -> min(abs(x), abs(x - 1)) <= 1e-12, raw_pi),
            "rank_frequency_within_roundoff_boundary" => sum(
                frequency[active[row]] for row in eachindex(active)
                if min(abs(rank_fit.ranks[row]), abs(rank_fit.ranks[row] - 1)) <= 1e-12)))
    end
    diagnostics = Dict{String, Any}(
        "source_profile" => KS_ESTIMATION_PROFILE, "degree" => prepared.degree,
        "evaluation_log_expenditure" => copy(prepared.evaluation_log_expenditure),
        "resampled" => indices !== nothing || counts !== nothing,
        "probability_policy" => String(probability_policy),
        "probability_clipping_applied" => probability_policy == :source_clip,
        "unclipped_affine_adding_up_tolerance" => 1e-7,
        "probability_renormalization" => false, "strict_open_patch_classification" => true,
        "rank_endpoint_minus_bandwidth_maps_to_zero" => true,
        "linear_solver" => "column-scaled full-rank pivoted QR", "by_year" => by_year,
        "implementation" => "exact integer-frequency weighted QR; evaluation adjoint",
        "roundoff_boundary_diagnostics_tolerance" => 1e-12,
        "qr_diagonal_ratio_comparability" => "pivot dependent; not required to equal duplicated-row ratio",
        "automatic_differentiation" => false)
    return (; pi, diagnostics)
end

"""Draw original-n bootstrap frequencies without constructing copied household rows."""
function resample_counts(rng::AbstractRNG, prepared::PreparedKSEstimator)
    return [begin
        counts = zeros(Int, year.n)
        for _ in 1:year.n
            counts[rand(rng, 1:year.n)] += 1
        end
        counts
    end for year in prepared.years]
end

"""Independent per-repetition RNGs and fitting buffers; no shared solver or RNG.

Returns a coordinates×B matrix suitable for the unchanged inference routines.
Explicit `indices`/`counts` collections are used verbatim; otherwise each draw
uses a reproducible seed derived serially from `seed`, independent of thread
scheduling. Read-only preparation is shared; each iteration writes only its
own output column and diagnostics slot. No BLAS/global RNG settings are changed.
"""
function bootstrap_prepared_probabilities(prepared::PreparedKSEstimator;
        repetitions::Integer=1000, seed::Integer=1, parallel::Bool=true,
        indices=nothing, counts=nothing, probability_policy::Symbol=:source_clip)
    repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
    probability_policy in (:source_clip, :unclipped_affine) ||
        throw(ArgumentError("probability_policy must be :source_clip or :unclipped_affine"))
    indices === nothing || counts === nothing ||
        throw(ArgumentError("supply either indices or counts, not both"))
    explicit = indices === nothing ? counts : indices
    explicit === nothing || length(explicit) == repetitions ||
        throw(ArgumentError("explicit draw count must equal repetitions"))
    master = MersenneTwister(seed)
    seeds = rand(master, UInt64, repetitions)
    pi = Matrix{Float64}(undef, prepared.coordinates, repetitions)
    diagnostics = Vector{Dict{String, Any}}(undef, repetitions)
    function fit_draw(b)
        result = if indices !== nothing
            estimate_prepared_probabilities(prepared; indices=indices[b], probability_policy)
        elseif counts !== nothing
            estimate_prepared_probabilities(prepared; counts=counts[b], probability_policy)
        else
            rng = MersenneTwister(seeds[b])
            estimate_prepared_probabilities(prepared; counts=resample_counts(rng, prepared), probability_policy)
        end
        pi[:, b] = result.pi
        diagnostics[b] = result.diagnostics
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        Threads.@threads :dynamic for b in 1:repetitions
            fit_draw(b)
        end
    else
        for b in 1:repetitions
            fit_draw(b)
        end
    end
    return (; pi, diagnostics, seeds)
end

"""Benchmark the same explicit draws after warming both paths.

Reports allocations and time separately for preparation and fitting. Speedups
both including and excluding preparation are shown; no household records or
draw-index vectors are returned. This is a local measurement, not a general
performance claim, and does not time QP inference.
"""
function benchmark_prepared_estimator(annual::Vector{KSAnnual}, X::AbstractMatrix, draws;
        degree=3, evaluation_log_expenditure=nothing)
    !isempty(draws) || throw(ArgumentError("at least one precomputed draw is required"))
    prepared = prepare_estimator(annual, X; degree, evaluation_log_expenditure)
    estimate_probabilities(annual, X; degree, indices=draws[1],
        evaluation_log_expenditure=prepared.evaluation_log_expenditure)
    estimate_prepared_probabilities(prepared; indices=draws[1])
    preparation = @timed prepare_estimator(annual, X; degree, evaluation_log_expenditure)
    reference = @timed [estimate_probabilities(annual, X; degree, indices=draw,
        evaluation_log_expenditure=prepared.evaluation_log_expenditure).pi for draw in draws]
    optimized = @timed [estimate_prepared_probabilities(prepared; indices=draw).pi for draw in draws]
    error = maximum(maximum(abs.(reference.value[b] .- optimized.value[b])) for b in eachindex(draws))
    return Dict("repetitions" => length(draws), "degree" => degree,
        "preparation_seconds" => preparation.time, "preparation_bytes" => preparation.bytes,
        "reference_fit_seconds" => reference.time, "reference_fit_bytes" => reference.bytes,
        "prepared_fit_seconds" => optimized.time, "prepared_fit_bytes" => optimized.bytes,
        "speedup_excluding_preparation" => reference.time / optimized.time,
        "speedup_including_preparation" => reference.time / (optimized.time + preparation.time),
        "allocation_ratio_excluding_preparation" => reference.bytes / optimized.bytes,
        "maximum_probability_difference" => error,
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "scope" => "identical explicit draws; warm fits; estimator only; no QP timing")
end
