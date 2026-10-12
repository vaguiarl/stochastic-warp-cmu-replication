"""Annual inputs from the author computational-replication source (not raw FES).

Rows are households, columns of shares are goods. `normalized_prices` are
read verbatim from the source price files: this loader does not renormalize
prices, shares, income, or the instrument.
"""
struct KSAnnual
    year::Int
    shares::Matrix{Float64}
    expenditure::Vector{Float64}
    instrument::Vector{Float64}
    normalized_prices::Vector{Float64}
end

const KS_ESTIMATION_PROFILE = "author-CG-Endogenous.cpp-919ba915623273a208e25666a47120745b7510f9"

function _validate_annual(a::KSAnnual; share_sum_tolerance=1e-5)
    n, goods = size(a.shares)
    n > 0 && goods > 0 || throw(ArgumentError("annual inputs must be nonempty"))
    length(a.expenditure) == n && length(a.instrument) == n ||
        throw(ArgumentError("annual expenditure/instrument lengths must equal share rows"))
    length(a.normalized_prices) == goods ||
        throw(ArgumentError("annual price count must equal the number of goods"))
    all(isfinite, a.shares) && all(isfinite, a.expenditure) &&
        all(isfinite, a.instrument) && all(isfinite, a.normalized_prices) ||
        throw(ArgumentError("annual inputs must be finite"))
    all(>=(0), a.shares) || throw(ArgumentError("shares must be nonnegative"))
    all(>(0), a.expenditure) || throw(ArgumentError("expenditure must be positive"))
    all(>(0), a.normalized_prices) || throw(ArgumentError("normalized prices must be positive"))
    maximum(abs.(vec(sum(a.shares; dims=2)) .- 1)) <= share_sum_tolerance ||
        throw(ArgumentError("shares do not sum to one within source rounding tolerance"))
    return n, goods
end

"""Load the source's count-prefixed share/E/Z CSVs and price vectors.

The input format follows `RUM_Computation_Source.cpp:285–319`. The field named
`income` there is exposed here as `expenditure`, since its logarithm is the
expenditure argument used in the demand estimator. No household rows are
printed or exported. These are computational-replication inputs, not proof of
reproducing the original Kitamura–Stoye estimation sample.
"""
function load_cg_annual(repo_path; goods::Integer=3, years=1986:1992)
    goods > 0 || throw(ArgumentError("goods must be positive"))
    input = joinpath(repo_path, "C++ Source", "input")
    annual = KSAnnual[]
    for year in years
        year isa Integer || throw(ArgumentError("years must contain integers"))
        suffix = lpad(string(mod(year, 100)), 2, '0')
        qpath = joinpath(input, "$(goods)goods-q$(suffix).csv")
        ppath = joinpath(input, "$(goods)goods-p$(suffix).csv")
        isfile(qpath) && isfile(ppath) ||
            throw(ArgumentError("missing author input files for year $year"))
        first_line = open(readline, qpath)
        n = tryparse(Int, strip(first_line))
        n !== nothing && n > 0 || throw(ArgumentError("invalid source row count for year $year"))
        table = CSV.File(qpath; header=false, skipto=2, types=Float64, strict=true)
        data = Matrix{Float64}(DataFrame(table))
        size(data) == (n, goods + 2) ||
            throw(ArgumentError("count/column mismatch in source input for year $year"))
        prices = parse.(Float64, split(strip(read(ppath, String)), r"[\s,]+"))
        length(prices) == goods || throw(ArgumentError("incorrect source price count for year $year"))
        a = KSAnnual(Int(year), data[:, 1:goods], data[:, goods + 1],
                     data[:, goods + 2], prices)
        _validate_annual(a)
        push!(annual, a)
    end
    isempty(annual) && throw(ArgumentError("at least one year is required"))
    length(unique(a.year for a in annual)) == length(annual) ||
        throw(ArgumentError("annual years must be distinct"))
    return annual
end

function _checked_degree(degree)
    degree isa Integer && degree >= 0 || throw(ArgumentError("degree must be a nonnegative integer"))
    return Int(degree)
end

"""Raw-Z polynomial basis; no log-income transform is applied to Z."""
function _instrument_basis(instrument::AbstractVector, degree::Integer)
    degree = _checked_degree(degree)
    all(isfinite, instrument) || throw(ArgumentError("the instrument must be finite"))
    basis = ones(Float64, length(instrument), degree + 1)
    for power in 1:degree
        basis[:, power + 1] .= basis[:, power] .* instrument
    end
    all(isfinite, basis) || throw(ArgumentError("instrument powers overflowed"))
    return basis
end

"""Column-scaled pivoted QR, rejecting deficient rather than silently fitting."""
function _scaled_qr(basis::AbstractMatrix; label="basis")
    n, p = size(basis)
    n >= p && p > 0 || throw(ArgumentError("$label has too few observations"))
    all(isfinite, basis) || throw(ArgumentError("$label must be finite"))
    scales = [norm(view(basis, :, column)) for column in 1:p]
    all(x -> isfinite(x) && x > 0, scales) ||
        throw(ArgumentError("$label has a zero or invalid column"))
    factor = qr(Matrix{Float64}(basis) ./ transpose(scales), ColumnNorm())
    diagonals = abs.(diag(factor.R))
    tolerance = max(n, p) * eps(Float64) * maximum(diagonals)
    count(>(tolerance), diagonals) == p ||
        throw(ArgumentError("$label is numerically rank deficient"))
    return (; factor, scales, rank=p,
            qr_diagonal_ratio=maximum(diagonals) / minimum(diagonals))
end

"""Conditional empirical-CDF rank, including all expenditure ties.

For household h the response vector is 1{E_h >= E_k}, as in Endogenous.cpp.
Thin-Q cumulative sums compute the same full-rank LS projection without an
n×n indicator or projection matrix. The returned ranks are not yet clipped.
"""
function conditional_cdf_ranks(expenditure::AbstractVector,
                               instrument::AbstractVector; degree=3)
    length(expenditure) == length(instrument) && !isempty(expenditure) ||
        throw(ArgumentError("expenditure/instrument lengths must agree and be nonempty"))
    all(x -> isfinite(x) && x > 0, expenditure) ||
        throw(ArgumentError("expenditure must be positive and finite"))
    R = _instrument_basis(instrument, _checked_degree(degree))
    fit = _scaled_qr(R; label="instrument basis")
    n, p = size(R)
    Q = fit.factor.Q * Matrix{Float64}(I, n, p) # Thin Q, not Matrix(Q) (n×n).
    order = sortperm(expenditure; alg=MergeSort)
    cumulative = zeros(Float64, p)
    ranks = zeros(Float64, n)
    first = 1
    while first <= n
        last = first
        while last < n && expenditure[order[last + 1]] == expenditure[order[first]]
            last += 1
        end
        for position in first:last
            cumulative .+= view(Q, order[position], :)
        end
        for position in first:last
            row = order[position]
            ranks[row] = dot(view(Q, row, :), cumulative)
        end
        first = last + 1
    end
    all(isfinite, ranks) || throw(ArgumentError("conditional rank estimation was nonfinite"))
    return (; ranks, diagnostics=(rank=fit.rank, qr_diagonal_ratio=fit.qr_diagonal_ratio))
end

"""Source smooth truncation at bandwidth n^(-1/3), including endpoints.

The mathematical lower endpoint -v maps to zero (the C++ source's strict
comparison leaves exactly -v untreated). For n >= 8 the lower/upper smoothing
regions do not overlap; smaller samples are rejected explicitly.
"""
function smooth_clamp_ranks(ranks::AbstractVector, n::Integer=length(ranks))
    n >= 8 || throw(ArgumentError("smooth rank clamp requires n >= 8"))
    all(isfinite, ranks) || throw(ArgumentError("conditional ranks must be finite"))
    v = n^(-1 / 3)
    return [x <= -v ? 0.0 : x <= v ? (x + v)^2 / (4v) :
            x < 1 - v ? Float64(x) : x <= 1 + v ? 1 - (1 - x + v)^2 / (4v) :
            1.0 for x in ranks]
end

function _total_degree_powers(degree::Integer)
    degree = _checked_degree(degree)
    return [(k, l) for k in 0:degree for l in 0:degree if k + l <= degree]
end

function _demand_basis(log_expenditure::AbstractVector, ranks::AbstractVector, degree::Integer)
    length(log_expenditure) == length(ranks) || throw(ArgumentError("basis vector lengths disagree"))
    powers = _total_degree_powers(degree)
    basis = Matrix{Float64}(undef, length(ranks), length(powers))
    for (column, (k, l)) in enumerate(powers)
        basis[:, column] .= log_expenditure .^ k .* ranks .^ l
    end
    all(isfinite, basis) || throw(ArgumentError("demand basis must be finite"))
    return basis
end

"""Integrate rank monomials on [0,1] at a fixed log-expenditure argument."""
function _integrated_demand_basis(evaluation_log_expenditure::Real, degree::Integer)
    isfinite(evaluation_log_expenditure) || throw(ArgumentError("evaluation log expenditure must be finite"))
    return [Float64(evaluation_log_expenditure)^k / (l + 1)
            for (k, l) in _total_degree_powers(degree)]
end

function _classify_patch_rows(shares, own_prices, prices, own_budget, groups, X;
                              boundary_tolerance=1e-10)
    boundary_tolerance >= 0 && isfinite(boundary_tolerance) ||
        throw(ArgumentError("boundary tolerance must be finite and nonnegative"))
    lookup = Dict{Tuple, Int}()
    for row in groups[own_budget]
        key = Tuple(X[row, :])
        haskey(lookup, key) && throw(ArgumentError("duplicate open-patch sign patterns"))
        lookup[key] = row
    end
    rows = zeros(Int, size(shares, 1))
    for h in axes(shares, 1)
        quantity = view(shares, h, :) ./ own_prices
        signs = zeros(Int8, size(prices, 1))
        for budget in axes(prices, 1)
            budget == own_budget && continue # Source explicitly forces own zero.
            gap = dot(view(prices, budget, :), quantity) - 1
            abs(gap) > boundary_tolerance ||
                throw(ArgumentError("an observation is ambiguous on a budget intersection"))
            signs[budget] = gap < 0 ? -1 : 1
        end
        row = get(lookup, Tuple(signs), 0)
        row > 0 || throw(ArgumentError("an observation has an unrecognized open-patch sign pattern"))
        rows[h] = row
    end
    return rows
end

"""Draw source-profile within-year paired bootstrap indices, with replacement."""
function resample_indices(rng::AbstractRNG, annual::Vector{KSAnnual})
    isempty(annual) && throw(ArgumentError("annual inputs must be nonempty"))
    return [rand(rng, 1:_validate_annual(a)[1], size(a.shares, 1)) for a in annual]
end

"""Port the author control-function estimator into the supplied full patch coordinates.

`X[row,budget]` has exactly one own-budget zero and other signs -1/+1.
`indices` optionally supplies source-profile within-year bootstrap draws of
unchanged sample size. Default evaluation is the original annual median log E,
computed BEFORE any resampling; explicit evaluation arguments stay fixed too.

Raw Z is polynomially regressed, ranks are smoothly clipped, the demand basis
has total degree `degree` in (log E, rank), and integration is uniform in rank.
Only coordinate clipping to [0,1] is applied: block probabilities are NOT
renormalized. Diagnostics are aggregate, without household observations.
"""
function estimate_probabilities(annual::Vector{KSAnnual}, X::AbstractMatrix;
        degree=3, indices=nothing, evaluation_log_expenditure=nothing)
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
    draws = indices === nothing ? [collect(1:first(size)) for size in sizes] : indices
    length(draws) == length(annual) || throw(ArgumentError("one resample-index vector per year is required"))
    prices = reduce(vcat, transpose(a.normalized_prices) for a in annual)
    pi = zeros(Float64, size(X, 1))
    by_year = Dict{String, Any}[]
    for (budget, a) in enumerate(annual)
        ids = collect(draws[budget])
        n = sizes[budget][1]
        length(ids) == n && all(x -> x isa Integer && 1 <= x <= n, ids) ||
            throw(ArgumentError("bootstrap indices must retain annual sample size and valid integer rows"))
        shares, expenditure, instrument = a.shares[ids, :], a.expenditure[ids], a.instrument[ids]
        rank_fit = conditional_cdf_ranks(expenditure, instrument; degree)
        ranks = smooth_clamp_ranks(rank_fit.ranks, n)
        rows = _classify_patch_rows(shares, a.normalized_prices, prices, budget, groups, X)
        basis = _demand_basis(log.(expenditure), ranks, degree)
        fit = _scaled_qr(basis; label="demand basis for year $(a.year)")
        response = zeros(Float64, n, length(groups[budget]))
        positions = Dict(row => column for (column, row) in enumerate(groups[budget]))
        for h in 1:n
            response[h, positions[rows[h]]] = 1
        end
        coefficients_scaled = fit.factor \ response
        integrated = _integrated_demand_basis(evaluation[budget], degree) ./ fit.scales
        raw_pi = vec(transpose(integrated) * coefficients_scaled)
        all(isfinite, raw_pi) || throw(ArgumentError("estimated probabilities are nonfinite"))
        clipped_pi = clamp.(raw_pi, 0, 1)
        pi[groups[budget]] = clipped_pi
        push!(by_year, Dict("year" => a.year, "n" => n,
            "evaluation_log_expenditure" => evaluation[budget],
            "instrument_basis_rank" => rank_fit.diagnostics.rank,
            "instrument_qr_diagonal_ratio" => rank_fit.diagnostics.qr_diagonal_ratio,
            "demand_basis_rank" => fit.rank, "demand_qr_diagonal_ratio" => fit.qr_diagonal_ratio,
            "raw_probability_sum" => sum(raw_pi), "clipped_probability_sum" => sum(clipped_pi),
            "coordinates_clipped_low" => count(<(0), raw_pi),
            "coordinates_clipped_high" => count(>(1), raw_pi),
            "rank_below_zero_before_smoothing" => count(<(0), rank_fit.ranks),
            "rank_above_one_before_smoothing" => count(>(1), rank_fit.ranks),
            "patch_counts" => vec(sum(response; dims=1))))
    end
    diagnostics = Dict{String, Any}(
        "source_profile" => KS_ESTIMATION_PROFILE, "degree" => degree,
        "evaluation_log_expenditure" => evaluation, "resampled" => indices !== nothing,
        "probability_renormalization" => false, "strict_open_patch_classification" => true,
        "rank_endpoint_minus_bandwidth_maps_to_zero" => true,
        "linear_solver" => "column-scaled full-rank pivoted QR", "by_year" => by_year)
    return (; pi, diagnostics)
end
