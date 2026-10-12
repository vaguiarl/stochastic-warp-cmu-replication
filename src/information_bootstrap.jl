"""Sample an original-n draw from each supplied empirical frequency distribution.

Integer inverse-CDF draws implement probabilities count/n exactly. Reusing an
RNG seed reuses the same uniform integer ranks, not the same selected rows:
the selected rows change with the supplied outer empirical distribution.
Inputs are original-row frequency vectors, already validated by the caller.
No source household matrix or draw indices are copied or returned.
"""
function _information_conditional_counts(prepared::PreparedKSEstimator,
        parent_frequencies::Vector{Vector{Int}}, rng::AbstractRNG)
    length(parent_frequencies) == length(prepared.years) ||
        throw(DimensionMismatch("one empirical frequency vector per year is required"))
    return [begin
        length(frequency) == year.n && all(>=(0), frequency) && sum(frequency) == year.n ||
            throw(ArgumentError("empirical frequencies must retain original annual n"))
        cumulative = cumsum(frequency)
        counts = zeros(Int, year.n)
        for _ in 1:year.n
            rank = rand(rng, 1:year.n)
            counts[searchsortedfirst(cumulative, rank)] += 1
        end
        counts
    end for (year, frequency) in zip(prepared.years, parent_frequencies)]
end

"""Finite-simulation covariance diagnostics; no inversion or rank repair.

An explicit relative eigenvalue tolerance identifies numerical rank. A
singular estimated covariance is returned and reported, never silently
regularized or replaced by a pseudoinverse. The downstream test must choose
its admissibility/rank policy separately.
"""
function _information_covariance_diagnostics(C::AbstractMatrix;
        relative_tolerance::Real=1e-10)
    size(C, 1) == size(C, 2) && size(C, 1) > 0 ||
        throw(DimensionMismatch("nonempty square covariance required"))
    all(isfinite, C) || throw(ArgumentError("sampling covariance is nonfinite"))
    isfinite(relative_tolerance) && relative_tolerance > 0 ||
        throw(ArgumentError("positive finite covariance rank tolerance required"))
    symmetry_error = maximum(abs.(C .- transpose(C)))
    eigenvalues = eigvals(Symmetric((Matrix{Float64}(C) + transpose(C)) / 2))
    scale = maximum(abs, eigenvalues)
    tolerance = Float64(relative_tolerance) * scale
    numerical_rank = count(>(tolerance), eigenvalues)
    spd = minimum(eigenvalues) > tolerance
    return Dict{String, Any}("dimension" => size(C, 1),
        "numerical_rank" => numerical_rank, "positive_definite_at_tolerance" => spd,
        "minimum_eigenvalue" => minimum(eigenvalues),
        "maximum_eigenvalue" => maximum(eigenvalues),
        "relative_rank_tolerance" => Float64(relative_tolerance),
        "absolute_rank_tolerance" => tolerance,
        "negative_eigenvalues_beyond_tolerance" => count(<(-tolerance), eigenvalues),
        "condition_number_if_positive_definite" => spd ? maximum(eigenvalues) / minimum(eigenvalues) : nothing,
        "maximum_symmetry_error" => symmetry_error,
        "regularization_applied" => false, "pseudoinverse_applied" => false)
end

function _information_fit_summary(result)
    years = result.diagnostics["by_year"]
    return Dict{String, Any}(
        "maximum_raw_adding_up_error" => maximum(y["raw_probability_adding_up_error"] for y in years),
        "maximum_instrument_qr_diagonal_ratio" => maximum(y["instrument_qr_diagonal_ratio"] for y in years),
        "maximum_demand_qr_diagonal_ratio" => maximum(y["demand_qr_diagonal_ratio"] for y in years),
        "minimum_instrument_basis_rank" => minimum(y["instrument_basis_rank"] for y in years),
        "minimum_demand_basis_rank" => minimum(y["demand_basis_rank"] for y in years),
        "minimum_unrestricted_coordinate" => minimum(result.pi),
        "maximum_unrestricted_coordinate" => maximum(result.pi),
        "negative_unrestricted_coordinates" => count(<(0), result.pi),
        "unrestricted_coordinates_above_one" => count(>(1), result.pi),
        "minimum_unique_selected_rows" => minimum(y["unique_selected_rows"] for y in years))
end

function _information_summary_of_fits(summaries)
    return Dict{String, Any}(
        "fits" => length(summaries),
        "maximum_raw_adding_up_error" => maximum(s["maximum_raw_adding_up_error"] for s in summaries),
        "maximum_instrument_qr_diagonal_ratio" => maximum(s["maximum_instrument_qr_diagonal_ratio"] for s in summaries),
        "maximum_demand_qr_diagonal_ratio" => maximum(s["maximum_demand_qr_diagonal_ratio"] for s in summaries),
        "minimum_instrument_basis_rank" => minimum(s["minimum_instrument_basis_rank"] for s in summaries),
        "minimum_demand_basis_rank" => minimum(s["minimum_demand_basis_rank"] for s in summaries),
        "minimum_unrestricted_coordinate" => minimum(s["minimum_unrestricted_coordinate"] for s in summaries),
        "maximum_unrestricted_coordinate" => maximum(s["maximum_unrestricted_coordinate"] for s in summaries),
        "negative_unrestricted_coordinates_across_fits" => sum(s["negative_unrestricted_coordinates"] for s in summaries),
        "unrestricted_coordinates_above_one_across_fits" => sum(s["unrestricted_coordinates_above_one"] for s in summaries),
        "minimum_unique_selected_rows" => minimum(s["minimum_unique_selected_rows"] for s in summaries))
end

"""One conditional covariance experiment; both CF stages refit in every draw."""
function _information_covariance_experiment(prepared::PreparedKSEstimator,
        Bcontrast::Matrix{Float64}, frequencies::Vector{Vector{Int}}, inner_seeds,
        context::AbstractString)
    V = length(inner_seeds)
    draws = Matrix{Float64}(undef, size(Bcontrast, 1), V)
    summaries = Vector{Dict{String, Any}}(undef, V)
    for v in 1:V
        try
            counts = _information_conditional_counts(prepared, frequencies, MersenneTwister(inner_seeds[v]))
            fit = estimate_prepared_probabilities(prepared; counts, probability_policy=:unclipped_affine)
            draws[:, v] = Bcontrast * fit.pi
            summaries[v] = _information_fit_summary(fit)
        catch error
            # Abort, preserving the failing experiment/draw number. Never
            # replace a rank-deficient fit, discard a draw, or change its seed.
            throw(ErrorException("$context inner covariance draw $v failed: $(sprint(showerror, error))"))
        end
    end
    all(isfinite, draws) || throw(ArgumentError("$context contrast estimates are nonfinite"))
    centered = draws .- mean(draws; dims=2)
    C = centered * transpose(centered) / (V - 1)
    C = (C + transpose(C)) / 2
    return (; covariance=C, diagnostics=Dict{String, Any}(
        "fit_summary" => _information_summary_of_fits(summaries),
        "covariance" => _information_covariance_diagnostics(C)))
end

"""Joint outer bootstrap of an affine patch estimator and its full covariance.

`Bcontrast` is any fixed finite matrix mapping all patch coordinates to the
reported contrast coordinates; cross-year covariances are retained. The point
and every outer/inner estimate use `:unclipped_affine`, not the legacy clipped
estimator. Prices, patch geometry, degree, smoothing bandwidth convention and
evaluation log expenditures remain fixed in `prepared`. Every original annual
sample size is retained, including when some original-row frequencies vanish.

Outer draws sample the original uniform empirical distribution. Inner draws
sample their outer empirical distribution (counts/n), refitting both the
conditional-CDF and demand regressions. The same independently generated inner
seed list is used for the base covariance and every outer covariance, providing
deterministic common random numbers without changing their conditional laws.
Covariances are centered at the inner mean and divide by V-1. Finite V simulation
noise is reported, not claimed absent; no statistical validity/rate assertion
is supplied by this implementation. `rate` only defines the normalization.

Returns `zhat`, base sampling `C_hat`, `Sigma=rate^2*C_hat`, `z_bootstrap` (d×B),
`Sigma_bootstrap` (d×d×B), aggregate diagnostics, and reproducible seed lists.
Failed fits abort the batch; none are dropped, repaired, or redrawn. Each outer
worker has its own RNG/fitting arrays; no shared solver/global RNG is used.
"""
function joint_information_bootstrap(prepared::PreparedKSEstimator,
        Bcontrast::AbstractMatrix; repetitions::Integer=199,
        covariance_repetitions::Integer=199, seed::Integer=1, parallel::Bool=true,
        rate::Real=sqrt(sum(year.n for year in prepared.years)))
    repetitions > 0 || throw(ArgumentError("outer repetitions must be positive"))
    covariance_repetitions >= 2 || throw(ArgumentError("at least two covariance repetitions required"))
    size(Bcontrast, 2) == prepared.coordinates && size(Bcontrast, 1) > 0 ||
        throw(DimensionMismatch("contrast matrix must have a nonempty row dimension and one column per patch"))
    all(isfinite, Bcontrast) || throw(ArgumentError("contrast matrix must be finite"))
    isfinite(rate) && rate > 0 && isfinite(Float64(rate)^2) ||
        throw(ArgumentError("rate must be positive finite with a finite square"))
    contrast = Matrix{Float64}(Bcontrast)
    master = MersenneTwister(seed)
    outer_seeds = rand(master, UInt64, repetitions)
    inner_seeds = rand(master, UInt64, covariance_repetitions)
    original_frequencies = _frequency_counts(prepared)
    point = estimate_prepared_probabilities(prepared; probability_policy=:unclipped_affine)
    zhat = contrast * point.pi
    all(isfinite, zhat) || throw(ArgumentError("point contrast estimates are nonfinite"))
    base = _information_covariance_experiment(prepared, contrast, original_frequencies, inner_seeds, "base")
    C_hat = base.covariance
    rate_squared = Float64(rate)^2
    Sigma = rate_squared * C_hat
    d = size(contrast, 1)
    z_bootstrap = Matrix{Float64}(undef, d, repetitions)
    Sigma_bootstrap = Array{Float64}(undef, d, d, repetitions)
    outer_diagnostics = Vector{Dict{String, Any}}(undef, repetitions)
    function fit_outer(b)
        try
            frequencies = resample_counts(MersenneTwister(outer_seeds[b]), prepared)
            fit = estimate_prepared_probabilities(prepared; counts=frequencies,
                                                  probability_policy=:unclipped_affine)
            z_bootstrap[:, b] = contrast * fit.pi
            conditional = _information_covariance_experiment(prepared, contrast, frequencies,
                                                              inner_seeds, "outer draw $b")
            Sigma_bootstrap[:, :, b] = rate_squared * conditional.covariance
            outer_diagnostics[b] = Dict{String, Any}(
                "outer_fit" => _information_fit_summary(fit),
                "inner_experiment" => conditional.diagnostics)
        catch error
            throw(ErrorException("information bootstrap outer draw $b failed: $(sprint(showerror, error))"))
        end
        return nothing
    end
    if parallel && Threads.nthreads() > 1
        Threads.@threads :dynamic for b in 1:repetitions
            fit_outer(b)
        end
    else
        for b in 1:repetitions
            fit_outer(b)
        end
    end
    all(isfinite, z_bootstrap) && all(isfinite, Sigma_bootstrap) ||
        throw(ArgumentError("outer joint bootstrap output is nonfinite"))
    diagnostics = Dict{String, Any}(
        "probability_policy" => "unclipped_affine", "probability_renormalization" => false,
        "outer_repetitions" => repetitions, "covariance_repetitions" => covariance_repetitions,
        "finite_covariance_simulation" => true, "covariance_denominator" => covariance_repetitions - 1,
        "covariance_center" => "conditional inner sample mean",
        "rate" => Float64(rate), "rate_squared" => rate_squared,
        "rate_interpretation" => "user-selected normalization; no estimator-rate theorem asserted",
        "original_total_n" => sum(year.n for year in prepared.years),
        "original_n_by_year" => [year.n for year in prepared.years],
        "years" => [year.year for year in prepared.years],
        "degree" => prepared.degree,
        "evaluation_log_expenditure" => copy(prepared.evaluation_log_expenditure),
        "geometry_and_evaluation_fixed" => true, "both_control_function_stages_refit" => true,
        "inner_law" => "multinomial original annual n from outer empirical frequencies/n",
        "outer_law" => "multinomial original annual n from original uniform empirical distribution",
        "common_random_numbers" => "same inner uniform integer ranks for base and every outer covariance",
        "cross_contrast_covariances_retained" => true,
        "draws_discarded" => 0, "replacement_draws" => 0,
        "fit_failure_policy" => "abort entire batch; report original draw context",
        "point_fit" => _information_fit_summary(point),
        "point_fit_by_year" => point.diagnostics["by_year"], "base_experiment" => base.diagnostics,
        "outer_experiments" => outer_diagnostics,
        "parallel_requested" => parallel, "julia_threads" => Threads.nthreads())
    return (; zhat, C_hat, Sigma, z_bootstrap, Sigma_bootstrap, diagnostics,
             seeds=(outer=outer_seeds, inner=inner_seeds))
end
