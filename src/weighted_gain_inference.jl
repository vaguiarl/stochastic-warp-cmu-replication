function _weighted_gain_rate(rate)
    r = Float64(rate)
    isfinite(r) && r>0 && isfinite(r^2) && r^2>0 || throw(ArgumentError(
        "rate must be positive finite with a finite, representably positive square"))
    return r
end

function _weighted_gain_epsilons(epsilons,rate)
    return epsilons===nothing ? [0.5,1.0,2.0] ./ sqrt(rate) : epsilons
end

function _weighted_gain_absolute_tolerance(tolerance)
    isfinite(tolerance) && tolerance>=0 || throw(ArgumentError(
        "absolute projection tolerance must be nonnegative finite"))
    return Float64(tolerance)
end

function _fixed_gain_metric(metric_covariance,d)
    d>0 || throw(ArgumentError("at least one contrast coordinate is required"))
    S = metric_covariance isa UniformScaling ? Matrix{Float64}(metric_covariance,d,d) :
        metric_covariance
    S isa AbstractMatrix || throw(ArgumentError("metric covariance must be a matrix or UniformScaling"))
    return _information_covariance(S,d)
end

"""
    numerical_fixed_weight_test(zhat,zstars; rate,rum_projector,cmu_projector,
        metric_covariance=I,epsilons=nothing,alpha=.05,include_draws=true)

Experimental second-order numerical-delta bootstrap of the gain in minimum
distance under one prespecified, fixed positive-definite weight W0. The argument
`metric_covariance` is W0^-1; it is a chosen metric, NOT an estimated sampling
covariance. Its default is identity. `zstars` contains unscaled joint estimator
bootstrap columns, preserving their full dependence across all coordinates.
Their sampling covariance may be singular: it is never inverted or rank-cut.

The null is equal population minimum W0-weighted distance to the smaller and
larger nested models, including possible common misspecification. This is not
an efficient QLR or an automatic application of the KS specification theorem.
Bootstrap consistency, the estimator rate/bias and second-order assumptions
remain obligations of the application; no uniform validity is claimed here.

The shared numerical-information engine is used with constant metric matrices.
Thus its joint SPD perturbation has identically zero metric increment. Projection
certificates, signed numerical derivatives, MC+1 tails and failure behavior are
unchanged. The default epsilon grid is [.5,1,2]/sqrt(rate), not chosen from results.
"""
function numerical_fixed_weight_test(zhat::AbstractVector,zstars::AbstractMatrix;
        rate,rum_projector,cmu_projector,metric_covariance=I,epsilons=nothing,
        alpha=.05,absolute_tolerance=1e-10,include_draws=true)
    r = _weighted_gain_rate(rate)
    tolerance = _weighted_gain_absolute_tolerance(absolute_tolerance)
    d = length(zhat)
    S = _fixed_gain_metric(metric_covariance,d)
    size(zstars,1)==d || throw(DimensionMismatch("bootstrap contrast coordinates"))
    draws = size(zstars,2)
    draws>=2 || throw(ArgumentError("at least two joint bootstrap draws required"))
    fixed_metrics = repeat(S;outer=(1,1,draws))
    result = numerical_information_test(zhat,S,zstars,fixed_metrics;
        rate=r,rum_projector,cmu_projector,
        epsilons=_weighted_gain_epsilons(epsilons,r),alpha,
        absolute_tolerance=tolerance,include_draws)
    result["method"] = "fixed_weight_distance_gain"
    result["metric"] = Dict{String,Any}(
        "metric_covariance"=>S,"weighting_matrix"=>S\Matrix{Float64}(I,d,d),
        "metric_fixed_within_test"=>true,"metric_prespecification_required"=>true,
        "metric_prespecification_verified_by_routine"=>false,
        "sampling_covariance_inverse_used"=>false,
        "sampling_covariance_positive_definite_required"=>false,
        "coordinates_retained"=>d,"nullspace_coordinates_deleted"=>false,
        "rank_threshold_applied"=>false,"ridge_applied"=>false)
    protocol = result["protocol"]
    protocol["null"] = "equal population minimum distance under the prespecified fixed W0 metric"
    protocol["alternative"] = "CMU/WARP has strictly smaller population W0-weighted distance"
    protocol["covariance_chart"] = "metric is fixed; no sampling-covariance perturbation or inversion"
    protocol["bootstrap"] = "joint unscaled probability/contrast estimator columns only"
    protocol["degenerate_estimator_gaussian_limit_permitted"] = true
    protocol["degenerate_limit_validity_verified_by_routine"] = false
    protocol["efficient_qlr"] = false
    protocol["ks_specification_uniform_validity_transferred"] = false
    protocol["uniform_validity_claimed"] = false
    protocol["sampling_zeros_established_as_population_zeros"] = false
    protocol["all_contrast_coordinates_retained"] = true
    protocol["normalization_and_full_probability_rows"] = "responsibility of the supplied model projectors"
    return result
end

function _regularized_gain_covariance(S::AbstractMatrix,d,ridge,relative_tolerance,label)
    size(S)==(d,d) || throw(DimensionMismatch("$label must be $d by $d"))
    raw = _information_symmetric(S,label)
    eigenvalues = eigvals(Symmetric(raw))
    # Truly relative to the covariance scale; do not give a small matrix a
    # unit-sized absolute PSD allowance. The zero matrix has tolerance zero.
    scale = maximum(abs,eigenvalues)
    tolerance = relative_tolerance*scale
    minimum(eigenvalues)>=-tolerance || throw(ArgumentError(
        "$label is not positive semidefinite within the declared roundoff tolerance"))
    # No eigenvalues are changed. A tolerated tiny negative eigenvalue remains
    # in raw, and every diagonal receives precisely the declared absolute ridge.
    regularized = raw+ridge*I
    checked = _information_covariance(regularized,d)
    return checked,(minimum_eigenvalue=minimum(eigenvalues),
        maximum_eigenvalue=maximum(eigenvalues),psd_roundoff_tolerance=tolerance,
        negative_eigenvalues_within_roundoff=count(<(0),eigenvalues),
        raw_symmetry_error=maximum(abs,Matrix{Float64}(S)-Matrix{Float64}(S)'))
end

"""
    regularized_information_test(zhat,Sigmahat,zstars,Sigmastars; ridge,rate,
        rum_projector,cmu_projector,epsilons=nothing,alpha=.05,
        psd_relative_tolerance=1e-12,include_draws=true)

Prespecified regularized full-covariance sensitivity, NOT the fixed-weight
primary test. Adds the same positive ABSOLUTE ridge to Sigmahat and EVERY joint
bootstrap covariance before calling the existing log-SPD numerical-delta test.
Metric-estimation fluctuations are therefore retained rather than frozen.

Raw covariance matrices must be PSD, allowing only the explicitly reported
roundoff-scale tolerance. No eigenvalue flooring, pseudoinverse, rank selection
or failed-draw replacement is performed. The ridge can make a singular raw
covariance usable but does not establish positive population variance in its
nullspace, validate the bootstrap, or eliminate finite-inner-MC covariance error.

Changing the ridge changes the distance metric and may change the equal-fit
null when both models are misspecified. This wrapper is not an efficient QLR,
nor a theorem of uniform validity. The caller must choose ridge before outcomes.
"""
function regularized_information_test(zhat::AbstractVector,Sigmahat::AbstractMatrix,
        zstars::AbstractMatrix,Sigmastars::AbstractArray;
        ridge,rate,rum_projector,cmu_projector,epsilons=nothing,alpha=.05,
        absolute_tolerance=1e-10,include_draws=true,psd_relative_tolerance=1e-12)
    r = _weighted_gain_rate(rate)
    tolerance = _weighted_gain_absolute_tolerance(absolute_tolerance)
    rho = Float64(ridge)
    isfinite(rho) && rho>0 || throw(ArgumentError("absolute ridge must be positive finite"))
    psd_tolerance = Float64(psd_relative_tolerance)
    isfinite(psd_tolerance) && psd_tolerance>=0 || throw(ArgumentError(
        "PSD roundoff tolerance must be nonnegative finite"))
    d = length(zhat)
    d>0 || throw(ArgumentError("at least one contrast coordinate is required"))
    size(zstars,1)==d || throw(DimensionMismatch("bootstrap contrast coordinates"))
    draws = size(zstars,2)
    draws>=2 || throw(ArgumentError("at least two joint bootstrap draws required"))
    size(Sigmastars)==(d,d,draws) || throw(DimensionMismatch("joint covariance draws"))
    base,base_diagnostics = _regularized_gain_covariance(Sigmahat,d,rho,psd_tolerance,"base raw covariance")
    stars = Array{Float64}(undef,d,d,draws)
    diagnostics = Vector{Any}(undef,draws)
    for b in 1:draws
        stars[:,:,b],diagnostics[b] = _regularized_gain_covariance(view(Sigmastars,:,:,b),d,
            rho,psd_tolerance,"raw bootstrap covariance $b")
    end
    result = numerical_information_test(zhat,base,zstars,stars;
        rate=r,rum_projector,cmu_projector,
        epsilons=_weighted_gain_epsilons(epsilons,r),alpha,
        absolute_tolerance=tolerance,include_draws)
    result["method"] = "regularized_joint_full_covariance_sensitivity"
    result["regularization"] = Dict{String,Any}(
        "absolute_ridge"=>rho,"ridge_fixed_across_base_and_draws"=>true,
        "ridge_prespecification_required"=>true,"ridge_prespecification_verified_by_routine"=>false,
        "relative_psd_roundoff_tolerance"=>psd_tolerance,
        "regularized_base_metric_covariance"=>base,
        "raw_base_minimum_eigenvalue"=>base_diagnostics.minimum_eigenvalue,
        "raw_base_maximum_eigenvalue"=>base_diagnostics.maximum_eigenvalue,
        "base_psd_roundoff_tolerance"=>base_diagnostics.psd_roundoff_tolerance,
        "base_negative_eigenvalues_within_roundoff"=>base_diagnostics.negative_eigenvalues_within_roundoff,
        "minimum_raw_bootstrap_eigenvalue"=>minimum(x.minimum_eigenvalue for x in diagnostics),
        "maximum_raw_bootstrap_psd_roundoff_tolerance"=>maximum(x.psd_roundoff_tolerance for x in diagnostics),
        "bootstrap_negative_eigenvalues_within_roundoff"=>sum(x.negative_eigenvalues_within_roundoff for x in diagnostics),
        "maximum_raw_covariance_symmetry_error"=>max(base_diagnostics.raw_symmetry_error,
            maximum(x.raw_symmetry_error for x in diagnostics)),
        "coordinates_retained"=>d,"nullspace_coordinates_deleted"=>false,
        "rank_threshold_applied"=>false,"eigenvalue_flooring_applied"=>false,
        "pseudoinverse_applied"=>false)
    protocol = result["protocol"]
    protocol["null"] = "equal population minimum distance under the full covariance plus prespecified absolute ridge metric"
    protocol["alternative"] = "CMU/WARP has strictly smaller population distance under that ridge metric"
    protocol["covariance_chart"] = "joint symmetric-log perturbation AFTER adding the same absolute ridge to base and every covariance draw"
    protocol["metric_covariance_uncertainty_included"] = true
    protocol["sensitivity_not_primary_test"] = true
    protocol["null_may_change_under_common_misspecification"] = true
    protocol["efficient_qlr"] = false
    protocol["uniform_validity_claimed"] = false
    protocol["finite_covariance_mc_error_eliminated"] = false
    protocol["joint_estimator_covariance_rate_and_bootstrap_require_justification"] = true
    protocol["sampling_zeros_established_as_population_zeros"] = false
    protocol["all_contrast_coordinates_retained"] = true
    return result
end
