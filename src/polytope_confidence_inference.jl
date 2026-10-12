"""Read projection bounds without treating floating-point certificates as exact intervals."""
function _confidence_distance_bounds(fit; absolute_tolerance=1e-10)
    q = Float64(fit.squared_distance)
    isfinite(q) && q>=0 || throw(ArgumentError("invalid squared projection distance"))
    e = hasproperty(fit,:objective_error_bound) ? Float64(fit.objective_error_bound) : 0.0
    isfinite(e) && e>=0 || throw(ArgumentError("invalid projection error bound"))
    lo = hasproperty(fit,:objective_lower_bound) ? Float64(fit.objective_lower_bound) : q-e
    hi = hasproperty(fit,:objective_upper_bound) ? Float64(fit.objective_upper_bound) : q+e
    all(isfinite,(lo,hi)) && lo<=hi+absolute_tolerance &&
        lo<=q+absolute_tolerance && q<=hi+absolute_tolerance ||
        throw(ArgumentError("invalid projection certificate interval"))
    return (;lower=max(0.0,lo-absolute_tolerance),upper=max(0.0,hi+absolute_tolerance))
end

"""
    polytope_confidence_bounds(zhat, radius; rum_projector, cmu_projector)

Deterministic Euclidean bounds for two nonempty nested convex models R ⊆ C.
The radius is in unscaled estimator units. Supplied projectors must describe
these sets in the same coordinates. Every solve explicitly uses identity
weight, independent of the estimator's possibly singular sampling covariance.

If ||zhat-z0|| <= radius, distance to each set is 1-Lipschitz. In addition,
at ANY equal-distance population null (including common misspecification),
d_R(zhat)^2-d_C(zhat)^2 <= ||zhat-z0||^2. This stronger common-fit implication
is verified separately in Lean; it does not require a covariance inverse,
smooth active face, identifiable mixture weights, or a second-order bootstrap.

This routine does NOT establish confidence coverage. Numerical primal/dual
bounds are floating-point checks, not outward-rounded interval arithmetic.
"""
function polytope_confidence_bounds(zhat::AbstractVector,radius::Real;
        rum_projector,cmu_projector,absolute_tolerance=1e-10)
    d = length(zhat)
    d>0 && all(isfinite,zhat) || throw(ArgumentError("nonempty finite estimator required"))
    isfinite(radius) && radius>=0 || throw(ArgumentError("finite nonnegative radius required"))
    tolerance = _weighted_gain_absolute_tolerance(absolute_tolerance)
    point = information_distance_gain(zhat,Matrix{Float64}(I,d,d),
        rum_projector,cmu_projector;absolute_tolerance=tolerance)
    r = _confidence_distance_bounds(point.rum;absolute_tolerance=tolerance)
    c = _confidence_distance_bounds(point.cmu;absolute_tolerance=tolerance)
    rlower, rupper = sqrt(r.lower),sqrt(r.upper)
    clower, cupper = sqrt(c.lower),sqrt(c.upper)
    gap_lower = max(0.0,r.lower-c.upper)
    gap_upper = max(0.0,r.upper-c.lower)
    # A conservative lower bound throughout the confidence ball, separately
    # from the stronger exact implication that excludes an equal-fit null.
    ball_gap_lower = max(0.0,max(0.0,rlower-radius)^2-(cupper+radius)^2)
    ball_gap_upper = max(0.0,(rupper+radius)^2-max(0.0,clower-radius)^2)
    return Dict{String,Any}(
        "radius"=>Float64(radius),"point_gap"=>point.gap,"raw_point_gap"=>point.raw_gap,
        "point_gap_lower_numerical"=>gap_lower,"point_gap_upper_numerical"=>gap_upper,
        "rum_squared_distance_lower_numerical"=>r.lower,
        "cmu_squared_distance_lower_numerical"=>c.lower,
        "point_projection"=>Dict("rum"=>_information_projection_summary(point.rum),
            "cmu"=>_information_projection_summary(point.cmu)),
        "distance_intervals_on_ball"=>Dict(
            "rum"=>[max(0.0,rlower-radius),rupper+radius],
            "cmu"=>[max(0.0,clower-radius),cupper+radius]),
        "squared_gap_interval_on_ball"=>[ball_gap_lower,ball_gap_upper],
        "rum_membership_excluded_on_ball"=>r.lower>radius^2,
        "cmu_membership_excluded_on_ball"=>c.lower>radius^2,
        "equal_distance_null_excluded_on_ball"=>gap_lower>radius^2,
        "explicit_positive_gap_lower_bound_on_ball"=>ball_gap_lower>0,
        "metric"=>"fixed identity; Euclidean distance in all supplied coordinates",
        "confidence_coverage_established"=>false,
        "floating_point_certificates_not_interval_arithmetic"=>true)
end

"""
    bootstrap_polytope_confidence_test(zhat,zstars; rate,rum_projector,
        cmu_projector,alpha=.05,include_draws=false)

Conservative JOINT-confidence-region inversion for model membership and
strict improvement of a larger nested model. Calibrates ONE Euclidean norm
ball using centered JOINT estimator bootstrap columns. Dependence across
coordinates is preserved; a covariance matrix is neither estimated nor
inverted. All empirical zeros remain coordinates, not population restrictions.

The calibration is valid conditional on a valid bootstrap confidence ball for
the population target, with negligible estimator bias and correct treatment
of design uncertainty. This is NOT Shi's KL/exponential-tilting model-selection
test, the KS cone-tightening test, or a claim of raw-data efficient inference.
It is a conservative confidence-projection construction motivated by the
general confidence-set inversion literature, with an exact geometric bridge.
Its nominal coverage must not be described as verified by this routine.

The MC+1 upper tail is used with conservative numerical ties; it is a simulation
convention, not a finite-sample distribution-free guarantee for an estimated
bootstrap law. Failure of any projection or bootstrap column aborts the test.
"""
function bootstrap_polytope_confidence_test(zhat::AbstractVector,zstars::AbstractMatrix;
        rate,rum_projector,cmu_projector,alpha=.05,
        absolute_tolerance=1e-10,include_draws=false)
    r = _weighted_gain_rate(rate)
    tolerance = _weighted_gain_absolute_tolerance(absolute_tolerance)
    d = length(zhat)
    d>0 && size(zstars,1)==d || throw(DimensionMismatch("joint bootstrap estimator coordinates"))
    all(isfinite,zhat) && all(isfinite,zstars) || throw(ArgumentError("nonfinite estimator/bootstrap"))
    draws = size(zstars,2)
    draws>=2 || throw(ArgumentError("at least two joint bootstrap columns required"))
    0<alpha<1 || throw(ArgumentError("alpha must be in (0,1)"))
    noise = [r^2*sum(abs2,view(zstars,:,b).-zhat) for b in 1:draws]
    all(isfinite,noise) || throw(ArgumentError("nonfinite bootstrap norm statistic"))
    # Invert the MC+1 tail exactly: rejection requires T > this order statistic.
    # Below MC resolution there is no finite critical value and no rejection.
    order = ceil(Int,(1-alpha)*(draws+1))
    critical = order>draws ? Inf : sort(noise)[max(1,order)]
    radius = sqrt(critical)/r
    bounds = polytope_confidence_bounds(zhat,isfinite(radius) ? radius : 0.0;
        rum_projector,cmu_projector,absolute_tolerance=tolerance)
    statements = Dict{String,Any}()
    for (name,q) in (("rum_membership",bounds["rum_squared_distance_lower_numerical"]),
            ("cmu_membership",bounds["cmu_squared_distance_lower_numerical"]),
            ("equal_distance",bounds["point_gap_lower_numerical"]))
        stat = r^2*q
        tie = r^2*tolerance
        all(isfinite,(stat,tie)) || throw(ArgumentError("nonfinite scaled statistic/tolerance"))
        tail = stat<=tie ? draws : count(x->x+tie>=stat,noise)
        p = (tail+1)/(draws+1)
        statements[name] = Dict("statistic_lower_numerical"=>stat,
            "p_value_mc_conservative"=>p,"tail_count"=>tail,
            "bootstrap_tail_probability_mc_interval"=>bootstrap_tail_interval(tail,draws),
            "reject"=>(stat>tie && stat>critical+tie && p<=alpha),
            "critical_value_squared_joint_norm"=>isfinite(critical) ? critical : nothing,
            "bootstrap_repetitions"=>draws,"numerical_tie_tolerance"=>tie)
    end
    # Infinite radius is represented explicitly; do not accidentally report
    # zero-radius exclusion diagnostics when alpha is below MC resolution.
    if !isfinite(radius)
        bounds["radius"] = nothing
        bounds["radius_infinite_below_mc_resolution"] = true
        bounds["distance_intervals_on_ball"] = nothing
        bounds["squared_gap_interval_on_ball"] = nothing
        for key in ("rum_membership_excluded_on_ball","cmu_membership_excluded_on_ball",
                "equal_distance_null_excluded_on_ball","explicit_positive_gap_lower_bound_on_ball")
            bounds[key] = false
        end
    end
    out = Dict{String,Any}("status"=>"computed_conditional_conservative",
        "method"=>"joint_norm_confidence_projection","confidence_bounds"=>bounds,
        "tests"=>statements,"protocol"=>Dict(
            "alpha"=>Float64(alpha),"rate"=>r,"bootstrap_repetitions"=>draws,
            "joint_ball_calibration"=>"order statistic of rate^2 * ||zstar - zhat||^2",
            "ball_order_index"=>order,"all_coordinates_retained"=>true,
            "sampling_covariance_inverse_used"=>false,"ridge_applied"=>false,
            "adding_up_reduction_must_be_exact"=>true,
            "population_zeros_inferred_from_sample_zeros"=>false,
            "calibration_conditional_on_joint_ball_coverage"=>true,
            "confidence_coverage_verified_by_routine"=>false,
            "ks_uniform_validity_transferred"=>false,"shi_2015_test_implemented"=>false,
            "uniform_validity_claimed"=>false,"finite_sample_exactness_claimed"=>false,
            "raw_data_efficiency_claimed"=>false,"failed_or_replaced_draws"=>0,
            "single_joint_ball_for_all_three_statements"=>true,
            "geometry_reference"=>"NestedConeDistanceGeometry and PolytopeConfidenceComparison Lean files",
            "confidence_projection_reference"=>"Romano and Shaikh (2008), Section 3.4, Remark 3.9; general confidence-region image principle, not an application of their subsampling theorem"))
    include_draws && (out["bootstrap_squared_joint_norms"] = noise)
    return out
end
