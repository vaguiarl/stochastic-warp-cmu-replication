# New conditional Chen--Fang inference for Psi=d_RUM^2-d_product_simplex^2.
# Existing estimators, geometries, models, raw caches and result files are read-only.
using KSReplication, LinearAlgebra, Statistics, SHA
include(joinpath(@__DIR__,"cmu_empty_inner_projection.jl"))
isdefined(@__MODULE__,:NormalizedHullProjector) ||
    include(joinpath(@__DIR__,"wilfried_normalized_working_set.jl"))

const CSFI_MULTIPLIERS=[.5,1.,2.]
const CSFI_PRIMARY_MULTIPLIER=1.
const CSFI_Q=Rational{BigInt}

"""Only complete Cartesian coverage proves RUM equals the product simplex.

RUM==complete WARP is deliberately NOT an analytic-zero shortcut here.
"""
function csfi_cartesian_identity(f)
    expected=prod(big(length(g)) for g in f.budget_groups)
    unique_columns=Set(Tuple(view(f.complete_rows,:,h)) for h in 1:f.n_complete)
    length(unique_columns)==f.n_complete && big(f.n_complete)==expected
end

function csfi_compact_certificate(f,target,record,floor)
    cert=record["certificate"]
    indices=Int.(cert["primal_indices"])
    masses=parse.(CSFI_Q,String.(cert["primal_masses_exact"]))
    all(i->i<=f.n_complete,indices) || error("zero-cap fit uses an incomplete/null consumer")
    denominator=foldl(lcm,Base.denominator.(masses);init=big(1))
    numerators=[numerator(v)*div(denominator,Base.denominator(v)) for v in masses]
    lo=parse(CSFI_Q,String(cert["squared_distance_lower_exact"]))
    hi=parse(CSFI_Q,String(cert["squared_distance_upper_exact"]))
    Dict("target_float"=>Float64.(target),"complete_column_indices"=>indices,
        "population_mass_common_denominator"=>string(denominator),
        "population_mass_numerators"=>string.(numerators),
        "RUM_squared_distance_lower_exact"=>string(lo),"RUM_squared_distance_upper_exact"=>string(hi),
        "simplex_target_exact"=>string.(floor.q),"probability_domain_floor_exact"=>string(floor.floor),
        "Psi_lower_exact"=>string(lo-floor.floor),"Psi_upper_exact"=>string(hi-floor.floor),
        "all_complete_columns_priced"=>f.n_complete,"fixed_target_rational_certificate"=>true,
        "certificate_scope"=>"Binary-rational target, normalized complete-type primal, exact residual-support dual and exact product-simplex KKT; not a statistical theorem")
end

"""Exact full support pricing with common-denominator INTEGER arithmetic.

No per-column rational gcds: lambda=numerators/denominator. An explicit worst
sum bound permits Int128 additions; otherwise arbitrary-size integers are used.
All original complete vertices are priced, regardless of the proposal working set.
"""
function csfi_integer_support(f,numerators)
    bound=f.n_budgets*maximum(abs,numerators)
    small=bound<=big(typemax(Int128))
    y=small ? Int128.(numerators) : numerators
    best=sum(y[f.complete_rows[j,1]] for j in 1:f.n_budgets)
    for h in 2:f.n_complete
        score=zero(eltype(y))
        for j in 1:f.n_budgets
            score+=y[f.complete_rows[j,h]]
        end
        best=max(best,score)
    end
    (;maximum=BigInt(best),arithmetic=small ? "checked_Int128" : "BigInt")
end

function csfi_integer_certificate(f,target,proposal,floor;store_certificate=true)
    all(isfinite,proposal) || error("nonfinite working simplex proposal")
    active=findall(>(0),proposal)
    counts=[max(big(0),round(BigInt,ldexp(Float64(proposal[h]),48))) for h in active]
    keep=findall(>(0),counts);indices=active[keep];counts=counts[keep]
    M=sum(counts);M>0 || error("zero repaired complete mass")
    qnum=zeros(BigInt,f.n_patches)
    for (h,numerator) in zip(indices,counts),j in 1:f.n_budgets
        qnum[f.complete_rows[j,h]]+=numerator
    end
    all(sum(qnum[g])==M for g in f.budget_groups) || error("integer primal budget adding-up failed")
    p=CSFI_Q.(target)
    D=foldl(lcm,denominator.(p);init=big(1))
    pnum=[numerator(v)*div(D,denominator(v)) for v in p]
    L=lcm(D,M)
    residual_num=div(L,D)*pnum-div(L,M)*qnum
    support=csfi_integer_support(f,residual_num)
    norm=sum(abs2,residual_num)//L^2
    targetdot=sum(pnum.*residual_num)//(D*L)
    lower=max(floor.floor,2targetdot-norm-2*(support.maximum//L))
    upper=norm
    lower<=upper || error("exact common-denominator objective bounds inverted")
    certificate=store_certificate ? Dict("target_float"=>Float64.(target),"complete_column_indices"=>indices,
        "population_mass_common_denominator"=>string(M),"population_mass_numerators"=>string.(counts),
        "RUM_squared_distance_lower_exact"=>string(lower),"RUM_squared_distance_upper_exact"=>string(upper),
        "simplex_target_exact"=>string.(floor.q),"probability_domain_floor_exact"=>string(floor.floor),
        "Psi_lower_exact"=>string(lower-floor.floor),"Psi_upper_exact"=>string(upper-floor.floor),
        "all_complete_columns_priced"=>f.n_complete,"fixed_target_rational_certificate"=>true,
        "exact_support_arithmetic"=>support.arithmetic,
        "certificate_scope"=>"Binary-rational target, dyadic normalized primal, exhaustive common-denominator integer support and exact simplex KKT; not a statistical theorem") : nothing
    (;lower=lower-floor.floor,upper=upper-floor.floor,certificate,arithmetic=support.arithmetic)
end

function csfi_make_fit(A,groups; tolerance=1e-8,store_certificate=true)
    f=ceip_build_family(A,groups;label="unchanged_complete_RUM_for_Psi")
    identical=csfi_cartesian_identity(f)
    projector=identical ? nothing : NormalizedHullProjector(A;certificate_tolerance=tolerance/4,
        pricing_tolerance=1e-11,allow_full_fallback=false)
    fallback=Ref{Any}(nothing)
    function fit(target)
        floor=ceip_simplex_floor(target,groups)
        if identical
            certificate=store_certificate ? Dict("analytic_RUM_equals_product_simplex"=>true,
                "verified_complete_cartesian_columns"=>f.n_complete,"target_float"=>Float64.(target),
                "simplex_target_exact"=>string.(floor.q),"probability_domain_floor_exact"=>string(floor.floor),
                "Psi_lower_exact"=>"0//1","Psi_upper_exact"=>"0//1") : nothing
            return (;lower=CSFI_Q(0),upper=CSFI_Q(0),certificate,
                diagnostic=Dict("solver_status"=>"analytic_complete_Cartesian_identity","all_columns_priced"=>true))
        end
        reason=""
        try
            proposal=nh_fit!(projector,target)
            exact=csfi_integer_certificate(f,target,proposal.mass,floor;store_certificate)
            exact.upper-exact.lower<=CSFI_Q(tolerance) || error("working proposal fails EXACT objective gap")
            return (;exact.lower,exact.upper,exact.certificate,
                diagnostic=Dict("solver_status"=>proposal.diagnostic.solver_status,
                    "all_columns_priced"=>true,"RUM_complete_columns"=>f.n_complete,
                    "exact_objective_width"=>string(exact.upper-exact.lower),
                    "integer_support_arithmetic"=>exact.arithmetic,"working_cuts"=>proposal.diagnostic.working_cuts,
                    "full_Clarabel_fallback_used"=>false))
        catch err
            reason=sprint(showerror,err)
        end
        # Failure is not a replaced draw: solve the SAME target with the entire
        # unchanged hull, and still require the exact final objective gate.
        fallback[]===nothing && (fallback[]=CMUEmptyInnerProjector(f;tolerance))
        record=ceip_fit!(fallback[],target,0.)
        c=record["certificate"]
        lower=parse(CSFI_Q,String(c["squared_distance_lower_exact"]))-floor.floor
        upper=parse(CSFI_Q,String(c["squared_distance_upper_exact"]))-floor.floor
        0<=lower<=upper || error("RUM-minus-simplex-floor interval invalid")
        (;lower,upper,certificate=store_certificate ? csfi_compact_certificate(f,target,record,floor) : nothing,
          diagnostic=Dict("solver_status"=>record["solver_status"],"all_columns_priced"=>true,
            "RUM_complete_columns"=>f.n_complete,"exact_objective_width"=>string(upper-lower),
            "full_Clarabel_fallback_used"=>true,"fallback_reason"=>reason))
    end
    (;fit,model=projector===nothing ? nothing : projector.model,identical,family=f)
end

function csfi_check_fit(fit)
    fit.lower isa CSFI_Q && fit.upper isa CSFI_Q && 0<=fit.lower<=fit.upper ||
        error("fit requires exact rational nonnegative objective endpoints")
    fit
end

"""All-data single-window numerical second derivatives, no model-null recentering.

For analytic validation only, fit_factory(worker) may return a pure callable
target -> (;lower::Rational{BigInt},upper::Rational{BigInt},certificate,diagnostic).
Real-data callers leave fit_factory unset. Every numerical worker then owns an
independent fixed-RUM QP and computes the product-simplex floor analytically.
No old RUM==WARP shortcut, no clamp, covariance inverse, ridge, or deletion.
The target/h arithmetic is Float64; certificates enclose the resulting exact
binary-rational targets, not unrounded symbolic perturbation arithmetic.
"""
function csfi_window_envelopes(cell,point,stars,A,groups,rate;
        multipliers=CSFI_MULTIPLIERS,workers=1,tolerance=1e-8,
        store_draw_certificates=true,fit_factory=nothing)
    B=size(stars,2); I=length(point)
    B>=2 && size(stars,1)==I==size(A,1) && I>0 || error("window dimensions invalid")
    all(isfinite,point)&&all(isfinite,stars)&&isfinite(rate)&&rate>0 || error("nonfinite target/rate")
    Float64.(multipliers)==CSFI_MULTIPLIERS || error("frozen c grid is .5,1,2")
    1<=workers<=Threads.nthreads() || error("invalid worker count")
    f=ceip_build_family(A,groups)
    identical=csfi_cartesian_identity(f)
    sources=fit_factory===nothing ? [csfi_make_fit(A,groups;tolerance,
        store_certificate=(worker==0||store_draw_certificates)) for worker in 0:workers] : nothing
    if sources!==nothing
        models=[s.model for s in sources if s.model!==nothing]
        length(unique(objectid.(models)))==length(models) || error("mutable QP models shared")
        fits=[s.fit for s in sources]
    else
        fits=[fit_factory(worker) for worker in 0:workers]
    end
    base=csfi_check_fit(fits[1](point))
    epsilons=Float64.(multipliers)./sqrt(Float64(rate)); K=length(epsilons)
    central=zeros(B,K); lower=zeros(B,K); upper=zeros(B,K)
    lower_exact=Matrix{String}(undef,B,K);upper_exact=Matrix{String}(undef,B,K)
    certificates=store_draw_certificates ? Matrix{Any}(undef,B,K) : nothing
    statuses=Matrix{String}(undef,B,K)
    fallback_used=falses(B,K);integer_support=Matrix{String}(undef,B,K)
    function run_worker(worker)
        fit=fits[worker+1]
        for (k,h) in enumerate(epsilons),b in worker:workers:B
            try
                target=Float64.(point).+(h*Float64(rate)).*(view(stars,:,b).-point)
                all(isfinite,target) || error("nonfinite perturbation")
                p=csfi_check_fit(fit(target))
                denominator=CSFI_Q(h)^2
                lo=(p.lower-base.upper)/denominator
                hi=(p.upper-base.lower)/denominator
                signed=((p.lower+p.upper)-(base.lower+base.upper))/(2denominator)
                lo<=signed<=hi || error("signed derivative interval inverted")
                central[b,k]=Float64(signed)
                lower[b,k]=ceip_float_lower(lo);upper[b,k]=ceip_float_upper(hi)
                lower_exact[b,k]=string(lo);upper_exact[b,k]=string(hi)
                all(isfinite,(central[b,k],lower[b,k],upper[b,k])) || error("nonfinite derivative endpoint")
                statuses[b,k]=String(p.diagnostic["solver_status"])
                fallback_used[b,k]=get(p.diagnostic,"full_Clarabel_fallback_used",false)
                integer_support[b,k]=get(p.diagnostic,"integer_support_arithmetic","analytic_or_Clarabel_fallback")
                if store_draw_certificates
                    certificates[b,k]=Dict("RUM_floor_fit"=>p.certificate,
                        "derivative_lower_exact"=>string(lo),"derivative_upper_exact"=>string(hi),
                        "signed_midpoint_derivative_exact"=>string(signed),"h_exact"=>string(CSFI_Q(h)))
                end
            catch err
                error("Psi window=$cell c=$(multipliers[k]) draw=$b failed, NONE replaced: $(sprint(showerror,err))")
            end
        end
    end
    if workers==1
        run_worker(1)
    else
        @sync for worker in 1:workers
            Threads.@spawn run_worker(worker)
        end
    end
    Dict("cell"=>Int(cell),"analytic_RUM_equals_product_simplex"=>identical && fit_factory===nothing,
        "B"=>B,"coordinates"=>I,"c_dimension"=>K,"matrix_storage"=>"column_major_B_by_c",
        "analytic_validation_fit_factory_used"=>fit_factory!==nothing,
        "point_raw"=>Float64((base.lower+base.upper)/2),"point_lower"=>ceip_float_lower(base.lower),
        "point_upper"=>ceip_float_upper(base.upper),"point_lower_exact"=>string(base.lower),
        "point_upper_exact"=>string(base.upper),"point_exact_certificate"=>base.certificate,
        "derivative_central"=>central,"derivative_lower"=>lower,"derivative_upper"=>upper,
        "derivative_lower_exact"=>lower_exact,"derivative_upper_exact"=>upper_exact,
        "per_draw_exact_certificates"=>certificates,"epsilons"=>epsilons,
        "projection_audit"=>Dict("all_complete_columns_priced"=>true,"complete_RUM_columns"=>f.n_complete,
            "exact_rational_point_and_perturbed_intervals"=>true,"workers"=>workers,
            "private_mutable_QP_models"=>fit_factory===nothing,"failed_or_replaced_draws"=>0,
            "maximum_derivative_interval_width"=>maximum(upper-lower),
            "full_Clarabel_fallback_draws"=>count(fallback_used),
            "integer_support_arithmetic_counts"=>Dict(s=>count(==(s),integer_support) for s in unique(integer_support)),
            "solve_status_counts"=>Dict(s=>count(==(s),statuses) for s in unique(statuses))),
        "protocol"=>Dict("functional"=>"Psi(raw)=d_unchanged_complete_RUM(raw)^2-d_product_simplex(raw)^2",
            "raw_target_preprocessed_or_null_recentered"=>false,"RUM_equals_WARP_shortcut_used"=>false,
            "sampling_covariance_inverse_ridge_coordinate_deletion"=>false,"no_extra_half_factor"=>true,
            "c_primary"=>1.,"c_grid"=>CSFI_MULTIPLIERS,"c_selected_by_p_value"=>false,
            "h_condition"=>"h=c/sqrt(rate); h->0 and rate*h->infinity",
            "certificates_apply_to_computed_binary_rational_targets"=>true,
            "null"=>"equal population RUM and product-simplex distances; exactly every positive-cap actual-inner equal-distance null"))
end

function csfi_aggregate(envelopes,weights,rate;alpha=.05,require_full_family=true)
    cells=sort([Int(e["cell"]) for e in envelopes])
    allunique(cells) && Set(cells)==Set(keys(weights)) || error("weighted window registry mismatch")
    require_full_family && cells!=collect(1:19) && error("all19 windows are required")
    isfinite(rate)&&rate>0&&0<alpha<1 || error("aggregate rate/alpha invalid")
    bycell=Dict(Int(e["cell"])=>e for e in envelopes)
    B,K=size(first(envelopes)["derivative_central"]); B>=2&&K==3 || error("draw/c grid incomplete")
    rateq=CSFI_Q(Float64(rate)); ratetwo=rateq^2
    pointlo=CSFI_Q(0);pointhi=CSFI_Q(0)
    central=zeros(B,K);lower_exact=zeros(CSFI_Q,B,K);upper_exact=zeros(CSFI_Q,B,K)
    for cell in cells
        e=bycell[cell]; w=CSFI_Q(weights[cell]); w>0 || error("positive window weights required")
        size(e["derivative_central"])==(B,K) && e["protocol"]["c_grid"]==CSFI_MULTIPLIERS || error("draw dimensions/c grid changed")
        pointlo+=w*parse(CSFI_Q,String(e["point_lower_exact"]))
        pointhi+=w*parse(CSFI_Q,String(e["point_upper_exact"]))
        for k in 1:K,b in 1:B
            central[b,k]+=Float64(w)*e["derivative_central"][b,k]
            lower_exact[b,k]+=w*parse(CSFI_Q,String(e["derivative_lower_exact"][b,k]))
            upper_exact[b,k]+=w*parse(CSFI_Q,String(e["derivative_upper_exact"][b,k]))
        end
    end
    lower=ceip_float_lower.(lower_exact);upper=ceip_float_upper.(upper_exact)
    observedlo=ratetwo*pointlo;observedhi=ratetwo*pointhi
    observed_lower=ceip_float_lower(observedlo);observed_upper=ceip_float_upper(observedhi)
    order=ceil(Int,(1-alpha)*(B+1)); rows=Dict{String,Any}[]
    for k in 1:K
        l,u,c=lower[:,k],upper[:,k],central[:,k]
        width=maximum(u-l); critical=order>B ? Inf : sort(u)[order]
        centralcritical=order>B ? Inf : sort(c)[order]
        resolved=isfinite(centralcritical)&&centralcritical>width&&centralcritical>0
        tail=count(>=(observed_lower),u); p=(tail+1)/(B+1)
        conditional_reject=resolved&&observed_lower>critical&&p<=alpha
        push!(rows,Dict("c_multiplier"=>CSFI_MULTIPLIERS[k],"epsilon_multiplier"=>CSFI_MULTIPLIERS[k],
            "h"=>CSFI_MULTIPLIERS[k]/sqrt(rate),"epsilon"=>CSFI_MULTIPLIERS[k]/sqrt(rate),
            "role"=>k==2 ? "frozen_primary" : "retained_sensitivity_NOT_selected",
            "central_critical_value"=>isfinite(centralcritical) ? centralcritical : nothing,
            "critical_value_upper"=>isfinite(critical) ? critical : nothing,
            "critical_value_positive_resolved"=>resolved,"maximum_projection_uncertainty"=>width,
            "exceedances"=>tail,"B"=>B,"p_value_mc"=>p,
            "p_value_mc_interval"=>KSReplication.bootstrap_tail_interval(tail,B),
            "inferential_p_value_conditional"=>resolved ? p : nothing,
            "reject_conditional"=>conditional_reject,"negative_signed_derivative_draws"=>count(<(0),c),
            "quantile_order_statistic_index"=>order,
            "calibration_status"=>resolved ? "conditional_finite_step_positive_critical" : "degenerate_or_unresolved_NO_inferential_rejection",
            "bootstrap_statistics"=>c,"bootstrap_statistics_lower"=>l,"bootstrap_statistics_upper"=>u))
        rows[end]["bootstrap_statistics_lower_exact"]=string.(lower_exact[:,k])
        rows[end]["bootstrap_statistics_upper_exact"]=string.(upper_exact[:,k])
    end
    Dict("status"=>"computed_conditional_Psi_second_order_bootstrap",
        "method"=>"joint_raw_unchanged_RUM_minus_exact_product_simplex_floor_Chen_Fang",
        "observed_statistic_lower"=>observed_lower,"observed_statistic_upper"=>observed_upper,
        "observed_statistic_lower_exact"=>string(observedlo),"observed_statistic_upper_exact"=>string(observedhi),
        "point_Psi_lower_exact"=>string(pointlo),"point_Psi_upper_exact"=>string(pointhi),
        "epsilon_results"=>rows,"primary_result_index_one_based"=>2,"primary_epsilon_multiplier"=>1.,
        "alpha"=>alpha,"rate"=>Float64(rate),"B"=>B,"windows"=>cells,
        "weights_exact"=>Dict(string(c)=>string(CSFI_Q(weights[c])) for c in cells),
        "protocol"=>Dict("all19_required"=>require_full_family,"no_Bonferroni_or_Holm"=>true,
            "weights_are_exact_frozen_binary_float_or_supplied_rational"=>true,
            "rate_squared_is_exact_square_of_declared_binary_float_rate"=>true,
            "positive_cap_nulls_identical_by_beta_Psi_le_phi_le_Psi"=>true,
            "positive_cap_effect_quantities_are_not_the_Psi_statistic"=>true,
            "rejecting_joint_null_does_not_reject_each_individual_window"=>true,
            "cap0_equal_distance_null_not_asserted_equivalent"=>true,
            "nearest_RUM_null_recenter_used"=>false,"normal_residual_not_removed"=>true,
            "ordinary_centered_gain_bootstrap_used"=>false,"c_selected_by_smallest_real_data_p"=>false,
            "critical_value_continuity_proved_by_positive_finite_step_quantile"=>false,
            "bootstrap_CLT_bias_survey_and_geometry_assumptions_verified"=>false,
            "uniform_finite_sample_validity_claimed"=>false,"no_general_empty_CMU_adequacy_claim"=>true,
            "sampling_significance_is_conditional"=>true))
end

"""Optional exact interval bridge check against all saved positive-cap point fits.

The feasible candidate uses the saved RUM primal, not an assumed exact optimizer.
This arithmetic check supplements, and does not replace, the analytic theorem.
"""
function csfi_check_saved_bridge(window)
    parseq(x)=parse(CSFI_Q,String(x))
    r=window["fixed_RUM"]["certificate"]
    rlo=parseq(r["squared_distance_lower_exact"]);rup=parseq(r["squared_distance_upper_exact"])
    qr=parseq.(r["conditional_fitted_exact"])
    floor=parseq(window["probability_domain_floor"]["squared_distance_exact"])
    qs=parseq.(window["probability_domain_floor"]["simplex_target_exact"])
    checks=Dict{String,Any}[]
    for profile in window["profiles"]
        profile["empty_allowance_cap"]==0 && continue
        c=profile["CMU_actual_empty_inner"]["certificate"]
        beta=parseq(c["unary_mix_fraction_exact"])
        clo=parseq(c["squared_distance_lower_exact"]);cup=parseq(c["squared_distance_upper_exact"])
        gainlo=max(CSFI_Q(0),rlo-cup);gainup=max(CSFI_Q(0),rup-clo)
        beta*(rlo-floor)<=gainup && gainlo<=rup-floor || error("saved beta-Psi gain sandwich intervals conflict")
        stronger=rlo-((1-beta)*rup+beta*floor-beta*(1-beta)*sum(abs2,qr-qs))
        stronger<=gainup || error("saved feasible-mixture gain bound conflicts")
        push!(checks,Dict("cap"=>profile["empty_allowance_cap"],"beta_exact"=>string(beta),
            "beta_Psi_lower_exact"=>string(beta*(rlo-floor)),"Psi_upper_exact"=>string(rup-floor),
            "feasible_candidate_gain_lower_exact"=>string(stronger),"interval_bridge_verified"=>true))
    end
    Dict("cell"=>window["cell"],"positive_cap_checks"=>checks,
        "scope"=>"Interval-compatible finite arithmetic check, not proof of sampling inference")
end

csfi_implementation_sha256()=bytes2hex(sha256(read(@__FILE__)))
