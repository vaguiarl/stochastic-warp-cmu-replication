# New helper: joint calendar-year empirical bootstrap and one stacked functional.
# Frozen comparison/source modules are imported, never modified.
include(joinpath(@__DIR__,"run_wilfried_chen_fang.jl"))

struct WilfriedCalendarPlan
    years::Vector{Int}
    annual_n::Vector{Int}
    seeds::Matrix{UInt64}
    repetitions::Int
    master_seed::Int
end

function WilfriedCalendarPlan(sizes::AbstractDict;repetitions::Int=1000,seed::Int=13732)
    repetitions>=2 || error("at least two joint draws required")
    years=sort(Int.(collect(keys(sizes))))
    !isempty(years) && allunique(years) || error("invalid calendar years")
    n=Int.([sizes[y] for y in years])
    all(>(0),n) || error("positive annual n required")
    # One fixed, sorted-year matrix: rows=calendar year, columns=joint draw.
    # Window execution order never determines or consumes the RNG stream.
    seeds=rand(MersenneTwister(seed),UInt64,length(years),repetitions)
    return WilfriedCalendarPlan(years,n,seeds,repetitions,seed)
end

function wjc_counts(plan::WilfriedCalendarPlan,year::Int,b::Int)
    j=findfirst(==(year),plan.years)
    j===nothing && error("calendar year absent from plan")
    1<=b<=plan.repetitions || error("joint draw outside plan")
    rng=MersenneTwister(plan.seeds[j,b])
    n=plan.annual_n[j]
    counts=zeros(Int,n)
    for _ in 1:n
        counts[rand(rng,1:n)]+=1
    end
    return counts
end

function wjc_count_digests(plan)
    digests=Matrix{String}(undef,length(plan.years),plan.repetitions)
    for (j,y) in enumerate(plan.years),b in 1:plan.repetitions
        digests[j,b]=bytes2hex(sha256(JSON3.write(wjc_counts(plan,y,b))))
    end
    return digests
end

function wjc_year_signature(year::KSReplication.PreparedKSYear)
    return bytes2hex(sha256(JSON3.write((year=year.year,n=year.n,
        expenditure=year.expenditure,log_expenditure=year.log_expenditure,
        instrument_basis=year.instrument_basis,expenditure_order=year.expenditure_order))))
end

function wjc_bootstrap(prepared::KSReplication.PreparedKSEstimator,plan;
        parallel::Bool=true)
    for year in prepared.years
        j=findfirst(==(year.year),plan.years)
        j!==nothing && year.n==plan.annual_n[j] || error("canonical annual n mismatch")
    end
    pi=Matrix{Float64}(undef,prepared.coordinates,plan.repetitions)
    summaries=Vector{Dict{String,Any}}(undef,plan.repetitions)
    function fit_draw(b)
        counts=[wjc_counts(plan,year.year,b) for year in prepared.years]
        result=estimate_prepared_probabilities(prepared;counts,probability_policy=:unclipped_affine)
        pi[:,b]=result.pi
        byyear=result.diagnostics["by_year"]
        summaries[b]=Dict("maximum_raw_adding_up_error"=>maximum(r["raw_probability_adding_up_error"] for r in byyear),
            "minimum_instrument_rank"=>minimum(r["instrument_basis_rank"] for r in byyear),
            "minimum_demand_rank"=>minimum(r["demand_basis_rank"] for r in byyear),
            "maximum_instrument_qr_diagonal_ratio"=>maximum(r["instrument_qr_diagonal_ratio"] for r in byyear),
            "maximum_demand_qr_diagonal_ratio"=>maximum(r["demand_qr_diagonal_ratio"] for r in byyear))
    end
    if parallel && Threads.nthreads()>1
        Threads.@threads :dynamic for b in 1:plan.repetitions
            fit_draw(b)
        end
    else
        for b in 1:plan.repetitions
            fit_draw(b)
        end
    end
    return (;pi,diagnostics=Dict(
        "maximum_raw_adding_up_error"=>maximum(d["maximum_raw_adding_up_error"] for d in summaries),
        "minimum_instrument_rank"=>minimum(d["minimum_instrument_rank"] for d in summaries),
        "minimum_demand_rank"=>minimum(d["minimum_demand_rank"] for d in summaries),
        "maximum_instrument_qr_diagonal_ratio"=>maximum(d["maximum_instrument_qr_diagonal_ratio"] for d in summaries),
        "maximum_demand_qr_diagonal_ratio"=>maximum(d["maximum_demand_qr_diagonal_ratio"] for d in summaries),
        "both_CF_stages_refit"=>true,"failed_or_replaced_draws"=>0,
        "raw_affine_probability_policy"=>true,"all_original_annual_n_retained"=>true))
end

function wjc_window_envelopes(cell,point,stars,Ar,Aw,rate,epsilons;workers=6,tolerance=1e-10)
    B=size(stars,2)
    size(stars,1)==length(point) && size(Ar,1)==length(point) && size(Aw,1)==length(point) ||
        error("window coordinate mismatch")
    1<=workers<=Threads.nthreads() || error("invalid worker count")
    all(isfinite,point) && all(isfinite,stars) || error("nonfinite joint estimator")
    issubset(ww_columns(Ar),ww_columns(Aw)) || error("window models not nested")
    identical=ww_columns(Ar)==ww_columns(Aw)
    if identical
        # Complete column-set equality is a mathematical identity, not a
        # zero-observed-gap screen. Every estimator perturbation contributes0.
        return Dict("cell"=>cell,"analytic_identical_models"=>true,
            "point_raw"=>0.0,"point_lower"=>0.0,"point_upper"=>0.0,
            "derivative_central"=>zeros(B,length(epsilons)),
            "derivative_lower"=>zeros(B,length(epsilons)),
            "derivative_upper"=>zeros(B,length(epsilons)),
            "point_projection"=>nothing,"projection_audit"=>Dict("fits"=>0,"fallback_count"=>0))
    end
    pair=(WilfriedCFNNLSProjector(Ar;transformation=:source_clip_composite),
        WilfriedCFNNLSProjector(Aw;transformation=:source_clip_composite))
    pairs=[(WilfriedCFNNLSProjector(Ar;transformation=:source_clip_composite),
        WilfriedCFNNLSProjector(Aw;transformation=:source_clip_composite)) for _ in 1:workers]
    isolation=wcf_isolation(pair,pairs)
    S=Matrix{Float64}(I,length(point),length(point))
    base=KSReplication._chen_fang_gap_envelope(point,S,pair[1],pair[2],tolerance)
    central,lower,upper=zeros(B,length(epsilons)),zeros(B,length(epsilons)),zeros(B,length(epsilons))
    for (k,e) in enumerate(epsilons)
        function run_worker(w)
            rp,cp=pairs[w]
            for b in w:workers:B
                try
                    target=point .+ (e*rate).*(view(stars,:,b).-point)
                    p=KSReplication._chen_fang_gap_envelope(target,S,rp,cp,tolerance)
                    central[b,k]=(p.raw-base.raw)/e^2
                    lower[b,k]=(p.lower-base.upper)/e^2
                    upper[b,k]=(p.upper-base.lower)/e^2
                    all(isfinite,(central[b,k],lower[b,k],upper[b,k])) || error("nonfinite derivative")
                    lower[b,k]<=upper[b,k] || error("reversed derivative bounds")
                    rawlo=(p.raw_lower-base.raw_upper)/e^2
                    rawhi=(p.raw_upper-base.raw_lower)/e^2
                    rawlo-tolerance/e^2<=central[b,k]<=rawhi+tolerance/e^2 ||
                        error("signed derivative outside raw numerical envelope")
                catch err
                    error("joint window=$cell epsilon=$e draw=$b failed, no replacement: $(sprint(showerror,err))")
                end
            end
        end
        if workers==1
            run_worker(1)
        else
            @sync for w in 1:workers
                Threads.@spawn run_worker(w)
            end
        end
    end
    return Dict("cell"=>cell,"analytic_identical_models"=>false,
        "point_raw"=>base.raw,"point_lower"=>base.lower,"point_upper"=>base.upper,
        "derivative_central"=>central,"derivative_lower"=>lower,"derivative_upper"=>upper,
        "point_projection"=>Dict("rum"=>KSReplication._information_projection_summary(base.rum),
            "warp"=>KSReplication._information_projection_summary(base.cmu)),
        "state_isolation"=>isolation,"projection_audit"=>wcf_projection_audit(pair,pairs))
end

function wjc_aggregate(envelopes,weights,rate,epsilons;alpha=.05)
    cells=sort([Int(e["cell"]) for e in envelopes])
    allunique(cells) && Set(cells)==Set(keys(weights)) || error("aggregate family incomplete")
    bycell=Dict(Int(e["cell"])=>e for e in envelopes)
    B,K=size(first(envelopes)["derivative_central"])
    K==length(epsilons) || error("epsilon dimension")
    raw,lo,hi=0.0,0.0,0.0
    central,lower,upper=zeros(B,K),zeros(B,K),zeros(B,K)
    for cell in cells
        e=bycell[cell];w=weights[cell]
        isfinite(w) && w>0 || error("all window weights must be strictly positive")
        size(e["derivative_central"])==(B,K) || error("draw dimensions differ")
        raw+=w*e["point_raw"];lo+=w*e["point_lower"];hi+=w*e["point_upper"]
        central .+=w.*e["derivative_central"]
        lower .+=w.*e["derivative_lower"]
        upper .+=w.*e["derivative_upper"]
    end
    observed=rate^2*max(0.0,raw)
    observedlower,observedupper=rate^2*lo,rate^2*hi
    order=ceil(Int,(1-alpha)*(B+1))
    results=Dict{String,Any}[]
    for (k,e) in enumerate(epsilons)
        c,l,u=central[:,k],lower[:,k],upper[:,k]
        width=maximum(u-l)
        centralcritical=order>B ? Inf : sort(c)[order]
        critical=order>B ? Inf : sort(u)[order]
        resolved=isfinite(centralcritical) && centralcritical>width && centralcritical>0
        tail=observedlower<=0 ? B : count(>=(observedlower),u)
        p=(tail+1)/(B+1)
        push!(results,Dict("epsilon"=>e,"epsilon_multiplier"=>e*sqrt(rate),
            "rate_times_epsilon"=>rate*e,"p_value_mc"=>p,
            "inferential_p_value_conditional"=>resolved ? p : nothing,
            "critical_value_positive_resolved"=>resolved,
            "central_critical_value"=>isfinite(centralcritical) ? centralcritical : nothing,
            "critical_value_upper"=>isfinite(critical) ? critical : nothing,
            "exceedances"=>tail,"B"=>B,"p_value_mc_interval"=>bootstrap_tail_interval(tail,B),
            "reject_conditional"=>resolved && observedlower>critical && p<=alpha,
            "calibration_status"=>resolved ? "computed_conditional_positive_critical" : "unresolved_or_degenerate_no_inferential_rejection",
            "maximum_numerical_derivative_envelope_width"=>width,
            "negative_signed_derivative_draws"=>count(<(0),c),
            "derivative_central"=>c,"derivative_lower"=>l,"derivative_upper"=>u))
    end
    return Dict("method"=>"single_joint_calendar_year_chen_fang_composite_aggregate",
        "cells"=>cells,"statistic"=>observed,"statistic_lower_numerical"=>observedlower,
        "statistic_upper_numerical"=>observedupper,"weighted_point_functional"=>max(0.0,raw),
        "rate"=>rate,"epsilon_results"=>results,"window_weights"=>weights,
        "primary_epsilon_multiplier"=>1.0,"Bonferroni_or_Holm_applied"=>false,
        "ordinary_centered_gap_bootstrap_used"=>false,
        "critical_value_continuity_and_joint_raw_bootstrap_validity_verified"=>false)
end
