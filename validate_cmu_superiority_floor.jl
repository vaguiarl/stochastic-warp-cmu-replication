# Independent Gaussian validation for the RUM-minus-probability-domain-floor
# numerical derivative. It never reads empirical estimates or selects tuning
# from empirical tails. Analytic projections isolate statistical code from QPs.
using Random, LinearAlgebra, Statistics, Distributions, JSON3, SHA

const CSFV_VERSION = "cmu-superiority-floor-gaussian-validation-v1"
const CSFV_REPETITIONS = 500
const CSFV_BOOTSTRAP = 399
const CSFV_SAMPLE_SIZES = [1000,10000]
const CSFV_MULTIPLIERS = [0.5,1.0,2.0]
const CSFV_PRIMARY = 1.0
const CSFV_ALPHA = 0.05
const CSFV_SEED = 202610121
const CSFV_EXCESS_SIZE_GATE_LEVEL = 0.01
const CSFV_OUTPUT = joinpath(@__DIR__,"results","cmu_superiority_floor_validation_20261012_v1.json")
const CSFV_ENGINE = joinpath(@__DIR__,"cmu_superiority_floor_inference.jl")
const CSFV_GROUPS = [[1,2],[3,4]]
const CSFV_RUM_TRIANGLE = Int8[0 1 0;1 0 1;0 0 1;1 1 0]
const CSFV_PRODUCT_SQUARE = hcat(CSFV_RUM_TRIANGLE,Int8[1,0,1,0])

if isfile(CSFV_ENGINE)&&!isdefined(@__MODULE__,:csfi_window_envelopes)
    include(CSFV_ENGINE)
end

csfv_sha(path) = bytes2hex(sha256(read(path)))
csfv_fullpoint(t) = [t[1],1-t[1],t[2],1-t[2]]

function csfv_triangle_projection(u,v)
    s=(clamp(u,zero(u),one(u)),clamp(v,zero(v),one(v)))
    s[1]+s[2] <= 1 && return s
    t=clamp((u-v+1)/2,zero(u),one(u))
    (t,1-t)
end

"""Full-coordinate squared distance gain; identity geometry is exactly zero."""
function csfv_gap(u,v; identical=false)
    identical && return zero(u+v)
    r=csfv_triangle_projection(u,v)
    s=(clamp(u,zero(u),one(u)),clamp(v,zero(v),one(v)))
    max(zero(u+v),2*((u-r[1])^2+(v-r[2])^2-(u-s[1])^2-(v-s[2])^2))
end

function csfv_gap_full(z; identical=false)
    length(z)==4 || throw(DimensionMismatch("two adding-up budgets"))
    u=(z[1]-z[2]+1)/2
    v=(z[3]-z[4]+1)/2
    csfv_gap(u,v;identical)
end

"""Exact analytic fit adapter for the inference engine's optional fit factory."""
function csfv_analytic_fit(z; identical=false)
    Q=Rational{BigInt}
    exact=csfv_gap_full(Q.(z);identical)
    central=Float64(exact)
    certificate=Dict("gap_lower_exact"=>string(exact),"gap_upper_exact"=>string(exact),
        "analytic_exact_gap"=>string(exact),"reference"=>"independent triangle/square closed forms")
    (;raw=central,lower=exact,upper=exact,certificate,exact_certificate=certificate,
        diagnostic=Dict("solver_status"=>"exact_analytic_oracle",
            "analytic_closed_form"=>true,"all_columns_priced"=>true))
end

"""Known second-order derivative at the specified null or boundary anchor."""
function csfv_oracle(kind::Symbol,h1,h2)
    kind in (:interior,:identity) && return 0.0
    kind==:edge && return max(0.0,h1+h2)^2
    kind==:outside && return 2max(0.0,h1)^2
    kind==:vertex || throw(ArgumentError("unknown analytic critical-cone oracle"))
    s=(max(h1,0.0),min(h2,0.0))
    if h1>=0 && h2<=0 && h1+h2<=0
        r=(h1,h2)
    else
        t=max(0.0,(h1-h2)/2)
        candidates=((0.0,0.0),(0.0,min(h2,0.0)),(t,-t))
        distances=map(q->(h1-q[1])^2+(h2-q[2])^2,candidates)
        r=candidates[argmin(distances)]
    end
    max(0.0,2*((h1-r[1])^2+(h2-r[2])^2-(h1-s[1])^2-(h2-s[2])^2))
end

function csfv_scenarios()
    L=[0.25 0.0;0.08 0.30]
    rankone=reshape([0.20,0.20],2,1)
    [(;id="common_interior",theta=(0.2,0.25),kind=:interior,null=true,L,local_shift=0.0,identical=false),
     (;id="binding_edge",theta=(0.5,0.5),kind=:edge,null=true,L,local_shift=0.0,identical=false),
     (;id="common_vertex",theta=(0.0,1.0),kind=:vertex,null=true,L,local_shift=0.0,identical=false),
     (;id="rank_one_boundary",theta=(0.5,0.5),kind=:edge,null=true,L=rankone,local_shift=0.0,identical=false),
     (;id="local_interior_null_one_SE",theta=(0.5,0.5),kind=:edge,null=true,L,local_shift=-1.0,identical=false),
     (;id="raw_outside_S_shared_projection",theta=(0.0,1.2),kind=:outside,null=true,L,local_shift=0.0,identical=false),
     (;id="identical_R_equals_S",theta=(0.5,0.5),kind=:identity,null=true,L,local_shift=0.0,identical=true),
     (;id="local_alternative_one_SE",theta=(0.5,0.5),kind=:edge,null=false,L,local_shift=1.0,identical=false),
     (;id="local_alternative_two_SE",theta=(0.5,0.5),kind=:edge,null=false,L,local_shift=2.0,identical=false),
     (;id="fixed_positive_alternative",theta=(0.65,0.65),kind=:edge,null=false,L,local_shift=0.0,identical=false)]
end

function csfv_binomial_interval(k,m;level=.95)
    alpha=1-level
    lower=k==0 ? 0.0 : quantile(Beta(k,m-k+1),alpha/2)
    upper=k==m ? 1.0 : quantile(Beta(k+1,m-k),1-alpha/2)
    [lower,upper]
end

function csfv_decision(statistic,draws;alpha=CSFV_ALPHA)
    B=length(draws)
    order=ceil(Int,(1-alpha)*(B+1))
    critical=order>B ? Inf : sort(draws)[order]
    resolved=isfinite(critical)&&critical>0
    exceedances=statistic<=0 ? B : count(>=(statistic),draws)
    tail=(exceedances+1)/(B+1)
    (;critical,resolved,tail,reject=resolved&&statistic>critical&&tail<=alpha,
        negative_draws=count(<(0),draws))
end

function csfv_case(s,n;repetitions=CSFV_REPETITIONS,bootstrap=CSFV_BOOTSTRAP,
        seed=CSFV_SEED,scenario_index=1,verbose=false)
    repetitions>0&&bootstrap>=39&&n>0 || throw(ArgumentError("positive frozen simulation dimensions"))
    rng=MersenneTwister(seed+100_003scenario_index+17n)
    rate=sqrt(n)
    eps=CSFV_MULTIPLIERS./sqrt(rate)
    signal_sd=sqrt(sum(abs2,vec(sum(s.L;dims=1))))
    theta=(s.theta[1]+s.local_shift*signal_sd/(2rate),
        s.theta[2]+s.local_shift*signal_sd/(2rate))
    K=length(eps)
    rejects=zeros(Int,K);resolved=zeros(Int,K);tails=zeros(K);negative=zeros(Int,K)
    zero_statistics=zeros(Int,K);max_oracle_error=zeros(K);mean_oracle_error=zeros(K)
    oracle_rejects=0;oracle_resolved=0;oracle_tails=0.0
    for rep in 1:repetitions
        noise=s.L*randn(rng,size(s.L,2))
        estimate=(theta[1]+noise[1]/rate,theta[2]+noise[2]/rate)
        base=csfv_gap(estimate...;identical=s.identical)
        statistic=n*base
        G=s.L*randn(rng,size(s.L,2),bootstrap)
        oracle=[csfv_oracle(s.kind,G[1,b],G[2,b]) for b in 1:bootstrap]
        oracle_result=csfv_decision(statistic,oracle)
        oracle_rejects+=oracle_result.reject
        oracle_resolved+=oracle_result.resolved
        oracle_tails+=oracle_result.tail
        for (k,e) in enumerate(eps)
            derivative=[(csfv_gap(estimate[1]+e*G[1,b],estimate[2]+e*G[2,b];identical=s.identical)-base)/e^2
                for b in 1:bootstrap]
            result=csfv_decision(statistic,derivative)
            rejects[k]+=result.reject;resolved[k]+=result.resolved;tails[k]+=result.tail
            negative[k]+=result.negative_draws;zero_statistics[k]+=statistic==0
            errors=abs.(derivative-oracle)
            max_oracle_error[k]=max(max_oracle_error[k],maximum(errors))
            mean_oracle_error[k]+=mean(errors)
        end
    end
    E=[1.0 0.0;-1.0 0.0;0.0 1.0;0.0 -1.0]
    covariance=E*s.L*s.L'*E'
    rows=Dict{String,Any}[]
    for k in 1:K
        excess_tail=ccdf(Binomial(repetitions,CSFV_ALPHA),rejects[k]-1)
        push!(rows,Dict("c"=>CSFV_MULTIPLIERS[k],"role"=>CSFV_MULTIPLIERS[k]==CSFV_PRIMARY ? "frozen_primary" : "sensitivity_only",
            "epsilon"=>eps[k],"rate_times_epsilon"=>rate*eps[k],
            "rejections"=>rejects[k],"rejection_rate"=>rejects[k]/repetitions,
            "rejection_rate_CP95_interval"=>csfv_binomial_interval(rejects[k],repetitions),
            "excess_size_one_sided_binomial_tail"=>excess_tail,
            "primary_size_gate_failed"=>s.null&&CSFV_MULTIPLIERS[k]==CSFV_PRIMARY&&excess_tail<CSFV_EXCESS_SIZE_GATE_LEVEL,
            "positive_critical_replicates"=>resolved[k],"degenerate_or_unresolved_replicates"=>repetitions-resolved[k],
            "mean_raw_MC_tail"=>tails[k]/repetitions,"zero_observed_statistics"=>zero_statistics[k],
            "negative_signed_derivative_draws"=>negative[k],
            "maximum_draw_difference_from_null_critical_cone_oracle"=>max_oracle_error[k],
            "mean_draw_difference_from_null_critical_cone_oracle"=>mean_oracle_error[k]/repetitions))
    end
    verbose&&println("Gaussian ",s.id," n=",n," c1 rejection=",rows[2]["rejection_rate"],
        " CP95=",rows[2]["rejection_rate_CP95_interval"])
    Dict("scenario"=>s.id,"n"=>n,"population_null"=>s.null,"population_theta"=>collect(theta),
        "population_gain"=>csfv_gap(theta...;identical=s.identical),"oracle_kind"=>string(s.kind),
        "oracle_local_boundary_for_local_sequences"=>s.local_shift!=0,
        "local_signal_SE_units"=>s.local_shift,"projection_dimension"=>4,
        "sampling_covariance_full_coordinates"=>covariance,"sampling_rank"=>rank(covariance),
        "underlying_gaussian_dimension"=>size(s.L,2),"adding_up_coordinates_discarded"=>false,
        "sampling_covariance_inverse_used"=>false,"repetitions"=>repetitions,"bootstrap"=>bootstrap,
        "c_results"=>rows,"oracle"=>Dict("rejections"=>oracle_rejects,"rejection_rate"=>oracle_rejects/repetitions,
            "CP95_interval"=>csfv_binomial_interval(oracle_rejects,repetitions),
            "positive_critical_replicates"=>oracle_resolved,"mean_raw_MC_tail"=>oracle_tails/repetitions),
        "failed_or_replaced_replicates"=>0)
end

function csfv_validation(;repetitions=CSFV_REPETITIONS,bootstrap=CSFV_BOOTSTRAP,
        sample_sizes=CSFV_SAMPLE_SIZES,seed=CSFV_SEED,verbose=false)
    rows=Dict{String,Any}[]
    for n in sample_sizes,(i,s) in enumerate(csfv_scenarios())
        push!(rows,csfv_case(s,n;repetitions,bootstrap,seed,scenario_index=i,verbose))
    end
    failures=[(r["scenario"],r["n"]) for r in rows if any(c["primary_size_gate_failed"] for c in r["c_results"])]
    Dict("status"=>isempty(failures) ? "passed_prespecified_gaussian_size_gate" : "failed_gaussian_size_gate",
        "primary_gate_failures"=>failures,"cases"=>rows,
        "scope"=>"Finite Gaussian implementation and size diagnostics; no uniform finite-sample theorem or UK CF bootstrap validation.",
        "protocol"=>Dict("version"=>CSFV_VERSION,"M"=>repetitions,"B"=>bootstrap,"sample_sizes"=>sample_sizes,
            "c_grid"=>CSFV_MULTIPLIERS,"primary_c"=>CSFV_PRIMARY,"seed"=>seed,"alpha"=>CSFV_ALPHA,
            "gate"=>"Fail only if the primary null rejection count has one-sided Binomial(M,.05) excess-size tail below .01; retain all CP95 intervals.",
            "gate_level"=>CSFV_EXCESS_SIZE_GATE_LEVEL,"c_selected_by_real_data_p_values"=>false,
            "same_gaussian_draws_across_c"=>true,"analytic_projections_no_QP_noise"=>true,
            "ordinary_centered_gain_bootstrap_used"=>false,"signed_derivative_no_extra_half"=>true,
            "clipping_of_raw_estimator_or_bootstrap"=>false,"nonpositive_critical_means_no_inferential_rejection"=>true,
            "singular_covariance_without_inverse_or_rank_selection"=>true,
            "raw_outside_S_functional_null_included"=>true,"real_empirical_tail_computed"=>false))
end

function csfv_engine_integration()
    isdefined(@__MODULE__,:csfi_window_envelopes)||error("new inference engine not loaded")
    rng=MersenneTwister(CSFV_SEED+9181)
    B=79; rate=sqrt(1000)
    E=[1.0 0.0;-1.0 0.0;0.0 1.0;0.0 -1.0]
    maximum_point_error=0.0;maximum_derivative_error=0.0
    maximum_adding_up_error=0.0;decision_agreement=true;calls=0
    for s in csfv_scenarios(),rep in 1:2
        signal_sd=sqrt(sum(abs2,vec(sum(s.L;dims=1))))
        theta=(s.theta[1]+s.local_shift*signal_sd/(2rate),s.theta[2]+s.local_shift*signal_sd/(2rate))
        noise=s.L*randn(rng,size(s.L,2))
        point=csfv_fullpoint((theta[1]+noise[1]/rate,theta[2]+noise[2]/rate))
        G=s.L*randn(rng,size(s.L,2),B)
        stars=point.+E*G/rate
        maximum_adding_up_error=max(maximum_adding_up_error,
            maximum(abs.(vec(sum(stars[1:2,:];dims=1)).-1)),
            maximum(abs.(vec(sum(stars[3:4,:];dims=1)).-1)))
        Ar=s.identical ? CSFV_PRODUCT_SQUARE : CSFV_RUM_TRIANGLE
        # Each worker receives a distinct closure; its identity is not used in
        # the pure analytic calculation or the synthetic data stream.
        factory=worker->let identity=Ref(worker)
            z->(identity[];csfv_analytic_fit(z;identical=s.identical))
        end
        env=csfi_window_envelopes(1,point,stars,Ar,CSFV_GROUPS,rate;
            multipliers=CSFV_MULTIPLIERS,workers=1,tolerance=1e-12,
            store_draw_certificates=false,fit_factory=factory)
        aggregate=csfi_aggregate([env],Dict(1=>1.0),rate;alpha=CSFV_ALPHA,require_full_family=false)
        base=csfv_gap_full(point;identical=s.identical)
        maximum_point_error=max(maximum_point_error,abs(env["point_raw"]-base))
        for (k,c) in enumerate(CSFV_MULTIPLIERS)
            eps=c/sqrt(rate)
            direct=[(csfv_gap_full(point+eps*rate*(stars[:,b]-point);identical=s.identical)-base)/eps^2 for b in 1:B]
            maximum_derivative_error=max(maximum_derivative_error,
                maximum(abs.(vec(env["derivative_central"][:,k])-direct)))
            expected=csfv_decision(rate^2*base,direct)
            result=aggregate["epsilon_results"][k]
            decision_agreement&=result["reject_conditional"]==expected.reject
            decision_agreement&=result["critical_value_positive_resolved"]==expected.resolved
        end
        calls+=1
    end
    passed=maximum_point_error<1e-12&&maximum_derivative_error<1e-9&&
        maximum_adding_up_error<1e-12&&decision_agreement
    passed||error("engine callback/aggregation differs from independent oracle")
    Dict("status"=>"passed_independent_analytic_engine_integration","passed"=>passed,
        "controlled_replicates"=>calls,"bootstrap_draws_each"=>B,
        "maximum_point_error"=>maximum_point_error,
        "maximum_derivative_error"=>maximum_derivative_error,
        "maximum_adding_up_error"=>maximum_adding_up_error,
        "all_rejection_and_critical_gates_agree"=>decision_agreement,
        "numeric_QP_validation_is_separate"=>true)
end

function csfv_main(args)
    args==["--write-approved-report"] || error("fixed validation protocol requires --write-approved-report")
    ispath(CSFV_OUTPUT)&&error("refusing to overwrite a validation report")
    isfile(CSFV_ENGINE)||error("freeze the new inference engine before validation")
    hashes=Dict("validation_source"=>csfv_sha(@__FILE__),"inference_engine"=>csfv_sha(CSFV_ENGINE),
        "Project.toml"=>csfv_sha(joinpath(@__DIR__,"Project.toml")),
        "Manifest.toml"=>csfv_sha(joinpath(@__DIR__,"Manifest.toml")))
    start=time()
    report=csfv_validation(;verbose=true)
    report["engine_integration"]=csfv_engine_integration()
    csfv_sha(CSFV_ENGINE)==hashes["inference_engine"]||error("engine changed during validation")
    csfv_sha(@__FILE__)==hashes["validation_source"]||error("validation changed during run")
    report["implementation_sha256"]=hashes
    report["elapsed_seconds"]=time()-start
    report["maintained_unverified_UK_assumptions"]=["Joint annual-iid CF estimator CLT/bootstrap validity",
        "Negligible fixed-degree polynomial approximation bias","Fixed source prices, geometry and trimming",
        "No survey-design PSU/weight uncertainty correction"]
    open(CSFV_OUTPUT,"w") do io
        JSON3.pretty(io,report)
    end
    println("Saved NEW validation report: ",CSFV_OUTPUT," status=",report["status"])
    isempty(report["primary_gate_failures"])||error("primary synthetic null size gate failed; stop empirical inference")
    report
end

if abspath(PROGRAM_FILE)==@__FILE__
    csfv_main(ARGS)
end
