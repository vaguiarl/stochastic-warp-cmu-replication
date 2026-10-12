#!/usr/bin/env julia
# Independent read-only arithmetic audit of saved actual-empty-CMU inner reports.
# Does not call the projection/certificate library and never re-estimates data.
using KSReplication, JSON3, LinearAlgebra, SparseArrays, SHA, Random
include(joinpath(@__DIR__,"common_cmu_empty_witnesses.jl"))

const VCAE_Q=Rational{BigInt}
const VCAE_CAPS=[0.,.005,.01,.025,.05,.10,.20]
vcae_read(path)=JSON3.read(read(path,String))
vcae_q(x)=parse(VCAE_Q,String(x))
vcae_sha(path)=bytes2hex(sha256(read(path)))
vcae_require(condition,message)=condition || error(message)

function vcae_equal(a,b)
    if a isa AbstractDict && b isa AbstractDict
        return Set(keys(a))==Set(keys(b)) && all(vcae_equal(a[k],b[k]) for k in keys(a))
    elseif a isa AbstractArray && b isa AbstractArray
        return size(a)==size(b) && all(vcae_equal(a[i],b[i]) for i in eachindex(a))
    end
    a==b
end

function vcae_family(A,groups)
    I,H=size(A); J=length(groups)
    rows=Matrix{Int}(undef,J,H)
    owner=zeros(Int,I)
    for (j,g) in enumerate(groups)
        owner[g].=j
        for h in 1:H
            row=filter(i->A[i,h]!=0,g)
            vcae_require(length(row)==1,"invalid complete incidence")
            rows[j,h]=only(row)
        end
    end
    ordinary=hcat(sparse(Int8.(A)),sparse(1:I,1:I,ones(Int8,I),I,I),spzeros(Int8,I,1))
    responses=hcat(sparse(ones(Int8,J,H)),sparse(owner,1:I,ones(Int8,I),J,I),spzeros(Int8,J,1))
    (;A=ordinary,D=responses,complete_rows=rows,groups,owner,I,H,J,T=H+I+1,null=H+I+1)
end

function vcae_complete_support(f,y)
    maximum(sum(y[f.complete_rows[j,h]] for j in 1:f.J) for h in 1:f.H)
end
vcae_full_support(f,y)=max(VCAE_Q(0),maximum(y),vcae_complete_support(f,y))

function vcae_primal(f,indices,mass,rates,q)
    vcae_require(length(indices)==length(mass) && allunique(indices) &&
        all(i->1<=i<=f.T,indices),"invalid primal column indices")
    vcae_require(all(>=(0),mass) && sum(mass)==1,"nonnegative normalized population mixture fails")
    vcae_require(length(rates)==f.J && all(r->0<r<=1,rates),"invalid response rates")
    vcae_require(VCAE_Q.(f.A[:,indices])*mass==q.*rates[f.owner],"exact ordinary-choice marginals fail")
    vcae_require(VCAE_Q.(f.D[:,indices])*mass==rates,"exact response moments fail")
end

function vcae_outward(record,lowerfield,upperfield,lower,upper)
    vcae_require(VCAE_Q(Float64(record[lowerfield]))<=lower &&
        upper<=VCAE_Q(Float64(record[upperfield])),"reported Float64 endpoints are not outward bounds")
end

function vcae_point(f,raw,floor,fit,cap)
    c=fit["certificate"]
    q=vcae_q.(c["conditional_fitted_exact"])
    tau=vcae_q(c["response_rate_exact"])
    beta=vcae_q(c["unary_mix_fraction_exact"])
    vcae_require(tau==1-VCAE_Q(cap) && beta==min(VCAE_Q(1),(1-tau)/((f.J-1)*tau)),
        "exact declared cap arithmetic changed")
    vcae_require(length(q)==f.I && all(>=(0),q) && all(sum(q[g])==1 for g in f.groups),
        "conditional fit is not a product-simplex probability")
    indices=Int.(c["primal_indices"]); mass=vcae_q.(c["primal_masses_exact"])
    vcae_primal(f,indices,mass,fill(tau,f.J),q)
    lambda=raw-q
    vcae_require(lambda==vcae_q.(c["residual_patch_weights_exact"]),"residual dual weights differ")
    complete_support=vcae_complete_support(f,lambda)
    product_support=sum(maximum(lambda[g]) for g in f.groups)
    support=(1-beta)*complete_support+beta*product_support
    vcae_require(complete_support==vcae_q(c["complete_support_exact"]) &&
        product_support==vcae_q(c["product_support_exact"]) &&
        support==vcae_q(c["conditional_support_exact"]),"exhaustive Minkowski support differs")
    upper=sum(abs2,lambda)
    lower=max(floor,2dot(raw,lambda)-sum(abs2,lambda)-2support)
    vcae_require(floor==vcae_q(c["domain_floor_exact"]) && lower==vcae_q(c["squared_distance_lower_exact"]) &&
        upper==vcae_q(c["squared_distance_upper_exact"]) && 0<=lower<=upper,
        "exact squared conditional objective interval fails")
    vcae_require(upper-lower<=VCAE_Q(1e-8),"unresolved exact point interval")
    vcae_outward(fit,"squared_distance_lower","squared_distance_upper",lower,upper)
    vcae_require(VCAE_Q(Float64(fit["objective_gap"]))>=upper-lower,"gap diagnostic rounds inward")
    vcae_require(Float64.(q)==Float64.(fit["conditional_fitted"]) &&
        Float64(tau)==Float64(fit["response_rate"]) && Float64(beta)==Float64(fit["unary_conditional_mix_fraction"]),
        "reported fitted probabilities/rate differ from certificate")
    vcae_require(fit["point_bounds_exact_rational"] && fit["outward_rounded_certificate"] &&
        fit["all_columns_priced"] && fit["common_response_is_additional_restriction"] &&
        Int(fit["full_type_count"])==f.T && fit["new_p_values"]===nothing,"point scope/registry changed")
    (;q,tau,beta,indices,mass,lower,upper)
end

function vcae_floor_lp(f,q,record,policy)
    vcae_require(record["response_policy"]==policy,"LP response-policy mismatch")
    indices=Int.(record["primal_certificate_columns"])
    mass=vcae_q.(record["primal_certificate_mass_exact"])
    rates=vcae_q.(record["primal_response_rates_exact"])
    vcae_primal(f,indices,mass,rates,q)
    policy=="common" && vcae_require(all(==(first(rates)),rates),"common-rate LP witness has varying rates")
    y=vcae_q.(record["dual_certificate_patch_weights_exact"])
    vcae_require(length(y)==f.I && dot(q,y)==1,"LP dual normalization fails")
    coefficients=[dot(q[g],y[g]) for g in f.groups]
    vcae_require(coefficients==vcae_q.(record["dual_budget_coefficients_exact"]),"LP budget dual coefficients differ")
    policy=="variable" && vcae_require(all(>=(0),coefficients),"variable-rate dual budget coefficient negative")
    gamma=vcae_full_support(f,y)
    vcae_require(gamma==vcae_q(record["dual_support_bound_exact"]),"LP full-family dual support differs")
    lower=minimum(rates); upper=min(VCAE_Q(1),gamma)
    vcae_require(lower==vcae_q(record["response_lower_exact"]) && upper==vcae_q(record["response_upper_exact"]) &&
        0<lower<=upper<=1,"LP response interval fails")
    elo,ehi=1-upper,1-lower
    vcae_require(elo==vcae_q(record["epsilon_lower_exact"]) && ehi==vcae_q(record["epsilon_upper_exact"]) &&
        upper-lower==vcae_q(record["exact_interval_width"]) && upper-lower<=VCAE_Q(1e-7),
        "LP allowance interval direction/gap fails")
    vcae_outward(record,"response_lower","response_upper",lower,upper)
    vcae_outward(record,"epsilon_lower","epsilon_upper",elo,ehi)
    vcae_require(Int(record["all_inner_columns_priced_exactly"])==f.T && record["exact_rational_certificate"] &&
        record["common_response_is_additional_restriction"]==(policy=="common") &&
        !record["population_absence_share_identified"] && record["new_p_values"]===nothing,"LP scope changed")
    (;lower,upper,elo,ehi,gap=upper-lower)
end

function vcae_window(report,old,cache,reference)
    cell=Int(report["cell"])
    vcae_require(cell==old["cell"]==cache["cell"] && report["years"]==old["years"]==cache["years"] &&
        report["annual_n"]==old["annual_n"] && report["annual_n_sum"]==sum(Int.(old["annual_n"])),
        "window calendar/sample changed")
    vcae_require(report["status"]=="completed_actual_empty_inner_descriptive","incomplete window")
    X=reduce(vcat,permutedims(Int8.(r)) for r in old["sign_matrix"])
    vcae_require(vcae_equal(report["sign_matrix"],old["sign_matrix"]),"patch geometry changed")
    groups=KSReplication._patch_groups(X); J=length(groups)
    P=reshape(Float64.(old["saved_prices"]),J,3)
    vcae_require(Float64.(report["saved_normalized_prices"])==vec(P),"normalized prices changed")
    raw_float=Float64.(report["raw_affine_point_unchanged"])
    vcae_require(raw_float==Float64.(cache["raw_affine_point"]),"cached raw target changed")
    raw=VCAE_Q.(raw_float)
    qfloor=vcae_q.(report["probability_domain_floor"]["simplex_target_exact"])
    vcae_require(all(>=(0),qfloor) && length(qfloor)==size(X,1),"invalid simplex-floor diagnostic")
    for group in groups
        vcae_require(sum(qfloor[group])==1,"simplex floor block does not add up")
        active=filter(i->qfloor[i]>0,group)
        theta=sum(raw[i]-qfloor[i] for i in active)/length(active)
        vcae_require(all(i->qfloor[i]>0 ? raw[i]-qfloor[i]==theta : raw[i]<=theta,group),
            "independent exact simplex-floor KKT fails")
    end
    floor=sum(abs2,raw-qfloor)
    vcae_require(floor==vcae_q(report["probability_domain_floor"]["squared_distance_exact"]),"domain floor differs")
    Ar=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
    Aw=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
    warp_columns=Set(Tuple(view(Aw,:,h)) for h in axes(Aw,2))
    vcae_require(all(Tuple(view(Ar,:,h)) in warp_columns for h in axes(Ar,2)),"complete RUM/WARP not nested")
    family=vcae_family(Aw,groups); rumfamily=vcae_family(Ar,groups)
    vcae_require(report["types"]["complete_RUM"]==size(Ar,2) && report["types"]["complete_WARP"]==size(Aw,2) &&
        report["types"]["actual_one_budget_responders"]==size(X,1) && report["types"]["actual_all_null"]==1 &&
        report["types"]["actual_CMU_inner_total"]==family.T,"inner type count differs")
    witnesses=report["actual_empty_single_budget_witnesses"]
    vcae_require(length(witnesses)==family.I && Int.([w["row"] for w in witnesses])==collect(1:family.I),
        "responder witness registry incomplete")
    for witness in witnesses
        vcae_require(cmue_verify_record(P,X,witness),"serialized responder witness fails")
    end
    vcae_require(cmue_verify_record(P,X,report["actual_empty_all_null_witness"]) &&
        report["witness_geometry_key"]==cmue_geometry_key(P,X),"serialized all-null witness fails")
    rum=vcae_point(rumfamily,raw,floor,report["fixed_RUM"],0.)
    oldrum=reference["comparison"]["policies"]["raw_affine"]["models"]["RUM"]["exact_full_projection_certificate"]
    vcae_require(rum.lower<=vcae_q(oldrum["upper_bound_exact"]) && vcae_q(oldrum["lower_bound_exact"])<=rum.upper,
        "new exact fixed-RUM interval does not overlap prior exact certificate")
    lps=report["allowance_to_domain_floor"]
    common=vcae_floor_lp(family,qfloor,lps["common_response"],"common")
    variable=vcae_floor_lp(family,qfloor,lps["variable_response"],"variable")
    vcae_require(variable.elo<=common.ehi,"common/variable response inclusion fails")
    vcae_require(length(report["profiles"])==length(VCAE_CAPS),"cap grid incomplete")
    prior=nothing; maxgap=0.; inherited=0
    for (k,profile) in enumerate(report["profiles"])
        cap=VCAE_CAPS[k]
        vcae_require(Float64(profile["empty_allowance_cap"])==cap,"cap grid changed")
        fit=profile["CMU_actual_empty_inner"]
        point=vcae_point(family,raw,floor,fit,cap)
        maxgap=max(maxgap,Float64(point.upper-point.lower))
        vcae_require(point.lower<=rum.upper,"fixed RUM/inner nesting violates exact intervals")
        if prior!==nothing
            vcae_require(point.lower<=prior.point.upper,"cap nesting violates exact intervals")
        end
        if fit["upper_inherited_by_global_all_null_mixture"]
            inherited+=1
            vcae_require(prior!==nothing && Float64(fit["inherited_fit_cap"])==prior.cap && point.q==prior.point.q &&
                point.upper==prior.point.upper && point.upper<vcae_q(fit["solver_proposal_upper_exact"]),
                "global-null inheritance bookkeeping fails")
            before=zeros(VCAE_Q,family.T); before[prior.point.indices]=prior.point.mass
            expected=(point.tau/prior.point.tau)*before
            expected[family.null]+=1-point.tau/prior.point.tau
            actual=zeros(VCAE_Q,family.T); actual[point.indices]=point.mass
            vcae_require(expected==actual,"inherited population is not an exact global-null mixture")
        end
        if cap==0
            oldwarp=reference["comparison"]["policies"]["raw_affine"]["models"]["WARP"]["exact_full_projection_certificate"]
            vcae_require(point.lower<=vcae_q(oldwarp["upper_bound_exact"]) &&
                vcae_q(oldwarp["lower_bound_exact"])<=point.upper,"zero-cap exact WARP baseline changed")
        end
        expectedlow=max(0.,Float64(report["fixed_RUM"]["squared_distance_lower"])-Float64(fit["squared_distance_upper"]))
        expectedhigh=max(0.,Float64(report["fixed_RUM"]["squared_distance_upper"])-Float64(fit["squared_distance_lower"]))
        vcae_require(Float64(profile["fixed_RUM_gain_lower"])==expectedlow &&
            Float64(profile["fixed_RUM_gain_upper"])==expectedhigh &&
            profile["numerically_resolved_gain"]==(expectedlow>1e-10),"gain interval/classification differs")
        prior=(;cap,point)
    end
    vcae_require(report["new_p_values"]===nothing && !report["observed_empty_choice_frequencies_available"] &&
        !report["actual_population_empty_share_estimated"],"empirical scope changed")
    Dict("cell"=>cell,"serialized_witnesses_verified"=>family.I+1,"exact_point_profiles_verified"=>length(VCAE_CAPS)+1,
        "exact_LP_policies_verified"=>2,"largest_exact_QP_interval_width"=>maxgap,
        "common_floor_allowance"=>[Float64(common.elo),Float64(common.ehi)],
        "variable_floor_allowance"=>[Float64(variable.elo),Float64(variable.ehi)],
        "largest_exact_LP_interval_width"=>Float64(max(common.gap,variable.gap)),
        "global_all_null_inheritance_records_verified"=>inherited)
end

function vcae_joint_norm(summary,checkpoints)
    root=@__DIR__; cachedir=joinpath(root,"results/wilfried_joint_gap_20261010_v1")
    old=vcae_read(joinpath(cachedir,"summary.json")); protocol=old["protocol"]
    years=Int.(protocol["calendar_years"]); annual_n=Int.(protocol["annual_n"])
    B=Int(protocol["B"]); seed=Int(protocol["seed"]); N=sum(annual_n); rate2=N/3
    vcae_require(years==collect(1975:1999) && N==46391 && B==1000 && seed==13732 &&
        protocol["N_unique"]==N,"canonical raw-bootstrap plan changed")
    seeds=rand(MersenneTwister(seed),UInt64,length(years),B)
    vcae_require(bytes2hex(sha256(JSON3.write(seeds)))==protocol["seed_grid_sha256"],"canonical seed-grid hash fails")
    digests=Matrix{String}(undef,length(years),B)
    for (j,year) in enumerate(years),b in 1:B
        rng=MersenneTwister(seeds[j,b]); counts=zeros(Int,annual_n[j])
        for draw in 1:annual_n[j]
            counts[rand(rng,1:annual_n[j])]+=1
        end
        vcae_require(sum(counts)==annual_n[j],"canonical household frequency total fails")
        digests[j,b]=bytes2hex(sha256(JSON3.write(counts)))
    end
    vcae_require(bytes2hex(sha256(JSON3.write(digests)))==protocol["count_digest_matrix_sha256"],
        "canonical count-digest matrix fails")
    norms=zeros(B); maxadding=0.
    for cell in 1:19
        report=checkpoints[cell]
        cache=vcae_read(joinpath(cachedir,"raw_bootstrap_cell_$(lpad(cell,2,'0')).json"))
        rows=[findfirst(==(y),years) for y in Int.(report["years"])]
        vcae_require(Int.(cache["canonical_year_seed_rows"])==rows && cache["B"]==B &&
            cache["year_count_digests_sha256"]==bytes2hex(sha256(JSON3.write(digests[rows,:]))),
            "raw cache joint-calendar alignment fails")
        vcae_require(Int.(report["annual_n"])==annual_n[rows],"raw cache annual denominators changed")
        I=Int(cache["coordinates"]); stars=reshape(Float64.(cache["raw_affine_bootstrap"]),I,B)
        point=Float64.(cache["raw_affine_point"])
        vcae_require(point==Float64.(report["raw_affine_point_unchanged"]) && all(isfinite,stars) &&
            bytes2hex(sha256(JSON3.write(stars)))==cache["raw_affine_bootstrap_sha256"],
            "cached raw estimator point/matrix hash fails")
        d=cache["refit_diagnostics"]
        vcae_require(d["both_CF_stages_refit"] && d["failed_or_replaced_draws"]==0 &&
            d["raw_affine_probability_policy"] && d["all_original_annual_n_retained"],"raw cache refit scope changed")
        weight=report["annual_n_sum"]/N
        vcae_require(weight==Float64(protocol["window_weights"][string(cell)]),"joint raw-bootstrap weight changed")
        X=reduce(vcat,permutedims(Int8.(r)) for r in report["sign_matrix"])
        groups=KSReplication._patch_groups(X)
        for group in groups
            maxadding=max(maxadding,maximum(abs.(vec(sum(stars[group,:];dims=1)).-1)))
        end
        for b in 1:B
            norms[b]+=rate2*weight*sum(abs2,view(stars,:,b).-point)
        end
    end
    vcae_require(maxadding<=1e-10,"raw bootstrap probability adding-up fails")
    criticals=Dict{String,Any}()
    sorted=sort(norms)
    for alpha in (.05,.10)
        order=ceil(Int,(1-alpha)*(B+1))
        criticals[string(alpha)]=Dict("order_statistic_index"=>order,"critical_value"=>sorted[order])
    end
    profiles=Dict{String,Any}[]
    for profile in summary["weighted_profiles"]
        statlo=rate2*Float64(profile["fixed_RUM_gain_lower"])
        stathi=rate2*Float64(profile["fixed_RUM_gain_upper"])
        tail=count(>=(statlo),norms)
        push!(profiles,Dict("empty_allowance_cap"=>profile["empty_allowance_cap"],
            "observed_statistic_lower"=>statlo,"observed_statistic_upper"=>stathi,
            "error_norm_exceedances"=>tail,"conditional_bootstrap_error_envelope_tail_mc"=>(tail+1)/(B+1),
            "reject_under_conditional_95pct_envelope"=>statlo>criticals["0.05"]["critical_value"],
            "reject_under_conditional_90pct_envelope"=>statlo>criticals["0.1"]["critical_value"]))
    end
    Dict("method"=>"uniform cap-free joint raw-estimator error-norm envelope for nested convex sets",
        "norm"=>"E_b=(46391/3)*sum_w (N_window_w/46391)*||raw_star_w-raw_point_w||^2",
        "finite_deterministic_bound"=>"If the two nested sets have equal population projection distance, their sample squared-distance gain is at most ||sample-population||^2; one joint error event bounds every cap simultaneously.",
        "sampling_calibration"=>"Conditional on a valid joint raw-CF bootstrap CLT and negligible bias; these assumptions remain UNVERIFIED.",
        "not_a_sharp_Chen_Fang_test"=>true,"not_an_ordinary_gain_bootstrap"=>true,
        "no_Bonferroni_or_Holm"=>true,"no_old_model_p_values_reused"=>true,
        "unobserved_empty_rate_still_a_sensitivity_parameter"=>true,"not_a_test_of_general_empty_CMU_adequacy"=>true,
        "canonical_seed_count_and_raw_matrix_hashes_verified"=>true,"B"=>B,"N_unique"=>N,
        "rate_squared"=>rate2,"maximum_bootstrap_blocksum_error"=>maxadding,"critical_values"=>criticals,
        "error_norm_sha256"=>bytes2hex(sha256(JSON3.write(norms))),"profiles"=>profiles,
        "statistical_significance_claim_unconditional"=>false)
end

function verify_cmu_actual_empty_comparison(directory; joint_error_envelope=false)
    directory=abspath(directory); root=@__DIR__
    summary=vcae_read(joinpath(directory,"summary.json"))
    frozen=vcae_read(joinpath(directory,"frozen_protocol.json"))
    vcae_require(summary["status"]=="completed_descriptive_actual_empty_CMU_inner_comparison" &&
        summary["source_integrity_unchanged"] && vcae_equal(summary["protocol"],frozen["protocol"]),"completed protocol changed")
    vcae_require(Int.(summary["protocol"]["windows"])==collect(1:19) &&
        Float64.(summary["protocol"]["empty_allowance_caps"])==VCAE_CAPS,"frozen windows/caps changed")
    for (before,after) in (("protected_sha256_before","protected_sha256_after"),
        ("implementation_sha256_before","implementation_sha256_after"))
        vcae_require(vcae_equal(summary[before],summary[after]) && vcae_equal(summary[before],frozen[before]),
            "saved before/after/protocol hashes differ")
        for (name,digest) in pairs(summary[after])
            vcae_require(vcae_sha(normpath(joinpath(root,String(name))))==String(digest),"current protected file changed: $name")
        end
    end
    refs=vcae_read(joinpath(root,"results/wilfried_normalized_fit_robustness_20261010_v1.json"))
    reference=Dict(Int(w["cell"])=>w for w in refs["windows"])
    checkpoints=Dict{Int,Any}(); audits=Dict{String,Any}[]
    vcae_require(length(summary["windows"])==19,"summary windows incomplete")
    for saved in summary["windows"]
        cell=Int(saved["cell"]); filename="cell_$(lpad(cell,2,'0')).json"
        report=vcae_read(joinpath(directory,filename))
        vcae_require(vcae_equal(saved,report) && !haskey(checkpoints,cell),"checkpoint/summary mismatch or duplicated cell")
        old=vcae_read(joinpath(root,"results/wilfried_windows_20261009_v2",filename))
        cache=vcae_read(joinpath(root,"results/wilfried_joint_gap_20261010_v1","raw_bootstrap_"*filename))
        try
            push!(audits,vcae_window(report,old,cache,reference[cell]))
        catch err
            error("cell $cell: "*sprint(showerror,err))
        end
        checkpoints[cell]=report
    end
    vcae_require(sort(collect(keys(checkpoints)))==collect(1:19),"all19 cells not verified")
    vcae_require(length(summary["weighted_profiles"])==length(VCAE_CAPS),"weighted profiles incomplete")
    for (k,profile) in enumerate(summary["weighted_profiles"])
        vcae_require(Float64(profile["empty_allowance_cap"])==VCAE_CAPS[k],"weighted cap grid differs")
        for (field,expected) in (
            ("fixed_RUM_distance",sum(checkpoints[c]["annual_n_sum"]/46391*checkpoints[c]["fixed_RUM"]["squared_distance_upper"] for c in 1:19)),
            ("actual_empty_CMU_inner_distance",sum(checkpoints[c]["annual_n_sum"]/46391*checkpoints[c]["profiles"][k]["CMU_actual_empty_inner"]["squared_distance_upper"] for c in 1:19)),
            ("fixed_RUM_gain_lower",sum(checkpoints[c]["annual_n_sum"]/46391*checkpoints[c]["profiles"][k]["fixed_RUM_gain_lower"] for c in 1:19)),
            ("fixed_RUM_gain_upper",sum(checkpoints[c]["annual_n_sum"]/46391*checkpoints[c]["profiles"][k]["fixed_RUM_gain_upper"] for c in 1:19)))
            vcae_require(abs(Float64(profile[field])-expected)<=1e-14,"weighted $field differs")
        end
        reduction=(profile["fixed_RUM_distance"]-profile["actual_empty_CMU_inner_distance"])/profile["fixed_RUM_distance"]
        vcae_require(abs(Float64(profile["distance_reduction_fraction"])-reduction)<=1e-14 &&
            Int.(profile["numerically_positive_gain_windows"])==[c for c in 1:19 if checkpoints[c]["profiles"][k]["numerically_resolved_gain"]],
            "weighted reduction/positive-window registry differs")
    end
    Dict("status"=>"all19_actual_empty_inner_certificates_verified","directory"=>directory,
        "verifier_sha256"=>vcae_sha(@__FILE__),"exact_point_certificates_verified"=>sum(a["exact_point_profiles_verified"] for a in audits),
        "exact_LP_certificates_verified"=>sum(a["exact_LP_policies_verified"] for a in audits),
        "serialized_actual_empty_witnesses_verified"=>sum(a["serialized_witnesses_verified"] for a in audits),
        "maximum_exact_QP_interval_width"=>maximum(a["largest_exact_QP_interval_width"] for a in audits),
        "maximum_exact_LP_interval_width"=>maximum(a["largest_exact_LP_interval_width"] for a in audits),
        "protected_and_implementation_hashes_unchanged"=>true,"summary_checkpoints_and_weighted_metrics_verified"=>true,
        "exact_scope"=>"Saved rational population mixtures, response/marginal equations, complete/singleton/null LP support, Minkowski residual-dual point intervals, outward floating endpoints and baseline overlap; witness algebra assumes the separately audited whole-budget CMU theorem.",
        "statistical_significance_established"=>false,"audits"=>audits,
        "joint_error_envelope"=>joint_error_envelope ? vcae_joint_norm(summary,checkpoints) : nothing)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (1,2) || error("Usage: julia --project=. verify_cmu_actual_empty_comparison.jl REPORT_DIRECTORY [--joint-error-envelope]")
    length(ARGS)==1 || ARGS[2]=="--joint-error-envelope" || error("unknown verifier option")
    println(JSON3.write(verify_cmu_actual_empty_comparison(ARGS[1];joint_error_envelope=length(ARGS)==2)))
end
