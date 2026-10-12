# Independent read-only audit of the frozen finite-anchor Gaussian study.
# This verifier does not include the calibration driver or reuse its scoring,
# decision, threshold, population-mixture or summary functions.
using KSReplication, JSON3, Random, LinearAlgebra, Statistics, Distributions, SHA

const CCPV_Q=Rational{BigInt}
ccpv_q(x)=parse(CCPV_Q,String(x))
ccpv_sha(path)=bytes2hex(sha256(read(path)))
ccpv_require(ok,message)=ok || error(message)
ccpv_read(path)=JSON3.read(read(path,String))

function ccpv_mixture(A,indices,numerators,denominator)
    ids=Int.(indices);counts=parse.(BigInt,String.(numerators));D=parse(BigInt,String(denominator))
    ccpv_require(length(ids)==length(counts)&&!isempty(ids),"mixture support dimensions")
    ccpv_require(length(unique(ids))==length(ids)&&all(>(0),counts)&&sum(counts)==D,"positive normalized mixture")
    ccpv_require(all(h->h in axes(A,2),ids),"mixture column index")
    masses=counts.//D
    q=CCPV_Q.(A[:,ids])*masses
    (;ids,masses,q)
end

"""Recompute the whole-budget shifted-log majority arithmetic independently.

The global majority-core bridge remains an analytic premise, not a new Lean
theorem. No finite quantity grid substitutes for the boundary/determinant audit.
"""
function ccpv_utility(P,X,record)
    J=size(P,1);PQ=CCPV_Q.(P)
    a=[ccpv_q(record["utilities"][v]["a"][k]) for v in 1:3,k in 1:3]
    d=[ccpv_q(record["utilities"][v]["d"][k]) for v in 1:3,k in 1:3]
    ccpv_require(all(>(0),a)&&all(>(0),d),"positive shifted-log primitives")
    chosen=Int(record["chosen_budget"]);anchor=ccpv_q.(record["anchor"])
    recordedP=[ccpv_q(record["prices_exact_binary_rational"][j][k]) for j in 1:J,k in 1:3]
    ccpv_require(PQ==recordedP,"utility price geometry")
    for j in 1:J
        cert=record["budget_certificates"][j]
        ccpv_require(Int(cert["budget"])==j,"utility budget index")
        upper=maximum(a[v,k]/(1//3+PQ[j,k]*d[v,k]) for v in 1:3,k in 1:3)
        for k in 1:3
            margin=minimum(a[v,k]/(PQ[j,k]*d[v,k]) for v in 1:3)-upper
            ccpv_require(margin>0&&margin==ccpv_q(cert["boundary_entry_margins"][k]),"whole-boundary entry certificate")
        end
        for v in 1:3
            lambda=sum(a[v,:])/(1+dot(PQ[j,:],d[v,:]))
            ideal=[a[v,k]/(lambda*PQ[j,k])-d[v,k] for k in 1:3]
            ccpv_require(all(>(0),ideal)&&dot(PQ[j,:],ideal)==1,"interior voter optimum")
            ccpv_require(ideal==ccpv_q.(cert["voter_ideals"][v])&&lambda==ccpv_q(cert["voter_multipliers"][v]),"stored voter certificates")
            j==chosen && ccpv_require(ideal==anchor,"common chosen-budget optimum")
            others=[i for i in 1:3 if i!=v]
            tangent=[a[i,k]/(ideal[k]+d[i,k])/PQ[j,k]-a[i,3]/(ideal[3]+d[i,3])/PQ[j,3] for i in others,k in 1:2]
            determinant=tangent[1,1]*tangent[2,2]-tangent[1,2]*tangent[2,1]
            ccpv_require(determinant==ccpv_q(cert["candidate_pair_tangent_determinants"][v]),"stored tangent determinant")
            j!=chosen && ccpv_require(determinant!=0,"no majority core at any voter optimum on null budget")
        end
    end
    ccpv_require(Int.(record["chosen_budgets"])==(chosen==0 ? Int[] : [chosen]),"chosen budgets")
    ccpv_require(Int.(record["empty_budgets"])==[j for j in 1:J if j!=chosen],"empty budgets")
    row=Int(record["row"])
    if row==0
        ccpv_require(chosen==0&&isempty(anchor),"true all-null model")
    else
        signs=Int8.(record["patch_signs"])
        ccpv_require(signs==vec(X[row,:])&&only(findall(==(0),signs))==chosen,"responder patch")
        costs=PQ*anchor
        ccpv_require(all(>(0),anchor)&&costs[chosen]==1,"strict responder anchor")
        ccpv_require(all(j->j==chosen||signs[j]*(costs[j]-1)>0,1:J),"responder cross-budget patch signs")
    end
    true
end

function ccpv_geometry(record,rate,source_root)
    cell=Int(record["cell"])
    source=ccpv_read(joinpath(source_root,"results/wilfried_windows_20261009_v2/summary.json"))
    case=only(c for c in source["cases"] if Int(c["cell"])==cell)
    X=reduce(vcat,permutedims(Int8.(row)) for row in case["sign_matrix"])
    groups=KSReplication._patch_groups(X);I,J=size(X)
    ccpv_require(J==7&&sort(reduce(vcat,groups))==collect(1:I),"seven-budget coordinate partition")
    A=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
    W=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
    ccpv_require(all(sum(A[g,h])==1 for g in groups,h in axes(A,2)),"complete RUM one patch per budget")
    ccpv_require(all(sum(W[g,h])==1 for g in groups,h in axes(W,2)),"complete WARP one patch per budget")
    Rset=Set(Tuple(view(A,:,h)) for h in axes(A,2));Wset=Set(Tuple(view(W,:,h)) for h in axes(W,2))
    ccpv_require(issubset(Rset,Wset),"RUM subset WARP")
    original=record["original_geometry_design"]
    ccpv_require(Int(original["RUM_types"])==size(A,2)&&Int(original["WARP_types"])==size(W,2),"regenerated type counts")
    centroid=ccpv_mixture(A,collect(axes(A,2)),original["centroid_positive_weight_numerators"],original["centroid_weight_denominator"])
    ccpv_require(centroid.q==ccpv_q.(original["centroid_exact"]),"centroid exact moments")
    firstexcluded=findfirst(h->!(Tuple(view(W,:,h)) in Rset),axes(W,2))
    ccpv_require(firstexcluded==Int(original["excluded_WARP_column_index"]),"canonical WARP-not-RUM vertex")
    w=Int.(original["w"]);ccpv_require(w==W[:,firstexcluded],"excluded WARP selector")
    rscore=[dot(w,Int.(A[:,h])) for h in axes(A,2)];support=maximum(rscore)
    ccpv_require(rscore==Int.(original["all_RUM_column_scores_exact"])&&support<J,"RUM exposed support")
    face=ccpv_mixture(A,original["face_column_indices"],original["face_positive_weight_numerators"],original["face_weight_denominator"])
    ccpv_require(face.ids==findall(==(support),rscore)&&face.q==ccpv_q.(original["q0_exact"]),"original exposed-face mixture")
    ccpv_require(dot(CCPV_Q.(w),face.q)==support,"original exposed-face equality")
    common=record["common_R_W_face"];s=Int.(common["s"]);selected=Int.(common["selected_rows"])
    ccpv_require(length(selected)==J&&all(selected[j] in groups[j] for j in 1:J),"common selector row partition")
    expected=zeros(Int,I);expected[selected].=1
    ccpv_require(s==expected&&!(Tuple(s) in Wset),"common selector outside complete WARP")
    sr=[dot(s,Int.(A[:,h])) for h in axes(A,2)];sw=[dot(s,Int.(W[:,h])) for h in axes(W,2)]
    common_support=maximum(sr)
    ccpv_require(sr==Int.(common["R_scores_exact"])&&sw==Int.(common["W_scores_exact"]),"full common support pricing")
    ccpv_require(common_support==maximum(sw)<J,"common R/W exposed support")
    commonface=ccpv_mixture(A,common["R_face_indices"],common["R_face_mass_numerators"],common["R_face_mass_denominator"])
    ccpv_require(commonface.ids==findall(==(common_support),sr)&&commonface.q==ccpv_q.(common["q_empty0_exact"]),"common exposed-face exact mixture")
    canonical=nothing;candidates=0
    for reverse_rows in Iterators.product(reverse(groups)...)
        rows=reverse(collect(reverse_rows));candidate=zeros(Int,I);candidate[rows].=1;candidates+=1
        Tuple(candidate) in Wset && continue
        hR=maximum(dot(candidate,Int.(A[:,h])) for h in axes(A,2))
        hW=maximum(dot(candidate,Int.(W[:,h])) for h in axes(W,2))
        if hR==hW<J
            canonical=candidate;break
        end
    end
    ccpv_require(canonical==s&&candidates==Int(common["Cartesian_candidates_examined"]),"first lexicographic common face")
    means=Dict("centroid_null"=>centroid.q,"WARP_exposed_face_null"=>face.q,"common_R_W_face_null"=>commonface.q)
    for (id,alt) in zip(("WARP_one_SE","WARP_two_SE"),original["alternatives"])
        t=ccpv_q(alt["mixture_exact"]);q=(1-t)*face.q+t*CCPV_Q.(w)
        ccpv_require(q==ccpv_q.(alt["point_exact"])&&dot(CCPV_Q.(w),q)>support,"original WARP alternative moments")
        means[id]=q
    end
    registry_path=joinpath(source_root,"results/cmu_actual_empty_comparison_20261011_v1","cell_$(lpad(cell,2,'0')).json")
    registry=ccpv_read(registry_path)
    registryX=reduce(vcat,permutedims(Int8.(row)) for row in registry["sign_matrix"])
    ccpv_require(registryX==X,"actual utility registry geometry")
    P=reshape(Float64.(registry["saved_normalized_prices"]),J,3)
    for row in selected
        witness=only(z for z in registry["actual_empty_single_budget_witnesses"] if Int(z["row"])==row)
        ccpv_utility(P,X,witness)
    end
    ccpv_utility(P,X,registry["actual_empty_all_null_witness"])
    for (id,alt) in zip(("actual_empty_cap10","actual_empty_cap20"),record["actual_empty_alternatives"])
        epsilon=CCPV_Q(Float64(alt["epsilon_cap"]));tau=1-epsilon
        beta=ccpv_q(alt["beta_down_exact"]);allowed=epsilon/((J-1)*tau);D=big(1)<<32
        ccpv_require(tau==ccpv_q(alt["response_rate_exact"])&&allowed==ccpv_q(alt["allowed_beta_exact"]),"consistent exact cap/response rates")
        ccpv_require(0<beta<=allowed&&beta==(floor(BigInt,allowed*D)//D),"fixed dyadic-down beta")
        q=(1-beta)*commonface.q+beta*CCPV_Q.(s)
        ccpv_require(q==ccpv_q.(alt["mean_exact"])&&dot(CCPV_Q.(s),q)>common_support,"actual-empty mean outside complete WARP")
        complete=ccpv_q.(alt["complete_population_masses_exact"]);single=ccpv_q(alt["population_mass_each_singleton_exact"]);null=ccpv_q(alt["all_null_population_mass_exact"])
        ccpv_require(Int.(alt["complete_RUM_columns"])==commonface.ids&&Int.(alt["singleton_rows"])==selected,"primitive population support")
        ccpv_require(complete==tau*(1-beta)*commonface.masses&&single==tau*beta,"primitive population masses")
        ccpv_require(all(>=(0),complete)&&single>=0&&null>=0&&sum(complete)+J*single+null==1,"primitive population simplex")
        ordinary=CCPV_Q.(A[:,commonface.ids])*complete+single*CCPV_Q.(s)
        ccpv_require(ordinary==tau*q&&all(sum(ordinary[g])==tau for g in groups),"exact conditional CMU marginalization")
        ccpv_require(ccpv_sha(registry_path)==alt["utility_registry_sha256"],"utility registry source hash")
        means[id]=q
    end
    for scenario in record["scenarios"]
        q=means[String(scenario["id"])]
        ccpv_require(q==ccpv_q.(scenario["mean_exact"]),"saved scenario mean")
        ccpv_require(CCPV_Q.(Float64.(q))==q&&all(sum(q[g])==1&&all(>=(0),q[g]) for g in groups),"exact probability simplex scenario")
    end
    dimensions=Int.(record["Gaussian_factor_dimensions"]);L=reshape(Float64.(record["Gaussian_factor"]),Tuple(dimensions))
    ccpv_require(size(L)==(I,I),"all-coordinate Gaussian factor")
    cache=ccpv_read(joinpath(source_root,"results/wilfried_joint_gap_20261010_v1","raw_bootstrap_cell_$(lpad(cell,2,'0')).json"))
    stars=reshape(Float64.(cache["raw_affine_bootstrap"]),Int(cache["coordinates"]),Int(cache["B"]))
    centered=rate*(stars.-mean(stars;dims=2));covariance=centered*centered'/(size(stars,2)-1)
    covariance_error=maximum(abs,L*L'-covariance)
    ccpv_require(covariance_error<=1e-10*(1+maximum(abs,covariance)),"full raw-cache covariance reconstruction")
    (;cell,A,W,groups,L,means,covariance_error,utility_witnesses=length(selected)+1)
end

function ccpv_interval(k,m)
    [k==0 ? 0. : quantile(Beta(k,m-k+1),.025),k==m ? 1. : quantile(Beta(k+1,m-k),.975)]
end

function ccpv_run(directory;save=false)
    BLAS.set_num_threads(1)
    directory=abspath(directory);source_root=@__DIR__;start=time()
    summary=ccpv_read(joinpath(directory,"summary.json"))
    protocol=summary["protocol"];initial=ccpv_read(joinpath(directory,"protocol_frozen_before_outcomes.json"))
    ccpv_require(protocol==initial["protocol"],"frozen protocol unchanged")
    cells=Int.(protocol["cells"]);grid=Float64.(protocol["c_methods"]);alphas=Float64.(protocol["alphas"])
    Mref=Int(protocol["reference_M_per_anchor"]);Mval=Int(protocol["validation_M_per_case"]);B=Int(protocol["B_fixed"]);rate=Float64(protocol["rate"])
    ccpv_require(cells==[7,12]&&grid==[1.,2.]&&alphas==[.05,.10]&&rate==sqrt(46391/3),"approved frozen specifications")
    probe=Bool(protocol["probe_only"])
    ccpv_require((Mref,Mval,B)==(probe ? (19,3,39) : (199,250,199)),"approved sample sizes")
    ccpv_require(!protocol["c_selection"]&&protocol["existing_UK_c1_primary_unchanged"],"no observed empirical c selection")
    ccpv_require(summary["protected_sha256_before"]==summary["protected_sha256_after"]==initial["protected_sha256_before"],"protected before/after hashes")
    for (file,sha) in pairs(summary["protected_sha256_after"])
        ccpv_require(ccpv_sha(joinpath(source_root,String(file)))==sha,"current protected hash: $file")
    end
    threshold_path=joinpath(directory,"reference_thresholds_frozen_before_validation.json")
    thresholds=ccpv_read(threshold_path)
    ccpv_require(ccpv_sha(threshold_path)==summary["reference_thresholds_sha256"],"frozen threshold file hash")
    geometry=ccpv_read(joinpath(directory,"geometry_and_exact_means.json"))
    designs=Dict(Int(r["cell"])=>ccpv_geometry(r,rate,source_root) for r in geometry)
    ccpv_require(sort(collect(keys(designs)))==cells,"geometry cell menu")
    allbanks=Dict{Tuple{Int,String},Any}();seen_seeds=Set{Int}();experiments=0;fits=0;pruned=0;fallbacks=0;utilities=0
    for d in values(designs);utilities+=d.utility_witnesses;end
    for cell in cells,phase in ("reference","validation")
        d=designs[cell];file=joinpath(directory,"$(phase)_cell_$(lpad(cell,2,'0'))_experiments.json")
        document=ccpv_read(file);banks=document["replicate_banks"]
        ccpv_require(Int(document["cell"])==cell&&document["phase"]==phase,"phase file identity")
        ccpv_require(length(banks)==(phase=="reference" ? Mref : Mval),"complete fixed-denominator bank")
        phase=="validation" && ccpv_require(stat(threshold_path).mtime<=stat(file).mtime,"threshold saved before validation file")
        allbanks[(cell,phase)]=banks
        for (rep,bank) in enumerate(banks)
            master=Int(protocol[phase=="reference" ? "reference_seed_master" : "validation_seed_master"])
            seed=master+1_000_003cell+10_007rep
            ccpv_require(Int(bank["replication"])==rep&&Int(bank["seed"])==seed&&!(seed in seen_seeds),"independent phase/replication seed")
            push!(seen_seeds,seed);rng=MersenneTwister(seed)
            outer=randn(rng,size(d.L,2));inner=randn(rng,size(d.L,2),B)
            latent=bank["latent_draws"]
            ccpv_require(outer==Float64.(latent["outer_standard_Gaussian"])&&vec(inner)==Float64.(latent["inner_standard_Gaussian"]),"every latent Gaussian draw regenerated")
            ccpv_require(Int.(latent["inner_dimensions"])==[size(d.L,2),B],"latent draw dimensions")
            ccpv_require(bytes2hex(sha256(JSON3.write(latent)))==bank["latent_draws_sha256"],"latent draw byte digest")
            ccpv_require(Int(bank["B_fixed"])==B&&bank["fresh_independent_outer_and_inner"],"fixed bootstrap denominator")
            expected_menu=phase=="reference" ? String.(protocol["null_menu"]) : vcat(String.(protocol["null_menu"]),String.(protocol["alternative_menu"]))
            ccpv_require([String(e["scenario"]) for e in bank["experiments"]]==expected_menu,"predeclared scenario menu")
            for e in bank["experiments"]
                experiments+=1;id=String(e["scenario"]);mean=Float64.(d.means[id]);point=mean.+(d.L*outer)/rate
                ccpv_require(point==Float64.(e["point_float"])&&e["fresh_projector_initialized_per_experiment"],"scenario point/fresh private model")
                result=e["whole_procedure_trace"];lo=Float64(result["observed_statistic_lower"]);hi=Float64(result["observed_statistic_upper"])
                base_lo=ccpv_q(result["base_exact_lower"]);base_hi=ccpv_q(result["base_exact_upper"])
                ccpv_require(0<=base_lo<=base_hi&&CCPV_Q(lo)<=CCPV_Q(rate)^2*base_lo<=CCPV_Q(rate)^2*base_hi<=CCPV_Q(hi),"exact observed outward interval")
                totalinner=0
                for (k,row) in enumerate(result["results"])
                    ccpv_require(Float64(row["c"])==grid[k]&&Float64(row["h"])==grid[k]/sqrt(rate)&&Int(row["B_fixed"])==B,"fixed numerical derivative method")
                    N=Int(row["inner_targets_evaluated"]);lower=Float64.(row["derivative_lower"]);upper=Float64.(row["derivative_upper"]);middle=Float64.(row["derivative_midpoint"])
                    ccpv_require(0<=N<=B&&length(lower)==length(upper)==length(middle)==N,"saved derivative trace completeness")
                    ccpv_require(all(isfinite,lower)&&all(isfinite,upper)&&all(lower.<=middle.<=upper),"signed derivative intervals")
                    tail=count(>=(lo),upper);ccpv_require(tail==Int(row["upper_tail_exceedances_evaluated"]),"all tail comparisons including ties")
                    ccpv_require(Int(row["inner_targets_decision_pruned"])==B-N,"fixed denominator after pruning")
                    if N<B
                        limit=maximum(floor(Int,a*(B+1)+1e-10)-1 for a in alphas)
                        ccpv_require(lo<=0 ? N==0 : tail>limit,"decision-only nonrejection pruning")
                    end
                    for (ai,decision) in enumerate(row["decisions"])
                        a=alphas[ai];ccpv_require(Float64(decision["alpha"])==a,"alpha method order")
                        expected=false
                        if N==B
                            order=ceil(Int,(1-a)*(B+1));critical=sort(upper)[order];central=sort(middle)[order];width=maximum(upper-lower)
                            positive=isfinite(central)&&central>max(0.,width)
                            expected=positive&&lo>critical&&(tail+1)/(B+1)<=a
                            ccpv_require(decision["critical_upper"]==critical&&decision["central_critical"]==central&&decision["maximum_derivative_width"]==width,"full-B quantile/width arithmetic")
                            ccpv_require(decision["critical_positive_resolved"]==positive&&decision["tail_MC_plus_one"]==(tail+1)/(B+1),"positive-critical and MC-plus-one gate")
                        else
                            ccpv_require(decision["critical_upper"]===nothing&&decision["central_critical"]===nothing&&decision["tail_MC_plus_one"]===nothing,"no partial bootstrap quantile")
                        end
                        ccpv_require(decision["reject"]==expected,"whole-procedure rejection decision")
                        ccpv_require(decision["tail_lower_bound_MC_plus_one"]==(tail+1)/(B+1)&&decision["tail_upper_bound_MC_plus_one"]==(tail+B-N+1)/(B+1),"partial tail bounds with original B")
                    end
                    original10=row["decisions"][2]["reject"]
                    score=original10 ? CCPV_Q(1)-(tail+1)//(B+1) : CCPV_Q(0)
                    saved=e["scores"][k]
                    ccpv_require(Float64(saved["c"])==grid[k]&&ccpv_q(saved["score_exact"])==score&&Float64(saved["score"])==Float64(score),"fixed truncated exact scalar score")
                    ccpv_require(saved["original_10pct_conditional_rejection"]==original10,"score source gate")
                    original10 && ccpv_require(N==B&&row["decisions"][2]["critical_positive_resolved"],"positive score complete calculation")
                    if phase=="validation"
                        method=e["calibrated_decisions"][k]
                        ccpv_require(Float64(method["c"])==grid[k],"calibrated method identity")
                        for (ai,decision) in enumerate(method["decisions"])
                            a=alphas[ai];t=thresholds["cells"][string(cell)]["methods"][string(grid[k])]["alphas"][string(a)]
                            critical=t["critical_score_exact"]===nothing ? nothing : ccpv_q(t["critical_score_exact"])
                            ccpv_require(decision["reject"]==(critical!==nothing&&score>critical),"strict independent-bank calibrated decision")
                            ccpv_require(ccpv_q(decision["score_exact"])==score&&decision["frozen_critical_score_exact"]==t["critical_score_exact"],"calibrated primitive scalars")
                        end
                    end
                    totalinner+=N;pruned+=B-N
                end
                diagnostic=e["exact_objective_bounds_diagnostics"]
                ccpv_require(Int(diagnostic["fits"])==totalinner+1&&Int(diagnostic["all_original_RUM_columns_priced"])==size(d.A,2),"full-family objective diagnostic bookkeeping")
                ccpv_require(0<=ccpv_q(diagnostic["maximum_Psi_interval_width_exact"])<=CCPV_Q(1e-8),"objective interval precision gate")
                fits+=Int(diagnostic["fits"]);fallbacks+=Int(diagnostic["fallbacks"])
            end
        end
    end
    rank_bounds=Dict{String,String}()
    for cell in cells,k in eachindex(grid),a in alphas
        t=thresholds["cells"][string(cell)]["methods"][string(grid[k])]["alphas"][string(a)]
        criticals=CCPV_Q[];order=ceil(Int,(1-a)*(Mref+1))
        for id in String.(protocol["null_menu"])
            scores=sort([ccpv_q(only(e for e in bank["experiments"] if e["scenario"]==id)["scores"][k]["score_exact"]) for bank in allbanks[(cell,"reference")]])
            anchor=t["null_anchor_thresholds"][id]
            ccpv_require(Int(anchor["order_statistic_index"])==order&&ccpv_q(anchor["critical_score_exact"])==scores[order],"independent exact reference order statistic")
            push!(criticals,scores[order])
        end
        ccpv_require(!t["infinite_threshold"]&&ccpv_q(t["critical_score_exact"])==maximum(criticals),"maximum over all three null anchors")
        rank_bounds[string(a)]=string(CCPV_Q(Mref+1-order,Mref+1))
    end
    ccpv_require(length(summary["rows"])==56,"complete calibrated summary menu")
    for row in summary["rows"]
        cell=Int(row["cell"]);k=findfirst(==(Float64(row["c"])),grid);ai=findfirst(==(Float64(row["alpha"])),alphas)
        selected=[only(e for e in bank["experiments"] if e["scenario"]==row["scenario"]) for bank in allbanks[(cell,"validation")]]
        calibrated=count(e->e["calibrated_decisions"][k]["decisions"][ai]["reject"],selected)
        original=count(e->e["whole_procedure_trace"]["results"][k]["decisions"][ai]["reject"],selected)
        ccpv_require(Int(row["validation_M"])==Mval&&Int(row["calibrated_rejections"])==calibrated&&Int(row["original_unadjusted_rejections"])==original,"all summary counts")
        ccpv_require(row["calibrated_rejection_rate"]==calibrated/Mval&&row["original_unadjusted_rejection_rate"]==original/Mval,"summary rejection rates")
        ccpv_require(maximum(abs,Float64.(row["calibrated_CP95_conditional_on_frozen_reference_bank"])-ccpv_interval(calibrated,Mval))<=1e-14,"conditional calibrated CP interval")
        ccpv_require(maximum(abs,Float64.(row["original_unadjusted_CP95"])-ccpv_interval(original,Mval))<=1e-14,"original CP interval")
    end
    validation_evaluated=0;validation_pruned=0;validation_fallbacks=0
    for cell in cells,bank in allbanks[(cell,"validation")],e in bank["experiments"]
        validation_fallbacks+=Int(e["exact_objective_bounds_diagnostics"]["fallbacks"])
        for row in e["whole_procedure_trace"]["results"]
            validation_evaluated+=Int(row["inner_targets_evaluated"])
            validation_pruned+=Int(row["inner_targets_decision_pruned"])
        end
    end
    ccpv_require(summary["validation_inner_targets_evaluated"]==validation_evaluated&&summary["validation_targets_decision_pruned"]==validation_pruned&&summary["validation_full_Clarabel_fallbacks"]==validation_fallbacks,"summary solver counts")
    expected_experiments=length(cells)*(3Mref+7Mval)
    ccpv_require(experiments==expected_experiments,"no omitted or replaced experiments")
    report=Dict("status"=>"independent_saved_bank_geometry_gate_threshold_count_hash_audit_passed",
        "verifier_sha256"=>ccpv_sha(@__FILE__),"summary_sha256"=>ccpv_sha(joinpath(directory,"summary.json")),
        "reference_thresholds_sha256"=>ccpv_sha(threshold_path),"probe_only"=>probe,"elapsed_seconds"=>time()-start,
        "cells"=>cells,"reference_M"=>Mref,"validation_M"=>Mval,"B_fixed"=>B,
        "whole_procedure_experiments_verified"=>experiments,"latent_outer_inner_banks_regenerated"=>length(seen_seeds),
        "bounded_fit_calls_bookkeeping_verified"=>fits,"decision_pruned_inner_targets_verified"=>pruned,
        "full_Clarabel_fallbacks_reported"=>fallbacks,"actual_empty_utility_witnesses_recomputed_exactly"=>utilities,
        "independent_primitive_population_mass_and_conditional_moment_checks"=>true,
        "all_reference_order_statistics_maximum_anchors_and_strict_ties_verified"=>true,
        "all_scalar_rejection_counts_and_conditional_CP_intervals_recomputed"=>true,
        "all_protected_sources_and_frozen_empirical_results_unchanged"=>true,
        "rank_size_bound_exact_per_method_finite_anchor_menu"=>rank_bounds,
        "scope"=>["Independent arithmetic and full saved-seed/trace/decision audit, not a duplicate simulation campaign",
            "The saved trace does not contain every individual QP primal/dual certificate; no independent all-target KKT reproof is claimed here",
            "Whole-budget majority-core interpretation depends on the existing analytic bridge, not a new Lean theorem",
            "Rank size bounds are unconditional over reference-and-test randomness, in the fixed known-Gaussian experiment and finite three-anchor menu only",
            "Validation CP intervals are conditional on this frozen reference bank and fixed estimated covariance",
            "Neither uniform continuous-null UK-CF calibration nor an optimal-c claim follows"],
        "geometry"=>[Dict("cell"=>cell,"RUM_types"=>size(designs[cell].A,2),"WARP_types"=>size(designs[cell].W,2),"covariance_max_abs_error"=>designs[cell].covariance_error) for cell in cells])
    if save
        output=joinpath(directory,"independent_audit.json")
        ispath(output)&&error("refusing to overwrite $output")
        open(output,"w") do io;JSON3.write(io,report);end
    end
    println(JSON3.write(report));report
end

if abspath(PROGRAM_FILE)==@__FILE__
    1<=length(ARGS)<=2||error("usage: verify_cmu_calibrated_power.jl RESULT_DIRECTORY [--save]")
    length(ARGS)==2&&ARGS[2]!="--save"&&error("unknown verifier option")
    ccpv_run(ARGS[1];save=length(ARGS)==2)
end
