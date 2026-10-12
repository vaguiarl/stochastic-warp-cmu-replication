# New independent-reference, finite-anchor Gaussian power study.
# Existing empirical c=1 results, geometries, estimators and sources are read-only.
include(joinpath(@__DIR__,"simulate_cmu_c_geometry.jl"))
isdefined(@__MODULE__,:cmue_verify_record) || include(joinpath(@__DIR__,"common_cmu_empty_witnesses.jl"))

const CCP_CELLS=[7,12]
const CCP_GRID=[1.,2.]
const CCP_ALPHAS=[.05,.10]
const CCP_RATE=sqrt(46391/3)
const CCP_REFERENCE_M=199
const CCP_VALIDATION_M=250
const CCP_B=199
const CCP_REFERENCE_SEED=902610121
const CCP_VALIDATION_SEED=1902610121
const CCP_TOL=1e-8
ccp_q(x)=parse(CSFI_Q,String(x))
ccp_sha(path)=bytes2hex(sha256(read(path)))

function ccp_new(path,object)
    ispath(path) && error("refusing to overwrite $path")
    open(path,"w") do io
        JSON3.write(io,object)
    end
end

"""First lexicographic Cartesian vertex outside W with common R/W support.

The selector normal is the vertex itself. All R/W support scores are integers;
their exact equality and strict separation make a common exposed face, not a
facet claim. No observed probability, empirical distance or p-value is used.
"""
function ccp_common_face(d;maximum_candidates=2_000_000)
    A=d.A; W=d.design.Aw; groups=d.groups; I,J=size(A,1),length(groups)
    prod(big(length(g)) for g in groups)<=maximum_candidates || error("Cartesian design search exceeds declared limit")
    Wset=Set(Tuple(view(W,:,h)) for h in axes(W,2))
    candidates=0; chosen=nothing
    # Reverse product axes so that the original first budget is lexicographically slowest.
    for reverse_rows in Iterators.product(reverse(groups)...)
        rows=reverse(collect(reverse_rows)); s=zeros(Int8,I);s[rows].=1
        candidates+=1
        Tuple(s) in Wset && continue
        rscore=[sum(Int(s[i])*Int(A[i,h]) for i in 1:I) for h in axes(A,2)]
        wscore=[sum(Int(s[i])*Int(W[i,h]) for i in 1:I) for h in axes(W,2)]
        hr,hw=maximum(rscore),maximum(wscore)
        hr==hw<J || continue
        chosen=(;s,rows,rscore,wscore,support=hr)
        break
    end
    chosen===nothing && error("no canonical common R/W exposed face found")
    face_indices=findall(==(chosen.support),chosen.rscore)
    face=csdg_column_mixture(A,face_indices)
    dot(CSFI_Q.(chosen.s),face.point_exact)==chosen.support || error("common-face null is not on support plane")
    (;chosen...,face_indices,face,candidates,
      record=Dict("enumeration"=>"lexicographic selected patch rows, ascending rows in original budget order",
        "Cartesian_candidates_examined"=>candidates,"s"=>Int.(chosen.s),"selected_rows"=>Int.(chosen.rows),
        "R_support_exact"=>string(CSFI_Q(chosen.support)),"W_support_exact"=>string(CSFI_Q(chosen.support)),
        "S_vertex_score_exact"=>string(CSFI_Q(J)),"R_scores_exact"=>chosen.rscore,"W_scores_exact"=>chosen.wscore,
        "R_face_indices"=>face_indices,"R_face_mass_numerators"=>string.(face.weights.counts),
        "R_face_mass_denominator"=>string(face.weights.denominator),"q_empty0_exact"=>string.(face.point_exact),
        "face_not_claimed_facet"=>true,"observed_UK_fit_or_p_values_used"=>false))
end

function ccp_empty_alternative(d,common,epsilon)
    I,J=size(d.A,1),length(d.groups);J==7 || error("approved common-rate construction uses seven budgets")
    eps=CSFI_Q(Float64(epsilon));tau=1-eps
    beta_allowed=min(CSFI_Q(1),eps/((J-1)*tau))
    denominator=big(1)<<32
    beta=floor(BigInt,beta_allowed*denominator)//denominator
    0<beta<=beta_allowed || error("dyadic-down beta invalid")
    q0=common.face.point_exact;s=CSFI_Q.(common.s)
    mean_exact=(1-beta)*q0+beta*s
    mean=Float64.(mean_exact)
    CSFI_Q.(mean)==mean_exact || error("empty alternative not exactly dyadic in Float64")
    all(sum(mean_exact[g])==1&&all(>=(0),mean_exact[g]) for g in d.groups) || error("empty mean outside product simplex")
    score=dot(s,mean_exact)
    score>common.support || error("empty mean not strictly outside complete WARP")
    complete_mass=tau*(1-beta)
    complete_weights=complete_mass*common.face.weights.weights
    singleton_mass=tau*beta
    null_mass=1-complete_mass-J*singleton_mass
    all(>=(0),complete_weights)&&singleton_mass>=0&&null_mass>=0 || error("negative actual population mass")
    sum(complete_weights)+J*singleton_mass+null_mass==1 || error("actual population normalization failed")
    ordinary=CSFI_Q.(d.A[:,common.face_indices])*complete_weights+singleton_mass*s
    ordinary==tau*mean_exact && complete_mass+singleton_mass==tau || error("actual response/conditional moments failed")
    # The utility registry implements each singleton with ONE preference across
    # all original budgets and has a genuine all-null preference; no deletion rule.
    path=joinpath(@__DIR__,"results/cmu_actual_empty_comparison_20261011_v1","cell_$(lpad(d.cell,2,'0')).json")
    registry=JSON3.read(read(path,String))
    X=d.design.X
    oldX=reduce(vcat,permutedims(Int8.(row)) for row in registry["sign_matrix"])
    oldX==X || error("actual CMU witness geometry changed")
    P=reshape(Float64.(registry["saved_normalized_prices"]),J,3)
    witnesses=registry["actual_empty_single_budget_witnesses"]
    for row in common.rows
        selected=only(w for w in witnesses if Int(w["row"])==row)
        cmue_verify_record(P,X,selected) || error("selected genuine-empty responder witness fails")
    end
    cmue_verify_record(P,X,registry["actual_empty_all_null_witness"]) || error("genuine all-null witness fails")
    record=Dict("epsilon_cap"=>Float64(epsilon),"response_rate_exact"=>string(tau),
        "allowed_beta_exact"=>string(beta_allowed),"beta_down_exact"=>string(beta),"beta_denominator_power"=>32,
        "mean_exact"=>string.(mean_exact),"complete_RUM_columns"=>common.face_indices,
        "complete_population_masses_exact"=>string.(complete_weights),
        "singleton_rows"=>Int.(common.rows),"population_mass_each_singleton_exact"=>string(singleton_mass),
        "all_null_population_mass_exact"=>string(null_mass),"conditional_moments_verified_exactly"=>true,
        "complete_WARP_support_exact"=>string(CSFI_Q(common.support)),"mean_WARP_selector_score_exact"=>string(score),
        "strictly_outside_complete_WARP"=>true,"inside_witnessed_common_rate_CMU_at_declared_cap"=>true,
        "selected_singleton_and_all_null_utility_records_verified"=>true,
        "utility_registry_sha256"=>ccp_sha(path),"ordinary_complete_CMU_bridge"=>"existing complete WARP--CMU interpolation",
        "no_observed_UK_empty_share_or_fit_gain_used"=>true)
    (;mean,mean_exact,record)
end

function ccp_design(cell;rate=CCP_RATE)
    d=ccgd_design(cell;rate)
    common=ccp_common_face(d)
    empty10=ccp_empty_alternative(d,common,.10)
    empty20=ccp_empty_alternative(d,common,.20)
    scenarios=[
        (;id="centroid_null",null=true,mean=d.scenarios[1].mean,kind="complete_RUM_null"),
        (;id="WARP_exposed_face_null",null=true,mean=d.scenarios[2].mean,kind="complete_RUM_null"),
        (;id="common_R_W_face_null",null=true,mean=common.face.point,kind="complete_RUM_null"),
        (;id="WARP_one_SE",null=false,mean=d.scenarios[3].mean,kind="complete_WARP_not_RUM"),
        (;id="WARP_two_SE",null=false,mean=d.scenarios[4].mean,kind="complete_WARP_not_RUM"),
        (;id="actual_empty_cap10",null=false,mean=empty10.mean,kind="actual_empty_CMU_outside_complete_WARP"),
        (;id="actual_empty_cap20",null=false,mean=empty20.mean,kind="actual_empty_CMU_outside_complete_WARP")]
    record=Dict("cell"=>cell,"years"=>d.years,"original_geometry_design"=>d.record,
        "common_R_W_face"=>common.record,"actual_empty_alternatives"=>[empty10.record,empty20.record],
        "Gaussian_factor"=>vec(d.L),"Gaussian_factor_dimensions"=>collect(size(d.L)),
        "scenarios"=>[Dict("id"=>s.id,"population_null"=>s.null,"kind"=>s.kind,
            "mean_exact"=>string.(CSFI_Q.(s.mean))) for s in scenarios],
        "local_geometry_and_fixed_estimated_covariance_only"=>true)
    (;cell=d.cell,years=d.years,A=d.A,groups=d.groups,L=d.L,scenarios,record,common,empty10,empty20)
end

"""A fixed WHOLE numerical procedure is scored, not its favorable raw tail.

Only the original 10% positive-critical, full-B conditional rejection gets
score 1-p. All other outcomes get zero. This predeclared truncation is arbitrary
but is a valid scalar statistic for the subsequent finite-anchor rank experiment.
"""
function ccp_scores(result)
    length(result["results"])==2 || error("c1/c2 score requires exactly two methods")
    [begin
        row=result["results"][k]
        Float64(row["c"])==CCP_GRID[k] || error("score method order changed")
        decision=only(d for d in row["decisions"] if Float64(d["alpha"])==.10)
        if decision["reject"]
            B=Int(row["B_fixed"]);tail=Int(row["upper_tail_exceedances_evaluated"])
            row["inner_targets_evaluated"]==B && decision["critical_positive_resolved"]===true ||
                error("positive score lacks complete positive-critical calculation")
            score=CSFI_Q(1)-(tail+1)//(B+1)
        else
            score=CSFI_Q(0)
        end
        Dict("c"=>CCP_GRID[k],"score_exact"=>string(score),"score"=>Float64(score),
            "original_10pct_conditional_rejection"=>decision["reject"])
    end for k in 1:2]
end

function ccp_rank_threshold(scores,alpha)
    M=length(scores);M>0 || error("reference scores empty")
    values=sort(CSFI_Q.(scores));order=ceil(Int,(1-Float64(alpha))*(M+1))
    critical=order>M ? nothing : values[order]
    Dict("alpha"=>Float64(alpha),"reference_M"=>M,"order_statistic_index"=>order,
        "critical_score_exact"=>critical===nothing ? nothing : string(critical),
        "critical_score"=>critical===nothing ? nothing : Float64(critical),
        "infinite_threshold"=>critical===nothing,"strict_greater_rejection"=>true)
end

function ccp_freeze_thresholds(reference,designs;reference_M)
    cells=Dict{String,Any}()
    for d in designs
        records=reference[d.cell]
        length(records)==reference_M || error("reference bank incomplete")
        methods=Dict{String,Any}()
        for k in 1:2
            alphas=Dict{String,Any}()
            for alpha in CCP_ALPHAS
                anchors=Dict{String,Any}()
                for s in filter(s->s.null,d.scenarios)
                    scores=[ccp_q(only(e for e in bank["experiments"] if e["scenario"]==s.id)["scores"][k]["score_exact"]) for bank in records]
                    anchors[s.id]=ccp_rank_threshold(scores,alpha)
                end
                infinite=any(a["infinite_threshold"] for a in values(anchors))
                critical=infinite ? nothing : maximum(ccp_q(a["critical_score_exact"]) for a in values(anchors))
                alphas[string(alpha)]=Dict("alpha"=>alpha,"null_anchor_thresholds"=>anchors,
                    "critical_score_exact"=>critical===nothing ? nothing : string(critical),
                    "critical_score"=>critical===nothing ? nothing : Float64(critical),
                    "infinite_threshold"=>infinite,"max_over_declared_null_anchors"=>true)
            end
            methods[string(CCP_GRID[k])]=Dict("c"=>CCP_GRID[k],"alphas"=>alphas)
        end
        cells[string(d.cell)]=Dict("cell"=>d.cell,"methods"=>methods)
    end
    Dict("cells"=>cells,"reference_M"=>reference_M,"score_rule"=>"1-MCtail only after original10% conditionalrejection; zero otherwise",
        "rank_level_scope"=>"Unconditional over independent reference and test experiments, within the specified known Gaussian experiment and FINITE declared null-anchor menu; ties make it conservative",
        "critical_frozen_before_validation"=>true,"method_selection"=>false,
        "not_a_uniform_composite_null_or_UK_CF_guarantee"=>true)
end

function ccp_apply_thresholds(scores,thresholds,cell)
    [Dict("c"=>CCP_GRID[k],"decisions"=>[begin
        t=thresholds["cells"][string(cell)]["methods"][string(CCP_GRID[k])]["alphas"][string(alpha)]
        score=ccp_q(scores[k]["score_exact"])
        critical=t["critical_score_exact"]===nothing ? nothing : ccp_q(t["critical_score_exact"])
        Dict("alpha"=>alpha,"reject"=>critical!==nothing&&score>critical,
            "score_exact"=>string(score),"frozen_critical_score_exact"=>t["critical_score_exact"],
            "strict_ties_not_rejected"=>true)
    end for alpha in CCP_ALPHAS]) for k in 1:2]
end

function ccp_phase(designs,phase,repetitions,B,output;thresholds=nothing,
        workers=min(6,Threads.nthreads()))
    phase in (:reference,:validation) || error("unknown phase")
    1<=workers<=Threads.nthreads() || error("invalid simulation workers")
    phase==:validation && thresholds===nothing && error("validation requires FROZEN reference thresholds")
    master=phase==:reference ? CCP_REFERENCE_SEED : CCP_VALIDATION_SEED
    allbanks=Dict{Int,Any}()
    for d in designs
        scenarios=phase==:reference ? filter(s->s.null,d.scenarios) : d.scenarios
        banks=Vector{Any}(undef,repetitions);complete=fill(false,repetitions)
        failures=Vector{Any}(undef,repetitions);fill!(failures,nothing)
        done=Threads.Atomic{Int}(0);start=time()
        @sync for worker in 1:workers
            Threads.@spawn begin
                for rep in worker:workers:repetitions
                    try
                        seed=master+1_000_003d.cell+10_007rep
                        rng=MersenneTwister(seed)
                        Zouter=randn(rng,size(d.L,2));Zinner=randn(rng,size(d.L,2),B)
                        Gouter=d.L*Zouter;Ginner=d.L*Zinner
                        experiments=Dict{String,Any}[]
                        for s in scenarios
                            # Fresh optimizer initialization per experiment is
                            # part of the fixed score map. No cross-replicate history.
                            source=csfi_make_fit(d.A,d.groups;tolerance=CCP_TOL,store_certificate=false)
                            maxwidth=CSFI_Q(0);fits=0;fallbacks=0
                            function audited_fit(target)
                                p=source.fit(target)
                                maxwidth=max(maxwidth,p.upper-p.lower);fits+=1
                                fallbacks+=get(p.diagnostic,"full_Clarabel_fallback_used",false)
                                p
                            end
                            point=s.mean.+Gouter/CCP_RATE
                            result=ccg_decisions(audited_fit,point,Ginner,CCP_RATE;
                                grid=CCP_GRID,alphas=CCP_ALPHAS,stop_nonrejection=true,store_trace=true)
                            scores=ccp_scores(result)
                            push!(experiments,Dict("scenario"=>s.id,"population_null"=>s.null,"kind"=>s.kind,
                                "point_float"=>point,"scores"=>scores,"whole_procedure_trace"=>result,
                                "fresh_projector_initialized_per_experiment"=>true,
                                "exact_objective_bounds_diagnostics"=>Dict("fits"=>fits,"fallbacks"=>fallbacks,
                                    "maximum_Psi_interval_width_exact"=>string(maxwidth),
                                    "all_original_RUM_columns_priced"=>size(d.A,2)),
                                "calibrated_decisions"=>phase==:validation ? ccp_apply_thresholds(scores,thresholds,d.cell) : nothing))
                        end
                        latent=Dict("outer_standard_Gaussian"=>Zouter,"inner_standard_Gaussian"=>vec(Zinner),
                            "inner_dimensions"=>collect(size(Zinner)),"matrix_storage"=>"column_major_factor_by_B")
                        banks[rep]=Dict("cell"=>d.cell,"phase"=>String(phase),"replication"=>rep,"seed"=>seed,
                            "latent_draws"=>latent,"latent_draws_sha256"=>bytes2hex(sha256(JSON3.write(latent))),
                            "common_random_numbers_across_c_and_scenarios_within_replication"=>true,
                            "fresh_independent_outer_and_inner"=>true,"B_fixed"=>B,"experiments"=>experiments)
                        complete[rep]=true
                        count=Threads.atomic_add!(done,1)+1
                        if count%25==0 || count==repetitions
                            println("calibrated ",phase," cell=",d.cell," outer=",count,"/",repetitions,
                                " seconds=",round(time()-start;digits=1));flush(stdout)
                        end
                    catch err
                        failures[rep]=sprint(showerror,err,catch_backtrace())
                    end
                end
            end
        end
        if any(!isnothing,failures)
            ccp_new(joinpath(output,"failed_$(phase)_cell_$(d.cell).json"),Dict("phase"=>String(phase),"cell"=>d.cell,
                "failures"=>[Dict("replication"=>rep,"error"=>failures[rep]) for rep in 1:repetitions if failures[rep]!==nothing],
                "completed_replications"=>findall(complete),"failed_experiments_NOT_replaced_or_omitted"=>true))
            error("calibrated phase failed; no completed inference summary")
        end
        all(complete) || error("incomplete phase")
        filename="$(phase)_cell_$(lpad(d.cell,2,'0'))_experiments.json"
        ccp_new(joinpath(output,filename),Dict("phase"=>String(phase),"cell"=>d.cell,"replicate_banks"=>banks))
        allbanks[d.cell]=banks
    end
    allbanks
end

function ccp_summary(validation,designs,thresholds;validation_M)
    rows=Dict{String,Any}[];evaluated=0;pruned=0;fallbacks=0
    for d in designs,s in d.scenarios,k in 1:2,ai in 1:2
        experiments=[only(e for e in bank["experiments"] if e["scenario"]==s.id) for bank in validation[d.cell]]
        length(experiments)==validation_M || error("validation bank incomplete")
        rejects=count(e->e["calibrated_decisions"][k]["decisions"][ai]["reject"],experiments)
        original=count(e->e["whole_procedure_trace"]["results"][k]["decisions"][ai]["reject"],experiments)
        push!(rows,Dict("cell"=>d.cell,"years"=>d.years,"scenario"=>s.id,"kind"=>s.kind,
            "population_null"=>s.null,"c"=>CCP_GRID[k],"alpha"=>CCP_ALPHAS[ai],"validation_M"=>validation_M,
            "calibrated_rejections"=>rejects,"calibrated_rejection_rate"=>rejects/validation_M,
            "calibrated_CP95_conditional_on_frozen_reference_bank"=>ccg_interval(rejects,validation_M),
            "original_unadjusted_rejections"=>original,"original_unadjusted_rejection_rate"=>original/validation_M,
            "original_unadjusted_CP95"=>ccg_interval(original,validation_M),
            "frozen_critical_score_exact"=>thresholds["cells"][string(d.cell)]["methods"][string(CCP_GRID[k])]["alphas"][string(CCP_ALPHAS[ai])]["critical_score_exact"]))
    end
    for banks in values(validation),bank in banks,e in bank["experiments"]
        fallbacks+=e["exact_objective_bounds_diagnostics"]["fallbacks"]
        for row in e["whole_procedure_trace"]["results"]
            evaluated+=row["inner_targets_evaluated"];pruned+=row["inner_targets_decision_pruned"]
        end
    end
    Dict("rows"=>rows,"validation_inner_targets_evaluated"=>evaluated,"validation_targets_decision_pruned"=>pruned,
        "validation_full_Clarabel_fallbacks"=>fallbacks,"conditional_binomial_interval_scope"=>
        "Given this stored independent reference bank and fixed estimated Gaussian nuisance, outer validation experiments are iid per scenario; CRN across scenarios/methods does not create cross-replication dependence",
        "unconditional_rank_scope"=>"Size bounded per method/alpha across reference-and-test randomness in the specified known Gaussian experiment, only for the finite three-anchor null menu",
        "no_optimal_c_or_UK_CF_sampling_claim"=>true,"existing_empirical_c1_primary_unchanged"=>true)
end

function ccp_hashes()
    names=["simulate_cmu_calibrated_power.jl","test/test_cmu_calibrated_power.jl",
        "simulate_cmu_c_geometry.jl","cmu_c_geometry_design.jl","cmu_superiority_floor_inference.jl",
        "cmu_empty_inner_projection.jl","common_cmu_empty_witnesses.jl","wilfried_normalized_working_set.jl",
        "Project.toml","Manifest.toml","selection_solver_environment/Project.toml","selection_solver_environment/Manifest.toml",
        "results/cmu_superiority_floor_20261012_v1/summary.json","results/cmu_c_geometry_simulation_20261012_v1/summary.json",
        "results/cmu_actual_empty_comparison_20261011_v1/summary.json","results/wilfried_windows_20261009_v2/summary.json"]
    for cell in CCP_CELLS
        push!(names,"results/wilfried_joint_gap_20261010_v1/raw_bootstrap_cell_$(lpad(cell,2,'0')).json",
            "results/cmu_actual_empty_comparison_20261011_v1/cell_$(lpad(cell,2,'0')).json")
    end
    Dict(name=>ccp_sha(joinpath(@__DIR__,name)) for name in names)
end

function ccp_run(;probe=false,workers=min(6,Threads.nthreads()))
    BLAS.set_num_threads(1)
    Mref=probe ? 19 : CCP_REFERENCE_M;Mval=probe ? 3 : CCP_VALIDATION_M;B=probe ? 39 : CCP_B
    output=joinpath(@__DIR__,"results",probe ? "cmu_calibrated_power_probe_20261012_v1" : "cmu_calibrated_power_20261012_v1")
    ispath(output) && error("refusing an existing calibrated-power output directory")
    before=ccp_hashes();mkpath(output);start=time()
    protocol=Dict("stage"=>"independent finite-anchor whole-procedure rank calibration and validation",
        "cells"=>CCP_CELLS,"c_methods"=>CCP_GRID,"c_selection"=>false,"existing_UK_c1_primary_unchanged"=>true,
        "alphas"=>CCP_ALPHAS,"reference_M_per_anchor"=>Mref,"validation_M_per_case"=>Mval,"B_fixed"=>B,
        "rate"=>CCP_RATE,"reference_seed_master"=>CCP_REFERENCE_SEED,"validation_seed_master"=>CCP_VALIDATION_SEED,
        "seed_rule"=>"phase_master+1000003*cell+10007*replication",
        "null_menu"=>["centroid_null","WARP_exposed_face_null","common_R_W_face_null"],
        "alternative_menu"=>["WARP_one_SE","WARP_two_SE","actual_empty_cap10","actual_empty_cap20"],
        "score"=>"1-MCtail iff original10pct positive-critical full-B conditionalreject, zero otherwise",
        "reference_quantile"=>"k=ceil((1-alpha)*(Mref+1)); maximum across declared null anchors; strict > handles ties",
        "reference_thresholds_frozen_before_validation"=>true,"fresh_projector_per_experiment"=>true,
        "outer_inner_independence"=>"Fresh independent Gaussian outer and inner draws each replication and phase; paired CRN across c/scenarios within replication",
        "no_UK_fit_gains_or_p_values_used_in_design"=>true,"no_inverse_ridge_rank_truncation_or_coordinate_deletion"=>true,
        "null_menu_rank_level"=>"Unconditional finite known-Gaussian rank bound for each method over this finite anchor menu; not uniform over the continuous RUM null",
        "validation_CP_scope"=>"Conditional on this frozen reference bank and estimated nuisance covariance",
        "actual_empty_alts"=>"Exact dyadic-down common-rate mixtures inside witnessed CMU and outside complete WARP at caps .10/.20",
        "all_latent_standard_Gaussian_draws_and_computed_traces_saved"=>true,"failed_draws_replaced"=>false,
        "probe_only"=>probe,"not_a_CF_refit_or_19_window_calibration"=>true,"oracle_not_in_this_source"=>true)
    ccp_new(joinpath(output,"protocol_frozen_before_outcomes.json"),Dict("protocol"=>protocol,"protected_sha256_before"=>before))
    designs=[ccp_design(cell) for cell in CCP_CELLS]
    ccp_new(joinpath(output,"geometry_and_exact_means.json"),[d.record for d in designs])
    reference=ccp_phase(designs,:reference,Mref,B,output;workers)
    thresholds=ccp_freeze_thresholds(reference,designs;reference_M=Mref)
    ccp_new(joinpath(output,"reference_thresholds_frozen_before_validation.json"),thresholds)
    threshold_hash=ccp_sha(joinpath(output,"reference_thresholds_frozen_before_validation.json"))
    validation=ccp_phase(designs,:validation,Mval,B,output;thresholds,workers)
    ccp_sha(joinpath(output,"reference_thresholds_frozen_before_validation.json"))==threshold_hash || error("reference thresholds changed during validation")
    after=ccp_hashes();before==after || error("protected input/source/frozen empirical result changed")
    summary=ccp_summary(validation,designs,thresholds;validation_M=Mval)
    summary["status"]=probe ? "completed_probe_not_power_or_size_evidence" : "completed_finite_anchor_calibrated_power"
    summary["protocol"]=protocol;summary["reference_thresholds"]=thresholds
    summary["elapsed_seconds"]=time()-start;summary["reference_thresholds_sha256"]=threshold_hash
    summary["protected_sha256_before"]=before;summary["protected_sha256_after"]=after
    summary["protected_and_frozen_results_unchanged"]=true
    ccp_new(joinpath(output,"summary.json"),summary)
    println("completed calibrated power ",output," seconds=",summary["elapsed_seconds"]);flush(stdout)
    summary
end

if abspath(PROGRAM_FILE)==@__FILE__
    ARGS==["--probe"] ? ccp_run(;probe=true) :
        ARGS==["--approved-full-stage"] ? ccp_run() :
        error("requires --probe or explicit approved --approved-full-stage; no automatic full campaign")
end
