#!/usr/bin/env julia
# Actual empty demand in a sufficient common-CMU class, not external deletion.
using KSReplication, JSON3, SHA, LinearAlgebra
push!(LOAD_PATH,joinpath(@__DIR__,"selection_solver_environment"))
include(joinpath(@__DIR__,"common_cmu_empty_witnesses.jl"))
include(joinpath(@__DIR__,"cmu_empty_inner_projection.jl"))

const CAEC_ROOT=@__DIR__
const CAEC_BASE=joinpath(CAEC_ROOT,"results/wilfried_windows_20261009_v2")
const CAEC_RAW=joinpath(CAEC_ROOT,"results/wilfried_joint_gap_20261010_v1")
const CAEC_REFERENCE=joinpath(CAEC_ROOT,"results/wilfried_normalized_fit_robustness_20261010_v1.json")
const CAEC_OUTPUT=joinpath(CAEC_ROOT,"results/cmu_actual_empty_comparison_20261011_v1")
const CAEC_CAPS=[0.0,0.005,0.01,0.025,0.05,0.10,0.20]
caec_sha(path)=bytes2hex(sha256(read(path)))
caec_read(path)=JSON3.read(read(path,String))

function caec_write_new(path,object)
    ispath(path) && error("refusing to overwrite $path")
    open(path,"w") do io
        JSON3.pretty(io,object)
    end
end

function caec_protected_hashes()
    paths=[joinpath(CAEC_ROOT,"Project.toml"),joinpath(CAEC_ROOT,"Manifest.toml"),
        CAEC_REFERENCE,joinpath(CAEC_BASE,"summary.json"),joinpath(CAEC_RAW,"summary.json"),
        joinpath(CAEC_ROOT,"results/wilfried_selection_sensitivity_20261011_v3/summary.json"),
        normpath(joinpath(CAEC_ROOT,"../../TheLawsOfDemand.tex")),
        normpath(joinpath(CAEC_ROOT,"../../TheLawsOfDemand.bib")),
        normpath(joinpath(CAEC_ROOT,"../../../../docs/SRUM_rational_cells_note_for_Nail_2026-10-10.tex"))]
    for name in readdir(joinpath(CAEC_ROOT,"src"))
        endswith(name,".jl") && push!(paths,joinpath(CAEC_ROOT,"src",name))
    end
    for cell in 1:19
        suffix="cell_$(lpad(cell,2,'0')).json"
        push!(paths,joinpath(CAEC_BASE,suffix),joinpath(CAEC_RAW,"raw_bootstrap_"*suffix))
    end
    for name in ("selection_partial_types.jl","wilfried_selection_sensitivity.jl",
            "run_wilfried_selection_sensitivity.jl","wilfried_normalized_working_set.jl",
            "verify_wilfried_selection_report.jl")
        push!(paths,joinpath(CAEC_ROOT,name))
    end
    for path in (get(ENV,"WILFRIED_ARCHIVE",joinpath(CAEC_ROOT,"external/RUMCODEFINAL.zip")),
            get(ENV,"FES_DATA_ARCHIVE",joinpath(CAEC_ROOT,"external/231552-V1.zip")))
        push!(paths,path)
    end
    Dict(relpath(p,CAEC_ROOT)=>caec_sha(p) for p in paths)
end

function caec_implementation_hashes()
    names=("run_cmu_actual_empty_comparison.jl","common_cmu_empty_witnesses.jl",
        "cmu_empty_inner_projection.jl","test/test_common_cmu_empty_witnesses.jl",
        "test/test_cmu_empty_inner_projection.jl","selection_solver_environment/Project.toml",
        "selection_solver_environment/Manifest.toml","../../../../docs/CMUEmptyDemandGeometry.lean")
    Dict(n=>caec_sha(normpath(joinpath(CAEC_ROOT,n))) for n in names)
end

function caec_window(cell,reference)
    start=time()
    old=caec_read(joinpath(CAEC_BASE,"cell_$(lpad(cell,2,'0')).json"))
    cache=caec_read(joinpath(CAEC_RAW,"raw_bootstrap_cell_$(lpad(cell,2,'0')).json"))
    old["cell"]==cache["cell"]==cell && old["years"]==cache["years"] || error("source alignment")
    X=reduce(vcat,permutedims(Int8.(r)) for r in old["sign_matrix"])
    J=size(X,2)
    P=reshape(Float64.(old["saved_prices"]),J,3)
    raw=Float64.(cache["raw_affine_point"])
    groups=KSReplication._patch_groups(X)
    maximum(abs.(clamp.(raw,0,1)-Float64.(old["pi_hat_regenerated"])))<=1e-12 ||
        error("frozen raw estimator mismatch")
    maximum(abs(sum(raw[g])-1) for g in groups)<=1e-10 || error("raw adding-up changed")
    Ar=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
    Aw=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
    size(Ar,2)==old["models"]["RUM"]["types"] &&
        size(Aw,2)==old["models"]["correct_WARP"]["types"] || error("complete types changed")
    # Every extra column receives a common preference-function witness, not a
    # rule deleting observed choices from a complete type.
    witnesses=Dict{String,Any}[]
    for row in axes(X,1)
        w=cmue_witness_for_patch(P,X,row)
        w.certified && cmue_verify_witness(P,X,w) || error("uncertified responder patch $row")
        record=cmue_witness_record(w)
        cmue_verify_record(P,X,JSON3.read(JSON3.write(record))) || error("witness serialization failed")
        push!(witnesses,record)
    end
    z=cmue_all_null_witness(P,X)
    z.certified && cmue_verify_witness(P,X,z) || error("uncertified all-null CMU")
    null_record=cmue_witness_record(z)
    cmue_verify_record(P,X,JSON3.read(JSON3.write(null_record))) || error("null serialization failed")
    family=ceip_build_family(Aw,groups;label="actual_empty_common_CMU_inner")
    fixed_rum=ceip_complete_fit(Ar,raw,groups;tolerance=1e-8)
    oldrum=reference["comparison"]["policies"]["raw_affine"]["models"]["RUM"]
    previous=oldrum["exact_full_projection_certificate"]
    oldlower=Float64(parse(Rational{BigInt},String(previous["lower_bound_exact"])))
    oldupper=Float64(parse(Rational{BigInt},String(previous["upper_bound_exact"])))
    fixed_rum["squared_distance_lower"]<=oldupper+1e-10 &&
        oldlower<=fixed_rum["squared_distance_upper"]+1e-10 || error("fixed RUM baseline changed")
    floor=ceip_simplex_floor(raw,groups)
    thresholds=Dict("common_response"=>ceip_floor_breakdown(family,floor.q;response_policy=:common),
        "variable_response"=>ceip_floor_breakdown(family,floor.q;response_policy=:variable))
    projector=CMUEmptyInnerProjector(family;tolerance=1e-8)
    profiles=Dict{String,Any}[]
    last_upper=Inf
    for epsilon in CAEC_CAPS
        fit=ceip_fit!(projector,raw,epsilon)
        fit["squared_distance_upper"]<=last_upper+2e-8 || error("inner conditional nesting failure")
        last_upper=fit["squared_distance_upper"]
        fit["squared_distance_upper"]>=floor.floor_float-1e-8 || error("below probability domain floor")
        fixed_rum["squared_distance_upper"]>=fit["squared_distance_lower"]-1e-9 || error("RUM/inner nesting")
        if epsilon==0
            oldwarp=reference["comparison"]["policies"]["raw_affine"]["models"]["WARP"]["exact_full_projection_certificate"]
            fit["squared_distance_lower"]<=Float64(parse(Rational{BigInt},String(oldwarp["upper_bound_exact"])))+1e-10 &&
                Float64(parse(Rational{BigInt},String(oldwarp["lower_bound_exact"])))<=fit["squared_distance_upper"]+1e-10 ||
                error("zero-empty baseline changed")
        end
        lower=max(0.0,fixed_rum["squared_distance_lower"]-fit["squared_distance_upper"])
        upper=max(0.0,fixed_rum["squared_distance_upper"]-fit["squared_distance_lower"])
        push!(profiles,Dict("empty_allowance_cap"=>epsilon,"common_response_rate"=>1-epsilon,
            "CMU_actual_empty_inner"=>fit,"fixed_RUM_gain_lower"=>lower,"fixed_RUM_gain_upper"=>upper,
            "numerically_resolved_gain"=>lower>1e-10,
            "additional_fit_gain_over_complete_WARP_lower"=>isempty(profiles) ? 0.0 :
                max(0.0,profiles[1]["CMU_actual_empty_inner"]["squared_distance_lower"]-fit["squared_distance_upper"])))
    end
    report=Dict("cell"=>cell,"years"=>Int.(old["years"]),"annual_n"=>Int.(old["annual_n"]),
        "annual_n_sum"=>sum(Int.(old["annual_n"])),"status"=>"completed_actual_empty_inner_descriptive",
        "sign_matrix"=>[collect(r) for r in eachrow(X)],"saved_normalized_prices"=>vec(P),
        "raw_affine_point_unchanged"=>raw,"fixed_RUM"=>fixed_rum,
        "types"=>Dict("complete_RUM"=>size(Ar,2),"complete_WARP"=>size(Aw,2),
            "actual_one_budget_responders"=>size(X,1),"actual_all_null"=>1,
            "actual_CMU_inner_total"=>size(Aw,2)+size(X,1)+1),
        "actual_empty_single_budget_witnesses"=>witnesses,"actual_empty_all_null_witness"=>null_record,
        "witness_geometry_key"=>cmue_geometry_key(P,X),"profiles"=>profiles,
        "probability_domain_floor"=>Dict("squared_distance_exact"=>string(floor.floor),
            "squared_distance"=>floor.floor_float,"simplex_target_exact"=>string.(floor.q),
            "used_only_for_floor_LP_not_distance_target"=>true),
        "allowance_to_domain_floor"=>thresholds,"new_p_values"=>nothing,
        "observed_empty_choice_frequencies_available"=>false,"actual_population_empty_share_estimated"=>false,
        "all_singleton_columns_have_same_preference_across_budgets"=>true,
        "entire_continuous_budget_emptiness_not_grid_test"=>true,"elapsed_seconds"=>time()-start)
    println("Actual-empty CMU cell $cell $(first(report["years"]))-$(last(report["years"])); ",
        size(X,1)," responder witnesses; fixed RUM distance ",fixed_rum["squared_distance_upper"])
    flush(stdout)
    report
end

function caec_main(args)
    args==["--write-approved-report"] || error("frozen all-window profile; requires --write-approved-report")
    ispath(CAEC_OUTPUT) && error("refusing existing output directory")
    refs=Dict(Int(w["cell"])=>w for w in caec_read(CAEC_REFERENCE)["windows"])
    before=caec_protected_hashes()
    codebefore=caec_implementation_hashes()
    protocol=Dict("analysis"=>"sufficient actual-empty coherent-CMU inner class versus fixed complete-choice RUM",
        "windows"=>collect(1:19),"empty_allowance_caps"=>CAEC_CAPS,"N_unique_records"=>46391,
        "RUM_demand_always_nonempty_and_model_unchanged"=>true,
        "CMU_family"=>"complete WARP types plus exactly witnessed one-budget-only and all-null majority-CMU types",
        "utility_primitives"=>"three positive-coefficient shifted-log utilities fixed across all budgets; three overlapping two-voter coalitions",
        "observable_outcome"=>"recorded conditional choice probabilities given preference-induced nonempty demand",
        "common_response_rate"=>"declared sufficient restriction at each cap, NOT without loss for general empty CMU",
        "missing_rates"=>"unobserved sensitivity parameters; neither manufactured frequencies nor estimated population shares",
        "domain_floor_LP"=>"separate common-response and variable-response maximin-rate inner feasibility certificates at exact simplex target",
        "distance_target"=>"unchanged full-coordinate raw-affine CF target; no clamp, simplex replacement or re-estimation",
        "metric"=>"conditional full-coordinate squared Euclidean distance",
        "sampling_prices_geometry_filters_and_CF_specification_unchanged"=>true,
        "analytic_scope"=>"whole-budget majority-core proof audited analytically; rational witness arithmetic checked; only four tangent-algebra leaves compiled in Lean",
        "empty_CMUs_knife_edge_in_prices"=>true,"unrestricted_participation_saturates_conditional_data"=>true,
        "primary_rates_selected_by_p_value"=>false,"new_p_values"=>false,"bootstrap_run"=>false,
        "complete_structural_CMU_characterization_established"=>false,
        "inner_rejection_does_not_reject_general_CMU"=>true,"statistical_significance_established"=>false)
    mkpath(CAEC_OUTPUT)
    caec_write_new(joinpath(CAEC_OUTPUT,"frozen_protocol.json"),Dict("protocol"=>protocol,
        "protected_sha256_before"=>before,"implementation_sha256_before"=>codebefore))
    BLAS.set_num_threads(1)
    windows=Vector{Any}(undef,19)
    failures=fill!(Vector{Any}(undef,19),nothing)
    start=time()
    Threads.@threads :dynamic for cell in 1:19
        try
            windows[cell]=caec_window(cell,refs[cell])
            caec_write_new(joinpath(CAEC_OUTPUT,"cell_$(lpad(cell,2,'0')).json"),windows[cell])
        catch err
            failures[cell]=sprint(showerror,err,catch_backtrace())
        end
    end
    if any(!isnothing,failures)
        caec_write_new(joinpath(CAEC_OUTPUT,"failed_checkpoint.json"),Dict("failures"=>[
            Dict("cell"=>c,"error"=>failures[c]) for c in 1:19 if failures[c]!==nothing],
            "completed_cells"=>[c for c in 1:19 if failures[c]===nothing],"failed_cells_not_omitted"=>true))
        error("actual-empty inner comparison failed; preserve checkpoints")
    end
    after=caec_protected_hashes()
    codeafter=caec_implementation_hashes()
    before==after && codebefore==codeafter || error("protected data/source/implementation changed")
    weighted=Dict{String,Any}[]
    r=sum(w["annual_n_sum"]/46391*w["fixed_RUM"]["squared_distance_upper"] for w in windows)
    for k in eachindex(CAEC_CAPS)
        d=sum(w["annual_n_sum"]/46391*w["profiles"][k]["CMU_actual_empty_inner"]["squared_distance_upper"] for w in windows)
        lo=sum(w["annual_n_sum"]/46391*w["profiles"][k]["fixed_RUM_gain_lower"] for w in windows)
        hi=sum(w["annual_n_sum"]/46391*w["profiles"][k]["fixed_RUM_gain_upper"] for w in windows)
        push!(weighted,Dict("empty_allowance_cap"=>CAEC_CAPS[k],"fixed_RUM_distance"=>r,
            "actual_empty_CMU_inner_distance"=>d,"fixed_RUM_gain_lower"=>lo,"fixed_RUM_gain_upper"=>hi,
            "distance_reduction_fraction"=>(r-d)/r,"numerically_positive_gain_windows"=>[
                w["cell"] for w in windows if w["profiles"][k]["numerically_resolved_gain"]]))
    end
    report=Dict("status"=>"completed_descriptive_actual_empty_CMU_inner_comparison",
        "protocol"=>protocol,"windows"=>windows,"weighted_profiles"=>weighted,
        "elapsed_seconds"=>time()-start,"source_integrity_unchanged"=>true,
        "protected_sha256_before"=>before,"protected_sha256_after"=>after,
        "implementation_sha256_before"=>codebefore,"implementation_sha256_after"=>codeafter,
        "interpretation"=>"Unlike external choice deletion, every added type has a common coherent-CMU witness with actual empty demand on whole continuous budgets. This sufficient class demonstrates conditional feasibility/relative fit under declared unobserved absence-rate bounds, not measured empty shares, sampling significance, robust counterfactual existence, or a full characterization of arbitrary empty-CMU preferences. Separate windows do not establish a common25-budget latent coupling.")
    caec_write_new(joinpath(CAEC_OUTPUT,"summary.json"),report)
    println("Saved actual-empty common-CMU comparison: $CAEC_OUTPUT")
    report
end

if abspath(PROGRAM_FILE)==@__FILE__
    caec_main(ARGS)
end
