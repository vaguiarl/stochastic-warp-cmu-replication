#!/usr/bin/env julia
# Portable Julia numerical release entry point. Originals stay unchanged.
using KSReplication, JSON3, SHA, LinearAlgebra
const RELEASE_ROOT=normpath(joinpath(@__DIR__,".."))
include(joinpath(RELEASE_ROOT,"common_cmu_empty_witnesses.jl"))
include(joinpath(RELEASE_ROOT,"cmu_superiority_floor_inference.jl"))
include(joinpath(RELEASE_ROOT,"wilfried_normalized_joint_significance.jl"))

rp_read(path)=JSON3.read(read(path,String))
rp_sha(path)=bytes2hex(sha256(read(path)))
function rp_options(args)
    out=Dict("data"=>joinpath(RELEASE_ROOT,"data"),"output"=>"","mode"=>"recorded", "cells"=>"all")
    iseven(length(args)) || error("Expected --name value pairs")
    for i in 1:2:length(args)
        startswith(args[i],"--") || error("Expected --name")
        key=args[i][3:end]; haskey(out,key) || error("Unknown argument $key")
        out[key]=args[i+1]
    end
    !isempty(out["output"]) || error("Required --output NEW_OUTPUT_DIRECTORY")
    out["mode"] in ("recorded","points","bootstrap") || error("Mode must be recorded, points or bootstrap")
    out
end
function rp_new(path,obj)
    ispath(path) && error("Refusing to overwrite $path")
    mkpath(dirname(path));open(path,"w") do io;JSON3.write(io,obj);println(io);end
end
function rp_assert_close(x,y,label;atol=1e-7)
    isapprox(Float64(x),Float64(y);atol,rtol=1e-10) || error("$label differs: $x vs $y")
end
function rp_check_fixture(data)
    manifest=rp_read(joinpath(data,"fixture_manifest.json"))
    for (name,digest) in pairs(manifest["files"])
        path=normpath(joinpath(data,String(name)))
        startswith(path,data*string(Base.Filesystem.path_separator)) || error("Fixture path escapes data directory")
        rp_sha(path)==String(digest) || error("Fixture content changed: $name")
    end
    fixture=rp_read(joinpath(data,"uk_aggregate_fixture.json"))
    Int(fixture["N_unique"])==46391 || error("Registered source count changed")
    sort(Int.(getindex.(fixture["windows"],"cell")))==collect(1:19) || error("All19 required")
    annual=Dict{Int,Int}()
    for w in fixture["windows"]
        years=Int.(w["years"]); n=Int.(w["annual_n"])
        for (y,k) in zip(years,n)
            get(annual,y,k)==k || error("Overlapping calendar year counts differ")
            annual[y]=k
        end
        rp_assert_close(w["original_window_weight"],sum(n)/46391,"Window weight";atol=0.)
        for (label,model) in pairs(w["recorded_receipts_not_new_estimates"]["source_cone_models"])
            values=Float64.(model["bootstrap_J_rounded_6"]);length(values)==1000 || error("Incomplete source-profile diagnostic bootstrap")
            J=Float64(model["J_rounded_6"]);tail=count(>=(J),values)
            tail==Int(model["tail_count"]) || error("Source-profile tail count differs for $label")
            p=J==0 ? 1. : tail/1000
            p==Float64(model["p_value_tau0_diagnostic"]) || error("Source-profile diagnostic p differs for $label")
        end
    end
    sum(values(annual))==46391 || error("Unique annual sample counts differ")
    fixture
end
function rp_restore_envelope(obj;floor=false)
    e=Dict(String(k)=>v for (k,v) in pairs(obj));B=1000;K=3
    for name in ("derivative_central","derivative_lower","derivative_upper")
        length(e[name])==B*K || error("Incomplete envelope $name")
        e[name]=reshape(Float64.(e[name]),B,K)
    end
    if floor
        for name in ("derivative_lower_exact","derivative_upper_exact")
            length(e[name])==B*K || error("Incomplete exact envelope")
            e[name]=reshape(String.(e[name]),B,K)
        end
    end
    e
end
function rp_compare_receipt(actual,expected;floor=false)
    if floor
        for k in ("observed_statistic_lower_exact","observed_statistic_upper_exact")
            actual[k]==expected[k] || error("Exact recorded statistic differs: $k")
        end
    else
        for k in ("statistic","statistic_lower_numerical","statistic_upper_numerical")
            rp_assert_close(actual[k],expected[k],"Recorded $k";atol=1e-10)
        end
    end
    for (r,s) in zip(actual["epsilon_results"],expected["epsilon_results"])
        for k in ("exceedances","B","reject_conditional","critical_value_positive_resolved")
            r[k]==s[k] || error("Recorded inference $k differs")
        end
        for k in ("epsilon_multiplier","p_value_mc","central_critical_value","critical_value_upper")
            r[k]===nothing && s[k]===nothing && continue
            rp_assert_close(r[k],s[k],"Recorded inference $k";atol=1e-9)
        end
    end
    true
end
function rp_recorded(data,fixture)
    record=rp_read(joinpath(data,"recorded_joint_inference.json"))
    weights=Dict(parse(Int,String(k))=>Float64(v) for (k,v) in pairs(fixture["weights"]))
    rate=Float64(fixture["rate"]);epsilons=[.5,1.,2.]./sqrt(rate)
    we=[rp_restore_envelope(e) for e in record["WARP_envelopes"]]
    ce=[rp_restore_envelope(e;floor=true) for e in record["CMU_Psi_envelopes"]]
    se=[rp_restore_envelope(e) for e in record["source_clipped_cone_envelopes"]]
    tests=Dict{String,Any}()
    for alpha in (.05,.10)
        w=wnj_aggregate(we,weights,rate,epsilons;alpha)
        c=csfi_aggregate(ce,weights,rate;alpha)
        rp_compare_receipt(w,record["recorded_WARP_comparisons"][string(alpha)])
        rp_compare_receipt(c,record["recorded_CMU_comparisons"][string(alpha)];floor=true)
        tests[string(alpha)]=Dict("ordinary_complete_WARP"=>w,"actual_empty_CMU_positive_cap_null_Psi"=>c)
    end
    source_test=wjc_aggregate(se,weights,rate,epsilons;alpha=.05)
    rp_compare_receipt(source_test,record["recorded_source_clipped_cone_comparison_5_percent"])
    Dict("status"=>"passed_all19_recorded_joint_envelope_reaggregation",
        "joint_tests"=>tests,"all_c_tail_counts_and_numerical_gates_match"=>true,
        "source_clipped_cone_joint_5_percent"=>source_test,
        "all38_source_profile_tau0_diagnostic_tails_checked"=>true,
        "scope"=>"Recomputed all aggregate tails, order statistics and gates from stored projection derivative envelopes; no QP or household-estimator refit",
        "empirical_sampling_validity_and_optimal_c_claimed"=>false)
end
function rp_point_window(w,fixture)
    cell=Int(w["cell"]);X=reduce(vcat,permutedims(Int8.(r)) for r in w["sign_matrix"])
    P=reshape(Float64.(w["prices_column_major"]),7,3);groups=KSReplication._patch_groups(X)
    p=Float64.(w["raw_affine_point"]); clipped=clamp.(p,0,1)
    maximum(abs.(clipped-Float64.(w["source_clipped_point"])))<=1e-12 || error("Source clipping relationship changed")
    maximum(abs(sum(p[g])-1) for g in groups)<=1e-10 || error("Raw adding-up differs")
    Ar=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
    Aw=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
    size(Ar,2)==Int(w["complete_types"]["RUM"]) || error("RUM count differs")
    size(Aw,2)==Int(w["complete_types"]["WARP"]) || error("WARP count differs")
    cmue_verify_record(P,X,w["all_null_witness"]) || error("Null utility witness fails")
    length(w["single_budget_witnesses"])==size(X,1) || error("Missing singleton witness")
    all(cmue_verify_record(P,X,r) for r in w["single_budget_witnesses"]) || error("Singleton utility witness fails")
    ref=w["recorded_receipts_not_new_estimates"]
    rawrum=ceip_complete_fit(Ar,p,groups;tolerance=1e-8)
    rawrum["squared_distance_lower"]<=Float64(ref["raw_normalized_RUM_distance_upper"])+1e-8 &&
        Float64(ref["raw_normalized_RUM_distance_lower"])<=rawrum["squared_distance_upper"]+1e-8 || error("RUM recorded interval does not overlap new exact interval")
    family=ceip_build_family(Aw,groups;label="witnessed_actual_empty_CMU_inner")
    projector=CMUEmptyInnerProjector(family;tolerance=1e-8)
    profiles=Any[]
    for (cap,stored) in zip(Float64.(fixture["caps"]),ref["inner_profiles"])
        cap==Float64(stored["cap"]) || error("Cap registry differs")
        fit=ceip_fit!(projector,p,cap)
        fit["squared_distance_lower"]<=Float64(stored["distance_upper"])+1e-8 &&
            Float64(stored["distance_lower"])<=fit["squared_distance_upper"]+1e-8 || error("Inner recorded interval does not overlap fresh interval")
        push!(profiles,Dict("cap"=>cap,"distance_lower"=>fit["squared_distance_lower"],
            "distance_upper"=>fit["squared_distance_upper"],"certificate"=>fit["certificate"]))
    end
    cone=Dict{String,Any}(); scale=sum(Int.(w["annual_n"]))/3
    for (label,A) in (("RUM",Ar),("correct_WARP",Aw))
        f=nnls_fit!(NNLSFullConeProjector(A),clipped)
        stored=ref["source_cone_models"][label]
        scale*f.lower<=Float64(stored["observed_J_bounds"][2])+1e-6 &&
            Float64(stored["observed_J_bounds"][1])<=scale*f.upper+1e-6 || error("Source-profile cone interval differs")
        cone[label]=Dict("J_lower"=>scale*f.lower,"J_upper"=>scale*f.upper,
            "recorded_bootstrap_p_NOT_recomputed"=>stored["p_value_tau0_diagnostic"])
    end
    floor=ceip_simplex_floor(p,groups)
    Dict("cell"=>cell,"years"=>w["years"],"type_counts"=>Dict("RUM"=>size(Ar,2),"WARP"=>size(Aw,2)),
        "weight"=>w["original_window_weight"],"fresh_raw_normalized_RUM"=>rawrum,"fresh_inner_profiles"=>profiles,
        "fresh_source_clipped_cone_J"=>cone,"probability_domain_floor_exact"=>string(floor.floor),
        "witnesses_checked"=>length(w["single_budget_witnesses"])+1,
        "all_fresh_objective_intervals_overlap_recorded_intervals"=>true),Ar,Aw,groups
end
function rp_pooled(rows,fixture;complete=true)
    profiles=Any[]
    rlo=sum(Float64(row["weight"])*row["fresh_raw_normalized_RUM"]["squared_distance_lower"] for row in rows)
    rhi=sum(Float64(row["weight"])*row["fresh_raw_normalized_RUM"]["squared_distance_upper"] for row in rows)
    for (k,cap) in enumerate(fixture["caps"])
        lo=sum(Float64(row["weight"])*row["fresh_inner_profiles"][k]["distance_lower"] for row in rows)
        hi=sum(Float64(row["weight"])*row["fresh_inner_profiles"][k]["distance_upper"] for row in rows)
        if complete
            expected=fixture["recorded_pooled_inner_profiles"][k]
            lo<=Float64(expected["actual_empty_CMU_inner_distance"])+1e-7 &&
                Float64(expected["actual_empty_CMU_inner_distance"])<=hi+1e-7 || error("Pooled recorded inner loss differs")
            rp_assert_close(rhi,expected["fixed_RUM_distance"],"Pooled RUM upper";atol=1e-7)
        end
        push!(profiles,Dict("cap"=>cap,"RUM_distance_interval"=>[rlo,rhi],"inner_distance_interval"=>[lo,hi],
            "gain_interval"=>[max(0.,rlo-hi),max(0.,rhi-lo)],
            "distance_reduction_fraction_interval"=>[max(0.,1-hi/rlo),max(0.,1-lo/rhi)]))
    end
    profiles
end
function rp_main(args)
    opt=rp_options(args);data=abspath(opt["data"]);output=abspath(opt["output"])
    ispath(output) && error("Output directory must be new")
    fixture=rp_check_fixture(data);mode=opt["mode"]
    before=rp_sha(joinpath(data,"fixture_manifest.json"));mkpath(output)
    BLAS.set_num_threads(1)
    if mode=="recorded"
        report=rp_recorded(data,fixture)
        report["fixture_manifest_sha256"]=before
        rp_new(joinpath(output,"summary.json"),report)
        println("Verified every registered joint tail at c=.5,1,2; output $output")
        return report
    end
    cells=opt["cells"]=="all" ? collect(1:19) : parse.(Int,split(opt["cells"],','))
    allunique(cells)&&all(c->c in 1:19,cells) || error("Invalid cells")
    mode=="bootstrap"&&sort(cells)!=collect(1:19)&&error("Functional joint rerun requires all19")
    rate=Float64(fixture["rate"]);weights=Dict(parse(Int,String(k))=>Float64(v) for (k,v) in pairs(fixture["weights"]))
    rows=Any[];we=Any[];ce=Any[];workers=min(6,Threads.nthreads())
    for cell in cells
        w=only(filter(w->Int(w["cell"])==cell,fixture["windows"]))
        println("Fresh projections: cell $cell");flush(stdout)
        row,Ar,Aw,groups=rp_point_window(w,fixture);push!(rows,row)
        rp_new(joinpath(output,"cell_$(lpad(cell,2,'0')).json"),row)
        if mode=="bootstrap"
            b=rp_read(joinpath(data,String(w["bootstrap_file"])))
            stars=reshape(Float64.(b["raw_affine_bootstrap"]),Int(b["coordinates"]),Int(b["B"]))
            p=Float64.(w["raw_affine_point"])
            push!(we,wnj_window_envelopes(cell,p,stars,Ar,Aw,rate,[.5,1.,2.]./sqrt(rate);workers,tolerance=1e-8))
            push!(ce,csfi_window_envelopes(cell,p,stars,Ar,groups,rate;workers,tolerance=1e-8,store_draw_certificates=false))
            rp_new(joinpath(output,"envelopes_$(lpad(cell,2,'0')).json"),Dict("WARP"=>we[end],"CMU_Psi"=>ce[end]))
        end
        GC.gc()
    end
    report=Dict{String,Any}("status"=>"completed_fresh_$(mode)_recalculation","cells"=>cells,
        "all19_completed"=>sort(cells)==collect(1:19),"pooled_profiles"=>rp_pooled(rows,fixture;complete=sort(cells)==collect(1:19)),
        "household_estimators_refit"=>false,"original_aggregate_bootstrap_probabilities_reused"=>mode=="bootstrap",
        "fixture_manifest_sha256"=>before,"actual_empty_share_estimated"=>false,
        "uniform_sampling_validity_or_optimal_c_claimed"=>false)
    if mode=="bootstrap"
        expected=rp_read(joinpath(data,"recorded_joint_inference.json"));tests=Dict{String,Any}()
        for alpha in (.05,.10)
            warp=wnj_aggregate(we,weights,rate,[.5,1.,2.]./sqrt(rate);alpha)
            cmu=csfi_aggregate(ce,weights,rate;alpha)
            # New warm-state numerical interval endpoints need not be identical,
            # but all fixed draw counts and economic decisions must reproduce.
            for (actual,old) in ((warp,expected["recorded_WARP_comparisons"][string(alpha)]),
                                 (cmu,expected["recorded_CMU_comparisons"][string(alpha)]))
                for (r,s) in zip(actual["epsilon_results"],old["epsilon_results"])
                    r["exceedances"]==s["exceedances"] || error("Fresh QP joint tail differs; do not silently replace")
                    r["reject_conditional"]==s["reject_conditional"] || error("Fresh QP decision differs")
                end
            end
            tests[string(alpha)]=Dict("ordinary_complete_WARP"=>warp,"actual_empty_CMU_positive_cap_null_Psi"=>cmu)
        end
        report["joint_tests"]=tests;report["all_fresh_joint_tail_counts_match_recorded"] = true
    end
    before==rp_sha(joinpath(data,"fixture_manifest.json")) || error("Fixture changed during run")
    rp_new(joinpath(output,"summary.json"),report)
    println("Completed $mode recalculation: $output")
    report
end
abspath(PROGRAM_FILE)==abspath(@__FILE__)&&rp_main(ARGS)
