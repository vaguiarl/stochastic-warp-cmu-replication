#!/usr/bin/env julia
# Export an explicitly allowlisted aggregate-data release. Never reads household
# microdata, MATLAB source, correspondence, manuscripts, or credentials.
using JSON3, SHA

function options(args)
    iseven(length(args)) || error("Expected --name value pairs")
    out=Dict("source-root"=>"", "output"=>"")
    for i in 1:2:length(args)
        startswith(args[i],"--") || error("Expected --name")
        key=args[i][3:end]; haskey(out,key) || error("Unknown argument $key")
        out[key]=args[i+1]
    end
    all(!isempty,values(out)) || error("Required: --source-root SOURCE_EMPIRICAL_DIRECTORY --output NEW_DATA_DIRECTORY")
    out
end
readjson(path)=JSON3.read(read(path,String))
digest(path)=bytes2hex(sha256(read(path)))
pick(obj,fields)=Dict{String,Any}(String(k)=>obj[String(k)] for k in fields if haskey(obj,String(k)))
function write_new(path,obj)
    ispath(path) && error("Refusing to overwrite $path")
    mkpath(dirname(path))
    open(path,"w") do io; JSON3.write(io,obj); println(io); end
end

function compact_joint(test)
    obj=pick(test,["alpha","statistic","statistic_lower_numerical","statistic_upper_numerical",
        "observed_statistic_lower","observed_statistic_upper","observed_statistic_lower_exact",
        "observed_statistic_upper_exact","primary_epsilon_multiplier","B"])
    obj["epsilon_results"]=[pick(r,["epsilon_multiplier","c_multiplier","exceedances","B",
        "p_value_mc","critical_value_upper","central_critical_value","critical_value_positive_resolved",
        "inferential_p_value_conditional","reject_conditional","calibration_status"])
        for r in test["epsilon_results"]]
    obj
end

function main(args)
    opt=options(args); source=abspath(opt["source-root"]); output=abspath(opt["output"])
    ispath(output) && error("Output directory must be new: $output")
    result=joinpath(source,"results")
    hashes=Dict{String,String}()
    function input(relative)
        path=joinpath(result,relative)
        hashes[relative]=digest(path)
        readjson(path)
    end
    cmu=input("cmu_actual_empty_comparison_20261011_v1/summary.json")
    warp_joint=input("wilfried_normalized_joint_significance_20261010_v1/summary.json")
    cmu_joint=input("cmu_superiority_floor_20261012_v1/summary.json")
    source_joint=input("wilfried_joint_gap_20261010_v1/summary.json")
    N=Int(warp_joint["protocol"]["N_unique"]); N==46391 || error("Registered unique sample size changed")
    windows=Any[]; warp_envelopes=Any[]; cmu_envelopes=Any[]; source_envelopes=Any[]
    witness_fields=["version","code_sha256","geometry_key","row","chosen_budget","anchor",
        "patch_signs","chosen_budgets","empty_budgets","utilities","prices_exact_binary_rational",
        "budget_certificates","analytic_majority_core_bridge_Lean_verified",
        "arithmetic_whole_budget_certificate_recomputed"]
    for cell in 1:19
        suffix="cell_$(lpad(cell,2,'0')).json"
        old=input("wilfried_windows_20261009_v2/$suffix")
        cache=input("wilfried_joint_gap_20261010_v1/raw_bootstrap_$suffix")
        actual=input("cmu_actual_empty_comparison_20261011_v1/$suffix")
        Int(old["cell"])==Int(cache["cell"])==Int(actual["cell"])==cell || error("Cell registry mismatch")
        old["years"]==cache["years"]==actual["years"] || error("Year registry mismatch")
        B=Int(cache["B"]); B==1000 || error("Expected complete 1000-draw cache")
        coordinates=Int(cache["coordinates"])
        length(cache["raw_affine_bootstrap"])==B*coordinates || error("Incomplete draw matrix")
        cache["refit_diagnostics"]["failed_or_replaced_draws"]==0 || error("Failed/replaced source draw")
        cache["refit_diagnostics"]["both_CF_stages_refit"]===true || error("Source cache did not refit both stages")
        bootstrap_name="bootstrap/raw_bootstrap_$suffix"
        write_new(joinpath(output,bootstrap_name),Dict("cell"=>cell,"years"=>cache["years"],
            "B"=>B,"coordinates"=>coordinates,"matrix_storage"=>"column_major_coordinates_by_B",
            "raw_affine_bootstrap"=>cache["raw_affine_bootstrap"],
            "paired_across_windows_by_original_calendar_year_and_draw"=>true,
            "source_probability_matrix_sha256"=>cache["raw_affine_bootstrap_sha256"]))
        raw=Float64.(cache["raw_affine_point"])
        raw==Float64.(actual["raw_affine_point_unchanged"]) || error("Raw source point changed")
        profiles=[Dict("cap"=>p["empty_allowance_cap"],
            "distance_lower"=>p["CMU_actual_empty_inner"]["squared_distance_lower"],
            "distance_upper"=>p["CMU_actual_empty_inner"]["squared_distance_upper"],
            "gain_lower"=>p["fixed_RUM_gain_lower"],"gain_upper"=>p["fixed_RUM_gain_upper"])
            for p in actual["profiles"]]
        source_models=Dict(k=>pick(old["models"][k],["types","J_unrounded","J_rounded_6",
            "observed_J_bounds","p_value_tau0_diagnostic","tail_count","bootstrap_J_rounded_6"])
            for k in ("RUM","correct_WARP"))
        push!(windows,Dict("cell"=>cell,"years"=>old["years"],"annual_n"=>old["annual_n"],
            "sign_matrix"=>old["sign_matrix"],"prices_column_major"=>old["saved_prices"],
            "price_dimensions"=>[7,3],"raw_affine_point"=>raw,
            "source_clipped_point"=>old["pi_hat_regenerated"],
            "original_window_weight"=>sum(Int.(old["annual_n"]))/N,
            "complete_types"=>Dict("RUM"=>old["models"]["RUM"]["types"],
                "WARP"=>old["models"]["correct_WARP"]["types"]),
            "bootstrap_file"=>bootstrap_name,"bootstrap_file_sha256"=>digest(joinpath(output,bootstrap_name)),
            "single_budget_witnesses"=>[pick(w,witness_fields) for w in actual["actual_empty_single_budget_witnesses"]],
            "all_null_witness"=>pick(actual["actual_empty_all_null_witness"],witness_fields),
            "recorded_receipts_not_new_estimates"=>Dict("source_cone_models"=>source_models,
                "raw_normalized_RUM_distance_lower"=>actual["fixed_RUM"]["squared_distance_lower"],
                "raw_normalized_RUM_distance_upper"=>actual["fixed_RUM"]["squared_distance_upper"],
                "inner_profiles"=>profiles)))
        w=input("wilfried_normalized_joint_significance_20261010_v1/$suffix")["envelope"]
        push!(warp_envelopes,pick(w,["cell","analytic_identical_models","point_raw","point_lower",
            "point_upper","derivative_central","derivative_lower","derivative_upper"]))
        s=input("wilfried_joint_gap_20261010_v1/$suffix")["envelope"]
        push!(source_envelopes,pick(s,["cell","analytic_identical_models","point_raw","point_lower",
            "point_upper","derivative_central","derivative_lower","derivative_upper"]))
        c=input("cmu_superiority_floor_20261012_v1/$suffix")
        push!(cmu_envelopes,pick(c,["cell","B","c_dimension","point_raw","point_lower","point_upper",
            "point_lower_exact","point_upper_exact","derivative_central","derivative_lower","derivative_upper",
            "derivative_lower_exact","derivative_upper_exact","protocol"]))
    end
    weights=Dict(string(w["cell"])=>w["original_window_weight"] for w in windows)
    warp_reference=Dict(string(a)=>compact_joint(warp_joint[a==.05 ? "comparison_5_percent" : "comparison_10_percent"])
        for a in (.05,.10))
    cmu_reference=Dict(string(a)=>compact_joint(cmu_joint["joint_tests"][a==.05 ? "0.05" : "0.10"]) for a in (.05,.10))
    write_new(joinpath(output,"uk_aggregate_fixture.json"),Dict("schema"=>"stochastic-warp-cmu-public-aggregates-v1",
        "N_unique"=>N,"rate"=>sqrt(N/3),"weights"=>weights,"caps"=>[0.,.005,.01,.025,.05,.10,.20],
        "windows"=>windows,"recorded_pooled_inner_profiles"=>cmu["weighted_profiles"],
        "original_source_content_sha256"=>hashes,
        "export_scope"=>"Aggregate sign geometry, prices, counts, estimated probabilities and hypothetical utility witnesses only; no household records, identifiers, per-household bootstrap frequencies, emails or third-party source",
        "target_policy"=>"Raw affine probabilities preserved; no normalization or clipping for normalized-hull/CMU comparison",
        "common_response_is_maintained_selection_sensitivity_not_measured_absence"=>true))
    write_new(joinpath(output,"recorded_joint_inference.json"),Dict("schema"=>"recorded-joint-envelope-replay-v1",
        "B"=>1000,"c_grid"=>[.5,1.,2.],"primary_c"=>1.,"rate"=>sqrt(N/3),"weights"=>weights,
        "WARP_envelopes"=>warp_envelopes,"CMU_Psi_envelopes"=>cmu_envelopes,
        "source_clipped_cone_envelopes"=>source_envelopes,
        "recorded_WARP_comparisons"=>warp_reference,"recorded_CMU_comparisons"=>cmu_reference,
        "recorded_source_clipped_cone_comparison_5_percent"=>compact_joint(source_joint["comparison"]),
        "scope"=>"Reaggregation of stored computed derivative envelopes checks all tails/weights/calibration arithmetic; it does not refit household estimators or re-solve projection QPs"))
    manifest=Dict("schema"=>"aggregate-fixture-file-digests-v1","files"=>Dict{String,String}())
    for (dir,_,files) in walkdir(output),f in files
        path=joinpath(dir,f); manifest["files"][relpath(path,output)]=digest(path)
    end
    write_new(joinpath(output,"fixture_manifest.json"),manifest)
    println("Exported 19 windows and all 1000 paired aggregate draws to $output")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && main(ARGS)
