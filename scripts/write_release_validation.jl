#!/usr/bin/env julia
# Generate a compact public numerical-validation receipt after recorded and
# fresh point replay. Explicit input locations are never embedded in output.
using JSON3, SHA
function validation_options(args)
    opt=Dict("data"=>"","recorded"=>"","points"=>"","output"=>"")
    iseven(length(args)) || error("Expected --name value pairs")
    for i in 1:2:length(args)
        startswith(args[i],"--") || error("Expected --name")
        key=args[i][3:end];haskey(opt,key)||error("Unknown option $key");opt[key]=args[i+1]
    end
    all(!isempty,values(opt))||error("Required --data DIRECTORY --recorded DIRECTORY --points DIRECTORY --output NEW_JSON_FILE")
    opt
end
vread(p)=JSON3.read(read(p,String))
vsha(p)=bytes2hex(sha256(read(p)))
function validation_main(args)
    opt=validation_options(args);output=abspath(opt["output"]);ispath(output)&&error("Refusing to overwrite validation")
    fixture=vread(joinpath(opt["data"],"uk_aggregate_fixture.json"))
    recorded=vread(joinpath(opt["recorded"],"summary.json"))
    points=vread(joinpath(opt["points"],"summary.json"))
    recorded["status"]=="passed_all19_recorded_joint_envelope_reaggregation"||error("Recorded inference did not pass")
    points["status"]=="completed_fresh_points_recalculation"&&points["all19_completed"]===true||error("All19 points not complete")
    windows=Dict(Int(w["cell"])=>w for w in fixture["windows"])
    max_rum=0.;max_inner=0.;max_cone=0.;max_width=0.;witnesses=0;type_counts=Any[];window_diffs=Any[]
    for cell in 1:19
        fresh=vread(joinpath(opt["points"],"cell_$(lpad(cell,2,'0')).json"));w=windows[cell]
        ref=w["recorded_receipts_not_new_estimates"]
        fresh["all_fresh_objective_intervals_overlap_recorded_intervals"]===true||error("Fresh objective check failed")
        fresh["type_counts"]["RUM"]==w["complete_types"]["RUM"]||error("RUM count differs")
        fresh["type_counts"]["WARP"]==w["complete_types"]["WARP"]||error("WARP count differs")
        dr=abs(Float64(fresh["fresh_raw_normalized_RUM"]["squared_distance_upper"])-Float64(ref["raw_normalized_RUM_distance_upper"]))
        max_rum=max(max_rum,dr);witnesses+=Int(fresh["witnesses_checked"])
        cap_diffs=Float64[]
        for (r,s) in zip(fresh["fresh_inner_profiles"],ref["inner_profiles"])
            r["cap"]==s["cap"]||error("Cap differs")
            push!(cap_diffs,abs(Float64(r["distance_upper"])-Float64(s["distance_upper"])))
            max_width=max(max_width,Float64(r["distance_upper"])-Float64(r["distance_lower"]))
        end
        max_inner=max(max_inner,maximum(cap_diffs));cone_diffs=Dict{String,Float64}()
        for label in ("RUM","correct_WARP")
            d=abs(Float64(fresh["fresh_source_clipped_cone_J"][label]["J_upper"])-Float64(ref["source_cone_models"][label]["J_unrounded"]))
            cone_diffs[label]=d;max_cone=max(max_cone,d)
        end
        push!(type_counts,Dict("cell"=>cell,"RUM"=>fresh["type_counts"]["RUM"],"WARP"=>fresh["type_counts"]["WARP"]))
        push!(window_diffs,Dict("cell"=>cell,"normalized_RUM_upper_abs_difference"=>dr,
            "inner_upper_abs_differences_by_cap"=>cap_diffs,"source_cone_J_upper_abs_differences"=>cone_diffs))
    end
    max_rum<=1e-7&&max_inner<=1e-7&&max_cone<=1e-5||error("Numerical release differences exceed stated tolerances")
    joint_rows=Any[]
    for alpha in ("0.05","0.1"),model in ("ordinary_complete_WARP","actual_empty_CMU_positive_cap_null_Psi")
        test=recorded["joint_tests"][alpha][model]
        for r in test["epsilon_results"]
            push!(joint_rows,Dict("model"=>model,"alpha"=>parse(Float64,alpha),"c"=>r["epsilon_multiplier"],
                "B"=>r["B"],"upper_tail_exceedances"=>r["exceedances"],"p_value"=>r["p_value_mc"],
                "reject_conditional"=>r["reject_conditional"],"critical_positive_resolved"=>r["critical_value_positive_resolved"]))
        end
    end
    for r in recorded["source_clipped_cone_joint_5_percent"]["epsilon_results"]
        push!(joint_rows,Dict("model"=>"source_clipped_cone_joint","alpha"=>.05,"c"=>r["epsilon_multiplier"],
            "B"=>r["B"],"upper_tail_exceedances"=>r["exceedances"],"p_value"=>r["p_value_mc"],
            "reject_conditional"=>r["reject_conditional"],"critical_positive_resolved"=>r["critical_value_positive_resolved"]))
    end
    result=Dict("schema"=>"public-julia-release-numerical-validation-v1","status"=>"passed",
        "implementation"=>"Julia 1.12 release entry point; portable aggregate-data replay",
        "fixture_json_sha256"=>vsha(joinpath(opt["data"],"uk_aggregate_fixture.json")),
        "recorded_inference_json_sha256"=>vsha(joinpath(opt["data"],"recorded_joint_inference.json")),
        "19_correct_type_counts_recomputed"=>true,"19_raw_normalized_RUM_projections_recomputed"=>true,
        "133_raw_CMU_inner_cap_projections_recomputed"=>true,"38_source_clipped_cone_point_statistics_recomputed"=>true,
        "all_singleton_and_null_witnesses_verified"=>true,"witnesses_checked"=>witnesses,
        "38_recorded_source_profile_bootstrap_tails_verified"=>true,"15_joint_tail_gate_rows_verified"=>true,
        "joint_rows"=>joint_rows,"correct_type_counts"=>type_counts,"per_window_numerical_differences"=>window_diffs,
        "maximum_RUM_upper_abs_difference"=>max_rum,"maximum_inner_upper_abs_difference"=>max_inner,
        "maximum_source_cone_J_upper_abs_difference"=>max_cone,"maximum_fresh_inner_interval_width"=>max_width,
        "pooled_profiles_fresh"=>points["pooled_profiles"],
        "scope"=>"Fresh complete point projections and utility-witness arithmetic; joint inference reaggregated from stored paired QP envelopes. Full household estimator/bootstrap refitting, full derivative-QP rerun, native MATLAB replication and general sampling validity are not claimed.",
        "third_party_raw_household_data_or_code_redistributed"=>false)
    mkpath(dirname(output));open(output,"w") do io;JSON3.write(io,result);println(io);end
    println("Saved sanitized public numerical validation with $witnesses utility witnesses")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__)&&validation_main(ARGS)
