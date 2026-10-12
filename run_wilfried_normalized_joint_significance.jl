#!/usr/bin/env julia
# New joint inference for the frozen normalized-hull distance, not old cone tails.
include(joinpath(@__DIR__,"wilfried_normalized_joint_significance.jl"))

const WNJ_OUTPUT = joinpath(@__DIR__,"results/wilfried_normalized_joint_significance_20261010_v1")

function wnj_source_guard()
    hashes=wnr_protected_hashes()
    for name in ("run_wilfried_normalized_fit_robustness.jl",
            "results/wilfried_normalized_fit_robustness_20261010_v1.json",
            "wilfried_normalized_joint_significance.jl","wilfried_normalized_working_set.jl",
            "run_wilfried_normalized_joint_significance.jl",
            "test/test_wilfried_normalized_joint_significance.jl",
            "test/test_wilfried_normalized_working_set.jl")
        hashes[name]=wnr_sha(joinpath(@__DIR__,name))
    end
    for path in (get(ENV,"WILFRIED_ARCHIVE",joinpath(@__DIR__,"external/RUMCODEFINAL.zip")),get(ENV,"FES_DATA_ARCHIVE",joinpath(@__DIR__,"external/231552-V1.zip")),
            normpath(joinpath(@__DIR__,"../../lean/NormalizedGapNullEnvelope.lean")))
        hashes[path]=wnr_sha(path)
    end
    hashes
end

function wnj_main(args)
    args==["--write-approved-report"] || error("frozen primary profile; no window, tuning, draw or target overrides")
    Threads.nthreads()>=6 || error("six worker-local Julia threads required")
    ispath(WNJ_OUTPUT) && error("refusing an existing output directory")
    wnr_sha(WNR_PRIOR)==WNR_PRIOR_SHA && wnr_sha(joinpath(WNR_JOINT,"summary.json"))==WNR_JOINT_SHA &&
        wnr_sha(WNR_OUTPUT)==WNJ_NORMALIZED_SHA || error("immutable earlier reports changed")
    before=wnj_source_guard()
    prior=JSON3.read(read(WNR_PRIOR,String))
    joint=JSON3.read(read(joinpath(WNR_JOINT,"summary.json"),String))
    calendar=JSON3.read(read(joinpath(WNR_JOINT,"frozen_protocol_and_calendar_digests.json"),String))
    cases=sort(collect(prior["cases"]);by=c->Int(c["cell"]))
    Int.([c["cell"] for c in cases])==collect(1:19) || error("all19 windows required")
    protocol0=joint["protocol"]
    N=Int(protocol0["N_unique"])
    N==46391 && Int(protocol0["B"])==1000 && Int(protocol0["seed"])==13732 || error("saved joint design changed")
    calendar["protocol"]==protocol0 || error("joint calendar protocol mismatch")
    rate=sqrt(N/3)
    epsilons=WNJ_MULTIPLIERS./sqrt(rate)
    weights=Dict(Int(c["cell"])=>sum(Int.(c["annual_n"]))/N for c in cases)
    records=Dict{Int,Any}()
    raw_hashes=Dict{Int,String}()
    digests=reshape(String.(calendar["calendar_count_digests"]),25,1000)
    for old in cases
        cell=Int(old["cell"])
        path=joinpath(WNR_JOINT,"raw_bootstrap_cell_$(lpad(cell,2,'0')).json")
        record=JSON3.read(read(path,String))
        record["cell"]==cell && record["years"]==old["years"] || error("saved point window alignment changed")
        restored=wnj_saved_matrix(record)
        maximum(abs.(clamp.(restored.point,0,1).-Float64.(old["pi_hat_regenerated"])))<=1e-12 || error("saved source point changed")
        year_rows=Int.(record["canonical_year_seed_rows"])
        year_rows==Int.(old["years"]).-1974 || error("canonical joint year rows changed")
        bytes2hex(sha256(JSON3.write(digests[year_rows,:])))==String(record["year_count_digests_sha256"]) ||
            error("paired calendar bootstrap count digest changed")
        records[cell]=restored
        raw_hashes[cell]=wnr_sha(path)
    end
    protocol=Dict("cells"=>collect(1:19),"B"=>1000,"seed"=>13732,"N_unique"=>N,
        "common_rate"=>rate,"window_weights"=>weights,"epsilon_multipliers"=>WNJ_MULTIPLIERS,
        "epsilons"=>epsilons,"primary_epsilon_multiplier"=>1.0,
        "functional"=>"Phi(a)=sum_j (N_window_j/N_unique)*[dist(a_j,conv(A_Rj))^2-dist(a_j,conv(A_Wj))^2]",
        "observed_statistic"=>"(N_unique/3)*Phi_hat",
        "derivative"=>"[Phi(a_hat+epsilon*r0*(a_star-a_hat))-Phi(a_hat)]/epsilon^2",
        "joint_null"=>"all19 population normalized-hull distance gains are zero; all weights positive",
        "target_policy"=>"raw_affine; no clamp or renormalization",
        "metric"=>"full-coordinate Euclidean identity; type masses nonnegative and sum one",
        "same_saved_joint_draws"=>true,"both_CF_stages_refit_in_saved_draws"=>true,
        "all_probability_coordinates_and_full_type_columns_retained"=>true,
        "c_primary_registered_before_new_tails"=>true,"tuning_selected_by_p_value"=>false,
        "Bonferroni_or_Holm_applied"=>false,"ordinary_centered_gap_bootstrap_used"=>false,
        "zero_point_nonidentical_windows_retained"=>true,"failed_draw_replacement_or_reselection"=>false,
        "fixed_source_prices_medians_trimming_geometry_and_degree3"=>true,
        "exact_point_bounds_numeric_bootstrap_bounds_with_error_propagation"=>true,
        "raw_bootstrap_checkpoint_sha256"=>raw_hashes)
    BLAS.set_num_threads(1)
    mkpath(WNJ_OUTPUT)
    wcf_write_new(joinpath(WNJ_OUTPUT,"frozen_protocol.json"),Dict("protocol"=>protocol,"protected_sha256_before"=>before))
    envelopes=Dict{String,Any}[]
    start=time()
    for old in cases
        cell=Int(old["cell"])
        println("Normalized joint window $cell $(first(old["years"]))-$(last(old["years"]))");flush(stdout)
        try
            X=reduce(vcat,permutedims(Int8.(row)) for row in old["sign_matrix"])
            Ar=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
            Aw=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
            size(Ar,2)==old["models"]["RUM"]["types"] && size(Aw,2)==old["models"]["correct_WARP"]["types"] || error("complete type counts changed")
            record=records[cell]
            groups=KSReplication._patch_groups(X)
            maximum(abs(sum(record.point[g])-1) for g in groups)<=1e-10 || error("raw point adding-up mismatch")
            maximum(abs(sum(record.stars[g,b])-1) for g in groups for b in axes(record.stars,2))<=1e-10 || error("raw bootstrap adding-up mismatch")
            timed=@timed wnj_window_envelopes(cell,record.point,record.stars,Ar,Aw,rate,epsilons;workers=6)
            envelope=timed.value
            push!(envelopes,envelope)
            wcf_write_new(joinpath(WNJ_OUTPUT,"cell_$(lpad(cell,2,'0')).json"),
                Dict("cell"=>cell,"years"=>old["years"],"complete_types"=>Dict("RUM"=>size(Ar,2),"WARP"=>size(Aw,2)),
                    "envelope"=>envelope,"projection_seconds"=>timed.time))
            println("  gain=$(envelope["point_raw"]); seconds=$(timed.time)");flush(stdout)
        catch err
            wcf_write_new(joinpath(WNJ_OUTPUT,"failed_checkpoint.json"),Dict("status"=>"failed_no_joint_tail",
                "failed_cell"=>cell,"error"=>sprint(showerror,err,catch_backtrace()),
                "completed_cells"=>[e["cell"] for e in envelopes],"failed_cells_not_zero_filled"=>true))
            rethrow()
        end
    end
    comparison05=wnj_aggregate(envelopes,weights,rate,epsilons;alpha=.05)
    comparison10=wnj_aggregate(envelopes,weights,rate,epsilons;alpha=.10)
    safety=wnj_null_error_envelope(records,weights,rate,comparison05["statistic_lower_numerical"])
    after=wnj_source_guard()
    before==after || error("protected source/report/code/manuscript changed; no valid summary")
    report=Dict("status"=>"completed_conditional_normalized_joint_comparison",
        "protocol"=>protocol,"comparison_5_percent"=>comparison05,"comparison_10_percent"=>comparison10,
        "c_free_conservative_companion"=>safety,"elapsed_seconds"=>time()-start,
        "source_integrity_unchanged"=>true,"protected_sha256_before"=>before,"protected_sha256_after"=>after,
        "maintained_unverified_inference_assumptions"=>[
            "Joint annual-iid raw-affine two-stage CF estimator CLT/bootstrap validity and negligible degree3 approximation bias",
            "Positive limiting relative annual and window sample proportions",
            "Polyhedral second-order directional expansion and positive continuous null critical quantile",
            "Conditioning on fixed estimated source prices, medians, trimming and patch geometry",
            "No correction for survey PSU clustering, survey design weights, or uncertainty of fixed source preparation"],
        "finite_sample_UK_coverage_Lean_verified"=>false,"unconditional_survey_design_inference_established"=>false,
        "ordinary_bootstrap_of_centered_distance_gain_used"=>false,"raw_household_identifiers_or_frequency_vectors_exported"=>false,
        "interpretation"=>"One direct joint comparison on all19 normalized nested-hull windows. Primary c=1 remains fixed; c=.5 and2 are sensitivities, never selected for a significance claim. Smaller CMU distance does not imply CMU adequacy or a WARP-pass/RUM-fail split. The c-free companion's deterministic envelope is Lean-checked but its joint UK error calibration remains conditional on the estimator bootstrap. Numerical bootstrap projection envelopes are not formal outward-rounded intervals.")
    wcf_write_new(joinpath(WNJ_OUTPUT,"summary.json"),report)
    println("Normalized joint statistic=",comparison05["statistic"],"; conditional c-grid tails=",
        [r["p_value_mc"] for r in comparison05["epsilon_results"]]);flush(stdout)
    report
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    wnj_main(ARGS)
end
