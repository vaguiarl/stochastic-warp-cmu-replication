#!/usr/bin/env julia
# New null-equivalent joint comparison. Existing reports and inputs are read-only.
include(joinpath(@__DIR__, "verify_cmu_actual_empty_comparison.jl"))
include(joinpath(@__DIR__, "cmu_superiority_floor_inference.jl"))

const CSFR_ROOT = @__DIR__
const CSFR_ACTUAL = joinpath(CSFR_ROOT, "results/cmu_actual_empty_comparison_20261011_v1")
const CSFR_RAW = joinpath(CSFR_ROOT, "results/wilfried_joint_gap_20261010_v1")
const CSFR_OUTPUT = joinpath(CSFR_ROOT, "results/cmu_superiority_floor_20261012_v1")
const CSFR_VALIDATION = joinpath(CSFR_ROOT, "results/cmu_superiority_floor_validation_20261012_v1.json")
const CSFR_MULTIPLIERS = [.5, 1., 2.]
const CSFR_RATE = sqrt(46391 / 3)

function csfr_new(path, object)
    ispath(path) && error("refusing overwrite: $path")
    open(path, "w") do io
        JSON3.pretty(io, object)
    end
end

function csfr_hashes(actual)
    paths = Set{String}()
    for field in ("protected_sha256_after", "implementation_sha256_after")
        for (relative, _) in pairs(actual[field])
            push!(paths, normpath(joinpath(CSFR_ROOT, String(relative))))
        end
    end
    for name in ("summary.json", "frozen_protocol.json", "independent_audit.json",
                 "conservative_joint_calibration.json")
        push!(paths, joinpath(CSFR_ACTUAL, name))
    end
    for cell in 1:19
        push!(paths, joinpath(CSFR_ACTUAL, "cell_$(lpad(cell, 2, '0')).json"))
    end
    for name in ("verify_cmu_actual_empty_comparison.jl", "cmu_superiority_floor_inference.jl",
                 "run_cmu_superiority_floor.jl", "validate_cmu_superiority_floor.jl",
                 "test/test_cmu_superiority_floor_inference.jl",
                 "test/test_cmu_superiority_floor_validation.jl",
                 "results/wilfried_joint_gap_20261010_v1/frozen_protocol_and_calendar_digests.json",
                 "../../../../docs/CMUEmptySuperiority.lean")
        push!(paths, normpath(joinpath(CSFR_ROOT, name)))
    end
    push!(paths, CSFR_VALIDATION)
    Dict(relpath(path, CSFR_ROOT) => vcae_sha(path) for path in paths)
end

function csfr_main(args)
    args == ["--frozen-all19"] || error("requires --frozen-all19; no sample, cap, c or draw overrides")
    Threads.nthreads() >= 6 || error("six Julia threads required")
    ispath(CSFR_OUTPUT) && error("refusing existing output directory")
    validation = vcae_read(CSFR_VALIDATION)
    validation["status"] == "passed_prespecified_gaussian_size_gate" || error("controlled validation has not passed")
    validation["engine_integration"]["passed"] || error("independent engine integration has not passed")
    aliases = Dict("validation_source"=>"validate_cmu_superiority_floor.jl",
                   "inference_engine"=>"cmu_superiority_floor_inference.jl")
    for (relative, digest) in pairs(validation["implementation_sha256"])
        path = get(aliases, String(relative), String(relative))
        vcae_sha(normpath(joinpath(CSFR_ROOT, path))) == String(digest) ||
            error("validated implementation changed: $relative")
    end
    actual = vcae_read(joinpath(CSFR_ACTUAL, "summary.json"))
    # This also regenerates the canonical year-by-draw seed and count digests,
    # verifies all raw bootstrap matrices, and audits the prior exact model fits.
    audit = verify_cmu_actual_empty_comparison(CSFR_ACTUAL; joint_error_envelope=true)
    before = csfr_hashes(actual)
    protocol = Dict(
        "target" => "equal population raw-target distances to fixed RUM and actual-empty-CMU inner sets",
        "null_equivalent_functional" => "Psi=sum_w alpha_w*(d_R_w(raw_w)^2-d_S_w(raw_w)^2)",
        "null_equivalence" => "beta*Psi_w <= actual_positive_cap_gain_w <= Psi_w, for every raw target and fixed beta>0",
        "S" => "unchanged product of original budget patch simplices; floor subtracted from distances, NOT a replacement estimation target",
        "windows" => collect(1:19), "N_unique" => 46391, "B" => 1000,
        "window_weights" => [w["annual_n_sum"] / 46391 for w in actual["windows"]],
        "rate" => CSFR_RATE, "rate_convention" => "sqrt(46391/3), unchanged from the previous joint profile",
        "epsilon_multipliers" => CSFR_MULTIPLIERS, "primary_epsilon_multiplier" => 1.,
        "epsilon" => CSFR_MULTIPLIERS ./ sqrt(CSFR_RATE),
        "derivative" => "[Psi(raw_hat+h*r*(raw_star-raw_hat))-Psi(raw_hat)]/h^2; signed; no extra half factor",
        "calibration" => "Chen--Fang numerical second-order derivative with exact objective envelopes",
        "reference" => "https://arxiv.org/abs/1901.04861",
        "alphas" => [.05, .10], "primary_alpha" => .05,
        "c_selected_by_real_data_tail" => false,
        "frozen_for_this_run_not_claimed_prospective_preregistration" => true,
        "all_fixed_positive_caps_share_one_null" => true,
        "zero_cap_excluded_different_null" => true,
        "variable_response_floor_LPs_are_not_this_distance_test" => true,
        "no_Bonferroni_or_Holm" => true,
        "no_old_model_p_values_reused" => true,
        "all_original_coordinates_retained" => true,
        "covariance_inverse_ridge_or_rank_truncation" => false,
        "prices_geometry_CF_stages_sample_filters_and_raw_targets_unchanged" => true,
        "joint_calendar_pairing_verified" => true,
        "bootstrap_draws_refit_both_CF_stages_in_the_original_frozen_cache" => true,
        "failed_or_replaced_draws_permitted" => false,
        "sampling_validity" => "conditional on valid joint CF estimator bootstrap, suitable rate, negligible bias and critical-value regularity; not established by Lean or Gaussian validation",
        "raw_pseudo_target_caveat" => "if fixed-degree CF approximations do not target true probabilities, the conclusion concerns the maintained raw population estimand",
        "economic_limits" => "not a test of measured empty-choice frequencies, CMU adequacy, robust demand existence, or a single 25-budget preference coupling",
        "Lean_scope" => "seven scalar mixture/sandwich/null consumers only; geometric and sampling bridges remain explicit"
    )
    mkpath(CSFR_OUTPUT)
    csfr_new(joinpath(CSFR_OUTPUT, "frozen_protocol.json"),
        Dict("protocol" => protocol, "source_and_implementation_sha256_before" => before))
    csfr_new(joinpath(CSFR_OUTPUT, "input_audit.json"), audit)
    BLAS.set_num_threads(1)
    envelopes = Any[]
    started = time()
    for cell in 1:19
        try
            old = actual["windows"][cell]
            old["cell"] == cell || error("window ordering changed")
            cache = vcae_read(joinpath(CSFR_RAW, "raw_bootstrap_cell_$(lpad(cell, 2, '0')).json"))
            X = reduce(vcat, permutedims(Int8.(row)) for row in old["sign_matrix"])
            groups = KSReplication._patch_groups(X)
            Ar = enumerate_types_fast(X; restriction=:rum, max_types=2_000_000, max_nodes=50_000_000)
            size(Ar, 2) == old["types"]["complete_RUM"] || error("RUM family changed")
            raw = Float64.(cache["raw_affine_point"])
            stars = reshape(Float64.(cache["raw_affine_bootstrap"]), length(raw), 1000)
            raw == Float64.(old["raw_affine_point_unchanged"]) || error("point target changed")
            println("Null-equivalent superiority window $cell: $(first(old["years"]))-$(last(old["years"])); $(size(Ar, 2)) RUM types")
            flush(stdout)
            envelope = csfi_window_envelopes(cell, raw, stars, Ar, groups, CSFR_RATE;
                multipliers=CSFR_MULTIPLIERS, workers=6, tolerance=1e-8, store_draw_certificates=true)
            envelope["saved_positive_cap_bridge_check"] = csfi_check_saved_bridge(old)
            push!(envelopes, envelope)
            csfr_new(joinpath(CSFR_OUTPUT, "cell_$(lpad(cell, 2, '0')).json"), envelope)
        catch err
            csfr_new(joinpath(CSFR_OUTPUT, "failed_checkpoint.json"),
                Dict("failed_cell" => cell, "error" => sprint(showerror, err, catch_backtrace()),
                     "completed_cells" => collect(1:length(envelopes)), "failed_draws_not_omitted" => true))
            rethrow()
        end
    end
    weights = protocol["window_weights"]
    comparison = Dict("0.05" => csfi_aggregate(envelopes, weights, CSFR_RATE; alpha=.05, require_full_family=true),
                      "0.10" => csfi_aggregate(envelopes, weights, CSFR_RATE; alpha=.10, require_full_family=true))
    after = csfr_hashes(actual)
    before == after || error("a protected input or implementation changed during the run")
    summary = Dict("status" => "completed_conditional_null_equivalent_superiority_test",
        "protocol" => protocol, "joint_tests" => comparison,
        "checkpoint_files" => ["cell_$(lpad(c, 2, '0')).json" for c in 1:19],
        "checkpoint_sha256" => [vcae_sha(joinpath(CSFR_OUTPUT, "cell_$(lpad(c, 2, '0')).json")) for c in 1:19],
        "source_and_implementation_sha256_before" => before,
        "source_and_implementation_sha256_after" => after,
        "prior_descriptive_results_unmodified" => true, "elapsed_seconds" => time()-started)
    csfr_new(joinpath(CSFR_OUTPUT, "summary.json"), summary)
    println("Saved fresh null-equivalent comparison: $CSFR_OUTPUT")
    summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    csfr_main(ARGS)
end
