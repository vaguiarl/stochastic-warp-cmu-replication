#!/usr/bin/env julia
# Extend the frozen focal replication without modifying it or its old reports.
include(joinpath(@__DIR__, "run_wilfried_replication.jl"))
include(joinpath(@__DIR__, "wilfried_working_set_projection.jl"))
include(joinpath(@__DIR__, "wilfried_nnls_projection.jl"))

function ww_options(args)
    opts = Dict("archive"=>get(ENV,"WILFRIED_ARCHIVE",joinpath(@__DIR__,"external/RUMCODEFINAL.zip")),
        "package"=>get(ENV,"WILFRIED_PACKAGE",joinpath(@__DIR__,"external/RUMCODEFINAL")),
        "data-archive"=>get(ENV,"FES_DATA_ARCHIVE",joinpath(@__DIR__,"external/231552-V1.zip")),
        "bootstrap"=>"1000", "seed"=>"13732", "cells"=>"all", "projection"=>"nnls",
        "output-directory"=>joinpath(@__DIR__, "results/wilfried_windows_20261009_v2"))
    iseven(length(args)) || error("arguments must be --name value pairs")
    for i in 1:2:length(args)
        startswith(args[i], "--") || error("expected --option")
        key = args[i][3:end]
        haskey(opts,key) || error("unknown option $key")
        opts[key] = args[i+1]
    end
    return opts
end

function ww_fit_report(A, pi, saved_pi, bootstrap, n, degree; projection="nnls")
    projection in ("nnls","working_set","full_reference") || error("unknown projection method")
    projection=="full_reference" && return wf_fit_report(A,pi,saved_pi,bootstrap,n,degree)
    projector = projection=="nnls" ? NNLSFullConeProjector(A) : CWNoFloorConeProjector(A)
    fit! = projection=="nnls" ? nnls_fit! : cw_fit!
    observed, stored = fit!(projector,pi),fit!(projector,saved_pi)
    scale = sum(n)/degree
    J = wf_round_J(scale*observed.upper)
    J_lower = wf_round_J(scale*observed.lower)
    values, lower_values = zeros(size(bootstrap,2)),zeros(size(bootstrap,2))
    diagnostics = [observed.diagnostic,stored.diagnostic]
    maximum_width = scale*(observed.upper-observed.lower)
    for b in axes(bootstrap,2)
        fit = fit!(projector,bootstrap[:,b]-pi+observed.fitted)
        values[b],lower_values[b] = wf_round_J(scale*fit.upper),wf_round_J(scale*fit.lower)
        maximum_width = max(maximum_width,scale*(fit.upper-fit.lower))
        push!(diagnostics,fit.diagnostic)
        if b%200==0
            println("    completed ",b,"/",size(bootstrap,2)," full-priced projections")
            flush(stdout)
        end
    end
    B = length(values)
    tail = count(>=(J),values)
    p = J==0 ? 1.0 : tail/B
    lower_p = J==0 ? 1.0 : count(>=(J),lower_values)/B
    upper_p = J_lower==0 ? 1.0 : count(>=(J_lower),values)/B
    sorted = sort(values)
    statuses = Dict{String,Int}()
    for d in diagnostics[3:end]
        statuses[d.solver_status] = get(statuses,d.solver_status,0)+1
    end
    return Dict("types"=>size(A,2),"J_unrounded"=>scale*observed.upper,"J_rounded_6"=>J,
        "observed_J_bounds"=>scale.*[observed.lower,observed.upper],
        "observed_J_rounding_bounds"=>[J_lower,J],
        "J_on_original_saved_pihat"=>scale*stored.upper,
        "J_on_original_saved_pihat_bounds"=>scale.*[stored.lower,stored.upper],
        "J_on_original_saved_pihat_rounded_6"=>wf_round_J(scale*stored.upper),
        "p_value_tau0_diagnostic"=>p,"tail_count"=>tail,
        "rounded_tail_numerical_bounds"=>[lower_p,upper_p],
        "rounded_tail_resolved_given_computed_recentered_targets"=>lower_p==upper_p,
        "bounds_do_not_certify_exact_MATLAB_draws_or_exact_recentered_projection"=>true,
        "p_value_mc_se"=>sqrt(p*(1-p)/B),"critical_95"=>sorted[ceil(Int,.95B)],
        "critical_99"=>sorted[ceil(Int,.99B)],"bootstrap_J_rounded_6"=>values,
        "bootstrap_J_lower_rounded_6"=>lower_values,
        "projection_fitted"=>observed.fitted,"projection_mass_sum"=>observed.mass_sum,
        "maximum_primal_dual_gap"=>maximum(d.primal_dual_gap for d in diagnostics),
        "maximum_equality_error"=>maximum(d.equality_error for d in diagnostics),
        "maximum_repaired_J_bound_width"=>maximum_width,
        "observed_projection_vector_error_bound"=>sqrt(observed.upper-observed.lower),
        "floating_dual_feasibility_safety"=>"1e-12*(1+maximum(abs,lambda)) per coordinate",
        "bootstrap_solver_status_counts"=>statuses,"failed_or_dropped_draws"=>0,
        "projection_optimization"=>projection=="nnls" ?
            "warm Lawson-Hanson NNLS; every original column globally priced; feasible primal/full-dual bounds; disclosed complete-model fallbacks" :
            "finite working-set dual constraints, exhaustive full-column pricing, feasible primal/full-dual bounds",
        "projection_diagnostics"=>Dict("backend"=>projection,"fits"=>length(diagnostics),
            "maximum_iterations"=>maximum(d.iterations for d in diagnostics),
            "full_pricing_passes"=>sum(d.full_pricing_passes for d in diagnostics),
            "columns_priced_including_repeated_passes"=>sum(d.priced_columns for d in diagnostics),
            "maximum_working_cuts"=>maximum(d.working_cuts for d in diagnostics),
            "full_type_count"=>size(A,2),"fallbacks_any"=>count(d->d.fallback_used,diagnostics),
            "NNLS_to_working_set_fallbacks"=>count(d->hasproperty(d,:nnls_completed) && !d.nnls_completed,diagnostics),
            "working_set_to_full_QP_fallbacks"=>count(d->hasproperty(d,:full_model_fallback) ? d.full_model_fallback : d.fallback_used,diagnostics),
            "NNLS_least_squares_calls"=>sum(hasproperty(d,:least_squares_calls) ? d.least_squares_calls : 0 for d in diagnostics),
            "NNLS_rank_guard_rejections"=>sum(hasproperty(d,:rank_guard_rejections) ? d.rank_guard_rejections : 0 for d in diagnostics),
            "fallback_reasons"=>unique([d.fallback_reason for d in diagnostics if d.fallback_used]),
            "maximum_repaired_full_dual_violation"=>maximum(d.repaired_full_dual_violation for d in diagnostics)))
end

function ww_columns(A)
    return Set(Tuple.(eachcol(A)))
end

function ww_pricing(pi, Ar, Aw, models, groups, years, scale)
    r, w = models["RUM"], models["correct_WARP"]
    residual = pi-r["projection_fitted"]
    rum_keys = ww_columns(Ar)
    only_warp = findall(h->!(Tuple(Aw[:,h]) in rum_keys),axes(Aw,2))
    scores = Aw'*residual
    worst = isempty(only_warp) ? nothing : only_warp[argmax(scores[only_warp])]
    # RUM's feasible primal is also WARP feasible. Repair the RUM residual to
    # satisfy ALL WARP dual constraints, yielding an independent equality bound.
    repair = max(0.0,maximum(scores))/length(groups)+1e-12*(1+maximum(abs,residual))
    lambda = residual.-repair
    maximum(Aw'*lambda)<=1e-12 || error("all-column repaired pricing dual infeasible")
    warp_lower = scale*max(0.0,2dot(pi,lambda)-sum(abs2,lambda))
    rum_upper = r["observed_J_bounds"][2]
    gain_bounds = [max(0.0,r["observed_J_bounds"][1]-w["observed_J_bounds"][2]),
        max(0.0,r["observed_J_bounds"][2]-w["observed_J_bounds"][1])]
    selected_rows = worst===nothing ? nothing :
        [only(filter(i->Aw[i,worst]==1,g)) for g in groups]
    return Dict("WARP_only_types"=>length(only_warp),
        "maximum_RUM_residual_dot_all_WARP_columns"=>maximum(scores),
        "maximum_RUM_residual_dot_WARP_only_columns"=>worst===nothing ? nothing : scores[worst],
        "maximizing_WARP_only_choice_rows"=>selected_rows,
        "choice_row_budget_years"=>collect(years),
        "fitted_vector_max_difference"=>maximum(abs.(r["projection_fitted"]-w["projection_fitted"])),
        "J_gain_RUM_minus_WARP"=>r["J_unrounded"]-w["J_unrounded"],
        "J_gain_float_primal_dual_bounds"=>gain_bounds,
        "RUM_primal_upper_minus_repaired_all_WARP_dual_lower"=>max(0.0,rum_upper-warp_lower),
        "pricing_descent_lower_bound_on_J_gain"=>worst===nothing ? 0.0 :
            max(0.0,scale*max(0.0,scores[worst])^2/length(groups)-diff(r["observed_J_bounds"])[1]),
        "interpretation"=>"Positive pricing identifies a local improvement direction from a computed RUM-feasible fit; the descent bound subtracts RUM optimization uncertainty. If the exact RUM projection has nonpositive scores against all added columns, the exact projections coincide. Floating bounds quantify computed-fit uncertainty. No identified nontransitive population share is claimed.")
end

function ww_main(args)
    opts = ww_options(args)
    output = abspath(opts["output-directory"])
    ispath(output) && error("refusing to reuse an existing output directory: $output")
    archive, package = abspath(opts["archive"]), abspath(opts["package"])
    wf_sha(archive)==WF_ARCHIVE_SHA || error("received ZIP identity changed")
    target = joinpath(package,"Output",WF_TARGET)
    wf_sha(target)==WF_TARGET_SHA || error("received saved output changed")
    audit_path = joinpath(@__DIR__,"results/wilfried_geometry_audit.20261009_v1.json")
    wf_sha(audit_path)=="655135ed5351db97b5eea6fc7fb549e19ab9fceec1a5abbf2e91e81a2dd22b1f" || error("registered geometry audit changed")
    audit = JSON3.read(read(audit_path,String))
    geometry_cases = audit["cases"]
    before = wf_hashes(package,target)
    additional_files = [@__FILE__,audit_path,joinpath(@__DIR__,"wilfried_working_set_projection.jl"),
        joinpath(@__DIR__,"wilfried_nnls_projection.jl"),
        joinpath(@__DIR__,"test/test_wilfried_windows.jl"),
        joinpath(@__DIR__,"test/test_wilfried_nnls_projection.jl")]
    for path in additional_files
        before[abspath(path)] = wf_sha(path)
    end
    input_hashes = wf_input_identity(package,opts["data-archive"])
    B, seed = parse(Int,opts["bootstrap"]), parse(Int,opts["seed"])
    B>=2 || error("at least two bootstrap repetitions are required")
    BLAS.set_num_threads(1)
    saved = matopen(target) do f
        first_year = Int(read(f,"start_year"))
        first_year<100 && (first_year+=1900)
        (; first_year, X=read(f,"X"), P=read(f,"budgets"), pi=read(f,"pi_hat"),
            J=vec(read(f,"Jstat")),p=vec(read(f,"pval")),tau=vec(read(f,"tau")),
            B=Int(read(f,"bootstrap_reps")),seed=Int(read(f,"seed")))
    end
    cells = opts["cells"]=="all" ? collect(eachindex(saved.X)) : parse.(Int,split(opts["cells"],','))
    allunique(cells) && all(c->c in eachindex(saved.X),cells) || error("invalid or duplicated cells")
    # All windows are selected before observing the new outcomes. Earlier large
    # windows run first for prompt numerical validation, not selective reporting.
    order = sort(cells;by=c->c==13 ? -2 : c==14 ? -1 : c)
    source = load_fes_source(opts["data-archive"])
    group = only(filter(g->g.id=="all",household_group_registry()))
    mkpath(output)
    cases = Dict{String,Any}[]
    start = time()
    for cell in order
        window_start = time()
        X, P, saved_pi = Int8.(saved.X[cell]),Float64.(saved.P[cell]),vec(Float64.(saved.pi[cell]))
        years = (saved.first_year+cell-1):(saved.first_year+cell+size(X,2)-2)
        groups = KSReplication._patch_groups(X)
        saved.tau[cell]==0 || error("window $cell does not use source tau0")
        geometry = only(filter(c->c["role"]=="reported_rpm_output" && c["cell_index"]==cell,geometry_cases))
        Ar = enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
        Aw = enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
        size(Ar,2)==geometry["correct_complete_rum_types"] || error("RUM audit count mismatch")
        size(Aw,2)==geometry["correct_complete_warp_types"] || error("WARP audit count mismatch")
        issubset(ww_columns(Ar),ww_columns(Aw)) || error("RUM/WARP nesting failure")
        prep = prepare_household_group(source,group;years)
        prep.supported || error("window $cell sample unsupported")
        P_new = reduce(vcat,permutedims(a.normalized_prices) for a in prep.annual)
        price_error = maximum(abs.(P_new-P))
        price_error<=1e-12 || error("window $cell saved/regenerated prices differ")
        annual = [KSAnnual(a.year,a.shares,a.expenditure,a.instrument,vec(P[j,:])) for (j,a) in enumerate(prep.annual)]
        estimator = prepare_estimator(annual,X;degree=3,evaluation_log_expenditure=prep.evaluation_log_expenditure)
        point = estimate_prepared_probabilities(estimator;probability_policy=:source_clip)
        pi_error = maximum(abs.(point.pi-saved_pi))
        pi_error<=1e-8 || error("window $cell saved/regenerated probabilities differ: $pi_error")
        n = [y.n for y in estimator.years]
        println("Window $cell $(first(years))-$(last(years)): WARP=$(size(Aw,2)), RUM=$(size(Ar,2)), pi error=$pi_error")
        flush(stdout)
        refit = @timed bootstrap_prepared_probabilities(estimator;repetitions=B,seed,
            parallel=true,probability_policy=:source_clip)
        boot = refit.value
        models = Dict{String,Any}()
        for (name,A) in (("RUM",Ar),("correct_WARP",Aw))
            fit_time = @timed ww_fit_report(A,point.pi,saved_pi,boot.pi,n,3;projection=opts["projection"])
            models[name] = fit_time.value
            models[name]["projection_seconds"] = fit_time.time
            println("  ",name," J=",models[name]["J_rounded_6"],", p=",models[name]["p_value_tau0_diagnostic"],", seconds=",round(fit_time.time;digits=2))
            flush(stdout)
        end
        r, w = models["RUM"],models["correct_WARP"]
        w["observed_J_bounds"][1]<=r["observed_J_bounds"][2]+1e-7 || error("numerically resolved nesting violation")
        case = Dict{String,Any}("status"=>"completed","cell"=>cell,"years"=>collect(years),
            "annual_n"=>n,"geometry_audit"=>geometry,"sign_matrix"=>[collect(row) for row in eachrow(X)],
            "saved_prices"=>P,"maximum_regenerated_price_difference"=>price_error,
            "pi_hat_regenerated"=>point.pi,"pi_hat_original_saved"=>saved_pi,
            "maximum_regenerated_pihat_difference"=>pi_error,
            "clipped_probability_block_sums"=>[sum(point.pi[g]) for g in groups],
            "estimation_diagnostics"=>point.diagnostics,"models"=>models,
            "source_recorded_checker_not_WARP"=>Dict("J"=>saved.J[cell],"p"=>saved.p[cell],
                "tau"=>saved.tau[cell],"B"=>saved.B,"seed"=>saved.seed),
            "RUM_to_WARP_pricing"=>ww_pricing(point.pi,Ar,Aw,models,groups,years,sum(n)/3),
            "bootstrap_refit_seconds"=>refit.time,
            "bootstrap_seed_vector_sha256"=>bytes2hex(sha256(JSON3.write(boot.seeds))),
            "elapsed_seconds"=>time()-window_start)
        push!(cases,case)
        checkpoint = joinpath(output,"cell_$(lpad(cell,2,'0')).json")
        ispath(checkpoint) && error("checkpoint unexpectedly already exists")
        open(checkpoint,"w") do io
            JSON3.pretty(io,case)
            println(io)
        end
        println("  Saved $checkpoint")
        flush(stdout)
    end
    after = wf_hashes(package,target)
    for path in additional_files
        after[abspath(path)] = wf_sha(path)
    end
    before==after || error("frozen source or implementation changed during run")
    wf_sha(archive)==WF_ARCHIVE_SHA || error("ZIP changed during run")
    report = Dict("status"=>"completed","method"=>"all predeclared received-package windows, corrected WARP/RUM Julia replication",
        "archive"=>archive,"archive_sha256"=>WF_ARCHIVE_SHA,"saved_target"=>target,
        "saved_target_sha256"=>WF_TARGET_SHA,"julia_version"=>string(VERSION),
        "received_inputs_identical_to_trusted_loader"=>input_hashes,
        "requested_cells"=>cells,"execution_order"=>order,"cases"=>sort(cases;by=c->c["cell"]),
        "source_integrity_unchanged"=>true,"received_source_and_port_sha256_before"=>before,
        "received_source_and_port_sha256_after"=>after,"elapsed_seconds"=>time()-start,
        "protocol"=>Dict("B"=>B,"seed"=>seed,"degree"=>3,"tau"=>0,
            "metric"=>"full-coordinate identity Euclidean cone; all types; no simplex or residual floor",
            "projection_backend"=>opts["projection"],
            "J_rounding"=>"6 decimals, RoundNearestTiesAway before inclusive bootstrap tail",
            "bootstrap"=>"annual iid empirical resampling; both CF stages refit; fixed prices/medians",
            "probability_policy"=>"source coordinate clipping without renormalization",
            "paired_draws_across_models_within_each_window"=>true,
            "joint_cross_window_inference_or_multiplicity_correction"=>false,
            "RNG"=>"Julia MersenneTwister, not exact MATLAB streams",
            "data_or_test_selected_to_increase_separation"=>false),
        "inference_caution"=>"Tau0 replication diagnostics, not bootstrap-validity certification or a joint test across overlapping windows. Floating bounds concern computed recentered targets. No household records/identifiers exported.",
        "actual_MATLAB_or_CVX_executed"=>false,"raw_household_records_or_identifiers_exported"=>false)
    open(joinpath(output,"summary.json"),"w") do io
        JSON3.pretty(io,report)
        println(io)
    end
    println("Completed $(length(cases)) windows; summary saved in $output")
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    ww_main(ARGS)
end
