#!/usr/bin/env julia
# Focused Julia port of the received package, not execution of MATLAB/CVX.
# Existing engines, prior reports, and the received archive remain unchanged.
using KSReplication, MAT, JSON3, SHA, Statistics, LinearAlgebra, Random, ZipFile, JuMP

const WF_ARCHIVE_SHA = "b72d55c823926b680f041320e0553138a2270b26d66d2ee2b6074c18efeb64e1"
const WF_TARGET = "RUM_7LengthBudget_3Classes_3PolyDegree_1000BSreps_Series_Endogenous0taunot0_20231216_202.mat"
const WF_TARGET_SHA = "1c24ca7e5a1d814ce1170ef72e935a2ff78b92005f41a8383209b967fdc7b634"
const WF_CELL = 12
wf_sha(path) = bytes2hex(open(sha256, path))
wf_round_J(x) = round(x,RoundNearestTiesAway;digits=6)

function wf_options(args)
    opts = Dict("archive"=>get(ENV,"WILFRIED_ARCHIVE",joinpath(@__DIR__,"external/RUMCODEFINAL.zip")),
        "package"=>get(ENV,"WILFRIED_PACKAGE",joinpath(@__DIR__,"external/RUMCODEFINAL")),
        "data-archive"=>get(ENV,"FES_DATA_ARCHIVE",joinpath(@__DIR__,"external/231552-V1.zip")),
        "bootstrap"=>"1000", "seed"=>"13732",
        "output"=>joinpath(@__DIR__, "results/wilfried_replication_20261009_v2.json"))
    iseven(length(args)) || error("arguments must be --name value pairs")
    for i in 1:2:length(args)
        startswith(args[i], "--") || error("expected --option")
        key = args[i][3:end]
        haskey(opts, key) || error("unknown option $key")
        opts[key] = args[i+1]
    end
    return opts
end

# This reproduces the faulty Boolean semantics ONLY for a source-code audit.
# It is never exposed as a WARP model in the production type enumerator.
function wf_faulty_types(X)
    groups = KSReplication._patch_groups(X)
    selected = zeros(Int, length(groups))
    choices = Vector{Vector{Int}}()
    function descend(level)
        for row in groups[level]
            selected[level] = row
            if level >= 2
                F = KSReplication._selected_patch_graph(X, selected, level)
                all(vec(maximum(F .& transpose(F); dims=1))) && continue
            end
            if level == length(groups)
                push!(choices, copy(selected))
            else
                descend(level+1)
            end
        end
    end
    descend(1)
    A = zeros(Int8, size(X,1), length(choices))
    for (h, rows) in enumerate(choices), row in rows
        A[row,h] = 1
    end
    return A
end

function wf_input_identity(package, older_archive)
    reader = ZipFile.Reader(older_archive)
    hashes = Dict{String,String}()
    try
        entries = Dict(f.name=>f for f in reader.files)
        for name in vcat(["data$(y).csv" for y in 75:99], ["pall.csv"])
            actual = wf_sha(joinpath(package, "Input", name))
            previous = bytes2hex(sha256(read(entries["data/Input/"*name])))
            actual == previous || error("raw source differs from trusted loader: $name")
            hashes[name] = actual
        end
    finally
        close(reader)
    end
    return hashes
end

function wf_hashes(package, target)
    paths = [joinpath(package, name) for name in readdir(package) if endswith(name, ".m")]
    push!(paths, target)
    hashes = Dict(abspath(p)=>wf_sha(p) for p in paths)
    for name in ("Project.toml", "Manifest.toml", "run_wilfried_replication.jl",
        "src/KSReplication.jl", "src/household_data.jl", "src/types.jl", "src/fast_types.jl",
        "src/geometry.jl", "src/estimation.jl", "src/fast_estimation.jl",
        "src/inference.jl", "src/dual_projection.jl")
        path = joinpath(@__DIR__, name)
        hashes[abspath(path)] = wf_sha(path)
    end
    return hashes
end

function wf_certified_fit(projector, target)
    diagnostic = project!(projector, target)
    # Independent feasible primal/dual repair for this no-floor cone.
    # Every deterministic column has one entry per budget, hence common column sum.
    mass = max.(0.0, -dual.(projector.column_constraints))
    fitted = projector.A*mass
    upper = sum(abs2,target-fitted)
    lambda = value.(projector.lambda)
    column_sum = only(unique(vec(sum(projector.A;dims=1))))
    safety = 1e-12*(1+maximum(abs,lambda))
    repair = max(0.0,maximum(projector.A'*lambda))/column_sum+safety
    feasible_lambda = lambda .- repair
    maximum(projector.A'*feasible_lambda)<=1e-12 || error("repaired cone dual is infeasible")
    lower = max(0.0,2dot(target,feasible_lambda)-sum(abs2,feasible_lambda))
    lower<=upper+1e-10 || error("primal/dual bound inversion")
    return (;fitted,upper,lower=min(lower,upper),mass_sum=sum(mass),diagnostic)
end

function wf_simplex_threshold(sorted, cumulative, mass)
    mass==0 && return first(sorted)
    active = 1
    for k in eachindex(sorted)
        sorted[k]>(cumulative[k]-mass)/k && (active=k)
    end
    return (cumulative[active]-mass)/active
end

# Exact mathematical H representation of the full Cartesian type cone:
# q>=0, with the same total mass in every budget. This is NOT used for WARP.
function wf_product_fit(A, groups, target)
    sorted = [sort(target[g];rev=true) for g in groups]
    cumulative = cumsum.(sorted)
    if sum(first,sorted)<=0
        mass = 0.0
    else
        lo,hi = 0.0,max(1.0,maximum(sum(max.(target[g],0.0)) for g in groups))
        for iteration in 1:80
            mid = (lo+hi)/2
            gradient = sum(wf_simplex_threshold(sorted[j],cumulative[j],mid) for j in eachindex(groups))
            gradient>0 ? (lo=mid) : (hi=mid)
        end
        mass = (lo+hi)/2
    end
    marginal = zeros(length(target))
    for j in eachindex(groups)
        threshold = wf_simplex_threshold(sorted[j],cumulative[j],mass)
        marginal[groups[j]] = max.(target[groups[j]].-threshold,0.0)
    end
    # Product coupling constructs explicit feasible masses for the original A.
    # This repairs only floating-point feasibility of a fitted vector, not data.
    weights = zeros(size(A,2))
    if mass>0
        for h in axes(A,2)
            probability = 1.0
            for g in groups
                row = only(filter(i->A[i,h]!=0,g))
                probability *= marginal[row]/mass
            end
            weights[h] = mass*probability
        end
    end
    fitted = A*weights
    upper = sum(abs2,target-fitted)
    lambda = target-fitted
    J = length(groups)
    safety = 1e-12*(1+maximum(abs,lambda))
    repair = max(0.0,maximum(A'*lambda))/J+safety
    feasible_lambda = lambda .- repair
    maximum(A'*feasible_lambda)<=1e-12 || error("product-cone repaired dual is infeasible")
    lower = max(0.0,2dot(target,feasible_lambda)-sum(abs2,feasible_lambda))
    lower<=upper+1e-10 || error("product-cone primal/dual bound inversion")
    diagnostic = (primal_dual_gap=(upper-min(lower,upper))/2,
        equality_error=0.0,solver_status="ANALYTIC_PRODUCT_CONE")
    return (;fitted,upper,lower=min(lower,upper),mass_sum=sum(weights),diagnostic)
end

function wf_fit_report(A, pi, saved_pi, bootstrap, n, degree; product_groups=nothing)
    if product_groups===nothing
        projector = DualConeProjector(A; residual_floor=nothing)
        fit_function = target->wf_certified_fit(projector,target)
    else
        fit_function = target->wf_product_fit(A,product_groups,target)
    end
    observed = fit_function(pi)
    stored_target_fit = fit_function(saved_pi)
    scale = sum(n)/degree
    raw_J = scale*observed.upper
    J = wf_round_J(raw_J)
    J_lower = wf_round_J(scale*observed.lower)
    values = zeros(size(bootstrap,2))
    lower_values = similar(values)
    upper_values = similar(values)
    statuses = Dict{String,Int}()
    max_gap = observed.diagnostic.primal_dual_gap
    max_equality = observed.diagnostic.equality_error
    max_bound_width = scale*(observed.upper-observed.lower)
    for b in axes(bootstrap,2)
        # Actual live package uses tau=0, so there is no nonzero tightening origin.
        fit = fit_function(bootstrap[:,b]-pi+observed.fitted)
        values[b] = wf_round_J(scale*fit.upper)
        lower_values[b] = wf_round_J(scale*fit.lower)
        upper_values[b] = values[b]
        status = fit.diagnostic.solver_status
        statuses[status] = get(statuses,status,0)+1
        max_gap = max(max_gap,fit.diagnostic.primal_dual_gap)
        max_equality = max(max_equality,fit.diagnostic.equality_error)
        max_bound_width = max(max_bound_width,scale*(fit.upper-fit.lower))
    end
    B = length(values)
    tail = count(>=(J), values)
    p = J==0 ? 1.0 : tail/B
    tail_lower = count(>=(J),lower_values)
    tail_upper = count(>=(J_lower),upper_values)
    p_lower,p_upper = J==0 ? (1.0,1.0) :
        (tail_lower/B,J_lower==0 ? 1.0 : tail_upper/B)
    sorted = sort(values)
    return Dict("types"=>size(A,2), "J_unrounded"=>raw_J, "J_rounded_6"=>J,
        "observed_J_bounds"=>[scale*observed.lower,raw_J],
        "observed_J_rounding_bounds"=>[J_lower,J],
        "J_on_original_saved_pihat"=>scale*stored_target_fit.upper,
        "J_on_original_saved_pihat_bounds"=>scale.*[stored_target_fit.lower,stored_target_fit.upper],
        "J_on_original_saved_pihat_rounded_6"=>wf_round_J(scale*stored_target_fit.upper),
        "p_value_tau0_diagnostic"=>p, "tail_count"=>tail,
        "rounded_tail_numerical_bounds"=>[p_lower,p_upper],
        "rounded_tail_resolved_given_computed_recentered_targets"=>p_lower==p_upper,
        "bounds_do_not_certify_exact_MATLAB_draws_or_exact_recentered_projection"=>true,
        "p_value_mc_se"=>sqrt(p*(1-p)/B),
        "critical_95"=>sorted[ceil(Int,.95B)], "critical_99"=>sorted[ceil(Int,.99B)],
        "bootstrap_J_rounded_6"=>values, "projection_fitted"=>observed.fitted,
        "projection_mass_sum"=>observed.mass_sum,
        "maximum_primal_dual_gap"=>max_gap, "maximum_equality_error"=>max_equality,
        "maximum_repaired_J_bound_width"=>max_bound_width,
        "observed_projection_vector_error_bound"=>sqrt(observed.upper-observed.lower),
        "floating_dual_feasibility_safety"=>"1e-12*(1+maximum(abs,lambda)) per coordinate",
        "bootstrap_solver_status_counts"=>statuses, "failed_or_dropped_draws"=>0)
end

function wf_main(args)
    opts = wf_options(args)
    output = abspath(opts["output"])
    isfile(output) && error("refusing to overwrite an existing report: $output")
    archive, package = abspath(opts["archive"]), abspath(opts["package"])
    wf_sha(archive)==WF_ARCHIVE_SHA || error("received archive identity changed")
    target = joinpath(package, "Output", WF_TARGET)
    wf_sha(target)==WF_TARGET_SHA || error("registered saved output changed")
    input_hashes = wf_input_identity(package, opts["data-archive"])
    before = wf_hashes(package, target)
    B, seed = parse(Int, opts["bootstrap"]), parse(Int, opts["seed"])
    B >= 2 || error("at least two bootstrap draws are required")
    BLAS.set_num_threads(1)
    X, A_saved, P_saved, pi_saved, recorded = matopen(target) do f
        X = Int8.(read(f,"X")[WF_CELL])
        A = Int8.(read(f,"A")[WF_CELL])
        P = Float64.(read(f,"budgets")[WF_CELL])
        pi = vec(Float64.(read(f,"pi_hat")[WF_CELL]))
        recorded = Dict("J"=>vec(read(f,"Jstat"))[WF_CELL],
            "p"=>vec(read(f,"pval"))[WF_CELL], "tau"=>vec(read(f,"tau"))[WF_CELL],
            "B"=>Int(read(f,"bootstrap_reps")), "seed"=>Int(read(f,"seed")))
        (X, A, P, pi, recorded)
    end
    size(X)==(24,7) && size(A_saved)==(24,1680) || error("wrong target geometry")
    recorded["tau"]==0 && recorded["B"]==1000 || error("source settings changed")
    groups = KSReplication._patch_groups(X)
    A_faulty = wf_faulty_types(X)
    Set(Tuple.(eachcol(A_faulty)))==Set(Tuple.(eachcol(A_saved))) ||
        error("saved columns do not match the as-written checker")
    Ar = enumerate_types(X; restriction=:rum)
    Aw = enumerate_types(X; restriction=:warp)
    Au = enumerate_types(X; restriction=:unrestricted)
    Set(Tuple.(eachcol(A_saved)))==Set(Tuple.(eachcol(Au))) ||
        error("source checker is not the unrestricted Cartesian product on this target")
    @assert size(Ar,2)==333 && size(Aw,2)==347
    @assert has_two_cycle(Bool[0 0 1; 0 0 0; 1 0 0])
    @assert !all(vec(maximum(Bool[0 0 1; 0 0 0; 1 0 0];dims=1)))

    source = load_fes_source(opts["data-archive"])
    group = only(filter(g->g.id=="all", household_group_registry()))
    prep = prepare_household_group(source, group; years=1986:1992)
    prep.supported || error("actual source cohort is unsupported")
    P_new = reduce(vcat, permutedims(a.normalized_prices) for a in prep.annual)
    maximum(abs.(P_saved-P_new))<=1e-12 || error("regenerated prices differ from saved budgets")
    # Keep exact source budget coordinates and source X ordering for parity.
    annual = [KSAnnual(a.year,a.shares,a.expenditure,a.instrument,vec(P_saved[j,:]))
        for (j,a) in enumerate(prep.annual)]
    estimator = prepare_estimator(annual,X; degree=3,
        evaluation_log_expenditure=prep.evaluation_log_expenditure)
    point = estimate_prepared_probabilities(estimator; probability_policy=:source_clip)
    n = [y.n for y in estimator.years]
    @assert n==[1800,1817,1856,1859,1772,1712,1822]
    println("Verified actual inputs; target types source-checker=$(size(A_saved,2)), WARP=$(size(Aw,2)), RUM=$(size(Ar,2))")
    println("Regenerated-vs-saved pi maximum difference: ",maximum(abs.(point.pi-pi_saved)))
    flush(stdout)
    timed = @timed bootstrap_prepared_probabilities(estimator; repetitions=B, seed,
        parallel=true, probability_policy=:source_clip)
    boot = timed.value
    println("Completed $B paired CF refits in $(round(timed.time;digits=2)) seconds")
    flush(stdout)
    models = Dict{String,Any}()
    for (name,A) in (("source_checker_not_WARP",A_saved),("correct_WARP",Aw),("RUM",Ar))
        models[name] = wf_fit_report(A,point.pi,pi_saved,boot.pi,n,3;
            product_groups=name=="source_checker_not_WARP" ? groups : nothing)
        println(name, ": J=",models[name]["J_rounded_6"],
            ", p(tau0 diagnostic)=",models[name]["p_value_tau0_diagnostic"])
        flush(stdout)
    end
    report = Dict("status"=>"completed", "method"=>"Julia port, received-package focal window",
        "actual_MATLAB_or_CVX_executed"=>false, "julia_version"=>string(VERSION),
        "archive"=>archive, "archive_sha256"=>WF_ARCHIVE_SHA,
        "saved_target"=>abspath(target), "saved_target_sha256"=>WF_TARGET_SHA,
        "saved_target_cell"=>WF_CELL, "source_recorded_result_not_new_estimation"=>recorded,
        "received_inputs_identical_to_trusted_loader"=>input_hashes,
        "years"=>collect(1986:1992), "annual_n"=>n,
        "sign_matrix"=>[collect(r) for r in eachrow(X)], "prices_saved"=>P_saved,
        "maximum_regenerated_price_difference"=>maximum(abs.(P_new-P_saved)),
        "pi_hat_regenerated"=>point.pi, "pi_hat_original_saved"=>pi_saved,
        "maximum_regenerated_pihat_difference"=>maximum(abs.(point.pi-pi_saved)),
        "clipped_probability_block_sums"=>[sum(point.pi[g]) for g in groups],
        "estimation_diagnostics"=>point.diagnostics, "models"=>models,
        "protocol"=>Dict("B"=>B,"seed"=>seed,"degree"=>3,"tau"=>0,
            "metric"=>"full-coordinate identity Euclidean cone, no simplex or residual floor",
            "scale"=>sum(n)/3,"J_rounding"=>"6 decimals before bootstrap tail, RoundNearestTiesAway to match MATLAB",
            "bootstrap"=>"annual iid empirical resampling, both CF stages refit, fixed prices/medians",
            "RNG"=>"Julia MersenneTwister, NOT MATLAB mlfg6331_64 substreams",
            "source_exact_p_value_bitwise_reproduction"=>false,
            "paired_draws_across_models"=>true,"data_or_test_selected_to_increase_separation"=>false,
            "probability_policy"=>"source coordinate clipping, no renormalization",
            "first_stage_solver"=>"stable column-scaled QR, not raw normal equations",
            "original_source_skips_variance_stage_when_tau0"=>false,
            "Julia_variance_stage_omitted_because_tau_fixed0"=>true),
        "bootstrap_refit_seconds"=>timed.time,
        "bootstrap_seed_vector_sha256"=>bytes2hex(sha256(JSON3.write(boot.seeds))),
        "received_source_and_port_sha256_before"=>before,
        "inference_caution"=>"tau0 replication diagnostics, not certification of bootstrap validity; source-checker model is not WARP",
        "raw_household_records_or_identifiers_exported"=>false)
    after = wf_hashes(package,target)
    before==after || error("source or implementation changed during replication")
    wf_sha(archive)==WF_ARCHIVE_SHA || error("archive changed during replication")
    report["received_source_and_port_sha256_after"] = after
    report["source_integrity_unchanged"] = true
    mkpath(dirname(output))
    open(output,"w") do io
        JSON3.pretty(io,report)
        println(io)
    end
    println("Saved ",output)
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    wf_main(ARGS)
end
