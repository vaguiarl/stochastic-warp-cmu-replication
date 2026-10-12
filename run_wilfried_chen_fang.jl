#!/usr/bin/env julia
# NEW conditional comparison of corrected received-package cones. No old report,
# MATLAB code, production module, or received input is modified or executed.
include(joinpath(@__DIR__, "run_wilfried_windows.jl"))

const WCF_PRIOR_SHA = "12141fd6314b9680ec7ef1aa19545c785aa4c42a28bcfb028bde6cbca986e4b1"
const WCF_DATA_SHA = "2a7c2473baf5a9eb68a392e087ca9d3c0499777d72efc7798f9419b18c13fece"
const WCF_CELLS = collect(3:7)
const WCF_MULTIPLIERS = [0.5, 1.0, 2.0]
const WCF_B = 1000
const WCF_SEED = 13732
const WCF_WORKERS = 6

mutable struct WilfriedCFNNLSProjector
    engine::NNLSFullConeProjector
    diagnostics::Vector{Any}
    transformation::Symbol
end
function WilfriedCFNNLSProjector(A; transformation::Symbol=:identity,kwargs...)
    transformation in (:identity,:source_clip_composite) || error("unknown target transformation")
    return WilfriedCFNNLSProjector(NNLSFullConeProjector(A; kwargs...),Any[],transformation)
end

function KSReplication.project_information!(p::WilfriedCFNNLSProjector,
        z::AbstractVector; covariance=I)
    d = size(p.engine.A, 1)
    length(z) == d || throw(DimensionMismatch("full cone coordinates"))
    all(isfinite, z) || throw(ArgumentError("nonfinite full cone target"))
    S = covariance isa UniformScaling ? Matrix{Float64}(covariance, d, d) :
        Matrix{Float64}(covariance)
    size(S) == (d, d) && S == Matrix{Float64}(I, d, d) ||
        throw(ArgumentError("received-profile adapter supports EXACT identity metric only"))
    # Primary g(a)=Delta(clamp(a)) composes the KNOWN source clamp with the
    # distance functional, while bootstrapping the raw affine estimator. The
    # secondary identity callback instead passes every signed target untouched.
    # Neither mode imposes normalization, simplex, or residual floor on the cone.
    target = p.transformation==:source_clip_composite ? clamp.(z,0,1) : z
    f = nnls_fit!(p.engine, target)
    all(isfinite, (f.lower, f.upper, f.mass_sum)) && 0 <= f.lower <= f.upper ||
        error("invalid full-column objective bounds")
    f.diagnostic.full_type_count == size(p.engine.A, 2) || error("incomplete cone fit")
    f.diagnostic.repaired_full_dual_violation <= 1e-12 || error("full dual not feasible")
    width = f.upper - f.lower
    width <= p.engine.certificate_tolerance * max(1.0, f.upper) ||
        error("full cone fit failed objective-width gate")
    push!(p.diagnostics, f.diagnostic)
    return (; squared_distance=f.upper, objective_lower_bound=f.lower,
        objective_upper_bound=f.upper, objective_error_bound=width,
        fitted_contrast=f.fitted, mass_sum=f.mass_sum,
        equality_error=f.diagnostic.equality_error,
        primal_dual_gap=f.diagnostic.primal_dual_gap,
        solver_status=f.diagnostic.solver_status, certified=true)
end

function wcf_isolation(point_pair, worker_pairs)
    projectors = [p for pair in vcat([point_pair], worker_pairs) for p in pair]
    length(unique(objectid(p) for p in projectors)) == length(projectors) ||
        error("shared mutable adapter")
    length(unique(objectid(p.engine) for p in projectors)) == length(projectors) ||
        error("shared mutable NNLS state")
    length(unique(objectid(p.diagnostics) for p in projectors)) == length(projectors) ||
        error("shared mutable diagnostic buffer")
    return Dict("point_states"=>2, "worker_states"=>2length(worker_pairs),
        "unique_adapter_states"=>length(projectors),
        "unique_NNLS_states"=>length(projectors), "shared_mutable_state"=>false)
end

function wcf_projection_audit(point_pair, worker_pairs)
    projectors = [p for pair in vcat([point_pair],worker_pairs) for p in pair]
    ds = [d for p in projectors for d in p.diagnostics]
    isempty(ds) && error("no recorded projections")
    return Dict("fits"=>length(ds), "failed_or_replaced_draws"=>0,
        "maximum_equality_error"=>maximum(d.equality_error for d in ds),
        "maximum_half_primal_dual_gap"=>maximum(d.primal_dual_gap for d in ds),
        "maximum_repaired_full_dual_violation"=>maximum(d.repaired_full_dual_violation for d in ds),
        "full_pricing_passes"=>sum(d.full_pricing_passes for d in ds),
        "priced_columns_including_repeated_passes"=>sum(d.priced_columns for d in ds),
        "fallback_count"=>count(d->d.fallback_used,ds),
        "fallback_reasons"=>unique([d.fallback_reason for d in ds if d.fallback_used]),
        "maximum_NNLS_iterations"=>maximum(d.iterations for d in ds),
        "no_residual_floor_or_simplex"=>true,
        "certificate_interpretation"=>"Float64 full-column feasible primal/dual numerical checks, not rigorous outward-rounded intervals")
end

function wcf_implementation_hashes()
    names = ["run_wilfried_chen_fang.jl", "test/test_wilfried_chen_fang.jl",
        "run_wilfried_windows.jl", "run_wilfried_replication.jl",
        "wilfried_nnls_projection.jl", "wilfried_working_set_projection.jl",
        "src/chen_fang_inference.jl", "src/weighted_gain_inference.jl",
        "src/information_inference.jl", "src/KSReplication.jl",
        "src/fast_estimation.jl", "src/estimation.jl", "src/household_data.jl",
        "src/fast_types.jl", "src/types.jl", "Project.toml", "Manifest.toml"]
    return Dict(n=>wf_sha(joinpath(@__DIR__,n)) for n in names)
end

function wcf_write_new(path, value)
    ispath(path) && error("refusing to overwrite $path")
    mkpath(dirname(path))
    temporary = path * ".tmp"
    ispath(temporary) && error("refusing to overwrite interrupted checkpoint")
    open(temporary, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
    chmod(temporary, 0o600)
    mv(temporary, path; force=false)
end

function wcf_options(args)
    options = Dict("archive"=>get(ENV,"WILFRIED_ARCHIVE",joinpath(@__DIR__,"external/RUMCODEFINAL.zip")),
        "package"=>get(ENV,"WILFRIED_PACKAGE",joinpath(@__DIR__,"external/RUMCODEFINAL")),
        "data-archive"=>get(ENV,"FES_DATA_ARCHIVE",joinpath(@__DIR__,"external/231552-V1.zip")),
        "prior-directory"=>joinpath(@__DIR__,"results/wilfried_windows_20261009_v2"),
        "validation-report"=>joinpath(@__DIR__,"results/wilfried_chen_fang_adapter_validation_20261010_v1.json"),
        "output-directory"=>joinpath(@__DIR__,"results/wilfried_chen_fang_20261010_v1"))
    iseven(length(args)) || error("expected --name value pairs")
    for i in 1:2:length(args)
        startswith(args[i],"--") || error("expected --option")
        key = args[i][3:end]
        haskey(options,key) || error("unknown option $key")
        options[key] = args[i+1]
    end
    return options
end

function wcf_prior_gate(path)
    wf_sha(path) == WCF_PRIOR_SHA || error("frozen all-window summary changed")
    prior = JSON3.read(read(path,String))
    prior["status"] == "completed" && prior["source_integrity_unchanged"] ||
        error("prior replication incomplete or source changed")
    Int.(prior["requested_cells"]) == collect(1:19) || error("prior family changed")
    Int(prior["protocol"]["B"]) == WCF_B && Int(prior["protocol"]["seed"]) == WCF_SEED ||
        error("prior bootstrap profile changed")
    Int(prior["protocol"]["degree"]) == 3 && prior["protocol"]["tau"] == 0 ||
        error("prior degree/tau changed")
    for (path, digest) in pairs(prior["received_source_and_port_sha256_before"])
        digest == prior["received_source_and_port_sha256_after"][path] || error("prior hash inconsistency")
        wf_sha(String(path)) == String(digest) || error("prior lineage no longer identical: $path")
    end
    return prior
end

function wcf_boundary_diagnostics(raw_point, raw_bootstrap, prepared)
    groups = [y.patch_rows for y in prepared.years]
    d, B = size(raw_bootstrap)
    per_coordinate = Dict{String,Any}[]
    for i in 1:d
        values = view(raw_bootstrap,i,:)
        push!(per_coordinate, Dict("coordinate"=>i, "affine_point"=>raw_point[i],
            "distance_to_zero"=>raw_point[i], "distance_to_one"=>1-raw_point[i],
            "raw_point_exact_boundary"=>raw_point[i] in (0.0,1.0),
            "raw_point_within_1e12_boundary"=>min(abs(raw_point[i]),abs(raw_point[i]-1))<=1e-12,
            "bootstrap_below_zero_fraction"=>count(<(0),values)/B,
            "bootstrap_above_one_fraction"=>count(>(1),values)/B,
            "bootstrap_clipping_regime_changed_fraction"=>count(x->
                (x<0 ? -1 : x>1 ? 1 : 0) != (raw_point[i]<0 ? -1 : raw_point[i]>1 ? 1 : 0),values)/B,
            "minimum_affine_bootstrap"=>minimum(values),"maximum_affine_bootstrap"=>maximum(values)))
    end
    empty_counts = [count(==(0),[count(==(p),y.patch_positions) for p in eachindex(y.patch_rows)])
        for y in prepared.years]
    return Dict("by_coordinate"=>per_coordinate,
        "raw_point_exact_boundary_coordinates"=>count(x->x in (0.0,1.0),raw_point),
        "raw_point_within_1e12_boundary_coordinates"=>count(x->min(abs(x),abs(x-1))<=1e-12,raw_point),
        "raw_point_strictly_below_zero_coordinates"=>count(<(0),raw_point),
        "raw_point_strictly_above_one_coordinates"=>count(>(1),raw_point),
        "empirically_empty_patch_counts_by_year"=>empty_counts,
        "empty_patches_are_not_known_population_zeros"=>true,
        "affine_budget_sums"=>[sum(raw_point[g]) for g in groups],
        "clipped_budget_sums"=>[sum(clamp.(raw_point[g],0,1)) for g in groups],
        "clipping_boundary_bootstrap_regularity_verified"=>false,
        "interpretation"=>"Strictly interior or strictly clipped pseudo-true coordinates can be locally smooth. Genuine population-boundary clipping can invalidate the ordinary empirical bootstrap. Finite-sample margins/crossings do not resolve that distinction.")
end

function wcf_multiplicity(cases; family=19)
    primary = Dict{Int,Float64}()
    for c in cases
        if get(c,"status","")=="computed_conditional"
            row = only(filter(r->isapprox(r["epsilon_multiplier"],1.0;atol=1e-12),
                c["comparison"]["epsilon_results"]))
            primary[c["cell"]] = row["critical_value_positive_resolved"] ? row["p_value_mc"] : 1.0
        end
    end
    # Untested/failed/unresolved windows remain p=1, not silently deleted from M.
    pvalues = [get(primary,i,1.0) for i in 1:family]
    order = sortperm(pvalues)
    holm = ones(family)
    previous = 0.0
    for (rank,i) in enumerate(order)
        previous = max(previous,min(1.0,(family-rank+1)*pvalues[i]))
        holm[i] = previous
    end
    return Dict("family_size"=>family,"tested_cells"=>sort(collect(keys(primary))),
        "untested_failed_or_unresolved_as_one"=>true,
        "primary_p_for_adjustment"=>pvalues,
        "primary_Bonferroni_adjusted_p"=>min.(1.0,family.*pvalues),
        "primary_Holm_adjusted_p"=>holm,
        "epsilon_search_sensitivity_family_size"=>3family,
        "validity"=>"All adjustments remain conditional on marginal bootstrap/critical-value validity; overlapping windows need not be independent. No selection among epsilon grids is authorized.")
end

function wcf_main(args)
    opts = wcf_options(args)
    Threads.nthreads() >= WCF_WORKERS || error("six Julia threads are required")
    output = abspath(opts["output-directory"])
    ispath(output) && error("refusing existing output directory")
    archive,package = abspath(opts["archive"]),abspath(opts["package"])
    target = joinpath(package,"Output",WF_TARGET)
    wf_sha(archive)==WF_ARCHIVE_SHA && wf_sha(target)==WF_TARGET_SHA || error("received source identity changed")
    wf_sha(opts["data-archive"])==WCF_DATA_SHA || error("raw data ZIP identity changed")
    prior_path = joinpath(opts["prior-directory"],"summary.json")
    prior = wcf_prior_gate(prior_path)
    before = wcf_implementation_hashes()
    validation_hash = wf_sha(opts["validation-report"])
    validation = JSON3.read(read(opts["validation-report"],String))
    validation["status"]=="passed" || error("adapter validation has not passed")
    for (name,digest) in pairs(validation["implementation_sha256"])
        get(before,String(name),nothing)==String(digest) || error("validation code no longer current: $name")
    end
    input_hashes = wf_input_identity(package,opts["data-archive"])
    saved = matopen(target) do f
        first_year=Int(read(f,"start_year")); first_year<100 && (first_year+=1900)
        (;first_year,X=read(f,"X"),P=read(f,"budgets"),pi=read(f,"pi_hat"),tau=vec(read(f,"tau")))
    end
    source = load_fes_source(opts["data-archive"])
    group = only(filter(g->g.id=="all",household_group_registry()))
    BLAS.set_num_threads(1)
    mkpath(output); chmod(output,0o700)
    cases = Dict{String,Any}[]
    start = time()
    for cell in WCF_CELLS
        case_start=time()
        prior_checkpoint=joinpath(opts["prior-directory"],"cell_$(lpad(cell,2,'0')).json")
        old=JSON3.read(read(prior_checkpoint,String))
        old==only(filter(c->c["cell"]==cell,prior["cases"])) ||
            error("checkpoint does not equal its registered frozen summary case")
        case=Dict{String,Any}("cell"=>cell,"status"=>"started",
            "prior_checkpoint_sha256"=>wf_sha(prior_checkpoint))
        try
            old["status"]=="completed" && old["cell"]==cell || error("prior cell unavailable")
            X,P=Int8.(saved.X[cell]),Float64.(saved.P[cell])
            years=(saved.first_year+cell-1):(saved.first_year+cell+size(X,2)-2)
            Int.(old["years"])==collect(years) && saved.tau[cell]==0 || error("window profile mismatch")
            Xold=reduce(vcat,permutedims(Int8.(row)) for row in old["sign_matrix"])
            Xold==X || error("saved patch geometry changed")
            Ar=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
            Aw=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
            size(Ar,2)==old["models"]["RUM"]["types"] &&
                size(Aw,2)==old["models"]["correct_WARP"]["types"] || error("type counts changed")
            issubset(ww_columns(Ar),ww_columns(Aw)) || error("model nesting failed")
            prep=prepare_household_group(source,group;years)
            prep.supported || error("sample unsupported")
            Pnew=reduce(vcat,permutedims(a.normalized_prices) for a in prep.annual)
            price_error=maximum(abs.(P-Pnew)); price_error<=1e-12 || error("price geometry changed")
            annual=[KSAnnual(a.year,a.shares,a.expenditure,a.instrument,vec(P[j,:])) for (j,a) in enumerate(prep.annual)]
            prepared=prepare_estimator(annual,X;degree=3,evaluation_log_expenditure=prep.evaluation_log_expenditure)
            affine=estimate_prepared_probabilities(prepared;probability_policy=:unclipped_affine)
            clipped=estimate_prepared_probabilities(prepared;probability_policy=:source_clip)
            clamp.(affine.pi,0,1)==clipped.pi || error("point policy mismatch")
            point_error=maximum(abs.(clipped.pi-Float64.(old["pi_hat_regenerated"])))
            point_error<=1e-12 || error("frozen regenerated point changed")
            n=[y.n for y in prepared.years]
            n==Int.(old["annual_n"]) || error("annual sample sizes changed")
            rate=sqrt(sum(n)/3)
            println("Chen–Fang cell $cell $(first(years))-$(last(years)); RUM=$(size(Ar,2)), WARP=$(size(Aw,2)), rate=$rate")
            flush(stdout)
            refit=@timed bootstrap_prepared_probabilities(prepared;repetitions=WCF_B,
                seed=WCF_SEED,parallel=true,probability_policy=:unclipped_affine)
            boot=refit.value
            seed_hash=bytes2hex(sha256(JSON3.write(boot.seeds)))
            seed_hash==old["bootstrap_seed_vector_sha256"] || error("original paired bootstrap seeds changed")
            boot_clip=clamp.(boot.pi,0,1)
            prefix_counts=[KSReplication.resample_counts(MersenneTwister(boot.seeds[b]),prepared) for b in 1:3]
            direct=bootstrap_prepared_probabilities(prepared;repetitions=3,seed=WCF_SEED,
                counts=prefix_counts,parallel=true,probability_policy=:source_clip)
            direct.pi==boot_clip[:,1:3] || error("affine-clamp differs from direct clipped bootstrap")
            point_pair=(WilfriedCFNNLSProjector(Ar;transformation=:source_clip_composite),
                WilfriedCFNNLSProjector(Aw;transformation=:source_clip_composite))
            pairs=[(WilfriedCFNNLSProjector(Ar;transformation=:source_clip_composite),
                WilfriedCFNNLSProjector(Aw;transformation=:source_clip_composite)) for _ in 1:WCF_WORKERS]
            isolation=wcf_isolation(point_pair,pairs)
            pointfits=[KSReplication.project_information!(p,affine.pi;covariance=I) for p in point_pair]
            for (k,name) in enumerate(("RUM","correct_WARP"))
                oldq=Float64.(old["models"][name]["observed_J_bounds"])/(sum(n)/3)
                f=pointfits[k]
                max(f.objective_lower_bound,oldq[1])<=min(f.objective_upper_bound,oldq[2])+1e-10 ||
                    error("same-point full-cone numerical intervals disagree")
            end
            timed=@timed chen_fang_fixed_metric_test(affine.pi,boot.pi;rate,
                rum_projector=point_pair[1],cmu_projector=point_pair[2],metric_covariance=I,
                epsilons=WCF_MULTIPLIERS./sqrt(rate),workers=WCF_WORKERS,
                projector_factory=w->pairs[w],include_draws=true)
            result=timed.value
            result["functional"]="g(a)=squared_distance_RUM(clamp(a,0,1))-squared_distance_WARP(clamp(a,0,1)); full identity cones"
            result["method"]="chen_fang_composite_source_clipping_with_raw_affine_joint_bootstrap"
            result["protocol"]["null"]="equal population source-clipped minimum cone distances: g(a0)=0"
            result["protocol"]["raw_affine_bootstrap_before_known_clipping_functional"]=true
            result["protocol"]["source_clipping_inside_every_functional_evaluation"]=true
            result["protocol"]["bootstrap_of_clipped_estimator_required"]=false
            result["protocol"]["composite_second_order_chain_rule_assumptions_verified_empirically"]=false
            for row in result["epsilon_results"]
                conditionalp=row["critical_value_positive_resolved"] ? row["p_value_mc"] : 1.0
                row["conditional_Bonferroni_p_19_windows"]=min(1.0,19conditionalp)
                row["conditional_Bonferroni_p_57_window_grid_search"]=min(1.0,57conditionalp)
                row["marginal_p_assumes_unverified_raw_affine_joint_bootstrap_regularity"]=true
            end
            secondary_point=(WilfriedCFNNLSProjector(Ar),WilfriedCFNNLSProjector(Aw))
            secondary_pairs=[(WilfriedCFNNLSProjector(Ar),WilfriedCFNNLSProjector(Aw)) for _ in 1:WCF_WORKERS]
            secondary_isolation=wcf_isolation(secondary_point,secondary_pairs)
            secondary_timed=@timed chen_fang_fixed_metric_test(clipped.pi,boot_clip;rate,
                rum_projector=secondary_point[1],cmu_projector=secondary_point[2],metric_covariance=I,
                epsilons=WCF_MULTIPLIERS./sqrt(rate),workers=WCF_WORKERS,
                projector_factory=w->secondary_pairs[w],include_draws=true)
            secondary=secondary_timed.value
            secondary["method"]="secondary_source_clipped_estimator_increments_diagnostic_only"
            secondary["protocol"]["primary_method"]=false
            secondary["protocol"]["genuine_boundary_clipped_estimator_bootstrap_validity_unverified"]=true
            secondary["not_used_for_primary_conclusion_or_favorable_method_selection"]=true
            merge!(case,Dict("status"=>"computed_conditional","years"=>collect(years),
                "annual_n"=>n,"rate"=>rate,"J_scale"=>sum(n)/3,
                "full_coordinate_count"=>size(X,1),"correct_type_counts"=>Dict("RUM"=>size(Ar,2),"WARP"=>size(Aw,2)),
                "type_column_sha256"=>Dict("RUM"=>bytes2hex(sha256(JSON3.write(Ar))),"WARP"=>bytes2hex(sha256(JSON3.write(Aw)))),
                "maximum_prior_point_difference"=>point_error,"maximum_saved_price_difference"=>price_error,
                "source_clipped_point"=>clipped.pi,"unclipped_affine_point"=>affine.pi,
                "source_clipped_bootstrap"=>boot_clip,"unclipped_affine_bootstrap"=>boot.pi,
                "bootstrap_seed_vector_sha256"=>seed_hash,
                "affine_bootstrap_sha256"=>bytes2hex(sha256(JSON3.write(boot.pi))),
                "clipped_bootstrap_sha256"=>bytes2hex(sha256(JSON3.write(boot_clip))),
                "direct_source_clipped_prefix_verified_draws"=>3,
                "direct_source_clipped_prefix_uses_explicit_counts_from_full_original_seed_vector"=>true,
                "source_clipping_diagnostics"=>wcf_boundary_diagnostics(affine.pi,boot.pi,prepared),
                "point_estimation_diagnostics"=>affine.diagnostics,
                "comparison"=>result,"state_isolation"=>isolation,
                "secondary_clipped_increment_comparison"=>secondary,
                "secondary_state_isolation"=>secondary_isolation,
                "full_column_projection_audit"=>wcf_projection_audit(point_pair,pairs),
                "secondary_full_column_projection_audit"=>wcf_projection_audit(secondary_point,secondary_pairs),
                "bootstrap_refit_seconds"=>refit.time,"comparison_seconds"=>timed.time,
                "secondary_comparison_seconds"=>secondary_timed.time))
            println("  gap statistic=",result["statistic"],"; conditional grid tails=",
                [r["p_value_mc"] for r in result["epsilon_results"]]); flush(stdout)
        catch err
            case["status"]="failed_no_replacement"
            case["error"]=sprint(showerror,err,catch_backtrace())
            println("  FAILED cell $cell: ",sprint(showerror,err)); flush(stdout)
        end
        case["elapsed_seconds"]=time()-case_start
        push!(cases,case)
        wcf_write_new(joinpath(output,"cell_$(lpad(cell,2,'0')).json"),case)
    end
    after=wcf_implementation_hashes()
    unchanged=before==after && wf_sha(archive)==WF_ARCHIVE_SHA &&
        wf_sha(opts["data-archive"])==WCF_DATA_SHA && wf_sha(prior_path)==WCF_PRIOR_SHA &&
        wf_sha(opts["validation-report"])==validation_hash
    wcf_prior_gate(prior_path)
    report=Dict("status"=>all(c->c["status"]=="computed_conditional",cases) && unchanged ?
        "completed_conditional_not_statistical_validity_certification" : "completed_with_failures",
        "cases"=>cases,"family_adjustments"=>wcf_multiplicity(cases),
        "requested_cells"=>WCF_CELLS,"uncomputed_family_cells"=>setdiff(collect(1:19),WCF_CELLS),
        "prior_summary_sha256"=>WCF_PRIOR_SHA,"prior_directory"=>abspath(opts["prior-directory"]),
        "received_archive_sha256"=>WF_ARCHIVE_SHA,"raw_data_archive_sha256"=>WCF_DATA_SHA,
        "saved_target_sha256"=>WF_TARGET_SHA,"received_input_hashes"=>input_hashes,
        "adapter_validation_sha256_before"=>validation_hash,
        "adapter_validation_sha256_after"=>wf_sha(opts["validation-report"]),
        "implementation_sha256_before"=>before,"implementation_sha256_after"=>after,
        "source_integrity_unchanged"=>unchanged,"elapsed_seconds"=>time()-start,
        "protocol"=>Dict("B"=>WCF_B,"seed"=>WCF_SEED,"degree"=>3,"tau"=>0,
            "workers"=>WCF_WORKERS,"rate"=>"sqrt(sum annual N / degree3)",
            "metric"=>"fixed identity full-coordinate nested type cones, NOT normalized probability polytopes",
            "probability_policy"=>"PRIMARY composite g(raw affine)=Delta(source_clip(raw affine)); observed source profile unchanged",
            "primary_numerical_perturbation"=>"raw affine increments, coordinate clamp INSIDE each known functional evaluation; no renormalization",
            "secondary_numerical_perturbation"=>"already-clipped estimator increments; signed perturbed targets NOT clamped; diagnostic only",
            "bootstrap"=>"annual iid empirical resampling; both CF stages re-fit; fixed source prices and evaluation medians",
            "RNG"=>"Julia MersenneTwister exact same saved Julia seeds; not exact MATLAB streams",
            "null"=>"equal population minimum distances to nested cones under fixed identity metric",
            "primary_epsilon_multiplier"=>1.0,"epsilon_multipliers"=>WCF_MULTIPLIERS,
            "selected_existing_positive_gain_cells"=>true,"selection_family_size"=>19,
            "epsilon_search_sensitivity_family_size"=>57,
            "rounded_source_J_or_membership_null_bootstrap_used_as_comparison_statistic"=>false,
            "all_coordinates_and_types_retained"=>true,"no_floor_or_simplex"=>true,
            "Monte_Carlo_tail"=>"inclusive observed-lower versus derivative-upper; plus one",
            "failed_draw_replacement_or_reselection"=>false),
        "unverified_inference_assumptions"=>[
            "Joint annual iid bootstrap consistency of the RAW AFFINE two-stage fixed-cubic estimator",
            "Negligible fixed-cubic approximation bias for a genuine population demand-probability target",
            "Composite second-order directional expansion of Delta composed with the known coordinate clamp under equal-distance null",
            "For SECONDARY diagnostic only: genuine population-boundary clipping can invalidate the clipped-estimator empirical bootstrap",
            "Positive continuous critical quantile in the second-order directional limit",
            "Survey dependence and expenditure/instrument identification; fixed price/median geometry uncertainty not bootstrapped"],
        "scope"=>"Profile-matched conditional Chen–Fang computation, not a literal published p-value replication or unconditional size guarantee. Nontransitive population shares are not identified. Prior reports and manuscript preserved.",
        "raw_household_records_or_identifiers_exported"=>false,"actual_MATLAB_or_CVX_executed"=>false)
    wcf_write_new(joinpath(output,"summary.json"),report)
    unchanged || error("source changed: inferential output invalid")
    println("Saved conditional comparison in $output")
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    wcf_main(ARGS)
end
