# New finite-step tuning diagnostic. Frozen empirical code/results are read-only.
# This is a two-window Gaussian geometry experiment, NOT a calibration theorem
# for the UK control-function estimator or a nineteen-window joint test.
using Random, LinearAlgebra, Statistics, Distributions, JSON3, SHA
include(joinpath(@__DIR__, "cmu_superiority_floor_inference.jl"))
include(joinpath(@__DIR__, "cmu_c_geometry_design.jl"))

const CCG_GRID = [.25, .5, .75, 1., 1.5, 2., 3., 4.]
const CCG_ALPHAS = [.05, .10]
const CCG_RATE = sqrt(46391 / 3)
const CCG_SEED = 202610123
const CCG_M = 250
const CCG_B = 199
const CCG_CELLS = [7, 12]
const CCG_TOL = 1e-8

ccg_sha(path) = bytes2hex(sha256(read(path)))
function ccg_interval(k, m)
    [k == 0 ? 0. : quantile(Beta(k, m-k+1), .025),
     k == m ? 1. : quantile(Beta(k+1, m-k), .975)]
end

"""The original conservative interval decision, generalized only in c-grid.

Early stopping is permitted ONLY when accumulated upper-bound exceedances
already force nonrejection at BOTH levels, whatever the remaining B draws.
The original B denominator is retained; this is decision pruning, not missing
or replaced bootstrap draws. A possible rejection requires ALL B evaluations
and the original positive-critical/numerical-resolution gate.
"""
function ccg_decisions(fit, point, G, rate; grid=CCG_GRID,
        alphas=CCG_ALPHAS, stop_nonrejection=true, store_trace=false)
    base = csfi_check_fit(fit(point))
    B = size(G, 2); length(point) == size(G, 1) || error("noise dimensions")
    lo = ceip_float_lower(CSFI_Q(rate)^2 * base.lower)
    hi = ceip_float_upper(CSFI_Q(rate)^2 * base.upper)
    outcomes = Dict{String,Any}[]
    max_tails = [floor(Int, a*(B+1) + 1e-10)-1 for a in alphas]
    all(0 .<= max_tails .< B) || error("bootstrap resolution insufficient")
    for c in grid
        h = c/sqrt(rate); h2 = CSFI_Q(h)^2
        central = Float64[]; lower = Float64[]; upper = Float64[]
        count_upper = 0; fallbacks = 0
        terminal = "complete"
        if stop_nonrejection && lo <= 0
            terminal = "observed_lower_nonpositive"
        else
            for b in 1:B
                target = point .+ h .* view(G, :, b)
                p = csfi_check_fit(fit(target))
                dl = (p.lower-base.upper)/h2
                du = (p.upper-base.lower)/h2
                dc = ((p.lower+p.upper)-(base.lower+base.upper))/(2h2)
                push!(central, Float64(dc))
                push!(lower, ceip_float_lower(dl))
                push!(upper, ceip_float_upper(du))
                count_upper += upper[end] >= lo
                fallbacks += get(p.diagnostic,"full_Clarabel_fallback_used",false)
                if stop_nonrejection && count_upper > maximum(max_tails)
                    terminal = "upper_tail_count_forces_nonrejection_at_both_levels"
                    break
                end
            end
        end
        evaluated = length(upper)
        decisions = Dict{String,Any}[]
        for (a, limit) in zip(alphas, max_tails)
            complete = evaluated == B
            order = ceil(Int,(1-a)*(B+1))
            critical = complete ? sort(upper)[order] : NaN
            critical_mid = complete ? sort(central)[order] : NaN
            width = complete ? maximum(upper-lower) : NaN
            resolved = complete && isfinite(critical_mid) && critical_mid > max(0.,width)
            reject = resolved && lo > critical && (count_upper+1)/(B+1) <= a
            certified_nonreject = complete || lo<=0 || count_upper>limit
            certified_nonreject || error("partial calculation cannot certify decision")
            push!(decisions,Dict("alpha"=>a,"reject"=>reject,
                "critical_positive_resolved"=>complete ? resolved : nothing,
                "critical_upper"=>complete ? critical : nothing,
                "central_critical"=>complete ? critical_mid : nothing,
                "maximum_derivative_width"=>complete ? width : nothing,
                "tail_MC_plus_one"=>complete ? (count_upper+1)/(B+1) : nothing,
                "tail_lower_bound_MC_plus_one"=>(count_upper+1)/(B+1),
                "tail_upper_bound_MC_plus_one"=>(count_upper+B-evaluated+1)/(B+1)))
        end
        row = Dict{String,Any}("c"=>c,"h"=>h,"decisions"=>decisions,
            "B_fixed"=>B,"inner_targets_evaluated"=>evaluated,
            "inner_targets_decision_pruned"=>B-evaluated,
            "termination"=>terminal,"upper_tail_exceedances_evaluated"=>count_upper,
            "fallbacks"=>fallbacks)
        if store_trace
            row["derivative_lower"]=lower; row["derivative_upper"]=upper
            row["derivative_midpoint"]=central
        end
        push!(outcomes,row)
    end
    Dict("observed_statistic_lower"=>lo,"observed_statistic_upper"=>hi,
        "base_exact_lower"=>string(base.lower),"base_exact_upper"=>string(base.upper),
        "results"=>outcomes)
end

function ccg_protocol(; repetitions=CCG_M, bootstrap=CCG_B, seed=CCG_SEED, probe=false)
    Dict("version"=>"actual-UK-two-geometry-c-diagnostic-v1",
        "c_grid"=>CCG_GRID,"existing_empirical_primary_c_unchanged"=>1.,
        "alpha_levels"=>CCG_ALPHAS,"outer_repetitions"=>repetitions,
        "inner_bootstrap_B"=>bootstrap,"seed"=>seed,"rate"=>CCG_RATE,
        "cells"=>CCG_CELLS,"probe_only"=>probe,
        "cell_selection"=>"Two smallest seven-budget windows with strict complete RUM versus WARP type-count difference, selected by geometry not fitted gain",
        "scenarios"=>["complete_RUM_centroid_null","exposed_face_null",
            "geometry_WARP_alternative_one_SE","geometry_WARP_alternative_two_SE"],
        "noise"=>"Centered local frozen raw-bootstrap covariance; full-coordinate QR factor, no inverse, rank truncation or nuisance refitting",
        "outer_inner_independence"=>"Fresh independent Gaussian outer and inner innovations for every replication; common random numbers across c and scenarios within replication",
        "simulation_outcomes_independent_of_observed_UK_fit_gains"=>true,
        "mean_design"=>"Geometry-only RUM centroid and exposed face; alternatives move towards the first canonical WARP column absent from RUM, at one/two normal-noise SE, capped to preserve probability simplex",
        "functional"=>"Psi=d_complete_RUM^2-d_product_simplex^2, unchanged positive-empty-cap equal-distance null",
        "empty_demand_scope"=>"Same certified inner-family null-equivalent statistic as the frozen positive-cap actual-empty-CMU run; no empty-choice observations manufactured",
        "calibration"=>"Signed numerical second derivative centered at each synthetic estimate, conservative exact objective endpoints and positive-critical gate",
        "no_real_data_c_selection"=>true,"no_candidate_promoted_by_empirical_p"=>true,
        "selection_or_optimality_claimed"=>false,
        "pruning"=>"All B latent Gaussian draws generated; objective solves may stop only after both nonrejections are mathematically forced, keeping B fixed",
        "evaluated_derivative_traces_saved"=>true,
        "limitations"=>["Per-window Gaussian stress experiment, not nineteen-window joint inference",
            "Conditional on estimated nuisance covariance, not household/control-function resimulation",
            "No uniform size guarantee or proof of optimal c",
            "Monte Carlo intervals describe each marginal rate; no multiple-comparison c selection"])
end

function ccg_summary(records, designs; repetitions)
    rows=Dict{String,Any}[]
    for d in designs, (sidx,s) in enumerate(d.scenarios), (k,c) in enumerate(CCG_GRID), (ai,a) in enumerate(CCG_ALPHAS)
        selected=[r for r in records if r["cell"]==d.cell && r["scenario_index"]==sidx]
        length(selected)==repetitions || error("incomplete outer campaign")
        decisions=[r["result"]["results"][k]["decisions"][ai] for r in selected]
        k_reject=count(x->x["reject"],decisions)
        resolved=count(x->x["critical_positive_resolved"]===true,decisions)
        evaluated=sum(r["result"]["results"][k]["inner_targets_evaluated"] for r in selected)
        push!(rows,Dict("cell"=>d.cell,"years"=>d.years,"scenario"=>s.id,
            "population_null"=>s.null,"c"=>c,"alpha"=>a,"repetitions"=>repetitions,
            "rejections"=>k_reject,"rejection_rate"=>k_reject/repetitions,
            "rejection_rate_CP95_interval"=>ccg_interval(k_reject,repetitions),
            "positive_critical_full_evaluations"=>resolved,"inner_targets_evaluated"=>evaluated,
            "population_normal_signal_SE"=>s.signal_se))
    end
    Dict("rows"=>rows,"repetitions_per_case"=>repetitions,
        "total_inner_target_evaluations"=>sum(row["inner_targets_evaluated"] for r in records for row in r["result"]["results"]),
        "total_decision_pruned_targets"=>sum(row["inner_targets_decision_pruned"] for r in records for row in r["result"]["results"]),
        "failed_replaced_outer_or_inner_draws"=>0,
        "full_Clarabel_fallbacks"=>sum(row["fallbacks"] for r in records for row in r["result"]["results"]),
        "tuning_conclusion"=>"Diagnostic only: existing c=1 remains primary; no optimum or uniform size established")
end

function ccg_run(; repetitions=CCG_M,bootstrap=CCG_B,seed=CCG_SEED,probe=false,
        workers=min(6,Threads.nthreads()))
    BLAS.set_num_threads(1)
    output=joinpath(@__DIR__,"results",probe ? "cmu_c_geometry_probe_20261012_v1" : "cmu_c_geometry_simulation_20261012_v1")
    ispath(output)&&error("refuse to overwrite existing campaign: $output")
    mkpath(output)
    protocol=ccg_protocol(;repetitions,bootstrap,seed,probe)
    protocol["source_hashes"]=Dict(f=>ccg_sha(joinpath(@__DIR__,f)) for f in
        ("simulate_cmu_c_geometry.jl","cmu_c_geometry_design.jl","cmu_superiority_floor_inference.jl",
         "wilfried_normalized_working_set.jl","cmu_empty_inner_projection.jl","Project.toml","Manifest.toml"))
    write(joinpath(output,"protocol_frozen_before_outcomes.json"),JSON3.write(protocol))
    start=time(); designs=[ccgd_design(c;rate=CCG_RATE) for c in CCG_CELLS]
    write(joinpath(output,"geometry_and_means.json"),JSON3.write([d.record for d in designs]))
    records=Dict{String,Any}[]
    for d in designs
        jobs=collect(1:repetitions); results=Vector{Vector{Dict{String,Any}}}(undef,repetitions)
        done=Threads.Atomic{Int}(0)
        @sync for worker in 1:workers
            Threads.@spawn begin
                source=csfi_make_fit(d.A,d.groups;tolerance=CCG_TOL,store_certificate=false)
                for rep in worker:workers:repetitions
                    rng=MersenneTwister(seed+1_000_003d.cell+10_007rep)
                    # Independent inner bank for each outer replicate; reuse only
                    # within that replicate to compare c/scenarios with less MC noise.
                    outer=d.L*randn(rng,size(d.L,2))
                    inner=d.L*randn(rng,size(d.L,2),bootstrap)
                    noise_digest=bytes2hex(sha256(reinterpret(UInt8,vcat(vec(outer),vec(inner)))))
                    localrows=Dict{String,Any}[]
                    for (sidx,s) in enumerate(d.scenarios)
                        point=s.mean .+ outer./CCG_RATE
                        result=ccg_decisions(source.fit,point,inner,CCG_RATE;store_trace=true)
                        push!(localrows,Dict("cell"=>d.cell,"replication"=>rep,
                            "scenario_index"=>sidx,"scenario"=>s.id,
                            "seed"=>seed+1_000_003d.cell+10_007rep,
                            "full_latent_outer_inner_noise_sha256"=>noise_digest,"result"=>result))
                    end
                    results[rep]=localrows
                    completed=Threads.atomic_add!(done,1)+1
                    if completed%25==0 || completed==repetitions
                        println("cell=",d.cell," completed outer=",completed,"/",repetitions,
                            " elapsed_seconds=",round(time()-start;digits=1));flush(stdout)
                    end
                end
            end
        end
        cellrecords=reduce(vcat,results)
        write(joinpath(output,"cell_$(lpad(d.cell,2,'0'))_replicates.json"),JSON3.write(cellrecords))
        append!(records,cellrecords)
    end
    summary=ccg_summary(records,designs;repetitions)
    summary["protocol"]=protocol;summary["elapsed_seconds"]=time()-start
    write(joinpath(output,"summary.json"),JSON3.write(summary))
    println("completed ",output," elapsed=",summary["elapsed_seconds"])
    summary
end

if abspath(PROGRAM_FILE)==@__FILE__
    probe="--probe" in ARGS
    ccg_run(;probe,repetitions=probe ? 3 : CCG_M,bootstrap=probe ? 39 : CCG_B)
end
