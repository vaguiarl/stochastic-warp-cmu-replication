# Isolated known-null-face prototype; no frozen empirical sources are edited.
# The finite Gaussian menu oracle, cone-law benchmark, and global norm bound
# have deliberately distinct validity labels below. No c is selected here.
using Random, LinearAlgebra, Statistics, Distributions, JSON3, SHA
include(joinpath(@__DIR__, "cmu_known_face_cones.jl"))
using .CMUKnownFaceCones
include(joinpath(@__DIR__, "cmu_c_geometry_design.jl"))
include(joinpath(@__DIR__, "cmu_superiority_floor_inference.jl"))

const KF_RATE = sqrt(46391 / 3)
const KF_B = 499
const KF_M = 250
const KF_SEED = 202610124
const KF_ALPHAS = [.05, .10]
const KF_NORMAL_DENOMINATOR = 64

kf_sha(path) = bytes2hex(sha256(read(path)))
kf_down(x) = begin f=Float64(x); Q(f)>x ? prevfloat(f) : f end
kf_up(x) = begin f=Float64(x); Q(f)<x ? nextfloat(f) : f end
kf_interval(k, m) = [k==0 ? 0. : quantile(Beta(k,m-k+1),.025),
                    k==m ? 1. : quantile(Beta(k+1,m-k),.975)]
kf_noise_sha(G) = bytes2hex(sha256(reinterpret(UInt8, vec(G))))

function kf_primitive_record(target, g, finite, gap, cone)
    mass_indices = findall(>(0),gap.rum.masses)
    Dict("target_exact"=>string.(target),"Gaussian_direction_exact"=>string.(Q.(g)),
        "finite_score_Psi_lower_exact"=>string(finite.lower),
        "finite_score_Psi_upper_exact"=>string(finite.upper),
        "finite_engine_certificate"=>finite.certificate,
        "finite_engine_diagnostic"=>finite.diagnostic,
        "finite_target_scope"=>"Exact symbolic rational q+n+Q(G)/Q(rate); engine target_float is the rounded numerical proposal ONLY, never the certificate's exact target. Reconstruct arithmetic from target_exact",
        "cone_Gamma_lower_exact"=>string(gap.lower),"cone_Gamma_upper_exact"=>string(gap.upper),
        "cone_RUM_loss_lower_exact"=>string(gap.rum.lower),
        "cone_RUM_loss_upper_exact"=>string(gap.rum.upper),
        "cone_simplex_loss_exact"=>string(gap.simplex.loss),
        "cone_primal_exact"=>string.(gap.rum.primal),
        "cone_dual_exact"=>string.(gap.rum.dual),
        "cone_primal_column_indices"=>cone.retained[mass_indices],
        "cone_primal_masses_exact"=>string.(gap.rum.masses[mass_indices]),
        "cone_simplex_projection_exact"=>string.(gap.simplex.point),
        "cone_analytic"=>gap.rum.analytic,
        "cone_certificate_scope"=>"Nonnegative unnormalized cone primal, exact global polar over every retained original vertex difference, exact simplex critical projection; no tolerance-as-polar or omitted zero coordinate")
end

function kf_decision(lower, upper, bank_lower, bank_upper, alpha;
        require_positive=false)
    B = length(bank_upper)
    length(bank_lower) == B || error("reference interval mismatch")
    order = ceil(Int, (1-alpha)*(B+1))
    1 <= order <= B || error("insufficient reference Monte Carlo resolution")
    critical = sort(bank_upper)[order]
    width = maximum(bank_upper .- bank_lower)
    midpoint_critical = sort((bank_lower .+ bank_upper) ./ 2)[order]
    positive = midpoint_critical > max(0., width)
    tails = count(>=(lower), bank_upper)
    p = (tails+1)/(B+1)
    reject = lower > critical && p <= alpha && (!require_positive || positive)
    Dict("alpha"=>alpha, "critical_upper"=>critical,
        "critical_lower"=>sort(bank_lower)[order], "tail_MC_plus_one"=>p,
        "reference_positive_resolved"=>positive,
        "reject"=>reject, "observed_lower"=>lower, "observed_upper"=>upper)
end

function kf_protocol(; B=KF_B, M=KF_M, pilot=false)
    Dict("version"=>"known-null-face-oracles-v1", "cells"=>[7,12],
        "B_reference_per_known_anchor"=>B, "M_validation_per_known_anchor"=>M,
        "seed"=>KF_SEED, "rate"=>KF_RATE, "alpha_levels"=>KF_ALPHAS,
        "pilot"=>pilot, "c_used_or_selected"=>false,
        "normal_denominator"=>KF_NORMAL_DENOMINATOR,
        "known_null_anchors"=>["centroid", "exposed_face", "raw_outside_simplex"],
        "reference_design"=>"One independent fixed Gaussian reference bank per known anchor; it does not depend on outer estimated means; fresh independent validation bank",
        "raw_normal"=>"n_i=(Z-k*1{q_i=0})/64 in each budget; exact zero adding-up; q+n may have negative coordinates but common RUM/simplex projection is q",
        "coordinates"=>"All original patch probabilities retained; no dropped coordinate, covariance inverse, ridge or rank truncation",
        "functional"=>"Psi(p)=d_complete_RUM(p)^2-d_product_simplex(p)^2 on the raw target",
        "finite_Gaussian_menu_oracle"=>"Reference and validation scores are the SAME finite T=r^2*Psi(p0+G/r); critical value is maximum across the three declared known null anchors. Monte Carlo rank control is unconditional over reference and validation banks for this finite menu under frozen Gaussian nuisance; not uniform over unknown null means",
        "cone_oracle"=>"Gamma=d_KR(G)^2-d_KS(G)^2 with exact critical cones at a known q,n; Gamma-versus-Gamma rank benchmark only. Finite T-versus-Gamma is a diagnostic and is NOT asserted size-valid from checked local samples",
        "global_norm_cap"=>"At every raw equal-distance null, finite T<=||G||^2; norm reference rank critical is conservative in the frozen Gaussian experiment",
        "intervals"=>"Clopper-Pearson marginal validation-rate intervals conditional on the realized shared reference bank, not unconditional size confidence or a c comparison",
        "UK_pvalues_changed"=>false,
        "limitations"=>["Known faces, means and synthetic normal; no feasible inferred-H/GMS theorem",
            "Conditional frozen nuisance, not household/control-function resimulation",
            "Per-window, not nineteen-window joint inference or selection of c",
            "Cone local transfer is checked direction by direction and not presumed for Gaussian tails"])
end

function kf_run(; B=KF_B, M=KF_M, pilot=false,
        output=joinpath(@__DIR__, "results", "cmu_known_face_oracle_20261012_v1"))
    19 <= B <= KF_B && 1 <= M <= KF_M || error("bounded campaign contract")
    ispath(output) && error("refusing to overwrite existing oracle output")
    BLAS.set_num_threads(1)
    protected = ["Project.toml", "Manifest.toml", "cmu_c_geometry_design.jl",
        "cmu_superiority_floor_inference.jl", "cmu_empty_inner_projection.jl",
        "run_cmu_superiority_floor.jl", "simulate_cmu_c_geometry.jl",
        "results/cmu_superiority_floor_20261012_v1/summary.json",
        "results/cmu_c_geometry_simulation_20261012_v1/summary.json"]
    protected_paths = [joinpath(@__DIR__, path) for path in protected]
    before = Dict(path=>kf_sha(full) for (path,full) in zip(protected,protected_paths))
    sources = Dict("cmu_known_face_cones.jl"=>kf_sha(joinpath(@__DIR__,"cmu_known_face_cones.jl")),
                   "test/test_cmu_known_face_cones.jl"=>kf_sha(joinpath(@__DIR__,"test/test_cmu_known_face_cones.jl")),
                   "simulate_cmu_known_face_oracle.jl"=>kf_sha(@__FILE__))
    protocol = kf_protocol(; B,M,pilot)
    mkdir(output)
    open(joinpath(output,"protocol.json"),"w") do io; JSON3.write(io,protocol); end
    summaries = Dict{String,Any}[]
    all_records = Dict{String,Any}[]
    started = time()
    for cell in (7,12)
        d = csdg_load_design(cell)
        centroid_weights = csdg_dyadic_weights(size(d.Ar,2)).weights
        nraw = raw_normal(d.q0_exact,d.groups; denominator=KF_NORMAL_DENOMINATOR)
        anchors = [
            (; id="centroid", q=d.centroid_exact, n=zeros(Q,length(d.q0)),
               cone=KnownFaceCone(d.Ar,d.centroid_exact,d.groups; centroid_weights)),
            (; id="exposed_face", q=d.q0_exact, n=zeros(Q,length(d.q0)),
               cone=KnownFaceCone(d.Ar,d.q0_exact,d.groups; face=d.face_indices,
                    exposing_normal=d.normal_exact,centroid_weights=d.face_weights_exact)),
            (; id="raw_outside_simplex", q=d.q0_exact, n=nraw,
               cone=KnownFaceCone(d.Ar,d.q0_exact,d.groups; face=d.face_indices,
                    exposing_normal=d.normal_exact,n=nraw,centroid_weights=d.face_weights_exact))]
        fit = csfi_make_fit(d.Ar,d.groups; tolerance=1e-8,store_certificate=true).fit
        banks = Dict{String,Any}[]
        validation_records = Dict{String,Any}[]
        for (ai,anchor) in enumerate(anchors)
            reference_seed = KF_SEED + 1_000_003cell + 100_003ai
            validation_seed = KF_SEED + 1_000_003cell + 100_003ai + 10_000_019
            Gref = d.factor * randn(MersenneTwister(reference_seed),length(d.q0),B)
            Gval = d.factor * randn(MersenneTwister(validation_seed),length(d.q0),M)
            mean_exact = anchor.q .+ anchor.n
            all(sum(mean_exact[group])==1 for group in d.groups) || error("raw anchor lost adding-up")
            mean = Float64.(mean_exact)
            Q.(mean) == mean_exact || error("anchor is not exactly representable")
            cone_lower,cone_upper = Float64[],Float64[]
            finite_lower,finite_upper = Float64[],Float64[]
            norm_lower,norm_upper = Float64[],Float64[]
            local_transfers, local_simplex_identities = 0,0
            max_cone_width,max_polar_price = 0.,Q(0)
            primitives = Dict{String,Any}[]
            for b in 1:B
                g = view(Gref,:,b)
                gap = critical_gap!(anchor.cone,g; include_primitives=true)
                target_exact = mean_exact .+ Q.(g) ./ Q(KF_RATE)
                finite = csfi_check_fit(fit(target_exact))
                scorelo = Q(KF_RATE)^2 * finite.lower
                scorehi = Q(KF_RATE)^2 * finite.upper
                push!(cone_lower,kf_down(gap.lower));push!(cone_upper,kf_up(gap.upper))
                push!(finite_lower,kf_down(scorelo));push!(finite_upper,kf_up(scorehi))
                norm = sum(abs2,Q.(g))
                push!(norm_lower,kf_down(norm));push!(norm_upper,kf_up(norm))
                scorelo <= norm || error("finite null score violates the global norm-cap bound")
                transfer = finite_gap_bound(anchor.cone,g,inv(Q(KF_RATE)),gap)
                if transfer !== nothing
                    local_transfers += 1
                    local_simplex_identities += transfer.simplex_local_identity
                    max(transfer.lower,finite.lower) <= min(transfer.upper,finite.upper) ||
                        error("same exact target finite/cone certificates disagree")
                end
                max_cone_width=max(max_cone_width,gap.width)
                max_polar_price=max(max_polar_price,gap.rum.max_polar_price)
                push!(primitives,kf_primitive_record(target_exact,g,finite,gap,anchor.cone))
            end
            bank = Dict("anchor"=>anchor.id,"q_exact"=>string.(anchor.q),
                "n_exact"=>string.(anchor.n),"mean_exact"=>string.(mean_exact),
                "reference_seed"=>reference_seed,"validation_seed"=>validation_seed,
                "reference_noise_sha256"=>kf_noise_sha(Gref),
                "validation_noise_sha256"=>kf_noise_sha(Gval),
                "critical_RUM_generator_count"=>length(anchor.cone.retained),
                "exposed_face_span_exact_dimension"=>size(anchor.cone.basis,2),
                "cone_lower"=>cone_lower,"cone_upper"=>cone_upper,
                "finite_lower"=>finite_lower,"finite_upper"=>finite_upper,
                "norm_lower"=>norm_lower,"norm_upper"=>norm_upper,
                "reference_finite_transfer_certificates"=>local_transfers,
                "reference_simplex_local_identities"=>local_simplex_identities,
                "reference_primitive_certificates"=>primitives,
                "maximum_cone_interval_width"=>max_cone_width,
                "maximum_exact_polar_price"=>string(max_polar_price))
            push!(banks,bank)
            for m in 1:M
                g = view(Gval,:,m)
                gap = critical_gap!(anchor.cone,g; include_primitives=true)
                target_exact = mean_exact .+ Q.(g) ./ Q(KF_RATE)
                finite = csfi_check_fit(fit(target_exact))
                scorelo,scorehi = Q(KF_RATE)^2*finite.lower,Q(KF_RATE)^2*finite.upper
                norm = sum(abs2,Q.(g))
                scorelo <= norm || error("validation null violates global norm cap")
                transfer = finite_gap_bound(anchor.cone,g,inv(Q(KF_RATE)),gap)
                transfer === nothing ||
                    max(transfer.lower,finite.lower) <= min(transfer.upper,finite.upper) ||
                    error("validation exact finite/cone certificates disagree")
                decisions = Dict{String,Any}[]
                for alpha in KF_ALPHAS
                    coned = kf_decision(kf_down(gap.lower),kf_up(gap.upper),cone_lower,cone_upper,alpha;
                        require_positive=true)
                    diagnostic = kf_decision(kf_down(scorelo),kf_up(scorehi),cone_lower,cone_upper,alpha;
                        require_positive=true)
                    normd = kf_decision(kf_down(scorelo),kf_up(scorehi),norm_lower,norm_upper,alpha)
                    push!(decisions,Dict("alpha"=>alpha,"cone_law_rank_benchmark"=>coned,
                        "finite_score_vs_cone_diagnostic_only"=>diagnostic,
                        "finite_global_norm_cap"=>normd))
                end
                push!(validation_records,Dict("cell"=>cell,"anchor"=>anchor.id,"replication"=>m,
                    "finite_lower"=>kf_down(scorelo),"finite_upper"=>kf_up(scorehi),
                    "cone_lower"=>kf_down(gap.lower),"cone_upper"=>kf_up(gap.upper),
                    "norm_squared_lower"=>kf_down(norm),"norm_squared_upper"=>kf_up(norm),
                    "finite_transfer_available"=>transfer!==nothing,
                    "simplex_local_identity"=>transfer===nothing ? false : transfer.simplex_local_identity,
                    "finite_transfer_exact_lower"=>transfer===nothing ? nothing : string(Q(KF_RATE)^2*transfer.lower),
                    "finite_transfer_exact_upper"=>transfer===nothing ? nothing : string(Q(KF_RATE)^2*transfer.upper),
                    "primitive_certificate"=>kf_primitive_record(target_exact,g,finite,gap,anchor.cone),
                    "cone_interval_width"=>gap.width,"decisions"=>decisions))
            end
        end
        # A finite-menu oracle must calibrate the finite score, not Gamma. The
        # maximum of individual-anchor reference quantiles dominates each
        # anchor's own reference critical. The scope is the declared finite menu.
        for record in validation_records, decision in record["decisions"]
            a = decision["alpha"]
            order = ceil(Int,(1-a)*(B+1))
            critical = maximum(sort(bank["finite_upper"])[order] for bank in banks)
            ownbank = only(filter(bank->bank["anchor"]==record["anchor"],banks))
            own = kf_decision(record["finite_lower"],record["finite_upper"],
                ownbank["finite_lower"],ownbank["finite_upper"],a)
            menu_reject = record["finite_lower"] > critical && own["tail_MC_plus_one"] <= a
            decision["finite_known_null_menu_oracle"] = Dict("alpha"=>a,"critical_upper"=>critical,
                "own_anchor_tail_MC_plus_one"=>own["tail_MC_plus_one"],"reject"=>menu_reject)
        end
        for anchor in anchors, a in KF_ALPHAS,
                method in ("cone_law_rank_benchmark","finite_score_vs_cone_diagnostic_only",
                           "finite_global_norm_cap","finite_known_null_menu_oracle")
            selected = filter(r->r["anchor"]==anchor.id,validation_records)
            k = count(r->only(filter(d->d["alpha"]==a,r["decisions"]))[method]["reject"],selected)
            push!(summaries,Dict("cell"=>cell,"anchor"=>anchor.id,"alpha"=>a,
                "method"=>method,"rejections"=>k,"validation_repetitions"=>M,
                "rejection_rate"=>k/M,"CP95_conditional_on_reference_bank"=>kf_interval(k,M),
                "finite_transfer_count"=>count(r->r["finite_transfer_available"],selected),
                "simplex_local_identity_count"=>count(r->r["simplex_local_identity"],selected)))
        end
        celloutput = Dict("cell"=>cell,"geometry"=>d.metadata,"banks"=>banks,
            "validation_records"=>validation_records,"protocol"=>protocol)
        open(joinpath(output,"cell_$(lpad(cell,2,'0')).json"),"w") do io;JSON3.write(io,celloutput);end
        append!(all_records,validation_records)
        println("KNOWN_FACE_CELL_DONE ",cell," elapsed=",round(time()-started;digits=2),
            " reference=",3B," validation=",3M)
        flush(stdout)
    end
    after = Dict(path=>kf_sha(full) for (path,full) in zip(protected,protected_paths))
    before == after || error("a frozen empirical source changed during oracle benchmark")
    summary = Dict("protocol"=>protocol,"rows"=>summaries,"elapsed_seconds"=>time()-started,
        "all_reference_draws"=>6B,"all_validation_draws"=>6M,
        "omitted_or_replaced_Gaussian_draws"=>0,"protected_before"=>before,"protected_after"=>after,
        "new_source_sha256"=>sources,"conditional_CP_intervals"=>true,
        "empirical_UK_test_or_pvalues_replaced"=>false,
        "scope"=>"Known-null-face and finite Gaussian MENU oracle only; feasible inferred-H selection and UK nuisance inference remain open")
    open(joinpath(output,"summary.json"),"w") do io;JSON3.write(io,summary);end
    println("KNOWN_FACE_CAMPAIGN_DONE ",round(time()-started;digits=2))
    summary
end

if abspath(PROGRAM_FILE)==@__FILE__
    if "--pilot" in ARGS
        kf_run(B=39,M=10,pilot=true,
            output=joinpath(@__DIR__,"results","cmu_known_face_oracle_pilot_20261012_v2"))
    else
        kf_run()
    end
end
