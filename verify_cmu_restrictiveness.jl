# Independent primitive/geometry audit. No benchmark, fitting or ray helper imports.
using KSReplication, JSON3, SHA, Random, Distributions, Statistics
include(joinpath(@__DIR__, "verify_cmu_superiority_floor.jl"))
const CRVA = IndependentSuperiorityFloorAudit
const CRVQ = Rational{BigInt}
const CRV_CAPS = [0., .005, .01, .025, .05, .10, .20]
const CRV_PRIORS = [1., .25]
crv_read(p) = JSON3.read(read(p,String))
crv_hash(p) = bytes2hex(sha256(read(p)))
crv_q(p) = parse(CRVQ,String(p))
crv_check(x,msg) = x || error(msg)

function crv_probability(seed,groups,alpha)
    rng=MersenneTwister(seed);p=zeros(CRVQ,maximum(reduce(vcat,groups)))
    for g in groups
        k=length(g)
        if alpha==1.
            u=sort(rand(rng,k-1))
            p[g]=diff(vcat(CRVQ(0),CRVQ.(u),CRVQ(1)))
        else
            z=CRVQ.(k==1 ? [1.] : rand(rng,Dirichlet(k,.25)))
            z ./= sum(z);D=big(2)^40;v=z.*D
            n=floor.(BigInt,v);remainder=Int(D-sum(n))
            order=sort(collect(1:k);by=i->(-(v[i]-n[i]),i))
            for i in order[1:remainder]; n[i]+=1; end
            p[g]=n .// D
        end
    end
    crv_check(all(g->sum(p[g])==1&&all(>=(0),p[g]),groups),"reference probability invalid")
    p
end

function crv_interval(record,lo,hi)
    crv_check(crv_q(record["lower_exact"])==lo && crv_q(record["upper_exact"])==hi,
        "saved exact interval differs")
    crv_check(Float64(record["lower"])==CRVA.lower_float(lo) &&
        Float64(record["upper"])==CRVA.upper_float(hi),"outward interval endpoint differs")
    crv_check(record["midpoint"]==Float64((lo+hi)/2) &&
        record["width"]==CRVA.upper_float(hi-lo),"interval midpoint/width differs")
    (lo,hi)
end

function crv_target(r,ctxR,ctxW,counters)
    cell=Int(r["cell"]);j=Int(r["prior_index"]);rep=Int(r["repetition"])
    seed=202610125+1_000_003cell+100_003j+10_007rep
    crv_check(r["seed"]==seed && r["alpha"]==CRV_PRIORS[j],"seed/prior changed")
    p=crv_probability(seed,ctxW.groups,CRV_PRIORS[j])
    crv_check(crv_q.(r["point_exact"])==p && CRVQ.(Float64.(r["point"]))==p,"generated target differs")
    crv_check(crv_q(r["simplex_floor_exact"])==0,"valid reference has nonzero floor")
    R=CRVA.verify_compact_certificate(ctxR,Float64.(p),r["RUM_certificate"],counters)
    W=CRVA.verify_compact_certificate(ctxW,Float64.(p),r["WARP_certificate"],counters)
    crv_check(R.floor==W.floor==0,"reference fit floor differs")
    crv_interval(r["RUM_loss"],R.lower,R.upper)
    crv_interval(r["WARP_loss"],W.lower,W.upper)
    crv_interval(r["Psi_RUM_minus_simplex_floor"],R.lower,R.upper)
    cert=r["WARP_certificate"]
    if get(cert,"analytic_RUM_equals_product_simplex",false)
        q=copy(p)
    else
        indices=Int.(cert["complete_column_indices"])
        counts=parse.(BigInt,String.(cert["population_mass_numerators"]))
        D=parse(BigInt,String(cert["population_mass_common_denominator"]))
        qnum=zeros(BigInt,ctxW.I)
        for (h,n) in zip(indices,counts),k in 1:ctxW.J; qnum[ctxW.rows[k,h]]+=n; end
        q=qnum .// D
    end
    crv_check(q==crv_q.(r["qW_exact"]),"saved WARP feasible point differs")
    v=p-q;V=sum(abs2,v);D=foldl(lcm,denominator.(v);init=big(1))
    vn=[numerator(x)*div(D,denominator(x)) for x in v]
    hW=CRVA.support_integer(ctxW.rows,vn).value//D
    hS=sum(maximum(v[g]) for g in ctxW.groups)
    a0=sum(p.*v)
    crv_check(v==crv_q.(r["ray_residual_v_exact"]) && V==crv_q(r["ray_norm_squared_exact"]) &&
        hW==crv_q(r["ray_WARP_support_exact"]) && hS==crv_q(r["ray_simplex_support_exact"]) &&
        a0==crv_q(r["ray_p_dot_v_exact"]),"exact support/ray primitive differs")
    crv_check(r["all_WARP_columns_priced"]==ctxW.H && hW<=hS,"WARP support scope differs")
    pairs=Tuple{CRVQ,CRVQ}[]
    crv_check(length(r["caps"])==length(CRV_CAPS),"cap registry incomplete")
    for (cap,b) in zip(CRV_CAPS,r["caps"])
        crv_check(b["cap"]==cap,"cap changed")
        tau=1-CRVQ(cap);beta=min(CRVQ(1),(1-tau)/((ctxW.J-1)*tau))
        support=(1-beta)*hW+beta*hS;a=a0-support
        t=V==0 ? CRVQ(0) : max(CRVQ(0),a)/V
        ray=V==0 ? CRVQ(0) : max(CRVQ(0),a)^2/V
        blend=(1-beta)^2*V
        lo,hi=beta==0 ? (W.lower,W.upper) : beta==1 ? (CRVQ(0),CRVQ(0)) : (ray,blend)
        crv_check(0<=lo<=hi && ray<=blend,"geometric interval invalid")
        for (key,value) in (("tau_exact",tau),("beta_exact",beta),("support_B_exact",support),
            ("ray_advantage_exact",a),("optimized_dual_ray_multiplier_exact",t),
            ("dual_ray_loss_lower_exact",ray),("feasible_blend_loss_upper_exact",blend),
            ("complete_population_mass_exact",tau*(1-beta)),
            ("singleton_population_mass_per_budget_exact",tau*beta),
            ("all_null_population_mass_exact",1-tau*(1+(ctxW.J-1)*beta)))
            crv_check(crv_q(b[key])==value,"cap primitive differs: $key")
        end
        crv_interval(b["loss"],lo,hi)
        crv_interval(b["RUM_minus_inner_gain"],max(CRVQ(0),R.lower-hi),R.upper-lo)
        crv_interval(b["WARP_minus_inner_gain"],max(CRVQ(0),W.lower-hi),W.upper-lo)
        push!(pairs,(lo,hi))
    end
    (;R=(R.lower,R.upper),W=(W.lower,W.upper),caps=pairs)
end

function crv_mean(saved,pairs)
    lo=sum(first,pairs)/length(pairs);hi=sum(last,pairs)/length(pairs)
    crv_interval(saved,lo,hi)
    mids=[Float64((a+b)/2) for (a,b) in pairs]
    se=length(pairs)>1 ? std(mids)/sqrt(length(pairs)) : nothing
    crv_check(se===nothing ? saved["Monte_Carlo_SE_midpoint"]===nothing :
        isapprox(saved["Monte_Carlo_SE_midpoint"],se;rtol=1e-13,atol=1e-15),"MC midpoint SE differs")
    (lo,hi)
end

function verify_cmu_restrictiveness(directory=joinpath(@__DIR__,"results/cmu_restrictiveness_20261012_v1"))
    summary=crv_read(joinpath(directory,"summary.json"));protocol=summary["protocol"]
    crv_check(protocol["cells"]==collect(1:19)&&Float64.(protocol["priors"])==CRV_PRIORS&&
        Float64.(protocol["empty_allowance_caps"])==CRV_CAPS,"protocol models/priors differ")
    crv_check(CRVA.same(protocol["source_and_input_sha256_before"],summary["source_and_input_sha256_after"]),
        "protected sources differ")
    for (path,digest) in pairs(protocol["source_and_input_sha256_before"])
        crv_check(crv_hash(joinpath(@__DIR__,String(path)))==String(digest),"source hash changed: $path")
    end
    M=Int(protocol["targets_per_prior_per_window"]);counters=CRVA.Counters()
    cells=Dict{Int,Any}();counts=Int[];filehashes=Dict{String,String}()
    for cell in 1:19
        name="cell_$(lpad(cell,2,'0'))_targets.json";path=joinpath(directory,name)
        g=crv_read(joinpath(directory,"geometry_$(lpad(cell,2,'0')).json"))
        X=reduce(vcat,permutedims(Int8.(row)) for row in g["sign_matrix"])
        groups=CRVA.patch_groups(X)
        AR=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
        AW=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
        for (A,key) in ((AR,"RUM_matrix_sha256"),(AW,"WARP_matrix_sha256"))
            crv_check(bytes2hex(sha256(reinterpret(UInt8,vec(A))))==String(g[key]),"regenerated matrix differs")
        end
        records=crv_read(path);crv_check(length(records)==2M,"reference target count differs")
        crv_check(Set((Int(r["prior_index"]),Int(r["repetition"])) for r in records)==
            Set((j,k) for j in 1:2 for k in 1:M),"reference target registry incomplete")
        ctxR=CRVA.complete_context(AR,groups);ctxW=CRVA.complete_context(AW,groups)
        parsed=Dict((Int(r["prior_index"]),Int(r["repetition"]))=>crv_target(r,ctxR,ctxW,counters) for r in records)
        cells[cell]=parsed;push!(counts,Int(g["frozen_annual_n_sum"]))
        filehashes[name]=crv_hash(path)
        println("restrictiveness independent audit cell=",cell," certificates=",counters.certificates);flush(stdout)
    end
    weights=CRVQ.(counts)./sum(counts)
    for row in summary["per_window"]
        cell=Int(row["cell"]);j=findfirst(==(Float64(row["alpha"])),CRV_PRIORS)
        rs=[cells[cell][(j,k)] for k in 1:M]
        crv_mean(row["RUM_MSE"],[r.R for r in rs]);crv_mean(row["WARP_MSE"],[r.W for r in rs])
        for (i,b) in enumerate(row["inner_caps"])
            crv_mean(b["MSE"],[r.caps[i] for r in rs])
            crv_mean(b["RUM_minus_inner_mean_gain"],[(max(CRVQ(0),r.R[1]-r.caps[i][2]),r.R[2]-r.caps[i][1]) for r in rs])
            crv_mean(b["WARP_minus_inner_mean_gain"],[(max(CRVQ(0),r.W[1]-r.caps[i][2]),r.W[2]-r.caps[i][1]) for r in rs])
        end
    end
    for row in summary["pooled"]
        j=findfirst(==(Float64(row["alpha"])),CRV_PRIORS)
        function aggregate(f)
            [(sum(weights[c]*f(cells[c][(j,k)])[1] for c in 1:19),
              sum(weights[c]*f(cells[c][(j,k)])[2] for c in 1:19)) for k in 1:M]
        end
        R=crv_mean(row["RUM_MSE"],aggregate(r->r.R));crv_mean(row["WARP_MSE"],aggregate(r->r.W))
        for (i,b) in enumerate(row["inner_caps"])
            B=crv_mean(b["MSE"],aggregate(r->r.caps[i]))
            crv_mean(b["RUM_minus_inner_mean_gain"],aggregate(r->(max(CRVQ(0),r.R[1]-r.caps[i][2]),r.R[2]-r.caps[i][1])))
            crv_mean(b["WARP_minus_inner_mean_gain"],aggregate(r->(max(CRVQ(0),r.W[1]-r.caps[i][2]),r.W[2]-r.caps[i][1])))
            if R[1]>0
                ratio=b["inner_to_RUM_ratio_of_mean_losses"]
                crv_check(crv_q(ratio["lower_exact"])==B[1]/R[2]&&
                    crv_q(ratio["upper_exact"])==min(CRVQ(1),B[2]/R[1]),"pooled loss ratio differs")
            end
        end
    end
    Dict("status"=>"passed_independent_restrictiveness_primitive_and_aggregate_audit",
        "verifier_sha256"=>crv_hash(@__FILE__),"summary_sha256"=>crv_hash(joinpath(directory,"summary.json")),
        "target_file_sha256"=>filehashes,"regenerated_reference_targets"=>19*2M,
        "exact_full_support_projection_certificates"=>counters.certificates,
        "exact_ray_cap_intervals"=>19*2M*length(CRV_CAPS),
        "maximum_RUM_WARP_exact_objective_width"=>string(counters.maximum_objective_width),
        "sampling_or_full_CMU_validity_proved"=>false,
        "scope"=>"Exact probability/RNG, complete type, primitive projection support, geometric ray/population accounting and aggregation audit; not statistical power or all-CMU characterization")
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)<=1 || error("usage: verify_cmu_restrictiveness.jl [completed-directory]")
    result=isempty(ARGS) ? verify_cmu_restrictiveness() : verify_cmu_restrictiveness(ARGS[1])
    println(JSON3.write(result))
end
