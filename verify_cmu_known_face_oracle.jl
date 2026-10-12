#!/usr/bin/env julia
"""Independent exact replay of the isolated known-face Gaussian oracle.

Imports no campaign driver, geometry helper, projection helper or inference
engine. Existing KS enumeration alone regenerates the original type columns.
This verifies deterministic certificates, seeded banks and Monte Carlo
aggregation. It does not establish feasible inferred-face, UK nuisance,
joint-window or uniform-over-unknown-means inference validity.
"""
module IndependentCMUKnownFaceOracleAudit

using KSReplication, JSON3, LinearAlgebra, Statistics, Random, SHA, Distributions

const ROOT = @__DIR__
const DEFAULT_DIRECTORY = joinpath(ROOT,"results/cmu_known_face_oracle_20261012_v1")
const Q = Rational{BigInt}
const RATE = sqrt(46391/3)
const SEED = 202610124
const CELLS = [7,12]
const ALPHAS = [.05,.10]
const ANCHORS = ["centroid","exposed_face","raw_outside_simplex"]
const METHODS = ["cone_law_rank_benchmark","finite_score_vs_cone_diagnostic_only",
                 "finite_global_norm_cap","finite_known_null_menu_oracle"]

check(ok,message) = ok || throw(ArgumentError(message))
readjson(path) = JSON3.read(read(path,String),Any)
filehash(path) = bytes2hex(sha256(read(path)))
parseq(x) = parse(Q,String(x))
qdot(x,y) = sum(x.*y;init=Q(0))
qnorm(x) = qdot(x,x)
noisehash(G) = bytes2hex(sha256(reinterpret(UInt8,vec(G))))
down(x) = begin f=Float64(x);Q(f)>x ? prevfloat(f) : f end
up(x) = begin f=Float64(x);Q(f)<x ? nextfloat(f) : f end
afraction(a) = a==.05 ? (1,20) : a==.10 ? (1,10) : error("unfrozen alpha")
order(a,B) = begin n,d=afraction(a);cld((d-n)*(B+1),d) end
tailok(k,a,B) = begin n,d=afraction(a);d*(k+1)<=n*(B+1) end

function equal(a,b)
    if a isa AbstractDict && b isa AbstractDict
        return Set(keys(a))==Set(keys(b)) && all(equal(a[k],b[k]) for k in keys(a))
    elseif a isa AbstractArray && b isa AbstractArray
        return size(a)==size(b) && all(equal(a[i],b[i]) for i in eachindex(a))
    end
    a==b
end

function dyadic(A,columns)
    D=big(1)
    while D<length(columns);D*=2;end
    base,remainder=divrem(D,length(columns))
    counts=[base+(i<=remainder ? 1 : 0) for i in eachindex(columns)]
    numerator=zeros(BigInt,size(A,1))
    for (h,m) in zip(columns,counts),i in axes(A,1)
        A[i,h]==1 && (numerator[i]+=m)
    end
    (;point=numerator.//D,counts,denominator=D)
end

function geometry(record)
    cell=Int(record["cell"])
    source=readjson(joinpath(ROOT,"results/wilfried_windows_20261009_v2/summary.json"))
    case=only(filter(c->Int(c["cell"])==cell,source["cases"]))
    X=reduce(vcat,permutedims(Int8.(row)) for row in case["sign_matrix"])
    check(all(i->count(==(0),X[i,:])==1,axes(X,1)),"invalid patch partition")
    groups=[findall(==(0),X[:,j]) for j in axes(X,2)]
    A=enumerate_types_fast(X;restriction=:rum,max_types=2_000_000,max_nodes=50_000_000)
    Aw=enumerate_types_fast(X;restriction=:warp,max_types=2_000_000,max_nodes=50_000_000)
    check(all(x->x==0||x==1,A),"non-incidence regenerated RUM")
    check(size(A,2)==record["RUM_types"] && size(Aw,2)==record["WARP_types"],"type count mismatch")
    rum=Set(Tuple(A[:,h]) for h in axes(A,2))
    check(length(rum)==size(A,2) && all(Tuple(A[:,h]) in Set(Tuple(Aw[:,k]) for k in axes(Aw,2)) for h in axes(A,2)),"type nesting mismatch")
    excluded=first(h for h in axes(Aw,2) if !(Tuple(Aw[:,h]) in rum))
    w=Int.(Aw[:,excluded]);scores=[sum(w.*A[:,h]) for h in axes(A,2)]
    face=findall(==(maximum(scores)),scores)
    check(excluded==record["excluded_WARP_column_index"] && w==Int.(record["w"]),"canonical excluded type mismatch")
    check(scores==Int.(record["all_RUM_column_scores_exact"]) && face==Int.(record["face_column_indices"]),"exposed face mismatch")
    check(maximum(scores)<sum(w),"strict RUM/WARP separation lost")
    f=dyadic(A,face);c=dyadic(A,collect(axes(A,2)))
    check(f.point==parseq.(record["q0_exact"]) && c.point==parseq.(record["centroid_exact"]),"known mean mismatch")
    eta=Q.(w)
    for g in groups;eta[g].-=sum(eta[g])/length(g);end
    check(eta==parseq.(record["affine_centered_normal_exact"]),"exposing normal mismatch")
    rows=hcat([[only(filter(i->A[i,h]==1,g)) for g in groups] for h in axes(A,2)]...)
    for hashes in (record["protected_sha256_before"],record["protected_sha256_after"])
        for (relative,digest) in hashes
            check(filehash(joinpath(ROOT,relative))==digest,"geometry source changed: $relative")
        end
    end
    check(equal(record["protected_sha256_before"],record["protected_sha256_after"]),"geometry source mutation")
    check(record["helper_source_sha256"]==filehash(joinpath(ROOT,"cmu_c_geometry_design.jl")),"geometry helper source changed")
    cache=readjson(joinpath(ROOT,"results/wilfried_joint_gap_20261010_v1/raw_bootstrap_cell_"*lpad(string(cell),2,"0")*".json"))
    stars=reshape(Float64.(cache["raw_affine_bootstrap"]),size(A,1),Int(cache["B"]))
    F=(stars.-vec(mean(stars;dims=2))).*(RATE/sqrt(size(stars,2)-1))
    L=Matrix(transpose(Matrix(qr(Matrix(transpose(F))).R)))
    check(size(L)==(size(A,1),size(A,1)),"full-coordinate QR changed dimension")
    check(record["covariance_inverse_ridge_or_rank_truncation"]===false && record["observed_fit_gains_or_empirical_pvalues_used"]===false,"unsupported geometry selection/metric")
    (;A,groups,rows,face,eta,L,centroid=c.point,q0=f.point)
end

function nums(values)
    D=foldl(lcm,denominator.(values);init=big(1))
    [numerator(v)*div(D,denominator(v)) for v in values],D
end

function support(values,rows,columns=collect(axes(rows,2)))
    yn,D=nums(values)
    maximum(sum(yn[rows[j,h]] for j in axes(rows,1)) for h in columns)//D
end

function mixture(rows,I,columns,mass)
    yn,D=nums(mass);p=zeros(BigInt,I)
    for (h,m) in zip(columns,yn),j in axes(rows,1);p[rows[j,h]]+=m;end
    p.//D
end

function simplex_kkt(p,x,groups)
    check(length(p)==length(x) && all(x.>=0),"finite simplex coordinates invalid")
    for group in groups
        check(sum(x[group])==1,"finite simplex adding-up failed")
        positive=filter(i->x[i]>0,group)
        level=p[first(positive)]-x[first(positive)]
        check(all(p[i]-x[i]==level for i in positive) && all(p[i]-x[i]<=level for i in group),"finite simplex exact KKT failed")
    end
    qnorm(p-x)
end

function critical_simplex_kkt(g,z,q,n,groups)
    check(length(g)==length(z)==length(q)==length(n),"critical simplex dimension mismatch")
    for group in groups
        free=filter(i->q[i]>0,group);level=n[first(free)]
        check(all(n[i]==level for i in free) && all(n[i]<=level for i in group),"invalid known simplex normal")
        theta=g[first(free)]-z[first(free)]
        check(sum(z[group])==0 && all(g[i]-z[i]==theta for i in free),"critical simplex free KKT failed")
        for i in group
            q[i]>0 && continue
            if n[i]<level
                check(z[i]==0,"positive normal multiplier coordinate moved")
            else
                check(z[i]>=0 && g[i]-z[i]<=theta && (z[i]==0||g[i]-z[i]==theta),"critical simplex inequality KKT failed")
            end
        end
    end
    qnorm(g-z)
end

function replay(c,g,q,n,geo; rate=RATE)
    A,groups,rows=geo.A,geo.groups,geo.rows
    I,H=size(A);p=parseq.(c["target_exact"])
    check(parseq.(c["Gaussian_direction_exact"])==Q.(g),"Gaussian exact direction differs from seeded bank")
    check(p==q+n+Q.(g)/Q(rate),"symbolic rational target mismatch")
    f=c["finite_engine_certificate"]
    columns=Int.(f["complete_column_indices"])
    masses=parseq.(f["population_mass_numerators"])./parseq(f["population_mass_common_denominator"])
    check(all(h->1<=h<=H,columns) && length(columns)==length(masses) && all(masses.>=0) && sum(masses)==1,"finite normalized complete primal invalid")
    x=mixture(rows,I,columns,masses);residual=p-x
    s=parseq.(f["simplex_target_exact"]);floor=simplex_kkt(p,s,groups)
    check(floor==parseq(f["probability_domain_floor_exact"]),"finite simplex loss mismatch")
    upper=qnorm(residual);lower=max(floor,2qdot(p,residual)-upper-2support(residual,rows))
    check(lower==parseq(f["RUM_squared_distance_lower_exact"]) && upper==parseq(f["RUM_squared_distance_upper_exact"]),"finite complete-support endpoint mismatch")
    flo,fhi=lower-floor,upper-floor
    check(0<=flo<=fhi && flo==parseq(c["finite_score_Psi_lower_exact"]) && fhi==parseq(c["finite_score_Psi_upper_exact"]),"finite Psi endpoint mismatch")
    check(f["all_complete_columns_priced"]==H && f["fixed_target_rational_certificate"]===true,"incomplete finite support audit")
    retained=[h for h in 1:H if qdot(n,Q.(A[:,h])-q)==0]
    check(support(n,rows)<=qdot(n,q),"known normal fails complete RUM support")
    hs=Int.(c["cone_primal_column_indices"]);mass=parseq.(c["cone_primal_masses_exact"])
    check(length(hs)==length(mass) && all(>(0),mass) && all(h in retained for h in hs),"cone masses/critical generators invalid")
    z=parseq.(c["cone_primal_exact"]);y=parseq.(c["cone_dual_exact"])
    check(length(z)==length(y)==I && mixture(rows,I,hs,mass)-q*sum(mass)==z,"unnormalized cone primal identity failed")
    maxprice=support(y,rows,retained)-qdot(y,q)
    check(maxprice<=0,"exact cone dual is not polar")
    gq=Q.(g);rlo=max(Q(0),2qdot(gq,y)-qnorm(y));rhi=qnorm(gq-z)
    check(rlo<=rhi && rlo==parseq(c["cone_RUM_loss_lower_exact"]) && rhi==parseq(c["cone_RUM_loss_upper_exact"]),"cone dual/primal loss mismatch")
    zs=parseq.(c["cone_simplex_projection_exact"])
    sloss=critical_simplex_kkt(gq,zs,q,n,groups)
    check(sloss==parseq(c["cone_simplex_loss_exact"]),"critical simplex loss mismatch")
    glo,ghi=max(Q(0),rlo-sloss),rhi-sloss
    check(0<=glo<=ghi && glo==parseq(c["cone_Gamma_lower_exact"]) && ghi==parseq(c["cone_Gamma_upper_exact"]),"Gamma endpoint mismatch")
    norm=qnorm(gq);scorelo,scorehi=Q(rate)^2*flo,Q(rate)^2*fhi
    check(scorelo<=norm,"known-null finite score lower violates proved global norm cap")
    t=inv(Q(rate));dual=(2qdot(gq,y)-qnorm(y)>=0 ? y : zeros(Q,I));fd=n+t*dual
    transfer=t*sum(mass)<=1 && support(fd,rows)<=qdot(fd,q)
    tlo,thi,localidentity=nothing,nothing,false
    if transfer
        candidate=q+t*z
        check(all(candidate.>=0),"transferred finite primal negative")
        lr=max(Q(0),2qdot(fd,p-q)-qnorm(fd))
        ur=min(qnorm(p-candidate),qnorm(p-q))
        tlo,thi=max(Q(0),lr-floor),ur-floor
        check(tlo<=thi && max(tlo,flo)<=min(thi,fhi),"same-target finite transfer disagreement")
        localidentity=floor==qnorm(n)+2t*qdot(n,gq)+t^2*sloss
    end
    (;flo,fhi,scorelo,scorehi,glo,ghi,norm,maxprice,retained,
       transfer,tlo,thi,localidentity,width=Float64(ghi-glo),
       rounded_target_diff=Q.(f["target_float"])!=p)
end

function decision(lo,hi,bl,bh,a;positive=false)
    B=length(bh);o=order(a,B)
    critical=sort(bh)[o];width=maximum(bh.-bl)
    resolved=sort((bl.+bh)./2)[o]>max(0.,width)
    tails=count(>=(lo),bh)
    reject=lo>critical && tailok(tails,a,B) && (!positive||resolved)
    Dict("alpha"=>a,"critical_upper"=>critical,"critical_lower"=>sort(bl)[o],
         "tail_MC_plus_one"=>(tails+1)/(B+1),"reference_positive_resolved"=>resolved,
         "reject"=>reject,"observed_lower"=>lo,"observed_upper"=>hi)
end

"""Endpoint-order envelope, including ties, without treating numerical bounds
as independent exact scores. Noncollapsed tail counts at zero-distance atoms
need not create rejection ambiguity when the fixed critical separates them.
"""
function robustness(lo,hi,bl,bh,a;gate=true,critical=nothing)
    B=length(bh)
    cl,ch=critical===nothing ? (sort(bl)[order(a,B)],sort(bh)[order(a,B)]) : critical
    tail_min=count(>=(hi),bl)
    tail_max=count(>=(lo),bh)
    must=gate && lo>ch && tailok(tail_max,a,B)
    may=gate && hi>cl && tailok(tail_min,a,B)
    check(!must||may,"endpoint rejection envelope inverted")
    (;must,may,tail_min,tail_max,tail_collapsed=tail_min==tail_max)
end

function audit(directory=DEFAULT_DIRECTORY;save=true)
    BLAS.set_num_threads(1)
    summary=readjson(joinpath(directory,"summary.json"));protocol=readjson(joinpath(directory,"protocol.json"))
    check(equal(summary["protocol"],protocol),"summary protocol mismatch")
    B,M=Int(protocol["B_reference_per_known_anchor"]),Int(protocol["M_validation_per_known_anchor"])
    check(protocol["cells"]==CELLS && protocol["alpha_levels"]==ALPHAS && protocol["known_null_anchors"]==ANCHORS &&
          protocol["seed"]==SEED && protocol["rate"]==RATE && protocol["normal_denominator"]==64,"frozen protocol changed")
    check(protocol["c_used_or_selected"]===false && protocol["UK_pvalues_changed"]===false && summary["empirical_UK_test_or_pvalues_replaced"]===false,"oracle scope mutated")
    check(summary["all_reference_draws"]==6B && summary["all_validation_draws"]==6M && summary["omitted_or_replaced_Gaussian_draws"]==0,"draw count mismatch")
    check(equal(summary["protected_before"],summary["protected_after"]),"protected campaign mutation")
    for (relative,digest) in summary["protected_before"]
        check(filehash(joinpath(ROOT,relative))==digest,"frozen empirical source changed: $relative")
    end
    for (relative,digest) in summary["new_source_sha256"]
        check(filehash(joinpath(ROOT,relative))==digest,"campaign source changed: $relative")
    end
    countchecked,roundeddiffs,decisionchecks,summarychecks=0,0,0,0
    cellchecks=Dict{String,Any}[]
    endpointrows=Dict{String,Any}[]
    for cell in CELLS
        record=readjson(joinpath(directory,"cell_"*lpad(string(cell),2,"0")*".json"))
        check(equal(record["protocol"],protocol),"cell protocol mismatch")
        geo=geometry(record["geometry"])
        banks=record["banks"];validation=record["validation_records"]
        check([b["anchor"] for b in banks]==ANCHORS && length(validation)==3M,"anchor/validation count mismatch")
        for (ai,bank) in enumerate(banks)
            anchor=bank["anchor"];q=parseq.(bank["q_exact"]);n=parseq.(bank["n_exact"])
            expectedq=anchor=="centroid" ? geo.centroid : geo.q0
            expectedn=zeros(Q,length(q))
            if anchor=="raw_outside_simplex"
                for group in geo.groups
                    k,Z=length(group),count(i->q[i]==0,group)
                    for i in group;expectedn[i]=Q(Z-k*Int(q[i]==0),64);end
                end
                check(any(q+n .<0),"declared raw anchor is not outside simplex")
            end
            check(q==expectedq && n==expectedn && parseq.(bank["mean_exact"])==q+n,"known anchor mismatch")
            check(all(sum(q[g])==1 && sum(n[g])==0 for g in geo.groups),"known anchor adding-up failed")
            rs=SEED+1_000_003cell+100_003ai;vs=rs+10_000_019
            check(bank["reference_seed"]==rs && bank["validation_seed"]==vs,"Gaussian seed mismatch")
            Gref=geo.L*randn(MersenneTwister(rs),length(q),B)
            Gval=geo.L*randn(MersenneTwister(vs),length(q),M)
            check(noisehash(Gref)==bank["reference_noise_sha256"] && noisehash(Gval)==bank["validation_noise_sha256"],"full Gaussian bank digest mismatch")
            check(length(bank["reference_primitive_certificates"])==B,"reference primitive count mismatch")
            maxwidth,maxpolar,transfers,identities=0.,Q(0),0,0
            for b in 1:B
                r=replay(bank["reference_primitive_certificates"][b],Gref[:,b],q,n,geo)
                for (key,value) in (("cone_lower",down(r.glo)),("cone_upper",up(r.ghi)),
                     ("finite_lower",down(r.scorelo)),("finite_upper",up(r.scorehi)),
                     ("norm_lower",down(r.norm)),("norm_upper",up(r.norm)))
                    check(bank[key][b]==value,"reference rounded endpoint mismatch: $key")
                end
                check(length(r.retained)==bank["critical_RUM_generator_count"],"critical slice count mismatch")
                transfers+=r.transfer;identities+=r.localidentity
                maxwidth=max(maxwidth,r.width);maxpolar=max(maxpolar,r.maxprice)
                countchecked+=1;roundeddiffs+=r.rounded_target_diff
            end
            check(maxwidth==bank["maximum_cone_interval_width"] && maxpolar==parseq(bank["maximum_exact_polar_price"]),"reference cone diagnostics mismatch")
            check(transfers==bank["reference_finite_transfer_certificates"] && identities==bank["reference_simplex_local_identities"],"reference local transfer aggregation mismatch")
            vals=filter(v->v["anchor"]==anchor,validation)
            check(length(vals)==M && [v["replication"] for v in vals]==collect(1:M),"validation replication mismatch")
            envelopes=Dict((a,method)=>Dict("must"=>0,"may"=>0,"tail_collapsed"=>0)
                           for a in ALPHAS for method in METHODS)
            for (m,v) in enumerate(vals)
                r=replay(v["primitive_certificate"],Gval[:,m],q,n,geo)
                for (key,value) in (("cone_lower",down(r.glo)),("cone_upper",up(r.ghi)),
                     ("finite_lower",down(r.scorelo)),("finite_upper",up(r.scorehi)),
                     ("norm_squared_lower",down(r.norm)),("norm_squared_upper",up(r.norm)))
                    check(v[key]==value,"validation rounded endpoint mismatch: $key")
                end
                check(v["finite_transfer_available"]==r.transfer && v["simplex_local_identity"]==r.localidentity && v["cone_interval_width"]==r.width,"validation transfer flag mismatch")
                check(v["finite_transfer_exact_lower"]==(r.transfer ? string(Q(RATE)^2*r.tlo) : nothing) &&
                      v["finite_transfer_exact_upper"]==(r.transfer ? string(Q(RATE)^2*r.thi) : nothing),"validation transfer endpoint mismatch")
                for a in ALPHAS
                    dec=only(filter(d->d["alpha"]==a,v["decisions"]))
                    expected=[decision(down(r.glo),up(r.ghi),bank["cone_lower"],bank["cone_upper"],a;positive=true),
                              decision(down(r.scorelo),up(r.scorehi),bank["cone_lower"],bank["cone_upper"],a;positive=true),
                              decision(down(r.scorelo),up(r.scorehi),bank["norm_lower"],bank["norm_upper"],a)]
                    for (method,e) in zip(METHODS[1:3],expected)
                        check(equal(dec[method],e),"rank/tail decision mismatch: $method")
                        decisionchecks+=1
                    end
                    bounds=[robustness(down(r.glo),up(r.ghi),bank["cone_lower"],bank["cone_upper"],a;
                                       gate=expected[1]["reference_positive_resolved"]),
                            robustness(down(r.scorelo),up(r.scorehi),bank["cone_lower"],bank["cone_upper"],a;
                                       gate=expected[2]["reference_positive_resolved"]),
                            robustness(down(r.scorelo),up(r.scorehi),bank["norm_lower"],bank["norm_upper"],a)]
                    critical=maximum(sort(b["finite_upper"])[order(a,B)] for b in banks)
                    tails=count(>=(down(r.scorelo)),bank["finite_upper"])
                    menu=Dict("alpha"=>a,"critical_upper"=>critical,"own_anchor_tail_MC_plus_one"=>(tails+1)/(B+1),
                              "reject"=>down(r.scorelo)>critical && tailok(tails,a,B))
                    check(equal(dec["finite_known_null_menu_oracle"],menu),"maximum known-anchor rank decision mismatch")
                    decisionchecks+=1
                    critical_lower=maximum(sort(b["finite_lower"])[order(a,B)] for b in banks)
                    push!(bounds,robustness(down(r.scorelo),up(r.scorehi),bank["finite_lower"],bank["finite_upper"],a;
                                           critical=(critical_lower,critical)))
                    for (method,bound) in zip(METHODS,bounds)
                        check(!bound.must||dec[method]["reject"],"current decision violates guaranteed rejection")
                        check(bound.may||!dec[method]["reject"],"current decision violates guaranteed nonrejection")
                        counts=envelopes[(a,method)]
                        counts["must"]+=bound.must;counts["may"]+=bound.may
                        counts["tail_collapsed"]+=bound.tail_collapsed
                    end
                end
                countchecked+=1;roundeddiffs+=r.rounded_target_diff
            end
            for a in ALPHAS,method in METHODS
                selected=only(filter(row->row["cell"]==cell && row["anchor"]==anchor && row["alpha"]==a && row["method"]==method,summary["rows"]))
                k=count(v->only(filter(d->d["alpha"]==a,v["decisions"]))[method]["reject"],vals)
                ci=[k==0 ? 0. : quantile(Beta(k,M-k+1),.025),k==M ? 1. : quantile(Beta(k+1,M-k),.975)]
                check(selected["rejections"]==k && selected["validation_repetitions"]==M && selected["rejection_rate"]==k/M,"rejection aggregation mismatch")
                check(maximum(abs.(selected["CP95_conditional_on_reference_bank"].-ci))<=1e-14,"conditional Clopper-Pearson interval mismatch")
                check(selected["finite_transfer_count"]==count(v->v["finite_transfer_available"],vals) &&
                      selected["simplex_local_identity_count"]==count(v->v["simplex_local_identity"],vals),"local transfer summary mismatch")
                envelope=envelopes[(a,method)]
                check(envelope["must"]<=k<=envelope["may"],"observed rejection count outside exact-score envelope")
                klo,khi=envelope["must"],envelope["may"]
                cp_envelope=[klo==0 ? 0. : quantile(Beta(klo,M-klo+1),.025),
                             khi==M ? 1. : quantile(Beta(khi+1,M-khi),.975)]
                push!(endpointrows,Dict("cell"=>cell,"anchor"=>anchor,"alpha"=>a,"method"=>method,
                    "exact_score_rejection_count_lower"=>klo,"exact_score_rejection_count_upper"=>khi,
                    "rejection_ambiguities"=>khi-klo,"tail_count_envelopes_collapsed"=>envelope["tail_collapsed"],
                    "tail_count_envelopes_not_collapsed"=>M-envelope["tail_collapsed"],
                    "conditional_CP95_exact_score_envelope"=>cp_envelope))
                summarychecks+=1
            end
        end
        push!(cellchecks,Dict("cell"=>cell,"RUM_columns"=>size(geo.A,2),"all_coordinates"=>size(geo.A,1),
                             "reference_certificates"=>3B,"validation_certificates"=>3M))
        println("KNOWN_FACE_INDEPENDENT_AUDIT_CELL ",cell," certificates=",3*(B+M));flush(stdout)
    end
    check(countchecked==6*(B+M) && decisionchecks==6M*length(ALPHAS)*length(METHODS) && summarychecks==48,"final audit counts mismatch")
    ambiguities=sum(r["rejection_ambiguities"] for r in endpointrows)
    result=Dict("all_checks_passed"=>true,"primitive_certificates_checked"=>countchecked,
        "symbolic_targets_distinct_from_rounded_proposal_target"=>roundeddiffs,
        "rank_decisions_checked"=>decisionchecks,"rejection_and_conditional_CP_rows_checked"=>summarychecks,
        "cell_checks"=>cellchecks,"auditor_source_sha256"=>filehash(@__FILE__),
        "campaign_summary_sha256"=>filehash(joinpath(directory,"summary.json")),
        "endpoint_order_rejection_ambiguities"=>ambiguities,
        "all_reported_rejection_counts_equal_iid_exact_score_counts"=>ambiguities==0,
        "endpoint_robustness_rows"=>endpointrows,
        "imports_campaign_driver_or_projection_helpers"=>false,
        "exact_target_reconstruction"=>"q+n+Q(G)/Q(rate), never engine target_float",
        "scope"=>"Exact deterministic arithmetic and seeded finite Gaussian oracle replay only; conditional CP describes the realized reference bank; no feasible H/GMS, UK nuisance, joint-window or unknown-null uniform inference theorem")
    if save
        check(ambiguities==0,"rejection endpoint ambiguities remain; report exact-score rate/CP envelopes before saving a pass audit")
        destination=joinpath(directory,"audit.json")
        check(!ispath(destination),"refusing to overwrite an existing independent audit")
        open(destination,"w") do io;JSON3.write(io,result);end
    end
    println("KNOWN_FACE_INDEPENDENT_AUDIT_PASS ",countchecked)
    result
end

if abspath(PROGRAM_FILE)==@__FILE__
    directory=isempty(ARGS) ? DEFAULT_DIRECTORY : abspath(first(ARGS))
    audit(directory;save=!("--no-save" in ARGS))
end

end # module
