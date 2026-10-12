# Actual-empty-CMU INNER-family computations, conditional on an external
# microfoundation registry: complete types, certified one-budget responders,
# and a certified all-null type. No arbitrary budget-deletion closure is used.
using JuMP, HiGHS, LinearAlgebra, SparseArrays
const CEIP_SOLVER_ENVIRONMENT=joinpath(@__DIR__,"selection_solver_environment")
isfile(joinpath(CEIP_SOLVER_ENVIRONMENT,"Manifest.toml")) ||
    error("instantiate the isolated selection_solver_environment before running this extension")
CEIP_SOLVER_ENVIRONMENT in LOAD_PATH || push!(LOAD_PATH,CEIP_SOLVER_ENVIRONMENT)
using Clarabel

const CEIP_Q = Rational{BigInt}
const CEIP_CAPS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]

function ceip_float_lower(q::CEIP_Q)
    f = Float64(q)
    CEIP_Q(f) > q ? prevfloat(f) : f
end
function ceip_float_upper(q::CEIP_Q)
    f = Float64(q)
    CEIP_Q(f) < q ? nextfloat(f) : f
end

function ceip_build_family(complete, groups; label="CMU_empty_inner")
    I,H = size(complete)
    I>0 && H>0 && length(groups)>1 || error("nonempty complete face and at least two budgets required")
    all(x->x==0 || x==1,complete) || error("binary complete incidence required")
    flat=reduce(vcat,Int.(g) for g in groups)
    sort(flat)==collect(1:I) && all(!isempty,groups) || error("budget groups must partition patch rows")
    J=length(groups)
    Ac=sparse(Int8.(complete))
    rows=zeros(Int,J,H)
    for h in 1:H,j in 1:J
        selected=filter(i->Ac[i,h]!=0,groups[j])
        length(selected)==1 || error("complete type must select exactly one patch per budget")
        rows[j,h]=only(selected)
    end
    owner=zeros(Int,I)
    for (j,g) in enumerate(groups)
        owner[g].=j
    end
    A=hcat(Ac,sparse(1:I,1:I,ones(Int8,I),I,I),spzeros(Int8,I,1))
    D=hcat(sparse(ones(Int8,J,H)),sparse(owner,1:I,ones(Int8,I),J,I),spzeros(Int8,J,1))
    (;A,D,complete=Ac,complete_rows=rows,budget_groups=[Int.(g) for g in groups],owner,
      n_patches=I,n_budgets=J,n_complete=H,n_types=H+I+1,
      complete_indices=collect(1:H),singleton_indices=collect(H+1:H+I),null_index=H+I+1,
      label,external_microfoundation_registry_required=true)
end

function ceip_simplex_floor(raw,groups)
    all(isfinite,raw) || error("finite raw affine target required")
    p=CEIP_Q.(raw); q=zeros(CEIP_Q,length(p))
    for g in groups
        sorted=sort(p[g];rev=true); running=CEIP_Q(0); theta=CEIP_Q(0)
        for k in eachindex(sorted)
            running+=sorted[k]
            candidate=(running-1)/k
            sorted[k]>candidate && (theta=candidate)
        end
        q[g]=max.(p[g].-theta,CEIP_Q(0))
        sum(q[g])==1 || error("exact simplex floor failed")
    end
    floor=sum(abs2,p-q)
    (;q,q_float=Float64.(q),floor,floor_float=Float64(floor),
      maximum_adjustment=Float64(maximum(abs,p-q)))
end

function ceip_dyadic_probabilities(values; bits=48)
    all(isfinite,values) || error("nonfinite simplex proposal")
    integers=[max(big(0),round(BigInt,ldexp(max(0.0,Float64(v)),bits))) for v in values]
    total=sum(integers)
    total>0 || error("no positive simplex proposal mass")
    [m//total for m in integers]
end

function ceip_complete_marginal(f,v)
    q=zeros(CEIP_Q,f.n_patches)
    for h in 1:f.n_complete,j in 1:f.n_budgets
        v[h]!=0 && (q[f.complete_rows[j,h]]+=v[h])
    end
    q
end

function ceip_complete_support(f,y)
    best=sum(y[f.complete_rows[j,1]] for j in 1:f.n_budgets)
    for h in 2:f.n_complete
        score=sum(y[f.complete_rows[j,h]] for j in 1:f.n_budgets)
        best=max(best,score)
    end
    best
end

function ceip_population_support(f,y)
    max(CEIP_Q(0),ceip_complete_support(f,y),maximum(y))
end

function ceip_sparse_mass(mass)
    indices=findall(!iszero,mass)
    (;indices,mass=mass[indices])
end

function ceip_check_primal(f,indices,mass,rates,q)
    all(>=(0),mass) && sum(mass)==1 || error("exact population mixture invalid")
    ordinary=CEIP_Q.(f.A[:,indices])*mass
    response=CEIP_Q.(f.D[:,indices])*mass
    response==rates && ordinary==q.*rates[f.owner] || error("exact response/marginal certificate failed")
    true
end

mutable struct CMUEmptyInnerProjector
    family::Any
    model::Model
    complete_weights::Vector{VariableRef}
    singleton_conditionals::Vector{VariableRef}
    complete_marginals::Vector{VariableRef}
    residual::Vector{VariableRef}
    mix_equalities::Vector{ConstraintRef}
    tolerance::Float64
    previous::Any
end

function CMUEmptyInnerProjector(f; tolerance=1e-8)
    model=Model(Clarabel.Optimizer); set_silent(model)
    for (name,value) in (("tol_gap_abs",1e-12),("tol_gap_rel",1e-12),("tol_feas",1e-12),
        ("max_iter",400),("iterative_refinement_abstol",1e-14),("iterative_refinement_reltol",1e-14))
        set_optimizer_attribute(model,name,value)
    end
    I,H=f.n_patches,f.n_complete
    @variable(model,v[1:H]>=0)
    @variable(model,u[1:I]>=0)
    @variable(model,qc[1:I])
    @variable(model,z[1:I])
    @constraint(model,f.complete*v-qc==zeros(I))
    @constraint(model,sum(v)==1)
    for group in f.budget_groups
        @constraint(model,sum(u[i] for i in group)==1)
    end
    mix=@constraint(model,[i=1:I],qc[i]-z[i]==0)
    @objective(model,Min,0.5sum(z.^2))
    CMUEmptyInnerProjector(f,model,collect(v),collect(u),collect(qc),collect(z),
        collect(mix),Float64(tolerance),nothing)
end

function ceip_point_certificate(f,raw,tau,beta,q,indices,mass)
    p=CEIP_Q.(raw)
    ceip_check_primal(f,indices,mass,fill(tau,f.n_budgets),q)
    lambda=p-q
    complete_support=ceip_complete_support(f,lambda)
    product_support=sum(maximum(lambda[g]) for g in f.budget_groups)
    support=(1-beta)*complete_support+beta*product_support
    floor=ceip_simplex_floor(raw,f.budget_groups).floor
    upper=sum(abs2,lambda)
    lower=max(floor,2dot(p,lambda)-sum(abs2,lambda)-2support)
    lower<=upper || error("exact conditional projection bounds inverted")
    (;q,indices,mass,lambda,complete_support,product_support,support,floor,lower,upper,tau,beta)
end

"""Common-response INNER experiment; common rates are an additional restriction.

The exact conditional set is (1-beta)*conv(complete)+beta*product(simplexes),
beta=min(1,(1-tau)/((J-1)*tau)). This equivalence uses the actual singleton/null
population family, not arbitrary deletion of a complete consumer's choices.
All objective bounds below are exact rational certificates for the fixed
binary-rational raw target and cap, with outward-rounded Float64 diagnostics.
"""
function ceip_fit!(projector::CMUEmptyInnerProjector,raw,epsilon)
    f=projector.family
    length(raw)==f.n_patches && all(isfinite,raw) && 0<=epsilon<1 || error("invalid target/cap")
    tau=1-CEIP_Q(Float64(epsilon))
    beta=min(CEIP_Q(1),(1-tau)/((f.n_budgets-1)*tau))
    for i in 1:f.n_patches
        set_normalized_coefficient(projector.mix_equalities[i],projector.complete_marginals[i],Float64(1-beta))
        set_normalized_coefficient(projector.mix_equalities[i],projector.singleton_conditionals[i],Float64(beta))
        set_normalized_rhs(projector.mix_equalities[i],Float64(raw[i]))
    end
    optimize!(projector.model)
    status=termination_status(projector.model)
    status in (MOI.OPTIMAL,MOI.ALMOST_OPTIMAL,MOI.ITERATION_LIMIT) && result_count(projector.model)>0 ||
        error("inner QP has no certified proposal: $status")
    v=ceip_dyadic_probabilities(value.(projector.complete_weights))
    u=zeros(CEIP_Q,f.n_patches)
    if beta==1
        u=ceip_simplex_floor(raw,f.budget_groups).q
    else
        numerical=value.(projector.singleton_conditionals)
        for group in f.budget_groups
            u[group]=ceip_dyadic_probabilities(numerical[group])
        end
    end
    qc=ceip_complete_marginal(f,v)
    q=(1-beta)*qc+beta*u
    c=tau*(1-beta)
    population=zeros(CEIP_Q,f.n_types)
    population[f.complete_indices]=c*v
    population[f.singleton_indices]=tau*beta*u
    population[f.null_index]=1-f.n_budgets*tau+(f.n_budgets-1)*c
    witness=ceip_sparse_mass(population)
    cert=ceip_point_certificate(f,raw,tau,beta,q,witness.indices,witness.mass)
    proposal_upper=cert.upper
    inherited=false
    prior=projector.previous
    if prior!==nothing && prior.raw==raw && prior.epsilon<=epsilon && prior.cert.upper<cert.upper
        # Global mixture with the certified ALL-NULL CMU, not independent deletion.
        scaled=zeros(CEIP_Q,f.n_types)
        fraction=tau/prior.cert.tau
        scaled[prior.cert.indices]=fraction*prior.cert.mass
        scaled[f.null_index]+=1-fraction
        witness=ceip_sparse_mass(scaled)
        cert=ceip_point_certificate(f,raw,tau,beta,prior.cert.q,witness.indices,witness.mass)
        inherited=true
    end
    gap=cert.upper-cert.lower
    gap<=CEIP_Q(projector.tolerance) || error("exact inner QP objective gap unresolved: $(Float64(gap)); cap=$epsilon")
    record=Dict("epsilon"=>Float64(epsilon),"response_rate"=>Float64(tau),
        "unary_conditional_mix_fraction"=>Float64(beta),"common_response_is_additional_restriction"=>true,
        "squared_distance_lower"=>ceip_float_lower(cert.lower),"squared_distance_upper"=>ceip_float_upper(cert.upper),
        "objective_gap"=>ceip_float_upper(gap),"conditional_fitted"=>Float64.(cert.q),
        "maximum_response_moment_error"=>0.0,"simplex_mass_sum"=>1.0,
        "complete_face_used"=>epsilon==0,"solver_status"=>string(status),"QP_backend"=>"clarabel",
        "full_type_count"=>f.n_types,"all_columns_priced"=>true,"outward_rounded_certificate"=>true,
        "point_bounds_exact_rational"=>true,"new_p_values"=>nothing,
        "upper_inherited_by_global_all_null_mixture"=>inherited,
        "inherited_fit_cap"=>inherited ? prior.epsilon : nothing,
        "solver_proposal_upper_exact"=>string(proposal_upper),
        "certificate"=>Dict("primal_indices"=>cert.indices,"primal_masses_exact"=>string.(cert.mass),
            "response_rate_exact"=>string(tau),"unary_mix_fraction_exact"=>string(beta),
            "conditional_fitted_exact"=>string.(cert.q),"residual_patch_weights_exact"=>string.(cert.lambda),
            "complete_support_exact"=>string(cert.complete_support),"product_support_exact"=>string(cert.product_support),
            "conditional_support_exact"=>string(cert.support),"domain_floor_exact"=>string(cert.floor),
            "squared_distance_lower_exact"=>string(cert.lower),"squared_distance_upper_exact"=>string(cert.upper)))
    projector.previous=(;raw=copy(raw),epsilon=Float64(epsilon),cert)
    record
end

function ceip_complete_fit(A,raw,groups; tolerance=1e-8)
    ceip_fit!(CMUEmptyInnerProjector(ceip_build_family(A,groups;label="fixed_complete_RUM");tolerance),raw,0.0)
end

function ceip_exact_solve(C,d,proposal)
    m,n=size(C); aug=hcat(CEIP_Q.(C),CEIP_Q.(d)); pivots=Int[]; row=1
    for column in 1:n
        k=findfirst(i->aug[i,column]!=0,row:m)
        k===nothing && continue
        pivot=row+k-1
        aug[row,:],aug[pivot,:]=copy(aug[pivot,:]),copy(aug[row,:])
        aug[row,:]./=aug[row,column]
        for other in 1:m
            other==row && continue
            coefficient=aug[other,column]
            coefficient!=0 && (aug[other,:].-=coefficient.*aug[row,:])
        end
        push!(pivots,column); row+=1
        row>m && break
    end
    any(i->all(iszero,aug[i,1:n])&&aug[i,n+1]!=0,row:m) && return nothing
    x=CEIP_Q.(proposal); free=setdiff(1:n,pivots)
    for (r,c) in enumerate(pivots)
        x[c]=aug[r,n+1]-sum(aug[r,k]*x[k] for k in free;init=CEIP_Q(0))
    end
    C*x==d ? x : nothing
end

function ceip_constructive_floor_primal(f,q,numerical_complete,policy)
    candidates=Any[]
    for cutoff in (1e-10,1e-12,0.0)
        values=[v>cutoff ? Float64(v) : 0.0 for v in numerical_complete]
        sum(values)>0 || continue
        v=ceip_dyadic_probabilities(values); qc=ceip_complete_marginal(f,v)
        any(i->q[i]==0&&qc[i]>0,eachindex(q)) && continue
        if policy==:common
            z=min(CEIP_Q(1),minimum(q[i]/qc[i] for i in eachindex(q) if qc[i]>0))
            tau=inv(f.n_budgets-(f.n_budgets-1)*z)
            rates=fill(tau,f.n_budgets); c=z*tau
            masses=zeros(CEIP_Q,f.n_types)
            masses[f.complete_indices]=c*v
            masses[f.singleton_indices]=q.*rates[f.owner]-c*qc
            witness=ceip_sparse_mass(masses)
            ceip_check_primal(f,witness.indices,witness.mass,rates,q)
            push!(candidates,(;witness...,rates,tau,method="complete_singleton_constructive",cutoff))
        else
            M=[maximum(qc[i]==0 ? CEIP_Q(0) : qc[i]/q[i] for i in g) for g in f.budget_groups]
            for z in unique(vcat(CEIP_Q[0,1],[inv(v) for v in M if v>0]))
                0<=z<=1 || continue
                a=max.(CEIP_Q(1),z*M); base=inv(sum(a)-(f.n_budgets-1)*z)
                rates=base*a; c=base*z; tau=minimum(rates)
                masses=zeros(CEIP_Q,f.n_types)
                masses[f.complete_indices]=c*v
                masses[f.singleton_indices]=q.*rates[f.owner]-c*qc
                witness=ceip_sparse_mass(masses)
                ceip_check_primal(f,witness.indices,witness.mass,rates,q)
                push!(candidates,(;witness...,rates,tau,method="complete_singleton_constructive",cutoff))
            end
        end
    end
    # One-budget cohorts always provide a valid response 1/J witness.
    rates=fill(CEIP_Q(1)/f.n_budgets,f.n_budgets)
    masses=zeros(CEIP_Q,f.n_types); masses[f.singleton_indices]=q/f.n_budgets
    witness=ceip_sparse_mass(masses)
    push!(candidates,(;witness...,rates,tau=first(rates),method="one_budget_universal",cutoff=0.0))
    candidates[argmax([c.tau for c in candidates])]
end

"""Exact certificate bracket for minimum common maximum absence allowance.

:common imposes equal rates (a restricted inner experiment). :variable allows
different rates and maximizes their minimum; it is the broader inner floor LP.
The LP uses the declared exact simplex diagnostic, never a substituted QP target.
"""
function ceip_floor_breakdown(f,q; response_policy=:common,certificate_tolerance=1e-7)
    response_policy in (:common,:variable) || error("unknown response policy")
    q=CEIP_Q.(q)
    length(q)==f.n_patches && all(>=(0),q) && all(sum(q[g])==1 for g in f.budget_groups) || error("exact normalized LP target required")
    allowed=[h for h in 1:f.n_types if all(i->f.A[i,h]==0||q[i]>0,1:f.n_patches)]
    A=f.A[:,allowed]; I,J=f.n_patches,f.n_budgets
    model=Model(HiGHS.Optimizer); set_silent(model)
    for (name,value) in (("primal_feasibility_tolerance",1e-9),("dual_feasibility_tolerance",1e-9),("threads",1))
        set_optimizer_attribute(model,name,value)
    end
    @variable(model,nu[1:length(allowed)]>=0)
    @variable(model,tau>=0)
    if response_policy==:common
        rates=fill(tau,J)
    else
        @variable(model,s[1:J]>=0)
        @constraint(model,[j=1:J],s[j]>=tau)
        rates=collect(s)
    end
    totals=A*nu
    eq=@constraint(model,[i=1:I],totals[i]==Float64(q[i])*rates[f.owner[i]])
    @constraint(model,sum(nu)==1)
    @objective(model,Max,tau)
    optimize!(model)
    termination_status(model)==MOI.OPTIMAL || error("inner floor LP failed: $(termination_status(model))")
    numerical=value.(nu); numerical_tau=value(tau); numerical_rates=value.(rates)
    proposal=nothing
    for threshold in (1e-10,1e-12,0.0)
        active=findall(>(threshold),numerical)
        isempty(active) && continue
        K=length(active)
        if response_policy==:common
            C=zeros(CEIP_Q,I+1,K+1); C[1:I,1:K]=CEIP_Q.(Matrix(A[:,active])); C[1:I,end]=-q; C[end,1:K].=1
            d=vcat(zeros(CEIP_Q,I),CEIP_Q[1])
            x=ceip_exact_solve(C,d,vcat(numerical[active],numerical_tau))
            x===nothing && continue
            mass=x[1:K]; certified_rates=fill(x[end],J); certified_tau=x[end]
        else
            tight=findall(j->abs(numerical_rates[j]-numerical_tau)<=1e-8,1:J)
            C=zeros(CEIP_Q,I+1+length(tight),K+J+1)
            C[1:I,1:K]=CEIP_Q.(Matrix(A[:,active]))
            for i in 1:I
                C[i,K+f.owner[i]]=-q[i]
            end
            C[I+1,1:K].=1
            for (r,j) in enumerate(tight)
                C[I+1+r,K+j]=1; C[I+1+r,end]=-1
            end
            d=vcat(zeros(CEIP_Q,I),CEIP_Q[1],zeros(CEIP_Q,length(tight)))
            x=ceip_exact_solve(C,d,vcat(numerical[active],numerical_rates,numerical_tau))
            x===nothing && continue
            mass=x[1:K]; certified_rates=x[K+1:K+J]; certified_tau=x[end]
        end
        all(>=(0),mass) && sum(mass)==1 && 0<certified_tau<=minimum(certified_rates) && maximum(certified_rates)<=1 || continue
        indices=allowed[active]
        ceip_check_primal(f,indices,mass,certified_rates,q)
        proposal=(;indices,mass,rates=certified_rates,tau=minimum(certified_rates),method="exact_active_basis",cutoff=threshold)
        break
    end
    full_numerical=zeros(f.n_types); full_numerical[allowed]=numerical
    constructive=ceip_constructive_floor_primal(f,q,full_numerical[f.complete_indices],response_policy)
    if proposal===nothing || constructive.tau>proposal.tau
        proposal=constructive
    end
    y=CEIP_Q.(dual.(eq))
    if response_policy==:common
        normalization=dot(q,y)
        normalization!=0 || error("floor dual normalization absent")
        y/=normalization
    else
        for group in f.budget_groups
            t=dot(q[group],y[group])
            t<0 && (y[group].-=t)
        end
        total=dot(q,y)
        total>0 || error("variable-rate dual normalization absent")
        y/=total
        all(dot(q[g],y[g])>=0 for g in f.budget_groups) || error("negative variable dual budget coefficient")
    end
    penalty=sum(max(v,CEIP_Q(0)) for v in y)+1
    for i in eachindex(q)
        q[i]==0 && (y[i]=-penalty)
    end
    dot(q,y)==1 || error("exact floor dual normalization failed")
    gamma=ceip_population_support(f,y)
    upper=min(CEIP_Q(1),gamma); lower=proposal.tau
    lower<=upper || error("exact floor LP bounds inverted")
    gap=upper-lower
    gap<=CEIP_Q(certificate_tolerance) || error("exact inner floor LP gap unresolved: $(Float64(gap)); policy=$response_policy")
    Dict("response_policy"=>String(response_policy),"response_lower"=>ceip_float_lower(lower),
        "response_upper"=>ceip_float_upper(upper),"response_lower_exact"=>string(lower),"response_upper_exact"=>string(upper),
        "epsilon_lower"=>ceip_float_lower(1-upper),"epsilon_upper"=>ceip_float_upper(1-lower),
        "epsilon_lower_exact"=>string(1-upper),"epsilon_upper_exact"=>string(1-lower),
        "exact_interval_width"=>string(gap),"exact_interval_width_float"=>ceip_float_upper(gap),
        "primal_certificate_method"=>proposal.method,"primal_certificate_columns"=>proposal.indices,
        "primal_certificate_mass_exact"=>string.(proposal.mass),"primal_response_rates_exact"=>string.(proposal.rates),
        "dual_certificate_patch_weights_exact"=>string.(y),"dual_support_bound_exact"=>string(gamma),
        "dual_budget_coefficients_exact"=>[string(dot(q[g],y[g])) for g in f.budget_groups],
        "all_inner_columns_priced_exactly"=>f.n_types,"exact_rational_certificate"=>true,
        "common_response_is_additional_restriction"=>response_policy==:common,
        "population_absence_share_identified"=>false,"new_p_values"=>nothing)
end
