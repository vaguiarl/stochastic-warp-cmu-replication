using KSReplication, JuMP, OSQP, LinearAlgebra

"""Worker-local projection onto the COMPLETE normalized type hull.

The optimization working set never changes the model: every accepted proposal
is priced against all original columns. The dual includes an intercept, so
its recovered type multipliers sum to one (unlike a cone). Returned feasible
simplex weights and a full-support dual bound bracket the squared Euclidean
objective. These Float64 checks are NOT outward-rounded interval certificates.
No clipping, covariance inverse, probability repair, or residual floor is used.
"""
mutable struct NormalizedHullProjector
    A::Matrix{Float64}
    model::Model
    lambda::Vector{VariableRef}
    intercept::VariableRef
    indices::Vector{Int}
    constraints::Vector{ConstraintRef}
    included::BitVector
    batch_size::Int
    maximum_iterations::Int
    maximum_working_cuts::Int
    pricing_tolerance::Float64
    certificate_tolerance::Float64
    allow_full_fallback::Bool
    fallback::Any
end

function NormalizedHullProjector(A::AbstractMatrix; initial_indices=nothing,
        batch_size::Integer=32, maximum_iterations::Integer=200,
        maximum_working_cuts::Integer=5000, pricing_tolerance::Real=1e-9,
        certificate_tolerance::Real=1e-8, allow_full_fallback::Bool=true)
    I,H=size(A)
    I>0 && H>0 || throw(ArgumentError("nonempty complete type matrix required"))
    all(x->x==0 || x==1,A) || throw(ArgumentError("binary complete type columns required"))
    sums=vec(sum(A;dims=1))
    length(unique(sums))==1 && first(sums)>0 ||
        throw(ArgumentError("equal positive type column sums required"))
    all(x->x>0,(batch_size,maximum_iterations,maximum_working_cuts)) ||
        throw(ArgumentError("positive finite working limits required"))
    all(x->isfinite(x) && x>0,(pricing_tolerance,certificate_tolerance)) ||
        throw(ArgumentError("finite positive numerical tolerances required"))
    Af=Matrix{Float64}(A)
    model=Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model,"eps_abs",1e-10)
    set_optimizer_attribute(model,"eps_rel",1e-10)
    set_optimizer_attribute(model,"max_iter",200000)
    set_optimizer_attribute(model,"polish",true)
    @variable(model,lambda[1:I])
    @variable(model,intercept)
    @objective(model,Min,.5sum(lambda[i]^2 for i in 1:I)+intercept)
    p=NormalizedHullProjector(Af,model,collect(lambda),intercept,Int[],
        ConstraintRef[],falses(H),Int(batch_size),Int(maximum_iterations),
        min(H,Int(maximum_working_cuts)),Float64(pricing_tolerance),
        Float64(certificate_tolerance),allow_full_fallback,nothing)
    seeds=initial_indices===nothing ? collect(1:min(Int(batch_size),H)) : collect(initial_indices)
    !isempty(seeds) && all(h->h isa Integer && 1<=h<=H,seeds) ||
        throw(ArgumentError("nonempty valid initial full-column indices required"))
    length(unique(seeds))<=p.maximum_working_cuts ||
        throw(ArgumentError("initial working set exceeds declared cut limit"))
    _nh_add_cuts!(p,Int.(seeds))
    return p
end

function _nh_add_cuts!(p,indices)
    for h in indices
        p.included[h] && continue
        length(p.indices)<p.maximum_working_cuts || error("working-set limit reached")
        c=@constraint(p.model,sum(p.A[i,h]*p.lambda[i] for i in axes(p.A,1))<=p.intercept)
        push!(p.indices,h);push!(p.constraints,c);p.included[h]=true
    end
    nothing
end

function _nh_bounds(A,target,indices,rawmass,lambda,intercept)
    all(isfinite,rawmass) && all(isfinite,lambda) && isfinite(intercept) ||
        error("nonfinite normalized-hull solver proposal")
    weights=max.(0.,rawmass)
    sum(weights)>0 || error("normalized-hull proposal has no positive mass")
    weights ./= sum(weights)
    mass=zeros(size(A,2));mass[indices]=weights
    fitted=A[:,indices]*weights
    residual=target-fitted
    upper=sum(abs2,residual)
    residual_scores=A'*residual
    support=maximum(residual_scores)
    lower=max(0.,2dot(target,residual)-upper-2support)
    lower<=upper+1e-10 || error("normalized-hull objective bracket inverted")
    # Exhaustive pricing checks both the QP cut violation and the independent
    # globally feasible dual used in the accepted objective bracket.
    scores=A'*lambda
    violation=maximum(scores)-intercept
    all(isfinite,(upper,lower,violation)) || error("nonfinite full hull pricing")
    return (;fitted,upper,lower=min(lower,upper),mass,mass_sum=sum(weights),
        raw_mass_sum=sum(rawmass),mass_correction_l1=sum(abs,weights-rawmass),
        stationarity_error=maximum(abs,lambda-residual),scores,violation,
        full_support_maximum=support,support_variational_gap=support-dot(residual,fitted))
end

function _nh_fallback!(p,z,reason,iterations,passes)
    p.allow_full_fallback || error("complete normalized-hull working solver failed: $reason")
    if p.fallback===nothing
        p.fallback=NormalizedHullProjector(p.A;initial_indices=collect(axes(p.A,2)),
            maximum_working_cuts=size(p.A,2),maximum_iterations=1,
            pricing_tolerance=p.pricing_tolerance,certificate_tolerance=p.certificate_tolerance,
            allow_full_fallback=false)
    end
    fit=nh_fit!(p.fallback,z)
    d=fit.diagnostic
    info=merge(d,(fallback_used=true,fallback_reason=String(reason),
        iterations=iterations+d.iterations,full_pricing_passes=passes+d.full_pricing_passes,
        priced_columns=(passes+d.full_pricing_passes)*size(p.A,2)))
    return merge(fit,(diagnostic=info,))
end

"""Complete normalized hull fit with an ABSOLUTE squared-objective gap gate."""
function nh_fit!(p::NormalizedHullProjector,target::AbstractVector)
    length(target)==size(p.A,1) || throw(DimensionMismatch("full hull target"))
    all(isfinite,target) || throw(ArgumentError("finite full hull target required"))
    z=Float64.(target)
    for i in eachindex(p.lambda)
        set_objective_coefficient(p.model,p.lambda[i],-z[i])
    end
    passes=0
    for iteration in 1:p.maximum_iterations
        try
            optimize!(p.model)
        catch err
            return _nh_fallback!(p,z,"solver exception $(typeof(err))",iteration,passes)
        end
        status=termination_status(p.model)
        status in (JuMP.MOI.OPTIMAL,JuMP.MOI.ALMOST_OPTIMAL) ||
            return _nh_fallback!(p,z,"solver status $status",iteration,passes)
        bound=_nh_bounds(p.A,z,p.indices,-dual.(p.constraints),value.(p.lambda),value(p.intercept))
        passes+=1
        if bound.upper-bound.lower<=p.certificate_tolerance
            info=(solver_status=string(status),iterations=iteration,full_pricing_passes=passes,
                priced_columns=passes*size(p.A,2),full_type_count=size(p.A,2),
                working_cuts=length(p.indices),full_support_maximum=bound.full_support_maximum,
                support_variational_gap=bound.support_variational_gap,
                stationarity_error=bound.stationarity_error,
                maximum_working_dual_violation=max(0.,bound.violation),
                absolute_objective_gap=bound.upper-bound.lower,
                simplex_mass_sum=bound.mass_sum,raw_mass_sum=bound.raw_mass_sum,
                simplex_correction_l1=bound.mass_correction_l1,
                fallback_used=false,fallback_reason="",full_pricing_required=true,
                normalized_hull_not_cone=true,outward_rounded_certificate=false)
            return (;bound.fitted,bound.upper,bound.lower,bound.mass,bound.mass_sum,diagnostic=info)
        end
        candidates=findall(h->!p.included[h] && bound.scores[h]-value(p.intercept)>p.pricing_tolerance,
            axes(p.A,2))
        isempty(candidates) && return _nh_fallback!(p,z,
            "no missing cut but full objective bracket unresolved",iteration,passes)
        sort!(candidates;by=h->(-bound.scores[h],h))
        room=p.maximum_working_cuts-length(p.indices)
        room>0 || return _nh_fallback!(p,z,"working cut limit",iteration,passes)
        _nh_add_cuts!(p,candidates[1:min(length(candidates),p.batch_size,room)])
    end
    _nh_fallback!(p,z,"working iteration limit",p.maximum_iterations,passes)
end
