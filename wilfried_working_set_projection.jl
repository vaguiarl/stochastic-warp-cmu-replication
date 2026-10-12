using KSReplication, JuMP, OSQP, LinearAlgebra

"""Finite working-set optimization of the FULL, no-floor binary type cone.

Only the solver's dual constraints are generated incrementally. Every returned
solution is exhaustively priced against every column of the original A, with
an explicitly feasible full-cone primal and a repaired full-cone dual bound.
This is not a sampled/truncated preference model. Each instance owns a mutable
solver and must be used by one worker at a time. Floating-point bounds are
diagnostics, not outward-rounded formal interval certificates.
"""
mutable struct CWNoFloorConeProjector
    A::Matrix{Float64}
    column_sum::Float64
    model::Model
    lambda::Vector{VariableRef}
    indices::Vector{Int}
    constraints::Vector{ConstraintRef}
    included::BitVector
    batch_size::Int
    maximum_iterations::Int
    maximum_working_cuts::Int
    pricing_tolerance::Float64
    certificate_tolerance::Float64
    equality_tolerance::Float64
    allow_full_fallback::Bool
    fallback::Union{Nothing,DualConeProjector}
end

function CWNoFloorConeProjector(A::AbstractMatrix;
        initial_indices=nothing, batch_size::Integer=32,
        maximum_iterations::Integer=200, maximum_working_cuts::Integer=5000,
        pricing_tolerance::Real=1e-9, certificate_tolerance::Real=1e-8,
        equality_tolerance::Real=1e-7, allow_full_fallback::Bool=true)
    I,H=size(A)
    I>0 && H>0 || throw(ArgumentError("empty full type matrix"))
    all(x->x==0 || x==1,A) || throw(ArgumentError("full A must be binary"))
    sums=vec(sum(A;dims=1))
    length(unique(sums))==1 && first(sums)>0 ||
        throw(ArgumentError("all full columns must have the same positive sum"))
    batch_size>0 && maximum_iterations>0 && maximum_working_cuts>0 ||
        throw(ArgumentError("positive finite working-set limits required"))
    all(x->isfinite(x) && x>0,(pricing_tolerance,certificate_tolerance,equality_tolerance)) ||
        throw(ArgumentError("finite positive numerical tolerances required"))
    Af=Float64.(A)
    model=Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model,"eps_abs",1e-10)
    set_optimizer_attribute(model,"eps_rel",1e-10)
    set_optimizer_attribute(model,"max_iter",200000)
    set_optimizer_attribute(model,"polish",true)
    @variable(model,lambda[1:I])
    @objective(model,Min,0.5sum(lambda[i]^2 for i in 1:I))
    projector=CWNoFloorConeProjector(Af,Float64(first(sums)),model,collect(lambda),
        Int[],ConstraintRef[],falses(H),Int(batch_size),Int(maximum_iterations),
        min(Int(maximum_working_cuts),H),Float64(pricing_tolerance),
        Float64(certificate_tolerance),Float64(equality_tolerance),allow_full_fallback,nothing)
    seeds=initial_indices===nothing ? collect(1:min(Int(batch_size),H)) : collect(initial_indices)
    !isempty(seeds) && all(h->h isa Integer && 1<=h<=H,seeds) ||
        throw(ArgumentError("initial indices must be nonempty valid full-column indices"))
    length(unique(seeds))<=projector.maximum_working_cuts ||
        throw(ArgumentError("initial working set exceeds declared cut limit"))
    _cw_add_cuts!(projector,Int.(seeds))
    return projector
end

function _cw_add_cuts!(p::CWNoFloorConeProjector,indices)
    for h in indices
        p.included[h] && continue
        length(p.indices)<p.maximum_working_cuts || error("working-set limit reached")
        constraint=@constraint(p.model,sum(p.A[i,h]*p.lambda[i] for i in axes(p.A,1))<=0)
        push!(p.indices,h);push!(p.constraints,constraint);p.included[h]=true
    end
    return nothing
end

function _cw_bounds(A, column_sum, target, lambda, indices, multipliers)
    all(isfinite,lambda) && all(isfinite,multipliers) || error("nonfinite dual solver output")
    mass=max.(0.0,-multipliers)
    fitted=A[:,indices]*mass
    all(isfinite,fitted) || error("nonfinite feasible primal")
    upper=sum(abs2,target-fitted)
    support=A'*lambda
    all(isfinite,support) || error("nonfinite full pricing")
    maxsupport=maximum(support)
    safety=1e-12*(1+maximum(abs,lambda))
    repair=max(0.0,maxsupport)/column_sum+safety
    feasible_lambda=lambda .- repair
    repaired_support=A'*feasible_lambda
    maximum(repaired_support)<=1e-12 || error("full repaired dual is infeasible")
    lower=max(0.0,2dot(target,feasible_lambda)-sum(abs2,feasible_lambda))
    isfinite(upper) && isfinite(lower) || error("nonfinite projection bounds")
    lower<=upper+1e-10 || error("primal/dual bound inversion")
    return (;fitted,upper,lower=min(lower,upper),mass_sum=sum(mass),support,
        maxsupport,equality_error=maximum(abs.(fitted+lambda-target)),
        repaired_full_violation=max(0.0,maximum(repaired_support)),repair)
end

function _cw_full_fallback!(p::CWNoFloorConeProjector,target,reason,iterations,passes)
    p.allow_full_fallback || error("full-cone working-set optimization failed: $reason")
    p.fallback===nothing && (p.fallback=DualConeProjector(p.A;residual_floor=nothing))
    diagnostic=project!(p.fallback,target)
    result=_cw_bounds(p.A,p.column_sum,target,value.(p.fallback.lambda),
        collect(axes(p.A,2)),dual.(p.fallback.column_constraints))
    result.equality_error<=p.equality_tolerance ||
        error("full fallback equality error exceeds declared tolerance")
    result.upper-result.lower<=p.certificate_tolerance*max(1.0,result.upper) ||
        error("full fallback certificate width exceeds declared tolerance")
    info=(primal_dual_gap=(result.upper-result.lower)/2,
        equality_error=result.equality_error,solver_status=diagnostic.solver_status,
        iterations=iterations,full_pricing_passes=passes+1,
        priced_columns=(passes+1)*size(p.A,2),working_cuts=length(p.indices),
        full_type_count=size(p.A,2),full_support_maximum=result.maxsupport,
        repaired_full_dual_violation=result.repaired_full_violation,
        dual_uniform_repair=result.repair,fallback_used=true,fallback_reason=String(reason))
    return (;result.fitted,result.upper,result.lower,result.mass_sum,diagnostic=info)
end

"""Fit the complete no-floor cone; accepts no silent truncation or failed draw.

The returned named tuple matches `wf_certified_fit`. The full A is immutable by
contract. Additional diagnostics disclose the working set and any full-model
fallback. Stopping requires full-column pricing, a primal/dual bracket, and
the recovered primal equality check—not a restricted-QP objective alone.
"""
function cw_fit!(p::CWNoFloorConeProjector,target::AbstractVector)
    length(target)==size(p.A,1) || throw(DimensionMismatch("full cone target"))
    all(isfinite,target) || throw(ArgumentError("nonfinite full cone target"))
    z=Float64.(target)
    for i in eachindex(p.lambda)
        set_objective_coefficient(p.model,p.lambda[i],-z[i])
    end
    passes=0
    for iteration in 1:p.maximum_iterations
        try
            optimize!(p.model)
        catch error
            return _cw_full_fallback!(p,z,"working solver exception: $(typeof(error))",iteration,passes)
        end
        status=termination_status(p.model)
        status in (JuMP.MOI.OPTIMAL,JuMP.MOI.ALMOST_OPTIMAL) ||
            return _cw_full_fallback!(p,z,"working solver status $status",iteration,passes)
        result=_cw_bounds(p.A,p.column_sum,z,value.(p.lambda),p.indices,dual.(p.constraints))
        passes+=1
        if result.maxsupport<=p.pricing_tolerance
            if result.equality_error<=p.equality_tolerance &&
                    result.upper-result.lower<=p.certificate_tolerance*max(1.0,result.upper)
                info=(primal_dual_gap=(result.upper-result.lower)/2,
                    equality_error=result.equality_error,solver_status=string(status),
                    iterations=iteration,full_pricing_passes=passes,
                    priced_columns=passes*size(p.A,2),working_cuts=length(p.indices),
                    full_type_count=size(p.A,2),full_support_maximum=result.maxsupport,
                    repaired_full_dual_violation=result.repaired_full_violation,
                    dual_uniform_repair=result.repair,fallback_used=false,fallback_reason="")
                return (;result.fitted,result.upper,result.lower,result.mass_sum,diagnostic=info)
            end
            return _cw_full_fallback!(p,z,"full pricing passed but numerical certificate did not",iteration,passes)
        end
        candidates=findall(h->!p.included[h] && result.support[h]>p.pricing_tolerance,axes(p.A,2))
        isempty(candidates) && return _cw_full_fallback!(p,z,
            "only existing cuts violate pricing tolerance",iteration,passes)
        sort!(candidates;by=h->(-result.support[h],h))
        room=p.maximum_working_cuts-length(p.indices)
        room>0 || return _cw_full_fallback!(p,z,"working cut limit",iteration,passes)
        added=candidates[1:min(length(candidates),p.batch_size,room)]
        _cw_add_cuts!(p,added)
    end
    return _cw_full_fallback!(p,z,"iteration limit",p.maximum_iterations,passes)
end
