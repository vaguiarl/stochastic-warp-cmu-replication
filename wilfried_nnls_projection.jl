using KSReplication, LinearAlgebra
if !isdefined(@__MODULE__,:CWNoFloorConeProjector)
    include(joinpath(@__DIR__,"wilfried_working_set_projection.jl"))
end

"""Warm Lawson--Hanson NNLS for the COMPLETE no-floor binary type cone.

The passive variables are an optimization support, never a truncated model.
Every outer iteration prices every original column. Accepted fits have a
nonnegative full-cone primal and an independently repaired full-cone dual.
Numerical QR rank guards never authorize excluding a type from the model:
unresolved dependence, iteration limits, or certificate failure use a disclosed
complete-model fallback or throw. Instances are stateful and worker-local.
"""
mutable struct NNLSFullConeProjector
    A::Matrix{Float64}
    column_sum::Float64
    active::Vector{Int}
    weights::Vector{Float64}
    pricing_tolerance::Float64
    certificate_tolerance::Float64
    rank_multiplier::Float64
    cleanup_tolerance::Float64
    maximum_iterations::Int
    maximum_least_squares::Int
    allow_working_fallback::Bool
    fallback::Union{Nothing,CWNoFloorConeProjector}
end

function NNLSFullConeProjector(A::AbstractMatrix;
        pricing_tolerance::Real=1e-9,certificate_tolerance::Real=1e-8,
        rank_multiplier::Real=10.0,cleanup_tolerance::Real=1e-13,
        maximum_iterations::Integer=2000,maximum_least_squares::Integer=10000,
        allow_working_fallback::Bool=true)
    I,H=size(A)
    I>0 && H>0 || throw(ArgumentError("empty full type matrix"))
    all(x->x==0 || x==1,A) || throw(ArgumentError("full A must be binary"))
    sums=vec(sum(A;dims=1))
    length(unique(sums))==1 && first(sums)>0 ||
        throw(ArgumentError("all columns must have the same positive sum"))
    all(x->isfinite(x) && x>0,(pricing_tolerance,certificate_tolerance,rank_multiplier,cleanup_tolerance)) ||
        throw(ArgumentError("finite positive NNLS tolerances required"))
    maximum_iterations>0 && maximum_least_squares>0 ||
        throw(ArgumentError("positive finite NNLS iteration limits required"))
    return NNLSFullConeProjector(Float64.(A),Float64(first(sums)),Int[],Float64[],
        Float64(pricing_tolerance),Float64(certificate_tolerance),Float64(rank_multiplier),
        Float64(cleanup_tolerance),Int(maximum_iterations),Int(maximum_least_squares),
        allow_working_fallback,nothing)
end

function _nnls_qr(p::NNLSFullConeProjector,indices)
    length(indices)<=size(p.A,1) || return nothing
    factor=qr(p.A[:,indices],ColumnNorm())
    diagonals=abs.(diag(factor.R))
    isempty(diagonals) && return nothing
    tolerance=p.rank_multiplier*max(size(p.A,1),length(indices))*eps(Float64)*maximum(diagonals)
    count(>(tolerance),diagonals)==length(indices) || return nothing
    return factor
end

# Each step starts from feasible weights, solves LS on the passive support,
# and walks only as far as the first nonnegativity boundary if LS is infeasible.
function _nnls_passive_solve(p,z,indices,weights,ls_calls)
    while !isempty(indices)
        ls_calls+=1
        ls_calls<=p.maximum_least_squares || return indices,weights,ls_calls,"least-squares iteration limit"
        factor=_nnls_qr(p,indices)
        factor===nothing && return indices,weights,ls_calls,"dependent passive QR support"
        proposal=factor\z
        all(isfinite,proposal) || return indices,weights,ls_calls,"nonfinite passive QR solution"
        if all(>(0),proposal)
            return indices,proposal,ls_calls,""
        end
        nonpositive=findall(<=(0),proposal)
        isempty(nonpositive) && error("NNLS positivity branch inconsistency")
        alpha=minimum(weights[i]/(weights[i]-proposal[i]) for i in nonpositive
            if weights[i]-proposal[i]>0;init=1.0)
        0<=alpha<=1+1e-12 || return indices,weights,ls_calls,"invalid NNLS boundary step"
        alpha=min(1.0,alpha)
        stepped=weights.+alpha.*(proposal.-weights)
        minimum(stepped)>=-1e-11 || return indices,weights,ls_calls,"NNLS feasibility lost"
        stepped=max.(stepped,0.0)
        tolerance=p.cleanup_tolerance*max(1.0,maximum(stepped;init=0.0))
        retained=findall(>(tolerance),stepped)
        length(retained)<length(indices) || return indices,weights,ls_calls,"NNLS boundary step made no progress"
        indices,weights=indices[retained],stepped[retained]
    end
    return indices,weights,ls_calls,""
end

function _nnls_fallback!(p,z,reason,iterations,passes,ls_calls,rank_guards)
    p.allow_working_fallback || error("full-cone NNLS failed: $reason")
    p.fallback===nothing && (p.fallback=CWNoFloorConeProjector(p.A))
    fit=cw_fit!(p.fallback,z)
    old=fit.diagnostic
    diagnostic=(primal_dual_gap=old.primal_dual_gap,equality_error=old.equality_error,
        solver_status=old.solver_status,iterations=iterations+old.iterations,
        full_pricing_passes=passes+old.full_pricing_passes,
        priced_columns=(passes+old.full_pricing_passes)*size(p.A,2),
        working_cuts=old.working_cuts,full_type_count=size(p.A,2),
        full_support_maximum=old.full_support_maximum,
        repaired_full_dual_violation=old.repaired_full_dual_violation,
        dual_uniform_repair=old.dual_uniform_repair,fallback_used=true,
        fallback_reason="NNLS: $reason; working fallback: $(old.fallback_reason)",
        nnls_used=true,nnls_completed=false,least_squares_calls=ls_calls,
        rank_guard_rejections=rank_guards,full_model_fallback=old.fallback_used)
    return (;fit.fitted,fit.upper,fit.lower,fit.mass_sum,diagnostic)
end

"""Return the same fit fields as `wf_certified_fit` and `cw_fit!`.

Warm starts retain only an optimization support. Every full column remains in
the mandatory pricing pass. This preserves the full Euclidean cone, including
zero masses, with no simplex equality, residual floor, or statistical changes.
"""
function nnls_fit!(p::NNLSFullConeProjector,target::AbstractVector)
    length(target)==size(p.A,1) || throw(DimensionMismatch("full cone target"))
    all(isfinite,target) || throw(ArgumentError("nonfinite full cone target"))
    z=Float64.(target)
    indices,weights=copy(p.active),copy(p.weights)
    ls_calls=0;passes=0;rank_guards=0
    indices,weights,ls_calls,reason=_nnls_passive_solve(p,z,indices,weights,ls_calls)
    isempty(reason) || return _nnls_fallback!(p,z,reason,0,passes,ls_calls,rank_guards)
    for iteration in 1:p.maximum_iterations
        fitted=isempty(indices) ? zeros(length(z)) : p.A[:,indices]*weights
        residual=z-fitted
        result=_cw_bounds(p.A,p.column_sum,z,residual,indices,-weights)
        passes+=1
        if result.maxsupport<=p.pricing_tolerance
            if result.upper-result.lower<=p.certificate_tolerance*max(1.0,result.upper)
                p.active=copy(indices);p.weights=copy(weights)
                diagnostic=(primal_dual_gap=(result.upper-result.lower)/2,
                    equality_error=result.equality_error,solver_status="NNLS_FULL_PRICED",
                    iterations=iteration,full_pricing_passes=passes,
                    priced_columns=passes*size(p.A,2),working_cuts=length(indices),
                    full_type_count=size(p.A,2),full_support_maximum=result.maxsupport,
                    repaired_full_dual_violation=result.repaired_full_violation,
                    dual_uniform_repair=result.repair,fallback_used=false,fallback_reason="",
                    nnls_used=true,nnls_completed=true,least_squares_calls=ls_calls,
                    rank_guard_rejections=rank_guards,full_model_fallback=false)
                return (;result.fitted,result.upper,result.lower,result.mass_sum,diagnostic)
            end
            return _nnls_fallback!(p,z,"full pricing passed but certificate did not",iteration,passes,ls_calls,rank_guards)
        end
        active_set=Set(indices)
        candidates=findall(h->!(h in active_set) && result.support[h]>p.pricing_tolerance,axes(p.A,2))
        sort!(candidates;by=h->(-result.support[h],h))
        selected=0
        for h in candidates
            if _nnls_qr(p,vcat(indices,h))!==nothing
                selected=h
                break
            end
            rank_guards+=1
        end
        selected!=0 || return _nnls_fallback!(p,z,
            "positive full pricing but no independent entering support",iteration,passes,ls_calls,rank_guards)
        push!(indices,selected);push!(weights,0.0)
        indices,weights,ls_calls,reason=_nnls_passive_solve(p,z,indices,weights,ls_calls)
        isempty(reason) || return _nnls_fallback!(p,z,reason,iteration,passes,ls_calls,rank_guards)
    end
    return _nnls_fallback!(p,z,"NNLS outer iteration limit",p.maximum_iterations,passes,ls_calls,rank_guards)
end
