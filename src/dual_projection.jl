"""Low-dimensional dual of the same literal source master projection.

Primal: min 1/2 ||e||², e + A*v = t, v>=0, e>=b.
Dual (as a minimization): min 1/2 ||lambda+mu||² - t'lambda - b*sum(mu),
A'lambda<=0, mu>=0. Without a residual floor, mu is fixed to zero.
The primal type masses are minus the JuMP upper-inequality duals. All columns
remain present as dual inequalities: there is no sampling or truncation.
"""
struct DualConeProjector
    model::Model
    lambda::Vector{VariableRef}
    mu::Vector{VariableRef}
    column_constraints::Vector{ConstraintRef}
    A::Matrix{Float64}
    residual_floor::Union{Nothing,Float64}
end

function DualConeProjector(A::AbstractMatrix; residual_floor::Union{Nothing,Float64}=nothing)
    I,H = size(A)
    I>0 && H>0 || throw(ArgumentError("empty type matrix"))
    all(x->x==0 || x==1,A) || throw(ArgumentError("nonbinary type matrix"))
    residual_floor===nothing || isfinite(residual_floor) || throw(ArgumentError("nonfinite residual floor"))
    Af = Float64.(A)
    model = Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model,"eps_abs",1e-9)
    set_optimizer_attribute(model,"eps_rel",1e-9)
    set_optimizer_attribute(model,"max_iter",200000)
    set_optimizer_attribute(model,"polish",true)
    @variable(model, lambda[1:I])
    @variable(model, mu[1:I] >= 0)
    if residual_floor===nothing
        for variable in mu
            fix(variable,0.0;force=true)
        end
    end
    column_constraints = @constraint(model,[h=1:H],sum(Af[i,h]*lambda[i] for i=1:I)<=0)
    floor_coefficient = residual_floor===nothing ? 0.0 : residual_floor
    @objective(model,Min,0.5sum((lambda[i]+mu[i])^2 for i=1:I)-floor_coefficient*sum(mu))
    return DualConeProjector(model,collect(lambda),collect(mu),collect(column_constraints),Af,residual_floor)
end

function project!(projector::DualConeProjector, pi::AbstractVector;
                  origin::AbstractVector=zeros(length(pi)))
    I = size(projector.A,1)
    length(pi)==I==length(origin) || throw(DimensionMismatch("projection input"))
    all(isfinite,pi) && all(isfinite,origin) || throw(ArgumentError("nonfinite projection target"))
    target = pi-origin
    if projector.residual_floor!==nothing
        minimum(target)>=projector.residual_floor-1e-8 ||
            throw(ArgumentError("source residual floor makes this target infeasible"))
    end
    for i=1:I
        set_objective_coefficient(projector.model,projector.lambda[i],-target[i])
    end
    optimize!(projector.model)
    status = termination_status(projector.model)
    status in (JuMP.MOI.OPTIMAL,JuMP.MOI.ALMOST_OPTIMAL) || error("dual cone QP failed: $status")
    lambda = value.(projector.lambda)
    mu = value.(projector.mu)
    residual = lambda+mu
    mass = -dual.(projector.column_constraints)
    fitted = projector.A*mass+origin
    mismatch = maximum(abs.(fitted+residual-pi))
    mismatch<=1e-7 || error("dual QP primal equality residual $mismatch")
    minimum(mass)>=-1e-8 || error("dual QP recovered negative type mass")
    gradient = projector.A'*lambda
    maximum(gradient)<=1e-6 || error("dual QP column infeasibility")
    abs(dot(mass,gradient))<=1e-6 || error("dual QP type complementarity")
    if projector.residual_floor===nothing
        maximum(abs.(mu))<=1e-7 || error("nonzero multiplier without a floor")
    else
        floor = projector.residual_floor
        minimum(residual)>=floor-1e-7 || error("dual QP primal residual floor")
        minimum(mu)>=-1e-7 || error("dual QP floor multiplier infeasibility")
        maximum(abs.(mu.*(residual.-floor)))<=1e-6 || error("dual QP floor complementarity")
    end
    squared_distance = sum(abs2,residual)
    floor_term = projector.residual_floor===nothing ? 0.0 : projector.residual_floor*sum(mu)
    dual_bound = dot(target,lambda)+floor_term-0.5squared_distance
    gap = abs(0.5squared_distance-dual_bound)
    gap<=1e-6 || error("QP primal/dual gap $gap exceeds tolerance")
    return (fitted=fitted, squared_distance=squared_distance, mass_sum=sum(mass),
        equality_error=mismatch, dual_violation=max(0.0,maximum(gradient)),
        minimum_residual=minimum(residual),
        active_residual_floors=projector.residual_floor===nothing ? 0 :
            count(x->x<=projector.residual_floor+1e-7,residual), primal_dual_gap=gap,
        solver_status=string(status))
end
