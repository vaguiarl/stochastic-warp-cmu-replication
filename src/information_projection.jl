"""Orthonormal budgetwise Helmert contrasts, with all probability rows retained.

`blocks` must partition `1:I`. Each block contributes length(block)-1 rows.
Singleton budgets contribute no row. The returned matrix B has B*B'=I and
annihilates every budget's constant vector. Contrasts are information-preserving
only together with the known budget adding-up constraints; they are not an
excuse to discard normalization violations in an estimator.
"""
function adding_up_contrasts(blocks::AbstractVector, I::Integer)
    I > 0 || throw(ArgumentError("probability dimension must be positive"))
    groups = _information_check_blocks(blocks, Int(I))
    d = I-length(groups)
    B = zeros(Float64, d, I)
    row = 0
    for block in groups, k in 1:length(block)-1
        row += 1
        scale = inv(sqrt(Float64(k*(k+1))))
        for j in 1:k
            B[row,block[j]] = scale
        end
        B[row,block[k+1]] = -k*scale
    end
    return B
end

"""Helmert contrasts from the validated one-own-zero patch sign matrix."""
adding_up_contrasts(X::AbstractMatrix) = adding_up_contrasts(_patch_groups(X), size(X,1))

function _information_check_blocks(blocks::AbstractVector, I::Int)
    isempty(blocks) && throw(ArgumentError("budget blocks must be nonempty"))
    groups = Vector{Vector{Int}}()
    seen = falses(I)
    for block in blocks
        isempty(block) && throw(ArgumentError("a budget block is empty"))
        group = Int[]
        for index in block
            index isa Integer || throw(ArgumentError("budget row indices must be integers"))
            1 <= index <= I || throw(ArgumentError("budget row index is out of range"))
            seen[index] && throw(ArgumentError("budget blocks overlap or repeat a row"))
            seen[index] = true
            push!(group,index)
        end
        push!(groups,group)
    end
    all(seen) || throw(ArgumentError("budget blocks do not cover all probability rows"))
    return groups
end

# B may be a row-rotation of Helmert contrasts. B'B, not individual row
# supports, identifies its blockwise contrast subspace, including singletons.
function _information_infer_blocks(B::Matrix{Float64}; tolerance::Float64=1e-10)
    I = size(B,2)
    gram = B'*B
    seen = falses(I)
    groups = Vector{Vector{Int}}()
    for first in 1:I
        seen[first] && continue
        block = Int[]
        stack = [first]
        seen[first] = true
        while !isempty(stack)
            i = pop!(stack)
            push!(block,i)
            for j in 1:I
                if !seen[j] && abs(gram[i,j]) > tolerance
                    seen[j] = true
                    push!(stack,j)
                end
            end
        end
        push!(groups,sort!(block))
    end
    return groups
end

function _information_symmetric(M::AbstractMatrix, label::String)
    Base.require_one_based_indexing(M)
    size(M,1)==size(M,2) || throw(DimensionMismatch("$label must be square"))
    all(isfinite,M) || throw(ArgumentError("$label contains nonfinite values"))
    result = Matrix{Float64}(M)
    scale = max(1.0, isempty(result) ? 0.0 : maximum(abs,result))
    symmetry_error = isempty(result) ? 0.0 : maximum(abs,result-result')
    symmetry_error <= 1e-12*scale || throw(ArgumentError("$label is not symmetric"))
    # Only roundoff-level antisymmetry is averaged, never eigenvalues/jitter.
    return (result+result')/2
end

function _information_covariance(S::AbstractMatrix, d::Int)
    size(S)==(d,d) || throw(DimensionMismatch("contrast covariance must be $d by $d"))
    result = _information_symmetric(S,"contrast covariance")
    isposdef(Symmetric(result)) || throw(ArgumentError(
        "contrast covariance is not strictly positive definite; no ridge is added"))
    return result
end

"""Symmetric matrix logarithm of a strictly SPD matrix; no jitter or flooring.

Roundoff-level antisymmetry (relative tolerance 1e-12) is explicitly averaged.
Zero/negative eigenvalues, nonfinite values and substantive asymmetry fail.
"""
function matrix_log_spd(S::AbstractMatrix)
    result = _information_symmetric(S,"SPD logarithm input")
    eigenpairs = eigen(Symmetric(result))
    all(>(0),eigenpairs.values) || throw(ArgumentError("SPD logarithm requires positive eigenvalues"))
    output = eigenpairs.vectors*Diagonal(log.(eigenpairs.values))*eigenpairs.vectors'
    all(isfinite,output) || throw(ArgumentError("nonfinite SPD logarithm"))
    return (output+output')/2
end

"""Matrix exponential on the symmetric log-covariance chart; no jitter.

Floating-point underflow to a zero eigenvalue or overflow fails visibly.
"""
function matrix_exp_symmetric(L::AbstractMatrix)
    result = _information_symmetric(L,"symmetric exponential input")
    eigenpairs = eigen(Symmetric(result))
    values = exp.(eigenpairs.values)
    all(x -> isfinite(x) && x>0,values) || throw(ArgumentError(
        "symmetric exponential overflows or underflows the SPD chart"))
    output = eigenpairs.vectors*Diagonal(values)*eigenpairs.vectors'
    all(isfinite,output) && isposdef(Symmetric(output)) || throw(ArgumentError(
        "symmetric exponential is not numerically SPD"))
    return (output+output')/2
end

"""Full-covariance projection onto a normalized type convex hull.

Primal: min_(nu>=0,sum(nu)=1) 1/2 (z-B*A*nu)'Sigma^-1(z-B*A*nu).
Dual: min_(lambda,t) 1/2 lambda'Sigma*lambda-z'lambda+t,
subject to (B*A_h)'lambda<=t for EVERY supplied type h.

The dual has d+1 variables, not one variable per type. Type masses are recovered
as minus the JuMP upper-inequality duals; t-stationarity enforces their sum one.
No types are omitted, no covariance ridge is added, and the model is a convex
hull, not the unnormalized source cone. A projector is mutable and must not be
shared between concurrent solves. Use one projector per worker.
"""
mutable struct InformationProjector
    model::Model
    lambda::Vector{VariableRef}
    support::VariableRef
    type_constraints::Vector{ConstraintRef}
    A::Matrix{Float64}
    B::Matrix{Float64}
    contrast_types::Matrix{Float64}
    blocks::Vector{Vector{Int}}
    covariance::Matrix{Float64}
    tolerance::Float64
end

function InformationProjector(A::AbstractMatrix, B::AbstractMatrix;
        covariance::AbstractMatrix, blocks=nothing, tolerance::Real=1e-7)
    Base.require_one_based_indexing(A,B)
    I,H = size(A)
    d = size(B,1)
    I>0 && H>0 || throw(ArgumentError("type matrix must be nonempty"))
    size(B,2)==I || throw(DimensionMismatch("contrast/type probability rows disagree"))
    d>0 || throw(ArgumentError("there are no informative contrast coordinates"))
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("invalid certificate tolerance"))
    all(isfinite,A) && all(isfinite,B) || throw(ArgumentError("nonfinite type/contrast matrix"))
    minimum(A)>=0 || throw(ArgumentError("type columns contain negative probabilities"))
    Af,Bf = Matrix{Float64}(A),Matrix{Float64}(B)
    groups = blocks===nothing ? _information_infer_blocks(Bf) :
        _information_check_blocks(blocks,I)
    length(groups)==I-d || throw(ArgumentError("contrast rank does not match budget blocks"))
    expected_B = adding_up_contrasts(groups,I)
    maximum(abs,Bf*Bf'-Matrix{Float64}(LinearAlgebra.I,d,d))<=1e-10 ||
        throw(ArgumentError("contrast rows are not orthonormal"))
    maximum(abs,Bf'*Bf-expected_B'*expected_B)<=1e-10 || throw(ArgumentError(
        "contrasts do not span the budgetwise adding-up subspace"))
    for group in groups
        maximum(abs,vec(sum(Af[group,:];dims=1)).-1)<=1e-10 || throw(ArgumentError(
            "each type must have probability mass one on every budget"))
    end
    S = _information_covariance(covariance,d)
    F = Bf*Af
    model = Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model,"eps_abs",1e-10)
    set_optimizer_attribute(model,"eps_rel",1e-10)
    set_optimizer_attribute(model,"max_iter",200000)
    set_optimizer_attribute(model,"polish",true)
    @variable(model,lambda[1:d])
    @variable(model,support)
    constraints = @constraint(model,[h=1:H],sum(F[i,h]*lambda[i] for i=1:d)<=support)
    projector = InformationProjector(model,collect(lambda),support,collect(constraints),
        Af,Bf,F,groups,S,Float64(tolerance))
    _set_information_objective!(projector,zeros(d))
    return projector
end

function _set_information_objective!(projector::InformationProjector,z::AbstractVector)
    d = length(projector.lambda)
    S,lambda = projector.covariance,projector.lambda
    @objective(projector.model,Min,
        0.5sum(S[i,j]*lambda[i]*lambda[j] for i=1:d,j=1:d)-
        sum(z[i]*lambda[i] for i=1:d)+projector.support)
    return nothing
end

"""Solve and independently certify the normalized full-information projection.

Returned `mass_raw` records the QP's recovered masses. For a feasible primal
certificate only, its tiny negative roundoff is clipped and its total explicitly
normalized; `mass_correction_l1` reports this correction and a correction above
the configured tolerance aborts. This does NOT alter the projection target,
covariance or model. Objective and support optimality are then recomputed from
the returned feasible certificate over all types, independently of QP values.

Failures throw, rather than returning a distance from an uncertified solve.
`covariance` updates the same reusable dual model when supplied. It is a d by d
contrast covariance, not an I by I singular probability covariance.
"""
function project_information!(projector::InformationProjector,z::AbstractVector;
        covariance::Union{Nothing,AbstractMatrix}=nothing)
    Base.require_one_based_indexing(z)
    d = size(projector.B,1)
    length(z)==d || throw(DimensionMismatch("contrast projection target"))
    all(isfinite,z) || throw(ArgumentError("nonfinite contrast target"))
    covariance===nothing || (projector.covariance=_information_covariance(covariance,d))
    target = Vector{Float64}(z)
    _set_information_objective!(projector,target)
    optimize!(projector.model)
    solver_status = termination_status(projector.model)
    solver_status in (JuMP.MOI.OPTIMAL,JuMP.MOI.ALMOST_OPTIMAL) ||
        error("information projection QP failed: $solver_status")
    lambda = value.(projector.lambda)
    support = value(projector.support)
    mass_raw = -dual.(projector.type_constraints)
    all(isfinite,mass_raw) && all(isfinite,lambda) && isfinite(support) ||
        error("information projection returned nonfinite solution")
    tol = projector.tolerance
    raw_mass_sum = sum(mass_raw)
    raw_nonnegative_violation = max(0.0,-minimum(mass_raw))
    raw_mass_error = abs(raw_mass_sum-1)
    mass = max.(mass_raw,0.0)
    sum(mass)>0 || error("information projection has no positive recovered mass")
    mass ./= sum(mass)
    correction = sum(abs,mass-mass_raw)
    max(raw_nonnegative_violation,raw_mass_error,correction)<=tol ||
        error("information projection simplex certificate failed; correction=$correction")

    fitted_probability = projector.A*mass
    fitted = projector.contrast_types*mass
    residual = target-fitted
    S = projector.covariance
    normal = cholesky(Symmetric(S))\residual
    squared_distance = dot(residual,normal)
    isfinite(squared_distance) && squared_distance>=0 ||
        error("information projection has invalid independently computed objective")
    scores = projector.contrast_types'*normal
    maximum_support = maximum(scores)
    independent_dual_bound = dot(target,normal)-0.5dot(normal,S*normal)-maximum_support
    primal_half = 0.5squared_distance
    signed_primal_dual_gap = primal_half-independent_dual_bound
    support_gap = maximum_support-dot(normal,fitted)
    dual_violation = max(0.0,maximum(projector.contrast_types'*lambda.-support))
    stationarity_error = maximum(abs,S*lambda-(target-projector.contrast_types*mass_raw))
    complementarity_error = maximum(abs,mass_raw.*(support.-projector.contrast_types'*lambda))
    budget_error = maximum(abs(sum(fitted_probability[group])-1) for group in projector.blocks)
    contrast_error = maximum(abs,projector.B*fitted_probability-fitted)
    probability_violation = max(0.0,-minimum(fitted_probability))
    solver_objective = 0.5dot(lambda,S*lambda)-dot(target,lambda)+support
    objective_value_error = abs(solver_objective-objective_value(projector.model))
    solver_primal_dual_gap = abs(primal_half+solver_objective)
    objective_scale = 1+abs(primal_half)+abs(independent_dual_bound)
    # Full squared-distance, not the half-objective used by the QP. This is
    # a numerical primal/support-dual certificate, not interval arithmetic.
    # Taking abs also exposes a tiny negative floating-point duality gap.
    objective_error_bound = 2abs(signed_primal_dual_gap)
    stationarity_scale = 1+maximum(abs,target)+maximum(abs,fitted)
    dual_scale = 1+abs(support)+maximum(abs,projector.contrast_types'*lambda)
    certified = abs(signed_primal_dual_gap)<=tol*objective_scale &&
        abs(support_gap)<=tol*objective_scale &&
        solver_primal_dual_gap<=tol*objective_scale &&
        objective_value_error<=tol*objective_scale &&
        dual_violation<=tol*dual_scale &&
        stationarity_error<=tol*stationarity_scale &&
        complementarity_error<=tol*objective_scale &&
        max(budget_error,contrast_error,probability_violation)<=tol
    certified || error("information projection failed independent certificate: " *
        "gap=$signed_primal_dual_gap, support_gap=$support_gap, " *
        "stationarity=$stationarity_error, adding_up=$budget_error")
    return (fitted_contrast=fitted, fitted_probability=fitted_probability,
        squared_distance=squared_distance, mass=mass, mass_raw=mass_raw,
        mass_sum=sum(mass), raw_mass_sum=raw_mass_sum,
        mass_correction_l1=correction, raw_mass_error=raw_mass_error,
        raw_nonnegative_violation=raw_nonnegative_violation,
        budget_adding_up_error=budget_error, probability_nonnegative_violation=probability_violation,
        contrast_consistency_error=contrast_error, stationarity_error=stationarity_error,
        equality_error=max(contrast_error,stationarity_error),
        nonnegativity_error=max(raw_nonnegative_violation,probability_violation),
        support_violation=max(0.0,support_gap), add_up_error=budget_error,
        dual_constraint_violation=dual_violation, complementarity_error=complementarity_error,
        support_variational_gap=support_gap, primal_dual_gap=abs(signed_primal_dual_gap),
        signed_primal_dual_gap=signed_primal_dual_gap,
        objective_error_bound=objective_error_bound,
        objective_upper_bound=squared_distance,
        objective_lower_bound=max(0.0,2independent_dual_bound),
        independent_dual_bound=independent_dual_bound, maximum_support=maximum_support,
        solver_primal_dual_gap=solver_primal_dual_gap,
        solver_objective_error=objective_value_error,
        covariance_condition=cond(S), certificate_tolerance=tol,
        certified=true, status="certified", solver_status=string(solver_status))
end
