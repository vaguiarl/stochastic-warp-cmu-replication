using KSReplication, JuMP, HiGHS, Random, SHA

const CMUE_Q = Rational{BigInt}
const CMUE_WITNESS_VERSION = "common-shifted-log-majority-3goods-v1"
const CMUE_SEARCH_SEED = 44213
const CMUE_SHIFT_SCALES = CMUE_Q[1//1000, 1//10000, 1//100000, 1//1000000]
const CMUE_SHAPES_PER_SCALE = 16
const CMUE_ANCHOR_SHARE_DENOMINATORS = BigInt[10^6, 10^9, 10^12]
const CMUE_ANCHOR_MARGIN_TOLERANCE = 1e-9
const CMUE_INTERPRETATION =
    "One common coherent CMU score: median of the three shifted-log utility " *
    "differences, equivalently max over the three two-voter coalitions of " *
    "their minimum. Null budgets have actual empty demand over their entire " *
    "continuous nonnegative linear budget. Exact algebra certificates depend " *
    "on the stated concavity/directional-derivative majority-core theorem; " *
    "the numerical LP only proposes a subsequently verified rational anchor. " *
    "Single-budget existence is knife-edge in prices; unknown unrestricted " *
    "participation rates can saturate conditional data. No empirical evidence " *
    "or stable counterfactual demand is claimed by witness generation alone."

cmue_source_sha256() = bytes2hex(sha256(read(@__FILE__)))

function cmue_exact_prices(P::AbstractMatrix)
    Base.require_one_based_indexing(P)
    size(P, 1) > 0 && size(P, 2) == 3 ||
        throw(ArgumentError("a nonempty three-good normalized-price matrix is required"))
    all(isfinite, P) && all(>(0), P) ||
        throw(ArgumentError("finite strictly positive normalized prices required"))
    CMUE_Q.(P)
end

function cmue_geometry_key(P, X)
    PQ = cmue_exact_prices(P)
    KSReplication._patch_groups(X)
    size(P, 1) == size(X, 2) || throw(DimensionMismatch("budget coordinates"))
    io = IOBuffer()
    print(io, CMUE_WITNESS_VERSION, '\n', size(P), '\n', size(X), '\n')
    for a in PQ
        print(io, a, '\n')
    end
    for a in X
        write(io, Int8(a))
    end
    bytes2hex(sha256(take!(io)))
end

"""Propose a positive strict patch anchor, then verify it in exact arithmetic.

Prices are normalized already and are not rescaled. The LP maximizes a common
strict cross-budget and positive-expenditure-share margin. Rounded shares are
normalized exactly, making the own-budget equality exact. Weak-only source
patches are explicitly rejected, not silently treated as strict cells.
"""
function cmue_strict_patch_anchor(P, X, row::Int)
    PQ = cmue_exact_prices(P)
    groups = KSReplication._patch_groups(X)
    size(P, 1) == size(X, 2) || throw(DimensionMismatch("budget coordinates"))
    row in axes(X, 1) || throw(BoundsError(X, row))
    own = only(findall(==(0), X[row, :]))
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "threads", 1)
    set_optimizer_attribute(model, "primal_feasibility_tolerance", 1e-9)
    set_optimizer_attribute(model, "dual_feasibility_tolerance", 1e-9)
    @variable(model, shares[1:3] >= 0)
    @variable(model, 0 <= margin <= 1/3)
    @constraint(model, sum(shares) == 1)
    @constraint(model, [k in 1:3], shares[k] >= margin)
    for j in axes(P, 1)
        j == own && continue
        cost = sum(Float64(PQ[j,k]/PQ[own,k])*shares[k] for k in 1:3)
        X[row,j] == -1 ? @constraint(model, cost + margin <= 1) :
            @constraint(model, cost - margin >= 1)
    end
    @objective(model, Max, margin)
    optimize!(model)
    status = termination_status(model)
    if status != JuMP.MOI.OPTIMAL
        return (; certified=false, reason="strict_anchor_LP_$status", row, own)
    end
    lp_margin = value(margin)
    lp_margin > CMUE_ANCHOR_MARGIN_TOLERANCE ||
        return (; certified=false, reason="no_certified_positive_strict_patch_margin", row, own, lp_margin)
    for denominator in CMUE_ANCHOR_SHARE_DENOMINATORS
        z = [round(BigInt, value(shares[k])*denominator)//denominator for k in 1:3]
        all(>(0), z) || continue
        z ./= sum(z)
        x = z ./ vec(PQ[own,:])
        costs = PQ*x
        costs[own] == 1 || error("internal exact anchor budget accounting")
        all(j -> j == own || X[row,j]*(costs[j]-1) > 0, axes(PQ,1)) || continue
        return (; certified=true, reason="exact_positive_strict_patch_anchor", row, own,
            x, shares=z, costs, lp_margin, share_denominator=denominator,
            minimum_positive_share=minimum(z),
            minimum_strict_cross_budget_margin=minimum(
                [abs(costs[j]-1) for j in axes(PQ,1) if j != own]; init=CMUE_Q(1)))
    end
    (; certified=false, reason="rationalized_anchor_failed_strict_sign_verification", row, own, lp_margin)
end

function _cmue_shapes(seed::Int)
    rng = MersenneTwister(seed)
    shapes = Matrix{Int}[Int[1 2 4; 2 4 1; 4 1 2]]
    for unused in 2:CMUE_SHAPES_PER_SCALE
        push!(shapes, rand(rng, 1:9, 3, 3))
    end
    shapes
end

# These calculations certify all of the continuous budget, not a finite grid.
# A majority core point at an interior budget boundary must be an individual
# voter optimum; if the other two tangent gradients have a nonzero determinant,
# a feasible tangent direction improves BOTH, contradicting majority dominance.
# At a goods boundary, the uniform marginal-ratio bound supplies a direction
# improving ALL voters by transferring expenditure from a good spending >=1/3.
function cmue_exact_model_audit(P, a::AbstractMatrix, d::AbstractMatrix;
        chosen_budget::Int=0, anchor=CMUE_Q[])
    PQ = cmue_exact_prices(P)
    size(a) == size(d) == (3,3) || throw(DimensionMismatch("three voters and three goods"))
    aq, dq = CMUE_Q.(a), CMUE_Q.(d)
    all(>(0), aq) && all(>(0), dq) ||
        throw(ArgumentError("positive shifted-log coefficients and shifts required"))
    J = size(PQ, 1)
    chosen_budget in 0:J || throw(ArgumentError("chosen budget must be zero or an input budget index"))
    xq = CMUE_Q.(anchor)
    chosen_budget == 0 || (length(xq) == 3 && all(>(0), xq)) ||
        throw(ArgumentError("positive three-good anchor required"))
    ideals = Array{CMUE_Q}(undef, J, 3, 3)
    multipliers = Matrix{CMUE_Q}(undef, J, 3)
    boundary_margins = Matrix{CMUE_Q}(undef, J, 3)
    tangent_determinants = Matrix{CMUE_Q}(undef, J, 3)
    for j in 1:J
        upper = maximum(aq[v,k]/(CMUE_Q(1//3)+PQ[j,k]*dq[v,k]) for v in 1:3, k in 1:3)
        for k in 1:3
            lower = minimum(aq[v,k]/(PQ[j,k]*dq[v,k]) for v in 1:3)
            boundary_margins[j,k] = lower-upper
            boundary_margins[j,k] > 0 || return (; certified=false,
                reason="boundary_entry_bound_failed", budget=j, good=k)
        end
        for v in 1:3
            lambda = sum(aq[v,:])/(1+sum(PQ[j,k]*dq[v,k] for k in 1:3))
            multipliers[j,v] = lambda
            for k in 1:3
                ideals[j,v,k] = aq[v,k]/(lambda*PQ[j,k])-dq[v,k]
                ideals[j,v,k] > 0 || return (; certified=false,
                    reason="individual_voter_optimum_not_interior", budget=j, voter=v, good=k)
                aq[v,k]/(ideals[j,v,k]+dq[v,k]) == lambda*PQ[j,k] ||
                    error("internal exact voter stationarity")
            end
            sum(PQ[j,k]*ideals[j,v,k] for k in 1:3) == 1 ||
                error("internal exact voter budget accounting")
            if j == chosen_budget && vec(ideals[j,v,:]) != xq
                return (; certified=false, reason="anchor_not_every_voter_optimum", budget=j, voter=v)
            end
            other = [i for i in 1:3 if i != v]
            tangent = Matrix{CMUE_Q}(undef, 2, 2)
            for (r, i) in enumerate(other)
                g3 = aq[i,3]/(ideals[j,v,3]+dq[i,3])/PQ[j,3]
                for k in 1:2
                    tangent[r,k] = aq[i,k]/(ideals[j,v,k]+dq[i,k])/PQ[j,k]-g3
                end
            end
            det = tangent[1,1]*tangent[2,2]-tangent[1,2]*tangent[2,1]
            tangent_determinants[j,v] = det
            j == chosen_budget || det != 0 || return (; certified=false,
                reason="individual_candidate_tangent_determinant_zero", budget=j, voter=v)
        end
    end
    (; certified=true, reason="exact_whole_budget_majority_core_certificate",
        chosen_budget, anchor=xq, a=aq, d=dq, P=PQ, ideals, multipliers,
        boundary_margins, tangent_determinants,
        chosen_budgets=chosen_budget == 0 ? Int[] : [chosen_budget],
        empty_budgets=[j for j in 1:J if j != chosen_budget])
end

"""Construct one common CMU with one chosen patch and real empty other budgets."""
function cmue_witness_for_patch(P, X, row::Int; seed::Int=CMUE_SEARCH_SEED)
    anchor = cmue_strict_patch_anchor(P, X, row)
    anchor.certified || return anchor
    PQ = cmue_exact_prices(P)
    attempts = 0
    last_reason = "not_attempted"
    for scale in CMUE_SHIFT_SCALES, (shape_index, shape) in enumerate(_cmue_shapes(seed))
        attempts += 1
        d = scale .* CMUE_Q.(shape)
        a = [PQ[anchor.own,k]*(anchor.x[k]+d[v,k]) for v in 1:3, k in 1:3]
        audit = cmue_exact_model_audit(PQ, a, d; chosen_budget=anchor.own, anchor=anchor.x)
        last_reason = audit.reason
        audit.certified || continue
        return (; audit..., row, anchor_certificate=anchor,
            patch_signs=Int8.(vec(X[row,:])), geometry_key=cmue_geometry_key(P,X),
            seed, shift_scale=scale, shape_index, shape, attempts,
            version=CMUE_WITNESS_VERSION, code_sha256=cmue_source_sha256(),
            interpretation=CMUE_INTERPRETATION)
    end
    (; certified=false, reason="registered_shift_search_exhausted", row,
        own=anchor.own, attempts, last_reason, anchor_certificate=anchor)
end

"""Construct a common shifted-log majority CMU empty on every input budget."""
function cmue_all_null_witness(P, X; seed::Int=CMUE_SEARCH_SEED)
    PQ = cmue_exact_prices(P)
    attempts = 0
    # These different Cobb-Douglas-like coefficient vectors form a triangle,
    # not three copies of one voter; shifts preserve continuity at zero goods.
    a = CMUE_Q[5//8 2//8 1//8; 1//8 5//8 2//8; 2//8 1//8 5//8]
    last_reason = "not_attempted"
    for scale in CMUE_SHIFT_SCALES, (shape_index, shape) in enumerate(_cmue_shapes(seed))
        attempts += 1
        d = scale .* CMUE_Q.(shape)
        audit = cmue_exact_model_audit(PQ, a, d)
        last_reason = audit.reason
        audit.certified || continue
        return (; audit..., row=0, anchor_certificate=nothing,
            patch_signs=Int8[], geometry_key=cmue_geometry_key(P,X),
            seed, shift_scale=scale, shape_index, shape, attempts,
            version=CMUE_WITNESS_VERSION, code_sha256=cmue_source_sha256(),
            interpretation=CMUE_INTERPRETATION)
    end
    (; certified=false, reason="registered_all_null_search_exhausted", attempts, last_reason)
end

"""Recompute the exact model audit, rather than trusting stored flags/margins."""
function cmue_verify_witness(P, X, witness)
    witness.certified || return false
    witness.version == CMUE_WITNESS_VERSION || return false
    witness.geometry_key == cmue_geometry_key(P,X) || return false
    witness.P == cmue_exact_prices(P) || return false
    audit = cmue_exact_model_audit(P, witness.a, witness.d;
        chosen_budget=witness.chosen_budget, anchor=witness.anchor)
    audit.certified || return false
    for key in (:ideals, :multipliers, :boundary_margins, :tangent_determinants,
            :chosen_budgets, :empty_budgets)
        getproperty(audit,key) == getproperty(witness,key) || return false
    end
    if witness.row != 0
        witness.row in axes(X,1) || return false
        witness.patch_signs == Int8.(vec(X[witness.row,:])) || return false
        only(findall(==(0),witness.patch_signs)) == witness.chosen_budget || return false
        costs = audit.P*audit.anchor
        costs[witness.chosen_budget] == 1 || return false
        all(j -> j == witness.chosen_budget || witness.patch_signs[j]*(costs[j]-1) > 0,
            axes(audit.P,1)) || return false
    else
        witness.chosen_budget == 0 || return false
    end
    true
end

"""Explicit JSON-safe certificate; rational inputs and claims are not floats."""
function cmue_witness_record(witness)
    witness.certified || throw(ArgumentError("cannot serialize an uncertified witness as a certificate"))
    Dict("version"=>witness.version, "code_sha256"=>witness.code_sha256,
        "geometry_key"=>witness.geometry_key, "row"=>witness.row,
        "chosen_budget"=>witness.chosen_budget, "anchor"=>string.(witness.anchor),
        "patch_signs"=>witness.patch_signs, "chosen_budgets"=>witness.chosen_budgets,
        "empty_budgets"=>witness.empty_budgets,
        "utilities"=>[Dict("a"=>string.(vec(witness.a[v,:])),
            "d"=>string.(vec(witness.d[v,:])),
            "formula"=>"u_v(x)=sum_k a_vk*log(x_k+d_vk)") for v in 1:3],
        "prices_exact_binary_rational"=>[string.(vec(witness.P[j,:])) for j in axes(witness.P,1)],
        "budget_certificates"=>[Dict("budget"=>j,
            "voter_ideals"=>[string.(vec(witness.ideals[j,v,:])) for v in 1:3],
            "voter_multipliers"=>string.(vec(witness.multipliers[j,:])),
            "boundary_entry_margins"=>string.(vec(witness.boundary_margins[j,:])),
            "candidate_pair_tangent_determinants"=>string.(vec(witness.tangent_determinants[j,:])))
            for j in axes(witness.P,1)],
        "search"=>Dict("seed"=>witness.seed,"shift_scale"=>string(witness.shift_scale),
            "shape_index"=>witness.shape_index,"shape"=>[vec(witness.shape[v,:]) for v in 1:3],
            "attempts"=>witness.attempts,"shapes_per_scale"=>CMUE_SHAPES_PER_SCALE,
            "registered_scales"=>string.(CMUE_SHIFT_SCALES),
            "strict_anchor_LP_tolerance"=>CMUE_ANCHOR_MARGIN_TOLERANCE,
            "registered_share_denominators"=>string.(CMUE_ANCHOR_SHARE_DENOMINATORS)),
        "interpretation"=>witness.interpretation,
        "analytic_majority_core_bridge_Lean_verified"=>false,
        "arithmetic_whole_budget_certificate_recomputed"=>true,
        "conditional_probability_saturation_is_not_empirical_evidence"=>true)
end

"""Parse a JSON certificate and independently recompute all stored claims."""
function cmue_verify_record(P, X, record)
    parseq(v) = parse(CMUE_Q, String(v))
    a = [parseq(record["utilities"][v]["a"][k]) for v in 1:3,k in 1:3]
    d = [parseq(record["utilities"][v]["d"][k]) for v in 1:3,k in 1:3]
    audit = cmue_exact_model_audit(P,a,d; chosen_budget=Int(record["chosen_budget"]),
        anchor=parseq.(record["anchor"]))
    audit.certified || return false
    record["version"] == CMUE_WITNESS_VERSION || return false
    record["geometry_key"] == cmue_geometry_key(P,X) || return false
    PQ = [parseq(record["prices_exact_binary_rational"][j][k]) for j in axes(P,1),k in 1:3]
    PQ == audit.P || return false
    for j in axes(P,1)
        cert = record["budget_certificates"][j]
        Int(cert["budget"]) == j || return false
        for v in 1:3
            parseq.(cert["voter_ideals"][v]) == vec(audit.ideals[j,v,:]) || return false
        end
        parseq.(cert["voter_multipliers"]) == vec(audit.multipliers[j,:]) || return false
        parseq.(cert["boundary_entry_margins"]) == vec(audit.boundary_margins[j,:]) || return false
        parseq.(cert["candidate_pair_tangent_determinants"]) == vec(audit.tangent_determinants[j,:]) || return false
    end
    Int.(record["chosen_budgets"]) == audit.chosen_budgets || return false
    Int.(record["empty_budgets"]) == audit.empty_budgets || return false
    row = Int(record["row"])
    if row != 0
        row in axes(X,1) || return false
        signs = Int8.(record["patch_signs"])
        signs == vec(X[row,:]) || return false
        only(findall(==(0),signs)) == audit.chosen_budget || return false
        costs = audit.P*audit.anchor
        costs[audit.chosen_budget] == 1 || return false
        all(j -> j == audit.chosen_budget || signs[j]*(costs[j]-1)>0,axes(P,1)) || return false
    else
        audit.chosen_budget == 0 || return false
    end
    true
end
