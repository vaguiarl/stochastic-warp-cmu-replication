using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "simulate_cmu_restrictiveness.jl"))

@testset "valid reference probabilities and fixed dyadic normalization" begin
    groups = [[1, 2], [3, 4, 5], [6]]
    for alpha in CRB_PRIORS, seed in 1:200
        p = crb_probability(MersenneTwister(seed), groups, alpha)
        @test all(g -> sum(p.exact[g]) == 1, groups)
        @test all(>=(0), p.exact)
        @test CRB_Q.(p.point) == p.exact
        @test iszero(ceip_simplex_floor(p.point, groups).floor)
        @test p.point == crb_probability(MersenneTwister(seed), groups, alpha).point
    end
    for values in ([1., 2., 3.], [0., 1., 0.], [1e-100, 1.], [.1, .2, .3, .4])
        n = crb_largest_remainder(values)
        @test sum(n.counts) == big(1) << 40
        @test sum(n.exact) == 1
        @test n.maximum_error <= CRB_Q(1) / (big(1) << 40)
        @test all(>=(0), n.counts)
    end
    sparse = crb_largest_remainder([1e-100, 1.])
    @test sparse.rounded_zero_coordinates == 1
    @test sparse.point == [0., 1.]
    @test_throws ArgumentError crb_largest_remainder([0., 0.])
    @test_throws ArgumentError crb_largest_remainder([-1., 2.])
    @test_throws ArgumentError crb_probability(MersenneTwister(1), groups, .5)
end

# Full-coordinate triangle u+v<=1 inside two two-patch budget simplices.
# The analytic squared distance to its Minkowski enlargement is
# max(0,.5-beta)^2 for p=(.75,.25,.75,.25).
const CRBT_GROUPS = [[1, 2], [3, 4]]
const CRBT_A = Int8[0 1 0; 1 0 1; 0 0 1; 1 1 0]
const CRBT_F = ceip_build_family(CRBT_A, CRBT_GROUPS; label="analytic_test_triangle")
const CRBT_P = CRB_Q[3//4, 1//4, 3//4, 1//4]
const CRBT_QW = fill(CRB_Q(1//2), 4)

@testset "exact support, optimized ray and population normalization" begin
    v = CRBT_P - CRBT_QW
    support = crb_support(CRBT_F, v)
    @test support.value == 0
    @test support.all_columns == 3
    @test support.value == maximum(dot(v, CRBT_A[:, h]) for h in axes(CRBT_A, 2))
    for cap in (0., .005, .01, .025, .05, .10, .20, .49, .75, .90)
        b = crb_ray_bounds(CRBT_P, CRBT_QW, CRBT_F, cap;
                          warp_lower=CRB_Q(1//4), warp_upper=CRB_Q(1//4))
        truth = max(CRB_Q(0), CRB_Q(1//2) - b.beta)^2
        @test b.lower <= truth <= b.upper
        @test b.ray_lower == truth
        @test b.complete_mass + 2b.singleton_budget_mass + b.null_mass == 1
        @test b.complete_mass + b.singleton_budget_mass == b.tau
        @test b.null_mass >= 0
        @test b.supportB == (1-b.beta)*b.supportW + b.beta*b.supportS
        @test b.ray_lower == 2b.ray_t*b.advantage - b.ray_t^2*b.norm2
        @test sum(abs2, CRBT_P - ((1-b.beta)*CRBT_QW + b.beta*CRBT_P)) == b.blend_upper
    end
    zero = crb_ray_bounds(CRBT_QW, CRBT_QW, CRBT_F, .10; warp_lower=CRB_Q(0), warp_upper=CRB_Q(0))
    @test zero.norm2 == zero.ray_t == zero.lower == zero.upper == 0
    saturated = crb_ray_bounds(CRBT_P, CRBT_QW, CRBT_F, .9; warp_lower=CRB_Q(1//4), warp_upper=CRB_Q(1//4))
    @test saturated.beta == 1
    @test saturated.lower == saturated.upper == 0
    @test saturated.method == "beta_one_exact_product_simplex_identity"
    @test_throws ArgumentError crb_ray_bounds(CRBT_P, CRBT_QW, CRBT_F, 1.; warp_lower=0, warp_upper=1)
    @test_throws ErrorException crb_ray_bounds(zeros(CRB_Q, 4), CRBT_QW, CRBT_F, .1; warp_lower=0, warp_upper=1)
end

@testset "existing complete fit certificates and independent full QP" begin
    source = csfi_make_fit(CRBT_A, CRBT_GROUPS; tolerance=1e-8, store_certificate=true)
    projector = CMUEmptyInnerProjector(CRBT_F; tolerance=1e-8)
    for repetition in 1:12
        probability = crb_probability(MersenneTwister(100 + repetition), CRBT_GROUPS, repetition%2 == 0 ? 1. : .25)
        fit = source.fit(probability.point)
        primal = crb_reconstruct_primal(source.family, fit, probability.exact)
        @test sum(abs2, probability.exact-primal.q) == fit.upper
        @test all(g -> sum(primal.q[g]) == 1, CRBT_GROUPS)
        @test fit.upper-fit.lower <= CRB_Q(1e-8)
        for cap in (.01, .10)
            b = crb_ray_bounds(probability.exact, primal.q, source.family, cap;
                              warp_lower=fit.lower, warp_upper=fit.upper)
            full = ceip_fit!(projector, probability.point, cap)
            lo = parse(CRB_Q, full["certificate"]["squared_distance_lower_exact"])
            hi = parse(CRB_Q, full["certificate"]["squared_distance_upper_exact"])
            @test b.lower <= hi
            @test lo <= b.upper
            @test hi-lo <= CRB_Q(1e-8)
        end
    end
    # Complete Cartesian coverage, not RUM==WARP alone, justifies analytic zero.
    cartesian = Int8[0 1 0 1; 1 0 1 0; 0 0 1 1; 1 1 0 0]
    identity_source = csfi_make_fit(cartesian, CRBT_GROUPS; store_certificate=true)
    fit = identity_source.fit(Float64.(CRBT_P))
    primal = crb_reconstruct_primal(identity_source.family, fit, CRBT_P)
    @test primal.q == CRBT_P
    @test fit.lower == fit.upper == 0
    @test primal.witness == "verified_full_Cartesian_product_distribution"
end

@testset "records, paired summaries and no overwrite" begin
    source = csfi_make_fit(CRBT_A, CRBT_GROUPS; store_certificate=true)
    records = Dict{String,Any}[]
    for prior in eachindex(CRB_PRIORS), repetition in 1:2
        seed = 500 + 100prior + repetition
        p = crb_probability(MersenneTwister(seed), CRBT_GROUPS, CRB_PRIORS[prior])
        fit = source.fit(p.point)
        push!(records, crb_target_record(1, prior, repetition, p, fit, fit, source; seed))
    end
    @test all(r -> r["simplex_floor_exact"] == "0//1", records)
    @test all(r -> length(r["caps"]) == length(CRB_CAPS), records)
    summary = crb_summarize(Dict(1=>records), [Dict("cell"=>1,"years"=>[1,2],"frozen_annual_n_sum"=>10)]; targets_per_prior=2)
    @test summary["normalized_frozen_window_weights_exact"] == ["1//1"]
    @test length(summary["pooled"]) == 2
    @test summary["p_values_power_membership_counts_or_model_volume_estimates"] == false
    @test all(p -> p["RUM_MSE"]["Monte_Carlo_SE_midpoint"] !== nothing, summary["pooled"])
    mktempdir() do path
        target = joinpath(path, "new.json")
        crb_new(target, Dict("original"=>true))
        original = read(target, String)
        @test_throws ErrorException crb_new(target, Dict("original"=>false))
        @test read(target, String) == original
    end
end
