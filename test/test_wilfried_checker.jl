using Test, Random, KSReplication
include(joinpath(@__DIR__, "..", "run_wilfried_replication.jl"))

@testset "Received checker: economically feasible missed two-cycle" begin
    @test wf_round_J(0.0000005)==0.000001
    @test wf_round_J(0.0000025)==0.000003
    P = [2 3 1; 3 2 3; 1 3 2]
    Q = [2 1 1; 1 2 1; 1 1 2]
    expenditures = [8,10,8]
    costs = P*Q'
    @test diag(costs)==expenditures
    @test all(>(0),P) && all(>(0),Q)
    F = costs .< reshape(expenditures,:,1)
    @test F==Bool[0 0 1;0 0 0;1 0 0]
    @test has_two_cycle(F)
    @test !all(vec(maximum(F .& F';dims=1)))
    X = Int8[0 1 -1;1 0 1;-1 1 0]
    @test size(wf_faulty_types(X),2)==1
    @test size(enumerate_types(X;restriction=:warp),2)==0
    @test size(enumerate_types(X;restriction=:rum),2)==0
    permutations = [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    faulty_counts = Int[]
    for order in permutations
        Y = X[order,order]
        push!(faulty_counts,size(wf_faulty_types(Y),2))
        @test size(enumerate_types(Y;restriction=:warp),2)==0
        @test size(enumerate_types(Y;restriction=:rum),2)==0
    end
    @test sort(faulty_counts)==[0,0,1,1,1,1]
end

@testset "Equivalent Cartesian cone projection, not a WARP projection" begin
    X = Int8[0 -1;0 1;-1 0;1 0]
    A = Float64.(enumerate_types(X;restriction=:unrestricted))
    groups = KSReplication._patch_groups(X)
    fit = wf_product_fit(A,groups,[.8,.2,.3,.7])
    @test fit.upper<1e-20
    @test isapprox(fit.mass_sum,1.0;atol=1e-12)
    zero_fit = wf_product_fit(A,groups,[-1.,-2.,-3.,-4.])
    @test zero_fit.mass_sum==0
    @test zero_fit.upper==30
    rng = MersenneTwister(4409)
    projector = DualConeProjector(A;residual_floor=nothing)
    for repetition in 1:30
        target = randn(rng,4)
        analytic = wf_product_fit(A,groups,target)
        reference = project!(projector,target)
        @test isapprox(analytic.upper,reference.squared_distance;atol=1e-7,rtol=1e-7)
        @test minimum(analytic.fitted)>=0
        @test isapprox(sum(analytic.fitted[groups[1]]),sum(analytic.fitted[groups[2]]);atol=1e-12)
        @test analytic.upper-analytic.lower<1e-8
    end
end

@testset "Checker set nesting and correct-model ordering invariance" begin
    rng = MersenneTwister(20261009)
    for J in 2:5, repetition in 1:15
        X = rand(rng,Int8[-1,1],3J,J)
        for budget in 1:J
            X[3budget-2:3budget,budget] .= 0
        end
        warp = enumerate_types(X;restriction=:warp)
        rum = enumerate_types(X;restriction=:rum)
        faulty = wf_faulty_types(X)
        warp_set = Set(Tuple.(eachcol(warp)))
        faulty_set = Set(Tuple.(eachcol(faulty)))
        @test all(Tuple(c) in warp_set for c in eachcol(rum))
        @test all(Tuple(c) in faulty_set for c in eachcol(warp))
        J==2 && @test warp_set==faulty_set
        order = randperm(rng,J)
        @test size(enumerate_types(X[:,order];restriction=:warp),2)==size(warp,2)
        @test size(enumerate_types(X[:,order];restriction=:rum),2)==size(rum,2)
    end
end
