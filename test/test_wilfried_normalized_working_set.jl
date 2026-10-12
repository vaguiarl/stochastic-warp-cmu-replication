using Test, Random, LinearAlgebra
include(joinpath(@__DIR__,"../wilfried_normalized_working_set.jl"))

@testset "complete normalized-hull working-set projection" begin
    A=Float64[1 0;0 1;1 0;0 1]
    p=NormalizedHullProjector(A;initial_indices=[1],batch_size=1)
    for z in ([.5,.5,.5,.5],[.8,.2,.3,.7],[2.,-1.,2.,-1.],[0.,0.,0.,0.])
        fit=nh_fit!(p,z)
        t=clamp((z[1]-z[2]+z[3]-z[4]+2)/4,0.,1.)
        truth=sum(abs2,z-[t,1-t,t,1-t])
        @test fit.lower-1e-10<=truth<=fit.upper+1e-10
        @test fit.upper-fit.lower<=1e-8
        @test abs(sum(fit.mass)-1)<1e-12
        @test minimum(fit.mass)>=0
        @test fit.diagnostic.full_type_count==2
        @test fit.diagnostic.priced_columns>=2
        @test fit.diagnostic.normalized_hull_not_cone
    end
    # Cone and normalized hull differ: zero is in the former, not the latter.
    @test nh_fit!(p,zeros(4)).lower>.99
    @test_throws ArgumentError NormalizedHullProjector(zeros(2,0))
    @test_throws ArgumentError NormalizedHullProjector([1. .5;0. .5])
    @test_throws ArgumentError NormalizedHullProjector([1 0;0 0])
    @test_throws ArgumentError NormalizedHullProjector(A;certificate_tolerance=0)
    @test_throws ArgumentError NormalizedHullProjector(A;initial_indices=[3])
    @test_throws ArgumentError nh_fit!(p,[NaN,0,0,0])
    @test_throws DimensionMismatch nh_fit!(p,zeros(3))
    # Forced disclosed full-model fallback does not omit the missing optimum.
    fallback=NormalizedHullProjector(A;initial_indices=[1],maximum_working_cuts=1)
    f=nh_fit!(fallback,[0.,1.,0.,1.])
    @test f.diagnostic.fallback_used
    @test f.upper<1e-8
    @test f.diagnostic.full_type_count==2
    strict=NormalizedHullProjector(A;initial_indices=[1],maximum_working_cuts=1,allow_full_fallback=false)
    @test_throws ErrorException nh_fit!(strict,[0.,1.,0.,1.])

    blocks=[[1,2],[3,4],[5,6]]
    full=hcat([Float64[v[1]==1,v[1]==2,v[2]==1,v[2]==2,v[3]==1,v[3]==2]
        for v in Iterators.product(1:2,1:2,1:2)]...)
    B=KSReplication.adding_up_contrasts(blocks,6)
    direct=InformationProjector(full,B;blocks,covariance=Matrix{Float64}(I,3,3),tolerance=1e-7)
    work=NormalizedHullProjector(full;initial_indices=[1],batch_size=2)
    rng=MersenneTwister(70091)
    for _ in 1:50
        z=randn(rng,6)
        w=nh_fit!(work,z)
        d=project_information!(direct,B*z)
        normal=sum((sum(z[g])-1)^2/length(g) for g in blocks)
        truth=d.squared_distance+normal
        @test w.lower-1e-8<=truth<=w.upper+1e-8
        @test w.upper-w.lower<=1e-8
        @test maximum(abs.(full*w.mass-w.fitted))<1e-12
        @test abs(sum(w.mass)-1)<1e-12
        @test minimum(w.mass)>=0
    end
end
