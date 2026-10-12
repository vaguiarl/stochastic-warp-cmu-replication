using Test, LinearAlgebra
include(joinpath(@__DIR__,"..","validate_cmu_superiority_floor.jl"))

@testset "independent full-coordinate triangle/product-simplex oracle" begin
    @test csfv_triangle_projection(.2,.25)==(.2,.25)
    @test csfv_triangle_projection(.7,.7)==(.5,.5)
    @test csfv_triangle_projection(-.2,1.2)==(0.,1.)
    @test csfv_gap(.7,.7)≈.16 atol=1e-14
    @test csfv_gap(-.2,1.2)==0
    @test csfv_gap(0.,1.2)==0
    @test csfv_gap(.7,.7;identical=true)==0
    Q=Rational{BigInt}
    @test csfv_gap(Q(7//10),Q(7//10))==Q(4//25)
    point=csfv_fullpoint((.7,.7))
    @test csfv_gap_full(point)≈.16 atol=1e-14
    # Adding-up-normal directions have the same contribution in both distances.
    @test csfv_gap_full(point+[.13,.13,-.24,-.24])≈csfv_gap_full(point) atol=1e-14
    fit=csfv_analytic_fit(point)
    @test fit.lower<=.16+1e-15 && fit.upper>=.16-1e-15
    @test parse(Q,fit.exact_certificate["gap_lower_exact"])==csfv_gap_full(Q.(point))
    @test_throws DimensionMismatch csfv_gap_full(ones(3))
end

@testset "new inference engine agrees with independent analytic adapter" begin
    integration=csfv_engine_integration()
    @test integration["passed"]
    @test integration["controlled_replicates"]==20
    @test integration["bootstrap_draws_each"]==79
    @test integration["maximum_point_error"]<1e-12
    @test integration["maximum_derivative_error"]<1e-9
    @test integration["all_rejection_and_critical_gates_agree"]
end

@testset "critical-cone derivatives and degeneracy gates" begin
    for h in ((1.,1.),(-1.,1.),(1.,-1.),(-1.,-1.),(.3,-.1),(.2,.7))
        e=1e-5
        @test csfv_gap(.5+e*h[1],.5+e*h[2])/e^2≈csfv_oracle(:edge,h...) atol=1e-8
        @test csfv_gap(e*h[1],1+e*h[2])/e^2≈csfv_oracle(:vertex,h...) atol=1e-6
        @test csfv_gap(e*h[1],1.2+e*h[2])/e^2≈csfv_oracle(:outside,h...) atol=1e-6
    end
    degenerate=csfv_decision(0.,zeros(399))
    @test !degenerate.reject && !degenerate.resolved && degenerate.tail==1
    @test csfv_decision(1.,zeros(399)).reject==false
    signed=csfv_decision(10.,vcat(fill(-1.,200),fill(1.,199)))
    @test signed.negative_draws==200
    @test signed.reject
    @test csfv_binomial_interval(0,500)[1]==0
    @test csfv_binomial_interval(500,500)[2]==1
end

@testset "small Gaussian protocol smoke: rank and immutable c roles" begin
    report=csfv_validation(;repetitions=20,bootstrap=39,sample_sizes=[1000],seed=9001)
    @test length(report["cases"])==10
    @test report["protocol"]["primary_c"]==1
    @test !report["protocol"]["real_empirical_tail_computed"]
    for c in report["cases"]
        @test c["failed_or_replaced_replicates"]==0
        @test c["projection_dimension"]==4
        @test c["sampling_rank"] in (1,2)
        @test [x["c"] for x in c["c_results"]]==[.5,1.,2.]
        @test c["c_results"][2]["role"]=="frozen_primary"
    end
    nullidentity=only(filter(c->c["scenario"]=="identical_R_equals_S",report["cases"]))
    @test all(r->r["rejections"]==0&&r["positive_critical_replicates"]==0,nullidentity["c_results"])
end
