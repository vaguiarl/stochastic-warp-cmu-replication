@testset "Equivalent low-dimensional projection dual" begin
    A = Int8[1 0;0 1;1 0;0 1]
    for floor in (nothing,-1.0), target in ([.3,.7,.3,.7],[.2,.8,.8,.2],[-.4,.4,1.6,.8])
        primal = project!(ConeProjector(A;residual_floor=floor),target)
        dual = project!(DualConeProjector(A;residual_floor=floor),target)
        @test dual.squared_distance ≈ primal.squared_distance atol=1e-7
        @test dual.fitted ≈ primal.fitted atol=1e-7
        @test dual.mass_sum ≈ primal.mass_sum atol=1e-7
        @test dual.primal_dual_gap < 1e-7
    end
    # Source floor actually binds in this analytical one-column projection.
    single = reshape(Int8[1,1],2,1)
    target = [-.5,3.5]
    primal = project!(ConeProjector(single;residual_floor=-1.0),target)
    dual = project!(DualConeProjector(single;residual_floor=-1.0),target)
    @test dual.fitted ≈ [.5,.5] atol=1e-7
    @test dual.squared_distance ≈ 10 atol=1e-7
    @test dual.squared_distance ≈ primal.squared_distance atol=1e-7
    @test dual.active_residual_floors == 1
end
