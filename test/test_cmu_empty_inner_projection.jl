using Test, LinearAlgebra
include(joinpath(@__DIR__,"../cmu_empty_inner_projection.jl"))

qr(x)=parse(CEIP_Q,String(x))
function check_point(f,raw,result)
    certificate=result["certificate"]
    indices=Int.(certificate["primal_indices"])
    mass=qr.(certificate["primal_masses_exact"])
    q=qr.(certificate["conditional_fitted_exact"])
    tau=qr(certificate["response_rate_exact"])
    beta=qr(certificate["unary_mix_fraction_exact"])
    @test ceip_check_primal(f,indices,mass,fill(tau,f.n_budgets),q)
    @test all(>=(0),mass)
    @test sum(mass)==1
    @test all(sum(q[g])==1 for g in f.budget_groups)
    @test beta==min(CEIP_Q(1),(1-tau)/((f.n_budgets-1)*tau))
    lambda=CEIP_Q.(raw)-q
    @test lambda==qr.(certificate["residual_patch_weights_exact"])
    upper=sum(abs2,lambda)
    @test upper==qr(certificate["squared_distance_upper_exact"])
    support=(1-beta)*ceip_complete_support(f,lambda)+beta*sum(maximum(lambda[g]) for g in f.budget_groups)
    @test support==qr(certificate["conditional_support_exact"])
    lower=max(ceip_simplex_floor(raw,f.budget_groups).floor,
        2dot(CEIP_Q.(raw),lambda)-sum(abs2,lambda)-2support)
    @test lower==qr(certificate["squared_distance_lower_exact"])
    @test lower<=upper
    @test result["squared_distance_lower"]<=Float64(lower)
    @test result["squared_distance_upper"]>=Float64(upper)
    @test result["point_bounds_exact_rational"]
    @test result["new_p_values"]===nothing
    (;lower,upper,q,tau)
end

function check_floor(f,q,result)
    indices=Int.(result["primal_certificate_columns"])
    mass=qr.(result["primal_certificate_mass_exact"])
    rates=qr.(result["primal_response_rates_exact"])
    y=qr.(result["dual_certificate_patch_weights_exact"])
    gamma=qr(result["dual_support_bound_exact"])
    @test ceip_check_primal(f,indices,mass,rates,q)
    @test all(>=(0),mass)
    @test sum(mass)==1
    @test dot(q,y)==1
    @test all(dot(q[g],y[g])>=0 for g in f.budget_groups) || result["response_policy"]=="common"
    @test gamma==ceip_population_support(f,y)
    @test all(dot(CEIP_Q.(f.A[:,h]),y)<=gamma for h in 1:f.n_types)
    @test qr(result["response_lower_exact"])==minimum(rates)
    @test qr(result["response_lower_exact"])<=qr(result["response_upper_exact"])
    @test qr(result["epsilon_lower_exact"])==1-qr(result["response_upper_exact"])
    @test qr(result["epsilon_upper_exact"])==1-qr(result["response_lower_exact"])
    @test result["population_absence_share_identified"]==false
    @test result["new_p_values"]===nothing
end

@testset "Actual singleton/null inner family, not arbitrary deletion" begin
    Ac=Int8[1 0 0;0 1 1;0 1 0;1 0 1]
    groups=[[1,2],[3,4]]
    f=ceip_build_family(Ac,groups)
    @test f.n_types==8
    @test Matrix(f.A[:,1:3])==Ac
    @test Matrix(f.A[:,4:7])==Matrix{Int8}(I,4,4)
    @test all(iszero,f.A[:,8])
    @test all(iszero,f.D[:,8])
    @test f.external_microfoundation_registry_required
    @test_throws ErrorException ceip_build_family(Ac,[[1,2],[2,3,4]])
    @test_throws ErrorException ceip_build_family(Int8[1;0;;],[[1],[2]])
    raw=[.9,.1,.9,.1]
    solver=CMUEmptyInnerProjector(f)
    for cap in (0.,.05,.2,.45)
        result=ceip_fit!(solver,raw,cap)
        certificate=check_point(f,raw,result)
        beta=min(1.,cap/(1-cap))
        truth=4*(.9-min(.9,.5*(1+beta)))^2
        @test Float64(certificate.lower)<=truth+1e-10
        @test Float64(certificate.upper)>=truth-1e-10
        @test Float64(certificate.upper-certificate.lower)<=1e-8
    end
    @test_throws ErrorException ceip_fit!(solver,raw,1.)
    @test_throws ErrorException ceip_fit!(solver,[NaN,0.,0.,0.],0.)
    complete=ceip_complete_fit(Ac,raw,groups)
    @test complete["squared_distance_lower"]<=.64<=complete["squared_distance_upper"]+1e-10
    q=CEIP_Q[9//10,1//10,9//10,1//10]
    for policy in (:common,:variable)
        breakdown=ceip_floor_breakdown(f,q;response_policy=policy)
        check_floor(f,q,breakdown)
        @test breakdown["response_lower"]<=5/9+1e-10
        @test breakdown["response_upper"]>=5/9-1e-10
        @test breakdown["epsilon_lower"]<=4/9+1e-10
        @test breakdown["epsilon_upper"]>=4/9-1e-10
    end
end

@testset "No mechanical shrinking of conditional noise floor" begin
    Ac=Int8[1 1 0 0;0 0 1 1;1 0 1 0;0 1 0 1]
    groups=[[1,2],[3,4]]
    f=ceip_build_family(Ac,groups)
    raw=[-.02,1.02,.25,.75]
    floor=ceip_simplex_floor(raw,groups)
    @test floor.q==CEIP_Q[0,1,1//4,3//4]
    @test abs(floor.floor_float-.0008)<1e-15
    solver=CMUEmptyInnerProjector(f)
    for cap in CEIP_CAPS
        result=ceip_fit!(solver,raw,cap)
        certificate=check_point(f,raw,result)
        @test certificate.lower==floor.floor
        @test certificate.upper-floor.floor<CEIP_Q(1e-8)
    end
    for policy in (:common,:variable)
        breakdown=ceip_floor_breakdown(f,floor.q;response_policy=policy)
        check_floor(f,floor.q,breakdown)
        @test breakdown["epsilon_upper"]<1e-10
    end
end

@testset "Common response is a substantive additional restriction" begin
    Ac=reshape(Int8[1,0,1,0,1,0],6,1)
    groups=[[1,2],[3,4],[5,6]]
    f=ceip_build_family(Ac,groups)
    q=CEIP_Q[2//3,1//3,1,0,1,0]
    common=ceip_floor_breakdown(f,q;response_policy=:common)
    variable=ceip_floor_breakdown(f,q;response_policy=:variable)
    check_floor(f,q,common);check_floor(f,q,variable)
    @test common["response_lower"]<=3/5+1e-10<=common["response_upper"]+1e-10
    @test variable["response_lower"]<=2/3+1e-10<=variable["response_upper"]+1e-10
    @test common["epsilon_lower"]>variable["epsilon_upper"]
    @test common["common_response_is_additional_restriction"]
    @test !variable["common_response_is_additional_restriction"]
    @test length(unique(variable["primal_response_rates_exact"]))>1
end

@testset "Exact cap arithmetic and global all-null inheritance" begin
    Ac=reshape(Int8[1,0,1,0,1,0],6,1)
    groups=[[1,2],[3,4],[5,6]]
    f=ceip_build_family(Ac,groups)
    raw=[.9,.1,.9,.1,.9,.1]
    solver=CMUEmptyInnerProjector(f)
    first=ceip_fit!(solver,raw,.01)
    prior=solver.previous
    # Force comparison with a mathematically feasible prior certificate whose
    # upper is less than the new raw solver proposal by using identical cap.
    second=ceip_fit!(solver,raw,.01)
    cert=check_point(f,raw,second)
    @test cert.tau==1-CEIP_Q(.01)
    @test cert.upper<=prior.cert.upper+CEIP_Q(1e-10)
    final=ceip_fit!(solver,raw,.8)
    full=check_point(f,raw,final)
    @test qr(final["certificate"]["unary_mix_fraction_exact"])==1
    @test full.upper==ceip_simplex_floor(raw,groups).floor
    @test sum(qr.(final["certificate"]["primal_masses_exact"]))==1
    @test (CEIP_Q(.01)/(6*(1-CEIP_Q(.01))))≈.0016835016835016834
    @test 1-CEIP_Q(.025)+CEIP_Q(.025)==1
    # An exact complete-choice optimum remains feasible at the next cap by
    # mixing the WHOLE population with the true all-null consumer.
    fullA=Int8[1 1 0 0;0 0 1 1;1 0 1 0;0 1 0 1]
    fullfamily=ceip_build_family(fullA,[[1,2],[3,4]])
    noisefloor_raw=[-.02,1.02,.25,.75]
    noisefloor=ceip_simplex_floor(noisefloor_raw,fullfamily.budget_groups)
    priorcert=ceip_point_certificate(fullfamily,noisefloor_raw,CEIP_Q(1),CEIP_Q(0),
        noisefloor.q,[3,4],CEIP_Q[1//4,3//4])
    fullsolver=CMUEmptyInnerProjector(fullfamily)
    fullsolver.previous=(;raw=copy(noisefloor_raw),epsilon=0.,cert=priorcert)
    inherited=ceip_fit!(fullsolver,noisefloor_raw,.01)
    inheritedcert=check_point(fullfamily,noisefloor_raw,inherited)
    @test inherited["upper_inherited_by_global_all_null_mixture"]
    @test inheritedcert.upper==noisefloor.floor
    inheritedindices=Int.(inherited["certificate"]["primal_indices"])
    inheritedmasses=qr.(inherited["certificate"]["primal_masses_exact"])
    @test inheritedmasses[findfirst(==(fullfamily.null_index),inheritedindices)]==CEIP_Q(.01)
    @test all(i<=fullfamily.n_complete||i==fullfamily.null_index for i in inheritedindices)
end

@testset "Tiny positive target retained in exact LP certificates" begin
    Ac=Int8[1 1 0 0;0 0 1 1;1 0 1 0;0 1 0 1]
    groups=[[1,2],[3,4]]
    f=ceip_build_family(Ac,groups)
    tiny=CEIP_Q(2.0^-48)
    q=CEIP_Q[tiny,1-tiny,1//4,3//4]
    for policy in (:common,:variable)
        result=ceip_floor_breakdown(f,q;response_policy=policy)
        check_floor(f,q,result)
        @test result["response_lower"]>1-1e-7
        @test result["response_upper"]==1
    end
end
