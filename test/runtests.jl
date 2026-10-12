using Test, KSReplication, JSON3, Random

# Run only public, self-contained acceptance tests. Historical empirical
# drivers are not silently invoked by Pkg.test().
@testset "core enumeration and projection dual" begin
    include("types_tests.jl")
    include("dual_projection_tests.jl")
end
@testset "received checker counterexample" begin
    include("test_wilfried_checker.jl")
end
# Engines expose similarly named globals. Fresh modules avoid accidental
# dependence on the order of separately authored research checks.
module NormalizedChecks
include("test_wilfried_normalized_working_set.jl")
end
module EmptyDemandChecks
include("test_cmu_empty_inner_projection.jl")
end
module InferenceChecks
include("test_cmu_superiority_floor_validation.jl")
end
module RestrictivenessChecks
include("test_cmu_restrictiveness.jl")
end
