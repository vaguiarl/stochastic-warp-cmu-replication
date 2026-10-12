using Test
using KSReplication
using JSON3

# This exhaustive oracle does not call production graph, grouping, pruning,
# or cache helpers. Its cycle check removes zero-indegree vertices (Kahn),
# independently of the production depth-first search.
function oracle_type_sets(X::AbstractMatrix)
    n = size(X, 2)
    choices = [[row for row in axes(X, 1) if X[row, budget] == 0] for budget in 1:n]
    unrestricted = Set{Tuple}()
    warp = Set{Tuple}()
    rum = Set{Tuple}()
    for selected in Iterators.product(choices...)
        rows = Tuple(sort(collect(selected)))
        push!(unrestricted, rows)
        outgoing = [[target for target in 1:n
                     if source != target && X[selected[target], source] <= 0]
                    for source in 1:n]
        pair = any(target in outgoing[source] && source in outgoing[target]
                   for source in 1:n for target in (source + 1):n)
        pair || push!(warp, rows)
        indegree = [count(source -> target in outgoing[source], 1:n) for target in 1:n]
        ready = [vertex for vertex in 1:n if indegree[vertex] == 0]
        removed = 0
        while !isempty(ready)
            source = pop!(ready)
            removed += 1
            for target in outgoing[source]
                indegree[target] -= 1
                indegree[target] == 0 && push!(ready, target)
            end
        end
        removed == n && push!(rum, rows)
    end
    return (; unrestricted, warp, rum)
end

function binary_type_sets(A::AbstractMatrix)
    return Set(Tuple(findall(x -> x != 0, A[:, column])) for column in axes(A, 2))
end

function assert_binary_budget_choices(X, A)
    @test eltype(A) == Int8
    @test size(A, 1) == size(X, 1)
    @test all(value -> value == 0 || value == 1, A)
    @test length(binary_type_sets(A)) == size(A, 2)
    for budget in axes(X, 2)
        rows = [row for row in axes(X, 1) if X[row, budget] == 0]
        @test all(vec(sum(A[rows, :]; dims=1)) .== 1)
    end
end

@testset "pure graph predicates" begin
    empty_graph = falses(0, 0)
    singleton = falses(1, 1)
    self_loop = trues(1, 1)
    two_cycle = Bool[0 1; 1 0]
    three_cycle = Bool[0 1 0; 0 0 1; 1 0 0]
    four_cycle = Bool[0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 0]
    acyclic = Bool[0 1 1; 0 0 1; 0 0 0]
    @test !has_two_cycle(empty_graph)
    @test !has_directed_cycle(empty_graph)
    @test !has_two_cycle(singleton)
    @test !has_directed_cycle(singleton)
    @test !has_two_cycle(self_loop)
    @test has_directed_cycle(self_loop)
    @test has_two_cycle(two_cycle)
    @test has_directed_cycle(two_cycle)
    @test !has_two_cycle(three_cycle)
    @test has_directed_cycle(three_cycle)
    @test !has_two_cycle(four_cycle)
    @test has_directed_cycle(four_cycle)
    @test !has_two_cycle(acyclic)
    @test !has_directed_cycle(acyclic)
    @test_throws ArgumentError has_two_cycle(falses(2, 3))
    @test_throws ArgumentError has_directed_cycle(falses(2, 3))
end

@testset "minimal deterministic type restrictions" begin
    # Graph orientation is source budget -> target's chosen patch.
    three_cycle = Int8[0 1 -1; -1 0 1; 1 -1 0]
    two_cycle = Int8[0 -1; -1 0]
    acyclic = Int8[0 1 1; -1 0 1; -1 -1 0]
    singleton = reshape(Int8[0], 1, 1)
    for X in (three_cycle, two_cycle, acyclic, singleton)
        expected = oracle_type_sets(X)
        for restriction in (:warp, :rum, :unrestricted)
            A = enumerate_types(X; restriction)
            assert_binary_budget_choices(X, A)
            @test binary_type_sets(A) == getproperty(expected, restriction)
        end
    end
    @test size(enumerate_types(three_cycle; restriction=:warp), 2) == 1
    @test size(enumerate_types(three_cycle; restriction=:rum), 2) == 0
    @test size(enumerate_types(two_cycle; restriction=:warp), 2) == 0
    @test size(enumerate_types(two_cycle; restriction=:rum), 2) == 0
    @test size(enumerate_types(acyclic), 2) == 1
    @test enumerate_types(singleton) == ones(Int8, 1, 1)
end

@testset "open-patch validation and bounded enumeration" begin
    @test_throws ArgumentError enumerate_types(Int8[0 0; 1 0])
    @test_throws ArgumentError enumerate_types(Int8[1 1; 1 0])
    @test_throws ArgumentError enumerate_types(Int8[0 2; 1 0])
    @test_throws ArgumentError enumerate_types([0.0 NaN; 1.0 0.0])
    @test_throws ArgumentError enumerate_types(reshape(Int8[0, 1], 1, 2))
    @test_throws ArgumentError enumerate_types(zeros(Int8, 0, 0))
    @test_throws ArgumentError enumerate_types(reshape(Int8[0], 1, 1); restriction=:rpm)
    @test_throws ArgumentError enumerate_types(reshape(Int8[0], 1, 1); max_types=0)
    multiple = Int8[0 1; 0 -1; 1 0; -1 0]
    @test size(enumerate_types(multiple; restriction=:warp, max_types=3), 2) == 3
    @test enumerate_types(multiple; restriction=:warp) ==
          Int8[1 1 0; 0 0 1; 1 0 1; 0 1 0]
    @test_throws ArgumentError enumerate_types(multiple; restriction=:warp, max_types=2)
    @test_throws ArgumentError enumerate_types(multiple; restriction=:unrestricted, max_types=3)
    # Confirm a failed capped call leaves no hidden accumulated state.
    @test size(enumerate_types(multiple; restriction=:unrestricted, max_types=4), 2) == 4
end

@testset "frozen 1986–1992 median geometry, exhaustive oracle" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "median_1986_1992.json")
    fixture = JSON3.read(read(fixture_path, String))
    rows = fixture.patch_sign_matrix
    X = Int8[rows[row][budget] for row in eachindex(rows), budget in eachindex(rows[1])]
    @test size(X) == (24, 7)
    @test [count(==(0), X[:, budget]) for budget in axes(X, 2)] == [1, 2, 3, 4, 2, 5, 7]
    expected = oracle_type_sets(X)
    @test length(expected.unrestricted) == fixture.expected_counts.unrestricted == 1680
    @test length(expected.warp) == fixture.expected_counts.warp == 347
    @test length(expected.rum) == fixture.expected_counts.rum == 333
    @test length(setdiff(expected.warp, expected.rum)) == 14
    @test issubset(expected.rum, expected.warp)
    for restriction in (:warp, :rum, :unrestricted)
        A = enumerate_types(X; restriction)
        assert_binary_budget_choices(X, A)
        @test binary_type_sets(A) == getproperty(expected, restriction)
        @test A == enumerate_types(X; restriction) # Exact ordering is deterministic.
    end
    # Interleave rows across budgets. Map selected rows back before comparison.
    row_permutation = vcat(collect(2:2:24), collect(1:2:24))
    Xrows = X[row_permutation, :]
    budget_permutation = [7, 2, 5, 1, 6, 3, 4]
    for restriction in (:warp, :rum, :unrestricted)
        A = enumerate_types(Xrows; restriction)
        assert_binary_budget_choices(Xrows, A)
        mapped = Set(Tuple(sort(row_permutation[collect(rows)])) for rows in binary_type_sets(A))
        @test mapped == getproperty(expected, restriction)
        @test binary_type_sets(enumerate_types(X[:, budget_permutation]; restriction)) ==
              getproperty(expected, restriction)
        @test binary_type_sets(A) == getproperty(oracle_type_sets(Xrows), restriction)
    end
end

@testset "versioned type cache identity" begin
    X = Int8[0 1 -1; -1 0 1; 1 -1 0]
    warp_key = type_cache_key(X; restriction=:warp)
    @test length(warp_key) == 64
    @test warp_key == type_cache_key(copy(X); restriction=:warp)
    @test warp_key == type_cache_key(Float64.(X); restriction=:warp)
    @test warp_key != type_cache_key(X; restriction=:rum)
    @test warp_key != type_cache_key(X; restriction=:unrestricted)
    changed = copy(X)
    changed[1, 2] = -1
    @test warp_key != type_cache_key(changed; restriction=:warp)
    @test warp_key != type_cache_key(X[[3, 1, 2], :]; restriction=:warp)
    @test warp_key != type_cache_key(X[:, [3, 1, 2]]; restriction=:warp)
    @test_throws ArgumentError type_cache_key(Int8[0 0; 1 0])
    @test_throws ArgumentError type_cache_key(X; restriction=:unknown)
end

@testset "synthetic finite end-to-end WARP/RUM separation" begin
    # Hand-written sign fixture, not household data and not the KS baseline.
    # Choosing the first patch on all three budgets creates one 3-cycle.
    # Every other assignment is acyclic; none has a 2-cycle.
    X = Int8[0 1 -1; 0 1 1; -1 0 1; 1 0 1; 1 -1 0; 1 1 0]
    expected = oracle_type_sets(X)
    Aw = enumerate_types(X; restriction=:warp)
    Ar = enumerate_types(X; restriction=:rum)
    @test length(expected.unrestricted) == 8
    @test size(Aw) == (6, 8)
    @test size(Ar) == (6, 7)
    @test binary_type_sets(Aw) == expected.warp
    @test binary_type_sets(Ar) == expected.rum

    target = Float64[1, 0, 1, 0, 1, 0]
    analytical_rum_fit = Float64[.8, .4, .8, .4, .8, .4]
    free_rum = project!(ConeProjector(Ar), target)
    bounded_rum = project!(ConeProjector(Ar; residual_floor=-1.0), target)
    free_warp = project!(ConeProjector(Aw), target)
    bounded_warp = project!(ConeProjector(Aw; residual_floor=-1.0), target)
    @test free_rum.fitted ≈ analytical_rum_fit atol=1e-7
    @test free_rum.squared_distance ≈ .6 atol=1e-7
    @test free_rum.mass_sum ≈ 1.2 atol=1e-7 # A cone, not a normalized simplex.
    @test bounded_rum.fitted ≈ free_rum.fitted atol=1e-7
    @test bounded_rum.squared_distance ≈ free_rum.squared_distance atol=1e-7
    @test free_warp.fitted ≈ target atol=1e-7
    @test free_warp.squared_distance < 1e-12
    @test bounded_warp.squared_distance < 1e-12
    @test bounded_rum.active_residual_floors == 0
    @test bounded_warp.active_residual_floors == 0

    # Distinct hand-crafted probability stages exercise numerical recentering.
    # They are valid probability blocks, not empirical bootstrap evidence.
    variance_stage = repeat(target, 1, 4)
    inference_stage = repeat(target, 1, 4)
    for budget in 1:3
        cycle_row, baseline_row = 2budget-1, 2budget
        variance_stage[cycle_row, :] .-= [.01, .02, .03, .04]
        variance_stage[baseline_row, :] .+= [.01, .02, .03, .04]
        inference_stage[cycle_row, :] .-= [.002, .004, .006, .008]
        inference_stage[baseline_row, :] .+= [.002, .004, .006, .008]
        @test vec(sum(variance_stage[[cycle_row, baseline_row], :]; dims=1)) ≈ ones(4)
        @test vec(sum(inference_stage[[cycle_row, baseline_row], :]; dims=1)) ≈ ones(4)
    end
    @test variance_stage != inference_stage
    @test all(value -> 0 <= value <= 1, variance_stage)
    @test all(value -> 0 <= value <= 1, inference_stage)
    comparison = compare_models(target, X, variance_stage, inference_stage, [100, 100, 100]; degree=3)
    rum, warp = comparison["models"]["rum"], comparison["models"]["warp"]
    @test comparison["sample_scale"] == 100
    @test comparison["tuning"]["variance_denominator"] == 4
    @test comparison["tuning"]["effective_n"] ≈ 8000
    @test comparison["tau"] ≈ sqrt(log(8000)/8000)
    @test rum["J"] ≈ 60 atol=1e-5
    @test warp["J"] <= comparison["zero_tolerance"]
    @test warp["squared_distance"] <= rum["squared_distance"]
    @test rum["p_value"] == 0
    @test warp["p_value"] == 1
    @test rum["tail_count"] == 0
    @test warp["tail_count"] == 4
    @test warp["tightening_origin"] ≈ fill(comparison["tau"]/2, 6)
    @test rum["tightening_origin"] ≈ comparison["tau"] .* [3/7, 4/7, 3/7, 4/7, 3/7, 4/7]
    @test rum["tightening_origin"] != warp["tightening_origin"]
    for result in (rum, warp)
        @test result["observed_active_residual_floors"] == 0
        @test result["bootstraps_with_active_residual_floor"] == 0
        @test all(isfinite, result["bootstrap_statistics"])
    end

    # A second synthetic experiment explicitly checks a nontrivial upper
    # tail: one statistic above J, one equal to J, and two below J. This
    # rejects either a lower-tail p-value or exclusion of exact ties.
    mixed_target = Float64[.75, .25, .75, .25, .75, .25]
    mixed_variance = zeros(6, 4)
    mixed_inference = zeros(6, 4)
    for budget in 1:3
        cycle_row, baseline_row = 2budget-1, 2budget
        mixed_variance[baseline_row, :] = [.24, .25, .26, .27]
        mixed_variance[cycle_row, :] = 1 .- mixed_variance[baseline_row, :]
        mixed_inference[baseline_row, :] = [0, 1/6, .25, .30]
        mixed_inference[cycle_row, :] = 1 .- mixed_inference[baseline_row, :]
    end
    mixed = compare_models(mixed_target, X, mixed_variance, mixed_inference, [3, 3, 3]; degree=3)
    mixed_rum = mixed["models"]["rum"]
    @test mixed_rum["J"] ≈ .1125 atol=1e-8
    @test mixed_rum["bootstrap_statistics"][1] > mixed_rum["J"]
    @test abs(mixed_rum["bootstrap_statistics"][2] - mixed_rum["J"]) <= mixed["zero_tolerance"]
    @test all(mixed_rum["bootstrap_statistics"][3:4] .< mixed_rum["J"])
    @test mixed_rum["tail_count"] == 2
    @test mixed_rum["p_value"] == .5
    @test mixed_rum["p_value_mc_se"] == .25
    @test mixed["models"]["warp"]["p_value"] == 1
end

@testset "literal residual floor and equality-dual KKT sign" begin
    # With one common fitted value, the free solution is 1.25. The bound
    # e>=-1 instead forces the fitted value to .5; its equality dual is not
    # twice the raw residual at the binding coordinate.
    A = ones(Int8, 2, 1)
    target = [-.5, 3.0]
    free = project!(ConeProjector(A), target)
    bounded = project!(ConeProjector(A; residual_floor=-1.0), target)
    @test free.fitted ≈ [1.25, 1.25] atol=1e-7
    @test free.squared_distance ≈ 6.125 atol=1e-7
    @test bounded.fitted ≈ [.5, .5] atol=1e-7
    @test bounded.squared_distance ≈ 7.25 atol=1e-7
    @test bounded.minimum_residual ≈ -1 atol=1e-7
    @test bounded.active_residual_floors == 1
    @test bounded.dual_violation <= 1e-6
    @test_throws DimensionMismatch project!(ConeProjector(A), [1.0])
    @test_throws ArgumentError project!(ConeProjector(A), [NaN, 1.0])
end
