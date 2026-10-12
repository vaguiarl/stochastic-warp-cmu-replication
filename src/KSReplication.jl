module KSReplication

using LinearAlgebra, Random, SHA, Statistics
using CSV, DataFrames, JSON3, ZipFile, MAT, JuMP, HiGHS, Distributions
import OSQP

include("types.jl")
include("fast_types.jl")
include("geometry.jl")
include("estimation.jl")
include("fast_estimation.jl")
include("household_data.jl")
include("inference.jl")
include("dual_projection.jl")
include("fast_inference.jl")
include("information_projection.jl")
include("information_bootstrap.jl")
include("information_inference.jl")
include("weighted_gain_inference.jl")
include("polytope_confidence_inference.jl")
include("chen_fang_inference.jl")

export enumerate_types, has_two_cycle, has_directed_cycle, type_cache_key
export build_patches, ConeProjector, project!, tuning_parameter, compare_models
export DualConeProjector
export KSAnnual, load_cg_annual, estimate_probabilities, resample_indices
export enumerate_types_fast, compare_models_fast, bootstrap_tail_interval, holm_adjust
export PreparedKSEstimator, prepare_estimator, estimate_prepared_probabilities
export resample_counts, bootstrap_prepared_probabilities, benchmark_prepared_estimator
export FESHouseholdSource, HouseholdGroup, load_fes_source, household_group_registry
export household_group_metadata, prepare_household_group
export adding_up_contrasts, InformationProjector, project_information!
export matrix_log_spd, matrix_exp_symmetric, joint_information_bootstrap
export information_distance_gain, numerical_information_test
export numerical_fixed_weight_test, regularized_information_test
export polytope_confidence_bounds, bootstrap_polytope_confidence_test
export chen_fang_fixed_metric_test

end
