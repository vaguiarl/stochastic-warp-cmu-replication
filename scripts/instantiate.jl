#!/usr/bin/env julia
using Pkg
const ROOT = normpath(joinpath(@__DIR__, ".."))
for directory in (ROOT, joinpath(ROOT, "selection_solver_environment"))
    Pkg.activate(directory)
    Pkg.instantiate()
end
Pkg.activate(ROOT)
