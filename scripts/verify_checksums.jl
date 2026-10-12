#!/usr/bin/env julia
using SHA
function main()
    root = normpath(joinpath(@__DIR__, ".."))
    index = joinpath(root, "SHA256SUMS")
    isfile(index) || error("SHA256SUMS is missing")
    verified = 0
    for line in eachline(index)
        isempty(strip(line)) && continue
        digest, relative = split(line, "  "; limit=2)
        isabspath(relative) && error("absolute path in checksum index")
        any(==(".."), splitpath(relative)) && error("parent traversal in checksum index")
        path = joinpath(root, relative)
        isfile(path) || error("missing file: $relative")
        bytes2hex(open(sha256, path)) == digest || error("checksum mismatch: $relative")
        verified += 1
    end
    println("Verified ", verified, " release file checksums.")
end
main()
