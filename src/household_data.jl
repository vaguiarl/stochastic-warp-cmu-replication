"""Raw FES inputs, with source household identifiers removed at ingestion.

The private frames support repeated, predeclared household filters. Public group
preparation returns only KSAnnual inputs and aggregate diagnostics, not raw IDs.
"""
struct FESHouseholdSource
    frames::Dict{Int,DataFrame}
    fine_prices::Matrix{Float64}
    provenance::Dict{String,Any}
end

struct HouseholdGroup
    id::String
    label::String
    predicate::Function
    fields::Vector{Symbol}
    notes::Vector{String}
end

const FES_SOURCE_SHA256 = "2a7c2473baf5a9eb68a392e087ca9d3c0499777d72efc7798f9419b18c13fece"
const FES_DEFAULT_ARCHIVE = get(ENV, "FES_DATA_ARCHIVE", joinpath(@__DIR__, "..", "external", "231552-V1.zip"))
const FES_FINE_GOODS = Symbol.(split("BREAD CEREALS BISCUITS BEEF LAMB PORK BACON POUL_OTH FISH BUTTER OIL_FATS CHEESE EGGS MILKFRES MILKPROD TEA COFFEE SOFTDRIN SUG_PRES SWE_CHOC POTATOES OTH_VEGS FRUIT OTH_FOOD CANTEEN OTH_SNAC BEER WINESPIR CIGS OTHTOB RENT MORTGAGE RATESETC REPAIR DIY COAL ELECTRIC GAS OIL_OTH FURNIT FURNISH ELEC_APP OTHHHEQ CONSUMA PETCARE POSTAGE TELEPHON DOMSERVS FEES_SUB MENOUTER WOMOUTER KIDOUTER OTHCLOTH FOOTWEAR CHEMGOOD P_SERVS MOTORVEH MAINTMOT PET_OIL TAX_INS RAILFARE BUSFARES OTHTRAV AUD_VIS REC_TOYS BOOK_NEW GARDEN TVLICEN ENTERTAI"))
# Exact one-based source indices: RUM_21_budgets_abi.m:61–64,77–80.
const FES_CLASS_MAP_3 = [collect(1:26), vcat(27:30,44:45,50:55,64:67),
                       vcat(36:39,46:49,56,58:63,68:69)]
const FES_NONDURABLE_INDICES = vcat(1:30,36:39,44:56,58:69)
const FES_REGION_FLAGS = Symbol.(split("NORTH YORKS NWEST EMIDS WMIDS EANGL GLOND SEAST SWEST WALES SCOTL NIREL"))
const FES_TENURE_FLAGS = [:OOCC,:PUBRENT,:PRRENT]

_fes_number(x) = x isa Real && isfinite(x)
_fes_eq(row,field,value) = _fes_number(row[field]) && row[field] == value
_fes_ge(row,field,value) = _fes_number(row[field]) && row[field] >= value
_fes_range(row,field,lo,hi) = _fes_number(row[field]) && lo <= row[field] < hi
_fes_adults(row) = _fes_number(row.NUMADMAL) && _fes_number(row.NUMADFEM) ?
    row.NUMADMAL + row.NUMADFEM : NaN

function _fes_combine(a::HouseholdGroup,b::HouseholdGroup)
    ap,bp = a.predicate,b.predicate
    return HouseholdGroup(a.id*"__"*b.id,a.label*" AND "*b.label,
        row->ap(row)&&bp(row),unique(vcat(a.fields,b.fields)),unique(vcat(a.notes,b.notes)))
end

"""Fixed exploratory family, declared without reference to fitted outcomes.

All predicates operate AFTER common eligibility: car ≥1, children >0,
positive total expenditure and household income. Categorical codes are labelled
literally where their detailed definitions are not provided by the archive.
This is a bounded set of main effects,all two-way contrasts of the six main
families,and selected exact-two-adult-couple contrasts,not all powersets.
"""
function household_group_registry()
    groups = HouseholdGroup[
        HouseholdGroup("all","All commonly eligible households",row->true,Symbol[],String[]),
        HouseholdGroup("source_couple_onecar","COUPLE == 1 AND NCARS == 1",
            row->_fes_eq(row,:COUPLE,1)&&_fes_eq(row,:NCARS,1),[:COUPLE,:NCARS],
            ["Exact eligibility of source MATLAB driver before annual trimming."])]
    couples = [HouseholdGroup("couple_$v","COUPLE == $v",row->_fes_eq(row,:COUPLE,v),
        [:COUPLE],["COUPLE is a source flag, not a verified marital-status indicator."]) for v in (1,0)]
    adults = [HouseholdGroup("adults_$v","NUMADMAL + NUMADFEM == $v",
        row->_fes_adults(row)==v,[:NUMADMAL,:NUMADFEM],
        ["Source adult age threshold is undocumented in this archive."]) for v in (1,2)]
    push!(adults,HouseholdGroup("adults_3plus","NUMADMAL + NUMADFEM >= 3",
        row->_fes_adults(row)>=3,[:NUMADMAL,:NUMADFEM],
        ["Source adult age threshold is undocumented in this archive."]))
    children = [HouseholdGroup("children_$v","NUMHHKID == $v",row->_fes_eq(row,:NUMHHKID,v),
        [:NUMHHKID],["Source child age threshold is undocumented in this archive."]) for v in (1,2)]
    push!(children,HouseholdGroup("children_3plus","NUMHHKID >= 3",row->_fes_ge(row,:NUMHHKID,3),
        [:NUMHHKID],["Source child age threshold is undocumented in this archive."]))
    cars = [HouseholdGroup("cars_1","NCARS == 1",row->_fes_eq(row,:NCARS,1),[:NCARS],String[]),
            HouseholdGroup("cars_2plus","NCARS >= 2",row->_fes_ge(row,:NCARS,2),[:NCARS],String[])]
    employment = [HouseholdGroup("numempl_$v","NUMEMPL == $v",row->_fes_eq(row,:NUMEMPL,v),
        [:NUMEMPL],["Literal source NUMEMPL count; definitions and special codes need a demographic codebook."]) for v in (0,1)]
    push!(employment,HouseholdGroup("numempl_2plus","NUMEMPL >= 2",row->_fes_ge(row,:NUMEMPL,2),
        [:NUMEMPL],["Literal source NUMEMPL count; definitions and special codes need a demographic codebook."]))
    ages = [HouseholdGroup("headage_18_34","18 <= AGEHEAD < 35",row->_fes_range(row,:AGEHEAD,18,35),
                [:AGEHEAD],["Literal AGEHEAD values; missing/special-code definitions remain unverified."]),
            HouseholdGroup("headage_35_49","35 <= AGEHEAD < 50",row->_fes_range(row,:AGEHEAD,35,50),
                [:AGEHEAD],["Literal AGEHEAD values; missing/special-code definitions remain unverified."]),
            HouseholdGroup("headage_50plus","AGEHEAD >= 50",row->_fes_ge(row,:AGEHEAD,50),
                [:AGEHEAD],["Literal AGEHEAD values; missing/special-code definitions remain unverified."])]
    families = (couples,adults,children,cars,employment,ages)
    for family in families
        append!(groups,family)
    end
    for flag in vcat(FES_REGION_FLAGS,FES_TENURE_FLAGS)
        push!(groups,HouseholdGroup("flag_"*lowercase(string(flag))*"_1","$flag == 1",
            row->_fes_eq(row,flag,1),[flag],["Literal source indicator; detailed definitions are not shipped."]))
    end
    for i in 1:length(families)-1,j in i+1:length(families),a in families[i],b in families[j]
        # Exactly the already-declared source eligibility,not a second test.
        a.id=="couple_1" && b.id=="cars_1" && continue
        push!(groups,_fes_combine(a,b))
    end
    for flag in (:LONE,:EXFAM,:HEADED)
        push!(groups,HouseholdGroup("flag_"*lowercase(string(flag))*"_1","$flag == 1",
            row->_fes_eq(row,flag,1),[flag],["Literal source indicator; demographic code meaning is unverified."]))
    end
    for code in 0:6
        push!(groups,HouseholdGroup("highed_code_$code","HIGHED == $code",
            row->_fes_eq(row,:HIGHED,code),[:HIGHED],
            ["Literal HIGHED code; no education-category label is inferred without a codebook."]))
    end
    two_adult_couple = _fes_combine(first(couples),adults[2])
    for family in (employment,children,cars),b in family
        push!(groups,_fes_combine(two_adult_couple,b))
    end
    length(unique(g.id for g in groups)) == length(groups) || error("Duplicate household group identifiers")
    return groups
end

household_group_metadata(g::HouseholdGroup) = Dict("id"=>g.id,"label"=>g.label,
    "required_fields"=>string.(g.fields),"notes"=>copy(g.notes),
    "selection_stage"=>"common eligibility, then subgroup, then annual source trimming",
    "outcome_based_selection"=>false)

"""Ingest the immutable derivative FES archive once, checking its exact hash.

This reconstructs the source preparation specification; it is not Wilfried's
unavailable replication code and not an original-KS full-precision archive.
Raw household reference/sequence identifiers are deliberately not retained.
"""
function load_fes_source(archive::AbstractString=FES_DEFAULT_ARCHIVE)
    isfile(archive) || throw(ArgumentError("Missing FES source archive"))
    checksum = bytes2hex(open(sha256,archive))
    checksum == FES_SOURCE_SHA256 || throw(ArgumentError("FES archive checksum mismatch"))
    reader = ZipFile.Reader(archive)
    frames = Dict{Int,DataFrame}()
    input_hashes = Dict{String,String}()
    raw_rows = 0
    prices = zeros(0,0)
    driver_hash = ""
    try
        entries = Dict(file.name=>file for file in reader.files)
        pricebytes = read(entries["data/Input/pall.csv"])
        prices = Matrix{Float64}(DataFrame(CSV.File(IOBuffer(pricebytes);types=Float64,strict=true)))
        size(prices)==(69,25) && all(isfinite,prices) && all(>(0),prices) ||
            throw(ArgumentError("Invalid source 69-good,25-year price panel"))
        driver_hash = bytes2hex(sha256(read(entries["data/RUM_21_budgets_abi.m"])))
        schema = String[]
        required = Set(vcat(FES_FINE_GOODS,[:NCARS,:NUMHHKID,:COUPLE,:TOTEXP,:HHINCOME],
            reduce(vcat,[g.fields for g in household_group_registry()])))
        for year in 1975:1999
            name = "data/Input/data$(lpad(string(year%100),2,'0')).csv"
            bytes = read(entries[name])
            input_hashes[name] = bytes2hex(sha256(bytes))
            frame = DataFrame(CSV.File(IOBuffer(bytes);types=Float64,strict=true))
            if year==1975
                schema = names(frame)
            elseif names(frame)!=schema
                throw(ArgumentError("Annual FES source schema changed"))
            end
            isempty(setdiff(required,Set(propertynames(frame)))) ||
                throw(ArgumentError("Required source household fields missing"))
            raw_rows += nrow(frame)
            select!(frame,Not([:HHREF,:HHSEQ]))
            frames[year] = frame
        end
    finally
        close(reader)
    end
    raw_rows == 176308 || throw(ArgumentError("Unexpected source household-year count"))
    bytes2hex(open(sha256,archive)) == checksum || error("FES source archive changed during ingestion")
    provenance = Dict{String,Any}("source_archive"=>abspath(archive),
        "source_archive_sha256"=>checksum,"source_driver"=>"data/RUM_21_budgets_abi.m",
        "source_driver_sha256"=>driver_hash,"input_file_sha256"=>input_hashes,
        "source_years"=>collect(1975:1999),"raw_household_years"=>raw_rows,
        "raw_household_identifiers_retained"=>false,"raw_records_exported"=>false,
        "reconstruction_not_wilfried_code"=>true,"source_integrity_checked"=>true)
    return FESHouseholdSource(frames,prices,provenance)
end

function _fes_common_eligible(row)
    return _fes_ge(row,:NCARS,1) && _fes_number(row.NUMHHKID) && row.NUMHHKID>0 &&
        _fes_number(row.TOTEXP) && row.TOTEXP>0 && _fes_number(row.HHINCOME) && row.HHINCOME>0
end

function _fes_source_trim(frame::DataFrame)
    n = nrow(frame)
    n == 0 && return (frame=frame,income_ceiling=NaN,expenditure_ceiling=NaN)
    # Both ceilings use the same subgroup-eligible annual sample, retaining ties.
    position = ceil(Int,.99*n)
    income_ceiling = sort(Float64.(frame.HHINCOME))[position]
    expenditure_ceiling = sort(Float64.(frame.TOTEXP))[position]
    selected = (frame.HHINCOME .<= income_ceiling) .& (frame.TOTEXP .<= expenditure_ceiling)
    return (;frame=frame[selected,:],income_ceiling,expenditure_ceiling)
end

function _fes_prepare_annual(source::FESHouseholdSource,frame::DataFrame,year::Int)
    fine = Matrix{Float64}(frame[:,FES_FINE_GOODS])
    all(isfinite,fine) || throw(ArgumentError("Nonfinite fine-good source shares"))
    nondurable_share = vec(sum(fine[:,FES_NONDURABLE_INDICES];dims=2))
    all(>(0),nondurable_share) || throw(ArgumentError("Nonpositive nondurable source expenditure"))
    normalized = fine ./ nondurable_share
    shares = hcat([vec(sum(normalized[:,indices];dims=2)) for indices in FES_CLASS_MAP_3]...)
    # Negative fine cells remain unchanged; no aggregate probability/share repair.
    all(>=(0),shares) || throw(ArgumentError("Negative composite shares in source subgroup"))
    maximum(abs.(vec(sum(shares;dims=2)).-1)) <= 1e-10 || error("Composite source partition does not add up")
    expenditure = Float64.(frame.TOTEXP) .* nondurable_share ./ (1 .+ .29 .* Float64.(frame.NUMHHKID))
    instrument = Float64.(frame.HHINCOME) # No equivalization or logarithm of Z.
    median_expenditure = median(expenditure)
    composite_prices = Float64[]
    for indices in FES_CLASS_MAP_3
        weights = vec(mean(normalized[:,indices];dims=1))
        sum(weights)>0 || throw(ArgumentError("No group expenditure support for a composite good"))
        weights ./= sum(weights)
        push!(composite_prices,dot(weights,source.fine_prices[indices,year-1974])/100)
    end
    annual = KSAnnual(year,shares,expenditure,instrument,composite_prices./median_expenditure)
    _validate_annual(annual)
    return (;annual,evaluation_log_expenditure=median(log.(expenditure)),median_expenditure,
        zero_composite_households=count(h->any(==(0),view(shares,h,:)),axes(shares,1)),
        negative_fine_cells=count(<(0),fine),composite_prices)
end

"""Prepare one predeclared group, without estimating its fit or test outcomes.

Default group-specific source preparation changes composite-price weights,
reference expenditure, and potentially patch geometry between groups.
`:common_budgets` instead freezes normalized prices and evaluation log expenditure
from an explicitly supplied prepared broad reference. It does not renormalize
household shares, transform the instrument, or remove boundary-zero purchases.
Unsupported groups return `supported=false` and no partial annual input vector.
"""
function prepare_household_group(source::FESHouseholdSource,group::HouseholdGroup;
        years=1986:1992,budget_mode::Symbol=:group_specific,reference=nothing,
        min_annual_n::Integer=1,require_evaluation_support::Bool=true)
    chosen_years = Int.(collect(years))
    !isempty(chosen_years) && issorted(chosen_years) && allunique(chosen_years) ||
        throw(ArgumentError("Household years must be nonempty,sorted,and distinct"))
    all(haskey(source.frames,y) for y in chosen_years) || throw(ArgumentError("Unavailable source year"))
    min_annual_n>=1 || throw(ArgumentError("Minimum annual sample must be positive"))
    budget_mode in (:group_specific,:common_budgets) || throw(ArgumentError("Unknown household budget mode"))
    refs = Dict{Int,Tuple{KSAnnual,Float64}}()
    if budget_mode==:common_budgets
        reference!==nothing && reference.supported || throw(ArgumentError("A supported broad reference is required"))
        reference.provenance["group"]["id"]=="all" || throw(ArgumentError("Common reference must be the all group"))
        reference.provenance["budget_mode"]=="group_specific" || throw(ArgumentError("Reference budgets must be source prepared"))
        reference.provenance["source_archive_sha256"]==source.provenance["source_archive_sha256"] ||
            throw(ArgumentError("Reference source differs from subgroup source"))
        refs = Dict(a.year=>(a,reference.evaluation_log_expenditure[i]) for (i,a) in enumerate(reference.annual))
        all(haskey(refs,y) for y in chosen_years) || throw(ArgumentError("Reference years incomplete"))
    end
    annual = KSAnnual[]
    evaluation = Float64[]
    diagnostics = Dict{String,Any}[]
    reasons = String[]
    for year in chosen_years
        frame = source.frames[year]
        isempty(setdiff(group.fields,propertynames(frame))) || throw(ArgumentError("Subgroup source field missing"))
        common = [_fes_common_eligible(row) for row in eachrow(frame)]
        subgroup = [common[i] && group.predicate(row) for (i,row) in enumerate(eachrow(frame))]
        pretrim = frame[subgroup,:]
        trimmed = _fes_source_trim(pretrim)
        n = nrow(trimmed.frame)
        d = Dict{String,Any}("year"=>year,"common_eligible_n"=>count(common),
            "subgroup_before_trim_n"=>nrow(pretrim),"n"=>n,
            "income_ceiling"=>isfinite(trimmed.income_ceiling) ? trimmed.income_ceiling : nothing,
            "expenditure_ceiling"=>isfinite(trimmed.expenditure_ceiling) ? trimmed.expenditure_ceiling : nothing)
        push!(diagnostics,d)
        if n<min_annual_n
            push!(reasons,"year $year: annual n=$n below required $min_annual_n")
            continue
        end
        prepared = _fes_prepare_annual(source,trimmed.frame,year)
        a,e = prepared.annual,prepared.evaluation_log_expenditure
        if budget_mode==:common_budgets
            ref,re = refs[year]
            a = KSAnnual(year,a.shares,a.expenditure,a.instrument,copy(ref.normalized_prices))
            e = re
        end
        support = minimum(log.(a.expenditure)) <= e <= maximum(log.(a.expenditure))
        if require_evaluation_support && !support
            push!(reasons,"year $year: common evaluation outside observed group expenditure range")
        end
        merge!(d,Dict("median_group_equivalized_expenditure"=>prepared.median_expenditure,
            "evaluation_log_expenditure"=>e,"evaluation_in_observed_expenditure_range"=>support,
            "zero_composite_households_retained"=>prepared.zero_composite_households,
            "negative_fine_cells_preserved"=>prepared.negative_fine_cells))
        push!(annual,a)
        push!(evaluation,e)
    end
    supported = isempty(reasons)
    provenance = copy(source.provenance)
    merge!(provenance,Dict("group"=>household_group_metadata(group),"years"=>chosen_years,
        "budget_mode"=>string(budget_mode),"goods"=>3,"instrument"=>"raw HHINCOME; not equivalized or logged",
        "trimming"=>"annual ceil(.99*n) order-statistic ceilings on subgroup eligible raw HHINCOME and TOTEXP; retain ties",
        "expenditure"=>"source nondurable share times TOTEXP divided by 1+.29*NUMHHKID",
        "equivalence_caution"=>"Source scale adjusts only children,not differing adult composition",
        "budget_geometry_changes_between_groups"=>budget_mode==:group_specific,
        "common_reference_group"=>budget_mode==:common_budgets ? "all" : nothing,
        "commodity_boundary_zeros_retained"=>true,"hidden_clipping_or_share_repair"=>false,
        "population_stability_not_verified"=>true,"outcome_based_group_selection"=>false))
    support_diagnostics = Dict("supported"=>supported,"reasons"=>reasons,"by_year"=>diagnostics,
        "annual_sample_sizes"=>[d["n"] for d in diagnostics],
        "minimum_annual_n"=>minimum(d["n"] for d in diagnostics),
        "total_household_years"=>sum(d["n"] for d in diagnostics),
        "required_minimum_annual_n"=>min_annual_n,
        "expenditure_range_check_not_full_control_function_support"=>true)
    return (;supported,annual=supported ? annual : KSAnnual[],
        evaluation_log_expenditure=supported ? evaluation : Float64[],provenance,support_diagnostics)
end
