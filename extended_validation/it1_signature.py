"""
IT1 signature definitions, exactly as specified in Methods 2.9 of the manuscript.

The 41-gene IT1 signature is transcribed verbatim from the manuscript text:
  myeloid/monocyte (12) | inflammatory mediator (5) | dendritic-cell (4)
  NK/cytotoxic (10)     | activation (5)            | interferon (5 additional)
Total = 41 unique symbols.
"""

IT1_MODULES = {
    "myeloid_monocyte": ["CD14", "ITGAM", "ITGAX", "FCGR3A", "FCGR1A", "CD68",
                         "CD86", "S100A8", "S100A9", "S100A12", "VCAN", "LYZ"],
    "inflammatory_mediator": ["IL1B", "TNF", "CCL2", "CXCL10", "CXCL9"],
    "dendritic_cell": ["CD1C", "CLEC4C", "IRF8", "IRF7"],
    "nk_cytotoxic": ["NKG7", "GNLY", "GZMB", "GZMA", "GZMK", "PRF1",
                     "KLRD1", "KLRK1", "NCAM1", "FCGR3B"],
    "activation": ["HLA-DRA", "HLA-DRB1", "CD38", "CD69", "CD274"],
    "interferon_extra": ["ISG15", "MX1", "OAS1", "IFNG", "STAT1"],
}

IT1_41 = []
for _m, _g in IT1_MODULES.items():
    for _s in _g:
        if _s not in IT1_41:
            IT1_41.append(_s)
assert len(IT1_41) == 41, len(IT1_41)

# The manuscript's canonical interferon sub-signatures, transcribed from the
# Supp. Fig. S9 caption:
#   type-I-IFN-responsive ISG core (13 genes):
#     "ISG15, MX1, OAS1, IRF7, IFI27, IFI44/44L, RSAD2, OAS2/3, IFIT1/3, STAT2"
#   type-II / IFN-gamma-induced (10 genes):
#     "CXCL9, CXCL10, CXCL11, IRF1, GBP1/5, CIITA, STAT1, IDO1, IFNG"
# The compound entries expand to individual symbols, giving exactly 13 and 10.
ISG_CORE_13 = ["ISG15", "MX1", "OAS1", "IRF7", "IFI27", "IFI44", "IFI44L",
               "RSAD2", "OAS2", "OAS3", "IFIT1", "IFIT3", "STAT2"]
IFN_II_10 = ["CXCL9", "CXCL10", "CXCL11", "IRF1", "GBP1", "GBP5", "CIITA",
             "STAT1", "IDO1", "IFNG"]
assert len(ISG_CORE_13) == 13 and len(IFN_II_10) == 10

# Kept for reference: the interferon genes lying inside the 41-gene signature.
IFN_I_IN_IT1 = ["ISG15", "MX1", "OAS1", "IRF7"]
IFN_II_IN_IT1 = ["IFNG", "CXCL9", "CXCL10"]

# Classical 4-gene clinical interferon score, fully specified in Methods 2.9 (ref [32]).
CLINICAL_IFN4 = ["MX1", "ISG15", "OAS1", "IFI27"]

# Gene-symbol aliases needed for older annotations (microarray / Ensembl gene_name fields).
ALIASES = {
    "NCAM1": ["CD56"],
    "ITGAM": ["CD11B", "CD11b"],
    "ITGAX": ["CD11C", "CD11c"],
    "FCGR3A": ["CD16", "FCGR3"],
    "FCGR3B": ["CD16B"],
    "FCGR1A": ["CD64"],
    "CD274": ["PDL1", "PD-L1", "B7H1"],
    "VCAN": ["CSPG2"],
    "CLEC4C": ["BDCA2", "CD303"],
    "CD1C": ["BDCA1"],
    "GNLY": ["LAG2"],
    "HLA-DRA": ["HLA_DRA", "HLADRA"],
    "HLA-DRB1": ["HLA_DRB1", "HLADRB1"],
    "IFI27": ["ISG12"],
}


def resolve(symbols, available):
    """Map requested symbols onto an available index, using aliases as fallback.

    Returns (found: dict symbol->matched_name, missing: list).
    """
    avail = {str(a).upper(): a for a in available}
    found, missing = {}, []
    for s in symbols:
        key = s.upper()
        if key in avail:
            found[s] = avail[key]
            continue
        hit = None
        for alt in ALIASES.get(s, []):
            if alt.upper() in avail:
                hit = avail[alt.upper()]
                break
        if hit is not None:
            found[s] = hit
        else:
            missing.append(s)
    return found, missing
