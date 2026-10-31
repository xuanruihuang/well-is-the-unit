"""Input classes and response programs for the routing test (notes/mainline_routing.md; fixed before running).
Input classes: perturbation targets grouped by function, by gene-name pattern (old and new HGNC symbols both accepted).
Programs: response gene sets (outputs). MAIN enter the direction profile; REPORT are computed but kept out of it
(cell-cycle programs carry the composition shift of arrest; MT-encoded transcripts are a local readout).
"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import re
import sys
sys.path.insert(0, (_WS + "/code/a100"))

CLASS_PATTERNS = {
    "cyto_ribo_L": r"^RPL\d+A?$",
    "cyto_ribo_S": r"^(RPS\d+A?|RPSA)$",
    "mito_ribo": r"^(MRPL\d+|MRPS\d+[A-C]?|DAP3|PTCD3|AURKAIP1|GADD45GIP1)$",
    "complex_I": r"^(NDUF(A|B|C|S|V)\d+|NDUFAF\d+|NDUFAB1|NUBPL|TIMMDC1|FOXRED1|ACAD9|ECSIT|TMEM126B)$",
    "complex_III": r"^(UQCR(C1|C2|FS1|B|Q|H|10|11)|CYC1|BCS1L|LYRM7|TTC19|UQCC[123])$",
    "complex_IV": r"^(COX(4I1|5A|5B|6A1|6B1|6C|7A2|7B|7C|8A|10|11|14|15|16|17|18|19|20)|SCO[12]|SURF1|PET100|PET117|COA[3-7])$",
    "complex_V": r"^(ATP5(F1A|F1B|F1C|F1D|F1E|PB|PD|PF|PO|MC1|MC3|ME|MF|MG|MK|A1|B|C1|D|E|O|J|L|I|H|G1|G3|F1)|ATPAF[12]|TMEM70)$",
    "proteasome_core": r"^PSM[AB]\d+$",
    "proteasome_reg": r"^PSM[CD]\d+$",
    "splice_U2": r"^(SF3A\d|SF3B\d|PHF5A|U2AF1|U2AF2)$",
    "snRNP": r"^(SNRP[A-G]\d?|SNRNP\d+|LSM[2-8])$",
    "splice_PRPF_NTC": r"^(PRPF\d+[AB]?|CDC5L|PLRG1|BCAS2|CWC\d+|XAB2|AQR|ISY1|CRNKL1|SNW1)$",
    "pol_I": r"^(POLR1[A-H]|ZNRD1)$",
    "pol_II": r"^POLR2[A-L]$",
    "pol_III": r"^POLR3[A-K]$",
    "mediator": r"^MED\d+$",
    "TFIID": r"^TAF(1|2|3|4|5|6|7|8|9|10|11|12|13)$",
    "eIF3": r"^EIF3[A-M]$",
    "eIF2B": r"^EIF2B[1-5]$",
    "aaRS_cyto": r"^(AARS1?|CARS1?|DARS1?|EPRS1?|FARSA|FARSB|GARS1?|HARS1?|IARS1?|KARS1?|LARS1?|MARS1?|NARS1?|QARS1?|RARS1?|SARS1?|TARS1?|VARS1?|WARS1?|YARS1?)$",
    "CCT": r"^(CCT\d[AB]?|TCP1)$",
    "exosome": r"^(EXOSC\d+|DIS3)$",
    "nuclear_pore": r"^(NUP\d+|SEH1L|SEC13|RAE1|AAAS|NDC1|POM121|TPR)$",
    "replication": r"^(MCM[2-7]|POLA1|POLA2|PRIM1|PRIM2|POLD[1-3]|POLE|POLE2|GINS[1-4]|CDC45|RFC[1-5]|PCNA)$",
    "cohesin_condensin": r"^(SMC1A|SMC3|RAD21|STAG2|SMC2|SMC4|NCAPD2|NCAPG|NCAPH|NCAPD3|NCAPG2|NCAPH2)$",
    "kinetochore": r"^(NDC80|NUF2|SPC24|SPC25|KNL1|ZWINT|DSN1|NSL1|PMF1|MIS12|CENPA|CENPC|CENPE|CENPN|CENPW)$",
    "APC_C": r"^(ANAPC\d+|CDC16|CDC23|CDC26|CDC27)$",
    "ER_translocon_OST": r"^(SEC61A1|SEC61B|SEC61G|SEC62|SEC63|SRP\d+|SRPRA|SRPRB|SSR[1-4]|DDOST|STT3A|STT3B|RPN1|RPN2|DAD1|OSTC|OST4)$",
    "N_glyc_precursor": r"^(DPAGT1|ALG\d+|DOLK|DPM[1-3]|GFPT1|PMM2)$",
    "ERAD_UFM": r"^(SEL1L|SYVN1|DERL[12]|UFL1|UFM1|UBA5|UFC1|DDRGK1|CDK5RAP3)$",
    "EMC": r"^(EMC\d+|MMGT1)$",
    "mTORC1_neg": r"^(TSC1|TSC2|TBC1D7|DEPDC5|NPRL2|NPRL3|SZT2|KPTN|ITFG2|CASTOR1)$",
}
COMPILED = {k: re.compile(v) for k, v in CLASS_PATTERNS.items()}


def classify(targets):
    """class -> list of indices into targets; a target belongs to at most one class (first match in dict order)."""
    out = {k: [] for k in COMPILED}
    for j, t in enumerate(targets):
        for k, r in COMPILED.items():
            if r.match(str(t)):
                out[k].append(j); break
    return out


def programs(hallmark_path=(_WS + "/genesets/MSigDB_Hallmark_2020.txt")):
    from tahoe_arms import ISR, HSF1, NRF2
    from cd4_m3 import P53
    H = {}
    for line in open(hallmark_path):
        parts = [x for x in line.rstrip("\n").split("\t") if x]
        if len(parts) > 1:
            H[parts[0]] = set(parts[1:])
    main = {"ISR_ATF4": set(ISR), "HSF1": set(HSF1), "NRF2": set(NRF2), "p53_core": set(P53),
            "UPR": H["Unfolded Protein Response"], "cholesterol": H["Cholesterol Homeostasis"],
            "mTORC1": H["mTORC1 Signaling"], "MYC_V1": H["Myc Targets V1"], "OXPHOS": H["Oxidative Phosphorylation"],
            "glycolysis": H["Glycolysis"], "hypoxia": H["Hypoxia"], "TNF_NFkB": H["TNF-alpha Signaling via NF-kB"],
            "IFN": H["Interferon Alpha Response"] | H["Interferon Gamma Response"], "apoptosis": H["Apoptosis"]}
    report = {"E2F": H["E2F Targets"], "G2M": H["G2-M Checkpoint"]}
    return main, report
