"""Arm-specific wiring on Tahoe-100M (A100): is a line's readout of each surveillance system set by the genotype lesion
on that system's arm? Z1 k found the p53 gain set by TP53 status (rho -0.76). The deepened frame (2026-10-17) predicts
the same for other arms: CDKN2A / RB1 loss -> E2F arm; MYC amplification -> MYC arm; KRAS / PIK3CA / PTEN -> mTORC1 arm.
Also: is the fate choice (elimination vs arrest, tahoe_m5 line_table) set by apoptotic priming read from baseline?
Arrays: tahoe_program.py cache (48 lines, responsive conditions, halves; base = DMSO mean log1p).
Per line c and program h (Hallmark + curated ISR / HSF1 / NRF2 / p53_core, >= 10 genes on the gene axis):
  program gain a[c, h] = sum_{p, g in h} (t_cA ref_B + t_cB ref_A) / 2 / sum ref_A ref_B, t_c = x_c - ref_{-c} (the
    line's deviation from the other lines' mean on that program, denoised; > 0 = reads the program more strongly)
  readout deviation d[c, h] = ||(M_c - mean M)[:, h]||_F / sqrt(|h|), M_c = k = 20 ridge readout (tahoe_readout)
Genotype: driver features gene_mechanism present in >= 3 lines (cell_line_metadata.parquet) + TP53_any; organ one-hot.
Tests: Pearson across lines between a[:, h] (and d[:, h]) and each feature; family-wise null = max |r| over all
(program, feature) pairs under 500 line permutations; the hypothesis pairs reported explicitly; organ eta^2 per program
with the same permutation null. Arm-level prediction: leave-one-line-out kernel ridge from genotype (+ organ) and from
baseline PCs to the program-gain vector a[c, :]; R^2 of the pooled LOO predictions.
Priming: score = mean log1p DMSO expression of pro-apoptotic (BAX, BAK1, BCL2L11, BID, BBC3, PMAIP1, BMF, HRK, BAD,
BIK) minus anti-apoptotic (BCL2, BCL2L1, MCL1, BCL2L2, BCL2A1, BCL2L10) genes; Spearman with the m5 elimination and
arrest propensities, partial on TP53; eliminate vs other fate (Mann-Whitney); single genes MCL1, BCL2L1, BCL2.
Usage: tahoe_arms.py <tahoe_program dir> <tahoe_m5 metrics json> <hallmark txt> <out_dir>. Exploratory."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from tahoe_readout import ridge, pick_lam

META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/cell_line_metadata.parquet")
K, NPERM = 20, 500
ISR = ["ATF4", "DDIT3", "ASNS", "TRIB3", "CHAC1", "PSAT1", "PHGDH", "SLC7A11", "SESN2", "VEGFA", "ATF3", "PPP1R15A", "XPOT",
       "SLC3A2", "SLC7A5", "MTHFD2", "SHMT2", "ALDH18A1", "CEBPG", "CARS1", "WARS1", "YARS1", "GARS1", "SARS1", "AARS1",
       "EPRS1", "MARS1", "IARS1", "LARS1", "NARS1", "TARS1", "VARS1", "CTH", "CBS", "GPT2", "PCK2", "STC2", "HERPUD1",
       "DDIT4", "EIF4EBP1", "JDP2", "NUPR1", "SLC1A5", "SLC38A2", "ASS1", "PYCR1"]
HSF1 = ["HSPA1A", "HSPA1B", "HSPA8", "HSPH1", "HSP90AA1", "HSP90AB1", "DNAJB1", "DNAJA1", "HSPB1", "HSPD1", "HSPE1", "BAG3",
        "CRYAB", "HSPA4", "CHORDC1", "AHSA1", "STIP1", "FKBP4", "DNAJB4", "HSPA6", "DNAJA4", "HSPA4L", "SERPINH1", "CCT2",
        "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8", "TCP1", "PTGES3", "HSPA9", "DNAJB6"]
NRF2 = ["NQO1", "HMOX1", "TXNRD1", "GCLM", "GCLC", "SLC7A11", "SRXN1", "TXN", "PRDX1", "G6PD", "PGD", "TALDO1", "ME1", "FTH1",
        "FTL", "AKR1C1", "AKR1C2", "AKR1B10", "GSR", "ABCC2", "OSGIN1", "SQSTM1", "KEAP1", "GPX2", "MGST1", "NQO2", "GSTA4",
        "UGDH", "PIR", "ABCC3", "TKT", "IDH1", "CBR1", "EPHX1"]
P53 = ["MDM2", "FDXR", "TP53I3", "RRM2B", "ZMAT3", "BAX", "CDKN1A", "GDF15", "SESN1", "SESN2", "TNFRSF10B", "AEN", "PLK3",
       "TP53INP1", "DDB2", "XPC", "RPS27L", "TRIAP1", "FAS", "BBC3", "PHLDA3", "SPATA18", "POLH", "TIGAR", "CCNG1", "APAF1",
       "EDA2R", "SULF2"]
HALL = ["p53 Pathway", "Unfolded Protein Response", "mTORC1 Signaling", "Myc Targets V1", "E2F Targets", "G2-M Checkpoint",
        "Hypoxia", "Reactive Oxygen Species Pathway", "TNF-alpha Signaling via NF-kB", "Interferon Alpha Response",
        "Cholesterol Homeostasis", "Oxidative Phosphorylation", "Glycolysis", "Apoptosis", "DNA Repair", "Protein Secretion",
        "Epithelial Mesenchymal Transition", "Inflammatory Response", "KRAS Signaling Up", "PI3K/AKT/mTOR  Signaling"]
HYP = [("p53_core", "TP53_any"), ("p53 Pathway", "TP53_any"), ("E2F Targets", "CDKN2A_LoF"), ("E2F Targets", "RB1_LoF"),
       ("G2-M Checkpoint", "CDKN2A_LoF"), ("Myc Targets V1", "MYC_GoF"), ("mTORC1 Signaling", "KRAS_GoF"),
       ("mTORC1 Signaling", "PTEN_LoF"), ("mTORC1 Signaling", "PIK3CA_GoF"), ("KRAS Signaling Up", "KRAS_GoF"),
       ("KRAS Signaling Up", "BRAF_GoF"), ("KRAS Signaling Up", "NRAS_GoF")]
PRO = ["BAX", "BAK1", "BCL2L11", "BID", "BBC3", "PMAIP1", "BMF", "HRK", "BAD", "BIK"]
ANTI = ["BCL2", "BCL2L1", "MCL1", "BCL2L2", "BCL2A1", "BCL2L10"]


def kr_loo(F, Y, lam=1.0):
    n = len(F); pred = np.zeros_like(Y)
    for i in range(n):
        m = np.arange(n) != i
        A = F[m] - F[m].mean(0); a = F[i] - F[m].mean(0); Kt = A @ A.T
        alpha = np.linalg.solve(Kt + lam * max(np.trace(Kt) / len(Kt), 1e-12) * np.eye(len(Kt)), Y[m] - Y[m].mean(0))
        pred[i] = Y[m].mean(0) + (A @ a) @ alpha
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main():
    src, m5, hpath, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    z = np.load(src + "/tahoe_arrays.npz", allow_pickle=True)
    resp = z["resp"]; XA = z["XA"][:, resp].astype(np.float64); XB = z["XB"][:, resp].astype(np.float64)
    base = z["base"].astype(np.float64); lines = [str(x) for x in z["lines"]]; genes = [str(g) for g in z["genes"]]
    gpos = {g: j for j, g in enumerate(genes)}; L, P, G = XA.shape
    sA, sB = XA.sum(0), XB.sum(0)
    refA = (sA[None] - XA) / (L - 1); refB = (sB[None] - XB) / (L - 1)
    tA, tB = XA - refA, XB - refB
    H = {}
    for line in open(hpath):
        parts = [x for x in line.rstrip("\n").split("\t") if x]
        if len(parts) > 1:
            H[parts[0]] = set(parts[1:])
    S = {h: H[h] for h in HALL if h in H}; S.update(ISR_ATF4=set(ISR), HSF1_heat_shock=set(HSF1), NRF2_oxidative=set(NRF2), p53_core=set(P53))
    S = {h: np.array(sorted(gpos[g] for g in s if g in gpos)) for h, s in S.items()}
    S = {h: ix for h, ix in S.items() if len(ix) >= 10}; names = list(S)
    # program gains
    a = np.zeros((L, len(names)))
    for j, h in enumerate(names):
        ix = S[h]
        num = ((tA[:, :, ix] * refB[:, :, ix]).sum((1, 2)) + (tB[:, :, ix] * refA[:, :, ix]).sum((1, 2))) / 2
        den = (refA[:, :, ix] * refB[:, :, ix]).sum((1, 2))
        a[:, j] = num / np.maximum(den, 1e-12)
    # readouts and deviation
    rng = np.random.default_rng(0)
    V = np.linalg.svd(sA / L, full_matrices=False)[2][:K].T
    Ms = np.array([ridge(refA[c] @ V, tB[c], pick_lam(refA[c] @ V, tB[c], rng)) for c in range(L)])
    Mbar = Ms.mean(0)
    d = np.array([[np.sqrt(((Ms[c] - Mbar)[:, S[h]] ** 2).sum() / len(S[h])) for h in names] for c in range(L)])
    # genotype
    m = pq.read_table(META).to_pydict(); feats, organ = {}, {}
    for cl, sym, mech, org in zip(m["Cell_ID_Cellosaur"], m["Driver_Gene_Symbol"], m["Driver_Mech_InferDM"], m["Organ"]):
        organ[cl] = org
        if sym and mech:
            feats.setdefault(cl, set()).add("%s_%s" % (sym, mech))
        if sym == "TP53":
            feats.setdefault(cl, set()).add("TP53_any")
    allf = sorted({f for c in lines for f in feats.get(c, ())}); cnt = {f: sum(f in feats.get(c, ()) for c in lines) for f in allf}
    keep = [f for f in allf if cnt[f] >= 3]
    Gm = np.array([[float(f in feats.get(c, ())) for f in keep] for c in lines])
    orgs = sorted({organ.get(c, "NA") for c in lines}); Om = np.array([[float(organ.get(c, "NA") == o) for o in orgs] for c in lines])
    print("programs", len(names), "features", len(keep), "organs", len(orgs), flush=True)
    def corr_mat(Y):
        C = np.zeros((Y.shape[1], len(keep)))
        for j in range(Y.shape[1]):
            for f in range(len(keep)):
                x = Gm[:, f]
                C[j, f] = np.corrcoef(Y[:, j], x)[0, 1] if x.std() > 0 and Y[:, j].std() > 0 else 0.0
        return C
    res = {"rules": __doc__, "programs": names, "features": keep, "feature_counts": {f: cnt[f] for f in keep}, "organs": orgs}
    for nm, Y in (("gain", a), ("deviation", d)):
        C = corr_mat(Y)
        null_max = []
        for i in range(NPERM):
            perm = np.random.default_rng(900 + i).permutation(L)
            null_max.append(np.abs(corr_mat(Y[perm])).max())
        null_max = np.array(null_max)
        fw = lambda r: float((null_max >= abs(r)).mean())
        pairs = sorted(((abs(C[j, f]), names[j], keep[f], float(C[j, f])) for j in range(len(names)) for f in range(len(keep))), reverse=True)
        res[nm] = dict(top_pairs=[(h, f, round(r, 3), fw(r)) for _, h, f, r in pairs[:20]],
                       null_max_q95=float(np.quantile(null_max, 0.95)),
                       hypothesis_pairs=[(h, f, round(float(C[names.index(h), keep.index(f)]), 3), fw(C[names.index(h), keep.index(f)]))
                                         for h, f in HYP if h in names and f in keep])
        # organ eta^2 per program
        def eta(Yj):
            gm = Yj.mean(); ss = ((Yj - gm) ** 2).sum()
            return float(sum(len(Yj[Om[:, o] > 0]) * (Yj[Om[:, o] > 0].mean() - gm) ** 2 for o in range(len(orgs)) if (Om[:, o] > 0).any()) / max(ss, 1e-12))
        et = np.array([eta(Y[:, j]) for j in range(len(names))])
        en = np.array([max(eta(Y[np.random.default_rng(700 + i).permutation(L), j]) for j in range(len(names))) for i in range(200)])
        res[nm]["organ_eta2"] = {names[j]: (round(float(et[j]), 3), float((en >= et[j]).mean())) for j in np.argsort(-et)[:8]}
        # arm-level prediction
        relb = base - base.mean(0); Vt = np.linalg.svd(relb, full_matrices=False)[2][:10]; Pc = relb @ Vt.T
        Pc = Pc / np.maximum(Pc.std(0), 1e-12)
        res[nm]["loo_R2"] = dict(genotype=kr_loo(Gm, Y), organ=kr_loo(Om, Y), geno_organ=kr_loo(np.hstack([Gm, Om]), Y),
                                 baseline=kr_loo(Pc, Y), all=kr_loo(np.hstack([Pc, Gm, Om]), Y),
                                 null_genotype=[kr_loo(Gm[np.random.default_rng(80 + i).permutation(L)], Y) for i in range(5)])
        print(nm, "top", res[nm]["top_pairs"][:8], flush=True)
        print(nm, "hyp", res[nm]["hypothesis_pairs"], flush=True)
        print(nm, "loo", {k: (round(v, 3) if not isinstance(v, list) else [round(x, 3) for x in v]) for k, v in res[nm]["loo_R2"].items()}, flush=True)
        print(nm, "organ", res[nm]["organ_eta2"], flush=True)
    # priming
    lt = json.load(open(m5))["line_table"]
    pro = [gpos[g] for g in PRO if g in gpos]; anti = [gpos[g] for g in ANTI if g in gpos]
    prime = base[:, pro].mean(1) - base[:, anti].mean(1)
    ok = [c for c in range(L) if lines[c] in lt]
    el = np.array([lt[lines[c]]["elimination"] for c in ok]); ar = np.array([lt[lines[c]]["arrest"] for c in ok])
    tp = np.array([lt[lines[c]]["tp53_mut"] for c in ok]); fate = np.array([lt[lines[c]]["fate"] for c in ok]); pr = prime[ok]
    def partial(x, y, zc):
        rx = x - np.polyval(np.polyfit(zc, x, 1), zc); ry = y - np.polyval(np.polyfit(zc, y, 1), zc)
        return float(spearmanr(rx, ry)[0])
    prim = dict(n=len(ok), genes_pro=[genes[j] for j in pro], genes_anti=[genes[j] for j in anti],
                rho_prime_elimination=float(spearmanr(pr, el)[0]), rho_prime_arrest=float(spearmanr(pr, ar)[0]),
                partial_prime_elimination_given_tp53=partial(pr, el, tp),
                rho_prime_p53gain=float(spearmanr(pr, [lt[lines[c]]["p53_gain"] for c in ok])[0]),
                fate_counts={f: int((fate == f).sum()) for f in set(fate)},
                prime_by_fate={f: float(np.median(pr[fate == f])) for f in set(fate)})
    if (fate == "eliminate").sum() >= 3:
        prim["mw_eliminate_vs_other_p"] = float(mannwhitneyu(pr[fate == "eliminate"], pr[fate != "eliminate"], alternative="greater")[1])
    for g in ("MCL1", "BCL2L1", "BCL2", "BAX", "BCL2L11", "BBC3"):
        if g in gpos:
            prim["rho_%s_elimination" % g] = float(spearmanr(base[ok, gpos[g]], el)[0])
    res["priming"] = prim
    print("priming", json.dumps(prim, default=lambda v: round(float(v), 3)), flush=True)
    np.savez(out + "/program_gains.npz", gain=a, deviation=d, programs=np.array(names), lines=np.array(lines), G=Gm, features=np.array(keep))
    json.dump(res, open(out + "/metrics_tahoe_arms.json", "w"), indent=1, default=float)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
