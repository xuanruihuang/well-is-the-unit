"""Step 2 (samewell/notes.md §2, written before running): what is the well-level shift?
  plate     2.2a correlation of well-mean vectors between different conditions on the same plate vs different plates
            (0.05 uM plates; also by well distance), with the plate-DMSO reference and with a pooled-DMSO reference
  pooled    2.2b same-well vs other-well evaluation (M1, M3, map) with the pooled-DMSO reference
  position  2.3 layout: are plate-6 / plate-14 conditions in the same well positions; technical differences of the
            6/14 pairs (mean pair difference removed) correlated by well distance; edge vs interior; row / column
  qc        2.4a QC differences vs size of the technical difference; 2.4b QC-regression correction (slopes per line and
            gene on 0.05 uM training conditions; covariates centred within plate) then re-evaluation
  genes     2.5 PCA of technical differences; loadings vs expression level, a depth signature and predefined gene sets
  ambient   2.6 per-line technical difference regressed on (pooled baseline - line baseline); line-specific residuals
            of near-baseline lines vs random lines
Output $WORKSPACE/rt/samewell/step2_<part>.json. Usage: sw_step2.py <part ...|all>"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, collections, itertools
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import sw_core as C

OUT = (_WS + "/rt/samewell/")
IEG = ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "ATF3", "IER2", "IER3", "IER5", "ZFP36", "ZFP36L1",
       "DUSP1", "NR4A1", "NR4A2", "KLF2", "KLF4", "KLF6", "CEBPD", "PPP1R15A", "BTG2", "CCN1", "CCN2", "GADD45B", "RHOB",
       "SOCS3", "NFKBIA", "MYC", "SRF", "ARC", "HES1"]
HSP = ["HSPA1A", "HSPA1B", "HSPA6", "HSPA8", "DNAJB1", "DNAJA1", "HSPH1", "HSPB1", "HSPD1", "HSPE1", "HSP90AA1",
       "HSP90AB1", "BAG3", "CRYAB", "DNAJB4", "HSPA4L", "CHORDC1"]


def wellmean(D):
    fin = np.isfinite(D).all(2)
    tot = np.nansum(np.where(fin[:, :, None], D, 0), 0, dtype=np.float64)
    return tot / fin.sum(0)[:, None]


def pooled_variant(d):
    dm, plates = C.plate_dmso(d)
    pi = {p: k for k, p in enumerate(plates)}
    ref = dm.mean(1)                                   # lines x genes, pooled DMSO (equal weight per plate)
    print("pooled vs cached base max abs diff", float(np.abs(ref - d["base"]).max()), flush=True)
    Dp = d["D"].copy()
    for j, p in enumerate(d["plate"]):
        Dp[:, j] += (dm[:, pi[p]] - ref).astype(np.float32)
    return Dp


def position(d):
    """Well position from sample numbers: samples of a plate are consecutive, row-major A1..H12, DMSO = H11, H12."""
    import pyarrow.parquet as pq
    sm = pq.read_table((_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")).to_pydict()
    first = collections.defaultdict(lambda: 10 ** 9)
    for s_, p in zip(sm["sample"], sm["plate"]):
        first[p] = min(first[p], int(s_[4:]))
    pos = np.array([int(s_[4:]) - first[p] for s_, p in zip(d["sample"], d["plate"])])
    return pos, pos // 12, pos % 12


def part_plate(d, M, tag):
    conc = d["conc"]; plate = d["plate"]; drug = d["drug"]
    pos, row, col = position(d)
    js = np.where(conc == "0.05")[0]
    rs = collections.defaultdict(list)
    rng = np.random.default_rng(0)
    for a, b in itertools.combinations(js, 2):
        if C.norm(drug[a]) == C.norm(drug[b]):
            continue
        if plate[a] == plate[b]:
            dist = max(abs(row[a] - row[b]), abs(col[a] - col[b]))
            k = "same_plate_adjacent" if dist == 1 else "same_plate_far" if dist >= 4 else "same_plate_mid"
            rs[k].append(C.pear(M[a], M[b]))
        elif rng.random() < 0.15:
            rs["different_plate"].append(C.pear(M[a], M[b]))
    out = {k: [round(float(np.median(v)), 3), len(v)] for k, v in rs.items()}
    print(tag, out, flush=True)
    return out


def run_plate(d):
    M = wellmean(d["D"]); Dp = pooled_variant(d); Mp = wellmean(Dp)
    res = {"plate_dmso_ref": part_plate(d, M, "plateref"), "pooled_dmso_ref": part_plate(d, Mp, "pooledref")}
    np.save(OUT + "D_pooled.npy", Dp)
    return res


def run_pooled(d):
    Dp = np.load(OUT + "D_pooled.npy")
    pairs, train = C.groups(d)
    rs = C.evaluate(Dp, d, pairs, train)
    res = {}
    for nm, gs in (("G614", ["G614"]), ("G13+Gother", ["G13", "Gother"]), ("within", ["within"]), ("Ada", ["Ada"])):
        res[nm] = C.summarise([r for r in rs if r["group"] in gs])
        print("pooled", nm, json.dumps(res[nm]), flush=True)
    return res


def tech_diffs(d, M):
    """Technical difference per replicated cross-plate pair (well a - well b); for plate 6/14 oriented 6 - 14 and with the
    mean 6/14 difference (plate-level, incl. the two plates' DMSO references) removed."""
    pairs, _ = C.groups(d)
    out = []
    six = [(c, a, b) if d["plate"][a] == "plate6" else (c, b, a) for c, a, b, g in pairs if g == "G614"]
    D6 = np.array([M[a] - M[b] for c, a, b in six]); mu6 = D6.mean(0)
    for (c, a, b), v in zip(six, D6):
        out.append(dict(cond=c, a=a, b=b, group="G614", d=v - mu6))
    for c, a, b, g in pairs:
        if g in ("G13", "Gother", "Ada", "within"):
            out.append(dict(cond=c, a=a, b=b, group=g, d=M[a] - M[b]))
    return out


def run_position(d):
    M = wellmean(d["D"]); pos, row, col = position(d)
    T = [t for t in tech_diffs(d, M) if t["group"] == "G614"]
    same_pos = float(np.mean([pos[t["a"]] == pos[t["b"]] for t in T]))
    res = {"G614_same_position_frac": round(same_pos, 3), "n": len(T)}
    cats = collections.defaultdict(list)
    for x, y in itertools.combinations(T, 2):
        a, b = x["a"], y["a"]
        dist = max(abs(row[a] - row[b]), abs(col[a] - col[b]))
        k = "adjacent" if dist == 1 else "near(2-3)" if dist <= 3 else "far(>=4)"
        cats[k].append(C.pear(x["d"], y["d"]))
        if row[a] == row[b]:
            cats["same_row"].append(C.pear(x["d"], y["d"]))
        if col[a] == col[b]:
            cats["same_col"].append(C.pear(x["d"], y["d"]))
    res["pair_corr_by_distance"] = {k: [round(float(np.median(v)), 3), round(float(np.mean(v)), 3), len(v)] for k, v in cats.items()}
    nrm = np.array([np.linalg.norm(t["d"]) for t in T]); a = np.array([t["a"] for t in T])
    edge = (row[a] == 0) | (row[a] == 7) | (col[a] == 0) | (col[a] == 11)
    res["norm_edge_vs_interior"] = [round(float(np.median(nrm[edge])), 4), round(float(np.median(nrm[~edge])), 4), int(edge.sum()), int((~edge).sum()),
                                    round(float(mannwhitneyu(nrm[edge], nrm[~edge]).pvalue), 4)]
    # fraction of variance of the PC1 score / the norm explained by row and column (one-way)
    Dm = np.array([t["d"] for t in T]); U, S, Vt = np.linalg.svd(Dm, full_matrices=False); sc = U[:, 0] * S[0]
    def eta2(v, gl):
        gm = {g: v[gl == g].mean() for g in set(gl)}
        return float(sum((v[gl == g].size) * (gm[g] - v.mean()) ** 2 for g in gm) / ((v - v.mean()) ** 2).sum())
    res["eta2_pc1"] = {"row": round(eta2(sc, row[a]), 3), "col": round(eta2(sc, col[a]), 3)}
    res["eta2_norm"] = {"row": round(eta2(nrm, row[a]), 3), "col": round(eta2(nrm, col[a]), 3)}
    # perm null for eta2 (row) of PC1
    rng = np.random.default_rng(1)
    nul = [eta2(sc, rng.permutation(row[a])) for _ in range(500)]
    res["eta2_pc1_row_null95"] = round(float(np.quantile(nul, 0.95)), 3)
    nul = [eta2(sc, rng.permutation(col[a])) for _ in range(500)]
    res["eta2_pc1_col_null95"] = round(float(np.quantile(nul, 0.95)), 3)
    print("position", json.dumps(res), flush=True)
    return res


def qc_covariates(d):
    """Per line x condition: line's mean log UMI per cell, log cells of the line; per condition: well mito fraction,
    log genes per cell; all centred within plate."""
    dep = json.load(open((_WS + "/rt/tahoe/line_depth.json")))
    nL, nC = len(d["lines"]), len(d["drug"])
    X = np.full((nL, nC, 4), np.nan)
    for j, c in enumerate(d["qc"]["conditions"]):
        ds = dep[c["sample"]]
        for i, ln in enumerate(d["lines"]):
            if ln in ds:
                X[i, j, 0] = ds[ln][0]; X[i, j, 1] = np.log(ds[ln][1])
        X[:, j, 2] = c["mito"]; X[:, j, 3] = np.log(c["genes"])
    for p in set(d["plate"]):
        js = np.where(d["plate"] == p)[0]
        X[:, js] -= np.nanmean(X[:, js], 1, keepdims=True)
    return X


def run_qc(d):
    M = wellmean(d["D"]); T = tech_diffs(d, M); X = qc_covariates(d)
    dep = json.load(open((_WS + "/rt/tahoe/line_depth.json")))
    def wq(j):
        c = d["qc"]["conditions"][j]; ds = dep[c["sample"]]; n = np.array([ds.get(l, [0, 0])[1] for l in d["lines"]])
        return np.array([np.log(c["tscp"]), np.log(c["genes"]), c["mito"], np.log(n.sum())]), n / n.sum()
    res = {}
    for grp in (["G614"], ["G13", "Gother"]):
        TT = [t for t in T if t["group"] in grp]
        nrm = np.array([np.linalg.norm(t["d"]) for t in TT])
        Dm = np.array([t["d"] for t in TT]); U, S, Vt = np.linalg.svd(Dm, full_matrices=False)
        qa = np.array([wq(t["a"])[0] - wq(t["b"])[0] for t in TT])
        comp = np.array([np.abs(wq(t["a"])[1] - wq(t["b"])[1]).sum() for t in TT])
        r = {}
        for k, nm in enumerate(("log_umi_per_cell", "log_genes_per_cell", "mito_frac", "log_cells")):
            r[nm] = {"vs_norm(|dQC|)": round(float(spearmanr(np.abs(qa[:, k]), nrm)[0]), 3),
                     "vs_pc1": round(float(spearmanr(qa[:, k], U[:, 0])[0]), 3), "vs_pc2": round(float(spearmanr(qa[:, k], U[:, 1])[0]), 3)}
        r["composition_L1_vs_norm"] = round(float(spearmanr(comp, nrm)[0]), 3)
        r["n"] = len(TT)
        res["+".join(grp)] = r
    print("qc a", json.dumps(res), flush=True)
    # 2.4b correction
    pairs, train = C.groups(d)
    D = d["D"]; Dq = D.copy(); low = np.array([j for j in train if d["conc"][j] == "0.05"])
    r2 = []
    for i in range(len(d["lines"])):
        js = low[np.isfinite(D[i, low]).all(1) & np.isfinite(X[i, low]).all(1)]
        Xi = np.c_[np.ones(len(js)), X[i, js]]
        B = np.linalg.lstsq(Xi, D[i, js].astype(np.float64), rcond=None)[0]
        res_ = D[i, js] - Xi @ B; r2.append(1 - (res_ ** 2).sum(0).sum() / ((D[i, js] - D[i, js].mean(0)) ** 2).sum(0).sum())
        ok = np.isfinite(X[i]).all(1)
        Dq[i, ok] = (D[i, ok] - X[i, ok] @ B[1:]).astype(np.float32)
    res["qc_fit_R2_lowdose_median"] = round(float(np.median(r2)), 3)
    rs = C.evaluate(Dq, d, pairs, train)
    for nm, gs in (("G614", ["G614"]), ("G13+Gother", ["G13", "Gother"])):
        res["corrected_" + nm] = C.summarise([r for r in rs if r["group"] in gs])
        print("qc corrected", nm, json.dumps(res["corrected_" + nm]), flush=True)
    return res


def gene_sets():
    H = {}
    for line in open((_WS + "/genesets/MSigDB_Hallmark_2020.txt")):
        p = [x for x in line.rstrip("\n").split("\t") if x]
        if len(p) > 2:
            H["HALLMARK " + p[0]] = set(p[2:]) if p[1] == "" or not p[1].isupper() else set(p[1:])
    return H


def run_genes(d):
    M = wellmean(d["D"]); T = [t for t in tech_diffs(d, M) if t["group"] in ("G614", "G13", "Gother")]
    genes = np.array(d["genes"]); Dm = np.array([t["d"] for t in T])
    U, S, Vt = np.linalg.svd(Dm, full_matrices=False); ve = S ** 2 / (S ** 2).sum()
    res = {"n_pairs": len(T), "var_explained_top5": [round(float(x), 3) for x in ve[:5]]}
    lvl = np.log1p(np.expm1(d["pooled"]))  # pooled DMSO mean log1p
    # depth signature: slope of well means on centred well log UMI per cell, 0.05 uM training conditions
    pairs, train = C.groups(d)
    low = np.array([j for j in train if d["conc"][j] == "0.05"])
    lt = np.array([np.log(d["qc"]["conditions"][j]["tscp"]) for j in range(len(d["drug"]))])
    for p in set(d["plate"]):
        js = np.where(d["plate"] == p)[0]; lt[js] -= lt[js].mean()
    x = lt[low] - lt[low].mean(); sdep = (x @ (M[low] - M[low].mean(0))) / (x @ x)
    generic = M.mean(0)
    sets = gene_sets(); sets.update({"IEG/dissociation": set(IEG), "heat shock": set(HSP),
                                     "ribosomal RPL/RPS": {g for g in genes if g.startswith(("RPL", "RPS")) and not g.startswith(("RPS6K", "RPL3L"))},
                                     "mito-encoded MT-": {g for g in genes if g.startswith("MT-")}})
    for k in range(3):
        v = Vt[k]; sgn = np.sign(np.corrcoef(v, sdep)[0, 1]) or 1.0; v = v * sgn
        r = {"orient": "sign set so loading correlates positively with depth signature",
             "rho_expression_level": round(float(spearmanr(v, lvl)[0]), 3), "r_depth_signature": round(float(np.corrcoef(v, sdep)[0, 1]), 3),
             "r_generic_drug_response": round(float(np.corrcoef(v, generic)[0, 1]), 3),
             "top_pos": list(genes[np.argsort(-v)[:25]]), "top_neg": list(genes[np.argsort(v)[:25]])}
        tests = []
        for nm, gs in sets.items():
            m = np.isin(genes, list(gs))
            if m.sum() >= 5:
                p = mannwhitneyu(v[m], v[~m]).pvalue
                tests.append((nm, int(m.sum()), float(np.median(v[m]) - np.median(v[~m])), float(p)))
        ps = np.array([t[3] for t in tests]); o = np.argsort(ps); q = np.empty_like(ps)
        q[o] = np.minimum.accumulate((ps[o] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
        r["sets"] = sorted([(t[0], t[1], round(t[2] * 1e3, 3), float("%.2g" % qq)) for t, qq in zip(tests, q) if qq < 0.05], key=lambda z: z[3])[:15]
        r["predefined"] = [(t[0], t[1], round(t[2] * 1e3, 3), float("%.2g" % qq)) for t, qq in zip(tests, q) if t[0] in ("IEG/dissociation", "heat shock", "ribosomal RPL/RPS", "mito-encoded MT-", "HALLMARK Hypoxia")]
        res["PC%d" % (k + 1)] = r
    # how much of each pair's d lies along the depth signature
    u = sdep / np.linalg.norm(sdep)
    res["share_of_d_on_depth_signature_median"] = round(float(np.median([(t["d"] @ u) ** 2 / (t["d"] @ t["d"]) for t in T])), 3)
    res["share_of_d_on_generic_response_median"] = round(float(np.median([(t["d"] @ (generic / np.linalg.norm(generic))) ** 2 / (t["d"] @ t["d"]) for t in T])), 3)
    print("genes", json.dumps(res), flush=True)
    np.savez(OUT + "step2_tech_pcs.npz", Vt=Vt[:5], S=S[:5], genes=genes, sdep=sdep, generic=generic)
    return res


def run_ambient(d):
    D = d["D"]; base = d["base"]; pooled = d["pooled"]; pairs, _ = C.groups(d); M = wellmean(D)
    fin = np.isfinite(D).all(2)
    r_amb, r_shared, r_both = [], [], []
    nn_corr, rnd_corr = [], []
    bz = base - base.mean(1, keepdims=True); bz /= np.linalg.norm(bz, axis=1, keepdims=True); sim = bz @ bz.T; np.fill_diagonal(sim, -np.inf)
    nn1 = np.argmax(sim, 1); rng = np.random.default_rng(2)
    for c, a, b, g in pairs:
        if g not in ("G614", "G13", "Gother"):
            continue
        dd = M[a] - M[b]; res_ = {}
        for t in range(len(d["lines"])):
            if not (fin[t, a] and fin[t, b]):
                continue
            dl = (D[t, a] - D[t, b]).astype(np.float64); amb = pooled - base[t]
            def r2(*cols):
                X = np.c_[np.ones(len(dl)), np.column_stack(cols)]; e = dl - X @ np.linalg.lstsq(X, dl, rcond=None)[0]
                return 1 - (e @ e) / ((dl - dl.mean()) @ (dl - dl.mean()))
            r_amb.append(r2(amb)); r_shared.append(r2(dd)); r_both.append(r2(amb, dd))
            X = np.c_[np.ones(len(dl)), dd]; res_[t] = dl - X @ np.linalg.lstsq(X, dl, rcond=None)[0]
        for t in res_:
            if nn1[t] in res_:
                nn_corr.append(C.pear(res_[t], res_[nn1[t]]))
                o = rng.choice([u for u in res_ if u not in (t, nn1[t])]); rnd_corr.append(C.pear(res_[t], res_[o]))
    res = {"R2_ambient": round(float(np.median(r_amb)), 3), "R2_shared_tech": round(float(np.median(r_shared)), 3),
           "R2_both": round(float(np.median(r_both)), 3),
           "line_specific_residual_corr_nearest_baseline": round(float(np.median(nn_corr)), 3),
           "line_specific_residual_corr_random": round(float(np.median(rnd_corr)), 3), "n": len(r_amb)}
    print("ambient", json.dumps(res), flush=True)
    return res


def main():
    parts = sys.argv[1:] or ["all"]
    if parts == ["all"]:
        parts = ["plate", "pooled", "position", "qc", "genes", "ambient"]
    d = C.load()
    for p in parts:
        res = globals()["run_" + p](d)
        json.dump(res, open(OUT + "step2_%s.json" % p, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
