"""M3 (A100): the framework on two new context axes in primary CD4 T cells (Zhu / Marson 2026, GWCD4i.pseudobulk_merged):
the state axis (Rest, Stim8hr, Stim48hr; same donors) and the donor axis (4 donors; same state).
Input rows = summed counts of one guide in one donor x culture condition x 10x run (csr, 18,129 genes).
Effect of target p in context k: log1p CP10K of the pooled pseudobulk of p minus that of the non-targeting guides of
the same context. Two independent estimates per context: the target's guides split in two (guide rank within the
target, even / odd; non-targeting guides likewise). Each sample was sequenced in one 10x run only, so a run split is
not available; with the guide split, guide-specific differences count as noise and are removed by the denoising.
Targets: >= MIN_CELLS cells in every half of every context analysed (P53 targets, proliferation genes and TP53
excluded, as the cell-line pipeline). Genes: control log1p CP10K >= MIN_EXPR in every context; own target zeroed.
For each axis (contexts = 3 states pooled over donors; or 4 donors within each state):
  gamma = two-way interaction over contexts x targets; gamma share of the perturbation-specific signal and gamma signal
  fraction of raw energy (cross-half products, as explore_U); principle one (ideal vs beta-picked neighbours, k = 10,
  mean over contexts); principle two (all ordered pairs; d = x_t - x_s; 80% measured, 5 splits; source-picked / ideal).
Reference values: lane floor <= 0.003 (Z1 c); cell-line pairs 0.30-0.35, four lines 0.46 (Z1 c, e).
Usage: cd4_m3.py <pseudobulk_merged.h5ad> <out_dir>. Exploratory."""
import sys, os, json
import numpy as np, h5py
from scipy.sparse import csr_matrix

MIN_CELLS, MIN_EXPR, K, N_RESP = 20, 0.1, 10, 1000
P53 = ["MDM2", "FDXR", "TP53I3", "RRM2B", "ZMAT3", "BAX", "CDKN1A", "GDF15", "SESN1", "SESN2", "TNFRSF10B", "AEN",
       "PLK3", "TP53INP1", "DDB2", "XPC", "RPS27L", "TRIAP1", "FAS", "BBC3", "PHLDA3", "SPATA18", "POLH", "TIGAR",
       "CCNG1", "APAF1", "EDA2R", "SULF2"]
PRO = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1",
       "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2",
       "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
       "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
       "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B",
       "GTSE1", "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2",
       "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3",
       "GAS2L3", "CBX5", "CENPA"]


def cat(g, k):
    n = g[k]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]])
        return c, n["codes"][:].astype(np.int64)
    v = np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]]); c, codes = np.unique(v, return_inverse=True)
    return c, codes


def wshare(W, RA, RB):
    cov = W.sum(1) > 0
    Wn = W[cov] / W[cov].sum(1, keepdims=True); P_ = Wn @ RA
    T, D = RB[cov], (RA - RB)[cov]
    sig = (T ** 2).sum() - (D ** 2).sum() / 2
    c = (T * P_).sum() / max((P_ ** 2).sum(), 1e-12)
    return float(((T ** 2).sum() - ((T - c * P_) ** 2).sum()) / sig)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def knn(S, k, rows=None, cols=None):
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    if cols is not None:
        m = np.full(S.shape[1], -np.inf); m[cols] = 0; S = S + m[None, :]
    W = np.zeros_like(S); np.put_along_axis(W, np.argsort(-S, 1)[:, :k], 1.0, 1)
    if rows is not None:
        keep = np.zeros(len(S), bool); keep[rows] = True; W[~keep] = 0
    return W


def interaction(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean((0, 1), keepdims=True)


def axis_stats(XA, XB, names):
    gA, gB = interaction(XA), interaction(XB)
    bA, bB = XA.mean(0) - XA.mean((0, 1)), XB.mean(0) - XB.mean((0, 1))
    C = len(names)
    gs, bs = (gA * gB).sum(), (bA * bB).sum() * C
    d = dict(contexts=names, gamma_share_of_specific=float(gs / (gs + bs)),
             gamma_signal_frac_of_raw=float(2 * gs / ((gA ** 2).sum() + (gB ** 2).sum())),
             beta_signal=float(bs / C), gamma_signal=float(gs / C))
    ide = [wshare(knn(unit(gA[c]) @ unit(gA[c]).T, K), gA[c], gB[c]) for c in range(C)]
    byb = [wshare(knn(unit(bA) @ unit(bA).T, K), gA[c], gB[c]) for c in range(C)]
    d["principle_one"] = dict(ideal=[round(x, 4) for x in ide], picked_by_beta=[round(x, 4) for x in byb])
    P = XA.shape[1]; rng = np.random.default_rng(29)
    splits = [np.sort(rng.choice(P, int(0.8 * P), replace=False)) for _ in range(5)]
    info = {}
    for t in range(C):
        for s in range(C):
            if s == t:
                continue
            dA, dB = XA[t] - XA[s], XB[t] - XB[s]
            Ss, Sd = unit(XA[s]) @ unit(XA[s]).T, unit(dA) @ unit(dA).T
            a, b = [], []
            for meas in splits:
                un = np.setdiff1d(np.arange(P), meas)
                a.append(wshare(knn(Ss, K, un, meas), dA, dB)); b.append(wshare(knn(Sd, K, un, meas), dA, dB))
            info["%s->%s" % (names[s], names[t])] = dict(source_picked=float(np.mean(a)), ideal=float(np.mean(b)),
                                                         ratio=float(np.mean(a) / np.mean(b)))
    d["principle_two"] = info
    return d


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with h5py.File(path) as f:
        o = f["obs"]
        donors, dc = cat(o, "donor_id"); conds, cc = cat(o, "culture_condition"); runs, _ = cat(o, "10xrun_id")
        guides, gc = cat(o, "guide_id")
        genes_t, tc = cat(o, "perturbed_gene_name"); gtypes, gtc = cat(o, "guide_type")
        ncell = o["n_cells"][:]
        gname = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_name"][:]])
        X = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=tuple(f["X"].attrs["shape"]))
    print("rows", X.shape, "donors", list(donors), "conditions", list(conds), "runs", list(runs), "guide types", list(gtypes), flush=True)
    ctrl_type = [i for i, g in enumerate(gtypes) if "non" in g.lower()][0]
    isc = gtc == ctrl_type
    D, S, R = len(donors), len(conds), 2
    T = len(genes_t)
    tgt = np.where(isc, T, tc)                                   # control rows -> group T
    # half = rank of the guide within its target (controls: within the non-targeting set), even / odd
    up, pinv = np.unique(tgt * (gc.max() + 1) + gc, return_inverse=True)   # sorted by target, then guide
    ut = up // (gc.max() + 1)
    first = np.searchsorted(ut, ut, side="left")
    rc = ((np.arange(len(up)) - first) % 2)[pinv]
    key = ((tgt * D + dc) * S + cc) * R + rc
    uk, inv = np.unique(key, return_inverse=True)
    Gm = csr_matrix((np.ones(len(key)), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
    agg = (Gm @ X).tocsr()                                       # groups x genes, summed counts
    cells = np.bincount(inv, weights=ncell, minlength=len(uk))
    lib = np.asarray(agg.sum(1)).ravel()
    print("groups", len(uk), flush=True)
    kt = uk // (D * S * R); kd = (uk // (S * R)) % D; ks = (uk // R) % S; kr = uk % R
    pos = {k: i for i, k in enumerate(uk)}
    def row(t, d, s, r):
        return pos.get(((t * D + d) * S + s) * R + r, -1)
    # gene filter on controls
    ctrl_mean = []
    for d in range(D):
        for s in range(S):
            rr = [row(T, d, s, r) for r in range(R) if row(T, d, s, r) >= 0]
            v = np.asarray(agg[rr].sum(0)).ravel(); ctrl_mean.append(np.log1p(v / v.sum() * 1e4))
    gk = np.where(np.min(ctrl_mean, 0) >= MIN_EXPR)[0]
    gpos = {g: k for k, g in enumerate(gname[gk])}
    excl = set(P53) | set(PRO) | {"TP53"}
    print("genes", len(gk), flush=True)
    def effect(t, ctx, r):
        """ctx: list of (donor, state) pooled; returns log1p CP10K diff vs pooled controls of the same contexts, run r."""
        rt = [row(t, d, s, r) for d, s in ctx]; rc_ = [row(T, d, s, r) for d, s in ctx]
        if min(rt) < 0 or min(rc_) < 0 or cells[rt].sum() < MIN_CELLS:
            return None
        a = np.asarray(agg[rt].sum(0)).ravel(); c = np.asarray(agg[rc_].sum(0)).ravel()
        e = (np.log1p(a / a.sum() * 1e4) - np.log1p(c / c.sum() * 1e4))[gk]
        if genes_t[t] in gpos:
            e[gpos[genes_t[t]]] = 0
        return e
    res = {"rules": __doc__, "n_genes": int(len(gk))}
    def build(contexts, names):
        keepT, XA, XB = [], [], []
        for t in range(T):
            if genes_t[t] in excl:
                continue
            ea, eb = [], []
            for ctx in contexts:
                x0, x1 = effect(t, ctx, 0), effect(t, ctx, 1)
                if x0 is None or x1 is None:
                    break
                ea.append(x0); eb.append(x1)
            else:
                keepT.append(t); XA.append(ea); XB.append(eb)
        XA = np.array(XA).transpose(1, 0, 2); XB = np.array(XB).transpose(1, 0, 2)
        # responsive targets: denoised energy of the context-averaged effect, top N_RESP with positive energy
        # (genome-wide screen: most targets barely respond and add only noise; the cell-line sets are mostly
        # responsive essential genes, so this subset is the comparable one)
        e = (XA.mean(0) * XB.mean(0)).sum(1)
        resp = np.argsort(-e)[:N_RESP]; resp = resp[e[resp] > 0]
        print(names, "targets", len(keepT), "responsive", len(resp), flush=True)
        d = {"all": axis_stats(XA, XB, names), "responsive": axis_stats(XA[:, resp], XB[:, resp], names)}
        d["n_targets"], d["n_responsive"] = len(keepT), int(len(resp))
        d["responsive_targets"] = [str(genes_t[keepT[i]]) for i in resp[:50]]
        return d
    # state axis: donors pooled
    res["state_axis"] = build([[(d, s) for d in range(D)] for s in range(S)], list(conds))
    print("state_axis", json.dumps({kk: {k2: v2 for k2, v2 in v.items() if k2 != "principle_two"} if isinstance(v, dict) else v for kk, v in res["state_axis"].items()}), flush=True)
    # donor axis within each state
    for s in range(S):
        k = "donor_axis_%s" % conds[s]
        res[k] = build([[(d, s)] for d in range(D)], list(donors))
        print(k, json.dumps({kk: {k2: v2 for k2, v2 in v.items() if k2 != "principle_two"} if isinstance(v, dict) else v for kk, v in res[k].items()}), flush=True)
    json.dump(res, open(out + "/metrics_cd4_m3.json", "w"), indent=1, default=float)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
