"""Routing test, step 1 (notes/mainline_routing.md): per context, per target, per replicate, fold-scale program scores.
Score of program p for target t in replicate h = mean over the program's genes expressed in the context (control
CP10K >= 0.5) of (log1p effect / s), s = c / (1 + c), c = control CP10K; the target's own gene is left out.
Contexts and replicates:
  replogle   K562 = day-6 screen vs day-8 screen (independent); K562_d6_rand, K562_d8_rand, RPE1 = random cell halves
             (the 1,820 perturbations of replogle_indep_readout; per-lane control correction as explore_V Line)
  orion      HCT116, HEK293T: random cell halves (orion_agg)
  pan        hESC: guide-parity halves (pan_agg_guide; non-targeting controls split by the same parity)
  cd4        Rest / Stim8hr / Stim48hr: donor halves {first two donors} vs {last two} (4-donor pseudobulk, summed counts)
Output: $WORKSPACE/rt/ctx/<context>.npz with targets, programs (main then report), scores (2, T, P), ncells (T, 2),
n_genes (P), unit. Usage: rt_extract.py <replogle|orion|pan|cd4|all>. Exploratory."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "16")
import numpy as np
sys.path.insert(0, (_WS + "/code")); sys.path.insert(0, (_WS + "/code/a100")); sys.path.insert(0, (_WS + "/rt"))
from rt_classes import programs

OUT = os.environ.get("RT_CTX_OUT", (_WS + "/rt/ctx"))
NOSCALE = bool(os.environ.get("RT_NOSCALE"))          # §19 T5: delta log1p, no fold-scale division
MIN_CP10K, MIN_GENES, MIN_GENES_MT, MIN_CELLS = 0.5, 8, 5, 10


def prog_sets(genes, cp10k):
    main, report = programs()
    gpos = {}
    for j, g in enumerate(genes):
        gpos.setdefault(str(g), j)
    expd = cp10k >= MIN_CP10K
    P = {}
    for name, st in list(main.items()) + list(report.items()):
        P[name] = np.array(sorted({gpos[g] for g in st if g in gpos and expd[gpos[g]]}), dtype=int)
    P["MT_local"] = np.array([j for j, g in enumerate(genes) if str(g).startswith("MT-") and expd[j]], dtype=int)
    return P, gpos


def union_cols(P):
    return np.array(sorted(set().union(*[set(v.tolist()) for v in P.values()])), dtype=int)


def score(effU, cols, targets, gpos, cp10k, P):
    """effU: (T, len(cols)) log1p effects on the union columns; returns (T, nP)"""
    cpos = {c: k for k, c in enumerate(cols)}
    s = cp10k[cols] / (1 + cp10k[cols])
    F = effU if NOSCALE else effU / s[None, :]
    own = np.array([cpos.get(gpos.get(str(t), -1), -1) for t in targets])
    out = np.full((len(targets), len(P)), np.nan)
    for k, (nm, ix) in enumerate(P.items()):
        need = MIN_GENES_MT if nm == "MT_local" else MIN_GENES
        if len(ix) < need:
            continue
        loc = np.array([cpos[c] for c in ix])
        sub = F[:, loc].copy()
        m = loc[None, :] == own[:, None]
        sub[m] = np.nan
        out[:, k] = np.nanmean(sub, 1)
    return out


def save(name, targets, P, scores, ncells, unit):
    os.makedirs(OUT, exist_ok=True)
    np.savez(OUT + "/%s.npz" % name, targets=np.array([str(t) for t in targets]), programs=np.array(list(P)),
             scores=scores.astype(np.float32), ncells=np.asarray(ncells, float), n_genes=np.array([len(v) for v in P.values()]), unit=unit)
    print("saved", name, scores.shape, "programs with genes", {k: len(v) for k, v in P.items()}, flush=True)


def replogle():
    from explore_V_day6_day8 import read_rows, Line, load
    from replogle_indep_readout import effects
    FILES = {"K6": (_WS + "/data_ext/replogle/K562_essential_raw_singlecell_01.h5ad"),
             "K8": (_WS + "/replogle/K562_gwps_raw_singlecell_01.h5ad"),
             "R": (_WS + "/data_ext/replogle/rpe1_raw_singlecell_01.h5ad")}
    perts = [str(p) for p in np.load((_WS + "/replogle_indep_readout/responses.npz"))["perts"]]
    D = {nm: load(p, set(perts)) for nm, p in FILES.items()}
    common = sorted(set.intersection(*[set(str(g) for g in d["genes"]) for d in D.values()]))
    E = {}
    for seed, (nm, p) in enumerate(FILES.items()):
        d = D[nm]; gidx = {str(g): j for j, g in enumerate(d["genes"])}
        X = read_rows(p, d["rows"], np.array([gidx[g] for g in common]))
        L = Line(X, d["codes"][d["rows"]], d["cats"], common, d["gem"][d["rows"]], d["ctrl"]); del X
        H, Fu, mu = effects(L, perts, seed)
        nc = np.array([L.n_cells(t) for t in perts], float)
        E[nm] = (H, Fu, np.expm1(mu), nc); del L
        print(nm, "effects done", flush=True)
    # K562 independent: day 6 vs day 8, programs on genes expressed in both screens
    cpK = np.minimum(E["K6"][2], E["K8"][2])
    P, gpos = prog_sets(common, cpK); cols = union_cols(P)
    sc = np.stack([score(E[nm][1][:, cols], cols, perts, gpos, E[nm][2], P) for nm in ("K6", "K8")])
    save("K562", perts, P, sc, np.stack([E["K6"][3], E["K8"][3]], 1), "independent screens (day 6 vs day 8)")
    for nm, lab in (("K6", "K562_d6_rand"), ("K8", "K562_d8_rand"), ("R", "RPE1")):
        P, gpos = prog_sets(common, E[nm][2]); cols = union_cols(P)
        H = E[nm][0]
        sc = np.stack([score(H[h][:, cols], cols, perts, gpos, E[nm][2], P) for h in range(2)])
        save(lab, perts, P, sc, np.stack([E[nm][3] / 2] * 2, 1), "random cell halves")


def orion():
    agg = (_WS + "/orion_agg")
    idx = json.load(open(agg + "/index.json")); lines, targets, genes = idx["lines"], idx["targets"], idx["genes"]
    T = len(targets); row = lambda L, t, h: (L * T + t) * 2 + h; ct = targets.index(idx["ctrl"])
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy"); n = cnt[:, 0].reshape(2, T, 2)
    for L, ln in enumerate(lines):
        ctrl = [np.asarray(S[row(L, ct, h)]) / cnt[row(L, ct, h), 0] for h in range(2)]
        cp = np.expm1((ctrl[0] + ctrl[1]) / 2)
        P, gpos = prog_sets(genes, cp); cols = union_cols(P)
        keep = [t for t in range(T) if t != ct and n[L, t].min() >= MIN_CELLS]
        sc = []
        for h in range(2):
            rows = [row(L, t, h) for t in keep]
            eff = np.asarray(S[rows][:, cols]) / cnt[rows, 0][:, None] - ctrl[h][cols][None, :]
            sc.append(score(eff, cols, [targets[t] for t in keep], gpos, cp, P))
        save(ln, [targets[t] for t in keep], P, np.stack(sc), n[L, keep], "random cell halves")


def pan():
    agg = (_WS + "/pan_agg_guide")
    idx = json.load(open(agg + "/index.json")); targets, genes = idx["targets"], idx["genes"]
    T = len(targets); ct = targets.index(idx["ctrl"])
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy")
    rows = lambda t, p: [t * 4 + p * 2 + h for h in range(2)]
    npar = np.array([[cnt[rows(t, p), 0].sum() for p in range(2)] for t in range(T)])
    ctrl = [np.asarray(S[rows(ct, p)]).sum(0) / npar[ct, p] for p in range(2)]
    cp = np.expm1((ctrl[0] + ctrl[1]) / 2)
    P, gpos = prog_sets(genes, cp); cols = union_cols(P)
    keep = [t for t in range(T) if t != ct and npar[t].min() >= MIN_CELLS]
    sc = []
    for p in range(2):
        eff = np.array([np.asarray(S[rows(t, p)])[:, cols].sum(0) / npar[t, p] for t in keep]) - ctrl[p][cols][None, :]
        sc.append(score(eff, cols, [targets[t] for t in keep], gpos, cp, P))
    save("hESC", [targets[t] for t in keep], P, np.stack(sc), npar[keep], "guide-parity halves")


def cd4():
    import h5py
    from scipy.sparse import csr_matrix
    from cd4_m3 import cat
    with h5py.File((_WS + "/data_ext/cd4/GWCD4i.pseudobulk_merged.h5ad")) as f:
        o = f["obs"]
        donors, dc = cat(o, "donor_id"); conds, cc = cat(o, "culture_condition")
        genes_t, tc = cat(o, "perturbed_gene_name"); gtypes, gtc = cat(o, "guide_type")
        ncell = o["n_cells"][:]
        gname = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_name"][:]])
        X = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=tuple(f["X"].attrs["shape"]))
    isc = gtc == [i for i, g in enumerate(gtypes) if "non" in g.lower()][0]
    T = len(genes_t); S_ = len(conds)
    tgt = np.where(isc, T, tc); dh = (dc >= 2).astype(int)          # donors sorted: first two vs last two
    key = (tgt * 2 + dh) * S_ + cc
    uk, inv = np.unique(key, return_inverse=True)
    Gm = csr_matrix((np.ones(len(key)), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
    agg = (Gm @ X).tocsr(); cells = np.bincount(inv, weights=ncell, minlength=len(uk)); lib = np.asarray(agg.sum(1)).ravel()
    pos = {k: i for i, k in enumerate(uk)}
    tnames = [str(t) for t in genes_t]
    for s in range(S_):
        rc = [pos[(T * 2 + h) * S_ + s] for h in range(2)]
        cvec = [np.log1p(np.asarray(agg[i].todense()).ravel() * 1e4 / lib[i]) for i in rc]
        cp = np.expm1((cvec[0] + cvec[1]) / 2)
        P, gpos = prog_sets(gname, cp); cols = union_cols(P)
        keep = [t for t in range(T) if all(((t * 2 + h) * S_ + s) in pos for h in range(2))
                and min(cells[pos[(t * 2 + h) * S_ + s]] for h in range(2)) >= MIN_CELLS]
        sc, nc = [], []
        for h in range(2):
            ri = [pos[(t * 2 + h) * S_ + s] for t in keep]
            A = agg[ri][:, cols].toarray() * 1e4 / lib[ri][:, None]
            eff = np.log1p(A) - cvec[h][cols][None, :]
            sc.append(score(eff, cols, [tnames[t] for t in keep], gpos, cp, P)); nc.append(cells[ri])
        save("CD4_%s" % conds[s], [tnames[t] for t in keep], P, np.stack(sc), np.stack(nc, 1), "donor halves (2 + 2 donors)")


if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time()
    for m in (["orion", "pan", "cd4", "replogle"] if mode == "all" else [mode]):
        {"replogle": replogle, "orion": orion, "pan": pan, "cd4": cd4}[m]()
        print(m, "done %.0fs" % (time.time() - t0), flush=True)
