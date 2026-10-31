"""Routing test §19 (notes/mainline_routing.md, written before running), A100 parts.
  depth    T8: mean log(transcripts per cell) per (Tahoe sample, cell line) from obs_metadata -> rt/tahoe/line_depth.json
  ntguide  T6: CD4 non-targeting guides one by one: program profile per guide and donor half relative to the mean of the
           other NT guides; cross-half product; offset of inert targets against all NT guides and against 'clean' NT
           guides (cross-half product below the threshold set by the most negative products); NT-half disagreement
           for all and clean NT guides                                                    -> rt/cd4_ntguide.json
  k562gw   T7: K562 genome-wide day-8 screen, all perturbations with >= 20 cells: program scores (random halves,
           fold scale) and the offset of targets absent from the screen's gene table or with control CP10K < 0.05
           and outside every class                                                        -> rt/k562_gw_offset.json
Usage: rt_tech_a100.py <depth|ntguide|k562gw>. Exploratory."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, collections
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "16")
import numpy as np
sys.path.insert(0, (_WS + "/code")); sys.path.insert(0, (_WS + "/code/a100")); sys.path.insert(0, (_WS + "/rt"))
RT = (_WS + "/rt")


def depth():
    import pyarrow.parquet as pq
    f = pq.ParquetFile((_WS + "/data_ext/tahoe/Tahoe-100M/metadata/obs_metadata.parquet"))
    acc = collections.defaultdict(lambda: [0.0, 0])
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=["sample", "cell_line", "tscp_count", "pass_filter"]).to_pandas()
        t = t[t["pass_filter"].astype(str) == "full"] if "full" in set(t["pass_filter"].astype(str)) else t
        t["lt"] = np.log(t["tscp_count"].astype(float))
        g = t.groupby(["sample", t["cell_line"].astype(str)])["lt"].agg(["sum", "count"])
        for (s, c), row in g.iterrows():
            a = acc[(s, c)]; a[0] += row["sum"]; a[1] += row["count"]
        if rg % 50 == 0:
            print("row group", rg, f.num_row_groups, flush=True)
    out = collections.defaultdict(dict)
    for (s, c), (sm, n) in acc.items():
        out[s][c] = [sm / n, n]
    json.dump(out, open(RT + "/tahoe/line_depth.json", "w"))
    print("saved samples", len(out), flush=True)


def ntguide():
    import h5py
    from scipy.sparse import csr_matrix
    from cd4_m3 import cat
    from rt_extract import prog_sets, MIN_GENES
    from rt_classes import classify
    with h5py.File((_WS + "/data_ext/cd4/GWCD4i.pseudobulk_merged.h5ad")) as f:
        o = f["obs"]
        donors, dc = cat(o, "donor_id"); conds, cc = cat(o, "culture_condition"); genes_t, tc = cat(o, "perturbed_gene_name")
        gtypes, gtc = cat(o, "guide_type"); gids, gic = cat(o, "guide_id")
        gname = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_name"][:]])
        X = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=tuple(f["X"].attrs["shape"]))
    isc = gtc == [i for i, g in enumerate(gtypes) if "non" in g.lower()][0]; dh = (dc >= 2).astype(int)
    inert = json.load(open(RT + "/inert_targets.json")); tn = np.array([str(t) for t in genes_t])
    res = {}
    for s in range(len(conds)):
        name = "CD4_%s" % conds[s]
        cls = {j for v in classify(sorted(set(inert[name]["inert"]))).values() for j in v}
        iset = {g for k, g in enumerate(sorted(set(inert[name]["inert"]))) if k not in cls}
        prof = lambda r: np.asarray(X[r].sum(0)).ravel()
        ntr = np.where(isc & (cc == s))[0]; guides = sorted(set(gic[ntr].tolist()))
        # per guide and donor half
        G = {}
        for g in guides:
            G[g] = [prof(ntr[(gic[ntr] == g) & (dh[ntr] == h)]) for h in range(2)]
        tot = [sum(G[g][h] for g in guides) for h in range(2)]
        cp = sum(t * 1e4 / t.sum() for t in tot) / 2; P, _ = prog_sets(gname, cp); sv = cp / (1 + cp)
        keepP = [nm for nm, ix in P.items() if len(ix) >= MIN_GENES and nm in ("ISR_ATF4", "HSF1", "NRF2", "UPR", "cholesterol", "mTORC1", "MYC_V1", "OXPHOS", "glycolysis", "hypoxia", "TNF_NFkB", "IFN", "apoptosis")]
        def score(x, ref):
            F = (np.log1p(x * 1e4 / x.sum()) - np.log1p(ref * 1e4 / ref.sum())) / np.maximum(sv, 1e-12)
            return np.array([F[P[nm]].mean() for nm in keepP])
        prods, devs = {}, {}
        for g in guides:
            if min(G[g][h].sum() for h in range(2)) < 2e4:
                continue
            d = [score(G[g][h], tot[h] - G[g][h]) for h in range(2)]
            prods[g] = float(d[0] @ d[1]); devs[g] = d
        pv = np.array(list(prods.values())); thr = np.percentile(-pv[pv < 0], 90) if (pv < 0).any() else 0.0
        active = [g for g, p in prods.items() if p > thr]
        clean = [g for g in guides if g not in set(active)]
        ir = np.where(~isc & (cc == s) & np.isin(tn[tc], list(iset)))[0]
        I = [prof(ir[dh[ir] == h]) for h in range(2)]
        def offset(gs):
            ref = [sum(G[g][h] for g in gs) for h in range(2)]
            return [score(I[h], ref[h]) for h in range(2)]
        Oall, Ocl = offset(guides), offset(clean)
        rng = np.random.default_rng(0)
        def halves_disagree(gs):
            gs = list(rng.permutation(gs)); a, b = gs[::2], gs[1::2]
            ra = sum(G[g][0] + G[g][1] for g in a); rb = sum(G[g][0] + G[g][1] for g in b); Ii = I[0] + I[1]
            oa, ob = score(Ii, ra), score(Ii, rb)
            return float(np.linalg.norm(score(ra, rb)) / np.linalg.norm((oa + ob) / 2)), float(oa @ ob / (np.linalg.norm(oa) * np.linalg.norm(ob)))
        res[name] = dict(n_guides=len(guides), n_scored=len(prods), threshold=thr, n_pos_extreme=int((pv > thr).sum()), n_neg_extreme=int((pv < -thr).sum()),
                         estimated_active_fraction=round(float(((pv > thr).sum() - (pv < -thr).sum()) / len(pv)), 3),
                         offset_norm_all=float(np.linalg.norm(np.mean(Oall, 0))), offset_norm_clean=float(np.linalg.norm(np.mean(Ocl, 0))),
                         offset_cos_all_vs_clean=float(np.mean(Oall, 0) @ np.mean(Ocl, 0) / (np.linalg.norm(np.mean(Oall, 0)) * np.linalg.norm(np.mean(Ocl, 0)))),
                         nt_halves_all=halves_disagree(guides), nt_halves_clean=halves_disagree(clean), programs=keepP,
                         offset_all={p: round(float(x), 4) for p, x in zip(keepP, np.mean(Oall, 0))}, offset_clean={p: round(float(x), 4) for p, x in zip(keepP, np.mean(Ocl, 0))})
        print(name, json.dumps({k: v for k, v in res[name].items() if k not in ("offset_all", "offset_clean", "programs")}), flush=True)
    json.dump(res, open(RT + "/cd4_ntguide.json", "w"), indent=1)


def k562gw():
    from explore_V_day6_day8 import read_rows, Line, load
    from replogle_indep_readout import effects, obs_info
    from rt_extract import prog_sets, union_cols, score
    from rt_classes import classify
    p = (_WS + "/replogle/K562_gwps_raw_singlecell_01.h5ad")
    codes, cats, gem, _ = obs_info(p); n = np.bincount(codes, minlength=len(cats))
    perts = [c for c, k in zip(cats, n) if k >= 20 and c not in ("non-targeting", "control", "CTRL")]
    d = load(p, set(perts)); genes = [str(g) for g in d["genes"]]
    X = read_rows(p, d["rows"], np.arange(len(genes)))
    L = Line(X, d["codes"][d["rows"]], d["cats"], genes, d["gem"][d["rows"]], d["ctrl"]); del X
    keep = [t for t in perts if L.n_cells(t) >= 20]
    H, Fu, mu = effects(L, keep, 0); cp = np.expm1(mu)
    P, gpos = prog_sets(genes, cp); cols = union_cols(P)
    sc = np.stack([score(np.asarray(H[h])[:, cols], cols, keep, gpos, cp, P) for h in range(2)])
    names = list(P)[:14]; main = [k for k, nm in enumerate(P) if nm in ("ISR_ATF4", "HSF1", "NRF2", "UPR", "cholesterol", "mTORC1", "MYC_V1", "OXPHOS", "glycolysis", "hypoxia", "TNF_NFkB", "IFN", "apoptosis") and len(P[nm]) >= 8]
    cl = classify(keep); incl = {j for v in cl.values() if len(v) >= 4 for j in v}
    inert = [j for j, t in enumerate(keep) if j not in incl and (t not in gpos or cp[gpos[t]] < 0.05)]
    S = sc[:, :, main]; O = np.nanmean(S[:, inert], 1)
    cn = np.median([np.linalg.norm(np.nanmean(S[:, v], 1).mean(0)) for v in cl.values() if len(v) >= 4])
    out = dict(n_perts=len(keep), n_inert=len(inert), programs=[list(P)[k] for k in main], offset=[round(float(x), 4) for x in O.mean(0)],
               offset_norm=float(np.linalg.norm(O.mean(0))), median_class_norm=float(cn), offset_over_class=float(np.linalg.norm(O.mean(0)) / cn),
               offset_half_cos=float(O[0] @ O[1] / (np.linalg.norm(O[0]) * np.linalg.norm(O[1]))))
    json.dump(out, open(RT + "/k562_gw_offset.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    {"depth": depth, "ntguide": ntguide, "k562gw": k562gw}[sys.argv[1]]()
