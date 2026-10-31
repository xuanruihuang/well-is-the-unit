"""Routing test §12 (notes/mainline_routing.md, written before running): inputs for the offset-source tests.
  kd      per CRISPR context and target: residual expression of the target's own gene (fold scale, 1 + effect / s,
          s = c / (1 + c); CD4 from summed pseudobulk CP10K) and its control CP10K           -> rt/kd_residual.json
  cd4nt   CD4 non-targeting controls split by guide_id into two halves; program scores (fold scale, donors pooled)
          of pooled inert targets vs NT half A, vs NT half B, and of NT half A vs half B     -> rt/cd4_nt_halves.json
  tahoe   per Tahoe condition (order of tahoe_arrays conds): sample id, plate, mean transcripts / genes per cell,
          percent mito, and the same for the plate's DMSO samples                          -> rt/tahoe/sample_qc.json
Usage: rt_offset_source.py <kd|cd4nt|tahoe|all>. Exploratory."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, ast
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "16")
import numpy as np
sys.path.insert(0, (_WS + "/code")); sys.path.insert(0, (_WS + "/code/a100")); sys.path.insert(0, (_WS + "/rt"))
from rt_extract import prog_sets, MIN_GENES
from rt_classes import classify

RT = (_WS + "/rt")


def kd():
    out = {}
    # orion
    agg = (_WS + "/orion_agg"); idx = json.load(open(agg + "/index.json")); lines, targets, genes = idx["lines"], idx["targets"], idx["genes"]
    T = len(targets); row = lambda L, t, h: (L * T + t) * 2 + h; ct = targets.index(idx["ctrl"]); gpos = {g: j for j, g in enumerate(genes)}
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy")
    for L, ln in enumerate(lines):
        ctrl = [np.asarray(S[row(L, ct, h)]) / cnt[row(L, ct, h), 0] for h in range(2)]; cm = (ctrl[0] + ctrl[1]) / 2; cp = np.expm1(cm)
        res = {}
        for t, g in enumerate(targets):
            if t == ct or g not in gpos or min(cnt[row(L, t, h), 0] for h in range(2)) < 10:
                continue
            j = gpos[g]; eff = np.mean([S[row(L, t, h)][j] / cnt[row(L, t, h), 0] - ctrl[h][j] for h in range(2)])
            s = cp[j] / (1 + cp[j]); res[g] = (float(1 + eff / s) if s > 0 else None, float(cp[j]))
        out[ln] = res; print("kd", ln, len(res), flush=True)
    # pan (hESC)
    agg = (_WS + "/pan_agg_guide"); idx = json.load(open(agg + "/index.json")); targets, genes = idx["targets"], idx["genes"]
    ct = targets.index(idx["ctrl"]); gpos = {g: j for j, g in enumerate(genes)}
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy")
    rows = lambda t: [t * 4 + p * 2 + h for p in range(2) for h in range(2)]
    cm = np.asarray(S[rows(ct)]).sum(0) / cnt[rows(ct), 0].sum(); cp = np.expm1(cm); res = {}
    for t, g in enumerate(targets):
        if t == ct or g not in gpos or cnt[rows(t), 0].sum() < 20:
            continue
        j = gpos[g]; eff = float(np.asarray(S[rows(t)])[:, j].sum() / cnt[rows(t), 0].sum() - cm[j]); s = cp[j] / (1 + cp[j])
        res[g] = (float(1 + eff / s) if s > 0 else None, float(cp[j]))
    out["hESC"] = res; print("kd hESC", len(res), flush=True)
    # cd4
    for st, res in cd4_residuals().items():
        out[st] = res; print("kd", st, len(res), flush=True)
    # replogle
    from explore_V_day6_day8 import read_rows, Line, load
    from replogle_indep_readout import effects
    FILES = {"K6": (_WS + "/data_ext/replogle/K562_essential_raw_singlecell_01.h5ad"),
             "K8": (_WS + "/replogle/K562_gwps_raw_singlecell_01.h5ad"),
             "R": (_WS + "/data_ext/replogle/rpe1_raw_singlecell_01.h5ad")}
    perts = [str(p) for p in np.load((_WS + "/replogle_indep_readout/responses.npz"))["perts"]]
    D = {nm: load(p, set(perts)) for nm, p in FILES.items()}
    common = sorted(set.intersection(*[set(str(g) for g in d["genes"]) for d in D.values()])); gpos = {g: j for j, g in enumerate(common)}
    E = {}
    for seed, (nm, p) in enumerate(FILES.items()):
        d = D[nm]; gidx = {str(g): j for j, g in enumerate(d["genes"])}
        X = read_rows(p, d["rows"], np.array([gidx[g] for g in common]))
        L = Line(X, d["codes"][d["rows"]], d["cats"], common, d["gem"][d["rows"]], d["ctrl"]); del X
        H, Fu, mu = effects(L, perts, seed); E[nm] = (Fu, np.expm1(mu)); del L
    for name, nms in (("K562", ("K6", "K8")), ("K562_d6_rand", ("K6",)), ("K562_d8_rand", ("K8",)), ("RPE1", ("R",))):
        res = {}
        for t, g in enumerate(perts):
            if g not in gpos:
                continue
            j = gpos[g]; vals = []
            for nm in nms:
                Fu, cp = E[nm]; s = cp[j] / (1 + cp[j])
                vals.append((1 + Fu[t, j] / s if s > 0 else np.nan, cp[j]))
            res[g] = (float(np.mean([v[0] for v in vals])), float(min(v[1] for v in vals)))
        out[name] = res; print("kd", name, len(res), flush=True)
    json.dump(out, open(RT + "/kd_residual.json", "w"))


def cd4_load():
    import h5py
    from scipy.sparse import csr_matrix
    from cd4_m3 import cat
    with h5py.File((_WS + "/data_ext/cd4/GWCD4i.pseudobulk_merged.h5ad")) as f:
        o = f["obs"]
        conds, cc = cat(o, "culture_condition"); genes_t, tc = cat(o, "perturbed_gene_name"); gtypes, gtc = cat(o, "guide_type"); gids, gic = cat(o, "guide_id")
        gname = np.array([x.decode() if isinstance(x, bytes) else x for x in f["var"]["gene_name"][:]])
        X = csr_matrix((f["X"]["data"][:], f["X"]["indices"][:], f["X"]["indptr"][:]), shape=tuple(f["X"].attrs["shape"]))
    isc = gtc == [i for i, g in enumerate(gtypes) if "non" in g.lower()][0]
    return conds, cc, [str(t) for t in genes_t], tc, isc, gids, gic, gname, X


def cd4_residuals():
    conds, cc, tn, tc, isc, gids, gic, gname, X = cd4_load()
    gpos = {}
    for j, g in enumerate(gname):
        gpos.setdefault(str(g), j)
    out = {}
    for s in range(len(conds)):
        v = np.asarray(X[np.where(isc & (cc == s))[0]].sum(0)).ravel(); cpc = v * 1e4 / v.sum()
        m = ~isc & (cc == s); rows = np.where(m)[0]; tt = tc[rows]
        res = {}
        for t in np.unique(tt):
            g = tn[t]
            if g not in gpos:
                continue
            r = rows[tt == t]; vt = np.asarray(X[r].sum(0)).ravel(); j = gpos[g]
            res[g] = (float((vt[j] * 1e4 / vt.sum()) / cpc[j]) if cpc[j] > 0 else None, float(cpc[j]))
        out["CD4_%s" % conds[s]] = res
    return out


def cd4nt():
    conds, cc, tn, tc, isc, gids, gic, gname, X = cd4_load()
    inert = json.load(open(RT + "/inert_targets.json"))
    out = {}
    for s in range(len(conds)):
        name = "CD4_%s" % conds[s]
        cls = {j for v in classify(sorted(set(inert[name]["inert"]))).values() for j in v}
        inert_set = {g for k, g in enumerate(sorted(set(inert[name]["inert"]))) if k not in cls}
        ntr = np.where(isc & (cc == s))[0]; ntg = sorted(set(gic[ntr].tolist()))
        half = {g: k % 2 for k, g in enumerate(ntg)}
        prof = lambda r: np.asarray(X[r].sum(0)).ravel()
        A = prof(ntr[[half[g] == 0 for g in gic[ntr]]]); B = prof(ntr[[half[g] == 1 for g in gic[ntr]]])
        ir = np.where(~isc & (cc == s) & np.isin(np.array(tn)[tc], list(inert_set)))[0]; I = prof(ir)
        cpA, cpB, cpI = (x * 1e4 / x.sum() for x in (A, B, I))
        cp = (cpA + cpB) / 2; P, gpos = prog_sets(gname, cp); sv = cp / (1 + cp)
        def score(x, ref):
            F = (np.log1p(x) - np.log1p(ref)) / np.maximum(sv, 1e-12)
            return {nm: (float(F[ix].mean()) if len(ix) >= MIN_GENES else None) for nm, ix in P.items()}
        out[name] = dict(n_nt_guides=len(ntg), n_inert_targets=len(inert_set), inert_vs_A=score(cpI, cpA), inert_vs_B=score(cpI, cpB), A_vs_B=score(cpA, cpB))
        print(name, len(ntg), len(inert_set), flush=True)
    json.dump(out, open(RT + "/cd4_nt_halves.json", "w"))


def tahoe():
    import pyarrow.parquet as pq
    z = np.load((_WS + "/tahoe_program/tahoe_arrays.npz"), allow_pickle=True, mmap_mode="r"); conds = [str(c) for c in z["conds"]]
    samples = json.load(open((_WS + "/tahoe_agg/index.json")))["samples"]
    d = pq.read_table((_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")).to_pydict()
    meta = {s: dict(plate=p, drug=dr.strip(), dc=dc, tscp=t, genes=g, mito=m) for s, p, dr, dc, t, g, m in
            zip(d["sample"], d["plate"], d["drug"], d["drugname_drugconc"], d["mean_tscp_count"], d["mean_gene_count"], d["mean_pcnt_mito"])}
    name = lambda s: "%s_%s" % (meta[s]["drug"], ast.literal_eval(meta[s]["dc"])[0][1])
    out, j = [], 0
    for s in samples:
        if j < len(conds) and not meta[s]["drug"].startswith("DMSO") and name(s) == conds[j]:
            out.append(dict(cond=conds[j], sample=s, **{k: meta[s][k] for k in ("plate", "tscp", "genes", "mito")})); j += 1
    dmso = {}
    for s, m in meta.items():
        if m["drug"].startswith("DMSO"):
            dmso.setdefault(m["plate"], []).append(dict(sample=s, tscp=m["tscp"], genes=m["genes"], mito=m["mito"]))
    json.dump(dict(conditions=out, dmso=dmso), open(RT + "/tahoe/sample_qc.json", "w"))
    print("tahoe conditions", len(out), "plates", len(dmso), flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    for f in ([tahoe, cd4nt, kd] if m == "all" else [{"kd": kd, "cd4nt": cd4nt, "tahoe": tahoe}[m]]):
        f()
