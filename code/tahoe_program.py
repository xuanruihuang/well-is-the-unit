"""Program layer on Tahoe-100M (A100): does a cell line's interaction (gamma) land on the gene programs that this line
expresses? Hypothesis: a line can rewrite the readout only of programs it has switched on, so the program layer would be
partly readable from control cells.
Usage: tahoe_program.py <agg_dir> <out_dir> <hallmark gmt-like txt>
Effects and filters exactly as tahoe_m2.py (same lines, responsive conditions, genes, halves; arrays cached to
<out_dir>/tahoe_arrays.npz for reuse). gamma = two-way interaction over lines x conditions.
Per line c and Hallmark program h (genes of h in the gene axis, >= 15):
  gE[c, h]  denoised gamma energy  sum_{p, g in h} gA * gB
  base[c, h] mean DMSO log1p expression of the program's genes in line c
Tests:
  (1) across lines, per program: Spearman(base[:, h], gE[:, h]); median rho over programs and the fraction positive;
      null: 200 random gene sets per program, same size, genes drawn from the same pooled-expression deciles as the
      program's genes (so expression level alone is matched); program rho minus the null mean, and its percentile.
  (2) within line, across programs: Spearman over programs between the line's relative baseline (base[c, h] minus the
      across-line mean, divided by the across-line sd) and its relative gamma share (gE[c, h] / sum_h gE[c, h] divided by
      the across-line mean of that share); median over lines; same expression-matched null.
  (3) per line, the three programs with the largest relative gamma share.
Exploratory, no multiple-testing correction."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import os, sys, json, ast
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

MIN_LINE_CELLS, MIN_HALF, MIN_EXPR = 40, 10, 0.1
META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata")
N_NULL = 200


def interaction(X):
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean((0, 1), keepdims=True)


def build(agg, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    idx = json.load(open(agg + "/index.json"))
    lines, samples, genes = idx["lines"], idx["samples"], np.array(idx["genes"])
    C, Ps = len(lines), len(samples)
    S64 = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy")
    sm = pq.read_table(META + "/sample_metadata.parquet").to_pydict()
    s2 = {s: (d.strip(), dc, p) for s, d, dc, p in zip(sm["sample"], sm["drug"], sm["drugname_drugconc"], sm["plate"])}
    drug = np.array([s2[s][0] for s in samples]); plate = np.array([s2[s][2] for s in samples])
    cond = np.array(["%s_%s" % (s2[s][0], ast.literal_eval(s2[s][1])[0][1]) for s in samples])
    isd = np.char.startswith(drug.astype(str), "DMSO")
    row = lambda c, s, h: (c * Ps + s) * 2 + h
    n = cnt[:, 0].reshape(C, Ps, 2)
    plates = sorted(set(plate)); ctrl = {}
    for pl in plates:
        ds = np.where(isd & (plate == pl))[0]
        for c in range(C):
            for h in range(2):
                r = [row(c, s, h) for s in ds]
                ctrl[(c, pl, h)] = np.asarray(S64[r]).sum(0) / max(cnt[r, 0].sum(), 1)
    present = n.sum((0, 2)) > 0
    med = np.array([np.median(n[c][~isd & present].sum(1)) for c in range(C)])
    kl = np.where(med >= MIN_LINE_CELLS)[0]
    ok = present & ~isd
    for c in kl:
        ok &= (n[c, :, 0] >= MIN_HALF) & (n[c, :, 1] >= MIN_HALF)
    conds = np.where(ok)[0]
    pooled = np.mean([ctrl[(c, pl, h)] for c in kl for pl in plates for h in range(2)], 0)
    gk = np.where(pooled >= MIN_EXPR)[0]
    XA = np.zeros((len(kl), len(conds), len(gk)), np.float32); XB = np.zeros_like(XA)
    for i, c in enumerate(kl):
        for j, s in enumerate(conds):
            for h, X in ((0, XA), (1, XB)):
                r = row(c, s, h)
                X[i, j] = np.asarray(S64[r, gk]) / cnt[r, 0] - ctrl[(c, plate[s], h)][gk]
    bA, bB = XA.mean(0).astype(np.float64), XB.mean(0).astype(np.float64)
    e_beta = (bA * bB).sum(1)
    null = []
    for pl in plates:
        ds = np.where(isd & (plate == pl))[0]
        if len(ds) < 2:
            continue
        a, b = ds[: len(ds) // 2], ds[len(ds) // 2:]
        ea = []
        for h in range(2):
            m1 = np.mean([np.asarray(S64[[row(c, s, h) for s in a]]).sum(0)[gk] / max(cnt[[row(c, s, h) for s in a], 0].sum(), 1) for c in kl], 0)
            m2 = np.mean([np.asarray(S64[[row(c, s, h) for s in b]]).sum(0)[gk] / max(cnt[[row(c, s, h) for s in b], 0].sum(), 1) for c in kl], 0)
            ea.append(m1 - m2)
        null.append(float((ea[0] * ea[1]).sum()))
    thr = np.percentile(null, 75) if null else 0.0
    resp = np.where(e_beta > thr)[0]
    base = np.stack([np.mean([ctrl[(c, pl, h)][gk] for pl in plates for h in range(2)], 0) for c in kl])
    d = dict(lines=np.array([lines[c] for c in kl]), conds=cond[conds], genes=genes[gk], resp=resp,
             XA=XA, XB=XB, base=base.astype(np.float32), pooled=pooled[gk].astype(np.float32))
    np.savez(cache, **d)
    return d


def hallmarks(path):
    H = {}
    for line in open(path):
        parts = [x for x in line.rstrip("\n").split("\t") if x]
        if len(parts) > 1:
            H[parts[0]] = set(parts[1:])
    return H


def main():
    agg, out, hpath = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    d = build(agg, out + "/tahoe_arrays.npz")
    lines, genes, resp = list(d["lines"]), list(d["genes"]), d["resp"]
    XA, XB = d["XA"][:, resp].astype(np.float64), d["XB"][:, resp].astype(np.float64)
    gA, gB = interaction(XA), interaction(XB)
    E = (gA * gB).sum(1)                      # lines x genes, denoised gamma energy per gene
    base, pooled = d["base"].astype(np.float64), d["pooled"].astype(np.float64)
    gpos = {g: j for j, g in enumerate(genes)}
    H = {h: np.array(sorted(gpos[g] for g in s if g in gpos)) for h, s in hallmarks(hpath).items()}
    H = {h: ix for h, ix in H.items() if len(ix) >= 15}
    names = list(H)
    dec = np.digitize(pooled, np.quantile(pooled, np.linspace(0.1, 0.9, 9)))
    bydec = {k: np.where(dec == k)[0] for k in range(10)}
    rng = np.random.default_rng(3)
    def stats(ix):
        return E[:, ix].sum(1), base[:, ix].mean(1)
    def null_sets(ix, n):
        out_ = []
        for _ in range(n):
            out_.append(np.concatenate([rng.choice(bydec[k], (dec[ix] == k).sum(), replace=False)
                                        for k in range(10) if (dec[ix] == k).sum()]))
        return out_
    L = len(lines)
    gE = np.zeros((L, len(names))); bs = np.zeros_like(gE)
    res1 = {}
    for k, h in enumerate(names):
        gE[:, k], bs[:, k] = stats(H[h])
        rho = spearmanr(bs[:, k], gE[:, k])[0]
        nr = []
        for ix in null_sets(H[h], N_NULL):
            a, b = stats(ix); nr.append(spearmanr(b, a)[0])
        nr = np.array(nr)
        res1[h] = dict(n_genes=int(len(H[h])), rho=float(rho), null_mean=float(np.nanmean(nr)),
                       excess=float(rho - np.nanmean(nr)), pct_of_null=float((nr < rho).mean()))
    rhos = np.array([v["rho"] for v in res1.values()]); exc = np.array([v["excess"] for v in res1.values()])
    pct = np.array([v["pct_of_null"] for v in res1.values()])
    summary1 = dict(median_rho=float(np.median(rhos)), frac_positive=float((rhos > 0).mean()),
                    median_excess_over_null=float(np.median(exc)), frac_above_null_95=float((pct > 0.95).mean()),
                    frac_above_null_50=float((pct > 0.5).mean()))
    # (2) within line across programs
    zb = (bs - bs.mean(0)) / np.maximum(bs.std(0), 1e-12)
    share = gE / np.maximum(gE.sum(1, keepdims=True), 1e-12)
    rs = share / np.maximum(share.mean(0, keepdims=True), 1e-12)
    within = np.array([spearmanr(zb[c], rs[c])[0] for c in range(L)])
    # null for (2): replace programs by expression-matched random sets (one draw per program), 50 repeats
    wn = []
    for _ in range(50):
        sets = [null_sets(H[h], 1)[0] for h in names]
        g2 = np.stack([E[:, ix].sum(1) for ix in sets], 1); b2 = np.stack([base[:, ix].mean(1) for ix in sets], 1)
        z2 = (b2 - b2.mean(0)) / np.maximum(b2.std(0), 1e-12); s2_ = g2 / np.maximum(g2.sum(1, keepdims=True), 1e-12)
        r2 = s2_ / np.maximum(s2_.mean(0, keepdims=True), 1e-12)
        wn.append(np.median([spearmanr(z2[c], r2[c])[0] for c in range(L)]))
    summary2 = dict(median_within_line_rho=float(np.median(within)), frac_lines_positive=float((within > 0).mean()),
                    null_median_mean=float(np.mean(wn)), null_median_q95=float(np.quantile(wn, 0.95)))
    top = {lines[c]: [(names[k], round(float(rs[c, k]), 2), round(float(zb[c, k]), 2)) for k in np.argsort(-rs[c])[:3]]
           for c in range(L)}
    res = {"rules": __doc__, "n_lines": L, "n_conditions": int(len(resp)), "n_genes": len(genes), "n_programs": len(names),
           "across_lines": summary1, "within_line": summary2, "per_program": res1, "top_programs_per_line": top}
    json.dump(res, open(out + "/metrics_tahoe_program.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k in ("n_lines", "n_conditions", "n_genes", "n_programs", "across_lines", "within_line")}), flush=True)
    for h, v in sorted(res1.items(), key=lambda kv: -kv[1]["excess"])[:8]:
        print("  top", h, v, flush=True)
    for h, v in sorted(res1.items(), key=lambda kv: kv[1]["excess"])[:4]:
        print("  bottom", h, v, flush=True)
    for c in list(top)[:6]:
        print("  line", c, top[c], flush=True)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
