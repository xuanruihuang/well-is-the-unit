"""Shared loading and evaluation for the same-well leakage study (samewell/notes.md), gene level, A100.
Data: cached Tahoe effects $WORKSPACE/tahoe_program/tahoe_arrays.npz (48 lines x 1,241 conditions x 6,134 genes; treated
minus same-plate DMSO, log1p means, two random cell halves; response = mean of halves), condition metadata
$WORKSPACE/rt/tahoe/tahoe_scores.npz, sample ids and QC $WORKSPACE/rt/tahoe/sample_qc.json, per line x sample depth
$WORKSPACE/rt/tahoe/line_depth.json, per-plate DMSO means from $WORKSPACE/tahoe_agg.
Models (as routing §32 / §33):
  map    A549 -> target ridge on PCA-50 of A549 training responses; copy = A549 response; mean = target's training mean
  M1     mean of the other lines' responses in the input well
  M2     mean of the 5 lines nearest to the target in DMSO baseline expression
  M3     per-gene calibration target = a + b * (other-lines mean), fitted on the target's training conditions
  ceiling = target's own response in the other well; deviation metric = Pearson of (prediction - input-well mean) with
  (observed - own-well mean).
Training conditions exclude every compound (salt-stripped name) that has more than one well."""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, re, collections, itertools
import numpy as np

REF = "CVCL_0023"
ARR = (_WS + "/tahoe_program/tahoe_arrays.npz")
SCO = (_WS + "/rt/tahoe/tahoe_scores.npz")
norm = lambda s: re.sub(r"\s*\(.*\)|\s+(citrate|hydrochloride|maleate|mesylate|sulfate|hcl)$", "", s, flags=re.I).strip().lower()


def pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def load():
    z = np.load(ARR, allow_pickle=True, mmap_mode="r")
    s = np.load(SCO, allow_pickle=True)
    lines = [str(x) for x in z["lines"]]; genes = [str(x) for x in z["genes"]]
    D = np.stack([(np.asarray(z["XA"][i], np.float32) + np.asarray(z["XB"][i], np.float32)) / 2 for i in range(len(lines))])
    drug, conc, plate = (np.array([str(x) for x in s[k]]) for k in ("drug", "conc", "plate"))
    qc = json.load(open((_WS + "/rt/tahoe/sample_qc.json")))
    sample = np.array([c["sample"] for c in qc["conditions"]])
    assert all(c["plate"] == p for c, p in zip(qc["conditions"], plate))
    return dict(D=D, lines=lines, genes=genes, base=np.asarray(z["base"], np.float64), pooled=np.asarray(z["pooled"], np.float64),
                drug=drug, conc=conc, plate=plate, sample=sample, qc=qc)


def plate_dmso(d):
    """Per line x plate DMSO mean (mean of the two halves' DMSO means, as in the cached responses), on the cached genes."""
    import pyarrow.parquet as pq, ast
    agg = (_WS + "/tahoe_agg"); idx = json.load(open(agg + "/index.json"))
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy")
    Ps = len(idx["samples"]); gpos = {g: k for k, g in enumerate(idx["genes"])}; gk = np.array([gpos[g] for g in d["genes"]])
    sm = pq.read_table((_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")).to_pydict()
    info = {s_: (dr.strip(), p) for s_, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    plates = sorted(set(d["plate"]), key=lambda p: int(p[5:]))
    out = np.zeros((len(d["lines"]), len(plates), len(gk)))
    for i, ln in enumerate(d["lines"]):
        c = idx["lines"].index(ln)
        for k, pl in enumerate(plates):
            ds = [j for j, s_ in enumerate(idx["samples"]) if info[s_][1] == pl and info[s_][0].startswith("DMSO")]
            hm = []
            for h in range(2):
                r = [(c * Ps + j) * 2 + h for j in ds]
                hm.append(np.asarray(S[r]).sum(0)[gk] / max(cnt[r, 0].sum(), 1))
            out[i, k] = (hm[0] + hm[1]) / 2
    return out, plates


def groups(d, multi_ok=True):
    """All unordered well pairs of the same drug x dose. Label: G614, G13, Gother, Ada (adagrasib 0.05 across plates),
    within (same plate)."""
    by = collections.defaultdict(list)
    for j in range(len(d["drug"])):
        by[(d["drug"][j], d["conc"][j])].append(j)
    pairs = []
    for k, js in by.items():
        if len(js) < 2:
            continue
        for ja, jb in itertools.combinations(js, 2):
            pa, pb = d["plate"][ja], d["plate"][jb]
            if pa == pb:
                g = "within"
            elif k[0] == "Adagrasib":
                g = "Ada"
            elif {pa, pb} == {"plate6", "plate14"}:
                g = "G614"
            elif "plate13" in (pa, pb):
                g = "G13"
            else:
                g = "Gother"
            pairs.append(("%s_%s" % k, ja, jb, g))
    names = {norm(k[0]) for k, js in by.items() if len(js) >= 2}
    train = np.array([j for j in range(len(d["drug"])) if norm(d["drug"][j]) not in names])
    return pairs, train


def evaluate(D, d, pairs, train, models=("map", "heldout")):
    """Per (pair, target line) records: same-well vs other-well inputs for the §32 and §33 models, plus well-mean R2."""
    lines, base = d["lines"], d["base"]; nL = len(lines); ir = lines.index(REF)
    fin = np.isfinite(D).all(2)
    total = np.nansum(np.where(fin[:, :, None], D, 0), 0, dtype=np.float64); count = fin.sum(0)
    rec = {}
    key = lambda p, t: (p[1], p[2], t)
    for p in pairs:
        for t in range(nL):
            rec[key(p, t)] = dict(cond=p[0], group=p[3], plates="%s|%s" % (d["plate"][p[1]], d["plate"][p[2]]), target=lines[t])
    # well-mean share
    for (c, ja, jb, g) in pairs:
        Ma, Mb = total[ja] / count[ja], total[jb] / count[jb]
        for t in range(nL):
            if fin[t, ja] and fin[t, jb]:
                r = rec[(ja, jb, t)]
                r["R2_same"] = 0.5 * (pear(D[t, ja], Ma) ** 2 + pear(D[t, jb], Mb) ** 2)
                r["R2_other"] = 0.5 * (pear(D[t, ja], Mb) ** 2 + pear(D[t, jb], Ma) ** 2)
    if "map" in models:
        Xr = D[ir].astype(np.float64); okr = fin[ir]; tr = train[okr[train]]
        mu, sd = Xr[tr].mean(0), Xr[tr].std(0); sd[sd == 0] = 1
        _, _, Vt = np.linalg.svd((Xr[tr] - mu) / sd, full_matrices=False); V50 = Vt[:50].T
        proj = lambda x: ((x - mu) / sd) @ V50
        Str = proj(Xr[tr]); ss = Str.std(0); Str = Str / ss
        for t in range(nL):
            if t == ir:
                continue
            expr = np.where((base[ir] >= 0.2) & (base[t] >= 0.2))[0]
            Xt = D[t].astype(np.float64); trt = tr[fin[t, tr]]
            Sx = Str[np.isin(tr, trt)]; Y = Xt[trt]; ym = Y.mean(0)
            lam = 0.1 * np.trace(Sx.T @ Sx) / Sx.shape[1]
            B = np.linalg.solve(Sx.T @ Sx + lam * np.eye(Sx.shape[1]), Sx.T @ (Y - ym))
            f = lambda x: (proj(x) / ss) @ B + ym
            for (c, a, b, g) in pairs:
                if not (okr[a] and okr[b] and fin[t, a] and fin[t, b]):
                    continue
                fa, fb = f(Xr[a]), f(Xr[b]); ix = expr
                r = rec[(a, b, t)]
                r.update(map_same=0.5 * (pear(fa[ix], Xt[a, ix]) + pear(fb[ix], Xt[b, ix])), map_cross=0.5 * (pear(fb[ix], Xt[a, ix]) + pear(fa[ix], Xt[b, ix])),
                         copy_same=0.5 * (pear(Xr[a, ix], Xt[a, ix]) + pear(Xr[b, ix], Xt[b, ix])), copy_cross=0.5 * (pear(Xr[b, ix], Xt[a, ix]) + pear(Xr[a, ix], Xt[b, ix])),
                         mean_pred=0.5 * (pear(ym[ix], Xt[a, ix]) + pear(ym[ix], Xt[b, ix])), ref_self=pear(Xr[a], Xr[b]))
    if "heldout" in models:
        for t in range(nL):
            G = np.where(base[t] >= 0.2)[0]
            bz = (base[:, G] - base[:, G].mean(1, keepdims=True)); bz /= np.linalg.norm(bz, axis=1, keepdims=True)
            sim = bz @ bz[t]; sim[t] = -np.inf; nn = np.argsort(-sim)[:5]
            def others_mean(j):
                own = D[t, j] if fin[t, j] else 0
                return ((total[j] - own) / (count[j] - fin[t, j]))[G]
            def knn_mean(j):
                ok = [i for i in nn if fin[i, j]]
                return D[ok, j][:, G].mean(0)
            trt = np.array([j for j in train if fin[t, j]])
            M = np.stack([others_mean(j) for j in trt]); Y = D[t, trt][:, G].astype(np.float64)
            mm, ym = M.mean(0), Y.mean(0); bb = ((M - mm) * (Y - ym)).sum(0) / np.maximum(((M - mm) ** 2).sum(0), 1e-12); aa = ym - bb * mm
            cal = lambda m: aa + bb * m
            for (c, ja, jb, g) in pairs:
                if not (fin[t, ja] and fin[t, jb]):
                    continue
                ya, yb = D[t, ja, G].astype(np.float64), D[t, jb, G].astype(np.float64)
                ma, mb = others_mean(ja), others_mean(jb); na, nb = knn_mean(ja), knn_mean(jb)
                r = rec[(ja, jb, t)]; r["ceiling"] = pear(ya, yb)
                for nm, fa, fb in (("M1", ma, mb), ("M2", na, nb), ("M3", cal(ma), cal(mb))):
                    r[nm + "_same"] = 0.5 * (pear(fa, ya) + pear(fb, yb)); r[nm + "_cross"] = 0.5 * (pear(fb, ya) + pear(fa, yb))
                for nm, fa, fb in (("M2", na, nb), ("M3", cal(ma), cal(mb))):
                    da, db = fa - ma, fb - mb; ta, tb = ya - ma, yb - mb
                    r[nm + "_dev_same"] = 0.5 * (pear(da, ta) + pear(db, tb)); r[nm + "_dev_cross"] = 0.5 * (pear(db, ta) + pear(da, tb))
    return [r for r in rec.values() if len(r) > 4]


PAIRS = (("map", "map_same", "map_cross"), ("copy", "copy_same", "copy_cross"), ("M1", "M1_same", "M1_cross"),
         ("M2", "M2_same", "M2_cross"), ("M3", "M3_same", "M3_cross"), ("M2dev", "M2_dev_same", "M2_dev_cross"),
         ("M3dev", "M3_dev_same", "M3_dev_cross"), ("R2", "R2_same", "R2_other"))


def summarise(rs, nboot=1000, seed=0):
    out = {"n_rec": len(rs), "n_cond": len({r["cond"] for r in rs})}
    for k in ("mean_pred", "ceiling"):
        v = [r[k] for r in rs if k in r]
        if v:
            out[k] = round(float(np.nanmedian(v)), 3)
    rng = np.random.default_rng(seed)
    for nm, a, b in PAIRS:
        v = [(r[a], r[b], r["cond"]) for r in rs if a in r and np.isfinite(r[a]) and np.isfinite(r[b])]
        if not v:
            continue
        out[nm] = [round(float(np.median([x[0] for x in v])), 3), round(float(np.median([x[1] for x in v])), 3)]
        byc = collections.defaultdict(list)
        for x in v:
            byc[x[2]].append(x[0] - x[1])
        cm = np.array([np.mean(u) for u in byc.values()])
        bs = [cm[rng.integers(0, len(cm), len(cm))].mean() for _ in range(nboot)]
        out[nm + "_diff"] = [round(float(cm.mean()), 3)] + [round(float(q), 3) for q in np.quantile(bs, [0.025, 0.975])]
    for nm in ("M1", "M3"):
        v = [r[nm + "_same"] > r["ceiling"] for r in rs if nm + "_same" in r and np.isfinite(r.get("ceiling", np.nan))]
        if v:
            out[nm + "_same_gt_ceiling"] = round(float(np.mean(v)), 3)
    return out
