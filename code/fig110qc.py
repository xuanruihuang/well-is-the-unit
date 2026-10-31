"""Method checks of the redrawn figures (samewell/notes.md §110 A3, A4; A100). Same profiles, lines, genes, A/B split and
stripe score as fig110.py (whole-well difference; genes = the K highest and K lowest of the A-line mean; score = Pearson of
each B line's row with the A mean over those genes; median over B lines).
  A3 depth (control pairs): per line, the deeper well's summed counts are thinned binomially to the shallower well's
     total (seed 0) before the difference; stripe score with and without thinning.
  A4 drug strength (drug pairs, G614 and diffpos): a proportional difference in drug strength between two wells moves
     each line along its own response; each line's difference is regressed (over the genes, no intercept) on its mean
     response in the two wells (D'), and genes and score are recomputed on the residuals.
Output $WORKSPACE/rt/samewell/fig110qc.json. Usage: fig110qc.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json
import numpy as np
import pyarrow.parquet as pq
import verified as V
import bp91_data as B

OUT = (_WS + "/rt/samewell/"); PB = OUT + "pb/"; META = (_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")
K = 150


def stripe(F, inA):
    mA = F[inA].mean(0); o = np.argsort(-mA); sel = np.concatenate([o[:K], o[-K:]])
    return float(np.median([np.corrcoef(F[k, sel], mA[sel])[0, 1] for k in np.where(~inA)[0]]))


def main():
    idx = json.load(open(PB + "index.json")); lines, samples = idx["lines"], idx["samples"]; L, P = len(lines), len(samples)
    cnt4 = np.load(PB + "cnt_lsh.npy").reshape(L, P, 2, 4); cnt = cnt4.sum(2)
    Xc = np.load(PB + "counts_lsh.npy", mmap_mode="r")
    sm = pq.read_table(META).to_pydict(); info = {s: (d.strip(), p) for s, d, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    drug = [info[s][0] for s in samples]; plate = [info[s][1] for s in samples]; isd = np.array([x.startswith("DMSO") for x in drug])
    exists = cnt[:, :, 0] > 0; qc = (cnt[:, :, 0] >= 50) & (cnt[:, :, 1] >= 35000)
    fail = 1 - (qc & exists).sum(1) / np.maximum(exists.sum(1), 1)
    keep_l = [i for i in range(L) if fail[i] <= 0.9 and exists[i].any()]
    ok = qc & (cnt4[:, :, :, 0].min(2) >= 10)
    lg = lambda c: np.log1p(1e4 * c / c.sum())
    raw = lambda i, j: np.asarray(Xc[(i * P + j) * 2], np.float64) + np.asarray(Xc[(i * P + j) * 2 + 1], np.float64)
    whole = lambda i, j: lg(raw(i, j))
    plates = sorted({p for p in plate if p != "plate13"}, key=lambda p: int(p[5:]))
    ctrl = {pl: sorted([j for j in range(P) if isd[j] and plate[j] == pl], key=lambda j: int(samples[j][4:])) for pl in plates}
    base = [np.mean([whole(i, j) for pl in plates for j in ctrl[pl] if qc[i, j]], 0) for i in keep_l]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    rng = np.random.default_rng(0); perm = rng.permutation(len(keep_l)); A = set(np.array(keep_l)[perm[: len(keep_l) // 2]].tolist())
    res = {"rules": __doc__, "A3": {}, "A4": {}}
    trng = np.random.default_rng(0)
    for pl in plates:
        if len(ctrl[pl]) != 2:
            continue
        a, b = ctrl[pl]; F, T, inA = [], [], []
        for i in keep_l:
            if not (ok[i, a] and ok[i, b]):
                continue
            ca, cb = raw(i, a), raw(i, b); F.append((lg(ca) - lg(cb))[G0])
            if ca.sum() > cb.sum():
                ca = trng.binomial(ca.astype(np.int64), cb.sum() / ca.sum()).astype(np.float64)
            else:
                cb = trng.binomial(cb.astype(np.int64), ca.sum() / cb.sum()).astype(np.float64)
            T.append((lg(ca) - lg(cb))[G0]); inA.append(i in A)
        inA = np.array(inA); res["A3"][pl] = dict(plain=stripe(np.array(F), inA), thinned=stripe(np.array(T), inA))
        print("A3", pl, json.dumps(res["A3"][pl]), flush=True)
    d = B.light_d(); si = {s: k for k, s in enumerate(samples)}
    lab = {(min(x["a"], x["b"]), max(x["a"], x["b"])): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    prs = []
    for e in V.pairs(d):
        if e["group"] in ("G614", "diffpos"):
            a, b = min(e["a"], e["b"]), max(e["a"], e["b"])
            prs.append((e["group"], e["cond"], lab.get((a, b)), si[str(d["sample"][a])], si[str(d["sample"][b])]))
    need = sorted({plate[x] for p in prs for x in p[3:]}); tsum, tn = {}, {}
    for i in keep_l:
        for pl in need:
            js = [j for j in range(P) if plate[j] == pl and not isd[j] and qc[i, j]]
            if js:
                tsum[(i, pl)] = np.sum([whole(i, j)[G0] for j in js], 0); tn[(i, pl)] = len(js)
    for g, cond, cls, a, b in prs:
        F, R, inA = [], [], []
        for i in keep_l:
            if not (ok[i, a] and ok[i, b]) or tn.get((i, plate[a]), 0) < 2 or tn.get((i, plate[b]), 0) < 2:
                continue
            wa, wb = whole(i, a)[G0], whole(i, b)[G0]
            ra = wa - (tsum[(i, plate[a])] - wa) / (tn[(i, plate[a])] - 1); rb = wb - (tsum[(i, plate[b])] - wb) / (tn[(i, plate[b])] - 1)
            dlt = ra - rb; m = 0.5 * (ra + rb); beta = float(dlt @ m / max(m @ m, 1e-12))
            F.append(dlt); R.append(dlt - beta * m); inA.append(i in A)
        if len(F) < 10:
            continue
        inA = np.array(inA); key = "%s|%s|%d|%d" % (g, cond, a, b)
        res["A4"][key] = dict(group=g, cls=cls, plain=stripe(np.array(F), inA), residual=stripe(np.array(R), inA))
    for g, f in (("G614", lambda v: v["group"] == "G614"), ("diffpos concordant", lambda v: v["group"] == "diffpos" and v["cls"] == "confirmed")):
        vs = [v for v in res["A4"].values() if f(v)]
        res["A4 summary " + g] = dict(pairs=len(vs), plain_median=float(np.median([v["plain"] for v in vs])), residual_median=float(np.median([v["residual"] for v in vs])),
                                      residual_range=[float(min(v["residual"] for v in vs)), float(max(v["residual"] for v in vs))])
        print("A4", g, json.dumps(res["A4 summary " + g]), flush=True)
    json.dump(res, open(OUT + "fig110qc.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
