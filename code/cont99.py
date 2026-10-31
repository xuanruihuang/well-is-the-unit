"""Contamination by other lines' RNA as an explanation of the same-well advantage (samewell/notes.md §105, independent
review 10-29 item A10; A100). Ambient RNA, doublets or misassigned cells put the other lines' RNA, including their
response to the drug in that well, into a line's profile; that is formally a same-well prediction.
Foreign genes of line t: protein-coding genes whose CPM in t's pooled control cells (raw-count pseudobulk
pb/counts_lsh.npy, all non-plate-13 DMSO wells) is below 5% of the median CPM of the other lines, with that median
>= 10 and CPM >= 10 in at least 25 other lines (revised 10-29 after the first run: the absolute rule CPM < 0.5 found
0-6 genes per line, since contamination itself lifts a line's foreign genes; no valid result had been seen). Lines
with fewer than 10 foreign genes get the median f of the lines in that well (count reported). f is an upper bound:
a foreign gene's own low expression in t is counted as contamination. Contamination fraction of
line t in well w: f = (foreign-gene share of t's UMIs in w) / (foreign-gene share of the pooled UMIs of the other
lines in w). Decontaminated response (first order, response space): y' = (y - f m) / (1 - f), m = mean response of the
other lines in the well (D', cached genes). Reported, for the verified pairs at different positions, one plate and
plates 6/14 (same position) and the pairs without viability effect: distribution of f; Spearman correlation over
records between f (mean of the two wells) and the record's advantage; advantage in records with f below / above the
median; advantage after decontamination (target and predictor lines all decontaminated). Control pairs: the same
correlation and decontamination on H11 - H12.
Output $WORKSPACE/rt/samewell/cont99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
import sw_v1a
from sw_step2 import position
from ver99 import sign_p

OUT = (_WS + "/rt/samewell/"); PB = OUT + "pb/"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def main(K=1.0, tag=""):
    """K multiplies every contamination fraction (10-29 review: sensitivity with f doubled, cont99.py 2)."""
    d = C.load(); D0 = d["D"]; base = d["base"]; plate = np.array(d["plate"]); L, J = D0.shape[:2]
    fin = np.isfinite(D0).all(2); Dz = np.where(fin[:, :, None], D0, 0).astype(np.float64); X = np.zeros_like(Dz)
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]; S = Dz[:, js].sum(1); n = fin[:, js].sum(1)
        for j in js:
            X[:, j] = Dz[:, j] - (S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]
    idx = json.load(open(PB + "index.json")); Xc = np.load(PB + "counts_lsh.npy", mmap_mode="r"); ns = len(idx["samples"])
    lpos = [idx["lines"].index(ln) for ln in d["lines"]]; spos = {s: k for k, s in enumerate(idx["samples"])}
    sm = pq.read_table(V.META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    def pbc(i, s):
        c = lpos[i]; k = spos[s]; r = (c * ns + k) * 2
        return np.asarray(Xc[r], np.float64) + np.asarray(Xc[r + 1], np.float64)
    dmso = [s for s in idx["samples"] if info[s][0].startswith("DMSO") and info[s][1] != "plate13"]
    pooled = np.stack([np.sum([pbc(i, s) for s in dmso], 0) for i in range(L)]); cpm = 1e6 * pooled / pooled.sum(1, keepdims=True)
    F = []
    for t in range(L):
        oth = np.delete(cpm, t, 0); mo = np.median(oth, 0)
        F.append(np.where((cpm[t] < 0.05 * mo) & (mo >= 10) & ((oth >= 10).sum(0) >= 25))[0])
    few = [t for t in range(L) if len(F[t]) < 10]
    res = {"rules": __doc__, "foreign_genes_per_line": [int(len(f)) for f in F], "lines_imputed": len(few)}; print("foreign genes", res["foreign_genes_per_line"], flush=True)
    samples = [str(s) for s in d["sample"]]
    fcache = {}; tot = {}
    def total(s):
        if s not in tot:
            tot[s] = np.sum([pbc(i, s) for i in range(L)], 0)
        return tot[s]
    def frac(t, j):
        if (t, j) in fcache:
            return fcache[(t, j)]
        s = samples[j]; ct = pbc(t, s); co = total(s) - ct
        num = ct[F[t]].sum() / max(ct.sum(), 1); den = co[F[t]].sum() / max(co.sum(), 1)
        fcache[(t, j)] = v = float(num / den) if den > 0 and t not in few else np.nan
        return v
    def fwell(t, j, ls):
        v = frac(t, j)
        return v if np.isfinite(v) else float(np.nanmedian([frac(i, j) for i in ls]))
    pos, _, _ = position(d)
    lab84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    lab85 = {(x["a"], x["b"]) for x in json.load(open(OUT + "labels85.json"))["pairs"] if x["cls"] == "nodvia"}
    P = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10]
    sets = {}
    for g in ("diffpos", "sameplate", "G614"):
        gp = [p for p in P if p["group"] == g and (g != "G614" or pos[p["a"]] == pos[p["b"]])]
        sets[g + " verified"] = [p for p in gp if lab84.get((p["a"], p["b"])) == "confirmed"]
        sets[g + " nodvia"] = [p for p in gp if (p["a"], p["b"]) in lab85]
    res["sets"] = {}
    for sn, ps in sets.items():
        rec = []
        for p in ps:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
            f = {(t, j): fwell(t, j, ls) for t in ls for j in (a, b)}
            M = {j: X[ls, j].mean(0) for j in (a, b)}
            Y = {}
            for j in (a, b):
                for k, t in enumerate(ls):
                    m = (M[j] * len(ls) - X[t, j]) / (len(ls) - 1); ff = min(max(K * f[(t, j)], 0), 0.5)
                    Y[(t, j)] = (X[t, j] - ff * m) / (1 - ff)
            for t in ls:
                Gt = np.where(base[t] >= 0.2)[0]; s0, o0, s1, o1 = [], [], [], []
                for tg, rp in ((a, b), (b, a)):
                    mo = lambda j: (X[ls, j][:, Gt].sum(0) - X[t, j, Gt]) / (len(ls) - 1)
                    mo1 = lambda j: np.mean([Y[(i, j)][Gt] for i in ls if i != t], 0)
                    s0.append(C.pear(mo(tg), X[t, tg, Gt])); o0.append(C.pear(mo(rp), X[t, tg, Gt]))
                    s1.append(C.pear(mo1(tg), Y[(t, tg)][Gt])); o1.append(C.pear(mo1(rp), Y[(t, tg)][Gt]))
                rec.append(dict(comp=p["comp"], f=0.5 * (f[(t, a)] + f[(t, b)]), adv=float(np.mean(s0) - np.mean(o0)), adv_dec=float(np.mean(s1) - np.mean(o1))))
        fs = np.array([r["f"] for r in rec]); ad = np.array([r["adv"] for r in rec]); ad1 = np.array([r["adv_dec"] for r in rec]); med = np.median(fs)
        by = collections.defaultdict(list)
        for r in rec:
            by[r["comp"]].append(r["adv_dec"])
        cm = [np.mean(v) for v in by.values()]; k = int(sum(x > 0 for x in cm))
        e = dict(pairs=len(ps), records=len(rec), f_median=round(float(med), 4), f_q90=round(float(np.quantile(fs, 0.9)), 4), f_max=round(float(fs.max()), 4),
                 spearman_f_advantage=round(spearman(fs, ad), 4), advantage=round(float(ad.mean()), 4),
                 advantage_f_low=round(float(ad[fs <= med].mean()), 4), advantage_f_high=round(float(ad[fs > med].mean()), 4),
                 advantage_decontaminated=round(float(ad1.mean()), 4), decontaminated_compounds_positive=k, sign_p=sign_p(k, len(cm)))
        res["sets"][sn] = e; print(sn, json.dumps(e), flush=True)
    # control pairs
    prof, plates = sw_v1a.control_profiles(d); ctrl = {pl: sorted([s for s in idx["samples"] if info[s][1] == pl and info[s][0].startswith("DMSO")], key=lambda s: int(s[4:])) for pl in plates}
    jx = {s: j for j, s in enumerate(samples)}; rec = []
    for pl in plates:
        if pl == "plate13" or len(ctrl[pl]) != 2:
            continue
        R = {i: (prof[(i, pl)][0] - prof[(i, pl)][1]).astype(np.float64) for i in range(L) if len(prof[(i, pl)]) == 2 and all(w is not None for w in prof[(i, pl)])}
        ff = {}
        for i in R:
            fv = []
            for s in ctrl[pl]:
                ct = pbc(i, s); co = total(s) - ct
                fv.append(float((ct[F[i]].sum() / ct.sum()) / (co[F[i]].sum() / co.sum())) if i not in few else np.nan)
            ff[i] = np.mean(fv)
        mf = np.nanmedian(list(ff.values()))
        ff = {i: (v if np.isfinite(v) else mf) for i, v in ff.items()}
        ls = sorted(R); Mm = np.mean([R[i] for i in ls], 0)
        Rd = {i: (R[i] - min(max(K * ff[i], 0), 0.5) * (Mm * len(ls) - R[i]) / (len(ls) - 1)) / (1 - min(max(K * ff[i], 0), 0.5)) for i in ls}
        for t in ls:
            Gt = np.where(base[t] >= 0.2)[0]
            s0 = C.pear(np.mean([R[i] for i in ls if i != t], 0)[Gt], R[t][Gt]); s1 = C.pear(np.mean([Rd[i] for i in ls if i != t], 0)[Gt], Rd[t][Gt])
            rec.append((ff[t], s0, s1))
    fs = np.array([r[0] for r in rec])
    res["control_pairs"] = dict(records=len(rec), f_median=round(float(np.median(fs)), 4), f_q90=round(float(np.quantile(fs, 0.9)), 4),
                                spearman_f_same=round(spearman(fs, np.array([r[1] for r in rec])), 4),
                                same_median=round(float(np.median([r[1] for r in rec])), 4), same_median_decontaminated=round(float(np.median([r[2] for r in rec])), 4))
    print("control", json.dumps(res["control_pairs"]), flush=True)
    json.dump(res, open(OUT + "cont99%s.json" % tag, "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(K=float(sys.argv[1]), tag="_f%s" % sys.argv[1])
    else:
        main()
