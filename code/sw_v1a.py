"""samewell/notes.md §6, written before running.
  ceil    A1: per (replicated condition, line): split-half reliability of the line's response within a well (cached
          halves XA, XB, each against its own half of the plate controls), Spearman-Brown corrected, vs the line's
          agreement with itself in the replicate well; for the full response and for the deviation from the well mean.
  negctrl A2: the two control wells of each plate (last two samples, H11 and H12). Pseudo-response of line t =
          H11 - H12 (mean log1p, halves averaged). Same-well: mean of the other lines' pseudo-responses on the same plate;
          other: the same on each other plate. Also well-mean R2 (same plate vs other plate).
  bscore  C2: per plate and gene, median polish of the well-mean responses over rows and columns; the row and column
          effects are subtracted from every line's response; then the same-well vs other-well evaluation.
Output $WORKSPACE/rt/samewell/v1a_<part>.json. Usage: sw_v1a.py <part ...>"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, collections
import numpy as np
import sw_core as C
from sw_step2 import wellmean, position

OUT = (_WS + "/rt/samewell/")
GROUPS = (("G614", ("G614",)), ("other_position", ("G13", "Gother")), ("same_plate", ("within",)), ("adagrasib", ("Ada",)))


def sb(r):
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def run_ceil(d):
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(len(d["lines"]))])
    XB = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(len(d["lines"]))])
    D = d["D"]; base = d["base"]; fin = np.isfinite(D).all(2)
    MA = np.nanmean(np.where(fin[:, :, None], XA, np.nan), 0); MB = np.nanmean(np.where(fin[:, :, None], XB, np.nan), 0)
    M = wellmean(D)
    pairs, _ = C.groups(d)
    recs = []
    for c, a, b, g in pairs:
        for t in range(len(d["lines"])):
            if not (fin[t, a] and fin[t, b]):
                continue
            G = np.where(base[t] >= 0.2)[0]
            half = np.nanmean([sb(C.pear(XA[t, j, G], XB[t, j, G])) for j in (a, b)])
            halfdev = np.nanmean([sb(C.pear(XA[t, j, G] - MA[j, G], XB[t, j, G] - MB[j, G])) for j in (a, b)])
            cross = C.pear(D[t, a, G], D[t, b, G]); crossdev = C.pear(D[t, a, G] - M[a, G], D[t, b, G] - M[b, G])
            recs.append(dict(cond=c, group=g, half=half, cross=cross, halfdev=halfdev, crossdev=crossdev))
    json.dump(recs, open(OUT + "v1a_ceil_records.json", "w"))
    res = {}
    for name, gs in GROUPS:
        sub = [r for r in recs if r["group"] in gs]
        if not sub:
            continue
        f = lambda k: round(float(np.nanmedian([r[k] for r in sub])), 3)
        res[name] = dict(n=len(sub), n_cond=len({r["cond"] for r in sub}), half=f("half"), cross=f("cross"), halfdev=f("halfdev"), crossdev=f("crossdev"),
                         frac_half_gt_cross=round(float(np.nanmean([r["half"] > r["cross"] for r in sub])), 3))
        print("ceil", name, res[name], flush=True)
    return res


def control_profiles(d):
    """Per line and plate: mean log1p of each of the two control samples (halves averaged), on the cached genes."""
    import pyarrow.parquet as pq
    agg = (_WS + "/tahoe_agg"); idx = json.load(open(agg + "/index.json"))
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); cnt = np.load(agg + "/counts.npy")
    Ps = len(idx["samples"]); gpos = {g: k for k, g in enumerate(idx["genes"])}; gk = np.array([gpos[g] for g in d["genes"]])
    sm = pq.read_table((_WS + "/data_ext/tahoe/Tahoe-100M/metadata/sample_metadata.parquet")).to_pydict()
    info = {s_: (dr.strip(), p) for s_, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    plates = sorted({p for _, p in info.values()}, key=lambda p: int(p[5:]))
    prof = {}
    for pl in plates:
        ds = sorted([j for j, s_ in enumerate(idx["samples"]) if info[s_][1] == pl and info[s_][0].startswith("DMSO")], key=lambda j: int(idx["samples"][j][4:]))
        for i, ln in enumerate(d["lines"]):
            c = idx["lines"].index(ln); w = []
            for j in ds:
                rows = [(c * Ps + j) * 2 + h for h in range(2)]
                n = cnt[rows, 0]
                w.append(np.mean([np.asarray(S[r])[gk] / n[h] for h, r in enumerate(rows)], 0) if n.min() >= 10 else None)
            prof[(i, pl)] = w
    return prof, plates


def run_negctrl(d):
    prof, plates = control_profiles(d); base = d["base"]; L = len(d["lines"])
    R = {}
    for pl in plates:
        for i in range(L):
            w = prof[(i, pl)]
            if len(w) == 2 and w[0] is not None and w[1] is not None:
                R[(i, pl)] = (w[0] - w[1]).astype(np.float64)
    recs = []
    for pl in plates:
        have = [i for i in range(L) if (i, pl) in R]
        Mp = np.mean([R[(i, pl)] for i in have], 0)
        for t in have:
            G = np.where(base[t] >= 0.2)[0]
            others_same = np.mean([R[(i, pl)] for i in have if i != t], 0)
            same = C.pear(others_same[G], R[(t, pl)][G]); r2s = C.pear(R[(t, pl)][G], Mp[G]) ** 2
            oth, r2o = [], []
            for q in plates:
                if q == pl:
                    continue
                hq = [i for i in range(L) if (i, q) in R and i != t]
                if len(hq) < 10:
                    continue
                mq = np.mean([R[(i, q)] for i in hq], 0)
                oth.append(C.pear(mq[G], R[(t, pl)][G])); r2o.append(C.pear(R[(t, pl)][G], mq[G]) ** 2)
            recs.append(dict(plate=pl, line=t, same=same, other=float(np.mean(oth)), r2_same=r2s, r2_other=float(np.mean(r2o))))
    f = lambda k: round(float(np.median([r[k] for r in recs])), 3)
    res = dict(n=len(recs), plates=len(plates), same=f("same"), other=f("other"), r2_same=f("r2_same"), r2_other=f("r2_other"),
               per_plate_same={pl: round(float(np.median([r["same"] for r in recs if r["plate"] == pl])), 3) for pl in plates})
    print("negctrl", json.dumps(res), flush=True)
    return res


def run_bscore(d):
    M = wellmean(d["D"]); pos, row, col = position(d); plate = d["plate"]
    Dc = d["D"].copy()
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]
        X = M[js].copy(); rr = row[js]; cc = col[js]
        reff = np.zeros((8, X.shape[1])); ceff = np.zeros((12, X.shape[1])); res_ = X - np.median(X, 0)
        for _ in range(10):
            for k in range(8):
                m = rr == k
                if m.any():
                    e = np.median(res_[m], 0); reff[k] += e; res_[m] -= e
            for k in range(12):
                m = cc == k
                if m.any():
                    e = np.median(res_[m], 0); ceff[k] += e; res_[m] -= e
        for jj, j in enumerate(js):
            Dc[:, j] -= (reff[rr[jj]] + ceff[cc[jj]]).astype(np.float32)
    pairs, train = C.groups(d)
    rs = C.evaluate(Dc, d, [p for p in pairs if p[3] in ("G614", "G13", "Gother", "within")], train)
    out = {}
    for name, gs in GROUPS[:3]:
        s = C.summarise([r for r in rs if r["group"] in gs], nboot=300)
        out[name] = {k: s[k] for k in ("n_cond", "ceiling", "map", "map_diff", "M1", "M1_diff", "R2") if k in s}
        print("bscore", name, json.dumps(out[name]), flush=True)
    return out


def main():
    d = C.load()
    for p in sys.argv[1:]:
        json.dump(globals()["run_" + p](d), open(OUT + "v1a_%s.json" % p, "w"), indent=1)


if __name__ == "__main__":
    main()
