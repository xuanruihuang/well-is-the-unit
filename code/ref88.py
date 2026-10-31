"""Eighth-review checks (samewell/notes.md §88; A100).
A Reference built from inert wells only. Under D' each response subtracts the mean of the plate's ~90 other treated
  wells, which contains the drug response common to all lines on that plate; for a cross-plate pair this term is shared
  by the target and the same-well predictor but not by the replicate well, and could produce a same-well advantage
  without any well shift. Here each half subtracts the line's mean of the same half over the plate's INERT treated wells
  only (dvia within the control-vs-control noise, check85 threshold 0.289; the well itself left out), so the subtracted
  term carries almost no drug response. Same-well vs replicate-well predictions (line genes, both wells as target,
  medians) and variance components (common genes centred; drug Q/W, shared well, own well) under D' and under this
  reference ('Dinert'), for verified pair classes.
B Ambient RNA / misassignment following each well's own line composition: per pair and line, the difference between
  the two wells (D', common genes) regressed on c = sum_k (pi_ak - pi_bk) base_k (pi = the wells' line fractions, base =
  line baselines) and on the wells' mean difference over lines; median R^2. Also for control-well pairs (H11 - H12,
  plate 13 excluded; composition from the two DMSO wells).
Output $WORKSPACE/rt/samewell/ref88.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
import sw_v1a

OUT = (_WS + "/rt/samewell/"); THR = 0.289


def main():
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L, J = d["D"].shape[:2]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA0 = np.stack([np.asarray(z["XA"][i], np.float32) for i in range(L)]).astype(np.float64)
    XB0 = np.stack([np.asarray(z["XB"][i], np.float32) for i in range(L)]).astype(np.float64)
    fin = np.isfinite(XA0).all(2) & np.isfinite(XB0).all(2)
    dv = V.dvia_all(); dvw = np.array([dv[str(s)] for s in d["sample"]]); inert = dvw < THR
    VAR = {}
    for name, mask in (("Dprime", np.ones(J, bool)), ("Dinert", inert)):
        hs = []
        for H0 in (XA0, XB0):
            Hz = np.where(fin[:, :, None], H0, 0); H = np.zeros_like(Hz)
            for p in sorted(set(plate)):
                js = np.where(plate == p)[0]; ref = js[mask[js]]
                S = Hz[:, ref].sum(1); n = fin[:, ref].sum(1)
                for j in js:
                    own = mask[j]
                    H[:, j] = Hz[:, j] - (S - (Hz[:, j] if own else 0)) / np.maximum(n - (fin[:, j] if own else 0), 1)[:, None]
            hs.append(H)
        VAR[name] = (hs[0], hs[1], (hs[0] + hs[1]) / 2)
    res = {"rules": __doc__, "n_inert_wells_per_plate_median": int(np.median([inert[plate == p].sum() for p in set(plate) if p != "plate13"]))}
    lab84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    lab85 = {(x["a"], x["b"]) for x in json.load(open(OUT + "labels85.json"))["pairs"] if x["cls"] == "nodvia"}
    P = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10]
    sets = {"diffpos all": [p for p in P if p["group"] == "diffpos"], "diffpos confirmed": [p for p in P if p["group"] == "diffpos" and lab84.get((p["a"], p["b"])) == "confirmed"],
            "diffpos nodvia": [p for p in P if p["group"] == "diffpos" and (p["a"], p["b"]) in lab85], "sameplate nodvia": [p for p in P if p["group"] == "sameplate" and (p["a"], p["b"]) in lab85],
            "sameplate confirmed": [p for p in P if p["group"] == "sameplate" and lab84.get((p["a"], p["b"])) == "confirmed"], "G614": [p for p in P if p["group"] == "G614"]}
    cen = lambda v: v - v.mean()
    res["A"] = {}
    for sname, ps in sets.items():
        e = {"n_pairs": len(ps), "n_comp": len({p["comp"] for p in ps})}
        for name, (HA, HB, X) in VAR.items():
            Ss = X.sum(0); nf = fin.sum(0); same, other, W, Q, SW = [], [], 0.0, 0.0, 0.0
            for p in ps:
                a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]; n = len(ls)
                Ya = X[ls, a][:, G0]; Yb = X[ls, b][:, G0]; Ya = Ya - Ya.mean(1, keepdims=True); Yb = Yb - Yb.mean(1, keepdims=True)
                Gaa, Gbb, Gab = Ya @ Ya.T, Yb @ Yb.T, Ya @ Yb.T
                for k, t in enumerate(ls):
                    Gt = np.where(base[t] >= 0.2)[0]
                    for tg, rp in ((a, b), (b, a)):
                        y = X[t, tg, Gt]; mo = lambda j: (Ss[j, Gt] - X[t, j, Gt]) / (nf[j] - 1)
                        same.append(C.pear(mo(tg), y)); other.append(C.pear(mo(rp), y))
                    ha = [cen(H[t, a, G0]) for H in (HA, HB)]; hb = [cen(H[t, b, G0]) for H in (HA, HB)]
                    W += 0.5 * (ha[0] @ ha[1] + hb[0] @ hb[1]); Q += 0.25 * sum(u @ v for u in ha for v in hb)
                    o = np.arange(n) != k
                    SW += 0.5 * (Gaa[k, o].mean() + Gbb[k, o].mean()) - 0.5 * (Gab[k, o].mean() + Gab[o, k].mean())
            e[name] = dict(same=round(float(np.nanmedian(same)), 4), other=round(float(np.nanmedian(other)), 4), adv_mean=round(float(np.nanmean(np.array(same) - np.array(other))), 4),
                           drug_share=round(Q / W, 4), shared_well=round(SW / W, 4), own_well=round(1 - (Q + SW) / W, 4))
        res["A"][sname] = e; print("A", sname, json.dumps(e), flush=True)
    # B composition-following ambient
    idx = json.load(open(V.AGG + "index.json")); cnt = np.load(V.AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); lpos = [idx["lines"].index(ln) for ln in d["lines"]]; spos = {s: k for k, s in enumerate(idx["samples"])}
    pi = lambda s: (lambda v: v / v.sum())(cells[lpos, spos[s]].astype(float))
    X = VAR["Dprime"][2]
    def r2(y, *cols):
        A_ = np.c_[np.ones(len(y)), np.column_stack(cols)]; e_ = y - A_ @ np.linalg.lstsq(A_, y, rcond=None)[0]
        return 1 - (e_ @ e_) / ((y - y.mean()) @ (y - y.mean()))
    res["B"] = {}
    for sname in ("diffpos nodvia", "diffpos confirmed", "sameplate nodvia", "G614"):
        rc, rd, rb = [], [], []
        for p in sets[sname]:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
            c = (pi(str(d["sample"][a])) - pi(str(d["sample"][b]))) @ base; dd = X[ls, a].mean(0) - X[ls, b].mean(0)
            for t in ls:
                y = (X[t, a] - X[t, b])[G0]; rc.append(r2(y, c[G0])); rd.append(r2(y, dd[G0])); rb.append(r2(y, c[G0], dd[G0]))
        res["B"][sname] = dict(R2_composition=round(float(np.median(rc)), 4), R2_well_mean=round(float(np.median(rd)), 4), R2_both=round(float(np.median(rb)), 4))
        print("B", sname, json.dumps(res["B"][sname]), flush=True)
    prof, plates = sw_v1a.control_profiles(d)
    sm = pq.read_table(V.META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    rc, rd, rb = [], [], []
    for pl in plates:
        if pl == "plate13":
            continue
        ds = sorted([s for s in idx["samples"] if info[s][1] == pl and info[s][0].startswith("DMSO")], key=lambda s: int(s[4:]))
        if len(ds) != 2:
            continue
        c = (pi(ds[0]) - pi(ds[1])) @ base
        R = {i: (prof[(i, pl)][0] - prof[(i, pl)][1]) for i in range(L) if len(prof[(i, pl)]) == 2 and all(w is not None for w in prof[(i, pl)])}
        dd = np.mean(list(R.values()), 0)
        for t, y in R.items():
            rc.append(r2(y[G0], c[G0])); rd.append(r2(y[G0], dd[G0])); rb.append(r2(y[G0], c[G0], dd[G0]))
    res["B"]["control pairs"] = dict(R2_composition=round(float(np.median(rc)), 4), R2_well_mean=round(float(np.median(rd)), 4), R2_both=round(float(np.median(rb)), 4))
    print("B control", json.dumps(res["B"]["control pairs"]), flush=True)
    json.dump(res, open(OUT + "ref88.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
