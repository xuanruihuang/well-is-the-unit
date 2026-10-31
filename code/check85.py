"""Fifth-review checks (samewell/notes.md §85; A100).
A Weak-drug class by viability alone: dvia of each well against the plate's DMSO wells; noise level = dvia of a plate's
  DMSO well against the plate's other DMSO well (plate 13 excluded). Classes written to labels85.json (format of
  labels84.json): 'lowdvia' = both wells below the 33rd percentile of treated wells' dvia; 'nodvia' = both wells below
  the 90th percentile of the DMSO-vs-DMSO noise level. Reported: number of pairs per group and how many of them were
  'confirmed' in labels84.
B Plate-13 control layout. (i) Control offset: per line, regular plate p's controls relative to its treated wells,
  c_p = -(mean over p's treated wells of D); q_-p = mean of c over the other regular plates; Pearson(c_p, q_-p) over the
  line's genes (median over lines) for each regular plate, against Pearson(D'_w, q) for plate-13 wells at H11 / H12
  (D' = against plate 13's other treated wells). Equal strength fits a control swap on plate 13; clearly weaker fits a
  position effect. (ii) Composition: SD over lines of the log2 odds of each well's line shares against the pooled
  DMSO wells of the other regular plates, for regular plates' DMSO wells, plate 13's labelled DMSO wells, plate 13's
  H11 / H12 wells and other treated wells.
C D' reference across plates: for different-position pairs, energy of the difference between the two wells'
  references (line's mean over the other treated wells of each plate; common genes centred) relative to the
  within-well reproducible energy W.
D Cell-cycle sign-flip on control pairs: sign of T per regular plate (cc_signflip.json).
Output $WORKSPACE/rt/samewell/check85.json and labels85.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
from sw_step2 import position

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); D = d["D"]; base = d["base"]; plate = np.array(d["plate"]); L, J = D.shape[:2]; res = {"rules": __doc__}
    idx = json.load(open(V.AGG + "index.json")); cnt = np.load(V.AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); sm = pq.read_table(V.META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    spos = {s: k for k, s in enumerate(idx["samples"])}; dmso = collections.defaultdict(list)
    for k, s in enumerate(idx["samples"]):
        if info[s][0].startswith("DMSO"):
            dmso[info[s][1]].append(k)
    lor = lambda n_, N: np.log2((n_ + 0.5) / (N - n_ + 0.5))
    def dvia_against(k, ref):
        nD = cells[:, ref].sum(1); n_ = cells[:, k]
        return float(np.std(lor(n_, n_.sum()) - lor(nD, nD.sum())))
    # A
    noise = [dvia_against(ks[i], [ks[1 - i]]) for p, ks in dmso.items() if p != "plate13" and len(ks) == 2 for i in range(2)]
    thr_noise = float(np.quantile(noise, 0.9))
    dv = V.dvia_all(); dvw = np.array([dv[str(s)] for s in d["sample"]]); q33 = float(np.quantile(dvw[plate != "plate13"], 1 / 3))
    lab84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    fin = np.isfinite(D).all(2); out = []
    for p in V.pairs(d, dv):
        a, b = p["a"], p["b"]
        if (fin[:, a] & fin[:, b]).sum() < 10:
            continue
        m = max(dvw[a], dvw[b])
        for cls, ok in (("lowdvia", m < q33), ("nodvia", m < thr_noise)):
            if ok:
                out.append(dict(a=a, b=b, group=p["group"], comp=p["comp"], salt=p["salt"], cls=cls, cls84=lab84.get((a, b))))
    cnts = collections.Counter((x["group"], x["cls"]) for x in out); conf = collections.Counter((x["group"], x["cls"]) for x in out if x["cls84"] == "confirmed")
    res["A"] = dict(dmso_noise_dvia_median=round(float(np.median(noise)), 3), dmso_noise_dvia_q90=round(thr_noise, 3), treated_dvia_q33=round(q33, 3),
                    treated_dvia_median=round(float(np.median(dvw[plate != "plate13"])), 3),
                    counts={"%s|%s" % k: [v, conf.get(k, 0)] for k, v in cnts.items()})
    print("A", json.dumps(res["A"]), flush=True)
    json.dump({"rules": __doc__, "pairs": out}, open(OUT + "labels85.json", "w"), indent=1, default=int)
    # B (i)
    regular = [p for p in sorted(set(plate), key=lambda x: int(x[5:])) if p != "plate13"]
    fin2 = np.isfinite(D).all(2); Dz = np.where(fin2[:, :, None], D, 0).astype(np.float64)
    cp = {p: -(Dz[:, plate == p].sum(1) / np.maximum(fin2[:, plate == p].sum(1), 1)[:, None]) for p in regular}
    own = {}
    for p in regular:
        qm = np.mean([cp[o] for o in regular if o != p], 0)
        own[p] = round(float(np.median([C.pear(cp[p][t, np.where(base[t] >= 0.2)[0]], qm[t, np.where(base[t] >= 0.2)[0]]) for t in range(L)])), 3)
    q = np.mean([cp[o] for o in regular], 0); pos, row, col = position(d)
    js13 = np.where(plate == "plate13")[0]; S13 = Dz[:, js13].sum(1); n13 = fin2[:, js13].sum(1)
    h = {}
    for j in js13:
        if row[j] == 7 and col[j] in (10, 11):
            Dpj = Dz[:, j] - (S13 - Dz[:, j]) / np.maximum(n13 - fin2[:, j], 1)[:, None]
            h["H%d" % (col[j] + 1)] = round(float(np.median([C.pear(Dpj[t, np.where(base[t] >= 0.2)[0]], q[t, np.where(base[t] >= 0.2)[0]]) for t in range(L) if fin2[t, j]])), 3)
    res["B_offset"] = dict(regular_controls_vs_q_leave_one_out=own, regular_median=round(float(np.median(list(own.values()))), 3), plate13_H11_H12_treated=h)
    print("B offset", json.dumps(res["B_offset"]), flush=True)
    # B (ii)
    def comp_dev(k, excl):
        ref = [kk for p in regular if p != excl for kk in dmso[p]]
        return dvia_against(k, ref)
    reg_dmso = [comp_dev(k, info[idx["samples"][k]][1]) for p in regular for k in dmso[p]]
    p13_dmso = [comp_dev(k, None) for k in dmso["plate13"]]
    samp13 = {str(d["sample"][j]): j for j in js13}
    p13_h = [comp_dev(spos[s], None) for s, j in samp13.items() if row[j] == 7 and col[j] in (10, 11)]
    p13_other = [comp_dev(spos[s], None) for s, j in samp13.items() if not (row[j] == 7 and col[j] in (10, 11))]
    reg_treated = [comp_dev(spos[str(d["sample"][j])], plate[j]) for j in np.where(plate == "plate5")[0]]
    res["B_composition"] = dict(regular_dmso_median=round(float(np.median(reg_dmso)), 3), regular_dmso_max=round(float(np.max(reg_dmso)), 3),
                                plate13_labelled_dmso=[round(x, 3) for x in p13_dmso], plate13_H11_H12=[round(x, 3) for x in p13_h],
                                plate13_other_treated_median=round(float(np.median(p13_other)), 3), plate5_treated_median=round(float(np.median(reg_treated)), 3))
    print("B composition", json.dumps(res["B_composition"]), flush=True)
    # C
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    z = np.load(C.ARR, allow_pickle=True, mmap_mode="r")
    XA = np.stack([np.asarray(z["XA"][i], np.float32)[:, G0] for i in range(L)]).astype(np.float64)
    XB = np.stack([np.asarray(z["XB"][i], np.float32)[:, G0] for i in range(L)]).astype(np.float64)
    fin3 = np.isfinite(XA).all(2) & np.isfinite(XB).all(2)
    R = {}
    for nm, H in (("A", XA), ("B", XB)):
        Hz = np.where(fin3[:, :, None], H, 0); Rh = np.zeros_like(Hz)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Hz[:, js].sum(1); n = fin3[:, js].sum(1)
            for j in js:
                Rh[:, j] = (S - Hz[:, j]) / np.maximum(n - fin3[:, j], 1)[:, None]
        R[nm] = Rh
    cen = lambda v: v - v.mean()
    eref, W = 0.0, 0.0
    for p in V.pairs(d):
        if p["group"] != "diffpos":
            continue
        a, b = p["a"], p["b"]
        for l in np.where(fin3[:, a] & fin3[:, b])[0]:
            eref += float(cen(R["A"][l, a] - R["A"][l, b]) @ cen(R["B"][l, a] - R["B"][l, b]))
            ha = [cen(H[l, a] - R[k][l, a]) for k, H in (("A", XA), ("B", XB))]; hb = [cen(H[l, b] - R[k][l, b]) for k, H in (("A", XA), ("B", XB))]
            W += 0.5 * (ha[0] @ ha[1] + hb[0] @ hb[1])
    res["C"] = dict(reference_difference_energy_over_W=round(eref / W, 3))
    print("C", json.dumps(res["C"]), flush=True)
    # D
    cc = json.load(open(OUT + "cc_signflip.json"))["pseudo"]
    res["D"] = {p: [round(v["T"], 3), round(v["p"], 4)] for p, v in cc.items()}
    print("D", json.dumps(res["D"]), flush=True)
    json.dump(res, open(OUT + "check85.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
