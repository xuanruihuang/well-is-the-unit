"""Per-well corrections on the current pair set (samewell/notes.md §92; A100). Table S3's correction rows came from the
early pipeline (step2_qc, v1a_bscore: 90 pairs of plates 6/14 before salt names were merged). Recomputed on
verified.pairs, plates 6 and 14 (96 pairs), control-well reference (sw_core D), with the corrections of those scripts:
  none     D
  qc       per line and gene, regression on the line's mean log UMI per cell and log cell number in the well, and the
           well's mitochondrial fraction and log genes per cell (sw_step2.qc_covariates, centred within plate); slopes
           fitted on the 0.05 uM wells of compounds without any replicate pair, then removed from every well
  rowcol   row and column effects of the well means removed per plate (median polish, sw_v1a.run_bscore)
Same-well advantage as dref92.py (mean over records of same - other; pair medians, median over pairs).
Output $WORKSPACE/rt/samewell/corr92.json. Usage: corr92.py"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json
import numpy as np
import sw_core as C
import verified as V
from sw_step2 import qc_covariates, wellmean, position

OUT = (_WS + "/rt/samewell/")


def advantage(D, d, ps):
    fin = np.isfinite(D).all(2); L = D.shape[0]; diffs, pm_s, pm_o = [], [], []
    for p in ps:
        a, b = p["a"], p["b"]; ok = [l for l in range(L) if fin[l, a] and fin[l, b]]; s_, o_ = [], []
        for t in ok:
            G = np.where(d["base"][t] >= 0.2)[0]; oth = [l for l in ok if l != t]
            for x, y in ((a, b), (b, a)):
                yt = D[t, x, G].astype(np.float64)
                s = C.pear(yt, D[oth, x][:, G].mean(0)); o = C.pear(yt, D[oth, y][:, G].mean(0)); s_.append(s); o_.append(o); diffs.append(s - o)
        pm_s.append(np.median(s_)); pm_o.append(np.median(o_))
    return dict(n_pairs=len(ps), same=round(float(np.median(pm_s)), 4), other=round(float(np.median(pm_o)), 4), advantage=round(float(np.mean(diffs)), 4))


def main():
    d = C.load(); D = d["D"]; plate = np.array(d["plate"]); d["plate"] = plate
    ps = [p for p in V.pairs(d) if p["group"] == "G614"]; test_comp = {p["comp"] for p in V.pairs(d)}
    res = {"rules": __doc__, "none": advantage(D, d, ps)}; print("none", res["none"], flush=True)
    X = qc_covariates(d); Dq = D.copy()
    low = np.array([j for j in range(D.shape[1]) if d["conc"][j] == "0.05" and plate[j] != "plate13" and V.norm(d["drug"][j]) not in test_comp])
    for i in range(D.shape[0]):
        js = low[np.isfinite(D[i, low]).all(1) & np.isfinite(X[i, low]).all(1)]
        Xi = np.c_[np.ones(len(js)), X[i, js]]; B = np.linalg.lstsq(Xi, D[i, js].astype(np.float64), rcond=None)[0]
        ok = np.isfinite(X[i]).all(1); Dq[i, ok] = (D[i, ok] - X[i, ok] @ B[1:]).astype(np.float32)
    res["qc"] = advantage(Dq, d, ps); res["qc_training_wells"] = int(len(low)); print("qc", res["qc"], flush=True)
    M = wellmean(D); pos, row, col = position(d); Dc = D.copy()
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]; Xm = M[js].copy(); rr = row[js]; cc = col[js]
        reff = np.zeros((8, Xm.shape[1])); ceff = np.zeros((12, Xm.shape[1])); r_ = Xm - np.median(Xm, 0)
        for _ in range(10):
            for k in range(8):
                m = rr == k
                if m.any():
                    e = np.median(r_[m], 0); reff[k] += e; r_[m] -= e
            for k in range(12):
                m = cc == k
                if m.any():
                    e = np.median(r_[m], 0); ceff[k] += e; r_[m] -= e
        for jj, j in enumerate(js):
            Dc[:, j] -= (reff[rr[jj]] + ceff[cc[jj]]).astype(np.float32)
    res["rowcol"] = advantage(Dc, d, ps); print("rowcol", res["rowcol"], flush=True)
    json.dump(res, open(OUT + "corr92.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
