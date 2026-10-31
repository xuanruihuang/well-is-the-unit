"""Averaging same-position replicates: adagrasib 0.05 uM (samewell/notes.md §58; A100).
Wells (sw_core conditions, positions from sample numbers): H10 on 13 plates; G10 on plate 7; B9 and B10 on plate 13.
Per line (genes with base >= 0.2) and k = 1..12: the mean response of k H10 wells drawn at random (30 draws, seed 0)
from the H10 wells on plates other than the target's plate, against
  same position     each H10 well in turn as the target
  other position    G10 (plate 7; plate 7's H10 excluded from the draws) and B9, B10 (plate 13)
Pearson of the k-mean with the target; medians over lines of the per-line mean over targets and draws. Also the lines
in the top tenth by the energy of their mean H10 response (the lines that respond to the drug).
Added after the first run: the other-position curve for G10 alone (same control position H11/H12 as the H10 wells; B9 and
B10 on plate 13 are referenced to controls at B11/B12), and all curves again on D' (D minus the line's mean over the
other treated wells of the plate, §53), which removes the control-position offset from every well.
Output $WORKSPACE/rt/samewell/ada_k.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json
import numpy as np
import sw_core as C
from sw_step2 import position

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); D = d["D"]; base = d["base"]; L = D.shape[0]; plate = d["plate"]
    pos, _, _ = position(d); fin = np.isfinite(D).all(2)
    Dz = np.where(fin[:, :, None], D, 0).astype(np.float32); Dp = np.zeros_like(Dz)
    for p in sorted(set(plate)):
        jp = np.where(plate == p)[0]; S = Dz[:, jp].sum(1, dtype=np.float64); n = fin[:, jp].sum(1)
        for j in jp:
            Dp[:, j] = Dz[:, j] - ((S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]).astype(np.float32)
    js = np.where((np.array([C.norm(x) for x in d["drug"]]) == "adagrasib") & (d["conc"] == "0.05"))[0]
    h10 = [j for j in js if pos[j] == 93]; oth = [j for j in js if pos[j] != 93]
    info = {int(j): (plate[j], int(pos[j])) for j in js}
    rng = np.random.default_rng(0); K = range(1, 13)
    kinds = ("same", "other", "G10")
    curves = {"%s_%s" % (r, k): np.full((L, 12), np.nan) for r in ("D", "Dprime") for k in kinds}; energy = np.full(L, np.nan)
    for t in range(L):
        G = np.where(base[t] >= 0.2)[0]; hh = [j for j in h10 if fin[t, j]]
        if len(hh) < 13:
            continue
        energy[t] = float((D[t, hh][:, G].astype(np.float64).mean(0) ** 2).sum())
        tsets = (("same", hh), ("other", [j for j in oth if fin[t, j]]), ("G10", [j for j in oth if fin[t, j] and pos[j] == 81]))
        for rname, V in (("D", D), ("Dprime", Dp)):
            for kind, targets in tsets:
                for k in K:
                    v = []
                    for tg in targets:
                        pool = [j for j in hh if plate[j] != plate[tg]]
                        if len(pool) < k:
                            continue
                        y = V[t, tg, G].astype(np.float64)
                        for _ in range(30):
                            pick = rng.choice(pool, k, replace=False)
                            v.append(C.pear(V[t, pick][:, G].astype(np.float64).mean(0), y))
                    if v:
                        curves["%s_%s" % (rname, kind)][t, k - 1] = np.mean(v)
        print("line", t, flush=True)
    top = energy >= np.nanquantile(energy, 0.9)
    res = {"rules": __doc__, "wells": {str(k): v for k, v in info.items()}}
    for name, m in (("all", np.isfinite(energy)), ("top_tenth", top)):
        res[name] = {kind: [round(float(np.nanmedian(c[m, k])), 4) for k in range(12)] for kind, c in curves.items()}
        res[name]["n_lines"] = int(m.sum())
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(OUT + "ada_k.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
