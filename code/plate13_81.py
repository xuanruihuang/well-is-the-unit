"""Are plate 13's wells labelled with the compounds they contain? (samewell/notes.md §81 item 8; A100). For every
compound-dose on plate 13 whose original well (other plate) has a strong line-mean D' signature, find the plate-13 well
whose signature matches it best; report its label, its sample-number offset from the labelled partner, and the
correlation with the labelled partner. A consistent offset points to a shifted sample-to-label mapping; best matches
labelled with the same compound at another dose point to dose relabelling; no match points to inactive wells.
Also: amplitude |M| of plate-13 replicate wells against their originals, and against other plates' wells at the same
dose. Output $WORKSPACE/rt/samewell/plate13_81.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json
import numpy as np
import sw_core as C
from sw_step2 import position


def main():
    d = C.load(); D = d["D"]; base = d["base"]; plate = np.array(d["plate"]); L, J = D.shape[:2]
    fin = np.isfinite(D).all(2); Dz = np.where(fin[:, :, None], D, 0).astype(np.float32); Dp = np.zeros_like(Dz)
    for p in sorted(set(plate)):
        js = np.where(plate == p)[0]; S = Dz[:, js].sum(1, dtype=np.float64); n = fin[:, js].sum(1)
        for j in js:
            Dp[:, j] = Dz[:, j] - ((S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]).astype(np.float32)
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    M = np.stack([Dp[fin[:, j], j][:, G0].mean(0) for j in range(J)]).astype(np.float64); M -= M.mean(1, keepdims=True)
    amp = np.linalg.norm(M, axis=1); Mn = M / amp[:, None]
    pos, _, _ = position(d); snum = np.array([int(str(s)[4:]) for s in d["sample"]])
    p13 = np.where(plate == "plate13")[0]; pairs, _ = C.groups(d); rows = []
    for c, a, b, g in pairs:
        if g != "G13":
            continue
        o, t = (a, b) if plate[b] == "plate13" else (b, a)
        r = Mn[p13] @ Mn[o]; k = p13[int(np.argmax(r))]
        rows.append(dict(drug=d["drug"][o], dose=d["conc"][o], orig_plate=plate[o], amp_orig=round(float(amp[o]), 2), amp_p13=round(float(amp[t]), 2),
                         r_labelled=round(float(Mn[o] @ Mn[t]), 3), best=d["drug"][k] + " " + d["conc"][k], r_best=round(float(r.max()), 3),
                         best_rank_of_labelled=int((r > Mn[o] @ Mn[t]).sum()), offset_samples=int(snum[k] - snum[t]),
                         pos_labelled=int(pos[t]), pos_best=int(pos[k])))
    rows.sort(key=lambda x: -x["amp_orig"])
    for x in rows:
        print(json.dumps(x), flush=True)
    others = np.where((plate != "plate13"))[0]
    res = dict(rules=__doc__, rows=rows, amp_plate13_median=round(float(np.median(amp[p13])), 2), amp_other_plates_median=round(float(np.median(amp[others])), 2),
               amp_ratio_p13_over_orig_median=round(float(np.median([x["amp_p13"] / x["amp_orig"] for x in rows])), 2))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("amp")}), flush=True)
    json.dump(res, open((_WS + "/rt/samewell/plate13_81.json"), "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
