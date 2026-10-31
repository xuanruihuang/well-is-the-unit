"""Part B of ref88.py for the control-well pairs only (samewell/notes.md §88): composition-following ambient RNA on
H11 - H12 (plate 13 excluded). Output $WORKSPACE/rt/samewell/ref88b.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
import sw_v1a

OUT = (_WS + "/rt/samewell/")


def main():
    d = C.load(); base = d["base"]; L = d["D"].shape[0]; G0 = np.where(np.median(base, 0) >= 0.2)[0]; res = {"B": {}}
    idx = json.load(open(V.AGG + "index.json")); cnt = np.load(V.AGG + "counts.npy"); Ps = len(idx["samples"]); nl = len(idx["lines"])
    cells = cnt[:, 0].reshape(nl, Ps, 2).sum(2); lpos = [idx["lines"].index(ln) for ln in d["lines"]]; spos = {s: k for k, s in enumerate(idx["samples"])}
    pi = lambda s: (lambda v: v / v.sum())(cells[lpos, spos[s]].astype(float))
    def r2(y, *cols):
        A_ = np.c_[np.ones(len(y)), np.column_stack(cols)]; e_ = y - A_ @ np.linalg.lstsq(A_, y, rcond=None)[0]
        return 1 - (e_ @ e_) / ((y - y.mean()) @ (y - y.mean()))
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
    res["B"]["control pairs"] = dict(R2_composition=round(float(np.median(rc)), 3), R2_well_mean=round(float(np.median(rd)), 3), R2_both=round(float(np.median(rb)), 3))
    print("B control", json.dumps(res["B"]["control pairs"]), flush=True)
    json.dump(res, open(OUT + "ref88b.json", "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
