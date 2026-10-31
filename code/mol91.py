"""Molina & Zhang 2026 cross-line transfer on verified replicate pairs (samewell/notes.md §91; A100). As mol_transfer.py
(§33: State's five lines; source = mean response of the other four lines in a well, >= 3 present; one-shot transfer =
the source itself; low-rank ridge = SVD basis (k = 30) of the training sources and a ridge map to the target response,
lambda from {1e-4 .. 1e3} (grid extended downwards in §93 after the lowest value was chosen) x mean feature variance on a 20% validation split of training compounds, seed 0; genes
expressed in the target, base >= 0.2), with these changes:
  - plate-13 wells are dropped everywhere (§81);
  - reference 'dprime' (the line's mean over the plate's other treated wells, well left out; main) or 'dmso' (the
    plate's DMSO wells; sw_core);
  - training: wells of compounds (verified.norm) that have no verified.pairs pair, excluding stereoisomers and other
    salts of those (bp91_data.py moved_to_test); test: every verified pair (a, b) in
    both directions, source from well t, scored against the target in t (their setting) and in the other well o;
  - pair label classes from labels84 (confirmed, ...) and labels85 (nodvia, ...).
Summaries per group and class: medians over records; same - other as the mean over compounds with a 95% interval from
1,000 bootstrap resamples of compounds (seed 0). Output $WORKSPACE/rt/samewell/mol91_<ref>.json (+ _records.json).
Usage: mol91.py <dprime|dmso>"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, collections
import numpy as np
import sw_core as C
import verified as V

OUT = (_WS + "/rt/samewell/")
TEST = ["CVCL_1097", "CVCL_1285", "CVCL_1098", "CVCL_0334", "CVCL_0480"]


def main():
    ref = sys.argv[1]; d = C.load(); D = d["D"]; lines = d["lines"]; plate = np.array(d["plate"]); fin = np.isfinite(D).all(2)
    if ref == "dprime":
        Dz = np.where(fin[:, :, None], D, 0).astype(np.float32); Dq = np.full_like(Dz, np.nan)
        for p in sorted(set(plate)):
            js = np.where(plate == p)[0]; S = Dz[:, js].sum(1, dtype=np.float64); n = fin[:, js].sum(1)
            for j in js:
                Dq[:, j] = Dz[:, j] - ((S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]).astype(np.float32)
        Dq[~fin] = np.nan; D = Dq
    fin[:, plate == "plate13"] = False
    lab = {}
    for fn in ("labels84", "labels85"):
        for x in json.load(open(OUT + fn + ".json"))["pairs"]:
            lab[(fn, min(x["a"], x["b"]), max(x["a"], x["b"]))] = x["cls"]
    pairs = V.pairs(d); test_comp = {p["comp"] for p in pairs}
    test_comp |= set(json.load(open(OUT + "bp91/dprime/meta.json"))["moved_to_test"])     # stereoisomers / salts of test compounds (bp91_data.py)
    train = [j for j in range(D.shape[1]) if plate[j] != "plate13" and V.norm(d["drug"][j]) not in test_comp]
    tid = [lines.index(c) for c in TEST]; rng = np.random.default_rng(0); recs = []
    for t in tid:
        src = [l for l in tid if l != t]; ex = np.where(d["base"][t] >= 0.2)[0]
        def S(j):
            ok = [l for l in src if fin[l, j]]
            return D[ok, j][:, ex].astype(np.float64).mean(0) if len(ok) >= 3 else None
        tr = [j for j in train if fin[t, j] and S(j) is not None]
        Xs = np.array([S(j) for j in tr]); Ys = D[t, tr][:, ex].astype(np.float64)
        cpd = np.array([V.norm(d["drug"][j]) for j in tr]); uc = np.unique(cpd); val = set(rng.choice(uc, len(uc) // 5, replace=False))
        iv = np.array([c in val for c in cpd])
        def fit(X):
            mx = X.mean(0); _, _, Vt = np.linalg.svd(X - mx, full_matrices=False); Vb = Vt[:30].T
            return mx, Vb, (X - mx) @ Vb
        ridge = lambda Z, Yc, lam: np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Yc)
        mx, Vb, Z = fit(Xs[~iv]); my = Ys[~iv].mean(0); bv = Z.var(0).mean(); best = None
        for f in (1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3):
            Wt = ridge(Z, Ys[~iv] - my, f * bv * len(Z)); P = ((Xs[iv] - mx) @ Vb) @ Wt + my
            sc = np.mean([C.pear(p, y) for p, y in zip(P, Ys[iv])])
            if best is None or sc > best[0]:
                best = (sc, f)
        mx, Vb, Z = fit(Xs); my = Ys.mean(0); Wt = ridge(Z, Ys - my, best[1] * Z.var(0).mean() * len(Z))
        pred = lambda s: ((s - mx) @ Vb) @ Wt + my
        for p in pairs:
            a, b = min(p["a"], p["b"]), max(p["a"], p["b"])
            for x, y in ((a, b), (b, a)):
                sx = S(x)
                if sx is None or not (fin[t, x] and fin[t, y]):
                    continue
                yx, yy = D[t, x, ex].astype(np.float64), D[t, y, ex].astype(np.float64); px = pred(sx)
                recs.append(dict(cond=p["cond"], comp=p["comp"], group=p["group"], cls84=lab.get(("labels84", a, b)), cls85=lab.get(("labels85", a, b)),
                                 t=int(x), o=int(y), target=lines[t], direct_same=C.pear(sx, yx), direct_other=C.pear(sx, yy),
                                 ridge_same=C.pear(px, yx), ridge_other=C.pear(px, yy), r=C.pear(yx, yy)))
        print(lines[t], "train", len(tr), "lambda factor", best[1], "val r", round(best[0], 3), flush=True)
    res = {"rules": __doc__, "ref": ref, "n_records": len(recs)}; rb = np.random.default_rng(0)
    subsets = [("G614 all", "G614", None), ("G614 confirmed", "G614", ("cls84", "confirmed")), ("diffpos all", "diffpos", None),
               ("diffpos confirmed", "diffpos", ("cls84", "confirmed")), ("diffpos nodvia", "diffpos", ("cls85", "nodvia")),
               ("sameplate confirmed", "sameplate", ("cls84", "confirmed")), ("sameplate nodvia", "sameplate", ("cls85", "nodvia")),
               ("Ada confirmed", "Ada", ("cls84", "confirmed"))]
    for name, g, c in subsets:
        rs = [r for r in recs if r["group"] == g and (c is None or r[c[0]] == c[1])]
        if not rs:
            continue
        comps = sorted({r["comp"] for r in rs}); out = dict(n_records=len(rs), n_pairs=len({(r["t"], r["o"]) for r in rs}) // 2, n_comp=len(comps),
                                                            r=round(float(np.median([r["r"] for r in rs])), 3))
        for m in ("direct", "ridge"):
            out[m + "_same"] = round(float(np.median([r[m + "_same"] for r in rs])), 3); out[m + "_other"] = round(float(np.median([r[m + "_other"] for r in rs])), 3)
            byc = collections.defaultdict(list)
            for r in rs:
                byc[r["comp"]].append(r[m + "_same"] - r[m + "_other"])
            cm = np.array([np.mean(byc[k]) for k in comps]); bs = [cm[rb.integers(0, len(cm), len(cm))].mean() for _ in range(1000)]
            out[m + "_diff"] = [round(float(cm.mean()), 3)] + [round(float(q), 3) for q in np.quantile(bs, [0.025, 0.975])]
        res[name] = out; print(name, json.dumps(out), flush=True)
    json.dump(res, open(OUT + "mol91_%s.json" % ref, "w"), indent=1); json.dump(recs, open(OUT + "mol91_%s_records.json" % ref, "w"))
    print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
