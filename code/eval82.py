"""Held-out cell lines under a clean protocol, on verified replicates (samewell/notes.md §82; A100). As sw_v1b.py
(State's five test lines; for each replicated drug-dose one well enters training and its other wells never do;
few-shot: test lines also contribute 30% of their single-well conditions, seed 42), with three changes:
  - plate-13 wells are dropped everywhere (§81: they do not reproduce their labelled treatments);
  - drug-doses are keyed by the salt-stripped name (verified.norm), so salt-name replicates count as replicated and
    never enter training under another name;
  - each (drug-dose, held-out well) is grouped by the pair of training well and held-out well: G614, sameplate,
    diffpos, samepos, Ada; strength = dvia of the pair.
Models: perturbation mean (mean response of the training lines in the training well) and the additive model
(alpha_line + beta_drugdose, 10 rounds of alternating means). Scores: Pearson over the line's genes against the
training well and against the held-out well; the line's own agreement between the two wells (ceiling).
Optional 'dprime': responses relative to the line's mean over the plate's other treated wells (D', leaving the well
out) instead of the plate's DMSO wells (§83). Output $WORKSPACE/rt/samewell/eval82_<mode>[_dprime].json.
Optional 4th argument: a label class of labels84.json; only held-out pairs of that class are scored (§84).
Optional 5th argument "swap": the training well is the other one (plate 14 for plates 6/14, else the highest plate
number), so each pair is evaluated in both directions (§89).
Usage: eval82.py <fewshot|zeroshot> [dprime|dmso] [class] [swap]"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import sys, json, collections
import numpy as np
import sw_core as C
import verified as V
from sw_step2 import position

OUT = (_WS + "/rt/samewell/")
TEST = ["CVCL_1097", "CVCL_1285", "CVCL_1098", "CVCL_0334", "CVCL_0480"]


def main():
    mode = sys.argv[1]; rng = np.random.default_rng(42)
    swap = len(sys.argv) > 4 and sys.argv[4] == "swap"
    d = C.load(); D = d["D"]; L, J, G = D.shape; lines = d["lines"]; plate = d["plate"]; B = d["base"]
    pos, _, _ = position(d); dv = V.dvia_all()
    fin = np.isfinite(D).all(2)
    dprime = len(sys.argv) > 2 and sys.argv[2] == "dprime"
    if dprime:
        pa_ = np.array(plate); Dz = np.where(fin[:, :, None], D, 0).astype(np.float32); Dq = np.full_like(Dz, np.nan)
        for p in sorted(set(plate)):
            js = np.where(pa_ == p)[0]; S = Dz[:, js].sum(1, dtype=np.float64); n = fin[:, js].sum(1)
            for j in js:
                Dq[:, j] = Dz[:, j] - ((S - Dz[:, j]) / np.maximum(n - fin[:, j], 1)[:, None]).astype(np.float32)
        Dq[~fin] = np.nan; D = Dq
    fin[:, np.array(plate) == "plate13"] = False
    tid = [lines.index(c) for c in TEST]
    key = [(V.norm(d["drug"][j]), d["conc"][j]) for j in range(J)]; kid = {k: i for i, k in enumerate(sorted(set(key)))}; K = len(kid)
    kj = np.array([kid[k] for k in key]); wells = collections.defaultdict(list)
    for j, k in enumerate(key):
        if plate[j] != "plate13":
            wells[k].append(j)
    pn = lambda j: int(plate[j][5:]); trainwell, heldwells = {}, {}
    for k, js in wells.items():
        if len(js) < 2:
            continue
        ps = {plate[j] for j in js}
        if swap:
            w = [j for j in js if plate[j] == "plate14"][0] if ("plate6" in ps and "plate14" in ps and k[0] != "adagrasib") else sorted(js, key=lambda j: (pn(j), j))[-1]
        else:
            w = [j for j in js if plate[j] == "plate6"][0] if ("plate6" in ps and "plate14" in ps and k[0] != "adagrasib") else sorted(js, key=lambda j: (pn(j), j))[0]
        trainwell[k] = w; heldwells[k] = [j for j in js if j != w]
    def grp(k, w, h):
        if plate[w] == plate[h]:
            return "sameplate"
        if k[0] == "adagrasib":
            return "Ada"
        if {plate[w], plate[h]} == {"plate6", "plate14"} and pos[w] == pos[h]:
            return "G614"   # same position only, as verified.pairs (10-29)
        return "samepos" if pos[w] == pos[h] else "diffpos"
    held = {j for v in heldwells.values() for j in v}; rows = []
    for l in range(L):
        for j in range(J):
            if not fin[l, j] or j in held:
                continue
            if l in tid and (mode == "zeroshot" or key[j] in trainwell or rng.random() >= 0.3):
                continue
            rows.append((l, j))
    rows = np.array(rows); print("training rows", len(rows), flush=True)
    yr = D[rows[:, 0], rows[:, 1]].astype(np.float64); alpha = np.zeros((L, G)); beta = np.zeros((K, G))
    for _ in range(10):
        s = np.zeros((K, G)); n = np.zeros(K); np.add.at(s, kj[rows[:, 1]], yr - alpha[rows[:, 0]]); np.add.at(n, kj[rows[:, 1]], 1); beta = s / np.maximum(n, 1)[:, None]
        s = np.zeros((L, G)); n = np.zeros(L); np.add.at(s, rows[:, 0], yr - beta[kj[rows[:, 1]]]); np.add.at(n, rows[:, 0], 1)
        seen = n > 0; alpha[seen] = s[seen] / n[seen, None]
    if mode == "zeroshot":
        alpha[tid] = alpha[[l for l in range(L) if l not in tid]].mean(0)
    pm = lambda j: np.mean([D[l, j] for l in range(L) if l not in tid and fin[l, j]], 0)
    cls = sys.argv[3] if len(sys.argv) > 3 else None
    keep = {(min(x["a"], x["b"]), max(x["a"], x["b"])) for x in json.load(open(OUT + "labels84.json"))["pairs"] if x["cls"] == cls} if cls else None
    recs = []
    for k, w in trainwell.items():
        pmw = pm(w)
        for t in tid:
            if not fin[t, w]:
                continue
            Gx = np.where(B[t] >= 0.2)[0]; yt = D[t, w, Gx].astype(np.float64)
            preds = {"additive": alpha[t] + beta[kid[k]], "pmean": pmw}
            for h in heldwells[k]:
                if not fin[t, h]:
                    continue
                yh = D[t, h, Gx].astype(np.float64)
                if keep is not None and (min(w, h), max(w, h)) not in keep:
                    continue
                rec = dict(cond="%s_%s" % k, comp=k[0], group=grp(k, w, h), target=lines[t], ceiling=C.pear(yt, yh), w=int(w), h=int(h),
                           dvia=0.5 * (dv[str(d["sample"][w])] + dv[str(d["sample"][h])]))
                for m, p in preds.items():
                    rec[m + "_train"] = C.pear(p[Gx], yt); rec[m + "_held"] = C.pear(p[Gx], yh)
                recs.append(rec)
    res = {"rules": __doc__, "mode": mode, "n_training_rows": int(len(rows))}
    rngb = np.random.default_rng(0)
    def summ(sub):
        out = dict(n_rec=len(sub), n_cond=len({r["cond"] for r in sub}), n_comp=len({r["comp"] for r in sub}),
                   ceiling=round(float(np.nanmedian([r["ceiling"] for r in sub])), 3))
        comps = sorted({r["comp"] for r in sub})
        for m in ("additive", "pmean"):
            byc = collections.defaultdict(list)
            for r in sub:
                byc[r["comp"]].append(r[m + "_train"] - r[m + "_held"])
            cm = np.array([np.mean(byc[c]) for c in comps]); bs = [cm[rngb.integers(0, len(cm), len(cm))].mean() for _ in range(1000)]
            out[m] = dict(train=round(float(np.nanmedian([r[m + "_train"] for r in sub])), 3), held=round(float(np.nanmedian([r[m + "_held"] for r in sub])), 3),
                          diff_by_compound=[round(float(cm.mean()), 3)] + [round(float(q), 3) for q in np.quantile(bs, [0.025, 0.975])])
        return out
    for g in ("diffpos", "sameplate", "G614", "samepos", "Ada"):
        sub = [r for r in recs if r["group"] == g]
        if not sub:
            continue
        res[g] = summ(sub); print(g, json.dumps(res[g]), flush=True)
        if g in ("diffpos", "G614") and len(sub) >= 30:
            q = np.quantile([r["dvia"] for r in sub], [1 / 3, 2 / 3])
            for t in range(3):
                st = [r for r in sub if int(np.digitize(r["dvia"], q)) == t]
                res["%s | dvia %d" % (g, t)] = summ(st); print(g, "dvia", t, json.dumps(res["%s | dvia %d" % (g, t)]), flush=True)
    json.dump(res, open(OUT + "eval82_%s%s%s%s.json" % (mode, "_dprime" if dprime else "", "_" + cls if cls else "", "_swap" if swap else ""), "w"), indent=1); json.dump(recs, open(OUT + "eval82_%s%s%s%s_records.json" % (mode, "_dprime" if dprime else "", "_" + cls if cls else "", "_swap" if swap else ""), "w"))
    print("SUMMARY done", flush=True)


if __name__ == "__main__":
    main()
