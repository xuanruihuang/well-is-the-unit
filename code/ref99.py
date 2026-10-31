"""References that cannot favour the same-well arm (samewell/notes.md §100, independent review 10-29 item A3; A100).
Review point: the treated-well reference subtracts the plate's treated-well mean, and the control-well reference the
plate's control wells, which carry their own well shift; for a pair on two plates either term is shared by the target
and the same-well predictor but not by the replicate well. Responses rebuilt from the per-sample, per-half sums of
log1p values ($WORKSPACE/tahoe_agg, as sw_v1a.control_profiles), on the cached genes; T[t, j] = line t's mean log1p in
sample j, per half. References for a pair (a on plate P, b on plate Q):
  ctrl      T - (cell-weighted mean of the plate's two control wells)  (must reproduce the paper's control-well numbers)
  split     the target line minus control well 1 of its plate, the other lines minus control well 2, and the reverse
            (scores averaged over the two assignments); the control wells' own shift is no longer shared
  treated   T - the line's mean over the plate's other treated wells (the paper's default)
  single    every line minus the same single control well of its plate (well 1, then well 2, averaged): the noise of
            'split' with the control shift still shared; split vs single isolates the sharing (added after the first run)
  common    both wells minus the SAME reference: the line's mean over the treated wells of plates P and Q, without a
            and b. INVALID for pairs on two plates (found in the first run, 10-29): without a plate term the plate's
            own shift returns and is shared by the target and the same-well predictor; kept only for one-plate pairs
  tsplit    as split, with the plate's other treated wells in two halves (alternate sample order) in place of the two
            control wells; tsingle, every line against the same half (the noise of tsplit, sharing kept). Tests whether
            the treated-well reference shares anything between the arms (10-29 review; ref99.py tsplit -> ref99_tsplit.json)
Per record (pair x line; both wells as target, averaged): same and replicate-well scores as in the paper. Also, under
'common', the variance components of the paper (W within-well half product, Q cross-well half product, common well
part from the other lines; common genes centred) and the ceiling (Pearson between the two halves of the target well,
Spearman-Brown corrected, against Pearson between the two wells' full responses).
Sets: verified pairs (labels84 'confirmed'), pairs without detectable viability effect (labels85 'nodvia'), for
different positions, one plate and plates 6/14 (same position only). Summary as ver99.py (compound-equal mean and
sign test over compounds). Output $WORKSPACE/rt/samewell/ref99.json"""
from config import WORKSPACE as _WS  # release: server root made configurable (code/config.py)
import json, collections, math
import numpy as np
import pyarrow.parquet as pq
import sw_core as C
import verified as V
from sw_step2 import position
from ver99 import sign_p

OUT = (_WS + "/rt/samewell/"); AGG = (_WS + "/tahoe_agg/")


def main(kinds=("ctrl", "split", "treated", "common"), components=True, tag=""):
    d = C.load(); base = d["base"]; plate = np.array(d["plate"]); L, J = d["D"].shape[:2]; D0 = d["D"]
    G0 = np.where(np.median(base, 0) >= 0.2)[0]
    idx = json.load(open(AGG + "index.json")); S = np.load(AGG + "sums64.npy", mmap_mode="r"); cnt = np.load(AGG + "counts.npy")
    Ps = len(idx["samples"]); gpos = {g: k for k, g in enumerate(idx["genes"])}; gk = np.array([gpos[g] for g in d["genes"]])
    lpos = [idx["lines"].index(ln) for ln in d["lines"]]; spos = {s: k for k, s in enumerate(idx["samples"])}
    sm = pq.read_table(V.META).to_pydict(); info = {s: (dr.strip(), p) for s, dr, p in zip(sm["sample"], sm["drug"], sm["plate"])}
    samples = [str(s) for s in d["sample"]]
    def prof(c, k):
        rows = [(c * Ps + k) * 2 + h for h in range(2)]; n = cnt[rows, 0]
        if n.min() < 10:
            return None, None, 0
        return np.asarray(S[rows[0]])[gk] / n[0], np.asarray(S[rows[1]])[gk] / n[1], int(n.sum())
    G = len(gk); TA = np.full((L, J, G), np.nan, np.float32); TB = np.full((L, J, G), np.nan, np.float32)
    for i, c in enumerate(lpos):
        for j, s in enumerate(samples):
            a, b, _ = prof(c, spos[s])
            if a is not None:
                TA[i, j], TB[i, j] = a, b
        print("line", i, flush=True)
    ctrl = {}                                  # (line, plate) -> [(half A, half B, cells) of control well 1, of well 2]
    for pl in sorted(set(plate)):
        ds = sorted([s for s in idx["samples"] if info[s][1] == pl and info[s][0].startswith("DMSO")], key=lambda s: int(s[4:]))
        for i, c in enumerate(lpos):
            ctrl[(i, pl)] = [prof(c, spos[s]) for s in ds]
    fin = np.isfinite(TA).all(2) & np.isfinite(TB).all(2) & np.isfinite(D0).all(2)
    T = (TA.astype(np.float64) + TB) / 2
    # sanity: T - pooled control should reproduce D
    chk = []
    for j in range(0, J, 97):
        for i in range(0, L, 7):
            w = ctrl[(i, plate[j])]
            if fin[i, j] and len(w) == 2 and all(x[0] is not None for x in w):
                cm = (w[0][2] * (w[0][0] + w[0][1]) / 2 + w[1][2] * (w[1][0] + w[1][1]) / 2) / (w[0][2] + w[1][2])
                chk.append(C.pear(T[i, j] - cm, D0[i, j].astype(np.float64)))
    res = {"rules": __doc__, "sanity_corr_with_D_median": round(float(np.median(chk)), 4), "sanity_corr_min": round(float(np.min(chk)), 4)}
    print("sanity", res["sanity_corr_with_D_median"], res["sanity_corr_min"], flush=True)
    pos, _, _ = position(d)
    lab84 = {(x["a"], x["b"]): x["cls"] for x in json.load(open(OUT + "labels84.json"))["pairs"]}
    lab85 = {(x["a"], x["b"]) for x in json.load(open(OUT + "labels85.json"))["pairs"] if x["cls"] == "nodvia"}
    P = [p for p in V.pairs(d) if (fin[:, p["a"]] & fin[:, p["b"]]).sum() >= 10]
    sets = {}
    for g in ("diffpos", "sameplate", "G614"):
        gp = [p for p in P if p["group"] == g and (g != "G614" or pos[p["a"]] == pos[p["b"]])]
        sets[g + " verified"] = [p for p in gp if lab84.get((p["a"], p["b"])) == "confirmed"]
        sets[g + " nodvia"] = [p for p in gp if (p["a"], p["b"]) in lab85]
    treated = np.array([not info[s][0].startswith("DMSO") for s in samples])
    Tz = np.where(fin[:, :, None], T, 0); plate_sum = {}; half_sum = {}; half_of = {}
    for p in set(plate):
        js = np.where((plate == p) & treated)[0]; plate_sum[p] = (Tz[:, js].sum(1), fin[:, js].sum(1))
        for h in (0, 1):          # 'tsplit'/'tsingle' (10-29 review): the plate's treated wells in two halves (alternate sample order)
            jh = js[h::2]; half_sum[(p, h)] = (Tz[:, jh].sum(1), fin[:, jh].sum(1))
            for j in jh:
                half_of[j] = h
    def cw(i, pl, k=None):
        w = ctrl[(i, pl)]
        if len(w) != 2 or any(x[0] is None for x in w):
            return None
        if k is None:
            return (w[0][2] * (w[0][0] + w[0][1]) / 2 + w[1][2] * (w[1][0] + w[1][1]) / 2) / (w[0][2] + w[1][2])
        return (w[k][0] + w[k][1]) / 2
    def resp(kind, i, j, pair, role=None):
        """Response of line i in well j under a reference; role = 0/1 picks the control well for 'split'."""
        x = T[i, j]
        if kind == "ctrl":
            r = cw(i, plate[j]); return None if r is None else x - r
        if kind in ("split", "single"):
            r = cw(i, plate[j], role); return None if r is None else x - r
        if kind == "treated":
            s, n = plate_sum[plate[j]]; return x - (s[i] - Tz[i, j]) / max(n[i] - fin[i, j], 1)
        if kind in ("tsplit", "tsingle"):
            s, n = half_sum[(plate[j], role)]; own = half_of.get(j) == role
            return x - (s[i] - own * Tz[i, j]) / max(n[i] - own * fin[i, j], 1)
        if kind == "common":
            a, b = pair; s = plate_sum[plate[a]][0][i] + (plate_sum[plate[b]][0][i] if plate[b] != plate[a] else 0)
            n = plate_sum[plate[a]][1][i] + (plate_sum[plate[b]][1][i] if plate[b] != plate[a] else 0)
            return x - (s - Tz[i, a] - Tz[i, b]) / max(n - fin[i, a] - fin[i, b], 1)
    def score(kind, ps):
        rec = []
        for p in ps:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
            roles = ((1, 0), (0, 1)) if kind in ("split", "tsplit") else ((0, 0), (1, 1)) if kind in ("single", "tsingle") else ((None, None),)
            cache = {}
            for rt, ro in roles:
                for j in (a, b):
                    for i in ls:
                        cache[(rt, ro, i, j)] = (resp(kind, i, j, (a, b), rt), resp(kind, i, j, (a, b), ro))
            for t in ls:
                Gt = np.where(base[t] >= 0.2)[0]; sv, ov = [], []
                for rt, ro in roles:
                    for tg, rp in ((a, b), (b, a)):
                        y = cache[(rt, ro, t, tg)][0]
                        if y is None:
                            continue
                        def mo(j):
                            v = [cache[(rt, ro, i, j)][1] for i in ls if i != t and cache[(rt, ro, i, j)][1] is not None]
                            return np.mean(v, 0)[Gt]
                        sv.append(C.pear(mo(tg), y[Gt])); ov.append(C.pear(mo(rp), y[Gt]))
                if sv:
                    rec.append((p["comp"], float(np.mean(sv)), float(np.mean(ov))))
        return rec
    def summ(rec, ps):
        by = collections.defaultdict(list)
        for c, s, o in rec:
            by[c].append(s - o)
        cm = [np.mean(v) for v in by.values()]; k = int(sum(x > 0 for x in cm))
        return dict(pairs=len(ps), compounds=len(by), same=round(float(np.median([r[1] for r in rec])), 4), other=round(float(np.median([r[2] for r in rec])), 4),
                    diff_record_mean=round(float(np.mean([r[1] - r[2] for r in rec])), 4), diff_compound_mean=round(float(np.mean(cm)), 4),
                    compounds_positive=k, sign_p=sign_p(k, len(cm)))
    res["advantage"] = {}
    for sn, ps in sets.items():
        for kind in kinds:
            res["advantage"]["%s | %s" % (sn, kind)] = e = summ(score(kind, ps), ps)
            print("adv", sn, kind, json.dumps(e), flush=True)
    # components and ceilings under the common reference
    cen = lambda v: v - v.mean(); res["common_components"] = {}
    for sn, ps in (sets.items() if components else []):
        W = Q = SW = 0.0; half, cross = [], []
        for p in ps:
            a, b = p["a"], p["b"]; ls = np.where(fin[:, a] & fin[:, b])[0]
            pa, pb = plate[a], plate[b]
            sA = plate_sum[pa]; sB = plate_sum[pb]
            def ref(i):
                s = sA[0][i] + (sB[0][i] if pb != pa else 0); n = sA[1][i] + (sB[1][i] if pb != pa else 0)
                return (s - Tz[i, a] - Tz[i, b]) / max(n - fin[i, a] - fin[i, b], 1)
            R = {i: ref(i) for i in ls}
            ha = {i: [cen((H[i, a] - R[i])[G0]) for H in (TA, TB)] for i in ls}; hb = {i: [cen((H[i, b] - R[i])[G0]) for H in (TA, TB)] for i in ls}
            ya = {i: (ha[i][0] + ha[i][1]) / 2 for i in ls}; yb = {i: (hb[i][0] + hb[i][1]) / 2 for i in ls}
            for t in ls:
                W += 0.5 * (ha[t][0] @ ha[t][1] + hb[t][0] @ hb[t][1]); Q += 0.25 * sum(u @ v for u in ha[t] for v in hb[t])
                o = [i for i in ls if i != t]
                SW += 0.5 * (np.mean([ya[t] @ ya[i] for i in o]) + np.mean([yb[t] @ yb[i] for i in o])) - 0.5 * (np.mean([ya[t] @ yb[i] for i in o]) + np.mean([yb[t] @ ya[i] for i in o]))
                for hh in (ha[t], hb[t]):
                    r = C.pear(hh[0], hh[1]); half.append(2 * r / (1 + r) if r > -1 else np.nan)
                cross.append(C.pear(ya[t], yb[t]))
        res["common_components"][sn] = dict(drug_share=round(Q / W, 4), common_well=round(SW / W, 4), line_well=round(1 - (Q + SW) / W, 4),
                                            ceiling_halves_SB=round(float(np.nanmedian(half)), 4), two_wells=round(float(np.median(cross)), 4))
        print("comp", sn, json.dumps(res["common_components"][sn]), flush=True)
    json.dump(res, open(OUT + "ref99%s.json" % tag, "w"), indent=1); print("SUMMARY done", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        main(kinds=("single", "split"), components=False, tag="_single")
    elif len(sys.argv) > 1 and sys.argv[1] == "tsplit":
        main(kinds=("tsingle", "tsplit", "treated"), components=False, tag="_tsplit")
    else:
        main()
