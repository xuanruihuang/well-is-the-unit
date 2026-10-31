"""Figures 1, 3 and 4 of the same-well manuscript redrawn after the rewrite (samewell/notes.md §111): each figure opens
with a schematic of the comparison, then shows the result over all the data, then an example, as in Squair et al. 2021.
Colour code throughout: red, taken from the same well (or pair); blue, taken from the replicate well (or other plates).
  Fig 1  a,b,c  halves schematic and the example heat maps (as fig110); d schematic of the prediction; e every plate
  Fig 3  a schematic of same-well against replicate-well prediction; b every pair; c,d example heat maps (as fig110)
  Fig 4  a schematic of the evaluation; b baseline and three published models; c ceilings from halves against two wells
Fig 2 and Supplementary Fig. 1 are unchanged (make_figs110.py). Reads fig110.json / fig110.npz / figdata89.json (v120
outputs, as make_figs110.py); the model scores of Fig. 4b are those of Supplementary Table S7 (tabS4 in supp_tables.tex).
Usage: python make_figs111.py <dir with fig110.json, fig110.npz, figdata89.json> <out_dir>"""
import sys, os, json
import numpy as np
import make_figs110 as M
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt = M.plt; MM = M.MM; INK, INK2, G1, G2, G3, SAME, OTHER = M.INK, M.INK2, M.G1, M.G2, M.G3, M.SAME, M.OTHER
W2 = 180
ST = {}
# Supplementary Table S7 (tabS4): score against the well that supplied training data or inputs, against the replicate
# well, and the target line's own response in the replicate well (treated-well reference, concordant pairs); for the
# perturbation mean, the test lines' own agreement between the two wells over the same records (eval82 'ceiling')
MODELS = [("Perturbation mean", "plates 6 and 14 (77)", 0.74, 0.51, 0.54), ("", "different positions (6)", 0.70, 0.36, 0.66),
          ("BioPert", "plates 6 and 14 (77)", 0.63, 0.43, 0.46), ("", "different positions (9)", 0.73, 0.51, 0.51),
          ("Transfer model", "plates 6 and 14 (77)", 0.71, 0.49, 0.54), ("", "different positions (9)", 0.88, 0.68, 0.67),
          ("Rhaister (own metric)", "plates 6 and 14", 0.93, 0.65, 0.67)]


def letter(fig, ax, l_, dx=0.012, y=None):
    bb = ax.get_position(); fig.text(bb.x0 - dx, bb.y1 + 0.02 if y is None else y, l_, fontsize=8, fontweight="bold", va="bottom", color=INK)


def wellbox(ax, x0, y0, s, cols, label=None, lw=0.6, ec=None):
    """A well of side s with 4 x 3 dots (cell lines); cols: one colour per dot (None = hollow black ring)."""
    ax.add_patch(FancyBboxPatch((x0, y0), s, s, boxstyle="round,pad=0.04,rounding_size=0.35", fc="white", ec=ec or INK2, lw=lw))
    for k, c in enumerate(cols):
        cx, cy = x0 + 0.3 * s / 1.6 + (k % 4) * (s - 0.6 * s / 1.6) / 3, y0 + 0.32 * s / 1.6 + (k // 4) * (s - 0.64 * s / 1.6) / 2
        if c is None:
            ax.add_patch(Circle((cx, cy), 0.12 * s / 1.6, fc="white", ec=INK, lw=0.9))
        else:
            ax.add_patch(Circle((cx, cy), 0.12 * s / 1.6, fc=c, ec="none"))
    if label:
        ax.text(x0 + s / 2, y0 + s + 0.15, label, ha="center", va="bottom", fontsize=5.5, color=INK)


def arrow(ax, a, b, c, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=c, connectionstyle="arc3,rad=%g" % rad))


def dumbbell(ax, ys, a, b, ref=None):
    for y, u, v in zip(ys, a, b):
        ax.plot([v, u], [y, y], color=G3, lw=1.0, zorder=1, solid_capstyle="round")
    ax.scatter(a, ys, s=16, color=SAME, zorder=3, lw=0); ax.scatter(b, ys, s=16, color=OTHER, zorder=3, lw=0)
    if ref is not None:
        for y, r in zip(ys, ref):
            if r is not None:
                ax.plot([r, r], [y - 0.36, y + 0.36], color=INK, lw=0.9, zorder=4)


def fig1(J, Z, D, out):
    ps = [p for p in J["pairs"] if p["kind"] == "control"]
    p = M.median_pick(ps); Bt, Wt = M.rowsB(Z, p, "between"), M.rowsB(Z, p, "within")
    vmax = float(np.percentile(np.abs(Bt), 98))
    fig = plt.figure(figsize=(W2 * MM, 118 * MM))
    top = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.12, left=0.02, right=0.9, bottom=0.53, top=0.93)
    axa = fig.add_subplot(top[0]); axa.set_xlim(0, 10); axa.set_ylim(0, 10); axa.set_aspect("equal"); axa.axis("off")
    M.well(axa, 0.4, 4.6, 4.0, 1, halves=True); M.well(axa, 5.6, 4.6, 4.0, 2, halves=True)
    axa.text(2.4, 9.0, "control well 1", ha="center", fontsize=6); axa.text(7.6, 9.0, "control well 2", ha="center", fontsize=6)
    axa.text(0.4, 3.5, "b", fontsize=6, fontweight="bold"); axa.text(1.1, 3.5, "half of well 1 minus half of well 2", fontsize=6)
    axa.text(0.4, 2.5, "c", fontsize=6, fontweight="bold"); axa.text(1.1, 2.95, "one half of well 1 minus\nthe other half", fontsize=6, va="top")
    axa.text(0.4, 0.3, "dot: a cell line; heat-map row: a cell line;\ncolumn: a gene", fontsize=5.5, color=INK2)
    axb = fig.add_subplot(top[1]); im = M.heat(axb, Bt, vmax, "Two wells")
    axc = fig.add_subplot(top[2]); M.heat(axc, Wt, vmax, "Two halves of one well")
    axb.set_ylabel("%d cell lines" % len(Bt)); axb.set_xlabel("%d genes" % Bt.shape[1]); axc.set_xlabel("same genes, same order")
    cax = fig.add_axes([0.915, 0.62, 0.01, 0.22]); cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5); cb.set_label("difference in expression (log)", fontsize=5.5)
    # d: the prediction
    axd = fig.add_axes([0.02, 0.05, 0.4, 0.36]); axd.set_xlim(0, 12); axd.set_ylim(0, 7.2); axd.set_aspect("equal"); axd.axis("off")
    axd.add_patch(FancyBboxPatch((0.2, 2.6), 5.0, 3.4, boxstyle="round,pad=0.05,rounding_size=0.3", fc="#f7f7f7", ec="none"))
    axd.add_patch(FancyBboxPatch((6.8, 2.6), 5.0, 3.4, boxstyle="round,pad=0.05,rounding_size=0.3", fc="#f7f7f7", ec="none"))
    axd.text(2.7, 6.3, "this plate (positions H11, H12)", ha="center", fontsize=6); axd.text(9.3, 6.3, "another plate (same positions)", ha="center", fontsize=6)
    cs = lambda c: [None] + [c] * 11
    wellbox(axd, 0.5, 3.0, 2.1, cs(SAME), "control well 1"); wellbox(axd, 2.8, 3.0, 2.1, cs(SAME), "control well 2")
    wellbox(axd, 7.1, 3.0, 2.1, [G2] + [OTHER] * 11, "control well 1"); wellbox(axd, 9.4, 3.0, 2.1, [G2] + [OTHER] * 11, "control well 2")
    axd.add_patch(Circle((0.6, 1.75), 0.14, fc="white", ec=INK, lw=0.9)); axd.text(0.95, 1.75, "a cell line: its difference between the two wells", va="center", fontsize=5.5)
    axd.add_patch(Circle((0.6, 1.05), 0.14, fc=SAME, ec="none")); axd.text(0.95, 1.05, "predicted by the other lines of the same pair", va="center", fontsize=5.5)
    axd.add_patch(Circle((0.6, 0.35), 0.14, fc=OTHER, ec="none")); axd.text(0.95, 0.35, "predicted by the other lines of the control pairs of other plates\n(Emerald Bay: of the other pair of the same plate)", va="center", fontsize=5.5)
    # e: every plate
    cp = D["control_per_plate"]; pl = sorted(cp, key=lambda k: cp[k][0])
    axe = fig.add_axes([0.56, 0.08, 0.4, 0.36])
    ys = list(range(len(pl))); a = [cp[k][0] for k in pl]; b = [cp[k][1] for k in pl]
    dumbbell(axe, ys, a, b)
    ye = len(pl) + 1.0
    dumbbell(axe, [ye], [0.65], [0.24])
    axe.set_yticks(ys + [ye]); axe.set_yticklabels(["plate %s" % k[5:] for k in pl] + ["Emerald Bay"], fontsize=5.5)
    axe.set_ylim(-0.8, ye + 0.8); axe.set_xlim(0, 0.85); axe.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axe.set_xlabel("prediction of a line's difference (median r)")
    for ax, l_ in ((axa, "a"), (axb, "b"), (axc, "c")):
        letter(fig, ax, l_, y=0.955)
    fig.text(0.02, 0.45, "d", fontsize=8, fontweight="bold", color=INK); fig.text(0.515, 0.45, "e", fontsize=8, fontweight="bold", color=INK)
    ST["fig1"] = dict(plate=p["plate"], per_plate=cp, all_plates_same_higher=int(sum(u > v for u, v in zip(a, b))))
    M.save(fig, out, "fig1")


def pair_xy(D, group, cls=None):
    ps = [p for p in D["pairs"] if p["group"] == group and (cls is None or p["cls"] == cls)]
    return ps


def fig3(J, Z, D, out):
    g1 = [p for p in J["pairs"] if p["kind"] == "diffpos" and p["cls"] == "confirmed"]
    g2 = [p for p in J["pairs"] if p["kind"] == "G614"]
    # example at different positions: among the concordant pairs whose lines' responses agree between the two wells at
    # least at the group median (so that the drug response reproduces), the pair at the median stripe score
    cross = {(q["comp"], tuple(sorted(q["plates"]))): q["cross"] for q in D["pairs"] if q["group"] == "diffpos" and q["cls"] == "confirmed"}
    cm = float(np.median(list(cross.values())))
    g1r = [p for p in g1 if cross[(p["comp"], tuple(sorted([p["pa"], p["pb"]])))] >= cm]
    p1, p2 = M.median_pick(g1r), M.median_pick(g2)
    B1, B2 = M.rowsB(Z, p1, "between"), M.rowsB(Z, p2, "between")
    vmax = float(np.percentile(np.abs(np.concatenate([B1.ravel(), B2.ravel()])), 98))
    fig = plt.figure(figsize=(W2 * MM, 125 * MM))
    # a: schematic
    axa = fig.add_axes([0.02, 0.56, 0.44, 0.38]); axa.set_xlim(0, 12); axa.set_ylim(0, 7.6); axa.set_aspect("equal"); axa.axis("off")
    axa.add_patch(FancyBboxPatch((0.3, 2.9), 4.6, 3.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc="#f7f7f7", ec="none"))
    axa.add_patch(FancyBboxPatch((7.1, 2.9), 4.6, 3.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc="#f7f7f7", ec="none"))
    axa.text(2.6, 6.8, "plate p", ha="center", fontsize=6); axa.text(9.4, 6.8, "plate q", ha="center", fontsize=6)
    wellbox(axa, 1.4, 3.3, 2.4, [None] + [SAME] * 11, "drug D, dose c")
    wellbox(axa, 8.2, 3.3, 2.4, [G2] + [OTHER] * 11, "drug D, dose c (replicate)")
    axa.add_patch(Circle((0.6, 2.25), 0.14, fc="white", ec=INK, lw=0.9)); axa.text(0.95, 2.25, "a cell line: its response in the well on plate p", va="center", fontsize=5.5)
    axa.add_patch(Circle((0.6, 1.6), 0.14, fc=G2, ec="none")); axa.text(0.95, 1.6, "the same line in the replicate (not used)", va="center", fontsize=5.5)
    axa.add_patch(Circle((0.6, 0.95), 0.14, fc=SAME, ec="none")); axa.text(0.95, 0.95, "same-well prediction: mean response of the other lines in that well", va="center", fontsize=5.5)
    axa.add_patch(Circle((0.6, 0.3), 0.14, fc=OTHER, ec="none")); axa.text(0.95, 0.3, "replicate-well prediction: mean response of the other lines in the replicate", va="center", fontsize=5.5)
    # b: every pair
    axb = fig.add_axes([0.6, 0.58, 0.3, 0.35])
    lo, hi = -0.3, 0.95
    axb.plot([lo, hi], [lo, hi], color=G2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    G = pair_xy(D, "G614"); C = pair_xy(D, "diffpos", "confirmed")
    axb.scatter([p["other"] for p in G], [p["same"] for p in G], s=7, color=G1, lw=0, alpha=0.85, zorder=2, label="plates 6 and 14, same position (%d pairs)" % len(G))
    axb.scatter([p["other"] for p in C], [p["same"] for p in C], s=12, facecolor="white", edgecolor=INK, lw=0.7, zorder=3, label="different positions, concordant (%d)" % len(C))
    axb.set_xlim(lo, hi); axb.set_ylim(lo, hi); axb.set_aspect("equal")
    axb.set_xticks([0, 0.4, 0.8]); axb.set_yticks([0, 0.4, 0.8])
    axb.set_xlabel("replicate-well prediction (r)", color=OTHER); axb.set_ylabel("same-well prediction (r)", color=SAME)
    axb.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1)
    axb.text(-0.05, -0.02, "equal", color=G1, fontsize=5, ha="left", va="bottom", rotation=45, rotation_mode="anchor")
    # c, d: example heat maps
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.05, right=0.88, bottom=0.06, top=0.42)
    t1 = "Same drug and dose, different positions\n(%s, plates %s and %s)" % (p1["comp"], p1["pa"][5:], p1["pb"][5:])
    t2 = "Same drug, dose and position\n(%s, plates 6 and 14)" % p2["comp"]
    axc = fig.add_subplot(gs[0]); im = M.heat(axc, B1, vmax, t1); axc.set_ylabel("%d cell lines" % len(B1)); axc.set_xlabel("%d genes" % B1.shape[1])
    axd = fig.add_subplot(gs[1]); M.heat(axd, B2, vmax, t2); axd.set_xlabel("%d genes" % B2.shape[1]); axd.set_ylabel("%d cell lines" % len(B2))
    cax = fig.add_axes([0.895, 0.14, 0.01, 0.2]); cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5); cb.set_label("difference in response (log)", fontsize=5.5)
    fig.text(0.02, 0.955, "a", fontsize=8, fontweight="bold", color=INK); fig.text(0.53, 0.955, "b", fontsize=8, fontweight="bold", color=INK)
    letter(fig, axc, "c", dx=0.03, y=0.455); letter(fig, axd, "d", dx=0.03, y=0.455)
    ST["fig3"] = dict(c=dict(cond=p1["cond"], plates=[p1["pa"], p1["pb"]], lines=len(B1), cross=cross[(p1["comp"], tuple(sorted([p1["pa"], p1["pb"]])))], cross_median=cm, eligible=len(g1r),
                             stripe_between=float(np.median(p1["stripe_between"])), stripe_within=float(np.median(p1["stripe_within"]))), d=dict(cond=p2["cond"], lines=len(B2)),
                      b=dict(G614=dict(pairs=len(G), above=int(sum(p["same"] > p["other"] for p in G)), median_same=float(np.median([p["same"] for p in G])), median_other=float(np.median([p["other"] for p in G]))),
                             diffpos_confirmed=dict(pairs=len(C), above=int(sum(p["same"] > p["other"] for p in C)), median_same=float(np.median([p["same"] for p in C])), median_other=float(np.median([p["other"] for p in C])))))
    M.save(fig, out, "fig3")


def fig4(D, out):
    fig = plt.figure(figsize=(W2 * MM, 68 * MM))
    # a: schematic
    axa = fig.add_axes([0.0, 0.04, 0.31, 0.86]); axa.set_xlim(-0.9, 12); axa.set_ylim(0, 8); axa.set_aspect("equal"); axa.axis("off")
    wellbox(axa, 0.2, 4.4, 2.4, [None] + [SAME] * 11, "one well")
    axa.text(2.9, 5.6, "taken from this well\n(training data or input)\nand scored in it", va="center", fontsize=5.5, color=SAME)
    wellbox(axa, 0.2, 0.9, 2.4, [G2] + [OTHER] * 11, "replicate well")
    wellbox(axa, 6.0, 0.9, 2.4, [None] + [G3] * 11, "scored well")
    arrow(axa, (2.75, 2.1), (5.85, 2.1), OTHER)
    axa.text(4.3, 2.35, "taken from", ha="center", va="bottom", fontsize=5.5, color=OTHER)
    axa.text(8.6, 2.1, "scored here", va="center", fontsize=5.5, color=INK)
    axa.add_patch(Circle((6.3, 7.4), 0.14, fc="white", ec=INK, lw=0.9)); axa.text(6.6, 7.4, "target line: its response to this\ndrug-dose never in training", va="center", fontsize=5.5)
    # b: models
    axb = fig.add_axes([0.43, 0.25, 0.25, 0.68])
    ys = [0, 1, 2.6, 3.6, 5.2, 6.2, 7.8]
    dumbbell(axb, ys, [m[2] for m in MODELS], [m[3] for m in MODELS], [m[4] for m in MODELS])
    axb.set_yticks(ys); axb.set_yticklabels([m[1] for m in MODELS], fontsize=5.5); axb.invert_yaxis()
    for y, m in zip(ys, MODELS):
        if m[0]:
            axb.text(-0.02, y - 0.75, m[0], transform=axb.get_yaxis_transform(), ha="right", va="center", fontsize=6, fontweight="bold", color=INK)
    axb.set_xlim(0.3, 1.0); axb.set_xticks([0.4, 0.6, 0.8, 1.0]); axb.set_xlabel("score of the target line (r)")
    axb.set_ylim(8.5, -1.4)
    axb.scatter([], [], s=16, color=SAME, label="taken from the scored well"); axb.scatter([], [], s=16, color=OTHER, label="taken from the replicate well")
    axb.plot([], [], color=INK, lw=0.9, label="target line's own replicate")
    axb.legend(loc="upper left", bbox_to_anchor=(-1.0, -0.13), ncol=3, frameon=False, fontsize=5.2, handletextpad=0.3, columnspacing=1.0)
    # c: ceilings
    axc = fig.add_axes([0.765, 0.25, 0.22, 0.68])
    lo, hi = -0.2, 1.05
    axc.plot([lo, hi], [lo, hi], color=G2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    G = pair_xy(D, "G614"); C = pair_xy(D, "diffpos", "confirmed")
    axc.scatter([p["ceiling"] for p in G], [p["cross"] for p in G], s=7, color=G1, lw=0, alpha=0.85, zorder=2, label="plates 6 and 14 (%d pairs)" % len(G))
    axc.scatter([p["ceiling"] for p in C], [p["cross"] for p in C], s=12, facecolor="white", edgecolor=INK, lw=0.7, zorder=3, label="different positions, concordant (%d)" % len(C))
    axc.set_xlim(lo, hi); axc.set_ylim(lo, hi); axc.set_aspect("equal"); axc.set_xticks([0, 0.5, 1]); axc.set_yticks([0, 0.5, 1])
    axc.set_xlabel("two halves of one well (r, corrected)"); axc.set_ylabel("two replicate wells (r)")
    axc.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.17), frameon=False, fontsize=5.2, handletextpad=0.2)
    axc.text(0.0, 0.03, "equal", color=G1, fontsize=5, ha="left", va="bottom", rotation=45, rotation_mode="anchor")
    fig.text(0.01, 0.95, "a", fontsize=8, fontweight="bold", color=INK); fig.text(0.31, 0.95, "b", fontsize=8, fontweight="bold", color=INK)
    fig.text(0.71, 0.95, "c", fontsize=8, fontweight="bold", color=INK)
    ST["fig4"] = dict(models=MODELS, c=dict(G614=dict(pairs=len(G), below=int(sum(p["cross"] < p["ceiling"] for p in G)), median_ceiling=float(np.median([p["ceiling"] for p in G])), median_cross=float(np.median([p["cross"] for p in G]))),
                                            diffpos_confirmed=dict(pairs=len(C), below=int(sum(p["cross"] < p["ceiling"] for p in C)), median_ceiling=float(np.median([p["ceiling"] for p in C])), median_cross=float(np.median([p["cross"] for p in C])))))
    M.save(fig, out, "fig4")


def main():
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    J = json.load(open(os.path.join(src, "fig110.json"))); Z = np.load(os.path.join(src, "fig110.npz")); D = json.load(open(os.path.join(src, "figdata89.json")))
    fig1(J, Z, D, out); fig3(J, Z, D, out); fig4(D, out)
    json.dump(ST, open(os.path.join(out, "figstats111.json"), "w"), indent=1); print(json.dumps(ST, indent=1))


if __name__ == "__main__":
    main()
