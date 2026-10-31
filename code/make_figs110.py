"""Figures 1-4 of the same-well manuscript, redrawn (samewell/notes.md §110): one sentence per figure.
  Fig 1  the cell lines of a well shift together; two halves of one well do not differ that way (control pairs)
  Fig 2  replicate labels need checking: strongly responding pairs recognise each other, plate 13 mostly does not
  Fig 3  the same holds for drug-treated replicate wells, also with the same drug and the same position
  Fig 4  a model scores higher in the well whose other lines it was trained on than in a replicate well
Reads fig110.json / fig110.npz (A100, fig110.py) and figdata89.json (fig3a_compounds, control_per_plate); writes
fig1-fig4 (.pdf, .png) to <out_dir> and figstats110.json with the numbers for the captions. Example pairs are chosen
by rule, not by eye: the one at the median stripe score of its group (lower middle for even counts).
Usage: python make_figs110.py <dir with fig110.json, fig110.npz, figdata89.json> <out_dir>"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap

SAME, OTHER = "#BB5566", "#004488"
INK, INK2, G1, G2, G3 = "#0b0b0b", "#52514e", "#7f8c8d", "#a6acaf", "#d5d8dc"
MM = 1 / 25.4; W = 130
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "pdf.fonttype": 42})
DIV = LinearSegmentedColormap.from_list("div", ["#5e3c99", "#ffffff", "#e66101"])   # purple-orange: red and blue keep their meaning (this well / another well)
ST = {}


def save(fig, out, name):
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(out, name + "." + e), dpi=300)
    plt.close(fig)


def median_pick(ps):
    """The pair at the median of the per-pair stripe score (median over B lines of 'between')."""
    s = [np.median(p["stripe_between"]) for p in ps]; o = np.argsort(s)
    return ps[o[(len(o) - 1) // 2]]


def rowsB(Z, p, kind):
    """B lines, ordered by the noise of their within-well difference (quietest on top), the same order in every panel."""
    inA = np.array(p["inA"]); M = Z["p%d_%s" % (p["n"], kind)].astype(np.float64)[~inA]
    o = np.argsort(Z["p%d_within" % p["n"]].astype(np.float64)[~inA].std(1))
    return M[o]


def heat(ax, M, vmax, title=None):
    im = ax.imshow(M, aspect="auto", cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, pad=3)
    return im


def well(ax, x0, y0, s, seed, halves=False):
    ax.add_patch(FancyBboxPatch((x0, y0), s, s, boxstyle="round,pad=0.05,rounding_size=0.4", fc="white", ec=INK2, lw=0.6))
    rng = np.random.default_rng(seed); cols = [G1, G2, G3, INK2]
    for k in range(20):
        cx, cy = x0 + 0.35 + (k % 5) * (s - 0.7) / 4, y0 + 0.4 + (k // 5) * (s - 0.8) / 3
        ax.add_patch(Circle((cx + rng.uniform(-0.1, 0.1), cy + rng.uniform(-0.1, 0.1)), 0.17, fc=cols[k % 4], ec="none"))
    if halves:
        ax.plot([x0 + s / 2, x0 + s / 2], [y0 - 0.1, y0 + s + 0.1], color=INK2, lw=0.6, ls=(0, (2, 1.5)))


def fig1(J, Z, D, out):
    ps = [p for p in J["pairs"] if p["kind"] == "control"]
    p = median_pick(ps); Bt, Wt = rowsB(Z, p, "between"), rowsB(Z, p, "within")
    vmax = float(np.percentile(np.abs(Bt), 98))
    fig = plt.figure(figsize=(W * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1, 1], wspace=0.12, left=0.02, right=0.89, bottom=0.12, top=0.86)
    axa = fig.add_subplot(gs[0]); axa.set_xlim(0, 10); axa.set_ylim(0, 10); axa.set_aspect("equal"); axa.axis("off")
    well(axa, 0.4, 4.6, 4.0, 1, halves=True); well(axa, 5.6, 4.6, 4.0, 2)
    axa.text(2.4, 9.0, "control well 1", ha="center", fontsize=6); axa.text(7.6, 9.0, "control well 2", ha="center", fontsize=6)
    axa.text(0.4, 3.5, "b", fontsize=6, fontweight="bold"); axa.text(1.1, 3.5, "well 1 minus well 2", fontsize=6)
    axa.text(0.4, 2.5, "c", fontsize=6, fontweight="bold"); axa.text(1.1, 2.95, "one half of well 1 minus\nthe other half", fontsize=6, va="top")
    axa.text(0.4, 0.3, "row: a cell line; column: a gene", fontsize=5.5, color=INK2)
    axb = fig.add_subplot(gs[1]); im = heat(axb, Bt, vmax, "Two wells")
    axc = fig.add_subplot(gs[2]); heat(axc, Wt, vmax, "Two halves of one well")
    axb.set_ylabel("%d cell lines" % len(Bt)); axb.set_xlabel("%d genes" % Bt.shape[1]); axc.set_xlabel("same genes, same order")
    cax = fig.add_axes([0.905, 0.3, 0.012, 0.4]); cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5); cb.set_label("difference (log)", fontsize=5.5)
    for ax, l_ in ((axa, "a"), (axb, "b"), (axc, "c")):
        bb = ax.get_position(); fig.text(bb.x0 - 0.012, 0.96, l_, fontsize=8, fontweight="bold", va="top", color=INK)
    allp = {q["plate"]: [float(np.median(q["stripe_between"])), float(np.median(q["stripe_within"]))] for q in ps}
    ST["fig1"] = dict(plate=p["plate"], lines_shown=len(Bt), lines_A=int(sum(p["inA"])), genes=Bt.shape[1], vmax=vmax,
                      stripe_between_median=float(np.median(p["stripe_between"])), stripe_within_median=float(np.median(p["stripe_within"])),
                      per_plate_between_within=allp, plates_between_range=[min(v[0] for v in allp.values()), max(v[0] for v in allp.values())],
                      plates_within_range=[min(v[1] for v in allp.values()), max(v[1] for v in allp.values())])
    save(fig, out, "fig1")


def fig2(L, out):
    """Each row: one strongly responding pair, in the direction where the partner ranks lower. Grey cells = similarity
    of the well to every treated well of the partner's plate, sorted from most to least similar; blue = the labelled
    replicate, placed at its rank. Rows sorted by that rank within a group."""
    keys = [("G614", "Plates 6 and 14"), ("diffpos same name", "Same name,\ndifferent plates"), ("diffpos salt", "Different salt names,\ndifferent plates"),
            ("sameplate", "Two wells on\none plate"), ("plate13", "Plate 13 against\nthe original well")]
    rows = []
    for k, lab in keys:
        e = L[k]; rs = []
        for (s1, p1), (s2, p2) in e["lineup"]:
            r1 = sum(x > p1 for x in s1) / len(s1); r2 = sum(x > p2 for x in s2) / len(s2)
            s, p, r = (s1, p1, r1) if r1 >= r2 else (s2, p2, r2)
            rs.append((r, s, p))
        rows.append((k, lab, sorted(rs, key=lambda t: t[0])))
    n = sum(len(r) for _, _, r in rows); gap = 1.6
    fig = plt.figure(figsize=(W * MM, 92 * MM)); ax = fig.add_axes([0.2, 0.11, 0.68, 0.86])
    cm = LinearSegmentedColormap.from_list("g", ["#ffffff", INK]); y = 0; ST["fig2"] = {}
    for k, lab, rs in rows:
        y0 = y
        for r, s, p in rs:
            v = np.array(sorted(s + [p], reverse=True)); k_ = int(sum(x > p for x in s))
            ax.imshow(np.clip(v, 0, 1)[None, :], cmap=cm, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                      extent=(0, 1, y + 0.85, y + 0.15))
            ax.add_patch(Rectangle((0, y + 0.15), 1, 0.7, fc="none", ec=G3, lw=0.3, zorder=2))
            w = 1 / len(v); ax.add_patch(Rectangle((k_ * w, y + 0.1), w, 0.8, fc=OTHER, ec="none", zorder=3))
            y += 1
        ax.text(-0.02, (y0 + y) / 2, lab, ha="right", va="center", fontsize=6, color=SAME if k == "plate13" else INK)
        ST["fig2"][k] = dict(pairs=len(rs), recognised=int(sum(r < 0.1 for r, _, _ in rs)), median_rank_fraction=float(np.median([r for r, _, _ in rs])),
                             wells_in_lineup=[min(len(s) for _, s, _ in rs) + 1, max(len(s) for _, s, _ in rs) + 1])
        y += gap
    ax.axvline(0.1, color=INK2, lw=0.5, ls=(0, (1.5, 1.5)), zorder=4)
    ax.text(0.1, -0.6, "top 10%", ha="center", va="bottom", fontsize=5.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(y - gap, -1.2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["most similar", "", "least similar"])
    ax.set_xlabel("treated wells of the partner's plate, by similarity to the well")
    cax = fig.add_axes([0.91, 0.35, 0.012, 0.3]); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1)), cax=cax)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5); cb.set_label("similarity (r)", fontsize=5.5)
    save(fig, out, "fig2")


def fig3(J, Z, out):
    g1 = [p for p in J["pairs"] if p["kind"] == "diffpos" and p["cls"] == "confirmed"]
    g2 = [p for p in J["pairs"] if p["kind"] == "G614"]
    p1, p2 = median_pick(g1), median_pick(g2)
    B1, B2 = rowsB(Z, p1, "between"), rowsB(Z, p2, "between")
    vmax = float(np.percentile(np.abs(np.concatenate([B1.ravel(), B2.ravel()])), 98))
    fig = plt.figure(figsize=(W * MM, 58 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.05, right=0.88, bottom=0.12, top=0.82)
    t1 = "Same drug and dose, different positions\n(%s, plates %s and %s)" % (p1["comp"], p1["pa"][5:], p1["pb"][5:])
    t2 = "Same drug, dose and position\n(%s, plates 6 and 14)" % p2["comp"]
    ax1 = fig.add_subplot(gs[0]); im = heat(ax1, B1, vmax, t1); ax1.set_ylabel("%d cell lines" % len(B1)); ax1.set_xlabel("%d genes" % B1.shape[1])
    ax2 = fig.add_subplot(gs[1]); heat(ax2, B2, vmax, t2); ax2.set_xlabel("%d genes" % B2.shape[1]); ax2.set_ylabel("%d cell lines" % len(B2))
    cax = fig.add_axes([0.895, 0.3, 0.012, 0.4]); cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5); cb.set_label("difference (log)", fontsize=5.5)
    for ax, l_ in ((ax1, "a"), (ax2, "b")):
        bb = ax.get_position(); fig.text(bb.x0 - 0.012, 0.97, l_, fontsize=8, fontweight="bold", va="top", color=INK)
    gm = lambda ps, k: [float(np.median(q[k])) for q in ps]
    ST["fig3"] = dict(a=dict(cond=p1["cond"], plates=[p1["pa"], p1["pb"]], lines=len(B1), stripe_between=float(np.median(p1["stripe_between"])), stripe_within=float(np.median(p1["stripe_within"]))),
                      b=dict(cond=p2["cond"], lines=len(B2), stripe_between=float(np.median(p2["stripe_between"])), stripe_within=float(np.median(p2["stripe_within"]))),
                      vmax=vmax, groups={k: dict(pairs=len(ps), between_median=float(np.median(gm(ps, "stripe_between"))), within_median=float(np.median(gm(ps, "stripe_within"))),
                                                 between_range=[min(gm(ps, "stripe_between")), max(gm(ps, "stripe_between"))])
                                         for k, ps in (("diffpos concordant", g1), ("G614", g2))})
    save(fig, out, "fig3")


def fig4(D, out):
    FC = D["fig3a_compounds"]
    fig = plt.figure(figsize=(80 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35, left=0.17, right=0.97, bottom=0.22, top=0.86)
    ST["fig4"] = {}
    for c, (g, lab) in enumerate((("G614", "Plates 6 and 14"), ("diffpos", "Different positions"))):
        cs = [x for x in FC if x["group"] == g]; a = np.array([x["pmean_train"] for x in cs]); b = np.array([x["pmean_held"] for x in cs])
        ax = fig.add_subplot(gs[c])
        for u, v in zip(a, b):
            ax.plot([0, 1], [u, v], color=G3, lw=0.5, zorder=1)
        ma, mb = np.median(a), np.median(b)
        ax.plot([0, 1], [ma, mb], color=INK, lw=1.2, zorder=3, solid_capstyle="round")
        ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.15, 0.9); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax.set_xticks([0, 1])
        ax.set_xticklabels(["well whose other\nlines were\ntrained on", "replicate well,\nno line\ntrained on"])
        ax.set_title("%s\n%d compounds" % (lab, len(cs)), pad=3)
        if c == 0:
            ax.set_ylabel("score of a held-out line (r)")
        fig.text(ax.get_position().x0 - (0.13 if c == 0 else 0.07), 0.97, "ab"[c], fontsize=8, fontweight="bold", va="top", color=INK)
        ST["fig4"][g] = dict(compounds=len(cs), median_train=float(ma), median_held=float(mb), falling=int((b < a).sum()))
    save(fig, out, "fig4")


def figS1(J, Z, out):
    """All 13 control pairs, as Fig. 1b (two wells), each on its own colour scale."""
    ps = sorted([p for p in J["pairs"] if p["kind"] == "control"], key=lambda p: int(p["plate"][5:]))
    fig = plt.figure(figsize=(W * MM, 112 * MM))
    gs = fig.add_gridspec(3, 5, wspace=0.12, hspace=0.42, left=0.02, right=0.98, bottom=0.03, top=0.95)
    for k, p in enumerate(ps):
        ax = fig.add_subplot(gs[k // 5, k % 5]); M = rowsB(Z, p, "between")
        heat(ax, M, float(np.percentile(np.abs(M), 98)))
        ax.set_title("plate %s   %.2f / %.2f" % (p["plate"][5:], np.median(p["stripe_between"]), np.median(p["stripe_within"])), pad=2, fontsize=5.5)
    save(fig, out, "figS1")


def main():
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    J = json.load(open(os.path.join(src, "fig110.json"))); Z = np.load(os.path.join(src, "fig110.npz")); D = json.load(open(os.path.join(src, "figdata89.json")))
    fig1(J, Z, D, out); figS1(J, Z, out); fig2(json.load(open(os.path.join(src, "fig110_labels.json"))), out); fig3(J, Z, out); fig4(D, out)
    json.dump(ST, open(os.path.join(out, "figstats110.json"), "w"), indent=1); print(json.dumps(ST, indent=1))


if __name__ == "__main__":
    main()
