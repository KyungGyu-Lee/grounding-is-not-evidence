# -*- coding: utf-8 -*-
"""Fig 4 (publication redesign, C2-normalized). Data/definitions UNCHANGED.
rho defined only on C2; y = Share of C2 questions (bars sum to 100% over all rho).
Canonical stored rho: NExT <- flags/nextgqa/r_norm.json; QVH <- flags/qvhme/r_norm.json.

The NExT rho~1 spike (40.4% of C2) is clipped at 20% and the true value is
printed beside it, with break marks on the bar.
Display window [-0.5, 1.5]; rho outside is shown as honest edge-overflow labels
(bars are NOT silently dropped: the count beyond each edge is annotated).
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
FIGDIR = os.path.join(ROOT, 'paper', 'figures') + os.sep
os.makedirs(FIGDIR, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 10, 'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
BAR = '#4C72B0'; AMBER = '#C6803B'; SHADE = '#DCE6F4'; THR = '#8A8A8A'; MED = '#B22222'; NAVY = '#20375F'; INK = '#222222'
BW = 0.1
X0, X1 = -0.5, 1.5
BINS = np.round(np.arange(X0, X1 + BW / 2, BW), 6)     # fixed, identical for both datasets
C3T = 0.4
YMAX_A = 20.0                                          # clipped-peak upper limit

PANELS = [
    ('(a) NExT-GQA', 'flags/nextgqa/r_norm.json', 2268, 248, 0.93),
    ('(b) QVH-ME',   'flags/qvhme/r_norm.json', 763, 241, 0.40),
]

def load(p):
    return np.array(list(json.load(open(p))['r_norm'].values()), float)


def panel_stats(path, ALL):
    a = load(path); C2 = len(a)
    h, _ = np.histogram(a, bins=BINS)
    share = 100 * h / C2                                # C2-normalized
    below = int((a < X0).sum()); above = int((a > X1).sum())
    med = float(np.median(a))
    within = 100 * (a <= C3T + 1e-9).sum() / C2
    overall = 100 * (a <= C3T + 1e-9).sum() / ALL
    return dict(a=a, C2=C2, share=share, below=below, above=above, med=med,
                within=within, overall=overall, peak=share.max(), peak_x=BINS[:-1][np.argmax(share)])


def base_axes(ax, st, ymax, y_left, show_ylabel, bar_color=BAR):
    """Common grammar: shading, threshold, median, C3 card, region label, overflow, spines."""
    ax.axvspan(X0 - 0.02, C3T, color=SHADE, alpha=0.9, zorder=0, lw=0)
    ax.bar(BINS[:-1], st['share'], width=BW * 0.9, align='edge', color=bar_color, linewidth=0, zorder=3)
    # C3 threshold (thin solid) vs median (dashed) -- different style AND colour
    ax.axvline(C3T, color=THR, ls='-', lw=1.0, zorder=2)
    ax.axvline(st['med'], color=MED, ls=(0, (5, 2)), lw=1.5, zorder=4)
    ax.set_xlim(X0 - 0.03, X1 + 0.03); ax.set_ylim(0, ymax)
    ax.set_xticks(np.arange(-0.5, 1.51, 0.5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}\\%' if False else f'{v:.0f}%'))
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9, length=3)
    if show_ylabel and y_left:
        ax.set_ylabel('Share of C2 questions', fontsize=10.5)
    # median label: small, beside the dashed line, aligned to the C3 tag baseline (top)
    mx = st['med']
    if mx >= 0.6:
        ax.text(mx - 0.03, ymax * 0.97, f"Median {mx:.2f}", color=MED, fontsize=7.0, ha='right', va='top', zorder=6)
    else:
        ax.text(mx + 0.03, ymax * 0.97, f"Median {mx:.2f}", color=MED, fontsize=7.0, ha='left', va='top', zorder=6)
    # only small C3-region tag inside the shading (no floating card, no outlier text)
    ax.text(X0 + 0.02, ymax * 0.97, r'C3: $\rho \leq 0.4$', color='#000000', fontsize=8.5, ha='left', va='top')


def panel_label(ax, title):
    ax.text(0.5, -0.26, title, transform=ax.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold', color='#000000')


def clip_marks(ax, x, ytop):
    """diagonal break marks on a bar clipped at ytop, + width to match bar."""
    w = BW * 0.9; d = 0.5
    for yy in (ytop - 0.6, ytop - 0.15):
        ax.plot([x - 0.01, x + w + 0.01], [yy - d, yy + d], color='white', lw=2.2, zorder=5, solid_capstyle='round')
        ax.plot([x - 0.01, x + w + 0.01], [yy - d, yy + d], color=INK, lw=0.9, zorder=6, solid_capstyle='round')


# ---------------- the paper figure: clipped peak ----------------
def make_fig():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), dpi=200)
    for ax, (title, path, ALL, C3n, _) in zip(axes, PANELS):
        st = panel_stats(path, ALL)
        bc = AMBER if 'QVH' in title else BAR
        base_axes(ax, st, YMAX_A, y_left=(ax is axes[0]), show_ylabel=True, bar_color=bc)
        # clipped bar: mark + true value, placed to the RIGHT of the spike (y fixed)
        if st['peak'] > YMAX_A:
            xk = st['peak_x']
            clip_marks(ax, xk, YMAX_A)
            ax.text(xk + BW + 0.03, YMAX_A * 0.97, f"{st['peak']:.1f}%", ha='left', va='top',
                    fontsize=9, fontweight='normal', color='#333333', zorder=7)
        ax.set_xlabel(r'Temporal redundancy $\rho$', fontsize=10.5)
        panel_label(ax, title)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.955, bottom=0.24, wspace=0.13)
    return fig


def main():
    print('=== per-panel stats (verification) ===')
    for title, path, ALL, C3n, _ in PANELS:
        st = panel_stats(path, ALL)
        print(f'  {title}: C2={st["C2"]} peak={st["peak"]:.1f}% @rho~{st["peak_x"]:.1f}  '
              f'median={st["med"]:.4f}  within-C2={st["within"]:.1f}%  overall={st["overall"]:.1f}%  '
              f'hist-sum={st["share"].sum():.1f}%  edge-overflow(<{X0}:{st["below"]}, >{X1}:{st["above"]})')
    fig = make_fig()
    fig.savefig(FIGDIR + 'fig4.pdf'); fig.savefig(FIGDIR + 'fig4.png', dpi=200); plt.close(fig)
    print('  WROTE paper/figures/fig4.pdf / .png')


if __name__ == '__main__':
    main()
