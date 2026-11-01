"""C3 - legend repositioning for the two supplementary figures.

Re-renders

  * Fig S1  fig_rho_sweep.pdf        (from A2, analysis/a2_threshold_sensitivity.py)
  * Fig S2  fig_draw_subsampling.pdf (from A3, analysis/a3_draw_variance.py)

with the legends moved off the data lines.  NOTHING is recomputed: the numbers
are read back out of the CSVs the two A-stage scripts wrote in the same run
that produced the original PDFs

  * analysis/A2_threshold_sensitivity/sweep.csv
  * analysis/A3_draw_variance/draw_subsampling.csv

and, after rendering, every vertex matplotlib actually drew is compared back
against those CSVs.  Figure size, fonts, colours, axes, ticks and line styles
are copied verbatim from the two source scripts; only legend placement (and,
for Fig S2, the model->colour assignment the shared legend forces) changes.

Existing files are not touched - the new PDFs go to
analysis/C3_figures/.
"""

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_A2 = f'{ROOT}/analysis/A2_threshold_sensitivity'
SRC_A3 = f'{ROOT}/analysis/A3_draw_variance'
OUTDIR = f'{ROOT}/analysis/C3_figures'

# ---- verbatim from the source scripts -------------------------------------
TAU_R_BASE = 0.4
TAU_R_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]
NDRAW = 5

MODELS = {'NExT': ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VTimeLLM'],
          'QVH': ['VideoChat-R1', 'Qwen', 'GVLLM', 'VideoMind', 'GroundVTS',
                  'TimeChat', 'VTimeLLM']}
DISP = {'Qwen': 'Qwen2.5-VL', 'VideoMind': 'VideoMind',
        'GVLLM': 'Grounded-VideoLLM', 'GroundVTS': 'GroundVTS',
        'VTimeLLM': 'VTimeLLM', 'TimeChat': 'TimeChat',
        'VideoChat-R1': 'VideoChat-R1'}
NOT_INTERPRETED = {'VTimeLLM', 'TimeChat'}
PALETTE = ['#4C72B0', '#C6803B', '#55A868', '#8172B2', '#C44E52',
           '#64B5CD', '#8C8C8C']
C_NEXT, C_QVH = '#4C72B0', '#C6803B'          # paper Fig 5 palette

RC = {'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8,
      'axes.linewidth': 0.7, 'axes.edgecolor': '#333333',
      'xtick.color': '#333333', 'ytick.color': '#333333',
      'pdf.fonttype': 42, 'ps.fonttype': 42}

# Table 5 order, union of the two panels.  No \ddagger: the current figure
# carries only the \dagger marks, so VideoMind stays unmarked here.
LEGEND_ORDER = ['Qwen', 'VideoMind', 'GVLLM', 'GroundVTS', 'VideoChat-R1',
                'TimeChat', 'VTimeLLM']

# ---- new: one colour per model, shared by both panels of Fig S2 -----------
# A single legend only works if a model has one colour, and today it does not
# (the source script indexes PALETTE by the model's position within each
# panel, so e.g. Qwen2.5-VL is #4C72B0 on the left and #C6803B on the right).
# The same PALETTE is kept, now indexed by the legend order above, which
# leaves the two non-interpreted models on the two muted entries.
MODEL_COLOR = {mk: PALETTE[i] for i, mk in enumerate(LEGEND_ORDER)}


def label_of(mk):
    return DISP[mk] + ('†' if mk in NOT_INTERPRETED else '')


# ----------------------------------------------------------------- Fig S1
def fig_rho_sweep(check):
    """A2 render_fig, legend -> lower left, tau label -> bottom of the rule."""
    sw = pd.read_csv(f'{SRC_A2}/sweep.csv')
    sw = sw[sw.sweep == 'rho'].copy()

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=200)
    drawn = {}
    for bench, key, col in (('NExT-GQA', 'NExT', C_NEXT),
                            ('QVH-ME', 'QVH', C_QVH)):
        sub = sw[sw.bench == bench].sort_values('tau_rho')
        xs = list(sub.tau_rho)
        assert xs == TAU_R_GRID, xs
        for mk in MODELS[key]:
            ys = list(sub[f'C3_SG__{mk}'])
            ln, = ax.plot(xs, ys, color=col, lw=0.6, alpha=0.38, zorder=2)
            drawn[(bench, mk)] = (ln, np.asarray(ys))
        ys = list(sub.meanSG_C3)
        ln, = ax.plot(xs, ys, color=col, lw=1.8, marker='o', ms=3.2,
                      label=bench, zorder=3)
        drawn[(bench, '<mean>')] = (ln, np.asarray(ys))

    ax.axvline(TAU_R_BASE, color='#777777', ls=':', lw=0.9, zorder=1)
    ax.set_xlabel(r'C3 threshold $\tau_{\rho}$')
    ax.set_ylabel('mean Support Gain on C3')
    ax.set_xticks(TAU_R_GRID)
    ax.grid(axis='y', color='#DDDDDD', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # was: loc='upper right', frameon=False -- sat on the QVH-ME thin lines.
    ax.legend(loc='lower left', fontsize=7, frameon=True, framealpha=0.9,
              facecolor='white', edgecolor='none', borderpad=0.3,
              handlelength=1.6, labelspacing=0.3).set_zorder(5)
    # was: anchored to the top of the axes, overlapping a thin line near .50.
    ax.annotate(r'paper $\tau_{\rho}{=}0.4$', xy=(TAU_R_BASE, ax.get_ylim()[0]),
                xytext=(3, 2), textcoords='offset points', ha='left',
                va='bottom', fontsize=6, color='#555555', zorder=5)

    fig.tight_layout(pad=0.3)
    # saved exactly as the original was (no bbox_inches) so the page size is
    # byte-for-byte the same 3.35 x 2.35 in; the legend is inside the axes.
    fig.savefig(f'{OUTDIR}/fig_rho_sweep.pdf')
    fig.savefig(f'{OUTDIR}/fig_rho_sweep.png', dpi=220)

    for (bench, mk), (ln, ys) in drawn.items():
        got = ln.get_ydata()
        check.append(('fig_rho_sweep', f'{bench}/{mk}',
                      float(np.max(np.abs(np.asarray(got) - ys))), len(ys)))
    plt.close(fig)


# ----------------------------------------------------------------- Fig S2
def fig_draw_subsampling(check):
    """A3 render_fig, per-panel legends -> one shared legend under both."""
    sub = pd.read_csv(f'{SRC_A3}/draw_subsampling.csv')
    s = sub[(sub['set'] == 'C3') & (sub.quantity == 'SG')]

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), dpi=200)
    drawn = {}
    for ax, (bench, key) in zip(axes, (('NExT-GQA', 'NExT'),
                                       ('QVH-ME', 'QVH'))):
        for mk in MODELS[key]:
            m = DISP[mk]
            r = s[(s.bench == bench) & (s.model == m)].sort_values('k')
            c = MODEL_COLOR[mk]
            ax.fill_between(r.k, r['min'], r['max'], color=c, alpha=0.18, lw=0)
            ln, = ax.plot(r.k, r['mean'], color=c, lw=1.3, marker='o', ms=2.8)
            drawn[(bench, mk)] = (ln, r['mean'].to_numpy())
        ax.set_title(bench, fontsize=8)
        ax.set_xlabel('number of draws used, $k$')
        ax.set_xticks(range(1, NDRAW + 1))
        ax.grid(axis='y', color='#DDDDDD', lw=0.5)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('mean Support Gain on C3')

    handles = [Line2D([], [], color=MODEL_COLOR[mk], lw=1.3, marker='o',
                      ms=2.8) for mk in LEGEND_ORDER]
    labels = [label_of(mk) for mk in LEGEND_ORDER]

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.30)
    leg = fig.legend(handles, labels, loc='lower center', ncol=4,
                     frameon=False, fontsize=5.6, handlelength=1.2,
                     columnspacing=1.6, bbox_to_anchor=(0.5, 0.005))

    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    assert bb.y0 >= 0 and bb.y1 <= 1 and bb.x0 >= 0 and bb.x1 <= 1, bb
    # The legend sits inside the figure, so nothing is clipped without
    # bbox_inches='tight' -- and the page then keeps the original 6.9 x 2.5 in.
    fig.savefig(f'{OUTDIR}/fig_draw_subsampling.pdf')
    fig.savefig(f'{OUTDIR}/fig_draw_subsampling.png', dpi=220)

    for (bench, mk), (ln, ys) in drawn.items():
        got = ln.get_ydata()
        check.append(('fig_draw_subsampling', f'{bench}/{mk}',
                      float(np.max(np.abs(np.asarray(got) - ys))), len(ys)))
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    check = []
    fig_rho_sweep(check)
    fig_draw_subsampling(check)

    worst = max(c[2] for c in check)
    npts = sum(c[3] for c in check)
    print(f'drawn-vs-csv: {len(check)} lines, {npts} points, '
          f'worst |drawn - csv| = {worst:.3e}')
    json.dump([{'figure': a, 'line': b, 'max_abs_diff': c, 'n': d}
               for a, b, c, d in check],
              open(f'{OUTDIR}/data_check.json', 'w'), indent=1)
    if worst != 0.0:
        raise SystemExit('FAIL: redrawn values differ from the stored CSVs')
    print('OK')


if __name__ == '__main__':
    main()
