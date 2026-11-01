# -*- coding: utf-8 -*-
"""Figure 5: Mean Support Gain progression across conditions.

Values are READ FROM figdata (``ablation_table[cut]['mean_of_means']``) — the
same quantity Table 2 prints — so the figure cannot drift from the tables.
SG only. Single-column, WACV/CVPR style.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'font.size': 8, 'axes.linewidth': 0.7,
    'axes.edgecolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

CONDS = ['All', 'C1', 'C2', 'C3']
CUTS = ['ALL', 'C1', 'C2', 'C3']


def series(path, key):
    D = json.load(open(path))[key]
    return [round(D['ablation_table'][c]['mean_of_means'], 3) for c in CUTS]


NEXT = series(f'{ROOT}/paper/figdata_nextgqa.json', 'NExT')
QVH = series(f'{ROOT}/paper/figdata_qvhme.json', 'QVH')

C_NEXT = '#4C72B0'   # calm blue
C_QVH = '#C6803B'    # muted amber
INK = '#222222'; LAB = '#444444'

x = np.arange(len(CONDS)); w = 0.38
fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=200)

b1 = ax.bar(x - w / 2, NEXT, w, color=C_NEXT, label='NExT-GQA', zorder=3)
b2 = ax.bar(x + w / 2, QVH, w, color=C_QVH, label='QVH-ME', zorder=3)


def annotate(bars, vals):
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.004, f'{v:.3f}'.lstrip('0'),
                ha='center', va='bottom', fontsize=6.2, color=LAB, zorder=4)


annotate(b1, NEXT); annotate(b2, QVH)

ax.set_ylabel('Mean Support Gain', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(CONDS, fontsize=8.5)
ax.set_ylim(0, 0.245)
ax.set_yticks(np.arange(0, 0.241, 0.05))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax.grid(axis='y', color='#000000', alpha=0.07, lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(length=2.5, labelsize=8)
ax.margins(x=0.04)
ax.legend(frameon=False, fontsize=7.5, loc='upper left', handlelength=1.1,
          handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3)

fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.13)
fig.savefig(f'{FIGDIR}/fig5_sg_progression.pdf')
fig.savefig(f'{FIGDIR}/fig5_sg_progression.png', dpi=300)
print('WROTE paper/figures/fig5_sg_progression.pdf / .png')
print('NExT', NEXT, ' QVH', QVH)

PAPER_HARDCODED = {'NExT': [0.023, 0.026, 0.032, 0.124],
                   'QVH': [0.125, 0.144, 0.161, 0.212]}
for name, got in (('NExT', NEXT), ('QVH', QVH)):
    want = PAPER_HARDCODED[name]
    print(f'{name}: from figdata {got}  vs the published figure {want}  '
          f'{"MATCH" if got == want else "<<< DIFFERS"}')
