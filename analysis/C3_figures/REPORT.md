# C3 — legend repositioning for Fig S1 and Fig S2

Produced by [analysis/camera_ready/c3_figures.py](analysis/camera_ready/c3_figures.py).
Nothing was recomputed and no model was re-run. The source scripts
(`a2_threshold_sensitivity.py`, `a3_draw_variance.py`) and the original PDFs
under `A2_threshold_sensitivity/` and `A3_draw_variance/` are unmodified —
the new files are written here.

The numbers are read back out of the CSVs those two scripts wrote in the same
run that produced the original PDFs (`A2/sweep.csv`, `A3/draw_subsampling.csv`),
so the plotted values cannot drift from the published ones.

## Output

```
outputs/camera_ready/C3_figures/
├── fig_rho_sweep.pdf        (+ .png)   supplementary Fig S1
├── fig_draw_subsampling.pdf (+ .png)   supplementary Fig S2
└── data_check.json          26 lines x 5 points, drawn-vs-CSV diff per line
```

## What changed

### Fig S2 `fig_draw_subsampling.pdf`

* The two in-panel legends are gone. One shared legend sits under both panels:
  `fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
  bbox_to_anchor=(0.5, 0.005))`, with `subplots_adjust(bottom=0.30)` reserving
  the strip. Seven entries, the union of the two panels, in Table 5 order:
  Qwen2.5-VL, VideoMind, Grounded-VideoLLM, GroundVTS, VideoChat-R1,
  TimeChat†, VTimeLLM†. The daggers are as in the current figure; no ‡ was
  added, since the current figure carries none.
* **Colours had to become model-keyed.** The old script indexed `PALETTE` by
  the model's position *within each panel*, so the same model had two
  different colours — Qwen2.5-VL was `#4C72B0` on the left and `#C6803B` on
  the right, VideoMind `#C6803B` / `#8172B2`, and so on. A single shared
  legend is only meaningful once a model has one colour, so the same seven
  `PALETTE` values are now indexed by the legend order above. Consequences:

  | | before | now |
  |---|---|---|
  | Qwen2.5-VL | blue / orange | blue `#4C72B0` |
  | VideoMind | orange / purple | orange `#C6803B` |
  | Grounded-VideoLLM | green / green | green `#55A868` |
  | GroundVTS | purple / red | purple `#8172B2` |
  | VideoChat-R1 | — / blue | red `#C44E52` |
  | TimeChat† | — / cyan | cyan `#64B5CD` |
  | VTimeLLM† | red / grey | grey `#8C8C8C` |

  The left panel changes in one place only (VTimeLLM† red → grey); the right
  panel is re-coloured throughout. A side effect worth keeping: both
  non-interpreted models now hold the two muted entries (cyan, grey).
  If you would rather the **left** panel stay pixel-identical, the
  alternative mapping is one line in `c3_figures.py` — keep
  `PALETTE[i] for i, mk in enumerate(MODELS['NExT'])` and give VideoChat-R1
  `PALETTE[5]`, TimeChat `PALETTE[6]`. That leaves VideoChat-R1, the top line
  of the right panel, on the pale cyan, which is why it is not the default.
* Legend font kept at the script's existing `5.6`. At full width `7` is
  comfortable; say so and it is a one-character change.
* **Not** saved with `bbox_inches='tight'`. The legend is placed *inside* the
  figure (verified: its bounding box is within `[0,1]²` in figure
  coordinates, asserted in the script), so nothing is clipped, and leaving
  the bbox alone is what keeps the page size identical to the published PDF —
  `tight` would have trimmed it to a different size.

### Fig S1 `fig_rho_sweep.pdf`

* Legend moved `upper right` → `lower left`, now with a white box
  (`frameon=True, framealpha=0.9, facecolor='white', edgecolor='none'`,
  `zorder=5`) as specified. The thin lines it covers in that corner are the
  near-zero per-model curves.
* The `paper τ_ρ = 0.4` annotation is moved off the top of the axes (where it
  crossed a thin line near .50) to just right of the dotted rule at the
  bottom of the axes, `fontsize=6` unchanged.
* Everything else — figsize `(3.35, 2.35)`, dpi, the `C_NEXT`/`C_QVH` colours,
  line widths, markers, ticks, grid, spines — is verbatim from `render_fig`.

## Checks

**Data unchanged.** After rendering, every vertex matplotlib actually holds is
read back off the artists and compared with the CSV column it came from:
**26 lines, 130 points, worst `|drawn − csv| = 0.000e+00`.** Per-line detail in
`data_check.json`. The script exits non-zero and this file is not trusted if
that is not exactly zero.

**`pdffonts` — no Type 3 in either file**, and the font set is identical to the
originals:

| | fonts |
|---|---|
| `fig_rho_sweep.pdf` | DejaVuSerif, Cmmi10, Cmr10 — all CID TrueType, embedded, subset |
| `fig_draw_subsampling.pdf` | DejaVuSerif, Cmmi10 — all CID TrueType, embedded, subset |

**`pdfinfo` — page size identical to the originals**, 1 page each:

| | original | new |
|---|---|---|
| `fig_rho_sweep.pdf` | 241.2 × 169.2 pts | **241.2 × 169.2 pts** |
| `fig_draw_subsampling.pdf` | 496.8 × 180 pts | **496.8 × 180 pts** |

So both files are drop-in replacements: `\includegraphics` needs no change.
