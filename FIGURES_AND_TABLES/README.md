# Figures and Tables

This directory contains the **figure and table generation scripts**, together with the corresponding input data, used in the associated publication.

The directory is organized into three main subdirectories:

- `Figs/` — Figures included in the main body of the article.
- `FigsSM/` — Figures included in the Supplementary Material.
- `Tables/` — Tables presented in the article and Supplementary Material.

Each figure or table is stored in its own subdirectory.

Two Gnuplot helpers used by every figure script are kept once, at the top of `FIGURES_AND_TABLES/` (not duplicated per figure):

- `my_color_palette.gpt` — color-blind-friendly palette definitions.
- `PNAS_term.gpt` — sets the `epslatex` terminal size for PNAS column widths (see below).

Each figure script loads them via a relative path, e.g. `load "../../my_color_palette.gpt"`.

---

## Directory structure

Most figure or table directories follow the same general layout:

```bash
figX/ or tableX/
├── figX.gpt / tableX.gpt # Gnuplot script
├── figX.pdf / tableX.pdf # Final generated figure or table
└── DATA/ # Input data used for plotting
```

A few directories are exceptions to this layout:

- `Figs/fig1/` has no `fig1.gpt`: `fig1-preview.gpt` generates a draft (`fig1-preview.pdf`/`.svg`), which is then finished by hand (e.g. in Inkscape) into the final `fig1.pdf`/`.svg`.
- `FigsSM/fig1/` is a static image (`figSM1.pdf`/`.svg`) with no generation script at all — it's produced and edited entirely by hand.
- `Tables/table1/` is regenerated with the Python script `update_table.py` rather than a Gnuplot script; it writes `table.tex` directly from the raw data in `../../../DATA` (extracting the files it needs on demand, like `fig2.gpt` does).
- `Tables/table2/` contains `get_singularity_spectrum.gpt`, a standalone analysis script that prints the singularity-spectrum values to the terminal rather than producing a `.pdf`/`.tex` table.

---

## How to generate figures and tables

To regenerate a Gnuplot-based figure or table:

1. **Enter the corresponding directory**, for example:
   
   ```bash
   cd Figs/fig2
   ```
2. **Run the Gnuplot script:**
   
   ```bash
   gnuplot fig2.gpt
   ```
   
   This generates intermediate LaTeX/EPS files and compiles them with `pdflatex` to produce the final PDF.

To regenerate `Tables/table1` (Python-based):

```bash
cd Tables/table1
python3 update_table.py
```

This pulls the required raw data out of `../../../DATA` and rewrites `table.tex`.

---

## Figure width (PNAS column sizes)

Each figure script sets a `C` variable before `load "../../PNAS_term.gpt"` to pick the figure width in PNAS units:

- `C=1` — one column (8.7 cm)
- `C=1.5` — 1.5 columns (11.4 cm)
- `C=2` — two columns (17.8 cm)

The `R` variable (also set before the `load`) is the height/width ratio, so the final size is `W x R*W` cm.

All figures share a common height of ~6 cm (`R=0.6897` for `C=1`, `R=0.3371` for `C=2`).

### Margins

Since `tmargin`/`bmargin` only depend on the shared height, they use the same values everywhere, together with the same `xtics`/`xlabel` offsets (so the gap axis→numbers→label→border is identical across figures, not just the margin fraction):

- `set xtics offset 0,0.5`
- `set xlabel '...' offset 0,1.0`
- `tmargin=0.97`, `bmargin=0.14` — comfortably fits this offset pair, including descender-heavy labels (`$\xi$`, `$r_{\mathrm{min}}$`), without clipping any figure's label or crowding the tick numbers.

`lmargin`/`rmargin` depend on each figure's width and y-axis content (label + tick-number width), so they're unified only within groups that share similar content — each group also uses a common `ytics`/`ylabel` offset:

- `rmargin=0.99` everywhere there's no secondary (y2) axis or wide last tick label in the way (e.g. `fig1-preview` uses `0.96` to fit its 3-digit rightmost `z` tick; `fig3`'s y2 side uses `0.937`).
- `lmargin=0.05` for panels with no `ylabel` (just bare tick numbers).
- `lmargin=0.11` for a short, parenthesis-free `ylabel` (e.g. `$U$`, `$\xi$`, `$x$`) with plain 1–2 digit tick numbers, `ytics`/`ylabel` offset `0.75`/`3.8`.
- `lmargin=0.16` for a `ylabel` with decimal/3-digit tick numbers, or containing parentheses/superscripts (e.g. `0.44`, `-14`, `$E(r)$`, `$C^{(1)}(r)$`) — parentheses extend further than bare letters when rotated, so they need the wider margin even with short tick numbers — `ytics`/`ylabel` offset `0.75`/`4.5`.
- `fig3` (two side-by-side panels at `C=2`) keeps its own margins — its layout is structurally different from the single-panel figures above.

A secondary (`y2`) axis, where present, mirrors the same idea on its own side (own tics/label offset chosen for a small, consistent gap, own outer margin).

### `FigsSM/` sizing (`C=2` instead of `C=1`)

The `FigsSM/fig2`–`fig9` scripts are generated at `C=2` (17.8 cm), not `C=1`, even though each is a single-panel figure that would normally fall in the `C=1` group above. This is because the SM figures get scaled to `\textwidth` (the full 2-column width) in the compiled supplement, so generating them at `C=1` and letting LaTeX scale them up by ~2.05× would inflate their `\normalsize` (12 pt) label text to ~24 pt — clearly oversized next to the surrounding body text. Generating them natively at the width they'll actually be displayed at means no LaTeX rescaling happens, so the labels render at their intended 12 pt, matching the terminal's `cmr,12` base font exactly (unlike the main-text `Figs/` figures above, which use smaller `\footnotesize`/`\scriptsize`, i.e. 10 pt/8 pt, labels).

To do this without changing how the figure looks (same aspect ratio, same relative margins, same font-to-plot proportions as the `C=1` figures), each script keeps its original `R` and rescales its four margins by the same `C=1→C=2` width ratio (17.8/8.7 ≈ 2.046), e.g. `lmargin_new = lmargin_old / 2.046` and `tmargin_new = 1-(1-tmargin_old)/2.046`. The `xtics`/`ytics`/`xlabel`/`ylabel` offsets are left untouched — they're in character units tied to the terminal's fixed font metric (`font cmr,12`), so they already represent the same absolute distance regardless of `C`.

---

## Requirements

- Gnuplot with LaTeX terminal support.

- A TeX distribution (e.g., TeX Live, MiKTeX) with `pdflatex` available in the system path.

---

## Cleaning up

At the end of each Gnuplot script, a cleanup command (rm) removes temporary files like .log, .aux, and .tex. This works on Linux/macOS but not directly on Windows. For running on Windows systems, substitute the Linux command `rm` with `del` in files figX.gpt.

## Notes

All figure files are version-controlled, but data files are excluded from Git history via .gitignore unless explicitly needed.

Output .pdf files are kept for convenience but can be regenerated as explained above.