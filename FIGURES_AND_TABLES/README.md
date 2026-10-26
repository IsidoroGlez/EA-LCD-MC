# Figures and Tables

This directory contains the **figure and table generation scripts**, together with the corresponding input data, used in the associated publication.

The directory is organized into three main subdirectories:

- `Figs/` — Figures included in the main body of the article.
- `FigsSM/` — Figures included in the Supplementary Material.
- `Tables/` — Tables presented in the article and Supplementary Material.

Each figure or table is stored in its own subdirectory.

The shared Gnuplot style used by every figure script is kept once, in `FIGURES_AND_TABLES/style/` (not duplicated per figure):

- `style/ealcd_pnas.gpt` — the file each figure loads. It sets the PNAS profile (column widths, see below), Helvetica text and the line/tick settings the figures were tuned with, and loads the files below.
- `style/core/` and `style/journals/pnas.gpt` — copy of the shared figure-style template (commit `c18e43e` of the `GNUPLOT` style repository): `epslatex` terminal, margins, tics and the color-blind-friendly palette (`core/palette.gpt`).

Each figure script loads it via a relative path: `load "../../style/ealcd_pnas.gpt"`. The text is typeset with Helvetica (`fontspec`), so figures are compiled with `xelatex`; if Helvetica is not installed, set `SANS_FONT = "Nimbus Sans"` before the `load`.

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
- `FigsSM/fig2/` is a static image (`figSM2.pdf`/`.svg`) with no generation script at all — it's drawn by hand in Inkscape.
- `Tables/table1/` is regenerated with the Python script `update_table.py` rather than a Gnuplot script; it writes `table.tex` directly from the raw data in `../../../DATA` (extracting the files it needs on demand, like `fig2.gpt` does).
- `Tables/table2/` is regenerated with the Python script `update_table.py`, like `table1`: it writes `table.tex` from the raw data in `../../../DATA`.
- `Tables/xi_summary/` is regenerated with `make_xi_summary.py`: it writes `xi_summary.dat`, a long-format summary of $\xi(L,T)$ and $\Delta\xi(L,T)$ (one row per system size and temperature), from the raw data in `../../../DATA`.

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
   
   This generates intermediate LaTeX/EPS files and compiles them with `xelatex` to produce the final PDF (and `dvisvgm --pdf` for the SVG).

To regenerate the Python-based tables (`Tables/table1`, `Tables/table2`) and the summary `Tables/xi_summary`:

```bash
cd Tables/table1 && python3 update_table.py        # rewrites table.tex
cd ../table2     && python3 update_table.py        # rewrites table.tex
cd ../xi_summary && python3 make_xi_summary.py     # rewrites xi_summary.dat
```

Each script pulls the required raw data out of `../../../DATA` (extracting only the files it needs into a local `DATA/` copy). The generated `table.tex` and `xi_summary.dat` are kept under version control.

---

## Figure width (PNAS column sizes)

Each figure script sets a `C` variable before `load "../../style/ealcd_pnas.gpt"` to pick the figure width in PNAS units:

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
- `lmargin=0.16` for a `ylabel` with decimal/3-digit tick numbers, or containing parentheses/superscripts (e.g. `0.44`, `-14`, `$E\,(r\,)$`, `$C^{\,(1)}(r\,)$`) — parentheses extend further than bare letters when rotated, so they need the wider margin even with short tick numbers — `ytics`/`ylabel` offset `0.75`/`4.5`.
- `fig3` (two side-by-side panels at `C=2`) keeps its own margins — its layout is structurally different from the single-panel figures above.

A secondary (`y2`) axis, where present, mirrors the same idea on its own side (own tics/label offset chosen for a small, consistent gap, own outer margin).

### `FigsSM/` sizing

The `FigsSM/` figures with a script (`figSM1`, `figSM3`–`figSM10`) are single-panel figures generated at `C=1` (8.7 cm) with the same `R=0.6897` as the main-text single-panel figures, and they use the same `style/ealcd_pnas.gpt`. `figSM2` has no script (it is drawn by hand in Inkscape).

### Typography

Text is set in Helvetica (via `fontspec`; maths in `mathastext`), so the `.tex` files must be compiled with `xelatex`. The base text size is 12 pt (`\normalsize`, set by the terminal font size in `style/ealcd_pnas.gpt`); axis labels and annotations use `\footnotesize` (10 pt) and `\scriptsize` (8 pt), as measured in the compiled figures. The figures are compiled at their final width, so these are the sizes printed in the figure. With Helvetica, superscripts and parentheses in labels would touch the adjacent symbols, so labels carry thin spaces (`\,`), e.g. `C^{\,(1)}(r\,)` or `K\,(r\,)`.

---

## Requirements

- Gnuplot with LaTeX terminal support.

- A TeX distribution (e.g., TeX Live, MiKTeX) with `xelatex` and `dvisvgm` available in the system path, and the Helvetica font (or a metric-compatible clone such as Nimbus Sans).

---

## Cleaning up

At the end of each Gnuplot script, a cleanup command (`rm -f`) removes the temporary `.aux`, `.tex`, `.log` and `.eps` files. This works on Linux/macOS but not directly on Windows. For running on Windows systems, substitute the Linux command `rm -f` with `del` in files figX.gpt.

## Notes

All figure files and the generated tables (`table.tex`, `xi_summary.dat`) are version-controlled, but the `DATA/` directories are excluded from Git history via .gitignore.

Output .pdf files are kept for convenience but can be regenerated as explained above.