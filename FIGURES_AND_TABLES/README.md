# Figures and Tables

This directory contains the **figure and table generation scripts**, together with the corresponding input data, used in the associated publication.

The directory is organized into three main subdirectories:

- `Figs/` — Figures included in the main body of the article.
- `FigsSM/` — Figures included in the Supplementary Material.
- `Tables/` — Tables presented in the article and Supplementary Material.

Each figure or table is stored in its own subdirectory.

---

## Directory structure

Most figure or table directories follow the same general layout:

```bash
figX/ or tableX/
├── figX.gpt / tableX.gpt # Gnuplot script
├── figX.pdf / tableX.pdf # Final generated figure or table
├── APS_term.gpt # Gnuplot formatting helpers
├── my_color_palette.gpt # Color palette definitions
└── DATA/ # Input data used for plotting
```

A few directories are exceptions to this layout:

- `Figs/fig1/` is generated from `fig1-preview.gpt` (producing `fig1-preview.pdf`/`.svg`) rather than `fig1.gpt`/`fig1.pdf`.
- `FigsSM/fig1/` is a static image (`figSM1.pdf`) with no generation script.
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

## Requirements

- Gnuplot with LaTeX terminal support.

- A TeX distribution (e.g., TeX Live, MiKTeX) with `pdflatex` available in the system path.

---

## Cleaning up

At the end of each Gnuplot script, a cleanup command (rm) removes temporary files like .log, .aux, and .tex. This works on Linux/macOS but not directly on Windows. For running on Windows systems, substitute the Linux command `rm` with `del` in files figX.gpt.

## Notes

All figure files are version-controlled, but data files are excluded from Git history via .gitignore unless explicitly needed.

Output .pdf files are kept for convenience but can be regenerated as explained above.