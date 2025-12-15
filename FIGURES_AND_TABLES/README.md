# Figures and Tables

This directory contains the **figure and table generation scripts**, together with the corresponding input data, used in the associated publication.

The directory is organized into three main subdirectories:

- `Figs/` — Figures included in the main body of the article.
- `FigsSM/` — Figures included in the Supplementary Material.
- `Tables/` — Tables presented in the article and Supplementary Material.

Each figure or table is stored in its own subdirectory.

---

## Directory structure

Each figure or table directory follows the same general layout:

```bash
figX/ or tableX/
├── figX.gpt / tableX.gpt # Gnuplot script
├── figX.pdf / tableX.pdf # Final generated figure or table
├── APS_term.gpt # Gnuplot formatting helpers
├── my_color_palete.gpt # Color palette definitions
└── DATA/ # Input data used for plotting
```

---

## How to generate figures and tables

To regenerate a figure or table:

1. **Enter the corresponding directory**, for example:
   
   ```bash
   cd Figs/fig2
   ```
2. **Run the Gnuplot script:**
   
   ```bash
   gnuplot fig2.gpt
   ```
   
   This generates intermediate LaTeX/EPS files and compiles them with `pdflatex` to produce the final PDF.

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