#!/usr/bin/env python3
"""
Regenerate table.tex from the raw simulation data.

For each OBC system (L, M), each temperature and each moment n=2,3,4 this
pulls the xi_2R_over_xi_2nR files out of the RAW data tarballs in
../../../DATA (extracting only the needed files into a local ./DATA/
copy, as table1/update_table.py and fig4.gpt do), and reads off

  tau_n = xi_{n=1}/xi_n   from  xi[12]_2R_over_xi[12]_2nR_Q_Q_n<n-1>_*.txt

at the same fixed rmin (column 3) used for these systems in table1 and
fig2.gpt.  The last row is the one-dimensional droplet-model prediction,
sum_{k=1}^{n} 1/(2k-1)  (= 4/3, 23/15, 176/105 for n=2,3,4), which the
previous get_singularity_spectrum.gpt printed as hard-coded literals.

Rows are selected by the beta value in column 2, not by the beta index in
column 1: the two systems do not share a beta grid.  T=0.8, 0.9 and 1.0
are simulated points for 16x48 (indices 3, 6, 9 -- the selectors used in
fig4_allT.gpt, which plots 16x48 only) but for 24x88 they exist only as
temperature-interpolated rows (column1=="I"), as does T~Tc for both
systems.  Matching on beta covers both cases uniformly, and each match is
checked to be unique.

This replaces get_singularity_spectrum.gpt, whose Tc selector read
    $1=="I" && $2="0.9075"
(an awk *assignment*, always true), so it silently returned the extremum
over all six interpolated temperatures instead of the value at beta_c.
"""
import math
import tarfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DATA_ROOT = HERE / "../../../DATA"
LOCAL_DATA_ROOT = HERE / "DATA"
BETA_C = 0.9075
# Tolerance for matching column 2.  The closest pair of beta values in any
# of these files is 0.9075 vs 0.90909 (1.6e-3 apart), so this is safe.
BETA_TOL = 1e-6

# Moments shown in the table.  Moment n lives in the file with suffix
# n<n-1> (as in fig4.gpt, where n0 <-> n=1).
MOMENTS = [2, 3, 4]

# Temperatures, in the order they appear in the table.  beta=None means
# "use BETA_C" (the critical point is not at a round temperature).
TEMPERATURES = [
    dict(label=r"$0.7$", beta=1.0 / 0.7),
    dict(label=r"$0.8$", beta=1.0 / 0.8),
    dict(label=r"$0.9$", beta=1.0 / 0.9),
    dict(label=r"$1.0$", beta=1.0),
    dict(label=r"$T_\mathrm{c}$", beta=BETA_C),
]

# subdir, nblocks, prefix ("xi12" for PBC systems, "xi" for OBC systems),
# and rmin (column 3) at which the ratio is read off -- rmin=8 for both OBC
# systems, the same value used in table1/update_table.py and fig2.gpt.
SYSTEMS = [
    dict(L=16, M=48, bc="OBC", subdir="OBC/INDEPENDENT/16x48", nblocks=16384, prefix="xi",   d=8),
    dict(L=24, M=88, bc="OBC", subdir="OBC/INDEPENDENT/24x88", nblocks=2048,  prefix="xi",   d=8),
]


def data_file_name(prefix, nblocks, n):
    """Name of the ratio file holding the n-th moment (suffix n<n-1>)."""
    return f"{prefix}_2R_over_{prefix}_2nR_Q_Q_n{n - 1}_NBLOCKS_{nblocks}.txt"


def ensure_extracted(subdir, nblocks, prefix):
    """Make sure the needed data files exist under LOCAL_DATA_ROOT,
    extracting them from the raw results_NBLOCKS*.tar.gz if missing."""
    dest_dir = LOCAL_DATA_ROOT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    names = {n: data_file_name(prefix, nblocks, n) for n in MOMENTS}
    paths = {n: dest_dir / name for n, name in names.items()}

    missing = [names[n] for n in MOMENTS if not paths[n].exists()]
    if not missing:
        return paths

    tarball = RAW_DATA_ROOT / subdir / f"results_NBLOCKS{nblocks}.tar.gz"
    if not tarball.exists():
        raise FileNotFoundError(f"raw data tarball not found: {tarball}")

    with tarfile.open(tarball, "r:gz") as tar:
        for member_name in missing:
            member = tar.getmember(member_name)
            with tar.extractfile(member) as src, open(dest_dir / member_name, "wb") as dst:
                dst.write(src.read())

    return paths


def read_row(filepath, beta, d):
    """Value and error of the unique row at this beta (column 2) and this
    rmin (column 3).  Raises if the row is missing or not unique."""
    matches = []
    with open(filepath) as f:
        for line in f:
            cols = line.split()
            if len(cols) < 5:
                continue
            if abs(float(cols[1]) - beta) < BETA_TOL and round(float(cols[2])) == d:
                matches.append(cols)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly 1 row with beta={beta:.6f}, rmin={d} in "
            f"{filepath}, found {len(matches)}"
        )
    return float(matches[0][3]), float(matches[0][4])


def format_val_err(value, error):
    """Format as LaTeX 'value(error)', rounding the error to 1 significant
    figure, or 2 if the leading digit would be 1 or 2 (matches the
    convention used throughout the existing tables)."""
    exponent = math.floor(math.log10(error))
    mantissa = error / 10 ** exponent
    lead = round(mantissa)
    if lead == 10:
        lead = 1
        exponent += 1
    sig_figs = 2 if lead in (1, 2) else 1
    decimals = max(sig_figs - 1 - exponent, 0)
    val_rounded = round(value, decimals)
    err_rounded = round(error, decimals)
    err_digits = int(round(err_rounded * 10 ** decimals))
    return f"{val_rounded:.{decimals}f}({err_digits})"


def prediction(n):
    """One-dimensional droplet-model prediction for tau_n,
    sum_{k=1}^{n} 1/(2k-1): 4/3, 23/15, 176/105 for n=2,3,4."""
    return sum(Fraction(1, 2 * k - 1) for k in range(1, n + 1))


def main():
    files = {}
    for sysinfo in SYSTEMS:
        files[sysinfo["L"]] = ensure_extracted(
            sysinfo["subdir"], sysinfo["nblocks"], sysinfo["prefix"]
        )

    body_lines = []
    for temp in TEMPERATURES:
        for sysinfo in SYSTEMS:
            L, d = sysinfo["L"], sysinfo["d"]
            taus = [
                format_val_err(*read_row(files[L][n], temp["beta"], d))
                for n in MOMENTS
            ]
            body_lines.append(
                f"         {temp['label']} & ${L}$ & " + " & ".join(taus) + "\\\\"
            )
    body_lines[-1] += "\\hline"

    preds = []
    for n in MOMENTS:
        pred = prediction(n)
        preds.append(f"${pred.numerator}/{pred.denominator} = {float(pred):.4f}$")
    body_lines.append(
        "         \\multicolumn{2}{c}{1D droplet} & " + " & ".join(preds) + "\\\\"
    )

    body = "\n".join(body_lines)
    tau_headers = " & ".join(f"$\\tau_{{{n}}}$" for n in MOMENTS)

    table_tex = f"""\\begin{{table}}[b]
    \\centering
    \\begin{{ruledtabular}}
    \\begin{{tabular}}{{{"c" * (2 + len(MOMENTS))}}}
        $T$ & $L$ & {tau_headers} \\\\\\hline
{body}
   \\end{{tabular}}
    \\end{{ruledtabular}}
    \\caption{{$\\tau_n=\\xi_{{n=1}}/\\xi_n$ for the two OBC systems, $L^2\\times M$ with $M=48$ ($L=16$) and $M=88$ ($L=24$), at several temperatures and at the critical point, read off at $r_\\mathrm{{min}}=8$ as in Table~\\ref{{tab:xi}}. The last row is the one-dimensional droplet-model prediction, $\\sum_{{k=1}}^{{n}}1/(2k-1)$.}}
    \\label{{tab:tau}}
\\end{{table}}
"""

    (HERE / "table.tex").write_text(table_tex)
    print(table_tex)


if __name__ == "__main__":
    main()
