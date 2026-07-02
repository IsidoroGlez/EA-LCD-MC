#!/usr/bin/env python3
"""
Regenerate table.tex from the raw simulation data.

For each system (L, M, boundary conditions along Z) this pulls the
xi12/xi and delta_xi12/delta_xi files out of the RAW data tarballs in
../../../DATA (extracting only the needed files into a local ./DATA/
copy, as fig2.gpt does), and reads off:

  - xi(T=0.7)        : xi[12]_Q_Q_n0_*.txt,       row with column1==0
  - delta_xi(T=0.7)  : delta_xi[12]_Q_Q_n0_*.txt, row with column1==0
  - xi(T~Tc)         : xi[12]_Q_Q_n0_*.txt,       row with column2==0.9075

at the same fixed rmin (column 3) used for that system in fig2.gpt.
"""
import math
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DATA_ROOT = HERE / "../../../DATA"
LOCAL_DATA_ROOT = HERE / "DATA"
BETA_C = 0.9075

# subdir, nblocks, prefix ("xi12" for PBC systems, "xi" for OBC systems),
# and rmin (column 3) at which xi/delta_xi is read off -- same values as
# d4,d6,d8,d12,d16,d24 in fig2.gpt (d16/d24 there are the OBC systems;
# the PBC 16x512 system is not part of fig2.gpt, so d=16 continues the
# PBC pattern d6=6, d8=8, d12=12 -> d=L, confirmed to reproduce the
# previous table's PBC 16x512 values).
SYSTEMS = [
    dict(L=4,  M=192, bc="PBC", subdir="PBC/4x192",              nblocks=32768, prefix="xi12", d=5),
    dict(L=6,  M=192, bc="PBC", subdir="PBC/6x192",               nblocks=32768, prefix="xi12", d=6),
    dict(L=8,  M=192, bc="PBC", subdir="PBC/8x192",               nblocks=32768, prefix="xi12", d=8),
    dict(L=12, M=320, bc="PBC", subdir="PBC/12x320",              nblocks=32768, prefix="xi12", d=12),
    dict(L=16, M=512, bc="PBC", subdir="PBC/16x512",              nblocks=1024,  prefix="xi12", d=16),
    dict(L=16, M=48,  bc="OBC", subdir="OBC/INDEPENDENT/16x48",   nblocks=16384, prefix="xi",   d=8),
    dict(L=24, M=88,  bc="OBC", subdir="OBC/INDEPENDENT/24x88",   nblocks=2048,  prefix="xi",   d=8),
]


def ensure_extracted(subdir, nblocks, prefix):
    """Make sure the two needed data files exist under LOCAL_DATA_ROOT,
    extracting them from the raw results_NBLOCKS*.tar.gz if missing."""
    dest_dir = LOCAL_DATA_ROOT / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)

    xi_name = f"{prefix}_Q_Q_n0_NBLOCKS_{nblocks}.txt"
    delta_name = f"delta_{prefix}_Q_Q_n0_NBLOCKS_{nblocks}.txt"

    xi_path = dest_dir / xi_name
    delta_path = dest_dir / delta_name

    if xi_path.exists() and delta_path.exists():
        return xi_path, delta_path

    tarball = RAW_DATA_ROOT / subdir / f"results_NBLOCKS{nblocks}.tar.gz"
    if not tarball.exists():
        raise FileNotFoundError(f"raw data tarball not found: {tarball}")

    with tarfile.open(tarball, "r:gz") as tar:
        for member_name in (xi_name, delta_name):
            member = tar.getmember(member_name)
            with tar.extractfile(member) as src, open(dest_dir / member_name, "wb") as dst:
                dst.write(src.read())

    return xi_path, delta_path


def read_row(filepath, matches):
    with open(filepath) as f:
        for line in f:
            cols = line.split()
            if cols and matches(cols):
                return float(cols[3]), float(cols[4])
    raise ValueError(f"no matching row found in {filepath}")


def format_val_err(value, error):
    """Format as LaTeX 'value(error)', rounding the error to 1 significant
    figure, or 2 if the leading digit would be 1 or 2 (matches the
    convention used throughout the existing table)."""
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


def main():
    rows = []
    for sysinfo in SYSTEMS:
        xi_path, delta_path = ensure_extracted(sysinfo["subdir"], sysinfo["nblocks"], sysinfo["prefix"])
        d = sysinfo["d"]

        def at_T07(cols, d=d):
            return cols[0] == "0" and round(float(cols[2])) == d

        def at_Tc(cols, d=d):
            return abs(float(cols[1]) - BETA_C) < 1e-6 and round(float(cols[2])) == d

        xi_T07, err_xi_T07 = read_row(xi_path, at_T07)
        dxi_T07, err_dxi_T07 = read_row(delta_path, at_T07)
        xi_Tc, err_xi_Tc = read_row(xi_path, at_Tc)

        rows.append(dict(
            L=sysinfo["L"], M=sysinfo["M"], bc=sysinfo["bc"],
            xi_T07=format_val_err(xi_T07, err_xi_T07),
            dxi_T07=format_val_err(dxi_T07, err_dxi_T07),
            xi_Tc=format_val_err(xi_Tc, err_xi_Tc),
        ))

    body_lines = "\n".join(
        f"         ${r['L']}$ & ${r['M']}$ & {r['bc']} & {r['xi_T07']} & {r['dxi_T07']} & {r['xi_Tc']}\\\\"
        for r in rows
    )

    table_tex = f"""\\begin{{table}}[b]
    \\centering
    \\begin{{ruledtabular}}
    \\begin{{tabular}}{{cccccc}}
        $L$ & $M$ & BC-Z & $\\xi_{{n=1}}(T=0.7)$ & $\\Delta\\xi_{{n=1}}(T=0.7)$ & $\\xi_{{n=1}}(T\\simeq T_\\mathrm{{c}})$ \\\\\\hline
{body_lines}
   \\end{{tabular}}
    \\end{{ruledtabular}}
    \\caption{{$\\xi_{{n=1}}$ and $\\Delta\\xi_{{n=1}}$ for the different system sizes, $L^2\\times M$, boundary conditions along the Z-axis (BC-Z), and temperatures. Data is represented in Fig.~\\ref{{fig:xi_vs_L}}.}}
    \\label{{tab:xi}}
\\end{{table}}
"""

    (HERE / "table.tex").write_text(table_tex)
    print(table_tex)


if __name__ == "__main__":
    main()
