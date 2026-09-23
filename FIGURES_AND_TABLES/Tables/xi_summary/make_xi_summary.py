#!/usr/bin/env python3
"""
Build a plain-text, long-format summary of the correlation length xi(L,T)
and its difference with the critical point, delta_xi(L,T), ready for
plotting (gnuplot / numpy / pandas), one row per (system size, temperature).

For each system this pulls the xi12/xi and delta_xi12/delta_xi files out of
the raw simulation data tarballs in ../../../DATA (extracting only the
needed files into a local ./DATA/ copy, same approach as
Tables/table1/update_table.py and Figs/fig2/fig2.gpt), and reads off xi and
delta_xi for every temperature available in those files (both directly
simulated points and temperature-interpolated ones), all at the same fixed
rmin (column 3) used for that system size in fig2.gpt/fig3.gpt.

delta_xi is the observable already tabulated in the delta_xi[12]_*.txt
files, on the same (index, beta) grid as xi; at T=Tc it is identically zero
by construction (delta_xi measures the difference with the critical point,
which vanishes at the critical point itself).
"""
import math
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DATA_ROOT = HERE / "../../../DATA"
LOCAL_DATA_ROOT = HERE / "DATA"
BETA_C = 0.9075  # T_c = 1/BETA_C =~ 1.1019, for reference when reading the output

# subdir, nblocks, prefix ("xi12" for PBC systems, "xi" for OBC systems),
# and rmin (column 3) at which xi/delta_xi is read off -- same six systems
# and same d4,d6,d8,d12,d16,d24 values as in fig2.gpt/fig3.gpt.
SYSTEMS = [
    dict(L=4,  M=192, bc="PBC", subdir="PBC/4x192",            nblocks=32768, prefix="xi12", d=5),
    dict(L=6,  M=192, bc="PBC", subdir="PBC/6x192",             nblocks=32768, prefix="xi12", d=6),
    dict(L=8,  M=192, bc="PBC", subdir="PBC/8x192",             nblocks=32768, prefix="xi12", d=8),
    dict(L=12, M=320, bc="PBC", subdir="PBC/12x320",            nblocks=32768, prefix="xi12", d=12),
    dict(L=16, M=48,  bc="OBC", subdir="OBC/INDEPENDENT/16x48", nblocks=16384, prefix="xi",   d=8),
    dict(L=24, M=88,  bc="OBC", subdir="OBC/INDEPENDENT/24x88", nblocks=2048,  prefix="xi",   d=8),
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


def read_all_rows(filepath, d):
    """Return every row at rmin column == d, keyed by beta (column 2), as
    {beta: (index_label, value, error)}. index_label is the raw index
    column: an integer string for a directly simulated temperature, or
    "I" for a temperature-interpolated one."""
    rows = {}
    with open(filepath) as f:
        for line in f:
            cols = line.split()
            if not cols or round(float(cols[2])) != d:
                continue
            beta = float(cols[1])
            rows[beta] = (cols[0], float(cols[3]), float(cols[4]))
    return rows


def main():
    rows = []
    for sysinfo in SYSTEMS:
        xi_path, delta_path = ensure_extracted(sysinfo["subdir"], sysinfo["nblocks"], sysinfo["prefix"])
        d = sysinfo["d"]

        xi_rows = read_all_rows(xi_path, d)
        delta_rows = read_all_rows(delta_path, d)

        # descending beta == ascending T
        for beta in sorted(xi_rows, reverse=True):
            idx, xi, err_xi = xi_rows[beta]
            if beta in delta_rows:
                _, dxi, err_dxi = delta_rows[beta]
            else:
                dxi, err_dxi = math.nan, math.nan

            rows.append(dict(
                L=sysinfo["L"], bc=sysinfo["bc"], d=d, idx=idx, T=1.0 / beta,
                xi=xi, err_xi=err_xi, delta_xi=dxi, err_delta_xi=err_dxi,
            ))

    header = (
        f"{'#L':>4} {'BC':>4} {'d':>4} {'idx':>4} {'T':>14} "
        f"{'xi':>14} {'err_xi':>14} {'delta_xi':>14} {'err_delta_xi':>14}"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['L']:>4} {r['bc']:>4} {r['d']:>4} {r['idx']:>4} {r['T']:14.7f} "
            f"{r['xi']:14.7f} {r['err_xi']:14.7f} "
            f"{r['delta_xi']:14.7f} {r['err_delta_xi']:14.7f}"
        )

    out_text = "\n".join(lines) + "\n"
    (HERE / "xi_summary.dat").write_text(out_text)
    print(out_text)


if __name__ == "__main__":
    main()
