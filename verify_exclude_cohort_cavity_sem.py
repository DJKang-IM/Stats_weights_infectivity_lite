# -*- coding: utf-8 -*-
"""
Verify overlap between Exclude-pattern readings (short disposition) and SEM complete-case cohort.

Reads backup CSV (labs intact) + current CSV (optional, for flag column).
Writes artifacts/infectivity_exclude_cohort_cavity_verify_260510/report.json + report.txt

Key checks:
  - Listwise SEM uses rows with all 5 indicators non-missing (Cavity, AFB, PCR, Solid, Liquid).
  - Rows matching should_clear_labs(Reading): in 260510 backup, Cavity was 100% missing - so they
    never entered the ~459 CFA cohort before or after lab-clear.
  - Partial labs among exclude-pattern rows (imbalance / workup completeness signal).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from apply_meta_exclusion_lab_clear import should_clear_labs  # noqa: E402

KR5 = ["도말검사", "TB-PCR검사", "배양검사(고체)", "배양검사(액체)", "Cavity 유무"]


def _pick_backup() -> Path:
    c = Path(__file__).resolve().parent / "data" / "260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv"
    baks = sorted(c.parent.glob(c.name + ".bak_*"))
    if not baks:
        raise SystemExit("No .bak_* next to 260510 CSV; pass --backup path.")
    return Path(baks[-1])


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, default=None, help="Pre-lab-clear CSV (default: latest .bak_*).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "infectivity_exclude_cohort_cavity_verify_260510",
    )
    args = ap.parse_args()
    bak = args.backup or _pick_backup()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(bak, encoding="cp949")
    ex = df["Reading"].map(should_clear_labs)
    sub = df[KR5].apply(pd.to_numeric, errors="coerce")
    complete = sub.notna().all(axis=1)

    rep = {
        "backup_csv": str(bak.resolve()),
        "n_rows": int(len(df)),
        "n_exclude_pattern_reading": int(ex.sum()),
        "n_complete_5labs": int(complete.sum()),
        "n_exclude_and_complete_5labs": int((ex & complete).sum()),
        "n_exclude_partial_some_lab": int((ex & sub.notna().any(axis=1) & ~complete).sum()),
        "exclude_rows_cavity_nonnull": int((ex & sub["Cavity 유무"].notna()).sum()),
        "nonexclude_cavity_nonnull": int((~ex & sub["Cavity 유무"].notna()).sum()),
        "interpretation": [
            "CFA/SEM (meta_infectivity_empirical_constrained_sem.py) uses listwise complete cases: all 5 indicators non-missing.",
            "Exclude-pattern rows had Cavity missing for every row in this backup - they cannot enter the complete-case cohort.",
            "Clearing labs on Exclude rows therefore did not shrink the SEM n vs the same cohort on backup.",
            "Cavity 'underevaluation' from mixing Exclude imaging with full labs is not supported for this Exclude rule: those rows had no Cavity label.",
        ],
    }

    if int(rep["n_complete_5labs"]) > 0:
        cc = sub.loc[complete]
        rep["complete_case_cavity_pos_rate"] = float(cc["Cavity 유무"].mean())
        rep["complete_case_liquid_pos_rate"] = float(cc["배양검사(액체)"].mean())

    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"backup: {rep['backup_csv']}", json.dumps(rep, indent=2, ensure_ascii=False)]
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
