# -*- coding: utf-8 -*-
"""
260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv

판독문(Reading)에 Exclude 표기가 있으나 실질 판독이 거의 없다고 판단되는 행에서
객관 지표 열만 비움(NaN). Reading·Study No. 는 그대로.

판정 규칙 (should_clear_labs):
  - Exclude 단어가 있고,
  - 아래 중 하나면 True:
      * 구조적 표기: ### ... Exclude ... ###, 줄 끝 `... : Exclude`, `로 Exclude`
      * 전체 길이 <= 120 이고, 다음이 아님:
          - cannot be excluded / cannot exclude
          - to exclude (예: PET/CT correlation to exclude malignancy)

검증용 열: lab_cleared_exclude_disposition (0/1)

사용:
  python apply_meta_exclusion_lab_clear.py
  python apply_meta_exclusion_lab_clear.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv"

FLAG_COL = "lab_cleared_exclude_disposition"
LAB_COLS_KR = [
    "도말검사",
    "TB-PCR검사",
    "배양검사(고체)",
    "배양검사(액체)",
    "Cavity 유무",
    "D6_NTM",
]


def should_clear_labs(text: object) -> bool:
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if not re.search(r"(?i)\bExclude\b", s):
        return False
    if re.search(r"###[^#]*\bExclude\b[^#]*###", s, re.I):
        return True
    if re.search(r"로\s+Exclude", s):
        return True
    for line in s.splitlines():
        if re.search(r":\s*Exclude\s*$", line.strip(), re.I):
            return True
    if len(s) > 120:
        return False
    sl = s.lower()
    if "cannot be excluded" in sl or "cannot exclude" in sl:
        return False
    if re.search(r"(?i)\bto\s+exclude\b", s):
        return False
    return True


def read_csv_any_enc(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak copy before overwrite.")
    args = ap.parse_args()
    path: Path = args.csv
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")

    df = read_csv_any_enc(path)
    if "Reading" not in df.columns:
        raise SystemExit(f"No Reading column. Columns: {list(df.columns)}")

    if FLAG_COL in df.columns:
        df = df.drop(columns=[FLAG_COL])

    missing = [c for c in LAB_COLS_KR if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing lab columns: {missing}")

    mask = df["Reading"].map(should_clear_labs)
    n_clear = int(mask.sum())
    print(f"Rows to clear lab fields: {n_clear} / {len(df)}")

    df2 = df.copy()
    insert_at = int(df2.columns.get_loc("Reading")) + 1
    df2.insert(insert_at, FLAG_COL, np.where(mask, 1, 0).astype(np.int64))

    for c in LAB_COLS_KR:
        df2.loc[mask, c] = np.nan

    if args.dry_run:
        print("[dry-run] not writing file.")
        print(df2.loc[mask, ["Study No.", "Reading", FLAG_COL] + LAB_COLS_KR].head(8).to_string())
        return 0

    if not args.no_backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".bak_{ts}")
        shutil.copy2(path, bak)
        print(f"Backup: {bak}")

    df2.to_csv(path, index=False, encoding="cp949", na_rep="")
    print(f"Wrote: {path} (encoding=cp949, na_rep empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
