# -*- coding: utf-8 -*-
"""
1) Relative-influence / Est.Std table from SEM summary JSONs (linear + log1p constrained).
2) Export complete-case 5-item matrix (CSV).
3) IRT without R: girth.twopl_mml = same binary 2PL family as mirt(data, model=1, itemtype="2PL")
   (unidimensional marginal MML). No R / mirt required.

Optional: --write-r-script if you ever want a reference R snippet (off by default).

Default inputs: 260510 linear + log1p artifact folders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import girth  # noqa: E402
from analyze_infectivity_latent import COLS  # noqa: E402
from meta_infectivity_empirical_constrained_sem import (  # noqa: E402
    _read_meta_csv,
    load_indicator_frame,
)


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, d: dict[str, float] | None) -> str:
    if not d:
        return f"{label}\t" + "\t".join(["nan"] * len(COLS))
    cells = [f"{float(d.get(k, float('nan'))):.6f}" for k in COLS]
    return f"{label}\t" + "\t".join(cells)


def build_table(
    *,
    label_linear: str,
    summary_linear: Path,
    label_log1p: str,
    summary_log1p: Path | None,
) -> str:
    sl = _load_summary(summary_linear)
    lines = [
        "Relative influence & related metrics (|Est.Std| / sum for CFA rows)",
        "Columns order: " + " | ".join(COLS),
        "",
        "Metric\t" + "\t".join(COLS),
        _row(f"PCA PC1 sq-loading share ({label_linear})", sl.get("pca", {}).get("weights_normalized_sq")),
        _row(f"Free CFA relative influence 0-1 ({label_linear})", sl.get("cfa_free_relative_influence_0_1")),
        _row(f"Constrained CFA relative influence 0-1 ({label_linear})", sl.get("constrained_cfa_relative_influence_0_1")),
    ]
    if summary_log1p is not None and summary_log1p.is_file():
        sg = _load_summary(summary_log1p)
        lines.append(_row(f"Constrained CFA relative influence 0-1 ({label_log1p})", sg.get("constrained_cfa_relative_influence_0_1")))
        lines.append(_row(f"Constrained CFA Est.Std loadings ({label_log1p})", sg.get("constrained_cfa_standardized_loadings")))
    lines.append(_row(f"Free CFA Est.Std loadings ({label_linear})", sl.get("cfa_free_standardized_loadings")))
    lines.append(_row(f"Constrained CFA Est.Std loadings ({label_linear})", sl.get("constrained_cfa_standardized_loadings")))
    lines.append("")
    lines.append("Notes:")
    lines.append("- CFA 'relative influence' = |standardized loading| normalized to sum to 1 across the 5 indicators.")
    lines.append("- PCA row uses squared PC1 correlation loadings normalized to sum 1 (different construct than CFA).")
    return "\n".join(lines) + "\n"


def run_girth_2pl(X: np.ndarray) -> dict:
    """X: (n_persons, n_items) 0/1 int/float. Returns dict with a,b per COLS order."""
    X = np.asarray(X, dtype=np.int32)
    if X.shape[1] != len(COLS):
        raise ValueError(f"Expected {len(COLS)} columns")
    out = girth.twopl_mml(X.T)
    a = np.asarray(out["Discrimination"], dtype=float).ravel()
    b = np.asarray(out["Difficulty"], dtype=float).ravel()
    items = {COLS[i]: {"a_2pl_discrimination": float(a[i]), "b_difficulty": float(b[i])} for i in range(len(COLS))}
    return {
        "engine": "girth.twopl_mml (Python, MML)",
        "mirt_equivalent_call": 'mirt(data, model = 1, itemtype = "2PL")  # binary 1-factor 2PL; no R in this pipeline',
        "note": "Unidimensional 2PL discrimination a and difficulty b on the same 0/1 matrix. "
        "Uses Python only (girth). Multidimensional girth MML may fail on SciPy>=2 polychoric path; use 1D here.",
        "n_persons": int(X.shape[0]),
        "n_items": int(X.shape[1]),
        "items": items,
        "discrimination_vector": [float(x) for x in a],
        "difficulty_vector": [float(x) for x in b],
    }


def write_r_script(csv_path: Path, out_dir: Path, out_r: Path) -> None:
    c = str(csv_path.resolve()).replace("\\", "/")
    coef_out = str((out_dir / "mirt_coef_output.txt").resolve()).replace("\\", "/")
    txt = f"""# Run after: install.packages(c("mirt"), repos="http<REDACTED_PATH>")
suppressPackageStartupMessages({{
  library(mirt)
}})
fn <- "{c}"
d <- read.csv(fn, fileEncoding = "UTF-8", check.names = FALSE)
stopifnot(ncol(d) == 5L)
mod <- mirt(d, model = 1, itemtype = "2PL", verbose = FALSE)
print(summary(mod))
cf <- coef(mod, simplify = TRUE)
print(cf)
sink("{coef_out}")
print(summary(mod))
print(cf)
sink()
cat("Wrote:", "{coef_out}", "\\n")
"""
    out_r.write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    _repo = Path(__file__).resolve().parent
    _art = _repo / "artifacts"
    _data = _repo / "data"
    ap.add_argument("--csv", type=Path, default=_data / "260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv")
    ap.add_argument(
        "--summary-linear",
        type=Path,
        default=_art / "infectivity_empirical_constrained_sem_260510" / "summary.json",
    )
    ap.add_argument(
        "--summary-log1p",
        type=Path,
        default=_art / "infectivity_empirical_constrained_sem_260510_log1p" / "summary.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_art / "infectivity_relative_influence_2pl_260510",
    )
    ap.add_argument(
        "--write-r-script",
        action="store_true",
        help="If set, also write run_mirt_2pl.R (optional; default is Python-only).",
    )
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    raw = _read_meta_csv(args.csv)
    wide = load_indicator_frame(raw)
    df_cc = wide.dropna()
    X = df_cc[COLS].values.astype(np.int32)

    table = build_table(
        label_linear="linear fixed loadings",
        summary_linear=args.summary_linear,
        label_log1p="log1p fixed loadings",
        summary_log1p=args.summary_log1p,
    )
    (out / "relative_influence_full_table.tsv").write_text(table, encoding="utf-8")

    csv_items = out / "infectivity_complete_case_5items.csv"
    df_cc.to_csv(csv_items, index=False, encoding="utf-8-sig")

    girth_res = run_girth_2pl(X)
    (out / "twopl_girth_mml.json").write_text(json.dumps(girth_res, indent=2, ensure_ascii=False), encoding="utf-8")

    (out / "irt_python_only.txt").write_text(
        "IRT (1D binary 2PL) was fit with Python package girth (twopl_mml). "
        "This is the same model class as mirt(..., 1, itemtype='2PL') but R is not used.\n"
        "See twopl_girth_mml.json for a and b.\n",
        encoding="utf-8",
    )

    if args.write_r_script:
        write_r_script(csv_items, out, out / "run_mirt_2pl.R")

    print(table)
    print("--- 2PL (girth twopl_mml) discrimination a ---")
    for k in COLS:
        print(f"  {k}: a = {girth_res['items'][k]['a_2pl_discrimination']:.6f}")
    print(f"\nWrote:\n  {out / 'relative_influence_full_table.tsv'}\n  {csv_items}\n  {out / 'twopl_girth_mml.json'}\n  {out / 'irt_python_only.txt'}")
    if args.write_r_script:
        print(f"  {out / 'run_mirt_2pl.R'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
