# -*- coding: utf-8 -*-
"""
Literature-AOR weighting for 5 infectivity indicators (Cavity, AFB, PCR, Solid, Liquid).

Background
----------
Earlier PC1/CFA-derived weights gave **Cavity > AFB** and **Culture > PCR**.
The user-supplied meta-analysis feedback says:
  - Sputum Smear (AFB) pooled AOR  2.15 (95% CI 1.47-3.17) is slightly higher than
    Cavitary Disease pooled AOR    1.90 (95% CI 1.26-2.84) for transmission.
  - In Lohmann 2012 (cited in the same paper):
        Sputum PCR positive       AOR 3.8 (95% CI 1.1-13.6)
        Sputum culture positive   AOR 1.7 (95% CI 1.1-2.6)
    so PCR > Culture.

This script:
  1. Builds weights directly from those published AORs.
  2. Produces both linear-AOR and log-AOR normalized weights (default = log).
  3. Optionally applies inverse-variance precision shrinkage of log(AOR) toward 0
     to discount the very wide PCR CI.
  4. Applies the weights to the 260510 complete-case cohort and writes scores +
     contributions, mirroring meta_infectivity_sensitivity_formulas_and_weighted_scores.py.

Outputs (default --out):
  artifacts/infectivity_literature_aor_weights_260510/
    summary.json
    report.txt
    scores_with_contributions.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_infectivity_latent import COLS  # noqa: E402
from meta_infectivity_empirical_constrained_sem import (  # noqa: E402
    DEFAULT_CSV,
    load_indicator_frame,
    _read_meta_csv,
)


LITERATURE_AOR: dict[str, dict[str, float | str]] = {
    "AFB_Smear": {
        "aor": 2.15,
        "ci_low": 1.47,
        "ci_high": 3.17,
        "source": "Pooled meta-analysis (Sputum Smear Positive); transmission outcome",
    },
    "Cavity": {
        "aor": 1.90,
        "ci_low": 1.26,
        "ci_high": 2.84,
        "source": "Pooled meta-analysis (Cavitary Disease); transmission outcome",
    },
    "TB_PCR": {
        "aor": 3.80,
        "ci_low": 1.10,
        "ci_high": 13.60,
        "source": "Lohmann 2012 (single study; Sputum PCR positive)",
    },
    "Solid_Culture": {
        "aor": 1.70,
        "ci_low": 1.10,
        "ci_high": 2.60,
        "source": "Lohmann 2012 (Sputum culture positive; not solid/liquid-specific in paper)",
    },
    "Liquid_Culture": {
        "aor": 1.70,
        "ci_low": 1.10,
        "ci_high": 2.60,
        "source": "Lohmann 2012 (same culture AOR; treated identically)",
    },
}


def _log(x: float) -> float:
    return float(math.log(x))


def var_log_aor(ci_low: float, ci_high: float) -> float:
    """Approx variance of log(AOR) from a 95% CI on AOR."""
    return ((math.log(ci_high) - math.log(ci_low)) / (2.0 * 1.959963984540054)) ** 2


def build_weights(
    *,
    mode: str = "log",
    precision_shrink: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return normalized weights (sum 1) keyed by COLS order, plus per-indicator details.

    mode:
      - "log": w_j ∝ log(AOR_j)
      - "linear": w_j ∝ AOR_j
    precision_shrink: shrink log(AOR) toward 0 using its CI width (inverse-variance).
      shrunk_log = log(AOR) * (prec / (prec + prec_median)), where prec=1/var.
    """
    details: dict[str, dict[str, float]] = {}
    raw_vals: list[float] = []
    for name in COLS:
        rec = LITERATURE_AOR[name]
        aor = float(rec["aor"])
        log_aor = _log(aor)
        v = var_log_aor(float(rec["ci_low"]), float(rec["ci_high"]))
        prec = 1.0 / v if v > 0 else float("inf")
        details[name] = {
            "aor": aor,
            "log_aor": log_aor,
            "ci_low": float(rec["ci_low"]),
            "ci_high": float(rec["ci_high"]),
            "var_log_aor": v,
            "precision": prec,
            "source": rec["source"],
        }

    if precision_shrink:
        precs = [details[n]["precision"] for n in COLS]
        prec_med = float(np.median(precs))
        for n in COLS:
            shrink = details[n]["precision"] / (details[n]["precision"] + prec_med)
            details[n]["shrink_factor"] = shrink
            details[n]["log_aor_shrunk"] = details[n]["log_aor"] * shrink

    for n in COLS:
        if mode == "log":
            v = details[n].get("log_aor_shrunk", details[n]["log_aor"]) if precision_shrink else details[n]["log_aor"]
        elif mode == "linear":
            v = details[n]["aor"]
        else:
            raise ValueError(f"unknown mode={mode}")
        details[n]["raw_weight"] = float(v)
        raw_vals.append(float(v))

    s = float(sum(raw_vals))
    if s <= 0:
        raise ValueError("nonpositive weight sum; check inputs")
    weights = {n: float(details[n]["raw_weight"] / s) for n in COLS}
    return weights, details


def compute_scores(X: np.ndarray, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Score = sum_j w_j * x_ij (raw 0/1, no z-scoring; weights sum to 1).
    Returns (S, contrib_n_by_5)."""
    w = np.array([weights[c] for c in COLS], dtype=float)
    contrib = X * w.reshape(1, -1)
    S = contrib.sum(axis=1)
    return S, contrib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "infectivity_literature_aor_weights_260510",
    )
    ap.add_argument("--mode", type=str, default="log", choices=("log", "linear"))
    ap.add_argument(
        "--precision-shrink",
        action="store_true",
        help="Shrink log(AOR) by its CI precision (down-weight wide-CI estimates such as PCR).",
    )
    args = ap.parse_args()

    weights, details = build_weights(mode=args.mode, precision_shrink=args.precision_shrink)

    weights_log_only, _ = build_weights(mode="log", precision_shrink=False)
    weights_linear_only, _ = build_weights(mode="linear", precision_shrink=False)
    weights_log_shrunk, _ = build_weights(mode="log", precision_shrink=True)

    raw = _read_meta_csv(args.csv)
    wide = load_indicator_frame(raw)
    df_cc = wide.apply(pd.to_numeric, errors="coerce").dropna()
    n_cc = len(df_cc)
    if n_cc < 30:
        print(f"[warn] n_complete_case={n_cc}", file=sys.stderr)
    X = df_cc[COLS].values.astype(float)
    S, contrib = compute_scores(X, weights)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.DataFrame(index=raw.index)
    if "Study No." in raw.columns:
        scores["Study No."] = raw["Study No."]
    mask_cc = wide.notna().all(axis=1)
    scores.loc[mask_cc, "score_literature_aor"] = S
    scores.loc[mask_cc, "score_raw_sum_0_5"] = X.sum(axis=1)
    for j, name in enumerate(COLS):
        scores.loc[mask_cc, f"contrib_{name}"] = contrib[:, j]
        scores.loc[mask_cc, f"raw_{name}"] = X[:, j]

    merged = raw.copy()
    for c in scores.columns:
        merged[c] = scores[c].values
    out_csv = out / "scores_with_contributions.csv"
    merged.to_csv(out_csv, index=False, encoding="cp949", na_rep="")

    summary = {
        "input_csv": str(args.csv.resolve()),
        "n_complete_case": int(n_cc),
        "mode_used": args.mode,
        "precision_shrink_used": bool(args.precision_shrink),
        "weights": weights,
        "weights_variants": {
            "log_aor": weights_log_only,
            "linear_aor": weights_linear_only,
            "log_aor_precision_shrunk": weights_log_shrunk,
        },
        "literature_aor_table": LITERATURE_AOR,
        "details": details,
        "ordering_check": {
            "AFB_ge_Cavity": weights["AFB_Smear"] >= weights["Cavity"],
            "PCR_gt_Solid": weights["TB_PCR"] > weights["Solid_Culture"],
            "PCR_gt_Liquid": weights["TB_PCR"] > weights["Liquid_Culture"],
        },
        "method_notes": [
            "Weights derived from published adjusted odds ratios for TB transmission:",
            " - AFB and Cavity from pooled meta-analysis (paper provided by user).",
            " - PCR and Culture from Lohmann 2012 (per-study), since the meta-analysis",
            "   did not pool PCR or split Solid/Liquid culture.",
            "Solid and Liquid culture share the same AOR=1.70; if you need to break the tie,",
            "use empirical sensitivity (Liquid > Solid) downstream.",
            "Score per row = sum_j w_j * 0/1 (no z-scoring); weights sum to 1.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("Literature-AOR weighted infectivity score")
    lines.append("=" * 60)
    lines.append(f"input_csv: {args.csv}")
    lines.append(f"n_complete_case: {n_cc}")
    lines.append(f"mode: {args.mode}   precision_shrink: {args.precision_shrink}")
    lines.append("")
    lines.append("Per-indicator (literature):")
    lines.append("  indicator        AOR    CI95          weight  source")
    for name in COLS:
        d = details[name]
        lines.append(
            f"  {name:<14}  {d['aor']:<5}  {d['ci_low']:.2f}-{d['ci_high']:.2f}     "
            f"{weights[name]:.4f}  {d['source']}"
        )
    lines.append("")
    lines.append("Ordering check vs user feedback:")
    lines.append(f"  AFB >= Cavity : {weights['AFB_Smear'] >= weights['Cavity']}")
    lines.append(f"  PCR  > Solid  : {weights['TB_PCR'] > weights['Solid_Culture']}")
    lines.append(f"  PCR  > Liquid : {weights['TB_PCR'] > weights['Liquid_Culture']}")
    lines.append("")
    lines.append("Variants (always normalized to sum 1):")
    for vname, w in (
        ("log_aor", weights_log_only),
        ("linear_aor", weights_linear_only),
        ("log_aor_precision_shrunk", weights_log_shrunk),
    ):
        line = "  " + vname.ljust(28) + " " + " ".join(f"{n}={w[n]:.4f}" for n in COLS)
        lines.append(line)
    lines.append("")
    lines.append("Output files:")
    lines.append(f"  {out / 'summary.json'}")
    lines.append(f"  {out_csv}")
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
