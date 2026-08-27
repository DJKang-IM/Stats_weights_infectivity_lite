# -*- coding: utf-8 -*-
"""
260509_META_Infectivity_weight-rule_CSV_Edited (single-hospital meta) ->

1) Positive counts D1..D5 (도말, PCR, 고체, 액체, Cavity 유무).
2) Empirical sensitivity ratios vs proxy gold = Liquid culture (액체 배양 양성 수 기준):
     X_AF = n(Liquid+) / n(AFB+),  Y_SO = n(Liquid+) / n(Solid+),
     Z_PC = n(Liquid+) / n(PCR+),  Z_CV = n(Liquid+) / n(Cavity+).
   Constrained CFA fixed loadings (Liquid = 1), from the same positive counts:
     - linear (default): lambda_j = N_j+ / N_L+
     - log1p:          lambda_j = ln(1 + N_j+/N_L+) / ln(2)  so lambda_Liquid=1 and low ratios are uplifted vs linear.
     - invlog1p:       lambda_j = ln(2) / ln(1 + N_j+/N_L+)  (reciprocal of log1p on r, Liquid=1); boosts very low r strongly.

3) StandardScaler on 0/1 indicators -> PCA + free CFA + constrained CFA (semopy).

Outputs default: artifacts/infectivity_empirical_constrained_sem_260509/
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_infectivity_latent as ail  # noqa: E402

COLS = ail.COLS

# Canonical English column order for ail / semopy (matches COLS)
KOREAN_TO_ENGLISH = {
    "도말검사": "AFB_Smear",
    "TB-PCR검사": "TB_PCR",
    "배양검사(고체)": "Solid_Culture",
    "배양검사(액체)": "Liquid_Culture",
    "Cavity 유무": "Cavity",
}

_REPO = Path(__file__).resolve().parent
_DATA = _REPO / "data"
_ARTIFACTS = _REPO / "artifacts"
DEFAULT_CSV = _DATA / "260509_META_Infectivity_weight-rule_CSV_Edited.csv"
DEFAULT_OUT = _ARTIFACTS / "infectivity_empirical_constrained_sem_260509"


def _read_meta_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def load_indicator_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing_kr = [kr for kr in KOREAN_TO_ENGLISH if kr not in df_raw.columns]
    if missing_kr:
        raise ValueError(f"CSV missing columns: {missing_kr}. Found: {list(df_raw.columns)}")
    out = pd.DataFrame()
    for kr, en in KOREAN_TO_ENGLISH.items():
        out[en] = pd.to_numeric(df_raw[kr], errors="coerce")
    # ail COLS order: Cavity first
    return out[COLS].copy()


def empirical_sensitivity_and_priors(
    df_bin: pd.DataFrame,
    *,
    loading_scale: str = "linear",
) -> tuple[dict, dict[str, float]]:
    """
    df_bin: all columns in {0,1} or NaN (missing).

    loading_scale:
      - "linear": fixed lambda_j = N_j+ / N_L+ (j != Liquid), Liquid=1.
      - "log1p":  fixed lambda_j = ln(1 + N_j+/N_L+) / ln(2); same counts, concave transform (boosts smaller ratios).
      - "invlog1p": fixed lambda_j = ln(2) / ln(1 + N_j+/N_L+); reciprocal-of-log1p on r, Liquid=1 (stronger boost for small r).

    Returns (report_dict, fixed_loadings for Liquid=1 convention).
    """
    scale = (loading_scale or "linear").strip().lower()
    if scale not in ("linear", "log1p", "invlog1p"):
        raise ValueError(f"loading_scale must be 'linear', 'log1p', or 'invlog1p', got {loading_scale!r}")
    df = df_bin.dropna()
    n = len(df)
    if n < 30:
        print(f"[warn] n={n} complete cases — fit indices indicative only.", file=sys.stderr)

    counts_pos = {c: int(df[c].sum()) for c in COLS}
    counts_neg = {c: int((df[c] == 0).sum()) for c in COLS}

    n_l = counts_pos["Liquid_Culture"]

    def ratio_liquid_over(name: str) -> float | None:
        d = counts_pos[name]
        if d <= 0:
            return None
        return float(n_l) / float(d)

    ratios = {
        "Liquid_per_AFB=(proxy_vs_AF)": ratio_liquid_over("AFB_Smear"),
        "Liquid_per_Solid": ratio_liquid_over("Solid_Culture"),
        "Liquid_per_PCR": ratio_liquid_over("TB_PCR"),
        "Liquid_per_Cavity": ratio_liquid_over("Cavity"),
    }

    if n_l <= 0:
        raise SystemExit(
            "No Liquid-culture positives in complete-case cohort; cannot set Liquid=1 loading prior. "
            "Check meta / cohort definition."
        )

    denom_log_ref = math.log1p(1.0)  # ln(2); lambda_liquid when N_L/N_L=1

    fixed: dict[str, float] = {"Liquid_Culture": 1.0}
    linear_lam: dict[str, float] = {"Liquid_Culture": 1.0}
    for c in COLS:
        if c == "Liquid_Culture":
            continue
        num = counts_pos[c]
        r = float(num) / float(n_l)
        if r <= 0:
            r = 1e-6
        linear_lam[c] = r
        if scale == "linear":
            fixed[c] = r
        elif scale == "log1p":
            fixed[c] = float(math.log1p(r) / denom_log_ref)
        else:
            lp = max(float(math.log1p(r)), 1e-12)
            fixed[c] = float(denom_log_ref / lp)

    if scale == "linear":
        loading_note = (
            "Constrained CFA: Liquid_Culture=1; other loadings = N_j+ / N_L+ (same evidence as linear ratio)."
        )
    elif scale == "log1p":
        loading_note = (
            "Constrained CFA: Liquid_Culture=1; other loadings = ln(1 + N_j+/N_L+) / ln(2). "
            "Same positive counts as linear; log1p compresses large ratios and raises small ratios vs linear scale."
        )
    else:
        loading_note = (
            "Constrained CFA: Liquid_Culture=1; other loadings = ln(2) / ln(1 + N_j+/N_L+). "
            "Double transform sense: after forming r=N_j+/N_L+, use reciprocal of log1p(r), normalized so Liquid=1; "
            "very small r get much larger fixed loadings than log1p or linear."
        )

    report = {
        "definition": {
            "proxy_gold": "Liquid_Culture (액체 배양) 양성 수 - 단일 병원 메타 기준 임시 기준.",
            "ratio_X_user_style": "n(Liquid+) / n(test+) - 같은 코호트·완전 사례 기준.",
            "fixed_loading_scale": scale,
            "loading_prior": loading_note,
        },
        "n_complete_case": n,
        "positive_counts": counts_pos,
        "negative_counts": counts_neg,
        "liquid_over_positive_ratios": ratios,
        "linear_loadings_Nj_over_NL": linear_lam,
        "inverse_for_loading": {c: (float(counts_pos[c]) / float(n_l)) for c in COLS},
        "constrained_fixed_loadings_unnormalized": fixed,
    }
    return report, fixed


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Empirical sensitivity + constrained SEM from 260509 infectivity meta CSV.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--fixed-loading-scale",
        type=str,
        default="linear",
        choices=("linear", "log1p", "invlog1p"),
        help="How to map count ratios N_j+/N_L+ to constrained fixed loadings (Liquid always 1).",
    )
    args = ap.parse_args()

    raw = _read_meta_csv(args.csv)
    wide = load_indicator_frame(raw)
    df = wide.apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 10:
        raise SystemExit(f"Too few complete rows after NA drop: n={len(df)}")

    emp_report, fixed_loadings = empirical_sensitivity_and_priors(wide, loading_scale=str(args.fixed_loading_scale))

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z = scaler.fit_transform(df.values)
    Z_df = pd.DataFrame(Z, columns=COLS)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "empirical_sensitivity.json").write_text(
        json.dumps(emp_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # PCA + free CFA
    pca_res = ail.pca_pc1_loadings_and_weights(Z, COLS)
    ail.plot_pca_biplot(Z, COLS, pca_res, out_dir / "pca_biplot.png")

    sem_model = sem_stats = sem_ins = sem_ins_std = None
    try:
        sem_model, sem_stats, sem_ins, sem_ins_std = ail.fit_cfa_sem(Z_df)
        srmr = ail.srmr_from_sigma(Z_df, sem_model)
        sem_stats = sem_stats.copy()
        sem_stats["SRMR"] = np.nan
        sem_stats.loc[sem_stats.index[0], "SRMR"] = srmr
    except Exception as e:
        print(f"[CFA] Failed: {e}", file=sys.stderr)
        sem_stats = pd.DataFrame({"Error": [str(e)]})

    if sem_ins is not None:
        ail.sem_path_diagram(
            sem_ins,
            out_dir / "sem_path_diagram_free.png",
            sem_stats,
            use_standardized_on_edges=sem_ins_std is not None,
            ins_std=sem_ins_std,
        )

    std_loadings = ail.extract_cfa_standardized_loadings(sem_ins_std) if sem_ins_std is not None else {}
    influence_01 = ail.influence_share_0_1(std_loadings) if std_loadings else {}

    # Constrained CFA
    sem_model_c = sem_stats_c = sem_ins_c = sem_ins_std_c = None
    try:
        sem_model_c, sem_stats_c, sem_ins_c, sem_ins_std_c = ail.fit_constrained_cfa_sem(Z_df, fixed_loadings=fixed_loadings)
        srmr_c = ail.srmr_from_sigma(Z_df, sem_model_c)
        sem_stats_c = sem_stats_c.copy()
        sem_stats_c["SRMR"] = np.nan
        sem_stats_c.loc[sem_stats_c.index[0], "SRMR"] = srmr_c
    except Exception as e:
        print(f"[CFA-Constrained] Failed: {e}", file=sys.stderr)
        sem_stats_c = pd.DataFrame({"Error": [str(e)]})

    if sem_ins_c is not None:
        ail.sem_path_diagram(
            sem_ins_c,
            out_dir / "sem_path_diagram_constrained.png",
            sem_stats_c,
            use_standardized_on_edges=sem_ins_std_c is not None,
            ins_std=sem_ins_std_c,
        )

    std_loadings_c = ail.extract_cfa_standardized_loadings(sem_ins_std_c) if sem_ins_std_c is not None else {}
    influence_01_c = ail.influence_share_0_1(std_loadings_c) if std_loadings_c else {}

    summary = {
        "input_csv": str(args.csv.resolve()),
        "n_complete_case": int(len(df)),
        "empirical_sensitivity": emp_report,
        "pca": {
            "pc1_variance_ratio": pca_res["pc1_explained_variance_ratio"],
            "loadings_pc1": pca_res["loadings_pc1_correlation"],
            "weights_normalized_sq": pca_res["normalized_weights_sq_loading"],
        },
        "cfa_free_fit": sem_stats.to_dict() if sem_stats is not None and "CFI" in sem_stats.columns else {},
        "cfa_free_standardized_loadings": std_loadings,
        "cfa_free_relative_influence_0_1": influence_01,
        "constrained_fixed_loadings": fixed_loadings,
        "fixed_loading_scale": str(args.fixed_loading_scale),
        "constrained_cfa_fit": sem_stats_c.to_dict() if sem_stats_c is not None and "CFI" in sem_stats_c.columns else {},
        "constrained_cfa_standardized_loadings": std_loadings_c,
        "constrained_cfa_relative_influence_0_1": influence_01_c,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "=== Empirical sensitivity (Liquid culture = proxy gold) ===",
        f"n complete-case rows (D1-D5 non-missing 0/1): {len(df)}",
        "",
        "Positive counts:",
        json.dumps(emp_report["positive_counts"], indent=2),
        "",
        "Ratios n(Liquid+)/n(test+) (higher => fewer positives vs liquid):",
        json.dumps(emp_report["liquid_over_positive_ratios"], indent=2),
        "",
        f"Constrained CFA fixed loadings (scale={args.fixed_loading_scale}; Liquid_Culture=1):",
        json.dumps(fixed_loadings, indent=2),
        "",
        "=== Free CFA fit ===",
        sem_stats.to_string() if sem_stats is not None else "",
        "",
        "=== Constrained CFA fit ===",
        sem_stats_c.to_string() if sem_stats_c is not None else "",
        "",
        "Constrained standardized loadings (Est. Std) & relative influence [0,1]:",
        json.dumps(std_loadings_c, indent=2),
        json.dumps(influence_01_c, indent=2),
    ]
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
