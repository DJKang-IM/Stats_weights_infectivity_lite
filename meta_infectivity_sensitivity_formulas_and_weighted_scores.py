# -*- coding: utf-8 -*-
"""
(1)(2) Empirical "민감도" 수식 정리 + (3) 변별력 기반 가중 점수 (IRT-style heuristic)

1) 리포트에 쓰는 민감도 비율 (proxy gold = 액체 배양 양성 수 N_L+)
   완전 사례 집합에서 각 지표 j의 양성 수 N_j+ (0/1 합)일 때:
     R_j = N_L+ / N_j+   (보고용; j에 양성이 적을수록 R_j 큼)

2) Constrained CFA 고정 적재 (비로그, 비추가 역수 보정 없음)
     λ_Liquid = 1
     λ_j      = N_j+ / N_L+   (j ≠ Liquid)
   즉 R_j = 1 / λ_j (곱셈 의미의 역수). log(R_j), log(λ_j) 변환은 사용하지 않음.

3) 점수 (IRT-like 휴리스틱)
   - 잠재축 proxy: 5개 지표를 표준화한 뒤 PCA 제1주성분 점수 θ (n×1).
   - 변별력: d_j = |corr(z_j, θ)| (이진 지표의 θ 대비 변별 강도).
   - 가중: w_j ∝ d_j, 정규화 후 Cavity에만 배수 --cavity-boost 적용 후 다시 정규화.
   - 행 i 점수: S_i = Σ_j w_j * z_ij ; 기여도 열: contrib_j_i = w_j * z_ij.

출력: --out 디렉터리에 report_formulas.txt, summary.json, scores.csv (Study No. 있으면 병합)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_infectivity_latent import COLS  # noqa: E402
from meta_infectivity_empirical_constrained_sem import (  # noqa: E402
    DEFAULT_CSV,
    empirical_sensitivity_and_priors,
    load_indicator_frame,
    _read_meta_csv,
)


def formulas_kr() -> str:
    return r"""
================================================================================
(1) 민감도(보고용 비율) - 완전 사례(5개 지표 모두 0/1 관측) 부분집합, 표본 크기 n
================================================================================
지표 j ∈ {Cavity, AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture}
  N_j+ = Σ_i x_ij   (양성 0/1 합)
  N_L+ = Liquid_Culture 열의 양성 수 (액체 배양을 proxy gold 로 둘 때)

보고에 쓰는 비율 (코드: liquid_over_positive_ratios):
  R_j = N_L+ / N_j+     (단, N_j+ = 0 이면 정의 불가 → None)

================================================================================
(2) Constrained CFA 고정 적재 (동일 양성 빈도 N_j+, N_L+)
================================================================================
  기본 linear:  λ_Liquid = 1 ,  λ_j = N_j+ / N_L+  (j ≠ Liquid)
  옵션 log1p:  λ_Liquid = 1 ,  λ_j = ln(1 + N_j+/N_L+) / ln(2)
               (작은 비율이 선형 대비 상대적으로 커짐)
  옵션 invlog1p: λ_Liquid = 1 ,  λ_j = ln(2) / ln(1 + N_j+/N_L+)
               (r=N_j+/N_L+에 대해 log1p(r)의 역수에 ln(2)로 스케일; log1p의 '역' 변환, 저비율 지표 고정적재가 크게 부풀어듦)

  보고 비율 R_j = N_L+/N_j+ = 1/λ_j (linear일 때의 곱 역수 관계)

JSON 키 linear_loadings_Nj_over_NL (meta 스크립트) / inverse_for_loading:
  linear 적재와 동일 스케일의 N_j+/N_L+ ; log1p일 때도 보고용 R_j는 동일 빈도로 계산.

================================================================================
(3) 변별력 가중 점수 (IRT 정식 추정 아님; θ=PC1 기반 휴리스틱)
================================================================================
  z_ij = (x_ij - μ_j) / σ_j  (완전 사례에서만 μ,σ 추정)
  θ_i  = PC1_i (z 행렬에 PCA, 1성분)
  d_j  = |corr(z_.j, θ)|
  w_j' = d_j / Σ_k d_k
  Cavity 부스트 k 적용: w_Cavity'' = k * w_Cavity' , 나머지 j≠Cavity: w_j'' = w_j'
           그 후 w_j = w_j'' / Σ w''
  contrib_ji = w_j * z_ij
  S_i = Σ_j contrib_ji
================================================================================
""".strip()


def compute_weighted_scores(
    X: np.ndarray,
    *,
    cavity_boost: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, float, dict[str, float], dict[str, float]]:
    """
    X: (n,5) binary complete-case.
    Returns (theta_pc1, S_total, weights_dict, contrib matrix n×5, pc1_var_ratio, d_map, w_pre_boost_map).
    """
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    theta = pca.fit_transform(Z).ravel()
    var1 = float(pca.explained_variance_ratio_[0])
    d = np.array([abs(np.corrcoef(Z[:, j], theta)[0, 1]) for j in range(5)], dtype=float)
    d = np.nan_to_num(d, nan=0.0)
    if d.sum() <= 0:
        d = np.ones(5)
    w_pre = d / d.sum()
    cav_idx = COLS.index("Cavity")
    w2 = w_pre.copy()
    w2[cav_idx] *= float(cavity_boost)
    w2 = w2 / w2.sum()
    contrib = Z * w2.reshape(1, -1)
    S = contrib.sum(axis=1)
    weights = {COLS[j]: float(w2[j]) for j in range(5)}
    d_map = {COLS[j]: float(d[j]) for j in range(5)}
    w_pre_map = {COLS[j]: float(w_pre[j]) for j in range(5)}
    return theta, S, weights, contrib, var1, d_map, w_pre_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "infectivity_sensitivity_formulas_weighted_scores_260510",
    )
    ap.add_argument(
        "--cavity-boost",
        type=float,
        default=1.5,
        help="Multiply Cavity discrimination weight after normalization (default 1.5).",
    )
    ap.add_argument(
        "--fixed-loading-scale",
        type=str,
        default="linear",
        choices=("linear", "log1p", "invlog1p"),
        help="For empirical_sensitivity JSON only (fixed loadings in summary).",
    )
    args = ap.parse_args()

    raw = _read_meta_csv(args.csv)
    wide = load_indicator_frame(raw)
    df_cc = wide.apply(pd.to_numeric, errors="coerce").dropna()
    if len(df_cc) < 30:
        print(f"[warn] n={len(df_cc)} complete cases", file=sys.stderr)

    emp_report, fixed_loadings = empirical_sensitivity_and_priors(wide, loading_scale=str(args.fixed_loading_scale))
    X = df_cc[COLS].values.astype(float)
    theta, S, weights, contrib, var1, d_map, w_pre_map = compute_weighted_scores(X, cavity_boost=args.cavity_boost)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report_formulas.txt").write_text(formulas_kr() + "\n", encoding="utf-8")

    # Align scores to full raw table (NaN for incomplete rows)
    scores = pd.DataFrame(index=raw.index)
    if "Study No." in raw.columns:
        scores["Study No."] = raw["Study No."]
    mask_cc = wide.notna().all(axis=1)
    scores.loc[mask_cc, "theta_pc1_z"] = theta
    scores.loc[mask_cc, "score_weighted_total"] = S
    for j, name in enumerate(COLS):
        scores.loc[mask_cc, f"contrib_{name}"] = contrib[:, j]
    scores.loc[mask_cc, "score_raw_sum_0_5"] = X.sum(axis=1)
    for j, name in enumerate(COLS):
        scores.loc[mask_cc, f"raw_{name}"] = X[:, j]

    summary = {
        "input_csv": str(args.csv.resolve()),
        "n_complete_case": int(len(df_cc)),
        "cavity_boost_applied": float(args.cavity_boost),
        "empirical_sensitivity_report_keys": list(emp_report.keys()),
        "fixed_loading_scale": str(args.fixed_loading_scale),
        "fixed_loadings_lambda_from_counts": fixed_loadings,
        "liquid_over_positive_Rj": emp_report.get("liquid_over_positive_ratios"),
        "discrimination_abs_corr_z_pc1": d_map,
        "weights_before_cavity_boost": w_pre_map,
        "weights_on_standardized_indicators": weights,
        "weighting_method": "abs corr(z_j, PC1) normalized; Cavity multiplied by cavity_boost then renormalized",
        "pca_pc1_variance_ratio": var1,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = out / "scores_with_contributions.csv"
    merged = raw.copy()
    for c in scores.columns:
        merged[c] = scores[c].values
    merged.to_csv(out_csv, index=False, encoding="cp949", na_rep="")

    print(formulas_kr())
    print("\n--- summary (weights) ---\n", json.dumps(weights, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
