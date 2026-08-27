# -*- coding: utf-8 -*-
"""
260509_META infectivity CSV:
  (1) 판독문(Reading)에서 *구조적* Exclude 표기가 있는 행 제거
      (영문 'cannot be excluded' 등 일반 문장 오탐 방지)
  (2) 지표 NaN 열별 균형 impute: Study No. 문자열 정렬 후 앞 절반 0 / 뒤 절반 1
  (3) empirical sensitivity → fixed loadings (Liquid=1), semopy free + constrained CFA
  (4) 부트스트랩: 행 재표본마다 fixed loadings 재계산 + constrained CFA 재적합
      → Cavity 표준화적재·CFI 경험적 불확실 구간
      (semopy는 MLE; '베이지안 SEM' 대신 사전/민감도의 표본 변동을 전파하는 용도)

옵션 --exclude-regex 로 추가 제외 패턴(주의: 오탐 가능).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import analyze_infectivity_latent as ail  # noqa: E402
from meta_infectivity_empirical_constrained_sem import (  # noqa: E402
    DEFAULT_CSV,
    empirical_sensitivity_and_priors,
    load_indicator_frame,
    _read_meta_csv,
)

COLS = ail.COLS


def reading_structured_exclude(text: object) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if re.search(r"###[^#]*\bExclude\b[^#]*###", text, flags=re.IGNORECASE):
        return True
    if re.search(r"로\s+Exclude", text):
        return True
    for line in text.splitlines():
        line = line.strip()
        if re.search(r":\s*Exclude\s*$", line, flags=re.IGNORECASE):
            return True
    return False


def mask_exclude_rows(df: pd.DataFrame, *, extra_regex: str | None) -> pd.Series:
    """True = DROP row."""
    m = pd.Series(False, index=df.index)
    if "Reading" in df.columns:
        m = df["Reading"].map(reading_structured_exclude)
    if extra_regex and extra_regex.strip():
        rx = re.compile(extra_regex.strip(), flags=re.IGNORECASE | re.DOTALL)
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype) == "string":
                m = m | df[c].fillna("").astype(str).map(lambda t: bool(rx.search(t)))
    return m.fillna(False)


def impute_na_balanced_half(ind: pd.DataFrame, study_no: pd.Series) -> tuple[pd.DataFrame, dict]:
    out = ind.copy()
    log: dict = {}
    sid = study_no.reindex(out.index)
    for c in COLS:
        miss = out[c].isna()
        nmiss = int(miss.sum())
        if nmiss == 0:
            continue
        sub_idx = out.index[miss.to_numpy()]
        sord = np.argsort(sid.loc[sub_idx].astype(str).to_numpy(), kind="mergesort")
        sorted_idx = sub_idx[sord]
        n = len(sorted_idx)
        n0 = n // 2
        out.loc[sorted_idx[:n0], c] = 0.0
        out.loc[sorted_idx[n0:], c] = 1.0
        log[c] = {"n_imputed": n, "n_zero": int(n0), "n_one": int(n - n0)}
    return out, log


def _summarize_vec(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.025, 0.05, 0.5, 0.95, 0.975]).tolist()
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q025": float(qs[0]),
        "q05": float(qs[1]),
        "q50": float(qs[2]),
        "q95": float(qs[3]),
        "q975": float(qs[4]),
    }


def bootstrap_constrained_cfa(
    df_bin: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    loading_scale: str = "linear",
) -> dict:
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(int(seed))
    n = len(df_bin)
    cfi_list: list[float] = []
    cavity_std: list[float] = []
    errs = 0
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        dfb = df_bin.iloc[idx].reset_index(drop=True)
        try:
            _rep, fixed_b = empirical_sensitivity_and_priors(dfb, loading_scale=loading_scale)
            scaler = StandardScaler()
            Z = scaler.fit_transform(dfb.values)
            Z_df = pd.DataFrame(Z, columns=COLS)
            _model, stats, _ins, ins_std = ail.fit_constrained_cfa_sem(Z_df, fixed_loadings=fixed_b)
            row = stats.iloc[0]
            cfi = float(row["CFI"]) if pd.notna(row.get("CFI", np.nan)) else float("nan")
            cfi_list.append(cfi)
            std_ld = ail.extract_cfa_standardized_loadings(ins_std) if ins_std is not None else {}
            cavity_std.append(float(std_ld.get("Cavity", float("nan"))))
        except Exception:
            errs += 1
            cfi_list.append(float("nan"))
            cavity_std.append(float("nan"))

    return {
        "n_bootstrap_requested": int(n_boot),
        "n_failed_fits": int(errs),
        "CFI": _summarize_vec(np.array(cfi_list)),
        "Cavity_standardized_loading": _summarize_vec(np.array(cavity_std)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Structured Exclude + balanced NaN impute + empirical constrained CFA + bootstrap."
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "infectivity_exclude_impute_bootstrap_sem_260510",
    )
    ap.add_argument(
        "--exclude-regex",
        type=str,
        default="",
        help="Optional extra regex over object columns; match => row dropped.",
    )
    ap.add_argument("--bootstrap", type=int, default=200, help="Bootstrap replicates (0=skip).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--fixed-loading-scale",
        type=str,
        default="linear",
        choices=("linear", "log1p", "invlog1p"),
        help="Same as meta_infectivity_empirical_constrained_sem.py: map N_j+/N_L+ to fixed loadings.",
    )
    args = ap.parse_args()

    raw = _read_meta_csv(args.csv)
    n0 = len(raw)
    excl = mask_exclude_rows(raw, extra_regex=(args.exclude_regex or "").strip() or None)
    raw_kept = raw.loc[~excl].reset_index(drop=True)
    n_drop = int(excl.sum())

    study = raw_kept["Study No."] if "Study No." in raw_kept.columns else pd.Series(np.arange(len(raw_kept)))

    wide = load_indicator_frame(raw_kept)
    wide_imp, imp_log = impute_na_balanced_half(wide, study)

    if wide_imp[COLS].isna().any().any():
        raise SystemExit(f"NaN remains after impute: {wide_imp[COLS].isna().sum().to_dict()}")

    emp_report, fixed_loadings = empirical_sensitivity_and_priors(wide_imp, loading_scale=str(args.fixed_loading_scale))

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z = scaler.fit_transform(wide_imp.values)
    Z_df = pd.DataFrame(Z, columns=COLS)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_block = {
        "input_csv": str(args.csv.resolve()),
        "n_rows_original": int(n0),
        "n_rows_dropped_exclude": int(n_drop),
        "n_rows_after_exclude": int(len(raw_kept)),
        "exclude_rule": "Reading: ###...Exclude...### OR line-end :Exclude OR 로 Exclude; optional --exclude-regex",
        "exclude_regex_extra": (args.exclude_regex or "").strip() or None,
        "impute": {"method": "per_column_balanced_half_sorted_by_StudyNo", "detail": imp_log},
        "fixed_loading_scale": str(args.fixed_loading_scale),
    }
    (out_dir / "preprocess.json").write_text(json.dumps(meta_block, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "empirical_sensitivity.json").write_text(
        json.dumps(emp_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pca_res = ail.pca_pc1_loadings_and_weights(Z, COLS)
    ail.plot_pca_biplot(Z, COLS, pca_res, out_dir / "pca_biplot.png")

    sem_stats = None
    sem_model = None
    try:
        sem_model, sem_stats, sem_ins, sem_ins_std = ail.fit_cfa_sem(Z_df)
        srmr = ail.srmr_from_sigma(Z_df, sem_model)
        sem_stats = sem_stats.copy()
        sem_stats["SRMR"] = np.nan
        sem_stats.loc[sem_stats.index[0], "SRMR"] = srmr
        ail.sem_path_diagram(
            sem_ins,
            out_dir / "sem_path_diagram_free.png",
            sem_stats,
            use_standardized_on_edges=sem_ins_std is not None,
            ins_std=sem_ins_std,
        )
    except Exception as e:
        sem_stats = pd.DataFrame({"Error": [str(e)]})
        print(f"[CFA free] {e}", file=sys.stderr)

    sem_model_c = None
    sem_stats_c = None
    sem_ins_c = None
    sem_ins_std_c = None
    std_loadings_c: dict[str, float] = {}
    influence_01_c: dict[str, float] = {}
    try:
        sem_model_c, sem_stats_c, sem_ins_c, sem_ins_std_c = ail.fit_constrained_cfa_sem(
            Z_df, fixed_loadings=fixed_loadings
        )
        srmr_c = ail.srmr_from_sigma(Z_df, sem_model_c)
        sem_stats_c = sem_stats_c.copy()
        sem_stats_c["SRMR"] = np.nan
        sem_stats_c.loc[sem_stats_c.index[0], "SRMR"] = srmr_c
        std_loadings_c = ail.extract_cfa_standardized_loadings(sem_ins_std_c) if sem_ins_std_c is not None else {}
        influence_01_c = ail.influence_share_0_1(std_loadings_c) if std_loadings_c else {}
        ail.sem_path_diagram(
            sem_ins_c,
            out_dir / "sem_path_diagram_constrained.png",
            sem_stats_c,
            use_standardized_on_edges=sem_ins_std_c is not None,
            ins_std=sem_ins_std_c,
        )
    except Exception as e:
        sem_stats_c = pd.DataFrame({"Error": [str(e)]})
        print(f"[CFA constrained] {e}", file=sys.stderr)

    boot: dict | None = None
    if int(args.bootstrap) > 0:
        boot = bootstrap_constrained_cfa(
            wide_imp,
            n_boot=int(args.bootstrap),
            seed=int(args.seed),
            loading_scale=str(args.fixed_loading_scale),
        )
        (out_dir / "bootstrap_constrained.json").write_text(
            json.dumps(boot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "preprocess": meta_block,
        "n_analysis_rows": int(len(wide_imp)),
        "point_estimate": {
            "constrained_fixed_loadings": fixed_loadings,
            "constrained_cfa_fit": sem_stats_c.to_dict()
            if sem_stats_c is not None and "CFI" in sem_stats_c.columns
            else {},
            "constrained_cfa_standardized_loadings": std_loadings_c,
            "constrained_cfa_relative_influence_0_1": influence_01_c,
            "cfa_free_fit": sem_stats.to_dict() if sem_stats is not None and "CFI" in sem_stats.columns else {},
        },
        "bootstrap_constrained": boot,
        "pca_pc1_variance_ratio": pca_res["pc1_explained_variance_ratio"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    caveats = [
        "Cavity 해석: 방사선학적 지표 vs 미생물 배양 중심 잠재요인이면 표준화적재가 상대적으로 낮게 나올 수 있음.",
        "균형 impute(50/50)는 정보가 없을 때의 *가정*이며, 민감도는 bootstrap 구간으로 요약.",
        "Bootstrap은 행 재표본 MLE SEM; 진한 의미의 베이지안 사후분포와 동일하지 않음.",
    ]
    lines = [
        "=== Preprocess ===",
        json.dumps(meta_block, indent=2, ensure_ascii=False),
        "",
        "=== Empirical sensitivity + fixed loadings ===",
        json.dumps(fixed_loadings, indent=2, ensure_ascii=False),
        "",
        "=== Free CFA ===",
        sem_stats.to_string() if sem_stats is not None else "",
        "",
        "=== Constrained CFA (empirical fixed loadings) ===",
        sem_stats_c.to_string() if sem_stats_c is not None else "",
        "",
        "Constrained standardized loadings & relative influence [0,1]:",
        json.dumps(std_loadings_c, indent=2, ensure_ascii=False),
        json.dumps(influence_01_c, indent=2, ensure_ascii=False),
        "",
        "=== Bootstrap constrained CFA ===",
        json.dumps(boot, indent=2, ensure_ascii=False) if boot else "(skipped)",
        "",
        "=== Caveats (KR) ===",
        "\n".join(f"- {c}" for c in caveats),
    ]
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
