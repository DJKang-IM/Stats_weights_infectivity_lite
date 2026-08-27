# -*- coding: utf-8 -*-
"""
Three-way comparison of infectivity weighting schemes for the 5 binary indicators
(Cavity, AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture).

  A. PR#7 Entropy Effective Rank
       Block-level weighting via Roy & Vetterli (2007) effective rank of polychoric
       block-correlation matrices, with within-microbiology shares from PLS-PM.
       Source: TBC_CAD_INFECTIVITY/TB Phase III/reports/
               20260520 PR7 Entropy Effective Rank - Results.json
  B. Literature AOR (log; user-supplied feedback)
       AFB 2.15, Cavity 1.90, PCR 3.8, Culture 1.7 for both Solid and Liquid.
       This was the basis for the earlier feedback. CULTURE AOR is borrowed from
       the paper's "Broncho-alveolar lavage AFB positive" row, NOT sputum culture
       (the paper does not give an AOR for sputum culture). Flagged here for
       transparency.
  C. Paper-faithful AOR (Asadi et al., Epidemiology and Infection, PMC9134570)
       Pooled meta-AOR:  Smear 2.15 (1.47-3.17),  Cavity 1.90 (1.26-2.84)
       Single study :    PCR 3.80 (1.10-13.60) [Lohmann 2013, ref [13]]
       Culture     :     NOT reported (paper has only references [24],[33] in the
                         non-significant column with no numerical AOR).
       For Culture we therefore provide three sensitivity variants (1.0 / 1.7 / 2.0)
       and one variant that mimics PR#7's within-block share for Solid vs Liquid.

The script:
  1. Loads the PR#7 JSON to seed Method A.
  2. Computes log-AOR-normalised weights for Methods B and C-variants.
  3. Applies all weight sets to the same 459-row complete-case CSV.
  4. Writes a side-by-side report (md + csv) plus per-patient scores under each
     scheme to artifacts/infectivity_pr7_vs_literature_aor_compare_260510/.
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
    load_indicator_frame,
    _read_meta_csv,
)


DEFAULT_PR7_JSON = Path(
    r"D:\TBC_CAD_INFECTIVITY\TB Phase III\reports\20260520 PR7 Entropy Effective Rank - Results.json"
)
DEFAULT_CSV = Path(r"D:\260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv")


def _log(x: float) -> float:
    return float(math.log(x))


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def log_aor_weights(aor_by_indicator: dict[str, float]) -> dict[str, float]:
    raw = {k: max(_log(v), 1e-9) for k, v in aor_by_indicator.items()}
    return normalize_weights(raw)


def load_pr7_weights(path: Path) -> tuple[dict[str, float], dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    w = {k: float(v) for k, v in obj["final_weights"].items()}
    return w, obj


def build_methods(pr7_json: Path) -> dict[str, dict]:
    """Return a dict of method -> {weights, aor_table, note}."""
    pr7_w, pr7_obj = load_pr7_weights(pr7_json)

    aor_B = {
        "Cavity": 1.90,
        "AFB_Smear": 2.15,
        "TB_PCR": 3.80,
        "Solid_Culture": 1.70,
        "Liquid_Culture": 1.70,
    }
    w_B = log_aor_weights(aor_B)

    paper_table = {
        "Cavity": {"aor": 1.90, "ci": (1.26, 2.84), "source": "Pooled meta-AOR (Asadi)"},
        "AFB_Smear": {"aor": 2.15, "ci": (1.47, 3.17), "source": "Pooled meta-AOR (Asadi)"},
        "TB_PCR": {"aor": 3.80, "ci": (1.10, 13.60), "source": "Single study Lohmann 2013 [13]"},
    }

    def variant(cult_aor: float, label: str, note: str) -> dict:
        aor = {
            "Cavity": 1.90,
            "AFB_Smear": 2.15,
            "TB_PCR": 3.80,
            "Solid_Culture": cult_aor,
            "Liquid_Culture": cult_aor,
        }
        return {
            "weights": log_aor_weights(aor),
            "aor_table": aor,
            "note": note,
            "label": label,
        }

    pr7_within = pr7_obj["within_micro_outer_weights_norm"]
    micro_share = pr7_obj["block_level"]["microbiology"]["share"]
    img_share = pr7_obj["block_level"]["imaging"]["share"]

    def variant_c_pr7_split(cult_aor: float) -> dict:
        """Smear/PCR/Cavity from log(AOR); Solid vs Liquid split by PR#7 within-block ratio."""
        log_cavity = _log(1.90)
        log_smear = _log(2.15)
        log_pcr = _log(3.80)
        log_cult = max(_log(cult_aor), 1e-9)
        sl = pr7_within["Solid_Culture"]
        ll = pr7_within["Liquid_Culture"]
        ssum = sl + ll
        w_solid_raw = log_cult * (sl / ssum)
        w_liquid_raw = log_cult * (ll / ssum)
        raw = {
            "Cavity": log_cavity,
            "AFB_Smear": log_smear,
            "TB_PCR": log_pcr,
            "Solid_Culture": w_solid_raw,
            "Liquid_Culture": w_liquid_raw,
        }
        return {
            "weights": normalize_weights(raw),
            "aor_table": {
                "Cavity": 1.90,
                "AFB_Smear": 2.15,
                "TB_PCR": 3.80,
                "Solid_Culture": cult_aor,
                "Liquid_Culture": cult_aor,
            },
            "note": (
                f"Paper-faithful AOR for smear/cavity/PCR; culture AOR={cult_aor:.2f}; "
                "Solid vs Liquid split by PR#7 PLS within-microbiology ratio."
            ),
            "label": f"C-pr7split-cult{cult_aor:.2f}",
        }

    methods = {
        "A_PR7_entropy_effective_rank": {
            "weights": pr7_w,
            "aor_table": None,
            "note": "Data-driven (polychoric + PLS within micro); no AOR.",
            "label": "PR#7 Entropy Effective Rank",
        },
        "B_literature_aor_log_user_feedback": {
            "weights": w_B,
            "aor_table": aor_B,
            "note": (
                "User's earlier feedback. CULTURE 1.70 is actually the BAL AFB AOR from "
                "Lohmann; sputum culture has no AOR in Asadi paper. Kept for direct comparison."
            ),
            "label": "Lit-AOR log (feedback)",
        },
        "C1_paper_correct_culture_1p00": variant(1.00, "Paper-correct, culture AOR=1.00 (null)",
            "Culture treated as no effect (null) since paper has no number."),
        "C2_paper_correct_culture_1p70": variant(1.70, "Paper-correct, culture AOR=1.70 (=BAL proxy)",
            "Same number as Method B but explicit that 1.70 came from BAL, not sputum culture."),
        "C3_paper_correct_culture_2p00": variant(2.00, "Paper-correct, culture AOR=2.00",
            "Mid-range: bacteriologic confirmation typically >= 2x; sensitivity check."),
        "C4_paper_pr7split_culture_2p00": variant_c_pr7_split(2.00),
        "paper_meta_extra_reference": {
            "weights": None,
            "aor_table": paper_table,
            "note": "Reference table of AORs verified from PMC9134570 (Asadi et al., EI 2022).",
            "label": "Asadi paper AOR table (reference)",
        },
    }
    return methods


def apply_weights(X: np.ndarray, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    w = np.array([weights[c] for c in COLS], dtype=float)
    contrib = X * w.reshape(1, -1)
    return contrib.sum(axis=1), contrib


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--pr7-json", type=Path, default=DEFAULT_PR7_JSON)
    ap.add_argument(
        "--out",
        type=Path,
        default=_SCRIPT_DIR / "artifacts" / "infectivity_pr7_vs_literature_aor_compare_260510",
    )
    args = ap.parse_args()

    if not args.pr7_json.exists():
        print(f"[error] PR#7 JSON not found: {args.pr7_json}", file=sys.stderr)
        return 1
    methods = build_methods(args.pr7_json)

    raw = _read_meta_csv(args.csv)
    wide = load_indicator_frame(raw)
    df_cc = wide.apply(pd.to_numeric, errors="coerce").dropna()
    X = df_cc[COLS].values.astype(float)
    n_cc = len(df_cc)
    pos_counts = {c: int(df_cc[c].sum()) for c in COLS}

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    score_df = pd.DataFrame(index=df_cc.index)
    if "Study No." in raw.columns:
        score_df["Study No."] = raw.loc[df_cc.index, "Study No."]
    for j, name in enumerate(COLS):
        score_df[f"raw_{name}"] = X[:, j]
    score_df["raw_sum_0_5"] = X.sum(axis=1)

    for key, m in methods.items():
        if m["weights"] is None:
            continue
        S, _ = apply_weights(X, m["weights"])
        score_df[f"score_{key}"] = S
        row = {"method": key, "label": m["label"], "note": m["note"]}
        for c in COLS:
            row[c] = float(m["weights"][c])
        row["score_mean"] = float(S.mean())
        row["score_std"] = float(S.std(ddof=1))
        row["AFB_ge_Cavity"] = bool(m["weights"]["AFB_Smear"] >= m["weights"]["Cavity"])
        row["PCR_gt_Solid"] = bool(m["weights"]["TB_PCR"] > m["weights"]["Solid_Culture"])
        row["PCR_gt_Liquid"] = bool(m["weights"]["TB_PCR"] > m["weights"]["Liquid_Culture"])
        rows.append(row)

    table_csv = out / "weights_comparison.csv"
    pd.DataFrame(rows).to_csv(table_csv, index=False, encoding="utf-8-sig")

    scores_csv = out / "scores_per_method.csv"
    score_df.to_csv(scores_csv, index=False, encoding="utf-8-sig")

    md_lines: list[str] = []
    md_lines.append("# PR#7 vs Literature-AOR weighting — comparison\n")
    md_lines.append(f"- Cohort (complete-case): **n = {n_cc}** (CSV: `{args.csv}`)")
    md_lines.append(f"- Positive counts in COLS order {COLS}: {pos_counts}\n")
    md_lines.append("## Paper of record (AORs)\n")
    md_lines.append(
        "Asadi et al., *Risk factors for infectiousness of patients with tuberculosis: "
        "a systematic review and meta-analysis*, Epidemiology and Infection 2022 (PMC9134570).\n"
    )
    md_lines.append("Three exposures had **pooled meta-AOR**: sputum smear, lung cavitation, HIV.")
    md_lines.append("- Sputum smear positive: AOR 2.15 (1.47-3.17), I²=38%")
    md_lines.append("- Lung cavitation: AOR 1.90 (1.26-2.84), I²=63%")
    md_lines.append("- HIV seropositivity: AOR 0.45 (0.26-0.80)\n")
    md_lines.append(
        "Other lines in Table 3 cite single studies and do **not** appear in any pooled analysis:"
    )
    md_lines.append("- Sputum PCR positive: AOR 3.8 (1.1-13.6) — Lohmann 2013 single study")
    md_lines.append(
        "- **Broncho-alveolar lavage AFB positive**: AOR 1.7 (1.1-2.6) — Lohmann 2013 single study"
    )
    md_lines.append("- **Sputum culture positive**: no numerical AOR in the paper.\n")
    md_lines.append(
        "**Correction to earlier feedback:** the value 1.70 for Solid/Liquid culture was taken from "
        "the *BAL AFB positive* row of the same single study, not from a culture AOR. The paper "
        "does not report a sputum-culture AOR.\n"
    )
    md_lines.append("## Side-by-side weights (all sum to 1)\n")
    header = "| method | Cavity | AFB_Smear | TB_PCR | Solid_Culture | Liquid_Culture | AFB>=Cav | PCR>Solid | PCR>Liquid |"
    sep = "|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|"
    md_lines.append(header)
    md_lines.append(sep)
    for r in rows:
        cell = lambda b: "Y" if b else "N"  # noqa: E731
        md_lines.append(
            f"| {r['label']} | {r['Cavity']:.4f} | {r['AFB_Smear']:.4f} | {r['TB_PCR']:.4f} | "
            f"{r['Solid_Culture']:.4f} | {r['Liquid_Culture']:.4f} | "
            f"{cell(r['AFB_ge_Cavity'])} | {cell(r['PCR_gt_Solid'])} | {cell(r['PCR_gt_Liquid'])} |"
        )
    md_lines.append("")
    md_lines.append("## Score statistics on the same n=459 cohort\n")
    md_lines.append("| method | mean | std |")
    md_lines.append("|---|---:|---:|")
    for r in rows:
        md_lines.append(f"| {r['label']} | {r['score_mean']:.4f} | {r['score_std']:.4f} |")
    md_lines.append("")
    md_lines.append("## Methods notes\n")
    for r in rows:
        md_lines.append(f"- **{r['label']}**: {r['note']}")
    md_lines.append("")
    md_lines.append("## Outputs\n")
    md_lines.append(f"- `{table_csv.name}` — one row per method with weights and ordering checks")
    md_lines.append(f"- `{scores_csv.name}` — per-patient (complete-case) score under each method")
    (out / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {
        "input_csv": str(args.csv),
        "pr7_json": str(args.pr7_json),
        "n_complete_case": n_cc,
        "positive_counts": pos_counts,
        "methods": {k: ({**m, "weights": m["weights"]} if m["weights"] else m) for k, m in methods.items()},
        "ordering_check": {
            r["label"]: {
                "AFB_ge_Cavity": r["AFB_ge_Cavity"],
                "PCR_gt_Solid": r["PCR_gt_Solid"],
                "PCR_gt_Liquid": r["PCR_gt_Liquid"],
            }
            for r in rows
        },
        "notes": [
            "Method B (user-feedback Lit-AOR) and Method C2 (paper-correct culture=1.70) yield "
            "identical weights; we keep both so the misattribution is visible in the comparison.",
            "Score scale differs per method because weights sum to 1 but indicator counts active "
            "per patient vary. Use Spearman rank correlation across method columns in "
            "scores_per_method.csv to compare patient orderings (not absolute values).",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        print("\n".join(md_lines))
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n".join(md_lines) + "\n").encode("utf-8", errors="replace"))
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
