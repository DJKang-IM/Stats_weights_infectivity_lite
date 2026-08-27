# Stats_weights_infectivity_lite

Full corpus available under NDA for research collaboration / employment. Contact: drkangim@naver.com


**Public lite release** of TB CAD / infectivity research code.

| | |
|--|--|
| Code | Full pipeline source (sanitized paths; no PHI) |
| JSON samples | **3** synthetic (`data/samples/`) |
| Preview images | **10** random 256px coarse (`data/coarse_256/`) |
| Full data / DICOM | **Not included** — gated: [DJKang-IM/Stats_weights_infectivity](https://github.com/DJKang-IM/Stats_weights_infectivity) |

See [`SERVING_RULES.md`](SERVING_RULES.md). Copied code files: ~102.

---

**CC BY-NC-SA 4.0, Non-commercial, Citation required. Full version gated — request access via GitHub Issues.**

**This is a coarse preview. For research collaboration contact the lite repo Issues page.**

---

# Stats_weights_infectivity

분석 결과는 아래 링크에서 확인하실 수 있습니다.

- [점수 및 기여도 결과 원본 데이터 보기 (CSV)](artifacts/infectivity_sensitivity_formulas_weighted_scores_260510/scores_with_contributions.csv)

- [이중 변환 데이터 보기(invlog1p)](artifacts/infectivity_empirical_constrained_sem_260510_invlog1p/report.txt)

Single-hospital **TB infectivity** meta-analysis: empirical sensitivity ratios, **constrained CFA/SEM** (semopy), **relative influence** weights, **2PL IRT** (Python `girth`), bootstrap stability, and formula-based weighted scores.

Split from [TBC_Phase_III](http<REDACTED_PATH> — Phase III imaging/CAD training lives there; **statistical weighting and latent infectivity** live here.

## Indicators (D1–D5)

| Code | Column (KR) | Role |
|------|-------------|------|
| AFB_Smear | 도말검사 | Smear |
| TB_PCR | TB-PCR검사 | PCR |
| Solid_Culture | 배양검사(고체) | Solid culture |
| Liquid_Culture | 배양검사(액체) | **Proxy gold** for empirical ratios |
| Cavity | Cavity 유무 | X-ray cavity (reading keyword rules) |

All scripts use **0/1** (or missing) on complete-case or imputed cohorts as documented per run.

## Pipeline scripts

| Script | Purpose |
|--------|---------|
| `meta_infectivity_empirical_constrained_sem.py` | Positive counts → empirical X_AF, Y_SO, Z_PC, Z_CV; fixed-loadings constrained CFA (`linear` / `log1p` / `invlog1p`) |
| `meta_infectivity_relative_influence_and_2pl.py` | Relative influence table + **girth** 1D 2PL (`twopl_mml`) |
| `meta_infectivity_exclude_impute_bootstrap_sem.py` | Exclude cohort + balanced impute + bootstrap CFA |
| `meta_infectivity_sensitivity_formulas_and_weighted_scores.py` | Sensitivity formulas + PCA-weighted infectivity score |
| `analyze_infectivity_latent.py` | PCA / free CFA / PLS-style weights (NPZ or CSV) |
| `compare_clinical_stat_infectivity_scores.py` | Clinical [3,2,2,1,1] vs CFA relative-influence scores |
| `apply_meta_exclusion_lab_clear.py` | Clear labs on Exclude-pattern readings (260510) |
| `verify_exclude_cohort_cavity_sem.py` | Verify Exclude rows vs SEM complete-case cohort |

## Setup

```bash
cd D:\Stats_weights_infectivity
pip install -r requirements.txt
```

Copy meta CSVs into `data/` (see [data/README.md](data/README.md)).

## Example runs

```bash
# Constrained SEM (260510 cohort, linear loadings)
py meta_infectivity_empirical_constrained_sem.py ^
  --csv data\260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv ^
  --out artifacts\infectivity_empirical_constrained_sem_260510

# log1p / invlog1p fixed loadings
py meta_infectivity_empirical_constrained_sem.py --fixed-loading-scale log1p --out artifacts\infectivity_empirical_constrained_sem_260510_log1p

# Relative influence + 2PL IRT
py meta_infectivity_relative_influence_and_2pl.py

# Bootstrap + imputation
py meta_infectivity_exclude_impute_bootstrap_sem.py --bootstrap 200
```

## Artifacts (committed)

Precomputed outputs under `artifacts/` (260509 / 260510 runs): `summary.json`, `report.txt`, `empirical_sensitivity.json`, `twopl_girth_mml.json`, TSV tables, etc.

Key result folders:

- `infectivity_empirical_constrained_sem_260510` — linear fixed loadings
- `infectivity_empirical_constrained_sem_260510_log1p` / `_invlog1p` — scale variants
- `infectivity_relative_influence_2pl_260510` — CFA weights + **girth** 2PL
- `infectivity_exclude_impute_bootstrap_sem_260510`
- `infectivity_sensitivity_formulas_weighted_scores_260510`

## Notes

- **IRT:** Python-only via `girth.twopl_mml` (1D binary 2PL). Optional R `mirt` script is generated only with `--write-r-script`.
- **D6 NTM** is not part of this infectivity five-indicator model.
- Imaging Phase III: see **TBC_Phase_III** repo.
