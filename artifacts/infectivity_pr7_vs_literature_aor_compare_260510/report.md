# PR#7 vs Literature-AOR weighting — comparison

- Cohort (complete-case): **n = 459** (CSV: `D:\260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv`)
- Positive counts in COLS order ['Cavity', 'AFB_Smear', 'TB_PCR', 'Solid_Culture', 'Liquid_Culture']: {'Cavity': 146, 'AFB_Smear': 202, 'TB_PCR': 254, 'Solid_Culture': 258, 'Liquid_Culture': 325}

## Paper of record (AORs)

Asadi et al., *Risk factors for infectiousness of patients with tuberculosis: a systematic review and meta-analysis*, Epidemiology and Infection 2022 (PMC9134570).

Three exposures had **pooled meta-AOR**: sputum smear, lung cavitation, HIV.
- Sputum smear positive: AOR 2.15 (1.47-3.17), I²=38%
- Lung cavitation: AOR 1.90 (1.26-2.84), I²=63%
- HIV seropositivity: AOR 0.45 (0.26-0.80)

Other lines in Table 3 cite single studies and do **not** appear in any pooled analysis:
- Sputum PCR positive: AOR 3.8 (1.1-13.6) — Lohmann 2013 single study
- **Broncho-alveolar lavage AFB positive**: AOR 1.7 (1.1-2.6) — Lohmann 2013 single study
- **Sputum culture positive**: no numerical AOR in the paper.

**Correction to earlier feedback:** the value 1.70 for Solid/Liquid culture was taken from the *BAL AFB positive* row of the same single study, not from a culture AOR. The paper does not report a sputum-culture AOR.

## Side-by-side weights (all sum to 1)

| method | Cavity | AFB_Smear | TB_PCR | Solid_Culture | Liquid_Culture | AFB>=Cav | PCR>Solid | PCR>Liquid |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| PR#7 Entropy Effective Rank | 0.3249 | 0.1742 | 0.1544 | 0.1810 | 0.1654 | N | N | N |
| Lit-AOR log (feedback) | 0.1687 | 0.2012 | 0.3510 | 0.1395 | 0.1395 | Y | Y | Y |
| Paper-correct, culture AOR=1.00 (null) | 0.2341 | 0.2791 | 0.4868 | 0.0000 | 0.0000 | Y | Y | Y |
| Paper-correct, culture AOR=1.70 (=BAL proxy) | 0.1687 | 0.2012 | 0.3510 | 0.1395 | 0.1395 | Y | Y | Y |
| Paper-correct, culture AOR=2.00 | 0.1555 | 0.1854 | 0.3234 | 0.1679 | 0.1679 | Y | Y | Y |
| C-pr7split-cult2.00 | 0.1868 | 0.2228 | 0.3886 | 0.1054 | 0.0963 | Y | Y | Y |

## Score statistics on the same n=459 cohort

| method | mean | std |
|---|---:|---:|
| PR#7 Entropy Effective Rank | 0.4844 | 0.3284 |
| Lit-AOR log (feedback) | 0.5137 | 0.3503 |
| Paper-correct, culture AOR=1.00 (null) | 0.4667 | 0.3802 |
| Paper-correct, culture AOR=1.70 (=BAL proxy) | 0.5137 | 0.3503 |
| Paper-correct, culture AOR=2.00 | 0.5232 | 0.3484 |
| C-pr7split-cult2.00 | 0.5000 | 0.3557 |

## Methods notes

- **PR#7 Entropy Effective Rank**: Data-driven (polychoric + PLS within micro); no AOR.
- **Lit-AOR log (feedback)**: User's earlier feedback. CULTURE 1.70 is actually the BAL AFB AOR from Lohmann; sputum culture has no AOR in Asadi paper. Kept for direct comparison.
- **Paper-correct, culture AOR=1.00 (null)**: Culture treated as no effect (null) since paper has no number.
- **Paper-correct, culture AOR=1.70 (=BAL proxy)**: Same number as Method B but explicit that 1.70 came from BAL, not sputum culture.
- **Paper-correct, culture AOR=2.00**: Mid-range: bacteriologic confirmation typically >= 2x; sensitivity check.
- **C-pr7split-cult2.00**: Paper-faithful AOR for smear/cavity/PCR; culture AOR=2.00; Solid vs Liquid split by PR#7 PLS within-microbiology ratio.

## Outputs

- `weights_comparison.csv` — one row per method with weights and ordering checks
- `scores_per_method.csv` — per-patient (complete-case) score under each method
