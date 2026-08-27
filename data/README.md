# Meta CSV (local, not in git)

Place infectivity meta spreadsheets here (paths match script defaults):

| File | Used by |
|------|---------|
| `260509_META_Infectivity_weight-rule_CSV_Edited.csv` | `meta_infectivity_empirical_constrained_sem.py` (260509 run) |
| `260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv` | 260510 SEM / IRT / bootstrap / exclusion scripts |

Typical source on this machine:

- `D:\260509_META_Infectivity_weight-rule_CSV_Edited.csv`
- `D:\260510_META_Infectivity_weight-rule_CSV_exclusion_delete.csv`

Columns (Korean): `도말검사`, `TB-PCR검사`, `배양검사(고체)`, `배양검사(액체)`, `Cavity 유무` (+ `Reading` for exclusion scripts).
