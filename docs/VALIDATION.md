# Validation record

| Scope | Evidence | Interpretation |
| --- | --- | --- |
| Original delivered notebook | Delivery notes report 45/45 code cells, 394 scientific quantities, and source/figure checks | Historical local verification, retained in archive/delivery/ |
| Kaggle Version 5 | Successful Save & Run All, 301.8 seconds | Directly observed in this chat |
| Final Kaggle output gate | “Aggregate-only output check passed; source and figures retained.” | Directly observed in run logs |
| Public rendering | Four paper PNGs loaded; paper and mathematics readable | Directly observed on the public notebook |
| Repository integrity | SHA-256 manifest and verify_repository.py | Checks packaging and preserved code; does not rerun models |

## Expected original frozen result

```text
Players:                  19
Charted observations:     7,801
Pressures:                668
Constant Brier:           0.08351348557970897
Conventional Brier:       0.08368440178100557
Mirror Brier:             0.08458661787147928
```

Small final-digit floating-point differences can depend on numerical libraries. The notebook uses explicit numerical assertions. The unfavorable mirror finding must not be replaced by a result selected from the exploratory extensions.

## Packaging checks

The repository checker confirms notebook structure, all 45 code cells’ identity between the portable and Kaggle-specific copies, the eight original code-cell hashes, the four embedded PNGs, and all manifest-listed file hashes. It runs without competition data.

CI runs this integrity checker on pushes and pull requests. CI does not access the restricted competition files and does not claim to reproduce the scientific experiments. The full research execution remains a separate, data-dependent step.
