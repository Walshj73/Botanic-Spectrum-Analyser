# Published numerical results

This directory contains 120 byte-identical numerical-result files retained from
the BSA manuscript experiments. They are organised by dataset:

| Dataset | Files |
|---|---:|
| Barley | 47 |
| Barley-SWIR | 25 |
| Dicot | 24 |
| Wheat | 24 |

Within each dataset, `files/` contains per-fold BSA CSV/configuration records and
`results/` contains aggregate metrics, model tables, histories, statistical
comparisons, and available cost measurements. The nested names preserve the
original context.

These values are published inspection records. They are not expected to be
regenerated automatically, and the benchmark workflow never writes here. New
experiment artifacts go into ignored, uniquely named directories under
`runs/`.

The retained `slic_superpixel_rf.json` and SLIC-RF cost records are historical
results. The original SLIC-RF implementation is missing; the workbench's new
`slic_rf_v1` baseline did not produce those values.
