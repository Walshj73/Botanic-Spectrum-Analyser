# Data layout

Raw image data is intentionally excluded from Git.

```text
data/raw/
  barley/{images,masks}/
  barley_swir/{images,masks}/
  dicot/{images,masks}/
  wheat/{images,masks_binary,masks_source_multilevel}/
```

The original filenames and file formats are unchanged. Wheat's
`masks_source_multilevel` directory is retained separately while its provenance
is investigated; active experiments use `masks_binary` as ground truth.

Future generated datasets belong under `data/interim/` or `data/processed/`,
which are also excluded from Git. Reproducible split manifests belong in each
experiment directory rather than in the raw-data tree.
