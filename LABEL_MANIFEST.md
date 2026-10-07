# BSA identity-validated label manifest (version 2)

The Label Creator and BSA analyser must exchange labels through
`label_manifest.py`. The module consumes only acquisitions already validated by
the shared implementation behind `hyperspectral_dataset.discover_dataset` and
`hyperspectral_dataset.discover_acquisitions`; it does not parse filenames
itself. The analyser uses the former, mask-required entry point. Label Creator
uses the latter, metadata-only entry point with optional mask validation.

## Identity rules

- A BIL sample uses Phase 6A's format-qualified, case-insensitive key
  `bil:<filename prefix>-<sensor>`. `VNIR` and `SWIR` are therefore distinct
  acquisitions even when the filename prefix is the same.
- A RAW sample uses Phase 6A's `raw:vnir_<n>_<n>_<n>` key. The existing rule
  that generic RAW references are valid for only one data cube remains in
  force. An ambiguous RAW folder never produces a manifestable acquisition.
- `acquisition_id` is `bsa-acq-v2:` plus a digest over metadata only: the
  validated identity key, format, sensor, data/dark/white filenames and
  data-file size reported by the directory entry. Mask existence, filename and
  location are deliberately excluded. No BIL or RAW file is opened, decoded,
  loaded or content-hashed. The ID is stable when the dataset tree is moved to
  a different root and before or after segmentation masks are generated.
- `dataset_id` is `bsa-dataset-v2:` plus an order-independent SHA-256 digest of
  all acquisition IDs. It binds a manifest to the complete selected dataset.
- The dark, data and white selected-directory-relative names are recorded as
  provenance and must exactly match the validated association. Discovery
  currently uses immediate directory entries, so these relative names are
  basenames. Original acquisition files are never opened, read, renamed or
  modified; only directory metadata is inspected.
- `discover_acquisitions` permits no mask folder, an empty mask folder or
  incomplete mask coverage. If a mask folder is supplied, every mask that is
  present must still be recognisable, match an acquisition uniquely and obey
  the existing duplicate-mask safeguards. `discover_dataset` remains strict
  and requires exactly one valid mask for every acquisition used by the main
  analyser.

## CSV schema

The UTF-8 CSV header is exact and versioned:

```text
manifest_version,dataset_id,acquisition_id,sample_id,identity_key,sensor,file_format,data_file,dark_file,white_file,data_size_bytes,label_status,class
```

There is exactly one row for every validated acquisition; row order has no
meaning.

| Column | Meaning |
| --- | --- |
| `manifest_version` | The literal `2`. |
| `dataset_id` | Complete selected-dataset identity, repeated on every row. |
| `acquisition_id` | Stable metadata-only acquisition identity. |
| `sample_id` | Phase 6A's human-readable sample identity. |
| `identity_key` | Phase 6A's canonical format-qualified identity key. |
| `sensor` | `VNIR` or `SWIR`, obtained from the validated association. |
| `file_format` | `BIL` or `RAW`, obtained from the validated association. |
| `data_file`, `dark_file`, `white_file` | Validated associated basenames. |
| `data_size_bytes` | Data-file size obtained from directory metadata without opening the cube. |
| `label_status` | `labelled` or `unlabelled`. |
| `class` | Any researcher-defined, non-blank text for `labelled`; empty for `unlabelled`. |

An empty class is invalid when status is `labelled`. An unlabelled acquisition
must be represented explicitly with status `unlabelled` and an empty class.
No biological class is derived from filenames or other metadata.

Labels.csv is a machine-readable interchange file. Its identity, filename and
class text is preserved exactly, including formula-like text. Do not open an
untrusted Labels.csv directly in spreadsheet software; use Label Creator's
**Export Spreadsheet Review (.xlsx)** option to inspect it. The review workbook
stores every manifest cell as literal text and is not an analysis input.

The importer accepts UTF-8 with or without a UTF-8 BOM and rejects invalid
byte sequences. It streams the CSV with limits of 64 MiB per file, 50,000 data
rows, exactly 13 schema columns, 32,767 characters per field, and 1 MiB per
physical CSV line. These limits comfortably exceed 1,000-acquisition datasets;
the field limit also matches Excel's maximum exact cell text length. A global
Python CSV parser field limit is set for each manifest read and restored after
it. Regular-file symlinks remain supported; directories, pipes, sockets and
devices are rejected before CSV parsing.

## Shared integration contract

Both applications use these functions:

1. `dataset_provenance(dataset)` exposes acquisition IDs for display/editing.
2. `create_label_manifest(dataset, assignments)` builds all rows. Assignment
   keys are acquisition IDs and values are a user class or `None`.
3. `save_label_manifest(path, manifest)` stages a CSV and publishes it without replacing an existing file.
4. `load_and_validate_label_manifest(path, dataset)` returns
   `ValidatedLabelAssignments` only after identity and provenance checks.

The validated result maps both acquisition IDs and Phase 6A identity keys to
classes. It omits explicitly unlabelled acquisitions from those class maps and
reports their number through `excluded_unlabelled_count`. Ordinary per-sample
analysis can still process every `DatasetSetPaths`; future class-based analysis
must use only the class maps and report that exclusion count.

Validation rejects malformed records, unsupported schemas, missing identity or
provenance fields, blank labelled classes, duplicate identities, conflicting
assignments, unknown rows, missing selected-dataset rows, provenance changes,
dataset mismatch and ambiguous selected acquisitions. Existing one-column
positional CSV files fail the manifest-header check and are never silently
reinterpreted.

Version 1 manifests are rejected explicitly rather than reinterpreted. Their
acquisition and dataset IDs incorporated `mask_file`, so they cannot provide
the required pre-/post-segmentation identity stability. Researchers must reopen
the unchanged acquisition dataset and export a version 2 manifest. The version
1 header and IDs are not accepted as version 2 data.

Because content inspection is intentionally forbidden, two independent dataset
trees with identical canonical identities, associated filenames and data-file
sizes are indistinguishable to the metadata-only scheme. They cannot coexist in
one selected dataset because duplicate canonical identities are rejected. If
such trees represent different experiments, researchers must keep separate
manifests and confirm the selected dataset; content hashing is not used as a
fallback.

## Application integration

The standalone Label Creator calls the shared optional-mask acquisition,
identity and manifest functions directly. Experimental labels can be created
and exported before BSA Segmentor produces any masks; that same manifest
validates after matching masks are added.

The analyser retains its strict mask-required discovery. Its existing label
CSV selector accepts structurally valid version 2 manifests, and the selected
manifest is validated against the discovered dataset before any cube is
loaded. Each processing iteration calls
`ValidatedLabelAssignments.class_for(set_paths)`; CSV row order is never used
for identity manifests. Every acquisition still receives ordinary per-sample
analysis, while `None` assignments are excluded from class means, class plots
and class exports. The completion status reports
`excluded_unlabelled_count`.

Legacy one-column positional CSVs are explicitly rejected. Supporting them
safely would require a separate mapping-review and confirmation workflow; they
are not silently mapped by discovery order.
