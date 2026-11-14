# Core and seam detection fusion

These programs construct core/seam samples from registered source keyframes, then restore detector outputs to global coordinates and suppress repeated observations using ownership intervals.

```text
Registered source keyframes and coordinates
    -> build_core_seam_dataset.py
    -> core_crops and seam_crops
    -> run a detector on each sample set
    -> merge_core_seam_results.py
    -> retained global detections
    -> separate plot partitioning
```

The detector is a separate program. Run the merger after both detection CSV files have been produced. The merger does not run inference, estimate plot boundaries, calculate phenotypes, or render a full-strip image.

## Package files

| File | Role |
| --- | --- |
| `build_core_seam_dataset.py` | Construct source-based core crops and pairwise seam crops |
| `merge_core_seam_results.py` | Restore local boxes, resolve ownership and clean residual duplicates |
| `core_seam_config.py` | Grouped defaults for both stages |
| `core_seam_io.py` | Shared image I/O, input checks and run records |
| `PARAMETERS.md` | Units, thresholds and decision rules |
| `requirements.txt` | Installation dependencies |
| `requirements-tested.txt` | Exact numerical package versions used in verification |
| `tests/test_core_seam.py` | Synthetic-coordinate and edge-case checks |
| `VALIDATION.md` and `validation.json` | Verification scope and recorded results |

Keep the four Python files together. Run the construction and merger scripts; the configuration and I/O modules are imported automatically.

## Installation

```shell
python -m pip install -r requirements.txt
```

Execution was tested with Python 3.12.14. Source syntax was also checked against Python 3.11; Python 3.11 execution was not tested here. The programs use NumPy, pandas and OpenCV. They do not require detector weights or a GPU training framework.

## Construct samples

```shell
python build_core_seam_dataset.py --keyframes results/strip/keyframes.csv --output results/core_seam
```

The default image directory is `keyframes_images` beside the keyframe table. Supply `--images` for another location. Paths containing spaces should be quoted. The keyframe table should contain `k`, `file`, `x_global_px`, `y_global_px`, `img_w` and `img_h`. Existing supported column aliases and positional fallbacks are retained, but explicit column names are recommended.

Coordinates describe the top-left corners of equal-sized, consistently oriented source images under a translation-only model. The source images and the table must use the same rotation. Images transformed onto unequal affine canvases are not compatible with this constructor.

Core samples are crops from individual source frames, with contextual extension and optional background fill. Seam samples combine patches from the two adjacent source frames with the specified weights. A full-strip panorama is not constructed and cropped to obtain the samples. Sample PNG files are saved at their constructed resolution without downsampling.

Construction outputs:

| Output | Contents |
| --- | --- |
| `core_crops/` | Contextual single-source samples |
| `seam_crops/` | Pairwise seam samples |
| `core_crops_meta.csv` | Source frame, crop origin, size, ownership and context intervals |
| `seam_crops_meta.csv` | Source pair, canvas origin, size, ownership and context intervals |
| `strip_meta.json` | Origin, strip axes, nominal band/context settings and image dimensions |
| `run_config.json` | Effective construction settings |
| `run_status.json` | Execution state, file hashes, numerical environment and sample counts |

Ownership bounds at strip ends can be infinite. This represents the absence of a neighboring separator and is distinct from an invalid crop origin.

## Detect seedlings

Run the detector separately on `core_crops` and `seam_crops`. Each CSV should contain:

```text
file,x1,y1,x2,y2,score
```

The filename must match `crop_file` in the corresponding metadata table. Coordinates are pixel positions in that crop at its saved resolution. Undo detector resizing or padding before exporting these coordinates. Additional detector columns are ignored. Use a header-only table when one sample set contains no detections; a zero-byte file is not a valid detection table.

## Restore and merge detections

```shell
python merge_core_seam_results.py --keyframes results/strip/keyframes.csv --dataset results/core_seam --core-detections results/core_detections/detections.csv --seam-detections results/seam_detections/detections.csv --output results/merged
```

`--dataset` selects the three metadata files produced by construction. Explicit `--core-meta`, `--seam-meta` and `--strip-meta` arguments support metadata files with other names. Command-line arguments override `core_seam_config.py`. Both scripts can also be run without arguments after editing their grouped path settings.

Fusion outputs:

| Output | Contents |
| --- | --- |
| `dedup_keep_global_core_seam_safe_light_clean.csv` | Retained global detections for plot partitioning |
| `dedup_all_with_flags.csv` | Ownership-retained detections, including residual-cleaning flags |
| `merge_decisions.csv` | Eligible core/seam observations, row hints and ownership decisions |
| `duplicate_pairs.csv` | Pairs removed by the final within-source cleaning stage |
| `run_config.json` | Effective fusion settings |
| `run_status.json` | Execution state, hashes, numerical environment and retained/removed counts |

The retained table preserves `anchor_x_global` and `anchor_y_global`, box coordinates, source information and existing decision fields. Its filename matches the partition program's `dedup_keep_global_*_safe_light_clean.csv` batch pattern. A single table can also be passed directly with the partition program's `--input` option.

The all-flags table starts after ownership filtering; it is not an archive of every raw detector box. The decision table covers ownership-eligible candidates. Context-only and score-filtered detections need to be inspected in the raw detector output if required. `duplicate_pairs.csv` can be empty when no pair is removed.

## Ownership and duplicate handling

Only detections whose centers lie inside their source ownership interval enter fusion. Context supplies detector input but does not grant ownership. Eligible seam detections are retained first. An eligible core detection outside every seam interval remains. Inside a seam interval, it is suppressed only if an eligible seam detection in the same coarse row meets both coordinate-distance thresholds; otherwise it remains as fallback. Matching tests for the existence of a qualifying seam observation and does not require a one-to-one assignment.

The final cleaning step acts only within the same `owner_frame` group: one source core frame or one source seam pair. A pair must meet all configured box-overlap, area, center-distance, strip-distance and coarse-row checks. This is not unrestricted global NMS. Coarse row groups support matching; they are not the final within-plot crop-row assignments.

Thresholds and decision rules are listed in `PARAMETERS.md`. The input score floor of 0.05 is a fusion-stage filter, not a statement that the detector was evaluated at confidence 0.05.

## Execution records and checks

Unreadable images, inconsistent dimensions, duplicate crop metadata and unmatched detection filenames raise explicit errors. A nonempty destination receives a new timestamped run subdirectory; use the printed `[OUTPUT]` path for the next stage. A seam-only detection input is supported. If no eligible detection remains from either source, fusion records a failed run and raises an explicit error.

```shell
python -m unittest discover -s tests -v
```

Verification records describe numerical comparisons and software checks. They do not establish seedling recall, duplicate-removal accuracy, or surveyed registration accuracy. Input registration, detector performance and ownership coverage remain separate sources of error.
