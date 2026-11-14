# Strip coordinate reconstruction

This program estimates a translation-based strip trajectory from ordered near-ground frames, refines geometry anchors, and exports source keyframes for subsequent core/seam sample construction.

```text
Ordered source frames
    -> coarse translations
    -> geometry keyframes and anchor correction
    -> detection keyframes and coordinate tables
    -> source-image export

Separate downstream stages:
    core/seam sample construction -> detection -> duplicate handling -> plot partitioning
```

The keyframe images are selected source frames at their original resolution, with the configured right-angle rotation. They are not crops of a stitched mosaic. Registration preprocessing and preview blending do not modify the images supplied for detection. PNG export is lossless; JPEG export re-encodes at the configured quality.

## Files

Keep `strip_registration.py` and `registration_config.py` together. Run only `strip_registration.py`.

| File | Purpose |
| --- | --- |
| `strip_registration.py` | Registration, keyframe selection, image export and command-line entry point |
| `registration_config.py` | Grouped default settings |
| `PARAMETERS.md` | Parameter values, coordinate conventions and fixed scoring rules |
| `requirements.txt` | Dependency specifications for installation |
| `requirements-tested.txt` | Exact numerical package versions used for the recorded checks |
| `tests/test_registration.py` | Synthetic-coordinate and execution checks |
| `VALIDATION.md` and `validation.json` | Recorded comparisons and their scope |

## Install and run

Use Python 3.11 or newer. Validation used Python 3.12.14; the source also passes the Python 3.11 syntax check. Runtime execution on Python 3.11 was not tested here. The program uses NumPy, pandas and OpenCV; it does not require a detector or GPU training framework.

```shell
python -m pip install -r requirements.txt
python strip_registration.py --input data/frames --output results/strip --rotate 180
```

For paths containing spaces, quote both paths:

```shell
python strip_registration.py --input "data/sequence 01" --output "results/sequence 01" --rotate 180
```

Alternatively, edit `INPUT_OUTPUT` in `registration_config.py` and run:

```shell
python strip_registration.py
```

Command-line options override the corresponding defaults. To skip inspection mosaics:

```shell
python strip_registration.py --input data/frames --output results/strip --no-preview
```

`--no-keyframe-export` writes coordinate tables without copying keyframe images. Leave image export enabled for the usual downstream core/seam workflow. If the requested output directory is nonempty, a new `run_YYYYMMDD_HHMMSS_microseconds` subdirectory is used. The actual destination is printed as `[OUTPUT]`.

## Input and coordinates

Provide at least two readable PNG, JPEG, BMP or TIFF images in one directory. Files are sorted by their numeric filename components. Use unique, consistently numbered frame names. Images must have the same dimensions after rotation.

The coordinate model has translation only. It does not fit an affine transform, a homography, camera calibration or a nonrigid warp. `x_global_px` and `y_global_px` locate the top-left corner of the processed source frame in the exported strip representation. `img_w` and `img_h` refer to that frame after rotation; they are not preview dimensions. Coordinates use image pixels, not metres, and are not absolute surveyed ground coordinates.

The automatic `sign_y` normalizes longitudinal motion. It does not rotate or reflect image pixels. A negative sign therefore requires review of image orientation and the preview before the tables are used as ordinary translated images downstream. A reverse-direction regression comparison establishes agreement with the supplied algorithm, not geometric validity of arbitrary reverse acquisitions.

## Outputs

| File or directory | Contents and use |
| --- | --- |
| `poses_all_coarse.csv` | Every input frame, coarse top-left trajectory, local displacement, method, response and quality scores |
| `poses_all_corrected.csv` | The same rows plus `x_corrected` and `y_corrected` after anchor interpolation |
| `geom_keyframes.csv` | Geometry keyframes and their corrected coordinates |
| `keyframes.csv` | Detection keyframes and their corrected coordinates; primary downstream coordinate table |
| `anchors_refined.csv` | Geometry-anchor pair estimates, coarse predictions, fallback reasons and accepted increments |
| `report_keyframe_gaps.csv` | Corrected geometry-keyframe y increments and estimated vertical overlap |
| `report_det_keyframe_gaps.csv` | Detection-keyframe x/y increments and estimated vertical overlap |
| `keyframes_images/` | Full-resolution detection source keyframes for the next sample-construction stage |
| `geom_keyframes_images/` | Geometry source keyframes, retained separately for inspection and reproduction |
| `mosaic_preview_*.png` | Downsampled geometry mosaics, segmented for display |
| `det_keyframes_preview/mosaic_preview_*.png` | Downsampled detection-keyframe mosaics |
| `meta.json` | Image dimensions, direction, principal settings and preview chunk extents |
| `poses_corrected_nonforward_warnings.csv` | Conditional diagnostic output when corrected y does not increase |
| `run_config.json` | Effective settings after command-line overrides |
| `run_status.json` | Run state, numerical package versions, code/configuration hashes and review reasons |
| `registration_handoff.json` | The coordinate/image handoff and its review state |

`geom_keyframes_images` and `keyframes_images` represent different selection roles. A source frame can belong to both sets. Such a duplicate file is intentional; neither directory contains a mosaic crop.

For the existing core/seam constructor, use `keyframes.csv` together with `keyframes_images/`. Their filenames and CSV columns are retained. Do not substitute a preview mosaic, a geometry-only table, or images with a different rotation. Existing fields are unchanged; the three run/handoff JSON files are additional outputs.

## Review state

`completed` means execution finished without the recorded review triggers. `completed_needs_review` still contains completed exports. It records predicted frame steps, coarse anchor fallbacks, non-forward corrected steps, overlap concerns, weak direction evidence, or insufficient detection keyframes. Inspect the per-pair tables and previews before quantitative use. A short sequence can select only one detection keyframe; this is reported and the original selection rule is retained. A detection-gap report for that case can be an empty CSV.

Predicted motion and coarse-anchor fallback are part of the supplied algorithm. They are retained and labeled rather than silently replaced by a new registration method. Neither a completion flag nor a high correlation response establishes field coordinate accuracy. The overlap target and guard are selection settings; corrected outputs should still be checked.

## Verification

```shell
python -m unittest discover -s tests -v
```

`VALIDATION.md` records the baseline comparison and numerical tolerances. The real-data comparison uses 20 frames, not the full long sequence. No detection, seam/core construction, deduplication or plot-partitioning accuracy claim is made by these checks.
