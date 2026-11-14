# Parameters

Edit `core_seam_config.py`. Construction and fusion have separate path settings. Distances use the common source-image coordinate scale, not metres.

## Build Paths

| Setting | Default | Meaning |
| --- | --- | --- |
| `KEYFRAME_CSV` | `'results/strip/keyframes.csv'` | Registered source-keyframe table. In fusion it is retained as an input record; coordinate restoration uses the crop metadata and strip axes. |
| `KEYFRAME_IMG_DIR` | `None` | Source image directory; None locates keyframes_images beside the keyframe CSV. |
| `OUT_DIR` | `'results/core_seam'` | Requested destination; nonempty directories receive a new run subdirectory. |

## Crop Geometry

| Setting | Default | Meaning |
| --- | --- | --- |
| `STRIP_AXIS_HINT` | `'y'` | Orient the estimated strip direction toward positive global y or x. |
| `SEAM_HALF_BAND` | `56.0` | Nominal seam ownership half-width in along-strip pixel units. |
| `CORE_CONTEXT` | `96.0` | Contextual extension beyond each finite core ownership edge, in along-strip pixel units. |
| `SEAM_CONTEXT` | `96.0` | Contextual extension beyond each seam ownership edge, in along-strip pixel units. |
| `SEAM_FEATHER` | `18.0` | Transition scale for pairwise seam weights, in along-strip pixel units. |

## Crop Background

| Setting | Default | Meaning |
| --- | --- | --- |
| `FILL_OUTSIDE` | `True` | Fill pixels outside the contextual mask within its bounding rectangle. |
| `FILL_VALUE` | `(0, 0, 0)` | BGR background value used for filled pixels. |

## Merge Paths

| Setting | Default | Meaning |
| --- | --- | --- |
| `KEYFRAME_CSV` | `'results/strip/keyframes.csv'` | Registered source-keyframe table. In fusion it is retained as an input record; coordinate restoration uses the crop metadata and strip axes. |
| `CORE_META_CSV` | `'results/core_seam/core_crops_meta.csv'` | Core crop metadata table. |
| `SEAM_META_CSV` | `'results/core_seam/seam_crops_meta.csv'` | Seam crop metadata table. |
| `STRIP_META_JSON` | `'results/core_seam/strip_meta.json'` | Common strip origin and axes from construction. |
| `CORE_DET_CSV` | `'results/core_detections/detections.csv'` | Detector output for core crops. |
| `SEAM_DET_CSV` | `'results/seam_detections/detections.csv'` | Detector output for seam crops. |
| `OUT_DIR` | `'results/merged'` | Requested destination; nonempty directories receive a new run subdirectory. |

## Ownership Fusion

| Setting | Default | Meaning |
| --- | --- | --- |
| `MIN_SCORE` | `0.05` | Inclusive confidence floor when loading detections for fusion. |
| `EXPECTED_ROWS_PER_STRIP` | `3` | Number of coarse transverse row groups; this is not a plot count or final row fit. |
| `FALLBACK_MATCH_S` | `36.0` | Inclusive along-strip distance threshold for a core-to-seam match, in pixels. |
| `FALLBACK_MATCH_T` | `20.0` | Inclusive transverse distance threshold for a core-to-seam match, in pixels. |

## Residual Cleaning

| Setting | Default | Meaning |
| --- | --- | --- |
| `DO_FINAL_LIGHT_CLEAN` | `True` | Enable conservative residual cleaning within each owner_frame group. |
| `FINAL_IOF_MIN` | `0.92` | Minimum intersection divided by the smaller box area. |
| `FINAL_IOU_MIN` | `0.55` | Minimum intersection-over-union. |
| `FINAL_CENTER_DIST_MAX` | `8.0` | Maximum Euclidean center distance in global pixels. |
| `FINAL_AREA_RATIO_MIN` | `0.45` | Minimum smaller-to-larger box-area ratio. |
| `FINAL_S_DIST_MAX` | `12.0` | Maximum along-strip center separation in pixels. |
| `FINAL_T_DIST_MAX` | `8.0` | Maximum transverse center separation in pixels. |
| `FINAL_REQUIRE_SAME_ROW` | `True` | Require equal coarse row labels in residual cleaning. |

## Construction rules

- The strip origin is the median source-frame center. The leading eigenvector of their covariance defines the along-strip axis. Vectors are normalized with epsilon 10⁻¹²; the transverse axis is its perpendicular.
- Adjacent frame-center midpoints define seam centers. An interior core half-band is capped at 0.45 times the distance between its neighboring seam centers. A seam half-band is capped at 0.45 times max(adjacent center distance, 1). These two caps are evaluated separately.
- Ownership and context masks include both interval endpoints. The contextual mask is cropped to its nonempty pixel bounding rectangle. Source crop origins are retained for coordinate restoration.
- For a pair A/B, let L = max(abs(sB − sA), 10⁻⁶). Before feathering, weights are clipped values of (sB − s)/L and (s − sA)/L. Feathering applies clip((w − 0.5) × L/max(feather, 10⁻⁶) + 0.5, 0, 1). Weights are multiplied by source validity masks, summed on the canvas and normalized where total weight >10⁻⁶. The canvas is zero elsewhere.
- Canvas positions use rounded source offsets; canvas bounds use floor/ceil. Samples are PNG-encoded with compression level 3.

## Fusion and cleaning rules

Local box coordinates are translated by their recorded crop origin. The box center is the anchor for ownership, strip projection and matching. Coarse rows use one-dimensional k-means initialized at quantiles j/(R+1), with at most 100 iterations and convergence tolerance 10⁻⁶. Row centers are sorted transversely. With at most R observations, sorted observed positions provide centers.

Core fallback is restricted to detections inside their own core ownership interval. It does not recover every detection from the context halo. All eligible seams are kept before residual cleaning. One eligible seam may satisfy the matching test for more than one core observation.

For residual cleaning, all six configured geometric thresholds and the enabled same-row condition must pass. Areas are floored at 10⁻⁶; 10⁻⁶ is added to the IoU denominator. Eligible pairs rank by:

```text
2.0 × IoU + 1.0 × IoF + 0.5 × area_ratio − 0.02 × center_distance
```

Tied pair scores retain enumeration order. Pairs whose detections have already been removed are skipped. Keep the higher-confidence observation; confidence values within 10⁻⁹ tie, then the larger box is kept. If confidence and area both tie, keep the earlier observation.

All distances and ownership tests apply to the coordinates after restoration. Detector boxes must already be expressed at crop resolution. These settings do not validate the upstream registration or identify false detector predictions.
