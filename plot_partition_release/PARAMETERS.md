# Parameters

Edit `partition_config.py`. Distance and window settings use reconstructed input-coordinate units. They are not lengths in metres unless the coordinates have been independently calibrated.

## Partition settings

The grouped partition settings retain the referenced values from the supplied English partition release. `MAIN_ROWS` is the number of coarse crop-row groups used for partition evidence; it is not a target plot count. `BAND_WIDTHS` contains gap-scanning windows rather than physical plot lengths.

Parameters that belonged only to the removed final-row, point-filtering and export routines are omitted. The isolated-point weights used to score gap evidence remain in `GAP_SCAN`; they do not apply a plot-level point-quality filter.

## Partition edges

| Setting | Default | Meaning |
| --- | --- | --- |
| `ENABLE_PARTITION_EDGE_REFINEMENT` | `True` | Refine edges before partition tables are exported |
| `EDGE_UNCERTAIN_POLICY` | `"visible_extent"` | Shrink unresolved edges toward the visible points; `"retain_initial"` keeps conservative limits |
| `EDGE_LOCAL_INTERVALS` | `20` | Maximum local spacing intervals at each coarse-row end |
| `EDGE_SUPPORT_BAND` | `2.0` | Support-band multiplier for the median row-specific 90th-percentile spacing |
| `EDGE_ISOLATED_STEP` | `3.0` | Maximum outermost spacing multiple for edge support |
| `EDGE_SPACING_MARGIN` | `0.5` | Margin beyond the supported point extent, in typical local spacings |
| `EDGE_MIN_SUPPORT_ROWS` | `2` | Required supporting coarse rows |
| `EDGE_MIN_REFERENCE_GAPS` | `3` | Other supported same-strip gaps needed for one-sided inference |

Rows need four positive local intervals to provide support. The target junction is excluded from its reference sample. A sufficiently supported side can move independently when its candidate does not cross the neighboring interval. No final row labels or later filtered-point tables are consulted.

With `"visible_extent"`, unresolved sides shrink inward to the point-supported candidate. The operation preserves all assigned points, does not invent new ownership and cannot increase overlap. The result retains a review flag. A wide blank shared by multiple rows can be reduced under this policy, which is an intentional visible-extent convention. With `"retain_initial"`, the conservative uncertainty treatment remains available.

The spacing margin is a processing convention. It is not a physical sowing-margin calibration. At the outer ends of a sequence, the absence of a neighboring plot is explicitly marked for review.
