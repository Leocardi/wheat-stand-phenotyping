# Plot partitioning with edge refinement

This program selects plot separators, assigns seedling detections to plots, and refines the plot edges before exporting partition results.

```text
Registered detections
    -> separator selection and initial assignment
    -> partition edge refinement
    -> boundary table, point groups and figures

Separate downstream processing:
    final crop-row estimation -> point-quality filtering -> phenotype analysis
```

The partition algorithm uses coarse `row_main` groups to assess multirow gaps and edge support. These are provisional hints for partitioning, not the final within-plot rows. The program does not fit final plot rows, apply a plot-level point-quality filter, or calculate phenotypes.

## Files and execution

Keep `plot_partition.py`, `partition_config.py` and `partition_edges.py` in the same directory. Run only the main script:

```shell
python plot_partition.py --input data/detections.csv --output results/sequence_a
```

Batch processing:

```shell
python plot_partition.py --batch-root data --pattern detections.csv --output results
```

The default batch pattern is `dedup_keep_global_*_safe_light_clean.csv`. A nonempty output directory receives a new timestamped run directory.

To export the initial partition edges as a control:

```shell
python plot_partition.py --input data/detections.csv --output results/control --no-edge-refinement
```

Dependencies are recorded in `requirements.txt`. The code was exercised with Python 3.12.14 and checked against the Python 3.11 grammar.

## Input

Provide one deduplicated CSV per registered image sequence with one of these column sets:

| Representation | Columns |
| --- | --- |
| Global points | `anchor_x_global`, `anchor_y_global` |
| Global boxes | `x1g`, `y1g`, `x2g`, `y2g` |
| Local boxes and crop origin | `x1l`, `y1l`, `x2l`, `y2l`, `crop_global_x1`, `crop_global_y1` |

The anchor of a bounding box is its center. All detections must use a common registered coordinate system.

## Edge refinement

The existing partition algorithm first supplies initial plot intervals and point ownership. Refinement then uses those assigned points and their coarse row hints, without relying on any later point filtering or final row labels. Local spacing and agreement between row ends support an edge estimate. Other supported gaps in the same strip may provide a reference for a weak side; the target gap is excluded from its own reference sample.

Plot IDs and the existing point-ownership decisions are preserved. Assigned partition points must remain inside their plot intervals, and adjacent intervals must not overlap. Conflicting ownership raises an error instead of deleting points.

A supported edge can update independently when the opposite edge lacks support. For unresolved edges, the default `EDGE_UNCERTAIN_POLICY = "visible_extent"` shrinks the interval inward toward its assigned visible points and records a review flag. This convention can exclude terminal space in which all rows failed to emerge. It describes a visible-detection extent, not a calibrated sowing boundary. Set the policy to `"retain_initial"` to retain uncertain initial limits while still allowing independently supported edges to update.

There is no target plot count or physical plot length input. The partitioning method still uses configured crop-row counts and pixel-scale thresholds. The half-spacing edge margin is an explicit convention, not a surveyed sowing margin.

## Outputs for the next processing stage

| File | Contents |
| --- | --- |
| `01_rectified_points_all.csv` | All input points in the rectified coordinate system |
| `04_dp_selected_boundaries.csv` | Selected separator evidence |
| `05_plot_boundaries.csv` | Refined plot intervals: use `s_start`, `s_end` and `length` |
| `05a_plot_boundaries_before_edge_refinement.csv` | Initial plot intervals |
| `06_plot_points.csv` | Existing partition-stage ownership and gap flags |
| `06a_partition_assigned_points.csv` | Points assigned to plots, before final row fitting or point-quality filtering |
| `06b_partition_gap_points.csv` | Other points from the partition table, including excluded gap points |
| `partitioned_points/plot_XXXX.csv` | One assigned-point table per plot for downstream processing |
| `09_points_with_plot_boundaries.png` | Partition points and the same refined intervals as the boundary table |
| `30_partition_edge_refinement.csv` | Edge shifts, support and review flags |
| `31_partition_junctions.csv` | Evidence at adjacent plot ends |
| `partition_edges.json` | Edge-refinement settings and detailed support |
| `partition_handoff.json` | Input files and stage information for the subsequent program |
| `run_status.json` | Run state and input/code/configuration hashes |

The all-input table preserves points outside the original partition assignment span. `06_plot_points.csv` retains the preceding assignment algorithm's span and gap treatment. Point-quality filtering has not been applied to the assigned-point exports. Counts in the partition boundary table describe initial assigned detections, not final plant counts.

Use the refined intervals and assigned-point tables as inputs to a separate final row-estimation and phenotype program. Do not interpret `row_main` as an already fitted final row label. Structural/full-cover fields in the boundary table retain their earlier diagnostic meanings; the canonical plot edges are `s_start/s_end`.

The downstream program should consume these saved plot intervals. It should not automatically refit the partition edges from its later filtered points. A deliberate revision of the partition is a separate operation.

## Verification and review

Run focused checks with:

```shell
python -m unittest discover -s tests -v
```

`VALIDATION.md` records the full-data comparisons. A `completed_needs_review` status means that separator or edge evidence needs inspection. Point-supported edges are not surveyed sowing boundaries, and preserving an existing partition does not correct its known errors.
