# Edge-policy verification

This update changes only the edge module and its explicit uncertainty policy. The main partition program is byte-identical to the preceding release.

## Checks on 2026-11-13

- Replayed the changed edge stage on 41 saved historical partition inputs and the newly supplied result: 42 input cases, 308 plot cases and 220,678 assigned-point cases.
- Both `visible_extent` and `retain_initial` were checked for point containment and valid intervals. The visible policy also passed adjacent-interval non-overlap checks.
- Previously accepted two-sided multirow edges retained their coordinates.
- Original point records and ownership were not changed. The original separator selection was reused, not rerun during this regression replay.
- All 22 focused tests passed, including independent upper/lower edges with no reference gaps, explicit visible fallback, conservative missingness treatment and the accepted shrinkage of shared terminal missingness.
- A fresh complete partition pipeline run on short-plot `data41` completed using the updated files.
- For the supplied result, only edges, edge diagnostics, summary and figures were regenerated. Input coordinates, selected separators and point/assignment tables were retained byte-for-byte.

These are numerical consistency and stage-separation checks. They are not surveyed edge-accuracy or independent generalization measurements. The visible policy is intentionally allowed to reduce common terminal missingness space.

## Supplied screenshot junction

| Edge | Previous coordinate | Updated coordinate | Method |
| --- | --- | --- | --- |
| Plot 14 end | 9319.251355 | 8811.868611 | Independently supported multirow edge |
| Plot 15 start | 9669.251355 | 10262.928900 | Visible-extent fallback, review required |

The supplied output directory was named `data11`; its recorded input path referred to `data12_2L_V6`. The replay used that supplied partition handoff, not a different historical `data11` input.

## Source fingerprints

- `plot_partition.py`: `00eab52fe44f7f3e26a3d4dcbd5ff4dbd3c1b7164d60fbdef82a9e0c8c9a4cbc`
- `partition_config.py`: `790082202e8c56f5f6387baa8a6b9b036e42b8f246ebd2b2a0e911f4da2b282d`
- `partition_edges.py`: `f29652f16e7e2243518fc3a255b3776db0ac43fa7aafef416d0c71d20be6ffb7`
