# Verification record

Verification date: 13 November 2026.

## Reference scripts and scope

The reference construction script has SHA256 `d48aa990d60273d9f4c42bfcff85cdd9c7edb3127eea96e3f3e753b171aa1eef`. The reference fusion script has SHA256 `f232c548a24165218804bafb8c854e9a50f94ea070afa2aabcaf2f73673678e8`.

After normalizing output filenames, docstrings and image-I/O calls, twelve construction functions and nine fusion functions have matching syntax trees. The comparison excludes added input checks, summary returns and the explicit empty-core table schema. Numerical defaults, ownership tests, blend weights, matching and cleaning thresholds are retained. Unused preview settings and an unused overlap helper were removed.

## Construction comparison

The input used six selected keyframes from a real 20-frame registration sequence. Eleven sample PNG files, two metadata CSV files and the strip metadata JSON matched the reference byte-for-byte after output filename mapping. This checks the construction on these registered source images, not on every frame of a longer acquisition.

## Fusion comparison

Generated detector boxes were placed in the actual crop coordinate frames: 143 core predictions and 52 seam predictions. The four fusion CSV outputs matched byte-for-byte after filename mapping, retaining 108 detections. These are controlled coordinate and decision fixtures, not measured detector predictions or manually annotated field accuracy references.

The existing core/seam output columns are retained. The retained-table filename was checked against the plot-partition program's batch pattern.

## Edge cases and execution checks

Eleven public tests passed. They cover source crop pixels and origins, box restoration, core fallback, seam replacement, context-only exclusion, seam-only input, unmatched filenames, empty results, residual cleaning, disabled cleaning and inconsistent source-image dimensions.

The reference fusion script fails on a header-only core detection table with valid seam detections (`KeyError` for a missing decision column). The public implementation defines the empty decision-table columns, so eligible seams proceed through the same ownership and residual-cleaning rules. Invalid filename joins raise errors rather than silently discarding unmatched observations. Equal-size checks prevent the first-image projection map from being applied to differently sized source images.

Execution used Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 and OpenCV 5.0.0. Source syntax was checked against Python 3.11. Command-line entry points were exercised separately from the callable interfaces.

## Interpretation

These checks establish behaviour preservation for the recorded valid inputs and the stated empty-input correction. They do not measure field recall, duplicate-removal accuracy, seedling count error or registration accuracy. A complete long field sequence and the subsequent detector/plot-trait pipeline were not rerun for this release.
