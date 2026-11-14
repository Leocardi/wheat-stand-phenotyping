# Wheat stand phenotyping

Code for reconstructing strip coordinates from image sequences, combining core/seam seedling detections, and partitioning wheat plots with longitudinal edge refinement.

## Workflow

```text
Ordered source frames
    |
    v
1. Strip coordinate reconstruction and source-keyframe export
    |
    v
2. Core/seam sample construction
    |
    v
   Seedling detection on both sample sets (external detector)
    |
    v
3. Coordinate restoration and duplicate handling
    |
    v
4. Plot partitioning and edge refinement
```

The repository contains three modules. The core/seam module has two entry points, which run before and after detection. Final crop-row estimation, point-quality filtering, and phenotype calculation are separate downstream steps.

## Repository structure

| Directory | Entry points | Purpose |
| --- | --- | --- |
| [`strip_registration_release`](strip_registration_release/README.md) | `strip_registration.py` | Estimate strip coordinates, select keyframes, and export source images |
| [`core_seam_release`](core_seam_release/README.md) | `build_core_seam_dataset.py`, `merge_core_seam_results.py` | Build detector samples, restore local detections, and handle duplicate observations |
| [`plot_partition_release`](plot_partition_release/README.md) | `plot_partition.py` | Select plot separators, assign points, and refine longitudinal plot edges |

Keep the Python files within each module together. Configuration and helper modules are imported by the entry points.

## Installation

Python 3.12 is recommended. The recorded checks used Python 3.12.14. Source syntax was also checked against Python 3.11; runtime execution on Python 3.11 was not tested.

Clone the repository and create an environment:

```shell
git clone https://github.com/Leocardi/wheat-stand-phenotyping.git
cd wheat-stand-phenotyping
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on Linux/macOS:

```shell
source .venv/bin/activate
```

The three modules use NumPy, pandas, and OpenCV. Install their shared dependencies using the pinned versions recorded for verification:

```shell
python -m pip install -r plot_partition_release/requirements.txt
```

This installs `numpy==2.3.5`, `pandas==3.0.1`, and `opencv-python-headless==5.0.0.93`. The headless OpenCV package supports the file-based image processing used here. Do not install both `opencv-python` and `opencv-python-headless` in this environment.

Detector inference requires its own compatible framework, model configuration, and dependencies. The installation above covers the three processing modules, not D-FINE inference or training.

## Usage

Run the following commands from the repository root. Paths are examples; replace them with your data locations. Quote paths that contain spaces.

When an output directory is already nonempty, the programs create a timestamped run subdirectory. Use the actual directory printed as `[OUTPUT]` in subsequent commands.

### 1. Reconstruct strip coordinates

Place ordered source frames in `data/frames/`, using unique, consistently numbered filenames. Images must have the same dimensions after rotation. Video-frame extraction is performed separately.

```shell
python strip_registration_release/strip_registration.py --input "data/frames" --output "results/strip" --rotate 180
```

Choose `--rotate 0`, `90`, `180`, or `270` to match the acquisition orientation. The example uses 180 degrees; it is not required for every dataset.

The main outputs for the next stage are:

- `results/strip/keyframes.csv`: source-keyframe positions and image dimensions.
- `results/strip/keyframes_images/`: selected source frames at their original resolution, with the configured rotation.

These keyframes are not crops of a stitched panorama. Preview mosaics are provided for inspection and must not replace the source images supplied to sample construction.

### 2. Construct core/seam samples

```shell
python core_seam_release/build_core_seam_dataset.py --keyframes "results/strip/keyframes.csv" --images "results/strip/keyframes_images" --output "results/core_seam"
```

The output directory contains:

- `core_crops/`: contextual crops from individual source keyframes.
- `seam_crops/`: samples combining patches from adjacent source keyframes.
- `core_crops_meta.csv`, `seam_crops_meta.csv`, and `strip_meta.json`: crop positions, ownership intervals, and strip geometry needed for coordinate restoration.

Preserve these metadata files and the crop filenames when running the detector.

### 3. Detect seedlings

Run a compatible detector separately on `core_crops/` and `seam_crops/`. Inference is an external step; the three processing modules do not execute the detection model.

Export two CSV files, for example:

```text
results/core_detections/detections.csv
results/seam_detections/detections.csv
```

Each table must contain these columns:

```csv
file,x1,y1,x2,y2,score
```

`file` must match the corresponding `crop_file` in the metadata table. Box coordinates must refer to pixels in the saved crop image. Undo any detector resizing or padding before exporting coordinates. Additional columns are ignored by the merger.

If one sample set has no detections, provide a header-only CSV rather than a zero-byte file.

### 4. Restore coordinates and merge detections

```shell
python core_seam_release/merge_core_seam_results.py --keyframes "results/strip/keyframes.csv" --dataset "results/core_seam" --core-detections "results/core_detections/detections.csv" --seam-detections "results/seam_detections/detections.csv" --output "results/merged"
```

The retained global detections are written to:

```text
results/merged/dedup_keep_global_core_seam_safe_light_clean.csv
```

The merger also exports decision records and residual-duplicate diagnostics. It does not run detection or render a full-strip image.

### 5. Partition plots and refine their edges

```shell
python plot_partition_release/plot_partition.py --input "results/merged/dedup_keep_global_core_seam_safe_light_clean.csv" --output "results/plots"
```

Partition edge refinement is enabled by default. It uses points assigned during partitioning, before later crop-row estimation or point-quality filtering. Plot IDs and point ownership are preserved by the edge-refinement step.

Main outputs include:

| Output | Purpose |
| --- | --- |
| `05_plot_boundaries.csv` | Refined longitudinal intervals in `s_start`, `s_end`, and `length` |
| `05a_plot_boundaries_before_edge_refinement.csv` | Initial intervals for comparison |
| `06a_partition_assigned_points.csv` | Assigned points for downstream processing |
| `partitioned_points/plot_XXXX.csv` | Assigned points for each plot |
| `09_points_with_plot_boundaries.png` | Visual inspection of partition points and refined intervals |
| `30_partition_edge_refinement.csv` | Edge shifts, supporting evidence, and review flags |

To produce a control run without edge refinement:

```shell
python plot_partition_release/plot_partition.py --input "results/merged/dedup_keep_global_core_seam_safe_light_clean.csv" --output "results/plots_initial" --no-edge-refinement
```

For batch processing, place deduplicated tables under a common directory:

```shell
python plot_partition_release/plot_partition.py --batch-root "results/batch_merged" --pattern "dedup_keep_global_*_safe_light_clean.csv" --output "results/batch_plots"
```

## Pretrained weights

The seedling detector checkpoint is available as a release asset:

- [Download `seedling.pth`](https://github.com/Leocardi/wheat-stand-phenotyping/releases/download/v1.0.0/seedling.pth)
- [Release v1.0.0](https://github.com/Leocardi/wheat-stand-phenotyping/releases/tag/v1.0.0)

Release assets are downloaded separately from the source repository. Use the checkpoint with the matching detector implementation, custom model components, and YAML configuration. A checkpoint alone does not specify the model architecture or preprocessing.

## Configuration and interpretation

The grouped defaults are in `registration_config.py`, `core_seam_config.py`, and `partition_config.py` within their respective modules. Module READMEs and parameter documentation describe the available settings and decision rules.

- Registration uses a translation-only model. Exported coordinates are image pixels, not calibrated physical ground coordinates. Source-image orientation must agree with the coordinate tables.
- Plot partitioning has no prescribed plot count or physical plot-length input. It still uses crop-row assumptions and pixel-scale thresholds, which should be reviewed for a new acquisition setup.
- The default uncertain-edge policy follows the visible assigned detections. It can exclude terminal space in which all rows failed to emerge; the refined limits are not surveyed sowing boundaries.
- Review the diagnostic tables and figures before quantitative use. `completed_needs_review` indicates that execution produced outputs requiring inspection.
- Downstream row estimation and phenotype analysis should consume the saved plot intervals and assigned-point tables as a separate processing stage.

## Software checks

Run each module's checks from its own directory:

```shell
cd strip_registration_release
python -m unittest discover -s tests -v
cd ../core_seam_release
python -m unittest discover -s tests -v
cd ../plot_partition_release
python -m unittest discover -s tests -v
cd ..
```

Each module includes `VALIDATION.md` describing its recorded comparisons and their scope. These software checks do not by themselves establish field registration accuracy, detector recall, or plot-boundary accuracy.
