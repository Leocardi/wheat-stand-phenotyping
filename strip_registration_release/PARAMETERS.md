# Registration parameters

Edit the grouped dictionaries in `registration_config.py`. Command-line overrides are saved in `run_config.json`. Distances refer to the rotated source images unless a registration-grid or preview unit is stated.

## Input Output

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `INPUT_DIR` | `'data/frames'` | Path | Source frame directory; the CLI --input option overrides it. |
| `OUTPUT_DIR` | `'results/strip'` | Path | Requested destination; a nonempty directory receives a new run subdirectory. |
| `IMG_EXTS` | `.bmp, .jpeg, .jpg, .png, .tif, .tiff` | Extensions | Case-insensitive source extensions. |
| `ROTATE_DEG` | `180` | Degrees clockwise | Right-angle rotation applied consistently to estimation and source export. |

## Registration

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `REG_SCALE` | `0.25` | Ratio | Downsampling for displacement estimation. NCC uses these same registration images. |
| `MARGIN_RATIO` | `0.08` | Fraction per edge | Border removed after registration-image downsampling. |
| `PHASE_MIN_RESP` | `0.15` | Response | Minimum phase-correlation response before local NCC is considered. |
| `DX_CLIP` | `120` | Full-resolution pixels | Symmetric horizontal displacement clipping, before the forward/anchor checks. |
| `MAX_DY_FRAC` | `0.9` | Fraction of rotated height | Maximum absolute vertical displacement used for clipping. |

## Preprocessing

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `USE_CLAHE` | `True` | Boolean | Apply CLAHE to registration grayscale images only. |
| `USE_VEG_MASK` | `True` | Boolean | Suppress vegetation in registration texture images only. |
| `USE_GLARE_SOFT` | `True` | Boolean | Attenuate registration texture in glare-mask regions. |
| `GLARE_V_TH` | `245` | HSV V | Minimum brightness for the glare mask. |
| `GLARE_S_TH` | `60` | HSV S | Maximum saturation for the glare mask. |
| `GLARE_REDUCE` | `0.3` | Multiplier | Registration-gradient multiplier within the glare mask. |

## Ncc Search

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `USE_NCC_FALLBACK` | `True` | Boolean | Search around the predicted shift when phase response is low. |
| `NCC_MIN_SCORE` | `0.25` | Normalized score | Minimum normalized template-correlation score. |
| `SEARCH_DX` | `30` | Registration-grid pixels | Horizontal half-range of the NCC search, not full-resolution pixels. |
| `SEARCH_DY` | `180` | Registration-grid pixels | Vertical half-range around the predicted NCC displacement. |
| `TEMPLATE_W_RATIO` | `0.7` | Fraction | NCC template width relative to registration-image width. |
| `TEMPLATE_H_RATIO` | `0.35` | Fraction | NCC template height relative to registration-image height. |

## Keyframe Selection

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `KEY_OVERLAP_TARGET` | `0.65` | Fraction of rotated height | Target geometry-keyframe overlap in the coarse trajectory. |
| `MIN_OVERLAP_GUARD` | `0.5` | Fraction of rotated height | Geometry candidate guard; final corrected overlaps are reported separately. |
| `Y_WINDOW_FRAC` | `0.2` | Fraction of rotated height | Candidate distance on either side of the target y location. |
| `CAND_MAX` | `12` | Candidate-loop budget | Nominal budget; left/right traversal can append one additional candidate in its final iteration. |
| `USE_CONTINUITY_AWARE_DET_KEYFRAMES` | `True` | Boolean | Combine image quality with interval position, drift and motion reliability. |

## Motion Checks

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `VALIDATE_FRAME_SHIFT` | `True` | Boolean | Replace too-small/backward frame motion with the supplied prediction rule. |
| `VALIDATE_ANCHOR_SHIFT` | `True` | Boolean | Check geometry-anchor refinements before accepting them. |
| `MIN_FRAME_FORWARD_PX` | `1.0` | Full-resolution pixels | Minimum forward frame increment. |
| `MIN_ANCHOR_FORWARD_PX` | `30.0` | Full-resolution pixels | Minimum forward geometry-anchor increment. |
| `MAX_ANCHOR_FORWARD_FRAC` | `0.85` | Fraction of rotated height | Maximum forward geometry-anchor increment. |
| `ANCHOR_STEP_TOL_FRAC` | `0.45` | Fraction of rotated height | One component of the maximum permitted deviation from the coarse anchor prediction. |
| `ANCHOR_STEP_TOL_RATIO` | `0.9` | Ratio | Other tolerance component; use max(height fraction, ratio × max(abs(prediction), 1)). |
| `ANCHOR_DX_MAX` | `240.0` | Full-resolution pixels | Guard on already clipped horizontal anchor displacement; at default DX_CLIP=120 this 240-pixel guard is inactive. |

## Image Export

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `EXPORT_KEYFRAME_IMAGES` | `True` | Boolean | Export source frames for both keyframe sets. |
| `KEYFRAMES_DIRNAME` | `'keyframes_images'` | Directory name | Detection source-frame export. |
| `GEOM_KEYFRAMES_DIRNAME` | `'geom_keyframes_images'` | Directory name | Geometry source-frame export. |
| `SAVE_ROTATED_KEYFRAMES` | `True` | Boolean | Keep True when rotation is nonzero, so exported images match the coordinate convention. |
| `JPEG_QUALITY` | `95` | 1 to 100 | JPEG re-encoding quality for exported keyframes. |
| `PNG_COMPRESSION` | `3` | 0 to 9 | Lossless PNG compression level for exported keyframes. |

## Preview

| Setting | Default | Unit | Meaning |
| --- | --- | --- | --- |
| `RENDER_PREVIEW` | `True` | Boolean | Render inspection mosaics; does not affect estimates or selections. |
| `RENDER_SCALE` | `0.2` | Ratio | Preview scale only; exported source images remain full resolution. |
| `FEATHER_PX` | `30` | Preview pixels | Blend-band size after preview downsampling. |
| `CHUNK_HEIGHT_FULL` | `60000` | Full-resolution equivalent pixels | Preview chunk step before multiplication by RENDER_SCALE. |
| `EXPORT_DET_KEYFRAME_PREVIEW` | `True` | Boolean | Render the detection-keyframe preview when RENDER_PREVIEW is also enabled. |

## Fixed processing rules

The following constants remain in the processing functions rather than the user-facing configuration:

- Quality scores are evaluated at scale 0.25. Detection quality is log(1 + grayscale Laplacian variance) − 8 × extreme-brightness fraction − 4 × glare fraction. Extreme brightness uses V > 250 or V < 5; the quality-score glare fraction uses V > 245 and S < 60. These quality thresholds are separate from the registration glare-mask settings.
- Registration preprocessing uses CLAHE clip limit 2.0 with an 8 × 8 tile grid, Sobel kernel 3, high-pass Gaussian sigma 3, and standard-deviation normalization with epsilon 10⁻⁶. Vegetation suppression combines excess-green Otsu thresholding with HSV H in [35, 95], S ≥ 40 and V ≥ 40, followed by the implemented morphology.
- Direction estimation inspects at most the first 30 frames. At least five shifts with sufficient response are needed to use their median; otherwise the direction defaults positive and is marked for review.
- Frame-step prediction uses the median absolute motion after eight history entries; history retains at most 30 steps. Before that, prediction is (1 − overlap target) × image height. A missing prediction inside displacement estimation uses 0.6 × image height × (1 − overlap target).
- Geometry candidate score is 1.2 × texture quality + 0.6 × detection quality − distance to target/(0.10 × image height + 10⁻⁶). First and last source frames are geometry anchors. The fallback candidate and the final anchor can bypass the candidate overlap test.
- Detection-keyframe selection starts from image quality. With continuity scoring enabled, it subtracts 0.0015 × distance to the interval midpoint and 0.0008 × distance from the interval median x. An out-of-interval position costs 100; unreliable method, low response and too-small forward step cost 0.35, 0.25 and 1.0 respectively. Each geometry interval includes both endpoint indices; duplicate selected indices are removed, and the final geometry frame is retained.
- Anchor corrections are interpolated in coarse y, using frame index only for a near-zero coarse-y denominator. This corrects translations; it does not warp source pixels.

Changes to these rules alter the method even if the grouped configuration is unchanged. Archive the processing file and configuration together when reporting results.
