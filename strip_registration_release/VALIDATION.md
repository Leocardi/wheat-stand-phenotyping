# Verification record

Checks performed on 13 November 2026. The supplied baseline has SHA256 `b7c71be456ce45e194d471caae16eced34ae64349ef3cf0bd48900c14905d3e8`.

## Scope

This release reorganizes configuration, removes development notes and private paths, and adds execution and file-I/O checks. It retains the numerical registration and selection rules. Eighteen retained computational functions, including the processing body, have matching syntax trees after normalizing configuration access, docstrings and image-I/O calls; the processing comparison also excludes the added dimension check and summary return. The unused `NCC_SCALE` setting and unused `is_valid_forward_step` function were removed.

The comparison used the same Python/NumPy/pandas/OpenCV environment for both implementations. It is a behaviour-preservation check, not an independent accuracy measurement.

## End-to-end baseline comparisons

| Case | Frames | Compared CSV files | Compared images | Maximum numeric difference |
| --- | --- | --- | --- | --- |
| real20 | 20 | 7 | 14 | 0 |
| synthetic_r0 | 12 | 7 | 10 | 0 |
| synthetic_r90 | 12 | 7 | 10 | 1.61e-07 |
| synthetic_r180 | 12 | 7 | 10 | 1.61e-07 |
| synthetic_r270 | 12 | 7 | 10 | 1.61e-07 |
| synthetic_reverse | 12 | 7 | 10 | 2.62e-07 |

The real 20-frame case produced byte-identical original outputs: seven CSV files, fourteen images (source keyframes and previews) and the original metadata JSON. All six cases retained the same image pixels and source selections. Some synthetic runs differed in final floating-point digits, with maximum difference below 3 × 10⁻⁷ in exported numeric fields; comparisons used absolute tolerance 10⁻⁵ and zero relative tolerance. Such outputs are not reported as universally byte-identical. No rounding was introduced to force agreement.

The reverse-order case verifies preservation of the supplied direction behaviour. It does not establish correct ordinary image placement under every acquisition orientation; negative normalized direction is marked for review.

## Focused checks

Nine public tests passed: known synthetic translation, rotated-source exports, Unicode image I/O, corrupt input, inconsistent dimensions, motion fallback reporting, nonempty output handling, NCC fallback, and prevention of image/coordinate rotation mismatch. A separate JPEG check confirmed identical decoding to the original reader for EXIF orientation values 1, 3, 6 and 8. Source syntax was checked using the Python 3.11 grammar; execution used Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 and OpenCV 5.0.0.

## Limits

The long field sequence, surveyed coordinate error, seam/core construction, detection, duplicate suppression and final plot traits were not rerun as part of this packaging task. The existing core/seam reader's required CSV fields and image-directory convention were checked. Retaining an algorithm's outputs does not correct its existing registration or coverage errors.
