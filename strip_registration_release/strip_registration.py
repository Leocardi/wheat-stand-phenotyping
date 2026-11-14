"""Reconstruct strip coordinates and export full-resolution source keyframes.

Run this file as a script. Edit grouped defaults in registration_config.py.
Mosaic previews are inspection products, not images for downstream detection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

from registration_config import CONFIG as DEFAULT_CONFIG

CONFIG = dict(DEFAULT_CONFIG)


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Decode an image from a filesystem path, including Unicode paths."""
    path = Path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, flags)
    except (OSError, cv2.error) as exc:
        raise OSError("Cannot decode source image: {}".format(path)) from exc
    if image is None:
        raise OSError("Cannot decode source image: {}".format(path))
    return image


def write_image(path, image, params=None):
    """Encode an image and fail explicitly if writing does not succeed."""
    path = Path(path)
    success, encoded = cv2.imencode(
        path.suffix, image, [] if params is None else params
    )
    if not success:
        raise OSError("Image encoding failed: {}".format(path))
    encoded.tofile(path)


def check_frame_shape(image, expected, path):
    if image.shape[:2] != expected:
        raise ValueError(
            "All rotated frames must have the same dimensions; {} has {}, expected {}".format(
                path, image.shape[:2], expected
            )
        )


def natural_key(p: Path):
    """Split a filename into numeric and text components for sorting."""
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split("(\\d+)", s)]


def rotate_img(img, deg):
    """Apply the configured right-angle rotation without resampling."""
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("ROTATE_DEG must be 0/90/180/270")


def list_images(input_dir):
    """List source images in natural filename order."""
    folder = Path(input_dir)
    if not folder.is_dir():
        raise NotADirectoryError("Input directory does not exist: {}".format(folder))
    files = sorted(
        (
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in CONFIG["IMG_EXTS"]
        ),
        key=natural_key,
    )
    if len(files) < 2:
        raise ValueError("At least two source images are required: {}".format(folder))
    return files


def crop_margin(img, margin_ratio):
    """Remove an equal fractional border from the registration image."""
    if margin_ratio <= 0:
        return img
    h, w = img.shape[:2]
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    if mx * 2 >= w or my * 2 >= h:
        return img
    return img[my : h - my, mx : w - mx]


def glare_mask(bgr_small: np.ndarray) -> np.ndarray:
    """Identify bright, low-saturation regions for soft suppression."""
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    m = ((v >= CONFIG["GLARE_V_TH"]) & (s <= CONFIG["GLARE_S_TH"])).astype(
        np.uint8
    ) * 255
    m = cv2.medianBlur(m, 5)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
    return m


def veg_mask_exg_hsv(bgr_small: np.ndarray) -> np.ndarray:
    """Combine excess green and HSV thresholds for vegetation suppression."""
    bgr = bgr_small.astype(np.float32)
    b, g, r = cv2.split(bgr)
    exg = 2 * g - r - b
    exg = cv2.GaussianBlur(exg, (0, 0), 1)
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 35) & (h <= 95) & (s >= 40) & (v >= 40)
    _, th = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (th > 0) & green
    m = m.astype(np.uint8) * 255
    m = cv2.medianBlur(m, 3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return m


def build_reg_image(bgr: np.ndarray, scale: float) -> np.ndarray:
    """Build a normalized texture image for displacement estimation only."""
    h, w = bgr.shape[:2]
    bgr_s = cv2.resize(
        bgr,
        (max(32, int(w * scale)), max(32, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    bgr_s = crop_margin(bgr_s, CONFIG["MARGIN_RATIO"])
    gray = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY)
    if CONFIG["USE_CLAHE"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag - cv2.GaussianBlur(mag, (0, 0), 3)
    mag = np.maximum(mag, 0)
    if CONFIG["USE_VEG_MASK"]:
        vm = veg_mask_exg_hsv(bgr_s) > 0
        mag[vm] = 0.0
    if CONFIG["USE_GLARE_SOFT"]:
        gm = glare_mask(bgr_s) > 0
        mag[gm] *= float(CONFIG["GLARE_REDUCE"])
    mag = mag - mag.mean()
    mag = mag / (mag.std() + 1e-06)
    return mag.astype(np.float32)


def phase_shift(prev_reg: np.ndarray, curr_reg: np.ndarray, scale: float):
    """Estimate prev(x, y) = curr(x + dx, y + dy), in full-resolution pixels."""
    h, w = prev_reg.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx_s, dy_s), resp = cv2.phaseCorrelate(prev_reg * win, curr_reg * win)
    dx = float(dx_s / scale)
    dy = float(dy_s / scale)
    return (dx, dy, float(resp))


def ncc_search_shift(
    prev_reg: np.ndarray,
    curr_reg: np.ndarray,
    pred_dx_s: float,
    pred_dy_s: float,
    sx: int,
    sy: int,
):
    """Search for a template near the predicted shift in scaled coordinates."""
    h, w = prev_reg.shape[:2]
    tw = int(w * CONFIG["TEMPLATE_W_RATIO"])
    th = int(h * CONFIG["TEMPLATE_H_RATIO"])
    tw = max(32, min(tw, w - 2))
    th = max(32, min(th, h - 2))
    x0 = (w - tw) // 2
    y0 = (h - th) // 2
    tpl = prev_reg[y0 : y0 + th, x0 : x0 + tw]
    sx0 = int(round(x0 + pred_dx_s - sx))
    sy0 = int(round(y0 + pred_dy_s - sy))
    sx1 = sx0 + tw + 2 * sx
    sy1 = sy0 + th + 2 * sy
    sx0c = max(0, sx0)
    sy0c = max(0, sy0)
    sx1c = min(w, sx1)
    sy1c = min(h, sy1)
    if sx1c - sx0c < tw or sy1c - sy0c < th:
        return (None, None, None)
    search = curr_reg[sy0c:sy1c, sx0c:sx1c]
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    found_x = sx0c + max_loc[0]
    found_y = sy0c + max_loc[1]
    dx_s = float(found_x - x0)
    dy_s = float(found_y - y0)
    return (dx_s, dy_s, float(max_val))


def estimate_shift(
    prev_bgr, curr_bgr, scale, pred_sy_full=None, sign_y=1.0, H_full=2000
):
    """Estimate a clipped translation, using local NCC if phase response is low."""
    max_dy_full = float(CONFIG["MAX_DY_FRAC"] * H_full)
    prev_reg = build_reg_image(prev_bgr, scale)
    curr_reg = build_reg_image(curr_bgr, scale)
    dx, dy, resp = phase_shift(prev_reg, curr_reg, scale)
    dx = float(np.clip(dx, -CONFIG["DX_CLIP"], CONFIG["DX_CLIP"]))
    dy = float(np.clip(dy, -max_dy_full, max_dy_full))
    if resp >= CONFIG["PHASE_MIN_RESP"]:
        return (dx, dy, resp, "phase")
    if not CONFIG["USE_NCC_FALLBACK"]:
        return (dx, dy, resp, "phase_lowresp")
    if pred_sy_full is None:
        pred_sy_full = 0.6 * H_full * (1.0 - CONFIG["KEY_OVERLAP_TARGET"])
    pred_dy_full = -(pred_sy_full / sign_y)
    pred_dx_s = 0.0
    pred_dy_s = pred_dy_full * scale
    dx_s, dy_s, score = ncc_search_shift(
        prev_reg,
        curr_reg,
        pred_dx_s,
        pred_dy_s,
        sx=CONFIG["SEARCH_DX"],
        sy=CONFIG["SEARCH_DY"],
    )
    if score is None or score < CONFIG["NCC_MIN_SCORE"]:
        return (dx, dy, resp, "fail")
    dx2 = float(dx_s / scale)
    dy2 = float(dy_s / scale)
    dx2 = float(np.clip(dx2, -CONFIG["DX_CLIP"], CONFIG["DX_CLIP"]))
    dy2 = float(np.clip(dy2, -max_dy_full, max_dy_full))
    return (dx2, dy2, float(score), "ncc")


def validate_anchor_shift(
    dx: float, dy: float, pred_sy: float, sign_y: float, h_full: float
):
    """Check forward displacement and agreement with the coarse anchor step."""
    sx = -float(dx)
    sy = float(sign_y) * -float(dy)
    reasons = []
    if not np.isfinite(sx) or not np.isfinite(sy):
        reasons.append("non_finite")
    if sy < float(CONFIG["MIN_ANCHOR_FORWARD_PX"]):
        reasons.append("non_forward_or_too_small_sy")
    if sy > float(CONFIG["MAX_ANCHOR_FORWARD_FRAC"] * h_full):
        reasons.append("too_large_sy")
    if abs(float(dx)) > float(CONFIG["ANCHOR_DX_MAX"]):
        reasons.append("too_large_dx")
    tol = max(
        float(CONFIG["ANCHOR_STEP_TOL_FRAC"] * h_full),
        float(CONFIG["ANCHOR_STEP_TOL_RATIO"] * max(abs(pred_sy), 1.0)),
    )
    if np.isfinite(pred_sy) and abs(sy - float(pred_sy)) > tol:
        reasons.append("far_from_coarse_prediction")
    return (len(reasons) == 0, sx, sy, ";".join(reasons))


def check_monotonic_sequence(xy: np.ndarray, name: str, out_dir: Path, files=None):
    """Export non-forward global-y increments for review."""
    rows = []
    for i in range(1, len(xy)):
        dy = float(xy[i, 1] - xy[i - 1, 1])
        dx = float(xy[i, 0] - xy[i - 1, 0])
        if dy <= 0:
            rows.append(
                {
                    "i_prev": i - 1,
                    "i_curr": i,
                    "file_prev": Path(files[i - 1]).name if files is not None else "",
                    "file_curr": Path(files[i]).name if files is not None else "",
                    "dx": dx,
                    "dy": dy,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / f"{name}_nonforward_warnings.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[WARN] {name}: {len(rows)} non-forward steps saved to {name}_nonforward_warnings.csv"
        )
    else:
        print(f"[INFO] {name}: no non-forward global-y steps detected.")


def detection_quality(bgr: np.ndarray, scale=0.25) -> float:
    """Score sharpness, near-black/white saturation, and glare."""
    h, w = bgr.shape[:2]
    bgr_s = cv2.resize(
        bgr,
        (max(64, int(w * scale)), max(64, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    sat = float(((v > 250) | (v < 5)).mean())
    glare = float(((v > 245) & (s < 60)).mean())
    return float(math.log(lap + 1.0) - 8.0 * sat - 4.0 * glare)


def texture_score_for_geom(bgr: np.ndarray, scale=0.25) -> float:
    """Score registration texture for geometry-keyframe selection."""
    reg = build_reg_image(bgr, scale)
    return float(np.mean(np.abs(reg)))


def select_geom_keyframes(files, y_coarse, q_det, q_tex, H_full):
    """Select geometry keyframes using overlap targets and image quality."""
    step_target = (1.0 - CONFIG["KEY_OVERLAP_TARGET"]) * float(H_full)
    max_step = (1.0 - CONFIG["MIN_OVERLAP_GUARD"]) * float(H_full)
    geom = [0]
    N = len(files)
    while True:
        i0 = geom[-1]
        y0 = y_coarse[i0]
        y_target = y0 + step_target
        if y_target >= y_coarse[-1]:
            break
        j = int(np.searchsorted(y_coarse, y_target, side="left"))
        j = max(i0 + 1, min(j, N - 1))
        win = CONFIG["Y_WINDOW_FRAC"] * float(H_full)
        cand = []
        left = j
        right = j
        while (left >= i0 + 1 or right < N) and len(cand) < CONFIG["CAND_MAX"]:
            if left >= i0 + 1 and abs(y_coarse[left] - y_target) <= win:
                cand.append(left)
            if right < N and abs(y_coarse[right] - y_target) <= win and (right != left):
                cand.append(right)
            left -= 1
            right += 1
            if left < i0 + 1 and right >= N:
                break
        if not cand:
            cand = [j]
        best = None
        best_score = -1e18
        for c in cand:
            dy = y_coarse[c] - y0
            if dy <= 0:
                continue
            if dy > max_step:
                continue
            close_pen = -abs(y_coarse[c] - y_target) / (0.1 * H_full + 1e-06)
            score = 1.2 * q_tex[c] + 0.6 * q_det[c] + close_pen
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            best = min(i0 + 1, N - 1)
        geom.append(best)
    if geom[-1] != N - 1:
        geom.append(N - 1)
    geom = sorted(set(geom))
    return geom


def anchor_correct_all(coarse_xy, key_idx, anchor_xy):
    """Interpolate anchor corrections along the coarse longitudinal trajectory."""
    N = coarse_xy.shape[0]
    corrected = coarse_xy.copy()
    for k in range(len(key_idx) - 1):
        i0, i1 = (key_idx[k], key_idx[k + 1])
        p0, p1 = (coarse_xy[i0], coarse_xy[i1])
        a0, a1 = (anchor_xy[k], anchor_xy[k + 1])
        d0 = a0 - p0
        d1 = a1 - p1
        denom = p1[1] - p0[1]
        for i in range(i0, i1 + 1):
            if abs(denom) > 1e-06:
                t = (coarse_xy[i, 1] - p0[1]) / denom
            else:
                t = (i - i0) / max(1, i1 - i0)
            t = float(np.clip(t, 0.0, 1.0))
            corrected[i] = coarse_xy[i] + (1 - t) * d0 + t * d1
    last_i = key_idx[-1]
    last_delta = anchor_xy[-1] - coarse_xy[last_i]
    for i in range(last_i, N):
        corrected[i] = coarse_xy[i] + last_delta
    return corrected


def _imwrite_smart(path: Path, img: np.ndarray):
    """Write a source keyframe using the configured PNG or JPEG encoder settings."""
    ext = path.suffix.lower()
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(CONFIG["JPEG_QUALITY"])]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(CONFIG["PNG_COMPRESSION"])]
    write_image(str(path), img, params)


def export_keyframe_images(files, indices, out_dir, rotate_deg, save_rotated=True):
    """Export selected source images; no mosaic pixels are used."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        source = Path(files[idx])
        destination = out_dir / source.name
        if save_rotated:
            image = rotate_img(read_image(source), rotate_deg)
            _imwrite_smart(destination, image)
        else:
            shutil.copy2(source, destination)


def render_preview(files, geom_idx, geom_xy, out_dir: Path, W_full, H_full):
    """Render downsampled mosaics for visual inspection only."""
    fw = int(round(W_full * CONFIG["RENDER_SCALE"]))
    fh = int(round(H_full * CONFIG["RENDER_SCALE"]))
    xs = (geom_xy[:, 0] * CONFIG["RENDER_SCALE"]).astype(int)
    ys = (geom_xy[:, 1] * CONFIG["RENDER_SCALE"]).astype(int)
    xs -= xs.min()
    ys -= ys.min()
    canvas_w = int(xs.max() + fw)
    y_max = int(ys.max() + fh)
    chunk_h = int(round(CONFIG["CHUNK_HEIGHT_FULL"] * CONFIG["RENDER_SCALE"]))
    if chunk_h < fh + 10:
        chunk_h = fh + 10

    def blend_overwrite(canvas, mask, img, x, y, feather=30):
        H, W = img.shape[:2]
        ch, cw = canvas.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(cw, x + W)
        y1 = min(ch, y + H)
        if x0 >= x1 or y0 >= y1:
            return
        ix0 = x0 - x
        iy0 = y0 - y
        ix1 = ix0 + (x1 - x0)
        iy1 = iy0 + (y1 - y0)
        roi_c = canvas[y0:y1, x0:x1]
        roi_m = mask[y0:y1, x0:x1]
        roi_i = img[iy0:iy1, ix0:ix1]
        overlap = roi_m > 0
        if np.any(overlap):
            ys_ = np.where(overlap)[0]
            ov_min, ov_max = (ys_.min(), ys_.max() + 1)
            bh = min(feather, ov_max - ov_min)
            if bh > 0:
                alpha = np.linspace(0, 1, bh, endpoint=False, dtype=np.float32)[
                    :, None, None
                ]
                band = slice(ov_min, ov_min + bh)
                roi_c[band] = (
                    roi_c[band].astype(np.float32) * (1 - alpha)
                    + roi_i[band].astype(np.float32) * alpha
                ).astype(np.uint8)
            roi_c[ov_min + bh : ov_max] = roi_i[ov_min + bh : ov_max]
            roi_c[~overlap] = roi_i[~overlap]
        else:
            roi_c[:] = roi_i
        roi_m[:] = 255

    chunks = []
    chunk_id = 0
    for y0 in range(0, y_max + 1, chunk_h):
        y1 = y0 + chunk_h
        idxs = [t for t, y in enumerate(ys) if y < y1 and y + fh > y0]
        if not idxs:
            continue
        canvas = np.zeros((chunk_h + fh + 10, canvas_w, 3), dtype=np.uint8)
        mask = np.zeros((chunk_h + fh + 10, canvas_w), dtype=np.uint8)
        used_h = 0
        for t in idxs:
            img = rotate_img(
                read_image(str(files[geom_idx[t]]), cv2.IMREAD_COLOR),
                CONFIG["ROTATE_DEG"],
            )
            img = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
            x = int(xs[t])
            y = int(ys[t] - y0)
            blend_overwrite(canvas, mask, img, x, y, feather=CONFIG["FEATHER_PX"])
            used_h = max(used_h, y + fh)
        out = canvas[:used_h]
        out_path = out_dir / f"mosaic_preview_{chunk_id:04d}.png"
        write_image(str(out_path), out)
        chunks.append(
            {"file": out_path.name, "y0_scaled": int(y0), "y1_scaled": int(y0 + used_h)}
        )
        chunk_id += 1
    return chunks


def run_pipeline():
    """Construct the trajectory and export source keyframes and diagnostics."""
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_images(CONFIG["INPUT_DIR"])
    first = rotate_img(
        read_image(str(files[0]), cv2.IMREAD_COLOR), CONFIG["ROTATE_DEG"]
    )
    H_full, W_full = first.shape[:2]
    max_dy_full = float(CONFIG["MAX_DY_FRAC"] * H_full)
    q_det = []
    q_tex = []
    for p in files:
        img = rotate_img(read_image(str(p), cv2.IMREAD_COLOR), CONFIG["ROTATE_DEG"])
        check_frame_shape(img, (H_full, W_full), p)
        q_det.append(detection_quality(img, scale=0.25))
        q_tex.append(texture_score_for_geom(img, scale=0.25))
    q_det = np.array(q_det, dtype=np.float32)
    q_tex = np.array(q_tex, dtype=np.float32)
    tmp_steps = []
    prev = first
    for i in range(1, min(len(files), 30)):
        cur = rotate_img(
            read_image(str(files[i]), cv2.IMREAD_COLOR), CONFIG["ROTATE_DEG"]
        )
        dx, dy, resp, method = estimate_shift(
            prev, cur, CONFIG["REG_SCALE"], pred_sy_full=None, sign_y=1.0, H_full=H_full
        )
        if resp >= CONFIG["PHASE_MIN_RESP"]:
            tmp_steps.append(-dy)
        prev = cur
    median_step = float(np.median(tmp_steps)) if len(tmp_steps) >= 5 else 1.0
    sign_y = 1.0 if median_step >= 0 else -1.0
    rows = []
    xg, yg = (0.0, 0.0)
    prev = first
    sy_hist = []
    rows.append(
        {
            "idx": 0,
            "file": files[0].name,
            "x_coarse": xg,
            "y_coarse": yg,
            "dx": 0.0,
            "dy": 0.0,
            "resp": 1.0,
            "method": "init",
            "sy": 0.0,
            "sy_abs": 0.0,
            "q_det": float(q_det[0]),
            "q_tex": float(q_tex[0]),
        }
    )
    for i in range(1, len(files)):
        cur = rotate_img(
            read_image(str(files[i]), cv2.IMREAD_COLOR), CONFIG["ROTATE_DEG"]
        )
        pred_sy = (
            float(np.median(sy_hist))
            if len(sy_hist) >= 8
            else (1.0 - CONFIG["KEY_OVERLAP_TARGET"]) * H_full
        )
        dx, dy, resp, method = estimate_shift(
            prev,
            cur,
            CONFIG["REG_SCALE"],
            pred_sy_full=pred_sy,
            sign_y=sign_y,
            H_full=H_full,
        )
        sy_test = sign_y * -dy
        if method == "fail" or (
            CONFIG["VALIDATE_FRAME_SHIFT"] and sy_test < CONFIG["MIN_FRAME_FORWARD_PX"]
        ):
            dx = 0.0
            dy = -(pred_sy / sign_y)
            resp = 0.0
            method = "pred"
        dx = float(np.clip(dx, -CONFIG["DX_CLIP"], CONFIG["DX_CLIP"]))
        dy = float(np.clip(dy, -max_dy_full, max_dy_full))
        sx = -dx
        sy = sign_y * -dy
        xg += sx
        yg += sy
        sy_hist.append(abs(sy))
        if len(sy_hist) > 30:
            sy_hist = sy_hist[-30:]
        rows.append(
            {
                "idx": i,
                "file": files[i].name,
                "x_coarse": xg,
                "y_coarse": yg,
                "dx": float(dx),
                "dy": float(dy),
                "resp": float(resp),
                "method": method,
                "sy": float(sy),
                "sy_abs": float(abs(sy)),
                "q_det": float(q_det[i]),
                "q_tex": float(q_tex[i]),
            }
        )
        prev = cur
    poses = pd.DataFrame(rows)
    poses.to_csv(out_dir / "poses_all_coarse.csv", index=False, encoding="utf-8-sig")
    y_coarse = poses["y_coarse"].values.astype(np.float64)
    x_coarse = poses["x_coarse"].values.astype(np.float64)
    geom_idx = select_geom_keyframes(files, y_coarse, q_det, q_tex, H_full)
    anchor_xy = np.zeros((len(geom_idx), 2), dtype=np.float64)
    anchor_xy[0] = np.array(
        [x_coarse[geom_idx[0]], y_coarse[geom_idx[0]]], dtype=np.float64
    )
    ref_rows = []
    for k in range(1, len(geom_idx)):
        i0, i1 = (geom_idx[k - 1], geom_idx[k])
        img0 = rotate_img(
            read_image(str(files[i0]), cv2.IMREAD_COLOR), CONFIG["ROTATE_DEG"]
        )
        img1 = rotate_img(
            read_image(str(files[i1]), cv2.IMREAD_COLOR), CONFIG["ROTATE_DEG"]
        )
        pred_sy = float(y_coarse[i1] - y_coarse[i0])
        dx, dy, resp, method = estimate_shift(
            img0,
            img1,
            CONFIG["REG_SCALE"],
            pred_sy_full=pred_sy,
            sign_y=sign_y,
            H_full=H_full,
        )
        pred_sy_anchor = float(y_coarse[i1] - y_coarse[i0])
        valid_anchor, sx, sy, invalid_reason = validate_anchor_shift(
            dx, dy, pred_sy_anchor, sign_y, H_full
        )
        if method == "fail" or (CONFIG["VALIDATE_ANCHOR_SHIFT"] and (not valid_anchor)):
            dx_raw = dx
            dy_raw = dy
            method_raw = method
            dx = -(x_coarse[i1] - x_coarse[i0])
            dy = -((y_coarse[i1] - y_coarse[i0]) / sign_y)
            resp = 0.0 if method == "fail" else float(resp)
            method = (
                "fallback_coarse"
                if method == "fail"
                else "fallback_coarse_invalid_anchor"
            )
            sx = -dx
            sy = sign_y * -dy
        else:
            dx_raw = dx
            dy_raw = dy
            method_raw = method
        anchor_xy[k, 0] = anchor_xy[k - 1, 0] + sx
        anchor_xy[k, 1] = anchor_xy[k - 1, 1] + sy
        ref_rows.append(
            {
                "k": k,
                "prev_idx": i0,
                "prev_file": files[i0].name,
                "curr_idx": i1,
                "curr_file": files[i1].name,
                "dx": float(dx),
                "dy": float(dy),
                "sx": float(sx),
                "sy": float(sy),
                "pred_sy": float(pred_sy_anchor),
                "resp": float(resp),
                "method": method,
                "raw_dx": float(dx_raw),
                "raw_dy": float(dy_raw),
                "raw_method": method_raw,
                "invalid_reason": invalid_reason,
            }
        )
    pd.DataFrame(ref_rows).to_csv(
        out_dir / "anchors_refined.csv", index=False, encoding="utf-8-sig"
    )
    coarse_xy = np.stack([x_coarse, y_coarse], axis=1)
    corrected_xy = anchor_correct_all(coarse_xy, geom_idx, anchor_xy)
    poses["x_corrected"] = corrected_xy[:, 0]
    poses["y_corrected"] = corrected_xy[:, 1]
    poses.to_csv(out_dir / "poses_all_corrected.csv", index=False, encoding="utf-8-sig")
    check_monotonic_sequence(corrected_xy, "poses_corrected", out_dir, files=files)
    det_idx = []
    for k in range(len(geom_idx) - 1):
        a = geom_idx[k]
        b = geom_idx[k + 1]
        seg = list(range(a, b + 1))
        if not CONFIG["USE_CONTINUITY_AWARE_DET_KEYFRAMES"]:
            j = max(seg, key=lambda t: float(q_det[t]))
        else:
            y_a = float(corrected_xy[a, 1])
            y_b = float(corrected_xy[b, 1])
            y_lo, y_hi = (min(y_a, y_b), max(y_a, y_b))
            y_mid = 0.5 * (y_a + y_b)
            x_mid = float(np.median(corrected_xy[seg, 0]))

            def det_score(t):
                q = float(q_det[t])
                x = float(corrected_xy[t, 0])
                y = float(corrected_xy[t, 1])
                score = q
                if y < y_lo - 1e-06 or y > y_hi + 1e-06:
                    score -= 100.0
                score -= 0.0015 * abs(y - y_mid)
                score -= 0.0008 * abs(x - x_mid)
                method_t = str(poses.loc[t, "method"])
                resp_t = float(poses.loc[t, "resp"])
                sy_t = float(poses.loc[t, "sy"])
                if method_t in ["fail", "pred", "phase_lowresp"]:
                    score -= 0.35
                if resp_t < CONFIG["PHASE_MIN_RESP"]:
                    score -= 0.25
                if sy_t < CONFIG["MIN_FRAME_FORWARD_PX"]:
                    score -= 1.0
                return score

            j = max(seg, key=det_score)
        det_idx.append(j)
    det_idx.append(geom_idx[-1])
    det_idx = sorted(set(det_idx))
    geom_rows = []
    for kk, idx in enumerate(geom_idx):
        geom_rows.append(
            {
                "k": kk,
                "idx": int(idx),
                "file": files[idx].name,
                "x_global_px": float(corrected_xy[idx, 0]),
                "y_global_px": float(corrected_xy[idx, 1]),
                "q_det": float(q_det[idx]),
                "q_tex": float(q_tex[idx]),
                "img_w": int(W_full),
                "img_h": int(H_full),
            }
        )
    geom_df = pd.DataFrame(geom_rows)
    geom_df.to_csv(out_dir / "geom_keyframes.csv", index=False, encoding="utf-8-sig")
    det_rows = []
    for kk, idx in enumerate(det_idx):
        det_rows.append(
            {
                "k": kk,
                "idx": int(idx),
                "file": files[idx].name,
                "x_global_px": float(corrected_xy[idx, 0]),
                "y_global_px": float(corrected_xy[idx, 1]),
                "q_det": float(q_det[idx]),
                "img_w": int(W_full),
                "img_h": int(H_full),
            }
        )
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(out_dir / "keyframes.csv", index=False, encoding="utf-8-sig")
    det_gaps = []
    for kk in range(1, len(det_idx)):
        i0, i1 = (det_idx[kk - 1], det_idx[kk])
        dx_det = float(corrected_xy[i1, 0] - corrected_xy[i0, 0])
        dy_det = float(corrected_xy[i1, 1] - corrected_xy[i0, 1])
        det_gaps.append(
            {
                "k": kk,
                "prev_idx": int(i0),
                "curr_idx": int(i1),
                "dx_px": dx_det,
                "dy_px": dy_det,
                "overlap_est_y": 1.0 - dy_det / float(H_full),
                "prev_file": files[i0].name,
                "curr_file": files[i1].name,
            }
        )
    pd.DataFrame(det_gaps).to_csv(
        out_dir / "report_det_keyframe_gaps.csv", index=False, encoding="utf-8-sig"
    )
    if CONFIG["EXPORT_KEYFRAME_IMAGES"]:
        det_dir = out_dir / CONFIG["KEYFRAMES_DIRNAME"]
        geom_dir = out_dir / CONFIG["GEOM_KEYFRAMES_DIRNAME"]
        export_keyframe_images(
            files,
            det_idx,
            det_dir,
            CONFIG["ROTATE_DEG"],
            save_rotated=CONFIG["SAVE_ROTATED_KEYFRAMES"],
        )
        export_keyframe_images(
            files,
            geom_idx,
            geom_dir,
            CONFIG["ROTATE_DEG"],
            save_rotated=CONFIG["SAVE_ROTATED_KEYFRAMES"],
        )
    gaps = []
    for k in range(1, len(geom_idx)):
        i0, i1 = (geom_idx[k - 1], geom_idx[k])
        dy = float(corrected_xy[i1, 1] - corrected_xy[i0, 1])
        overlap = 1.0 - dy / float(H_full)
        gaps.append(
            {
                "k": k,
                "prev_idx": i0,
                "curr_idx": i1,
                "dy_px": dy,
                "overlap_est": overlap,
                "prev_file": files[i0].name,
                "curr_file": files[i1].name,
            }
        )
    pd.DataFrame(gaps).to_csv(
        out_dir / "report_keyframe_gaps.csv", index=False, encoding="utf-8-sig"
    )
    chunks = []
    if CONFIG["RENDER_PREVIEW"]:
        geom_xy = geom_df[["x_global_px", "y_global_px"]].values.astype(np.float64)
        chunks = render_preview(files, geom_idx, geom_xy, out_dir, W_full, H_full)
    det_chunks = []
    if CONFIG["RENDER_PREVIEW"] and CONFIG["EXPORT_DET_KEYFRAME_PREVIEW"]:
        det_preview_dir = out_dir / "det_keyframes_preview"
        det_preview_dir.mkdir(parents=True, exist_ok=True)
        det_xy = det_df[["x_global_px", "y_global_px"]].values.astype(np.float64)
        det_chunks = render_preview(
            files, det_idx, det_xy, det_preview_dir, W_full, H_full
        )
    meta = {
        "num_frames": len(files),
        "H_full": int(H_full),
        "W_full": int(W_full),
        "key_overlap_target": float(CONFIG["KEY_OVERLAP_TARGET"]),
        "min_overlap_guard": float(CONFIG["MIN_OVERLAP_GUARD"]),
        "reg_scale": float(CONFIG["REG_SCALE"]),
        "phase_min_resp": float(CONFIG["PHASE_MIN_RESP"]),
        "sign_y": float(sign_y),
        "render_preview": bool(CONFIG["RENDER_PREVIEW"]),
        "chunks": chunks,
        "det_preview_chunks": det_chunks,
        "export_keyframe_images": bool(CONFIG["EXPORT_KEYFRAME_IMAGES"]),
        "save_rotated_keyframes": bool(CONFIG["SAVE_ROTATED_KEYFRAMES"]),
        "keyframes_dir": (
            CONFIG["KEYFRAMES_DIRNAME"] if CONFIG["EXPORT_KEYFRAME_IMAGES"] else None
        ),
        "geom_keyframes_dir": (
            CONFIG["GEOM_KEYFRAMES_DIRNAME"]
            if CONFIG["EXPORT_KEYFRAME_IMAGES"]
            else None
        ),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[DONE]")
    print("Saved:")
    print(" - poses_all_coarse.csv")
    print(" - poses_all_corrected.csv")
    print(" - geom_keyframes.csv")
    print(" - keyframes.csv   (for detection)")
    print(" - anchors_refined.csv")
    print(" - report_keyframe_gaps.csv (check overlap)")
    if CONFIG["RENDER_PREVIEW"]:
        print(" - mosaic_preview_*.png")
        if CONFIG["EXPORT_DET_KEYFRAME_PREVIEW"]:
            print(" - det_keyframes_preview/mosaic_preview_*.png")
    if CONFIG["EXPORT_KEYFRAME_IMAGES"]:
        print(f" - {CONFIG['KEYFRAMES_DIRNAME']}/ (det keyframe images)")
        print(f" - {CONFIG['GEOM_KEYFRAMES_DIRNAME']}/ (geom keyframe images)")
    return {
        "direction_sample_count": len(tmp_steps),
        "direction_median_step": median_step,
    }


def validate_config(config):
    """Validate execution settings without changing registration thresholds."""
    if config["ROTATE_DEG"] not in (0, 90, 180, 270):
        raise ValueError("ROTATE_DEG must be 0, 90, 180, or 270")
    for name in ("REG_SCALE", "RENDER_SCALE"):
        if not 0 < config[name] <= 1:
            raise ValueError(name + " must be in (0, 1]")
    if not 0 <= config["MARGIN_RATIO"] < 0.5:
        raise ValueError("MARGIN_RATIO must be in [0, 0.5)")
    if not 0 < config["MIN_OVERLAP_GUARD"] <= config["KEY_OVERLAP_TARGET"] < 1:
        raise ValueError("Require 0 < MIN_OVERLAP_GUARD <= KEY_OVERLAP_TARGET < 1")
    for name in ("SEARCH_DX", "SEARCH_DY", "FEATHER_PX"):
        if not isinstance(config[name], int) or config[name] < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if not isinstance(config["CAND_MAX"], int) or config["CAND_MAX"] < 1:
        raise ValueError("CAND_MAX must be a positive integer")
    if config["CHUNK_HEIGHT_FULL"] <= 0:
        raise ValueError("CHUNK_HEIGHT_FULL must be positive")
    if (
        not 1 <= config["JPEG_QUALITY"] <= 100
        or not 0 <= config["PNG_COMPRESSION"] <= 9
    ):
        raise ValueError("Invalid image encoder settings")
    for name in ("KEYFRAMES_DIRNAME", "GEOM_KEYFRAMES_DIRNAME"):
        value = config[name]
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(name + " must be a single directory name")
    if config["KEYFRAMES_DIRNAME"] == config["GEOM_KEYFRAMES_DIRNAME"]:
        raise ValueError("Detection and geometry image directories must differ")
    if (
        config["EXPORT_KEYFRAME_IMAGES"]
        and config["ROTATE_DEG"]
        and not config["SAVE_ROTATED_KEYFRAMES"]
    ):
        raise ValueError(
            "Export rotated keyframes when ROTATE_DEG is nonzero, so images match the coordinate tables"
        )


def save_json(path, data):
    def convert(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError("Cannot serialize {}".format(type(value).__name__))

    Path(path).write_text(
        json.dumps(data, indent=2, default=convert, allow_nan=False), encoding="utf-8"
    )


def summarize_run(out_dir, pipeline_info):
    """Summarize fallback use and coverage without altering exported coordinates."""
    poses = pd.read_csv(out_dir / "poses_all_corrected.csv")
    anchors = pd.read_csv(out_dir / "anchors_refined.csv")
    geometry = pd.read_csv(out_dir / "geom_keyframes.csv")
    detection = pd.read_csv(out_dir / "keyframes.csv")
    geom_gaps = pd.read_csv(out_dir / "report_keyframe_gaps.csv")
    try:
        det_gaps = pd.read_csv(out_dir / "report_det_keyframe_gaps.csv")
    except pd.errors.EmptyDataError:
        det_gaps = pd.DataFrame(columns=["overlap_est_y"])
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    flags = []
    predicted = int(poses.method.eq("pred").sum())
    coarse_anchors = (
        int(anchors.method.str.startswith("fallback_coarse").sum())
        if len(anchors)
        else 0
    )
    nonforward = int(np.count_nonzero(np.diff(poses.y_corrected.to_numpy()) <= 0))
    weak_geom = int((geom_gaps.overlap_est < CONFIG["MIN_OVERLAP_GUARD"]).sum())
    uncovered_det = int((det_gaps.overlap_est_y <= 0).sum()) if len(det_gaps) else 0
    if predicted:
        flags.append("predicted_frame_steps")
    if coarse_anchors:
        flags.append("coarse_anchor_fallbacks")
    if nonforward:
        flags.append("nonforward_corrected_steps")
    if weak_geom:
        flags.append("geometry_overlap_below_guard")
    if uncovered_det:
        flags.append("detection_keyframe_gap")
    if len(detection) < 2:
        flags.append("insufficient_detection_keyframes_for_seam_core")
    if pipeline_info["direction_sample_count"] < 5:
        flags.append("insufficient_direction_samples")
    if meta["sign_y"] < 0:
        flags.append("negative_direction_requires_orientation_review")
    return dict(
        num_frames=len(poses),
        geometry_keyframes=len(geometry),
        detection_keyframes=len(detection),
        predicted_frame_steps=predicted,
        coarse_anchor_fallbacks=coarse_anchors,
        nonforward_corrected_steps=nonforward,
        geometry_overlap_below_guard=weak_geom,
        detection_keyframe_gaps=uncovered_det,
        sign_y=meta["sign_y"],
        review_reasons=flags,
        **pipeline_info,
    )


def run(input_dir=None, output_dir=None, overrides=None):
    """Run one image sequence. A nonempty destination receives a new run directory.

    Configuration is process-local; invoke separate processes for parallel runs.
    """
    effective = dict(DEFAULT_CONFIG)
    if overrides:
        unknown = set(overrides) - set(effective)
        if unknown:
            raise ValueError("Unknown settings: {}".format(", ".join(sorted(unknown))))
        effective.update(overrides)
    if input_dir is not None:
        effective["INPUT_DIR"] = str(input_dir)
    if output_dir is not None:
        effective["OUTPUT_DIR"] = str(output_dir)
    validate_config(effective)
    source = Path(effective["INPUT_DIR"]).resolve()
    if not source.is_dir():
        raise NotADirectoryError("Input directory does not exist: {}".format(source))
    destination = Path(effective["OUTPUT_DIR"]).resolve()
    if source == destination:
        raise ValueError("Input and output directories must differ")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(
            "Output path is not a directory: {}".format(destination)
        )
    if destination.exists() and any(destination.iterdir()):
        destination = destination / (
            "run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    destination.mkdir(parents=True, exist_ok=True)
    effective.update(INPUT_DIR=str(source), OUTPUT_DIR=str(destination))
    CONFIG.clear()
    CONFIG.update(effective)
    status = dict(
        status="running",
        input_dir=str(source),
        output_dir=str(destination),
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        pandas=pd.__version__,
        opencv=cv2.__version__,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(
            Path(__file__).with_name("registration_config.py").read_bytes()
        ).hexdigest(),
        coordinate_model="translation_only",
        keyframe_source="rotated_source_frames",
    )
    save_json(destination / "run_status.json", status)
    save_json(destination / "run_config.json", effective)
    print("[OUTPUT] {}".format(destination), flush=True)
    try:
        info = run_pipeline()
        summary = summarize_run(destination, info)
        status.update(
            status=(
                "completed_needs_review" if summary["review_reasons"] else "completed"
            ),
            **summary,
        )
        save_json(
            destination / "registration_handoff.json",
            dict(
                keyframes_csv="keyframes.csv",
                keyframes_images=(
                    effective["KEYFRAMES_DIRNAME"]
                    if effective["EXPORT_KEYFRAME_IMAGES"]
                    else None
                ),
                geometry_keyframes_csv="geom_keyframes.csv",
                geometry_images=(
                    effective["GEOM_KEYFRAMES_DIRNAME"]
                    if effective["EXPORT_KEYFRAME_IMAGES"]
                    else None
                ),
                keyframe_images_are_rotated=bool(effective["SAVE_ROTATED_KEYFRAMES"]),
                keyframe_images_are_mosaic_crops=False,
                coordinate_model="translation_only",
                coordinate_units="pixels",
                review_reasons=summary["review_reasons"],
            ),
        )
    except Exception as exc:
        status.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
        raise
    finally:
        status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(destination / "run_status.json", status)
    print("[STATUS] {}".format(status["status"]), flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Construct strip coordinates and export original keyframes"
    )
    parser.add_argument("--input", help="Directory of naturally ordered source frames")
    parser.add_argument("--output", help="Registration output directory")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        help="Clockwise source-image rotation",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Skip inspection mosaics"
    )
    parser.add_argument(
        "--no-keyframe-export",
        action="store_true",
        help="Export coordinate tables without source keyframe images",
    )
    args = parser.parse_args(argv)
    overrides = {}
    if args.rotate is not None:
        overrides["ROTATE_DEG"] = args.rotate
    if args.no_preview:
        overrides["RENDER_PREVIEW"] = False
    if args.no_keyframe_export:
        overrides["EXPORT_KEYFRAME_IMAGES"] = False
    return run(args.input, args.output, overrides)


if __name__ == "__main__":
    main()
