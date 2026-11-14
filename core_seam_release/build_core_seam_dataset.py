"""Build core and seam detection samples from registered source keyframes."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from core_seam_config import BUILD_CONFIG as DEFAULT_CONFIG
from core_seam_io import (
    read_image,
    write_image,
    start_run,
    finish_run,
    validate_keyframes,
    validate_metadata,
    validate_detection_join,
)

CONFIG = dict(DEFAULT_CONFIG)
import cv2


def ensure_dir(p: Path) -> Path:
    """Create a destination directory."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def natural_col(
    df: pd.DataFrame, candidates: List[str], fallback_idx: Optional[int] = None
) -> str:
    """Select a supported column name, with positional fallback for existing tables."""
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    for c in candidates:
        if c in cols:
            return c
    if fallback_idx is not None and fallback_idx < len(cols):
        return cols[fallback_idx]
    raise KeyError(f"Cannot find any of columns: {candidates}")


def resolve_img_dir(keyframe_csv: str, keyframe_img_dir: Optional[str]) -> Path:
    """Locate the source keyframe directory."""
    if keyframe_img_dir:
        p = Path(keyframe_img_dir)
    else:
        p = Path(keyframe_csv).resolve().parent / "keyframes_images"
    if not p.exists():
        raise FileNotFoundError(f"KEYFRAME_IMG_DIR not found: {p}")
    return p


def estimate_strip_axes_from_frame_centers(points_xy: np.ndarray, hint: str = "y"):
    """Estimate strip axes from the distribution of source-frame centers."""
    origin = np.median(points_xy, axis=0)
    X = points_xy - origin
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    e_along = eigvecs[:, np.argsort(eigvals)[::-1][0]]
    e_along = e_along / (np.linalg.norm(e_along) + 1e-12)
    if hint == "y":
        if e_along[1] < 0:
            e_along = -e_along
    elif e_along[0] < 0:
        e_along = -e_along
    e_cross = np.array([-e_along[1], e_along[0]], dtype=float)
    e_cross = e_cross / (np.linalg.norm(e_cross) + 1e-12)
    return (origin, e_along, e_cross)


def project_points(
    points_xy: np.ndarray, origin: np.ndarray, e_along: np.ndarray, e_cross: np.ndarray
):
    """Project global coordinates onto the along-strip and transverse axes."""
    X = points_xy - origin
    s = X @ e_along
    t = X @ e_cross
    return (s, t)


def frame_corner_s_range(
    fx: float, fy: float, w: float, h: float, origin: np.ndarray, e_along: np.ndarray
):
    """Project the four image corners onto the along-strip axis."""
    corners = np.array(
        [[fx, fy], [fx + w, fy], [fx, fy + h], [fx + w, fy + h]], dtype=float
    )
    s = (corners - origin[None, :]) @ e_along
    return (float(s.min()), float(s.max()))


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the smallest pixel bounding box containing a nonempty mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def build_local_projection_maps(
    w: int, h: int, e_along: np.ndarray, e_cross: np.ndarray
):
    """Map source pixels to local strip coordinates."""
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    s_local = xx * float(e_along[0]) + yy * float(e_along[1])
    t_local = xx * float(e_cross[0]) + yy * float(e_cross[1])
    return (s_local.astype(np.float32), t_local.astype(np.float32))


def interval_mask_from_s(s_map: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Select pixels within an inclusive longitudinal interval."""
    m = np.ones_like(s_map, dtype=bool)
    if not math.isinf(lo):
        m &= s_map >= lo
    if not math.isinf(hi):
        m &= s_map <= hi
    return m


def apply_mask_and_crop(
    img: np.ndarray, mask: np.ndarray, fill_outside: bool = True, fill_value=(0, 0, 0)
):
    """Extract a context crop and optionally fill pixels outside its mask."""
    bbox = bbox_from_mask(mask.astype(np.uint8))
    if bbox is None:
        return (None, None)
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if fill_outside:
        bg = np.zeros_like(crop)
        bg[:, :] = np.array(fill_value, dtype=np.uint8)
        crop = np.where(crop_mask[:, :, None], crop, bg)
    return (crop, (x1, y1, x2, y2, crop_mask))


def blend_two_masks_to_canvas(
    canvas_h: int,
    canvas_w: int,
    g_x0: int,
    g_y0: int,
    piece_a: Optional[Dict],
    piece_b: Optional[Dict],
    sA: float,
    sB: float,
    feather: float,
):
    """Blend the two source patches using their strip-coordinate weights."""
    sum_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    sum_w = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    def paste(piece: Dict, side: str):
        nonlocal sum_img, sum_w
        img = piece["img"].astype(np.float32)
        mask = piece["mask"]
        s_global = piece["s_global"]
        gx1, gy1, _, _ = piece["global_bbox"]
        gh, gw = img.shape[:2]
        ox = int(round(gx1 - g_x0))
        oy = int(round(gy1 - g_y0))
        yy0 = max(0, oy)
        xx0 = max(0, ox)
        yy1 = min(canvas_h, oy + gh)
        xx1 = min(canvas_w, ox + gw)
        if yy0 >= yy1 or xx0 >= xx1:
            return
        iy0 = yy0 - oy
        ix0 = xx0 - ox
        iy1 = iy0 + (yy1 - yy0)
        ix1 = ix0 + (xx1 - xx0)
        roi_img = img[iy0:iy1, ix0:ix1]
        roi_mask = mask[iy0:iy1, ix0:ix1]
        roi_s = s_global[iy0:iy1, ix0:ix1]
        denom = max(abs(sB - sA), 1e-06)
        if side == "A":
            w = (sB - roi_s) / denom
        else:
            w = (roi_s - sA) / denom
        w = np.clip(w, 0.0, 1.0)
        if feather > 0:
            w = np.clip((w - 0.5) * (denom / max(feather, 1e-06)) + 0.5, 0.0, 1.0)
        w = w * roi_mask.astype(np.float32)
        sum_img[yy0:yy1, xx0:xx1] += roi_img * w[:, :, None]
        sum_w[yy0:yy1, xx0:xx1] += w

    if piece_a is not None:
        paste(piece_a, "A")
    if piece_b is not None:
        paste(piece_b, "B")
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    valid = sum_w > 1e-06
    out[valid] = np.clip(sum_img[valid] / sum_w[valid, None], 0, 255).astype(np.uint8)
    return (out, valid.astype(np.uint8))


def run_pipeline():
    """Construct source-based core and pairwise seam samples."""
    cfg = CONFIG
    out_dir = ensure_dir(Path(cfg["OUT_DIR"]))
    core_dir = ensure_dir(out_dir / "core_crops")
    seam_dir = ensure_dir(out_dir / "seam_crops")
    df = pd.read_csv(cfg["KEYFRAME_CSV"])
    file_col = natural_col(df, ["file", "frame_id", "image"], 0)
    k_col = natural_col(df, ["k", "frame_order", "order", "idx"], 0)
    x_col = natural_col(df, ["x_global_px", "x_global", "global_x", "gx"], 1)
    y_col = natural_col(df, ["y_global_px", "y_global", "global_y", "gy"], 2)
    w_col = natural_col(df, ["img_w", "width", "w"], 3)
    h_col = natural_col(df, ["img_h", "height", "h"], 4)
    df = (
        pd.DataFrame(
            {
                "k": df[k_col].astype(int),
                "file": df[file_col].astype(str),
                "fx1": df[x_col].astype(float),
                "fy1": df[y_col].astype(float),
                "img_w": df[w_col].astype(int),
                "img_h": df[h_col].astype(int),
            }
        )
        .sort_values("k")
        .reset_index(drop=True)
    )
    df["fx2"] = df["fx1"] + df["img_w"]
    df["fy2"] = df["fy1"] + df["img_h"]
    df["cxg"] = df["fx1"] + 0.5 * df["img_w"]
    df["cyg"] = df["fy1"] + 0.5 * df["img_h"]
    validate_keyframes(df)
    img_dir = resolve_img_dir(cfg["KEYFRAME_CSV"], cfg.get("KEYFRAME_IMG_DIR"))
    first_img = read_image(str(img_dir / df.iloc[0]["file"]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise FileNotFoundError(
            f"Cannot read first keyframe image: {img_dir / df.iloc[0]['file']}"
        )
    H, W = first_img.shape[:2]
    validate_keyframes(df, (H, W))
    centers = df[["cxg", "cyg"]].to_numpy(dtype=float)
    origin, e_along, e_cross = estimate_strip_axes_from_frame_centers(
        centers, cfg["STRIP_AXIS_HINT"]
    )
    center_s, center_t = project_points(centers, origin, e_along, e_cross)
    df["center_s"] = center_s
    df["center_t"] = center_t
    s_local, t_local = build_local_projection_maps(W, H, e_along, e_cross)
    seams = []
    for i in range(len(df) - 1):
        s_mid = 0.5 * (
            float(df.iloc[i]["center_s"]) + float(df.iloc[i + 1]["center_s"])
        )
        seams.append(s_mid)
    seams = np.asarray(seams, dtype=float)
    core_rows = []
    seam_rows = []
    strip_meta = {
        "origin": [float(origin[0]), float(origin[1])],
        "e_along": [float(e_along[0]), float(e_along[1])],
        "e_cross": [float(e_cross[0]), float(e_cross[1])],
        "seam_half_band": float(cfg["SEAM_HALF_BAND"]),
        "core_context": float(cfg["CORE_CONTEXT"]),
        "seam_context": float(cfg["SEAM_CONTEXT"]),
        "image_w": int(W),
        "image_h": int(H),
    }
    with open(out_dir / "strip_meta.json", "w", encoding="utf-8") as f:
        json.dump(strip_meta, f, ensure_ascii=False, indent=2)
    for i, row in df.iterrows():
        file = str(row["file"])
        img = read_image(str(img_dir / file), cv2.IMREAD_COLOR, expected=(H, W))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_dir / file}")
        s_offset = float(np.dot(np.array([row["fx1"], row["fy1"]]) - origin, e_along))
        s_map = s_local + s_offset
        left_mid = float(seams[i - 1]) if i > 0 else -math.inf
        right_mid = float(seams[i]) if i < len(df) - 1 else math.inf
        local_band = float(cfg["SEAM_HALF_BAND"])
        if i > 0 and i < len(df) - 1:
            avail = max(0.0, right_mid - left_mid)
            local_band = min(local_band, 0.45 * avail)
        core_left = left_mid + local_band if not math.isinf(left_mid) else -math.inf
        core_right = right_mid - local_band if not math.isinf(right_mid) else math.inf
        ctx_left = (
            core_left - float(cfg["CORE_CONTEXT"])
            if not math.isinf(core_left)
            else -math.inf
        )
        ctx_right = (
            core_right + float(cfg["CORE_CONTEXT"])
            if not math.isinf(core_right)
            else math.inf
        )
        mask_ctx = interval_mask_from_s(s_map, ctx_left, ctx_right)
        crop, info = apply_mask_and_crop(
            img,
            mask_ctx,
            fill_outside=bool(cfg["FILL_OUTSIDE"]),
            fill_value=tuple(cfg["FILL_VALUE"]),
        )
        if crop is None:
            continue
        x1, y1, x2, y2, crop_mask = info
        crop_name = f"core_k{i:04d}_{Path(file).stem}.png"
        write_image(core_dir / crop_name, crop)
        core_rows.append(
            {
                "k": int(row["k"]),
                "src_file": file,
                "crop_file": crop_name,
                "frame_fx1": float(row["fx1"]),
                "frame_fy1": float(row["fy1"]),
                "crop_x1_local": int(x1),
                "crop_y1_local": int(y1),
                "crop_w": int(x2 - x1),
                "crop_h": int(y2 - y1),
                "crop_global_x1": float(row["fx1"] + x1),
                "crop_global_y1": float(row["fy1"] + y1),
                "core_left_s": float(core_left),
                "core_right_s": float(core_right),
                "ctx_left_s": float(ctx_left),
                "ctx_right_s": float(ctx_right),
                "center_s": float(row["center_s"]),
                "center_t": float(row["center_t"]),
            }
        )
    for i in range(len(df) - 1):
        a = df.iloc[i]
        b = df.iloc[i + 1]
        file_a = str(a["file"])
        file_b = str(b["file"])
        img_a = read_image(str(img_dir / file_a), cv2.IMREAD_COLOR, expected=(H, W))
        img_b = read_image(str(img_dir / file_b), cv2.IMREAD_COLOR, expected=(H, W))
        if img_a is None or img_b is None:
            raise FileNotFoundError(f"Cannot read seam images: {file_a}, {file_b}")
        sA = float(a["center_s"])
        sB = float(b["center_s"])
        m = float(seams[i])
        band = float(cfg["SEAM_HALF_BAND"])
        band = min(band, 0.45 * max(abs(sB - sA), 1.0))
        seam_left = m - band
        seam_right = m + band
        ctx_left = seam_left - float(cfg["SEAM_CONTEXT"])
        ctx_right = seam_right + float(cfg["SEAM_CONTEXT"])
        pieces = []
        for tag, r, img in [("A", a, img_a), ("B", b, img_b)]:
            s_offset = float(np.dot(np.array([r["fx1"], r["fy1"]]) - origin, e_along))
            s_map = s_local + s_offset
            mask_ctx = interval_mask_from_s(s_map, ctx_left, ctx_right)
            ap = apply_mask_and_crop(
                img,
                mask_ctx,
                fill_outside=bool(cfg["FILL_OUTSIDE"]),
                fill_value=tuple(cfg["FILL_VALUE"]),
            )
            if ap[0] is None:
                continue
            crop, info = ap
            x1, y1, x2, y2, crop_mask = info
            s_global_piece = s_map[y1:y2, x1:x2]
            pieces.append(
                {
                    "tag": tag,
                    "img": crop,
                    "mask": crop_mask.astype(np.uint8),
                    "s_global": s_global_piece.astype(np.float32),
                    "global_bbox": (
                        float(r["fx1"] + x1),
                        float(r["fy1"] + y1),
                        float(r["fx1"] + x2),
                        float(r["fy1"] + y2),
                    ),
                }
            )
        if not pieces:
            continue
        gxs = []
        gys = []
        for p in pieces:
            gx1, gy1, gx2, gy2 = p["global_bbox"]
            gxs += [gx1, gx2]
            gys += [gy1, gy2]
        g_x0 = int(math.floor(min(gxs)))
        g_y0 = int(math.floor(min(gys)))
        g_x1 = int(math.ceil(max(gxs)))
        g_y1 = int(math.ceil(max(gys)))
        canvas_w = max(1, g_x1 - g_x0)
        canvas_h = max(1, g_y1 - g_y0)
        piece_a = next((p for p in pieces if p["tag"] == "A"), None)
        piece_b = next((p for p in pieces if p["tag"] == "B"), None)
        mosaic, valid_mask = blend_two_masks_to_canvas(
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            g_x0=g_x0,
            g_y0=g_y0,
            piece_a=piece_a,
            piece_b=piece_b,
            sA=sA,
            sB=sB,
            feather=float(cfg["SEAM_FEATHER"]),
        )
        crop_name = f"seam_{int(a['k']):04d}_{int(b['k']):04d}.png"
        write_image(seam_dir / crop_name, mosaic)
        seam_rows.append(
            {
                "seam_id": i,
                "k_left": int(a["k"]),
                "k_right": int(b["k"]),
                "file_left": file_a,
                "file_right": file_b,
                "crop_file": crop_name,
                "crop_global_x1": float(g_x0),
                "crop_global_y1": float(g_y0),
                "crop_w": int(canvas_w),
                "crop_h": int(canvas_h),
                "center_s_left": sA,
                "center_s_right": sB,
                "seam_mid_s": m,
                "seam_left_s": seam_left,
                "seam_right_s": seam_right,
                "ctx_left_s": ctx_left,
                "ctx_right_s": ctx_right,
            }
        )
    core_meta = pd.DataFrame(core_rows)
    seam_meta = pd.DataFrame(seam_rows)
    core_meta.to_csv(out_dir / "core_crops_meta.csv", index=False, encoding="utf-8-sig")
    seam_meta.to_csv(out_dir / "seam_crops_meta.csv", index=False, encoding="utf-8-sig")
    print("[DONE] Core/seam samples saved")
    print(f"core crops : {len(core_meta)} -> {core_dir}")
    print(f"seam crops : {len(seam_meta)} -> {seam_dir}")
    print(f"meta saved : {out_dir / 'core_crops_meta.csv'}")
    print(f"meta saved : {out_dir / 'seam_crops_meta.csv'}")
    return {"core_samples": len(core_rows), "seam_samples": len(seam_rows)}


def run(overrides=None):
    """Execute one sequence. Settings are process-local."""
    effective = dict(DEFAULT_CONFIG)
    if overrides:
        unknown = set(overrides) - set(effective)
        if unknown:
            raise ValueError("Unknown settings: {}".format(", ".join(sorted(unknown))))
        effective.update(overrides)
    if effective["STRIP_AXIS_HINT"] not in ("x", "y"):
        raise ValueError("STRIP_AXIS_HINT must be x or y")
    if any(
        effective[k] < 0
        for k in ("SEAM_HALF_BAND", "CORE_CONTEXT", "SEAM_CONTEXT", "SEAM_FEATHER")
    ):
        raise ValueError("Band and context widths must be nonnegative")
    for key in ["KEYFRAME_CSV"]:
        if not Path(effective[key]).is_file():
            raise FileNotFoundError("{}: {}".format(key, effective[key]))
    if (
        Path(effective["OUT_DIR"]).resolve()
        == Path(effective["KEYFRAME_CSV"]).resolve().parent
    ):
        raise ValueError(
            "Use an output directory separate from the registration inputs"
        )
    out, status, effective = start_run(
        effective["OUT_DIR"], "sample_construction", effective, __file__
    )
    CONFIG.clear()
    CONFIG.update(effective)
    try:
        summary = run_pipeline()
    except Exception as exc:
        finish_run(out, status, error=exc)
        raise
    finish_run(out, status, summary=summary)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build core and seam samples from source keyframes"
    )
    parser.add_argument("--keyframes", help="Registered keyframe CSV")
    parser.add_argument(
        "--images",
        help="Keyframe image directory; inferred beside the CSV when omitted",
    )
    parser.add_argument("--output", help="Sample output directory")
    args = parser.parse_args(argv)
    values = {
        k: v
        for k, v in {
            "KEYFRAME_CSV": args.keyframes,
            "KEYFRAME_IMG_DIR": args.images,
            "OUT_DIR": args.output,
        }.items()
        if v is not None
    }
    return run(values)


if __name__ == "__main__":
    main()
