"""Fuse core and seam detections in a common strip coordinate system."""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from core_seam_config import MERGE_CONFIG as DEFAULT_CONFIG
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


def project_points(
    points_xy: np.ndarray, origin: np.ndarray, e_along: np.ndarray, e_cross: np.ndarray
):
    """Project global coordinates onto the along-strip and transverse axes."""
    X = points_xy - origin
    s = X @ e_along
    t = X @ e_cross
    return (s, t)


def kmeans_1d(
    values: np.ndarray, k: int = 6, max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster transverse positions into the specified number of coarse row groups."""
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    if n == 0:
        return (np.array([], dtype=int), np.array([], dtype=float))
    if n <= k:
        order = np.argsort(values)
        labels = np.zeros(n, dtype=int)
        labels[order] = np.arange(n, dtype=int)
        return (labels, values[order])
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(values, qs)
    for _ in range(max_iter):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for i in range(k):
            m = labels == i
            if np.any(m):
                new_centers[i] = values[m].mean()
        if np.allclose(new_centers, centers, atol=1e-06):
            break
        centers = new_centers
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[int(x)] for x in labels], dtype=int)
    centers = centers[order]
    return (labels, centers)


def pairwise_box_metrics(a: pd.DataFrame, b: pd.DataFrame):
    """Compute intersection over smaller area, IoU and area similarity."""
    if all((c in a.columns and c in b.columns for c in ["x1g", "y1g", "x2g", "y2g"])):
        x1, y1, x2, y2 = ("x1g", "y1g", "x2g", "y2g")
    else:
        x1, y1, x2, y2 = ("x1l", "y1l", "x2l", "y2l")
    ax1 = a[x1].to_numpy(dtype=float)[:, None]
    ay1 = a[y1].to_numpy(dtype=float)[:, None]
    ax2 = a[x2].to_numpy(dtype=float)[:, None]
    ay2 = a[y2].to_numpy(dtype=float)[:, None]
    bx1 = b[x1].to_numpy(dtype=float)[None, :]
    by1 = b[y1].to_numpy(dtype=float)[None, :]
    bx2 = b[x2].to_numpy(dtype=float)[None, :]
    by2 = b[y2].to_numpy(dtype=float)[None, :]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 1e-06, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 1e-06, None)
    iof = inter / np.minimum(area_a, area_b)
    iou = inter / (area_a + area_b - inter + 1e-06)
    area_ratio = np.minimum(area_a, area_b) / np.maximum(area_a, area_b)
    return (iof, iou, area_ratio)


def load_det_csv(path: str) -> pd.DataFrame:
    """Read local detection boxes and apply the configured score threshold."""
    df = pd.read_csv(path)
    file_col = natural_col(df, ["file", "frame_id", "image"], 0)
    x1_col = natural_col(df, ["x1"], 1)
    y1_col = natural_col(df, ["y1"], 2)
    x2_col = natural_col(df, ["x2"], 3)
    y2_col = natural_col(df, ["y2"], 4)
    s_col = natural_col(df, ["score", "conf", "confidence"], 5)
    out = pd.DataFrame(
        {
            "file": df[file_col].astype(str),
            "x1l": df[x1_col].astype(float),
            "y1l": df[y1_col].astype(float),
            "x2l": df[x2_col].astype(float),
            "y2l": df[y2_col].astype(float),
            "score": df[s_col].astype(float),
        }
    )
    out = out[out["score"] >= CONFIG["MIN_SCORE"]].copy().reset_index(drop=True)
    return out


def add_row_ids(df: pd.DataFrame, expected_rows: int):
    """Assign coarse row labels for cross-source matching."""
    if len(df) == 0:
        df = df.copy()
        df["row_id"] = []
        return (df, np.array([], dtype=float))
    labels, centers = kmeans_1d(df["t_anchor"].to_numpy(dtype=float), expected_rows)
    df = df.copy()
    df["row_id"] = labels
    return (df, centers)


def final_light_clean(df: pd.DataFrame):
    """Remove qualifying residual duplicates within each source-image group."""
    out = df.copy()
    out["final_removed"] = False
    out["final_removed_reason"] = ""
    out["final_removed_by_tmp_id"] = np.nan
    out["final_remove_iof"] = np.nan
    out["final_remove_iou"] = np.nan
    out["final_remove_dist"] = np.nan
    out["final_remove_area_ratio"] = np.nan
    out["final_remove_ds"] = np.nan
    out["final_remove_dt"] = np.nan
    if not CONFIG.get("DO_FINAL_LIGHT_CLEAN", True) or len(out) == 0:
        out.attrs["final_light_clean_debug_rows"] = []
        return out
    debug_rows = []
    for owner_frame, g in out.groupby("owner_frame", dropna=False):
        gg = g.reset_index(drop=True)
        n = len(gg)
        if n <= 1:
            continue
        iof, iou, area_ratio = pairwise_box_metrics(gg, gg)
        pts = gg[["anchor_x_global", "anchor_y_global"]].to_numpy(dtype=float)
        dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        if "s_anchor" in gg.columns and "t_anchor" in gg.columns:
            s = gg["s_anchor"].to_numpy(dtype=float)
            t = gg["t_anchor"].to_numpy(dtype=float)
            ds_mat = np.abs(s[:, None] - s[None, :])
            dt_mat = np.abs(t[:, None] - t[None, :])
        else:
            ds_mat = np.zeros((n, n), dtype=float)
            dt_mat = np.zeros((n, n), dtype=float)
        if CONFIG.get("FINAL_REQUIRE_SAME_ROW", True) and "row_id" in gg.columns:
            row = gg["row_id"].to_numpy()
            same_row = row[:, None] == row[None, :]
        else:
            same_row = np.ones((n, n), dtype=bool)
        candidate_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if not same_row[i, j]:
                    continue
                if iof[i, j] < float(CONFIG.get("FINAL_IOF_MIN", 0.92)):
                    continue
                if iou[i, j] < float(CONFIG.get("FINAL_IOU_MIN", 0.55)):
                    continue
                if dist[i, j] > float(CONFIG.get("FINAL_CENTER_DIST_MAX", 8.0)):
                    continue
                if area_ratio[i, j] < float(CONFIG.get("FINAL_AREA_RATIO_MIN", 0.45)):
                    continue
                if ds_mat[i, j] > float(CONFIG.get("FINAL_S_DIST_MAX", 12.0)):
                    continue
                if dt_mat[i, j] > float(CONFIG.get("FINAL_T_DIST_MAX", 8.0)):
                    continue
                quality = (
                    2.0 * float(iou[i, j])
                    + 1.0 * float(iof[i, j])
                    + 0.5 * float(area_ratio[i, j])
                    - 0.02 * float(dist[i, j])
                )
                candidate_pairs.append((quality, i, j))
        candidate_pairs.sort(reverse=True, key=lambda x: x[0])
        active_tmp_ids = set((int(x) for x in gg["tmp_id"].to_numpy()))
        for _, i, j in candidate_pairs:
            tmp_i = int(gg.iloc[i]["tmp_id"])
            tmp_j = int(gg.iloc[j]["tmp_id"])
            if tmp_i not in active_tmp_ids or tmp_j not in active_tmp_ids:
                continue
            score_i = float(gg.iloc[i]["score"])
            score_j = float(gg.iloc[j]["score"])
            if abs(score_i - score_j) > 1e-09:
                drop_idx, keep_idx = (i, j) if score_i < score_j else (j, i)
            else:
                if all((c in gg.columns for c in ["x1g", "y1g", "x2g", "y2g"])):
                    area_i = max(
                        float(
                            (gg.iloc[i]["x2g"] - gg.iloc[i]["x1g"])
                            * (gg.iloc[i]["y2g"] - gg.iloc[i]["y1g"])
                        ),
                        0.0,
                    )
                    area_j = max(
                        float(
                            (gg.iloc[j]["x2g"] - gg.iloc[j]["x1g"])
                            * (gg.iloc[j]["y2g"] - gg.iloc[j]["y1g"])
                        ),
                        0.0,
                    )
                else:
                    area_i = max(
                        float(
                            (gg.iloc[i]["x2l"] - gg.iloc[i]["x1l"])
                            * (gg.iloc[i]["y2l"] - gg.iloc[i]["y1l"])
                        ),
                        0.0,
                    )
                    area_j = max(
                        float(
                            (gg.iloc[j]["x2l"] - gg.iloc[j]["x1l"])
                            * (gg.iloc[j]["y2l"] - gg.iloc[j]["y1l"])
                        ),
                        0.0,
                    )
                drop_idx, keep_idx = (i, j) if area_i < area_j else (j, i)
            drop_id = int(gg.iloc[drop_idx]["tmp_id"])
            keep_id = int(gg.iloc[keep_idx]["tmp_id"])
            active_tmp_ids.discard(drop_id)
            mask = out["tmp_id"] == drop_id
            out.loc[mask, "final_removed"] = True
            out.loc[mask, "final_removed_reason"] = "safe_light_clean_duplicate"
            out.loc[mask, "final_removed_by_tmp_id"] = keep_id
            out.loc[mask, "final_remove_iof"] = float(iof[i, j])
            out.loc[mask, "final_remove_iou"] = float(iou[i, j])
            out.loc[mask, "final_remove_dist"] = float(dist[i, j])
            out.loc[mask, "final_remove_area_ratio"] = float(area_ratio[i, j])
            out.loc[mask, "final_remove_ds"] = float(ds_mat[i, j])
            out.loc[mask, "final_remove_dt"] = float(dt_mat[i, j])
            debug_rows.append(
                {
                    "owner_frame": owner_frame,
                    "drop_tmp_id": drop_id,
                    "keep_tmp_id": keep_id,
                    "drop_source": gg.iloc[drop_idx].get("source", ""),
                    "keep_source": gg.iloc[keep_idx].get("source", ""),
                    "drop_score": float(gg.iloc[drop_idx]["score"]),
                    "keep_score": float(gg.iloc[keep_idx]["score"]),
                    "iof": float(iof[i, j]),
                    "iou": float(iou[i, j]),
                    "center_dist": float(dist[i, j]),
                    "area_ratio": float(area_ratio[i, j]),
                    "ds": float(ds_mat[i, j]),
                    "dt": float(dt_mat[i, j]),
                }
            )
    out.attrs["final_light_clean_debug_rows"] = debug_rows
    return out


def run_pipeline():
    """Restore global detections and apply ownership-based fusion."""
    cfg = CONFIG
    out_dir = ensure_dir(Path(cfg["OUT_DIR"]))
    keyf = pd.read_csv(cfg["KEYFRAME_CSV"])
    core_meta = pd.read_csv(cfg["CORE_META_CSV"])
    seam_meta = pd.read_csv(cfg["SEAM_META_CSV"])
    with open(cfg["STRIP_META_JSON"], "r", encoding="utf-8") as f:
        smeta = json.load(f)
    origin = np.array(smeta["origin"], dtype=float)
    e_along = np.array(smeta["e_along"], dtype=float)
    e_cross = np.array(smeta["e_cross"], dtype=float)
    core_det = load_det_csv(cfg["CORE_DET_CSV"])
    seam_det = load_det_csv(cfg["SEAM_DET_CSV"])
    validate_metadata(core_meta, "core")
    validate_metadata(seam_meta, "seam")
    validate_detection_join(core_det, core_meta, "core")
    validate_detection_join(seam_det, seam_meta, "seam")
    core = core_det.merge(core_meta, left_on="file", right_on="crop_file", how="inner")
    core["source"] = "core"
    core["owner_frame"] = core["src_file"]
    core["x1g"] = core["crop_global_x1"] + core["x1l"]
    core["y1g"] = core["crop_global_y1"] + core["y1l"]
    core["x2g"] = core["crop_global_x1"] + core["x2l"]
    core["y2g"] = core["crop_global_y1"] + core["y2l"]
    core["anchor_x_global"] = 0.5 * (core["x1g"] + core["x2g"])
    core["anchor_y_global"] = 0.5 * (core["y1g"] + core["y2g"])
    s_core, t_core = project_points(
        core[["anchor_x_global", "anchor_y_global"]].to_numpy(dtype=float),
        origin,
        e_along,
        e_cross,
    )
    core["s_anchor"] = s_core
    core["t_anchor"] = t_core
    core["inside_core"] = (core["s_anchor"] >= core["core_left_s"]) & (
        core["s_anchor"] <= core["core_right_s"]
    )
    core["inside_any_seam_zone"] = False
    if len(seam_meta) > 0 and len(core) > 0:
        sL = seam_meta["seam_left_s"].to_numpy(dtype=float)
        sR = seam_meta["seam_right_s"].to_numpy(dtype=float)
        s = core["s_anchor"].to_numpy(dtype=float)[:, None]
        hit = (s >= sL[None, :]) & (s <= sR[None, :])
        core["inside_any_seam_zone"] = hit.any(axis=1)
    core_keep_main = core[core["inside_core"]].copy().reset_index(drop=True)
    seam = seam_det.merge(seam_meta, left_on="file", right_on="crop_file", how="inner")
    seam["source"] = "seam"
    seam["owner_frame"] = seam["file_left"] + "|" + seam["file_right"]
    seam["x1g"] = seam["crop_global_x1"] + seam["x1l"]
    seam["y1g"] = seam["crop_global_y1"] + seam["y1l"]
    seam["x2g"] = seam["crop_global_x1"] + seam["x2l"]
    seam["y2g"] = seam["crop_global_y1"] + seam["y2l"]
    seam["anchor_x_global"] = 0.5 * (seam["x1g"] + seam["x2g"])
    seam["anchor_y_global"] = 0.5 * (seam["y1g"] + seam["y2g"])
    if len(seam) > 0:
        s_seam, t_seam = project_points(
            seam[["anchor_x_global", "anchor_y_global"]].to_numpy(dtype=float),
            origin,
            e_along,
            e_cross,
        )
        seam["s_anchor"] = s_seam
        seam["t_anchor"] = t_seam
    else:
        seam["s_anchor"] = []
        seam["t_anchor"] = []
    seam["inside_seam"] = (seam["s_anchor"] >= seam["seam_left_s"]) & (
        seam["s_anchor"] <= seam["seam_right_s"]
    )
    seam_keep_main = seam[seam["inside_seam"]].copy().reset_index(drop=True)
    both_for_rows = pd.concat(
        [
            core_keep_main[
                ["anchor_x_global", "anchor_y_global", "s_anchor", "t_anchor"]
            ],
            seam_keep_main[
                ["anchor_x_global", "anchor_y_global", "s_anchor", "t_anchor"]
            ],
        ],
        ignore_index=True,
    )
    _, row_centers = add_row_ids(both_for_rows, cfg["EXPECTED_ROWS_PER_STRIP"])

    def assign_nearest_row(df_in: pd.DataFrame):
        if len(df_in) == 0:
            df_in = df_in.copy()
            df_in["row_id"] = []
            return df_in
        if len(row_centers) == 0:
            df_in = df_in.copy()
            df_in["row_id"] = 0
            return df_in
        t = df_in["t_anchor"].to_numpy(dtype=float)
        df_in = df_in.copy()
        df_in["row_id"] = np.argmin(np.abs(t[:, None] - row_centers[None, :]), axis=1)
        return df_in

    core_keep_main = assign_nearest_row(core_keep_main)
    seam_keep_main = assign_nearest_row(seam_keep_main)
    fallback_rows = []
    for _, r in core_keep_main.iterrows():
        if not bool(r["inside_any_seam_zone"]):
            fallback_rows.append(
                {"tmp_id": None, "keep_reason": "core_outside_seam", "matched_seam": 0}
            )
            continue
        if len(seam_keep_main) == 0:
            fallback_rows.append(
                {
                    "tmp_id": None,
                    "keep_reason": "core_fallback_no_seam_dets",
                    "matched_seam": 0,
                }
            )
            continue
        cand = seam_keep_main[seam_keep_main["row_id"] == int(r["row_id"])]
        if len(cand) == 0:
            fallback_rows.append(
                {
                    "tmp_id": None,
                    "keep_reason": "core_fallback_no_same_row",
                    "matched_seam": 0,
                }
            )
            continue
        ds = np.abs(cand["s_anchor"].to_numpy(dtype=float) - float(r["s_anchor"]))
        dt = np.abs(cand["t_anchor"].to_numpy(dtype=float) - float(r["t_anchor"]))
        hit = (ds <= cfg["FALLBACK_MATCH_S"]) & (dt <= cfg["FALLBACK_MATCH_T"])
        if np.any(hit):
            fallback_rows.append(
                {
                    "tmp_id": None,
                    "keep_reason": "drop_replaced_by_seam",
                    "matched_seam": int(hit.sum()),
                }
            )
        else:
            fallback_rows.append(
                {"tmp_id": None, "keep_reason": "core_fallback_keep", "matched_seam": 0}
            )
    core_keep_main = core_keep_main.reset_index(drop=True)
    core_keep_main["tmp_id"] = np.arange(len(core_keep_main), dtype=int)
    fb_df = pd.DataFrame(
        fallback_rows, columns=["tmp_id", "keep_reason", "matched_seam"]
    )
    fb_df["tmp_id"] = core_keep_main["tmp_id"].values
    core_keep_main = core_keep_main.merge(fb_df, on="tmp_id", how="left")
    core_final = (
        core_keep_main[core_keep_main["keep_reason"] != "drop_replaced_by_seam"]
        .copy()
        .reset_index(drop=True)
    )
    seam_keep_main = seam_keep_main.reset_index(drop=True)
    seam_keep_main["tmp_id"] = np.arange(len(seam_keep_main), dtype=int) + 10000000
    seam_keep_main["keep_reason"] = "seam_primary"
    seam_keep_main["matched_seam"] = 0
    final = pd.concat([core_final, seam_keep_main], ignore_index=True, sort=False)
    if len(final) == 0:
        raise RuntimeError(
            "No final detections kept. Check detector outputs or metadata."
        )
    final = assign_nearest_row(final)
    final = final_light_clean(final)
    final_kept = final[~final["final_removed"]].copy().reset_index(drop=True)
    save_all = out_dir / "dedup_all_with_flags.csv"
    save_keep = out_dir / "dedup_keep_global_core_seam_safe_light_clean.csv"
    save_dbg = out_dir / "merge_decisions.csv"
    final.to_csv(save_all, index=False, encoding="utf-8-sig")
    final_kept.to_csv(save_keep, index=False, encoding="utf-8-sig")
    pd.concat(
        [
            core_keep_main[
                [
                    "source",
                    "owner_frame",
                    "score",
                    "s_anchor",
                    "t_anchor",
                    "row_id",
                    "keep_reason",
                    "matched_seam",
                ]
            ],
            seam_keep_main[
                [
                    "source",
                    "owner_frame",
                    "score",
                    "s_anchor",
                    "t_anchor",
                    "row_id",
                    "keep_reason",
                    "matched_seam",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(save_dbg, index=False, encoding="utf-8-sig")
    save_clean_dbg = out_dir / "duplicate_pairs.csv"
    pd.DataFrame(final.attrs.get("final_light_clean_debug_rows", [])).to_csv(
        save_clean_dbg, index=False, encoding="utf-8-sig"
    )
    print("[DONE] Detection fusion completed")
    print(f"core dets loaded     : {len(core)}")
    print(f"seam dets loaded     : {len(seam)}")
    print(f"before final clean   : {len(final)}")
    print(f"final removed clean  : {int(final['final_removed'].sum())}")
    print(f"final kept           : {len(final_kept)}")
    print(f"saved all flags      : {save_all}")
    print(f"saved keep globals   : {save_keep}")
    print(f"saved merge debug    : {save_dbg}")
    print(f"saved clean debug    : {save_clean_dbg}")
    return {
        "core_eligible": len(core_keep_main),
        "seam_eligible": len(seam_keep_main),
        "before_residual_cleaning": len(final),
        "residual_duplicates_removed": int(final["final_removed"].sum()),
        "retained_detections": len(final_kept),
    }


def run(overrides=None):
    """Execute one sequence. Settings are process-local."""
    effective = dict(DEFAULT_CONFIG)
    if overrides:
        unknown = set(overrides) - set(effective)
        if unknown:
            raise ValueError("Unknown settings: {}".format(", ".join(sorted(unknown))))
        effective.update(overrides)
    if effective["EXPECTED_ROWS_PER_STRIP"] < 1:
        raise ValueError("EXPECTED_ROWS_PER_STRIP must be positive")
    if any(
        effective[k] < 0
        for k in (
            "FALLBACK_MATCH_S",
            "FALLBACK_MATCH_T",
            "FINAL_CENTER_DIST_MAX",
            "FINAL_S_DIST_MAX",
            "FINAL_T_DIST_MAX",
        )
    ):
        raise ValueError("Matching distances must be nonnegative")
    for key in [
        "KEYFRAME_CSV",
        "CORE_META_CSV",
        "SEAM_META_CSV",
        "STRIP_META_JSON",
        "CORE_DET_CSV",
        "SEAM_DET_CSV",
    ]:
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
        effective["OUT_DIR"], "detection_fusion", effective, __file__
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
        description="Restore and merge core/seam detector outputs"
    )
    parser.add_argument("--keyframes", help="Registered keyframe CSV")
    parser.add_argument(
        "--dataset", help="Sample directory containing the three metadata files"
    )
    parser.add_argument("--core-detections", help="Core detection CSV")
    parser.add_argument("--seam-detections", help="Seam detection CSV")
    parser.add_argument("--core-meta", help="Explicit core metadata path")
    parser.add_argument("--seam-meta", help="Explicit seam metadata path")
    parser.add_argument("--strip-meta", help="Explicit strip metadata path")
    parser.add_argument("--output", help="Merged output directory")
    args = parser.parse_args(argv)
    values = {}
    if args.dataset:
        folder = Path(args.dataset)
        values.update(
            CORE_META_CSV=str(folder / "core_crops_meta.csv"),
            SEAM_META_CSV=str(folder / "seam_crops_meta.csv"),
            STRIP_META_JSON=str(folder / "strip_meta.json"),
        )
    for key, value in {
        "KEYFRAME_CSV": args.keyframes,
        "CORE_META_CSV": args.core_meta,
        "SEAM_META_CSV": args.seam_meta,
        "STRIP_META_JSON": args.strip_meta,
        "CORE_DET_CSV": args.core_detections,
        "SEAM_DET_CSV": args.seam_detections,
        "OUT_DIR": args.output,
    }.items():
        if value is not None:
            values[key] = value
    return run(values)


if __name__ == "__main__":
    main()
