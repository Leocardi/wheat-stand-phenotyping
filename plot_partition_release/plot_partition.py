"""Partition registered seedling detections and refine plot edges.

Final crop-row fitting and phenotype analysis are separate stages."""

from __future__ import annotations
import json
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

os.environ["PYTHONNOUSERSITE"] = "1"
import cv2
import numpy as np
import pandas as pd
from partition_config import CONFIG
from partition_edges import refine_partition_edges, partition_point_mask


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def kmeans_1d(
    values: np.ndarray, k: int = 3, max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    n = len(values)
    if n == 0:
        return (np.zeros((0,), dtype=int), np.zeros((0,), dtype=float))
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


def estimate_axes(
    points_xy: np.ndarray, hint: str = "y"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.median(points_xy, axis=0)
    X = points_xy - origin
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e_along = eigvecs[:, order[0]]
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
    X = points_xy - origin
    return (X @ e_along, X @ e_cross)


def fix_lr_mirror(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if len(out) < 8 or "anchor_x_global" not in out.columns:
        return out
    try:
        corr = np.corrcoef(
            out["anchor_x_global"].to_numpy(dtype=float), out["t_rot"].to_numpy(dtype=float)
        )[0, 1]
        if np.isfinite(corr) and corr < 0:
            out["t_rot"] = -out["t_rot"]
    except Exception:
        pass
    return out


def load_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if {"anchor_x_global", "anchor_y_global"}.issubset(df.columns):
        return df.copy()
    if {"x1g", "y1g", "x2g", "y2g"}.issubset(df.columns):
        df["anchor_x_global"] = 0.5 * (df["x1g"].astype(float) + df["x2g"].astype(float))
        df["anchor_y_global"] = 0.5 * (df["y1g"].astype(float) + df["y2g"].astype(float))
        return df
    if {"x1l", "y1l", "x2l", "y2l", "crop_global_x1", "crop_global_y1"}.issubset(df.columns):
        df["x1g"] = df["x1l"].astype(float) + df["crop_global_x1"].astype(float)
        df["y1g"] = df["y1l"].astype(float) + df["crop_global_y1"].astype(float)
        df["x2g"] = df["x2l"].astype(float) + df["crop_global_x1"].astype(float)
        df["y2g"] = df["y2l"].astype(float) + df["crop_global_y1"].astype(float)
        df["anchor_x_global"] = 0.5 * (df["x1g"] + df["x2g"])
        df["anchor_y_global"] = 0.5 * (df["y1g"] + df["y2g"])
        return df
    raise ValueError("Input CSV lacks anchor/global coordinate columns.")


def coarse_rectify_and_rows(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()
    pts = out[["anchor_x_global", "anchor_y_global"]].to_numpy(dtype=float)
    origin, e_along, e_cross = estimate_axes(pts, cfg["STRIP_AXIS_HINT"])
    s, t = project_points(pts, origin, e_along, e_cross)
    out["s_rot"] = s
    out["t_rot"] = t
    out = fix_lr_mirror(out)
    used_existing = False
    if cfg.get("PREFER_EXISTING_ROW_ID", False) and "row_id" in out.columns:
        try:
            row_series = out["row_id"].dropna().astype(int)
            uniq = sorted(row_series.unique().tolist())
            if len(uniq) >= int(cfg["MAIN_ROWS"]):
                centers_old = sorted(
                    [
                        (rid, float(np.median(out.loc[out["row_id"] == rid, "t_rot"])))
                        for rid in uniq
                    ],
                    key=lambda z: z[1],
                )[: int(cfg["MAIN_ROWS"])]
                remap = {rid: i for i, (rid, _) in enumerate(centers_old)}
                tmp = out["row_id"].map(remap)
                if tmp.notna().sum() >= max(20, len(out) * 0.5):
                    miss = tmp.isna()
                    out["row_main"] = tmp.fillna(-1).astype(int)
                    if miss.any():
                        known = np.array([c for _, c in centers_old], dtype=float)
                        tvals = out.loc[miss, "t_rot"].to_numpy(dtype=float)
                        out.loc[miss, "row_main"] = np.argmin(
                            np.abs(tvals[:, None] - known[None, :]), axis=1
                        )
                    used_existing = True
        except Exception:
            used_existing = False
    if not used_existing:
        labels, _ = kmeans_1d(out["t_rot"].to_numpy(dtype=float), k=int(cfg["MAIN_ROWS"]))
        out["row_main"] = labels.astype(int)
    centers = np.array(
        [
            np.median(out.loc[out["row_main"] == i, "t_rot"])
            for i in range(int(cfg["MAIN_ROWS"]))
        ],
        dtype=float,
    )
    order = np.argsort(centers)
    remap = {int(old): i for i, old in enumerate(order)}
    out["row_main"] = out["row_main"].map(remap).astype(int)
    centers = np.array(
        [
            np.median(out.loc[out["row_main"] == i, "t_rot"])
            for i in range(int(cfg["MAIN_ROWS"]))
        ],
        dtype=float,
    )
    out["keep_for_partition"] = True
    if cfg.get("USE_LIGHT_OUTER_RANGE_FILTER", True):
        gaps = np.diff(np.sort(centers))
        gap_med = float(np.median(gaps)) if len(gaps) else 40.0
        left = float(centers.min()) - 0.6 * gap_med - float(cfg["OUTER_RANGE_EXTRA_MARGIN"])
        right = float(centers.max()) + 0.6 * gap_med + float(cfg["OUTER_RANGE_EXTRA_MARGIN"])
        out["keep_for_partition"] = (out["t_rot"] >= left) & (out["t_rot"] <= right)
    meta = {
        "used_existing_row_id": int(used_existing),
        "row_centers_t": [float(x) for x in centers],
        "axis_origin": [float(x) for x in origin.tolist()],
        "axis_along": [float(x) for x in e_along.tolist()],
        "axis_cross": [float(x) for x in e_cross.tolist()],
    }
    return (out, meta)


def robust_norm_density(count: float, ref_count: float) -> float:
    return float(count) / max(float(ref_count), 1e-06)


def interval_points(svals: np.ndarray, s0: float, s1: float) -> np.ndarray:
    return svals[(svals >= s0) & (svals <= s1)]


def interval_count(svals: np.ndarray, s0: float, s1: float) -> int:
    return int(np.sum((svals >= s0) & (svals <= s1)))


def longest_gap_ratio(svals: np.ndarray, s0: float, s1: float) -> float:
    width = max(float(s1 - s0), 1e-06)
    pts = np.sort(interval_points(svals, s0, s1))
    if len(pts) == 0:
        return 1.0
    ext = np.concatenate([[s0], pts, [s1]])
    gaps = np.diff(ext)
    return float(np.max(gaps) / width) if len(gaps) else 1.0


def discounted_interval_count(svals: np.ndarray, s0: float, s1: float, cfg: Dict) -> float:
    pts = np.sort(interval_points(svals, s0, s1))
    n = len(pts)
    if n == 0:
        return 0.0
    if n == 1:
        return max(0.0, 1.0 - float(cfg.get("INSIDE_ISOLATED_WEIGHT", 0.85)))
    nn_min = float(cfg.get("INSIDE_ISOLATED_NN_MIN", 260.0))
    iso_weight = float(cfg.get("INSIDE_ISOLATED_WEIGHT", 0.85))
    max_frac = float(cfg.get("INSIDE_ISOLATED_MAX_FRAC", 0.7))
    iso = 0
    for i in range(n):
        left_gap = pts[i] - pts[i - 1] if i > 0 else np.inf
        right_gap = pts[i + 1] - pts[i] if i < n - 1 else np.inf
        if min(left_gap, right_gap) >= nn_min:
            iso += 1
    iso = min(iso, int(np.floor(max_frac * n)))
    eff = float(n) - iso_weight * float(iso)
    return max(0.0, eff)


def row_ref_counts(df: pd.DataFrame, window: float) -> Dict[int, float]:
    refs: Dict[int, float] = {}
    for r in range(int(CONFIG.get("MAIN_ROWS", 3))):
        s = np.sort(df.loc[df["row_main"].astype(int) == r, "s_rot"].to_numpy(dtype=float))
        if len(s) == 0:
            refs[int(r)] = 1.0
            continue
        centers = np.arange(float(s.min()), float(s.max()) + window, max(window / 4.0, 1.0))
        vals = [np.sum((s >= c - 0.5 * window) & (s <= c + 0.5 * window)) for c in centers]
        q = float(np.quantile(vals, 0.9)) if len(vals) else 1.0
        refs[int(r)] = max(q, 1.0)
    return refs


def boundary_width_score(width: float, cfg: Dict) -> float:
    lo = float(cfg["MIN_BAND_WIDTH"])
    hi = float(cfg["MAX_BAND_WIDTH"])
    if width < lo:
        return -0.8
    if width > hi:
        return -0.8
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return float(cfg["W_WIDTH"]) * max(0.0, 1.0 - abs(width - mid) / max(half, 1e-06))


def evaluate_candidate_band(
    df: pd.DataFrame, row_refs: Dict[int, float], s0: float, s1: float, cfg: Dict
) -> Dict[str, Any]:
    width = float(s1 - s0)
    side_w = float(cfg["SIDE_WINDOW"])
    left0, left1 = (s0 - side_w, s0)
    right0, right1 = (s1, s1 + side_w)
    row_stats = []
    inside_norms = []
    ref_norms = []
    cont_rows = 0
    for r in range(int(cfg["MAIN_ROWS"])):
        svals = np.sort(df.loc[df["row_main"].astype(int) == r, "s_rot"].to_numpy(dtype=float))
        inside_ct = interval_count(svals, s0, s1)
        inside_eff = discounted_interval_count(svals, s0, s1, cfg)
        left_ct = interval_count(svals, left0, left1)
        right_ct = interval_count(svals, right0, right1)
        inside_dn = robust_norm_density(inside_eff, row_refs.get(r, 1.0))
        left_dn = robust_norm_density(left_ct, row_refs.get(r, 1.0))
        right_dn = robust_norm_density(right_ct, row_refs.get(r, 1.0))
        ref_dn = max(left_dn, right_dn)
        drop_ratio = inside_dn / max(ref_dn, 1e-06)
        gap_ratio = longest_gap_ratio(svals, s0, s1)
        strong_break = drop_ratio <= float(cfg["STRONG_DROP_RATIO"]) and gap_ratio >= float(
            cfg["STRONG_GAP_RATIO"]
        )
        weak_break = drop_ratio <= float(cfg["WEAK_DROP_RATIO"]) and gap_ratio >= float(
            cfg["WEAK_GAP_RATIO"]
        )
        continuous = inside_ct >= int(
            cfg.get("CONTINUITY_ROW_MIN_POINTS", 4)
        ) and gap_ratio <= float(cfg.get("CONTINUITY_ROW_GAP_MAX", 0.4))
        cont_rows += int(continuous)
        row_stats.append(
            {
                "row": r,
                "inside_ct": int(inside_ct),
                "inside_ct_eff": float(inside_eff),
                "left_ct": int(left_ct),
                "right_ct": int(right_ct),
                "inside_dn": float(inside_dn),
                "left_dn": float(left_dn),
                "right_dn": float(right_dn),
                "ref_dn": float(ref_dn),
                "drop_ratio": float(drop_ratio),
                "gap_ratio": float(gap_ratio),
                "strong_break": int(strong_break),
                "weak_break": int(weak_break),
                "continuous": int(continuous),
            }
        )
        inside_norms.append(inside_dn)
        ref_norms.append(ref_dn)
    n_strong = int(sum((x["strong_break"] for x in row_stats)))
    n_weak = int(sum((x["weak_break"] for x in row_stats)))
    joint_inside = float(np.mean(inside_norms))
    joint_ref = float(np.mean(ref_norms))
    joint_drop = joint_inside / max(joint_ref, 1e-06)
    left_recover_rows = int(
        sum((x["left_dn"] >= float(cfg["RECOVER_ROW_DENSITY_MIN"]) for x in row_stats))
    )
    right_recover_rows = int(
        sum((x["right_dn"] >= float(cfg["RECOVER_ROW_DENSITY_MIN"]) for x in row_stats))
    )
    left_joint = float(np.mean([x["left_dn"] for x in row_stats]))
    right_joint = float(np.mean([x["right_dn"] for x in row_stats]))
    left_recover = int(
        left_recover_rows >= int(cfg["MIN_RECOVER_ROWS"])
        and left_joint >= float(cfg["RECOVER_JOINT_MIN"])
    )
    right_recover = int(
        right_recover_rows >= int(cfg["MIN_RECOVER_ROWS"])
        and right_joint >= float(cfg["RECOVER_JOINT_MIN"])
    )
    top2_gap = sorted([x["gap_ratio"] for x in row_stats], reverse=True)[:2]
    mean_top2_gap = float(np.mean(top2_gap)) if len(top2_gap) else 0.0
    drop_ratios = np.array([x["drop_ratio"] for x in row_stats], dtype=float)
    dominant_single_row_penalty = float(np.max(drop_ratios) - np.min(drop_ratios))
    continuity_penalty = float(cont_rows) / max(float(cfg["MAIN_ROWS"]), 1.0)
    score = 0.0
    score += float(cfg["W_STRONG_BREAK"]) * min(n_strong, 2)
    score += float(cfg["W_WEAK_BREAK"]) * min(n_weak, 3)
    score += float(cfg["W_JOINT_DROP"]) * max(0.0, 1.0 - joint_drop)
    score += float(cfg["W_GAP_RATIO"]) * mean_top2_gap
    score += float(cfg["W_LEFT_RECOVER"]) * left_recover
    score += float(cfg["W_RIGHT_RECOVER"]) * right_recover
    score += boundary_width_score(width, cfg)
    score -= float(cfg["PENALTY_DOMINANT_SINGLE_ROW"]) * max(
        0.0, dominant_single_row_penalty - 0.35
    )
    if left_recover == 0 or right_recover == 0:
        score -= float(cfg["PENALTY_NO_RECOVER"])
    score -= float(cfg.get("PENALTY_CONTINUITY", 2.2)) * continuity_penalty
    strict = (
        score >= float(cfg["HIGH_CONF_SCORE"])
        and n_strong >= int(cfg["HIGH_CONF_MIN_STRONG"])
        and (n_weak >= int(cfg["HIGH_CONF_MIN_WEAK"]))
        and (joint_drop <= float(cfg["HIGH_CONF_JOINT_DROP_MAX"]))
        and (
            not bool(cfg["HIGH_CONF_REQUIRE_BOTH_RECOVER"])
            or (left_recover == 1 and right_recover == 1)
        )
    )
    out: Dict[str, Any] = {
        "s0": float(s0),
        "s1": float(s1),
        "center_s": float(0.5 * (s0 + s1)),
        "width": float(width),
        "score": float(score),
        "kind": "dense_scan",
        "strict_high_conf": int(strict),
        "n_strong_break": int(n_strong),
        "n_weak_break": int(n_weak),
        "joint_inside": float(joint_inside),
        "joint_ref": float(joint_ref),
        "joint_drop": float(joint_drop),
        "left_recover": int(left_recover),
        "right_recover": int(right_recover),
        "left_joint": float(left_joint),
        "right_joint": float(right_joint),
        "left_recover_rows": int(left_recover_rows),
        "right_recover_rows": int(right_recover_rows),
        "mean_top2_gap_ratio": float(mean_top2_gap),
        "dominant_single_row_penalty": float(dominant_single_row_penalty),
        "continuous_rows": int(cont_rows),
        "continuity_penalty": float(continuity_penalty),
    }
    for x in row_stats:
        r = int(x["row"])
        for key in [
            "inside_ct",
            "inside_ct_eff",
            "left_ct",
            "right_ct",
            "inside_dn",
            "left_dn",
            "right_dn",
            "ref_dn",
            "drop_ratio",
            "gap_ratio",
            "strong_break",
            "weak_break",
            "continuous",
        ]:
            out[f"row{r}_{key}"] = x[key]
    return out


def scan_dense_candidates(work: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    step = float(cfg["SCAN_STEP"])
    row_refs = row_ref_counts(work, float(cfg["SIDE_WINDOW"]))
    rows: List[Dict[str, Any]] = []
    for width in cfg["BAND_WIDTHS"]:
        width = float(width)
        starts = np.arange(smin, smax - width + step, step, dtype=float)
        for s0 in starts:
            rows.append(
                evaluate_candidate_band(work, row_refs, float(s0), float(s0 + width), cfg)
            )
    return pd.DataFrame(rows)


def row_large_gap_proposals(work: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in range(int(cfg["MAIN_ROWS"])):
        svals = np.sort(
            work.loc[work["row_main"].astype(int) == r, "s_rot"].to_numpy(dtype=float)
        )
        if len(svals) < 4:
            continue
        gaps = np.diff(svals)
        positive = gaps[gaps > 1e-06]
        if len(positive) < 3:
            continue
        base = positive[positive <= np.quantile(positive, 0.8)]
        if len(base) == 0:
            base = positive
        typical_step = float(np.median(base))
        thr = max(
            float(np.quantile(positive, float(cfg["ROW_GAP_HIGH_Q"]))),
            typical_step * float(cfg["ROW_GAP_MIN_FACTOR"]),
        )
        centers = 0.5 * (svals[:-1] + svals[1:])
        for g, c in zip(gaps, centers):
            if float(g) >= thr:
                rows.append(
                    {
                        "row_main": int(r),
                        "gap_center_s": float(c),
                        "gap_len": float(g),
                        "row_typical_step": float(typical_step),
                        "row_gap_thr": float(thr),
                    }
                )
    return pd.DataFrame(rows)


def cluster_gap_centers(gap_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if len(gap_df) == 0:
        return pd.DataFrame(
            columns=[
                "center_s",
                "gap_width_est",
                "support_rows",
                "support_count",
                "support_row_count",
            ]
        )
    tol = float(cfg["ROW_GAP_CLUSTER_TOL"])
    gg = gap_df.sort_values("gap_center_s").reset_index(drop=True)
    groups: List[List[int]] = []
    cur = [0]
    for i in range(1, len(gg)):
        if (
            abs(float(gg.loc[i, "gap_center_s"]) - float(gg.loc[cur[-1], "gap_center_s"]))
            <= tol
        ):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    out: List[Dict[str, Any]] = []
    for idxs in groups:
        sub = gg.loc[idxs].copy()
        support_rows = sorted(sub["row_main"].astype(int).unique().tolist())
        if len(support_rows) < int(cfg["ROW_GAP_MIN_ROWS"]):
            continue
        out.append(
            {
                "center_s": float(np.median(sub["gap_center_s"].to_numpy(dtype=float))),
                "gap_width_est": float(np.median(sub["gap_len"].to_numpy(dtype=float))),
                "support_rows": ",".join(map(str, support_rows)),
                "support_count": int(len(sub)),
                "support_row_count": int(len(support_rows)),
            }
        )
    return (
        pd.DataFrame(out).sort_values("center_s").reset_index(drop=True)
        if out
        else pd.DataFrame(
            columns=[
                "center_s",
                "gap_width_est",
                "support_rows",
                "support_count",
                "support_row_count",
            ]
        )
    )


def build_row_gap_candidates(work: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if not bool(cfg.get("ENABLE_ROW_GAP_PROPOSALS", True)):
        return pd.DataFrame()
    gap_df = row_large_gap_proposals(work, cfg)
    clusters = cluster_gap_centers(gap_df, cfg)
    if len(clusters) == 0:
        return pd.DataFrame()
    row_refs = row_ref_counts(work, float(cfg["SIDE_WINDOW"]))
    rows: List[Dict[str, Any]] = []
    for _, rr in clusters.iterrows():
        center = float(rr["center_s"])
        width = float(rr.get("gap_width_est", np.median(cfg["BAND_WIDTHS"]))) * float(
            cfg.get("ROW_GAP_BAND_SCALE", 1.08)
        )
        width = float(
            np.clip(width, float(cfg["MIN_BAND_WIDTH"]), float(cfg["MAX_BAND_WIDTH"]))
        )
        cand = evaluate_candidate_band(
            work, row_refs, center - 0.5 * width, center + 0.5 * width, cfg
        )
        cand["kind"] = "row_gap_proposal"
        cand["row_gap_support_rows"] = rr.get("support_rows", "")
        cand["row_gap_support_count"] = int(rr.get("support_count", 0))
        cand["row_gap_support_row_count"] = int(rr.get("support_row_count", 0))
        cand["score"] = float(cand["score"]) + 0.45 * float(cand["row_gap_support_row_count"])
        rows.append(cand)
    return pd.DataFrame(rows)


def filter_and_cluster_candidates(cands: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if len(cands) == 0:
        return cands.copy()
    cand = cands.copy()
    for col in [
        "score",
        "center_s",
        "n_weak_break",
        "n_strong_break",
        "left_recover",
        "right_recover",
    ]:
        cand[col] = pd.to_numeric(cand[col], errors="coerce")
    cand = cand[
        (cand["score"] >= float(cfg["CAND_MIN_SCORE"]))
        & (cand["n_weak_break"] >= int(cfg["CAND_MIN_WEAK_BREAK"]))
        & (cand["n_strong_break"] >= int(cfg["CAND_MIN_STRONG_BREAK"]))
    ].copy()
    if len(cand) == 0:
        return cand
    cand = cand.sort_values("center_s").reset_index(drop=True)
    tol = float(cfg["CAND_CLUSTER_TOL"])
    rows: List[Dict[str, Any]] = []
    start = 0
    for i in range(1, len(cand) + 1):
        end_cluster = (
            i == len(cand)
            or float(cand.iloc[i]["center_s"]) - float(cand.iloc[i - 1]["center_s"]) > tol
        )
        if not end_cluster:
            continue
        block = cand.iloc[start:i].copy()
        block["cluster_rank"] = (
            block["score"].astype(float)
            + 1.0 * block["strict_high_conf"].fillna(0).astype(float)
            + 0.55 * block["n_strong_break"].astype(float)
            + 0.25 * block["n_weak_break"].astype(float)
            + 0.45
            * (block["left_recover"].astype(float) + block["right_recover"].astype(float))
            - 1.5 * block["continuity_penalty"].astype(float)
        )
        best = block.sort_values(["cluster_rank", "score"], ascending=False).iloc[0].to_dict()
        best["cluster_size"] = int(len(block))
        best["cluster_center_min"] = float(block["center_s"].min())
        best["cluster_center_max"] = float(block["center_s"].max())
        rows.append(best)
        start = i
    out = pd.DataFrame(rows).sort_values("center_s").reset_index(drop=True)
    max_keep = int(cfg.get("CAND_MAX_KEEP", 450))
    if len(out) > max_keep:
        out = (
            out.sort_values("cluster_rank", ascending=False)
            .head(max_keep)
            .sort_values("center_s")
            .reset_index(drop=True)
        )
    return out


def build_all_candidates(work: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dense = scan_dense_candidates(work, cfg)
    row_gap = build_row_gap_candidates(work, cfg)
    if len(row_gap):
        all_cands = pd.concat([dense, row_gap], ignore_index=True, sort=False)
    else:
        all_cands = dense
    clustered = filter_and_cluster_candidates(all_cands, cfg)
    return (all_cands, clustered)


def is_high_conf_candidate(row: pd.Series, cfg: Dict) -> bool:
    try:
        if float(row.get("score", -1000000000.0)) < float(cfg["HIGH_CONF_SCORE"]):
            return False
        if int(row.get("n_strong_break", 0)) < int(cfg["HIGH_CONF_MIN_STRONG"]):
            return False
        if int(row.get("n_weak_break", 0)) < int(cfg["HIGH_CONF_MIN_WEAK"]):
            return False
        if float(row.get("joint_drop", 1.0)) > float(cfg["HIGH_CONF_JOINT_DROP_MAX"]):
            return False
        if bool(cfg["HIGH_CONF_REQUIRE_BOTH_RECOVER"]):
            if int(row.get("left_recover", 0)) != 1 or int(row.get("right_recover", 0)) != 1:
                return False
        if float(row.get("continuity_penalty", 1.0)) > 0.35:
            return False
        return True
    except Exception:
        return False


def _candidate_inside_total(row: pd.Series, cfg: Dict) -> int:
    total = 0
    for r in range(int(cfg.get("MAIN_ROWS", 3))):
        try:
            total += int(row.get(f"row{r}_inside_ct", 0))
        except Exception:
            pass
    return int(total)


def is_ultra_empty_candidate(row: pd.Series, cfg: Dict) -> bool:
    """Identify an empty gap even when a flank recovery window is sparse."""
    try:
        if float(row.get("joint_drop", 1.0)) > float(
            cfg.get("ULTRA_EMPTY_JOINT_DROP_MAX", 0.08)
        ):
            return False
        if _candidate_inside_total(row, cfg) > int(cfg.get("ULTRA_EMPTY_MAX_INSIDE_TOTAL", 6)):
            return False
        if int(row.get("n_strong_break", 0)) < int(cfg.get("ULTRA_EMPTY_MIN_STRONG", 2)):
            return False
        if int(row.get("n_weak_break", 0)) < int(cfg.get("ULTRA_EMPTY_MIN_WEAK", 3)):
            return False
        if int(row.get("continuous_rows", 0)) > int(
            cfg.get("ULTRA_EMPTY_MAX_CONTINUOUS_ROWS", 0)
        ):
            return False
        support = row.get("row_gap_support_row_count", np.nan)
        if pd.notna(support) and float(support) < float(
            cfg.get("ULTRA_EMPTY_MIN_ROW_GAP_SUPPORT_ROWS", 2)
        ):
            return False
        return True
    except Exception:
        return False


def _local_peak_indices(y: np.ndarray, min_distance_bins: int = 1) -> np.ndarray:
    """Find local peaks without a SciPy dependency."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(y) < 3:
        return np.zeros((0,), dtype=int)
    raw = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if len(raw) == 0:
        return raw
    min_distance_bins = max(1, int(min_distance_bins))
    order = raw[np.argsort(y[raw])[::-1]]
    keep: List[int] = []
    for idx in order:
        if all((abs(int(idx) - int(k)) >= min_distance_bins for k in keep)):
            keep.append(int(idx))
    keep = sorted(keep)
    return np.asarray(keep, dtype=int)


def estimate_typical_length_from_density_acf(
    work: pd.DataFrame, cfg: Dict
) -> Tuple[float, Dict[str, Any]]:
    """Estimate the repetition scale from longitudinal point-density autocorrelation."""
    if not bool(cfg.get("ENABLE_DENSITY_ACF_LENGTH", True)) or work is None or len(work) < 20:
        return (np.nan, {"acf_length_source": "disabled_or_too_few_points"})
    s = work["s_rot"].to_numpy(dtype=float)
    span = float(np.nanmax(s) - np.nanmin(s)) if len(s) else 0.0
    if not np.isfinite(span) or span <= float(cfg.get("ACF_MIN_LAG", 12000.0)) * 2.0:
        return (np.nan, {"acf_length_source": "span_too_short", "acf_span_s": span})
    binw = float(cfg.get("ACF_BIN", 500.0))
    bins = np.arange(float(np.nanmin(s)), float(np.nanmax(s)) + binw, binw, dtype=float)
    if len(bins) < 8:
        return (np.nan, {"acf_length_source": "too_few_bins", "acf_span_s": span})
    hist, _ = np.histogram(s, bins=bins)
    smooth_win = max(
        3, int(round(float(cfg.get("ACF_SMOOTH_WIN_PX", 2500.0)) / max(binw, 1e-06)))
    )
    if smooth_win % 2 == 0:
        smooth_win += 1
    kernel = np.ones(smooth_win, dtype=float) / float(smooth_win)
    sm = np.convolve(hist.astype(float), kernel, mode="same")
    x = sm - float(np.mean(sm))
    denom = float(np.dot(x, x))
    if denom <= 1e-09:
        return (np.nan, {"acf_length_source": "flat_density", "acf_span_s": span})
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :] / denom
    lags = np.arange(len(ac), dtype=float) * binw
    min_lag = float(cfg.get("ACF_MIN_LAG", 12000.0))
    max_lag = min(float(cfg.get("ACF_MAX_LAG", 90000.0)), span * 0.75)
    mask = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(mask):
        return (np.nan, {"acf_length_source": "no_valid_lag_range", "acf_span_s": span})
    lagvals = lags[mask]
    vals = ac[mask]
    min_dist_bins = max(
        1, int(round(float(cfg.get("ACF_PEAK_DISTANCE_PX", 8000.0)) / max(binw, 1e-06)))
    )
    peaks = _local_peak_indices(vals, min_distance_bins=min_dist_bins)
    if len(peaks) == 0:
        return (np.nan, {"acf_length_source": "no_acf_peak", "acf_span_s": span})
    peak_vals = vals[peaks]
    order = np.argsort(peak_vals)[::-1]
    best = int(peaks[order[0]])
    best_len = float(lagvals[best])
    best_val = float(vals[best])
    if best_val < float(cfg.get("ACF_MIN_PEAK_VALUE", 0.08)):
        return (
            np.nan,
            {
                "acf_length_source": "weak_acf_peak",
                "acf_peak_len": best_len,
                "acf_peak_value": best_val,
                "acf_span_s": span,
            },
        )
    top_lags = lagvals[peaks][order[:5]].astype(float).tolist()
    top_vals = vals[peaks][order[:5]].astype(float).tolist()
    return (
        best_len,
        {
            "acf_length_source": "density_acf",
            "acf_peak_len": best_len,
            "acf_peak_value": best_val,
            "acf_top_lags": ";".join((f"{x:.1f}" for x in top_lags)),
            "acf_top_values": ";".join((f"{x:.4f}" for x in top_vals)),
            "acf_span_s": span,
        },
    )


def robust_typical_length_from_candidates(
    cands: pd.DataFrame, work: pd.DataFrame, cfg: Dict
) -> Tuple[float, Dict[str, Any]]:
    """Combine candidate spacing and density autocorrelation to estimate a reference scale."""
    acf_len, acf_meta = estimate_typical_length_from_density_acf(work, cfg)
    high_meta: Dict[str, Any] = {
        "length_source_high_conf": "none",
        "n_high_conf": 0,
        "n_intervals": 0,
    }
    high_typical = np.nan
    high_spread = np.nan
    if cands is not None and len(cands) > 0:
        high = cands[cands.apply(lambda r: is_high_conf_candidate(r, cfg), axis=1)].copy()
        high_meta["n_high_conf"] = int(len(high))
        if len(high) >= 2:
            centers = np.sort(high["center_s"].to_numpy(dtype=float))
            ds = np.diff(centers)
            ds = ds[np.isfinite(ds) & (ds > float(cfg.get("DP_SEGMENT_MIN_ABS", 2500.0)))]
            if len(ds) > 0:
                qlo, qhi = (
                    np.quantile(ds, [0.15, 0.85]) if len(ds) >= 4 else (np.min(ds), np.max(ds))
                )
                core = ds[(ds >= qlo) & (ds <= qhi)]
                if len(core) == 0:
                    core = ds
                high_typical = float(np.median(core))
                high_spread = float(qhi / max(qlo, 1e-06))
                high_meta.update(
                    {
                        "length_source_high_conf": "high_conf_center_spacing",
                        "n_intervals": int(len(ds)),
                        "typical_len_high_conf": float(high_typical),
                        "interval_median": float(np.median(ds)),
                        "interval_q15": float(qlo),
                        "interval_q85": float(qhi),
                        "interval_spread_q85_q15": float(high_spread),
                    }
                )
            else:
                high_meta["length_source_high_conf"] = "no_valid_intervals"
        else:
            high_meta["length_source_high_conf"] = "insufficient_high_conf"
    use_acf = False
    reason = ""
    if np.isfinite(acf_len) and acf_len > 1e-06:
        if not np.isfinite(high_typical):
            use_acf = True
            reason = "no_reliable_high_conf_length"
        elif int(high_meta.get("n_intervals", 0)) < int(
            cfg.get("DP_LENGTH_PRIOR_MIN_INTERVALS_STRONG", 3)
        ):
            use_acf = True
            reason = "few_high_conf_intervals"
        elif np.isfinite(high_spread) and high_spread > float(
            cfg.get("HIGH_CONF_INTERVAL_SPREAD_MAX", 2.8)
        ):
            use_acf = True
            reason = "high_conf_intervals_too_scattered"
        elif high_typical < 0.55 * acf_len:
            use_acf = True
            reason = "high_conf_length_too_short_vs_acf"
        elif high_typical > 2.2 * acf_len:
            use_acf = True
            reason = "high_conf_length_too_long_vs_acf"
    meta: Dict[str, Any] = {}
    meta.update(high_meta)
    meta.update(acf_meta)
    if use_acf:
        meta.update(
            {
                "length_source": "density_acf",
                "length_choose_reason": reason,
                "typical_len": float(acf_len),
            }
        )
        return (float(acf_len), meta)
    if np.isfinite(high_typical) and high_typical > 1e-06:
        meta.update(
            {
                "length_source": "high_conf_center_spacing",
                "length_choose_reason": "use_stable_high_conf_spacing",
                "typical_len": float(high_typical),
            }
        )
        return (float(high_typical), meta)
    if np.isfinite(acf_len) and acf_len > 1e-06:
        meta.update(
            {
                "length_source": "density_acf",
                "length_choose_reason": "fallback_to_acf",
                "typical_len": float(acf_len),
            }
        )
        return (float(acf_len), meta)
    meta.update(
        {
            "length_source": "none",
            "length_choose_reason": "no_reliable_length",
            "typical_len": np.nan,
        }
    )
    return (np.nan, meta)


def determine_dp_mode(
    work: pd.DataFrame, cands: pd.DataFrame, typical_len: float, cfg: Dict
) -> Dict[str, Any]:
    high_n = (
        int(cands.apply(lambda r: is_high_conf_candidate(r, cfg), axis=1).sum())
        if len(cands)
        else 0
    )
    span = float(work["s_rot"].max() - work["s_rot"].min()) if len(work) else 0.0
    est_plots_by_len = (
        float(span / typical_len)
        if np.isfinite(typical_len) and typical_len > 1e-06
        else np.nan
    )
    if (
        np.isfinite(typical_len)
        and typical_len > 1e-06
        and (
            high_n >= int(cfg["AUTO_MODE_MANY_MIN_HIGH_CONF"])
            or (
                np.isfinite(est_plots_by_len)
                and est_plots_by_len >= float(cfg["AUTO_MODE_MANY_MIN_EST_PLOTS"])
            )
        )
    ):
        mode = "many_plot_length_prior"
        length_weight = float(cfg["DP_LENGTH_WEIGHT_MANY"])
        boundary_penalty = float(cfg["DP_BOUNDARY_PENALTY_MANY"])
        length_prior_strength = "strong"
    elif (
        np.isfinite(typical_len)
        and typical_len > 1e-06
        and (
            np.isfinite(est_plots_by_len)
            and est_plots_by_len
            >= float(cfg.get("AUTO_MODE_FEW_MIN_EST_PLOTS_FOR_LENGTH", 2.5))
        )
    ):
        mode = "few_plot_moderate_length_prior"
        length_weight = float(cfg["DP_LENGTH_WEIGHT_FEW"])
        boundary_penalty = float(cfg["DP_BOUNDARY_PENALTY_FEW"])
        length_prior_strength = "moderate"
    elif high_n > 0:
        mode = "few_plot_evidence_first"
        length_weight = 0.35 * float(cfg["DP_LENGTH_WEIGHT_FEW"])
        boundary_penalty = float(cfg["DP_BOUNDARY_PENALTY_FEW"])
        length_prior_strength = "weak"
    else:
        mode = "evidence_only_no_reliable_length_prior"
        length_weight = 0.0
        boundary_penalty = float(
            cfg.get("DP_BOUNDARY_PENALTY_NOPRIOR", cfg["DP_BOUNDARY_PENALTY_FEW"])
        )
        length_prior_strength = "none"
    return {
        "mode": mode,
        "high_conf_count": high_n,
        "span_s": span,
        "estimated_plots_by_length": est_plots_by_len,
        "length_weight": length_weight,
        "boundary_penalty": boundary_penalty,
        "length_prior_strength": length_prior_strength,
    }


def candidate_reward(row: pd.Series, cfg: Dict, mode_info: Dict[str, Any]) -> float:
    score = float(row.get("score", 0.0))
    reward = float(cfg["DP_REWARD_SCALE"]) * (score - float(cfg["DP_SCORE_BASE"]))
    reward -= float(mode_info["boundary_penalty"])
    high_conf = is_high_conf_candidate(row, cfg)
    ultra_empty = is_ultra_empty_candidate(row, cfg)
    if high_conf:
        reward += float(cfg["DP_STRICT_BONUS"])
    rec = int(row.get("left_recover", 0)) + int(row.get("right_recover", 0))
    if rec == 2:
        reward += float(cfg.get("DP_BOTH_RECOVER_BONUS", 1.2))
    elif rec == 1:
        if not ultra_empty:
            reward -= float(cfg.get("DP_ONE_SIDE_RECOVER_PENALTY", 0.8))
    else:
        reward -= (
            0.5 * float(cfg.get("DP_NO_RECOVER_PENALTY", 1.8))
            if ultra_empty
            else float(cfg.get("DP_NO_RECOVER_PENALTY", 1.8))
        )
    if (
        str(row.get("kind", "")) == "row_gap_proposal"
        and rec == 2
        and (int(row.get("continuous_rows", 0)) == 0)
    ):
        reward += float(cfg.get("DP_EMPTY_BONUS", 0.3))
    if ultra_empty:
        reward += float(cfg.get("DP_ULTRA_EMPTY_BONUS", 5.2))
    reward -= float(cfg["DP_CONTINUITY_PENALTY"]) * float(row.get("continuity_penalty", 0.0))
    return float(reward)


def segment_cost(
    length: float,
    typical_len: float,
    mode_info: Dict[str, Any],
    cfg: Dict,
    is_edge: bool = False,
) -> float:
    length = float(length)
    if length <= 0:
        return 1000000000000.0
    if (
        not np.isfinite(typical_len)
        or typical_len <= 1e-06
        or mode_info.get("length_prior_strength") == "none"
    ):
        if length < float(cfg["DP_NO_PRIOR_MIN_SEGMENT_ABS"]):
            return 100000000.0 + (float(cfg["DP_NO_PRIOR_MIN_SEGMENT_ABS"]) - length)
        return 0.0
    ratio = length / max(typical_len, 1e-06)
    if ratio < float(cfg["DP_SHORT_RATIO_HARD"]):
        return 10000000.0 + 1000.0 * (float(cfg["DP_SHORT_RATIO_HARD"]) - ratio)
    if mode_info.get("length_prior_strength") in {"strong", "moderate"} and ratio > float(
        cfg["DP_LONG_RATIO_HARD"]
    ):
        return 50000.0 + 100.0 * (ratio - float(cfg["DP_LONG_RATIO_HARD"]))
    w = float(mode_info["length_weight"])
    if is_edge:
        w *= float(cfg["DP_EDGE_LENGTH_WEIGHT_FACTOR"])
    cost = w * abs(math.log(max(ratio, 1e-06)))
    if ratio < float(cfg["DP_SHORT_RATIO_SOFT"]):
        cost += (
            float(cfg.get("DP_SHORT_PENALTY_SCALE", 8.0))
            * w
            * (
                (float(cfg["DP_SHORT_RATIO_SOFT"]) - ratio)
                / max(float(cfg["DP_SHORT_RATIO_SOFT"]), 1e-06)
            )
            ** 2
        )
    if ratio > float(cfg["DP_LONG_RATIO_SOFT"]):
        cost += (
            float(cfg.get("DP_LONG_PENALTY_SCALE", 3.0))
            * w
            * (
                (ratio - float(cfg["DP_LONG_RATIO_SOFT"]))
                / max(float(cfg["DP_LONG_RATIO_SOFT"]), 1e-06)
            )
            ** 2
        )
    return float(cost)


def smooth1d(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(x.astype(float), k, mode="same")


def find_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def _candidate_objective_with_fixed_neighbors(
    row: pd.Series,
    prev_s: float,
    next_s: float,
    typical_len: float,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> float:
    c = float(row.get("center_s", np.nan))
    if not np.isfinite(c):
        return -1e18
    left_len = c - float(prev_s)
    right_len = float(next_s) - c
    if left_len <= 0 or right_len <= 0:
        return -1e18
    left_is_edge = bool(row.get("_left_is_edge", False))
    right_is_edge = bool(row.get("_right_is_edge", False))
    return (
        candidate_reward(row, cfg, mode_info)
        - segment_cost(left_len, typical_len, mode_info, cfg, is_edge=left_is_edge)
        - segment_cost(right_len, typical_len, mode_info, cfg, is_edge=right_is_edge)
    )


def refine_selected_boundaries_locally(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Move selected boundaries within their local candidate pools without changing their number."""
    if (
        not bool(cfg.get("ENABLE_DP_LOCAL_BOUNDARY_REFINE", True))
        or selected is None
        or len(selected) == 0
        or (cands is None)
        or (len(cands) == 0)
    ):
        return (selected.copy() if selected is not None else pd.DataFrame(), pd.DataFrame())
    typical_len = float(mode_info.get("typical_len", np.nan))
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    all_c = cands.copy().sort_values("center_s").reset_index(drop=True)
    logs: List[Dict[str, Any]] = []
    for i in range(len(current)):
        old = current.iloc[i].copy()
        old_c = float(old["center_s"])
        prev_s = s_start if i == 0 else float(current.loc[i - 1, "center_s"])
        next_s = s_end if i == len(current) - 1 else float(current.loc[i + 1, "center_s"])
        if np.isfinite(typical_len) and typical_len > 1e-06:
            window = max(
                float(cfg.get("LOCAL_REFINE_WINDOW_ABS", 8500.0)),
                float(cfg.get("LOCAL_REFINE_WINDOW_RATIO", 0.18)) * typical_len,
            )
            min_child = max(
                float(cfg.get("DP_SEGMENT_MIN_ABS", 2500.0)),
                float(cfg.get("LOCAL_REFINE_MIN_SEGMENT_RATIO", 0.32)) * typical_len,
            )
        else:
            window = float(cfg.get("LOCAL_REFINE_WINDOW_ABS", 8500.0))
            min_child = float(cfg.get("DP_NO_PRIOR_MIN_SEGMENT_ABS", 3500.0))
        sub = all_c[
            (all_c["center_s"].astype(float) >= old_c - window)
            & (all_c["center_s"].astype(float) <= old_c + window)
        ].copy()
        sub = sub[
            (sub["center_s"].astype(float) > prev_s + min_child)
            & (sub["center_s"].astype(float) < next_s - min_child)
        ].copy()
        sub = sub[
            pd.to_numeric(sub.get("score", 0.0), errors="coerce")
            >= float(cfg.get("LOCAL_REFINE_MIN_SCORE", 8.0))
        ].copy()
        if len(sub) == 0:
            logs.append(
                {
                    "boundary_order": i,
                    "old_center_s": old_c,
                    "new_center_s": old_c,
                    "decision": "keep_no_candidate",
                }
            )
            continue
        sub["_left_is_edge"] = i == 0
        sub["_right_is_edge"] = i == len(current) - 1
        old["_left_is_edge"] = i == 0
        old["_right_is_edge"] = i == len(current) - 1
        old_obj = _candidate_objective_with_fixed_neighbors(
            old, prev_s, next_s, typical_len, mode_info, cfg
        )
        objs = []
        for _, rr in sub.iterrows():
            objs.append(
                _candidate_objective_with_fixed_neighbors(
                    rr, prev_s, next_s, typical_len, mode_info, cfg
                )
            )
        sub["_local_refine_obj"] = objs
        best = sub.sort_values(["_local_refine_obj", "score"], ascending=False).iloc[0]
        best_obj = float(best["_local_refine_obj"])
        improve = best_obj - old_obj
        if (
            improve >= float(cfg.get("LOCAL_REFINE_MIN_IMPROVE", 0.6))
            and abs(float(best["center_s"]) - old_c) > 1e-06
        ):
            new_row = best.drop(
                labels=["_local_refine_obj", "_left_is_edge", "_right_is_edge"], errors="ignore"
            ).to_dict()
            new_row["selected_stage"] = "dp_local_refined"
            new_row["refined_from_center_s"] = old_c
            current.loc[i, list(new_row.keys())] = list(new_row.values())
            decision = "replace_boundary"
        else:
            decision = "keep_no_improvement"
        logs.append(
            {
                "boundary_order": int(i),
                "old_center_s": float(old_c),
                "best_center_s": float(best["center_s"]),
                "new_center_s": float(current.loc[i, "center_s"]),
                "old_score": float(old.get("score", np.nan)),
                "best_score": float(best.get("score", np.nan)),
                "old_obj": float(old_obj),
                "best_obj": float(best_obj),
                "improve": float(improve),
                "best_kind": str(best.get("kind", "")),
                "best_ultra_empty": int(is_ultra_empty_candidate(best, cfg)),
                "decision": decision,
            }
        )
    current = current.sort_values("center_s").reset_index(drop=True)
    return (current, pd.DataFrame(logs))


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _count_points_between(work: pd.DataFrame, a: float, b: float) -> int:
    if work is None or len(work) == 0:
        return 0
    lo, hi = (min(float(a), float(b)), max(float(a), float(b)))
    s = work["s_rot"].astype(float)
    return int(((s >= lo) & (s <= hi)).sum())


def is_local_anchor_candidate(row: pd.Series, cfg: Dict) -> bool:
    """Check multirow breaks, continuity and peak stability without a global length prior."""
    try:
        if bool(cfg.get("LOCAL_ANCHOR_ALLOW_ULTRA_EMPTY", True)) and is_ultra_empty_candidate(
            row, cfg
        ):
            if _safe_int(row.get("continuous_rows", 999), 999) <= int(
                cfg.get("LOCAL_ANCHOR_MAX_CONTINUOUS_ROWS", 0)
            ):
                return True
        if _safe_float(row.get("score", 0.0), 0.0) < float(
            cfg.get("LOCAL_ANCHOR_MIN_SCORE", 13.8)
        ):
            return False
        if _safe_float(row.get("joint_drop", 1.0), 1.0) > float(
            cfg.get("LOCAL_ANCHOR_JOINT_DROP_MAX", 0.38)
        ):
            return False
        if _safe_int(row.get("n_strong_break", 0), 0) < int(
            cfg.get("LOCAL_ANCHOR_MIN_STRONG", 3)
        ):
            return False
        if _safe_int(row.get("n_weak_break", 0), 0) < int(cfg.get("LOCAL_ANCHOR_MIN_WEAK", 3)):
            return False
        if bool(cfg.get("LOCAL_ANCHOR_REQUIRE_BOTH_RECOVER", True)):
            if (
                _safe_int(row.get("left_recover", 0), 0) != 1
                or _safe_int(row.get("right_recover", 0), 0) != 1
            ):
                return False
        if _safe_int(row.get("continuous_rows", 999), 999) > int(
            cfg.get("LOCAL_ANCHOR_MAX_CONTINUOUS_ROWS", 0)
        ):
            return False
        if _safe_int(row.get("cluster_size", 1), 1) < int(
            cfg.get("LOCAL_ANCHOR_MIN_CLUSTER_SIZE", 4)
        ):
            return False
        if _safe_float(row.get("dominant_single_row_penalty", 0.0), 0.0) > float(
            cfg.get("LOCAL_ANCHOR_MAX_DOMINANT_SINGLE_ROW", 0.8)
        ):
            return False
        return True
    except Exception:
        return False


def _existing_center_array(selected: pd.DataFrame) -> np.ndarray:
    if selected is None or len(selected) == 0 or "center_s" not in selected.columns:
        return np.zeros((0,), dtype=float)
    vals = pd.to_numeric(selected["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    return np.sort(vals)


def merge_weak_oversplit_boundaries(
    work: pd.DataFrame, selected: pd.DataFrame, cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove weak cuts that create very short segments; preserve strong gap evidence."""
    current = selected.copy() if selected is not None else pd.DataFrame()
    current = (
        current.sort_values("center_s").reset_index(drop=True)
        if len(current)
        else current.copy()
    )
    logs: List[Dict[str, Any]] = []
    if (
        not bool(cfg.get("ENABLE_FINAL_WEAK_OVERSPLIT_MERGE", True))
        or current is None
        or len(current) == 0
        or (work is None)
        or (len(work) == 0)
    ):
        return (current, pd.DataFrame([{"decision": "disabled_or_empty"}]))
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    min_child_abs = float(cfg.get("FINAL_MERGE_MIN_CHILD_ABS", 9000.0))
    weak_score_max = float(cfg.get("FINAL_MERGE_WEAK_SCORE_MAX", 9.5))
    max_iter = int(cfg.get("FINAL_MERGE_MAX_ITER", 20))
    for it in range(max_iter):
        if len(current) == 0:
            break
        centers = _existing_center_array(current)
        cuts = [s_start] + centers.tolist() + [s_end]
        lens = np.array([cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)], dtype=float)
        short_idx = np.where(lens < min_child_abs)[0]
        if len(short_idx) == 0:
            break
        pid0 = int(short_idx[np.argmin(lens[short_idx])])
        b_candidates = []
        if pid0 - 1 >= 0:
            b_candidates.append(pid0 - 1)
        if pid0 < len(current):
            b_candidates.append(pid0)
        removed = False
        for bidx in b_candidates:
            row = current.iloc[bidx]
            strong = is_strong_boundary_for_final_keep(row, cfg)
            score = _safe_float(row.get("score", np.nan), np.nan)
            weak_enough = not strong and (not np.isfinite(score) or score <= weak_score_max)
            logs.append(
                {
                    "iter": it,
                    "short_plot_index0": pid0,
                    "short_plot_len": float(lens[pid0]),
                    "try_boundary_index": int(bidx),
                    "try_center_s": _safe_float(row.get("center_s", np.nan)),
                    "try_score": score,
                    "is_strong_keep": int(strong),
                    "decision": (
                        "drop_weak_boundary" if weak_enough else "keep_boundary_not_weak"
                    ),
                }
            )
            if weak_enough:
                current = (
                    current.drop(current.index[bidx])
                    .reset_index(drop=True)
                    .sort_values("center_s")
                    .reset_index(drop=True)
                )
                removed = True
                break
        if not removed:
            break
    if not logs:
        logs.append({"decision": "no_short_weak_split"})
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def _sequence_context_expected_length_from_current_cuts(
    work: pd.DataFrame, selected: pd.DataFrame, cfg: Dict
) -> float:
    """Estimate a local reference extent from the current boundary sequence."""
    if work is None or len(work) == 0:
        return np.nan
    centers = _existing_center_array(selected)
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    cuts = np.array([s_start] + centers.tolist() + [s_end], dtype=float)
    lens = np.diff(cuts)
    lens = lens[np.isfinite(lens) & (lens > 1e-06)]
    if len(lens) == 0:
        return np.nan
    q = float(cfg.get("FALSE_SPLIT_EXPECTED_LEN_QUANTILE", 0.58))
    q = min(max(q, 0.45), 0.75)
    core = lens.copy()
    if len(core) >= 5:
        lo, hi = np.quantile(core, [0.1, 0.88])
        trimmed = core[(core >= lo) & (core <= hi)]
        if len(trimmed) >= 3:
            core = trimmed
    return float(np.quantile(core, q))


def _sequence_context_row_coverage_between(
    work: pd.DataFrame, a: float, b: float, min_points_per_row: int = 5
) -> int:
    if work is None or len(work) == 0 or b <= a:
        return 0
    n_rows = int(CONFIG.get("MAIN_ROWS", 3))
    cov = 0
    for r in range(n_rows):
        m = (
            (work["row_main"].astype(int) == r)
            & (work["s_rot"].astype(float) > a)
            & (work["s_rot"].astype(float) <= b)
        )
        if int(m.sum()) >= int(min_points_per_row):
            cov += 1
    return int(cov)


def _sequence_context_is_ultra_empty_protected(row: pd.Series, cfg: Dict) -> bool:
    cls = str(row.get("candidate_class", ""))
    joint = _safe_float(row.get("joint_drop", np.nan), np.nan)
    cont = _safe_int(row.get("continuous_rows", 999), 999)
    strong = _safe_int(row.get("n_strong_break", 0), 0)
    weak = _safe_int(row.get("n_weak_break", 0), 0)
    return bool(
        cls == "A_ultra_empty"
        and np.isfinite(joint)
        and (joint <= float(cfg.get("FALSE_SPLIT_PROTECT_ULTRA_EMPTY_JOINT_DROP", 0.085)))
        and (cont <= int(cfg.get("FALSE_SPLIT_PROTECT_ULTRA_EMPTY_MAX_CONT_ROWS", 0)))
        and (strong >= 3)
        and (weak >= 3)
    )


def prune_false_splits_by_child_segment_quality(
    work: pd.DataFrame, selected: pd.DataFrame, cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test short adjacent segments against the merged segment and retain strong empty gaps."""
    current = selected.copy() if selected is not None else pd.DataFrame()
    current = (
        current.sort_values("center_s").reset_index(drop=True)
        if len(current)
        else current.copy()
    )
    if (
        not bool(cfg.get("ENABLE_FINAL_FALSE_SPLIT_RECHECK", True))
        or work is None
        or len(work) == 0
        or (len(current) == 0)
    ):
        return (current, pd.DataFrame([{"decision": "disabled_or_empty"}]))
    logs: List[Dict[str, Any]] = []
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    max_iter = int(cfg.get("FALSE_SPLIT_MAX_ITER", 8))
    for it in range(max_iter):
        centers = _existing_center_array(current)
        if len(centers) == 0:
            break
        expected = _sequence_context_expected_length_from_current_cuts(work, current, cfg)
        if not np.isfinite(expected) or expected <= 1e-06:
            logs.append({"iter": it, "decision": "stop_no_expected_length"})
            break
        cuts = np.array([s_start] + centers.tolist() + [s_end], dtype=float)
        lens = np.diff(cuts)
        child_max = expected * float(cfg.get("FALSE_SPLIT_CHILD_MAX_RATIO", 0.62))
        merged_min = expected * float(cfg.get("FALSE_SPLIT_MERGED_MIN_RATIO", 0.72))
        merged_max = expected * float(cfg.get("FALSE_SPLIT_MERGED_MAX_RATIO", 1.45))
        candidates: List[Dict[str, Any]] = []
        for bidx in range(len(current)):
            left_len = float(lens[bidx])
            right_len = float(lens[bidx + 1])
            merged_len = left_len + right_len
            row = current.iloc[bidx]
            center = _safe_float(row.get("center_s", np.nan), np.nan)
            cont = _safe_int(row.get("continuous_rows", 999), 999)
            edgefit_fb = _safe_int(row.get("edgefit_fallback", 1), 1)
            ultra_protect = _sequence_context_is_ultra_empty_protected(row, cfg)
            merged_rows = _sequence_context_row_coverage_between(
                work, float(cuts[bidx]), float(cuts[bidx + 2]), min_points_per_row=5
            )
            merged_points = _count_points_between(
                work, float(cuts[bidx]), float(cuts[bidx + 2])
            )
            both_short = left_len <= child_max and right_len <= child_max
            merged_ok = merged_len >= merged_min and merged_len <= merged_max
            structure_ok = merged_rows >= int(
                cfg.get("FALSE_SPLIT_REQUIRE_MERGED_ROW_COVERAGE", 2)
            ) and merged_points >= int(cfg.get("FALSE_SPLIT_MIN_MERGED_POINTS", 120))
            not_clear_gap = cont >= int(
                cfg.get("FALSE_SPLIT_MIN_BOUNDARY_CONT_ROWS", 1)
            ) or edgefit_fb >= int(cfg.get("FALSE_SPLIT_MIN_EDGEFIT_FALLBACK", 0))
            should_drop = bool(
                both_short
                and merged_ok
                and structure_ok
                and (not ultra_protect)
                and not_clear_gap
            )
            info = {
                "iter": int(it),
                "boundary_index": int(bidx),
                "center_s": float(center) if np.isfinite(center) else np.nan,
                "score": _safe_float(row.get("score", np.nan), np.nan),
                "candidate_class": str(row.get("candidate_class", "")),
                "kind": str(row.get("kind", "")),
                "left_len": float(left_len),
                "right_len": float(right_len),
                "merged_len": float(merged_len),
                "expected_len_adaptive": float(expected),
                "left_ratio": float(left_len / expected),
                "right_ratio": float(right_len / expected),
                "merged_ratio": float(merged_len / expected),
                "continuous_rows": int(cont),
                "edgefit_fallback": int(edgefit_fb),
                "ultra_empty_protected": int(ultra_protect),
                "merged_row_coverage": int(merged_rows),
                "merged_points": int(merged_points),
                "both_children_short": int(both_short),
                "merged_ok": int(merged_ok),
                "structure_ok": int(structure_ok),
                "not_clear_gap": int(not_clear_gap),
                "decision": "drop_false_split_boundary" if should_drop else "keep",
            }
            logs.append(info)
            if should_drop:
                severity = (
                    (child_max - left_len) / max(child_max, 1e-06)
                    + (child_max - right_len) / max(child_max, 1e-06)
                    - abs(merged_len - expected) / max(expected, 1e-06)
                )
                info["severity"] = float(severity)
                candidates.append({"bidx": bidx, "severity": severity})
        if not candidates:
            break
        best = max(candidates, key=lambda z: z["severity"])
        drop_idx = int(best["bidx"])
        current = (
            current.drop(current.index[drop_idx])
            .reset_index(drop=True)
            .sort_values("center_s")
            .reset_index(drop=True)
        )
        logs.append(
            {
                "iter": int(it),
                "drop_boundary_index": drop_idx,
                "decision": "applied_drop_false_split_boundary",
            }
        )
    if not logs:
        logs.append({"decision": "no_action"})
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def diagnose_final_partition(
    work: pd.DataFrame, selected: pd.DataFrame, cands: pd.DataFrame, cfg: Dict
) -> pd.DataFrame:
    """Report strong unselected candidates inside the final plots."""
    if work is None or len(work) == 0:
        return pd.DataFrame()
    centers = _existing_center_array(selected)
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    cuts = [s_start] + centers.tolist() + [s_end]
    anchors = (
        cands[cands.apply(lambda r: is_local_anchor_candidate(r, cfg), axis=1)].copy()
        if cands is not None and len(cands)
        else pd.DataFrame()
    )
    rows = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        sub = (
            anchors[
                (anchors["center_s"].astype(float) > a)
                & (anchors["center_s"].astype(float) < b)
            ].copy()
            if len(anchors)
            else pd.DataFrame()
        )
        rows.append(
            {
                "plot_id": int(i),
                "s_start": float(a),
                "s_end": float(b),
                "length": float(b - a),
                "point_count": int(_count_points_between(work, a, b)),
                "residual_local_anchor_count": int(len(sub)),
                "best_residual_anchor_center_s": (
                    float(sub.sort_values("score", ascending=False).iloc[0]["center_s"])
                    if len(sub)
                    else np.nan
                ),
                "best_residual_anchor_score": (
                    float(sub.sort_values("score", ascending=False).iloc[0]["score"])
                    if len(sub)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _count_sorted_between(arr: np.ndarray, a: float, b: float) -> int:
    if arr is None or len(arr) == 0:
        return 0
    lo, hi = (min(float(a), float(b)), max(float(a), float(b)))
    return int(np.searchsorted(arr, hi, side="right") - np.searchsorted(arr, lo, side="left"))


def _row_support_between_ctx(
    ctx: Dict[str, Any], a: float, b: float, cfg: Dict
) -> Tuple[int, List[int]]:
    counts = []
    for arr in ctx.get("row_s", []):
        counts.append(_count_sorted_between(arr, a, b))
    rows_present = int(sum((c >= int(cfg.get("SEGMENT_ROW_MIN_POINTS", 5)) for c in counts)))
    return (rows_present, counts)


def _internal_anchor_weight_between_ctx(
    ctx: Dict[str, Any], a: float, b: float, cfg: Dict
) -> Tuple[float, float, int]:
    centers = ctx.get("centers", np.zeros((0,), dtype=float))
    if len(centers) == 0:
        return (0.0, 0.0, 0)
    lo, hi = (min(float(a), float(b)), max(float(a), float(b)))
    width = max(hi - lo, 1.0)
    margin = max(
        float(cfg.get("SEGMENT_INTERNAL_ANCHOR_MARGIN_ABS", 3200.0)),
        float(cfg.get("SEGMENT_INTERNAL_ANCHOR_MARGIN_RATIO", 0.08)) * width,
    )
    lo2, hi2 = (lo + margin, hi - margin)
    if hi2 <= lo2:
        return (0.0, 0.0, 0)
    l = int(np.searchsorted(centers, lo2, side="left"))
    r = int(np.searchsorted(centers, hi2, side="right"))
    if r <= l:
        return (0.0, 0.0, 0)
    w = float(ctx["prefix_anchor"][r] - ctx["prefix_anchor"][l])
    sw = float(ctx["prefix_anchor_score"][r] - ctx["prefix_anchor_score"][l])
    n = int(np.sum(ctx.get("anchor_weight", np.zeros((0,), dtype=float))[l:r] > 0))
    return (w, sw, n)


def correct_typical_length_by_harmonics(
    typical_len: float, cands: pd.DataFrame, cfg: Dict, meta: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Check autocorrelation harmonics against observed candidate spacing."""
    out_meta: Dict[str, Any] = {"harmonic_corrected": 0, "harmonic_reason": "not_checked"}
    if not bool(cfg.get("ENABLE_HARMONIC_LENGTH_CORRECTION", True)):
        out_meta["harmonic_reason"] = "disabled"
        return (typical_len, out_meta)
    if (
        not np.isfinite(typical_len)
        or typical_len <= 1e-06
        or cands is None
        or (len(cands) < 3)
    ):
        out_meta["harmonic_reason"] = "no_reliable_typical_or_candidates"
        return (typical_len, out_meta)
    strong = cands[
        cands.get("candidate_class_rank", pd.Series(np.zeros(len(cands), dtype=int))).astype(
            int
        )
        >= 3
    ].copy()
    if len(strong) < 3:
        out_meta["harmonic_reason"] = "insufficient_strong_candidates"
        return (typical_len, out_meta)
    centers = np.sort(strong["center_s"].to_numpy(dtype=float))
    ds = np.diff(centers)
    ds = ds[np.isfinite(ds) & (ds >= float(cfg.get("HARMONIC_MIN_LEN", 12000.0)))]
    if len(ds) == 0:
        out_meta["harmonic_reason"] = "no_valid_intervals"
        return (typical_len, out_meta)
    best_len = float(typical_len)
    best_support = 0
    tol_cfg = float(cfg.get("HARMONIC_SUPPORT_TOL", 0.34))
    for d in cfg.get("HARMONIC_DIVISORS", [2, 3]):
        cand_len = float(typical_len) / float(d)
        if cand_len < float(cfg.get("HARMONIC_MIN_LEN", 12000.0)):
            continue
        tol = max(2500.0, tol_cfg * cand_len)
        support = int(np.sum(np.abs(ds - cand_len) <= tol))
        if support > best_support:
            best_support = support
            best_len = cand_len
    support_fraction = float(best_support) / max(float(len(ds)), 1.0)
    if (
        best_support >= int(cfg.get("HARMONIC_MIN_SUPPORT_INTERVALS", 8))
        and support_fraction >= float(cfg.get("HARMONIC_MIN_SUPPORT_FRACTION", 0.28))
        and (best_len < 0.85 * float(typical_len))
    ):
        out_meta.update(
            {
                "harmonic_corrected": 1,
                "harmonic_reason": "strong_candidate_intervals_support_subharmonic",
                "typical_len_before_harmonic": float(typical_len),
                "typical_len_after_harmonic": float(best_len),
                "harmonic_support_intervals": int(best_support),
                "harmonic_support_fraction": float(support_fraction),
            }
        )
        return (float(best_len), out_meta)
    out_meta.update(
        {
            "harmonic_reason": "no_supported_subharmonic",
            "harmonic_support_intervals": int(best_support),
            "harmonic_support_fraction": float(support_fraction),
        }
    )
    return (typical_len, out_meta)


def run_graph_dp_interval(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    cfg: Dict,
    mode_info: Dict[str, Any],
    s_start: float,
    s_end: float,
    typical_len: float,
    stage: str = "graph_dp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select candidate boundaries by dynamic programming within an interval."""
    if cands is None or len(cands) == 0:
        return (pd.DataFrame(), pd.DataFrame([{"stage": stage, "decision": "no_candidates"}]))
    pool = _select_candidate_pool_for_graph(cands, cfg)
    pool = (
        pool[
            (pool["center_s"].astype(float) > float(s_start))
            & (pool["center_s"].astype(float) < float(s_end))
        ]
        .copy()
        .sort_values("center_s")
        .reset_index(drop=True)
    )
    if len(pool) == 0:
        return (pool, pd.DataFrame([{"stage": stage, "decision": "no_candidates_in_interval"}]))
    ctx = _make_graph_context(work, pool, cfg)
    centers = pool["center_s"].to_numpy(dtype=float)
    n = len(pool)
    rewards = np.array(
        [graph_candidate_reward(pool.iloc[i], cfg, mode_info) for i in range(n)], dtype=float
    )
    pool["graph_boundary_reward"] = rewards
    if (
        np.isfinite(typical_len)
        and typical_len > 1e-06
        and (mode_info.get("length_prior_strength") in {"strong", "moderate", "weak"})
    ):
        min_spacing = max(
            float(cfg.get("GRAPH_MIN_CENTER_SPACING_ABS", 1800.0)),
            float(cfg.get("GRAPH_DYNAMIC_MIN_SPACING_RATIO", 0.2)) * float(typical_len),
        )
    else:
        min_spacing = float(cfg.get("GRAPH_MIN_CENTER_SPACING_ABS", 1800.0))
    dp = np.full(n, -1e18, dtype=float)
    parent = np.full(n, -1, dtype=int)
    trans_cost = np.full(n, np.nan, dtype=float)
    for i in range(n):
        cost0 = graph_segment_quality_cost(
            ctx, s_start, centers[i], typical_len, mode_info, cfg, is_edge=True
        )
        best_val = rewards[i] - cost0
        best_parent = -1
        best_cost = cost0
        for j in range(i):
            if centers[i] - centers[j] < min_spacing:
                continue
            seg_cost = graph_segment_quality_cost(
                ctx, centers[j], centers[i], typical_len, mode_info, cfg, is_edge=False
            )
            val = dp[j] + rewards[i] - seg_cost
            if val > best_val:
                best_val = val
                best_parent = j
                best_cost = seg_cost
        dp[i] = best_val
        parent[i] = best_parent
        trans_cost[i] = best_cost
    no_boundary_score = -graph_segment_quality_cost(
        ctx, s_start, s_end, typical_len, mode_info, cfg, is_edge=True
    )
    best_end_score = no_boundary_score
    best_last = -1
    best_end_cost = -no_boundary_score
    for i in range(n):
        end_cost = graph_segment_quality_cost(
            ctx, centers[i], s_end, typical_len, mode_info, cfg, is_edge=True
        )
        val = dp[i] - end_cost
        if val > best_end_score:
            best_end_score = val
            best_last = i
            best_end_cost = end_cost
    path = []
    cur = best_last
    while cur >= 0:
        path.append(cur)
        cur = int(parent[cur])
    path = list(reversed(path))
    selected = (
        pool.iloc[path].copy().reset_index(drop=True)
        if path
        else pd.DataFrame(columns=pool.columns)
    )
    if len(selected):
        selected["selected_stage"] = stage
    trace_rows = []
    for i in range(n):
        trace_rows.append(
            {
                "stage": stage,
                "candidate_index": int(i),
                "center_s": float(centers[i]),
                "score": _safe_float(pool.iloc[i].get("score", np.nan)),
                "candidate_class": str(pool.iloc[i].get("candidate_class", "")),
                "graph_boundary_reward": float(rewards[i]),
                "dp_score": float(dp[i]),
                "parent": int(parent[i]),
                "transition_segment_cost": (
                    float(trans_cost[i]) if np.isfinite(trans_cost[i]) else np.nan
                ),
                "selected": int(i in set(path)),
            }
        )
    trace_rows.append(
        {
            "stage": stage,
            "candidate_index": "final",
            "n_selected": int(len(selected)),
            "best_end_score": float(best_end_score),
            "no_boundary_score": float(no_boundary_score),
            "best_end_cost": float(best_end_cost),
            "min_spacing": float(min_spacing),
            "decision": "graph_dp_done",
        }
    )
    return (selected.sort_values("center_s").reset_index(drop=True), pd.DataFrame(trace_rows))


def run_graph_dp_select_boundaries(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    classified = classify_boundary_candidates(cands, cfg)
    typical_len, len_meta = robust_typical_length_from_candidates(classified, work, cfg)
    typical_len2, harmonic_meta = correct_typical_length_by_harmonics(
        typical_len, classified, cfg, len_meta
    )
    mode_info = determine_dp_mode(work, classified, typical_len2, cfg)
    mode_info.update(len_meta)
    mode_info.update(harmonic_meta)
    mode_info["typical_len"] = float(typical_len2) if np.isfinite(typical_len2) else np.nan
    if np.isfinite(typical_len) and typical_len != typical_len2:
        mode_info["typical_len_raw_before_harmonic"] = float(typical_len)
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    selected, trace = run_graph_dp_interval(
        work,
        classified,
        cfg,
        mode_info,
        s_start,
        s_end,
        mode_info.get("typical_len", np.nan),
        stage="global_graph_dp",
    )
    return (selected, trace, mode_info)


def _graph_evidence_is_row_gap(row: pd.Series) -> bool:
    return str(row.get("kind", "")).lower() == "row_gap_proposal"


def _graph_evidence_recovery_count(row: pd.Series) -> int:
    return _safe_int(row.get("left_recover", 0), 0) + _safe_int(row.get("right_recover", 0), 0)


def _graph_evidence_inside_total(row: pd.Series, cfg: Dict) -> int:
    return int(
        sum(
            (
                _safe_int(row.get(f"row{r}_inside_ct", 0), 0)
                for r in range(int(cfg.get("MAIN_ROWS", 3)))
            )
        )
    )


def _graph_evidence_row_gap_strict_promotable(row: pd.Series, cfg: Dict) -> bool:
    """Require strong empty-gap evidence before promoting a row-gap proposal."""
    if not _graph_evidence_is_row_gap(row):
        return False
    score = _safe_float(row.get("score", 0.0), 0.0)
    joint = _safe_float(row.get("joint_drop", 1.0), 1.0)
    strong = _safe_int(row.get("n_strong_break", 0), 0)
    weak = _safe_int(row.get("n_weak_break", 0), 0)
    cont = _safe_int(row.get("continuous_rows", 999), 999)
    rec = _graph_evidence_recovery_count(row)
    inside_total = _graph_evidence_inside_total(row, cfg)
    if bool(cfg.get("ROWGAP_STRICT_PROMOTE_REQUIRE_BOTH_RECOVER", True)) and rec < 2:
        return False
    return bool(
        score >= float(cfg.get("ROWGAP_STRICT_PROMOTE_MIN_SCORE", 14.2))
        and joint <= float(cfg.get("ROWGAP_STRICT_PROMOTE_JOINT_DROP_MAX", 0.18))
        and (strong >= int(cfg.get("ROWGAP_STRICT_PROMOTE_MIN_STRONG", 3)))
        and (weak >= int(cfg.get("ROWGAP_STRICT_PROMOTE_MIN_WEAK", 3)))
        and (cont <= int(cfg.get("ROWGAP_STRICT_PROMOTE_MAX_CONTINUOUS_ROWS", 0)))
        and (inside_total <= int(cfg.get("ROWGAP_STRICT_PROMOTE_MAX_INSIDE_TOTAL", 18)))
    )


def candidate_boundary_class(row: pd.Series, cfg: Dict) -> str:
    """Classify boundary evidence; row-gap proposals remain auxiliary candidates."""
    try:
        if (
            str(row.get("kind", "")) == "feature_gap_interval"
            or str(row.get("selected_stage", "")) == "feature_gap_feature_gap_interval_proposal"
        ):
            fscore = _safe_float(
                row.get("feature_gap_boundary_likeness", row.get("score", 0.0)), 0.0
            )
            gap_w = _safe_float(row.get("feature_gap_gap_width", row.get("width", 0.0)), 0.0)
            active_le1 = _safe_float(row.get("feature_gap_active_le1_frac", 0.0), 0.0)
            post_rows = _safe_int(row.get("feature_gap_post_dense_rows", 0), 0)
            if (
                fscore >= float(cfg.get("FEATURE_GAP_FEATURE_A_MIN_SCORE", 13.0))
                and gap_w >= float(cfg.get("FEATURE_GAP_FEATURE_A_MIN_WIDTH", 4200.0))
                and (
                    active_le1
                    >= float(cfg.get("FEATURE_GAP_FEATURE_A_MIN_ACTIVE_LE1_FRAC", 0.58))
                )
                and (post_rows >= int(cfg.get("FEATURE_GAP_FEATURE_A_MIN_POST_DENSE_ROWS", 2)))
            ):
                return "A_feature_gap"
            if fscore >= float(cfg.get("FEATURE_GAP_FEATURE_B_MIN_SCORE", 10.8)):
                return "B_feature_gap"
            return "D_low"
        if (
            str(row.get("kind", "")) == "observed_gap_interval"
            or str(row.get("selected_stage", ""))
            == "observed_proposal_observed_gap_interval_proposal"
        ):
            support = _safe_int(
                row.get(
                    "observed_proposal_observed_gap_support_rows",
                    row.get("row_gap_support_row_count", 0),
                ),
                0,
            )
            med_w = _safe_float(
                row.get("observed_proposal_observed_gap_median_width", row.get("width", 0.0)),
                0.0,
            )
            if support >= int(
                cfg.get("OBSERVED_SEQUENCE_OBS_GAP_MIN_SUPPORT_ROWS", 2)
            ) and med_w >= float(cfg.get("OBSERVED_SEQUENCE_OBS_GAP_MIN_WIDTH_ABS", 6500.0)):
                return "A_observed_gap"
            return "B_observed_gap_weak"
        score = _safe_float(row.get("score", 0.0), 0.0)
        joint = _safe_float(row.get("joint_drop", 1.0), 1.0)
        strong = _safe_int(row.get("n_strong_break", 0), 0)
        weak = _safe_int(row.get("n_weak_break", 0), 0)
        cont = _safe_int(row.get("continuous_rows", 999), 999)
        rec = _graph_evidence_recovery_count(row)
        if _graph_evidence_is_row_gap(row):
            if _graph_evidence_row_gap_strict_promotable(row, cfg):
                return "B_rowgap_strict_aux"
            support_rows = _safe_int(row.get("row_gap_support_row_count", 0), 0)
            if (
                score >= float(cfg.get("ROWGAP_C_USABLE_MIN_SCORE", 9.2))
                and support_rows >= int(cfg.get("ROW_GAP_MIN_ROWS", 2))
                and (cont <= int(cfg.get("ROWGAP_C_MAX_CONTINUOUS_ROWS", 1)))
            ):
                return "C_row_gap_aux"
            return "D_low"
        if is_ultra_empty_candidate(row, cfg):
            return "A_ultra_empty"
        if is_high_conf_candidate(row, cfg):
            return "A_strong"
        if (
            score >= float(cfg.get("GRAPH_B_MIN_SCORE", 10.8))
            and joint <= float(cfg.get("GRAPH_B_JOINT_DROP_MAX", 0.22))
            and (strong >= int(cfg.get("GRAPH_B_MIN_STRONG", 2)))
            and (weak >= int(cfg.get("GRAPH_B_MIN_WEAK", 3)))
            and (cont <= int(cfg.get("GRAPH_B_MAX_CONTINUOUS_ROWS", 1)))
            and (rec >= 1 or joint <= 0.08)
        ):
            return "B_strong_one_side"
        if score >= float(cfg.get("GRAPH_C_WEAK_MIN_SCORE", 7.8)):
            return "C_weak_evidence"
        return "D_low"
    except Exception:
        return "D_low"


def candidate_class_rank(cls: str) -> int:
    cls = str(cls)
    if cls.startswith("A_"):
        return 4
    if cls.startswith("B_"):
        return 3
    if cls == "C_weak_evidence":
        return 2
    if cls == "C_row_gap_aux":
        return 1
    return 0


def classify_boundary_candidates(cands: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return pd.DataFrame() if cands is None else cands.copy()
    out = cands.copy().reset_index(drop=True)
    classes = [candidate_boundary_class(out.iloc[i], cfg) for i in range(len(out))]
    out["candidate_class"] = classes
    out["candidate_class_rank"] = [candidate_class_rank(x) for x in classes]
    out["is_row_gap_candidate"] = [
        int(_graph_evidence_is_row_gap(out.iloc[i])) for i in range(len(out))
    ]
    out["is_graph_strong"] = out["candidate_class_rank"].astype(int) >= 3
    out["is_graph_usable"] = out["candidate_class_rank"].astype(int) >= 1
    return out


def _make_graph_context(work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict) -> Dict[str, Any]:
    """Make graph context."""
    s_all = (
        np.sort(work["s_rot"].to_numpy(dtype=float))
        if work is not None and len(work)
        else np.zeros((0,), dtype=float)
    )
    row_s = []
    for r in range(int(cfg.get("MAIN_ROWS", 3))):
        if work is None or len(work) == 0:
            row_s.append(np.zeros((0,), dtype=float))
        else:
            row_s.append(
                np.sort(
                    work.loc[work["row_main"].astype(int) == r, "s_rot"].to_numpy(dtype=float)
                )
            )
    cc = (
        classify_boundary_candidates(cands, cfg)
        .copy()
        .sort_values("center_s")
        .reset_index(drop=True)
        if cands is not None and len(cands)
        else pd.DataFrame()
    )
    centers = cc["center_s"].to_numpy(dtype=float) if len(cc) else np.zeros((0,), dtype=float)
    ranks = (
        cc.get("candidate_class_rank", pd.Series(np.zeros(len(cc), dtype=int))).to_numpy(
            dtype=int
        )
        if len(cc)
        else np.zeros((0,), dtype=int)
    )
    scores = (
        pd.to_numeric(cc.get("score", pd.Series(np.zeros(len(cc)))), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        if len(cc)
        else np.zeros((0,), dtype=float)
    )
    is_rowgap = (
        cc.get("is_row_gap_candidate", pd.Series(np.zeros(len(cc), dtype=int)))
        .to_numpy(dtype=int)
        .astype(bool)
        if len(cc)
        else np.zeros((0,), dtype=bool)
    )
    anchor_weight = np.zeros(len(cc), dtype=float)
    anchor_weight[ranks >= 4] = 1.0
    anchor_weight[(ranks == 3) & ~is_rowgap] = 0.55
    anchor_weight[(ranks == 3) & is_rowgap] = 0.18
    anchor_score_weight = anchor_weight * np.maximum(
        0.0, (scores - float(cfg.get("DP_SCORE_BASE", 5.8))) / 8.0
    )
    prefix_anchor = np.concatenate([[0.0], np.cumsum(anchor_weight)])
    prefix_anchor_score = np.concatenate([[0.0], np.cumsum(anchor_score_weight)])
    return {
        "s_all": s_all,
        "row_s": row_s,
        "cands": cc,
        "centers": centers,
        "anchor_weight": anchor_weight,
        "anchor_score_weight": anchor_score_weight,
        "prefix_anchor": prefix_anchor,
        "prefix_anchor_score": prefix_anchor_score,
    }


def graph_candidate_reward(row: pd.Series, cfg: Dict, mode_info: Dict[str, Any]) -> float:
    reward = candidate_reward(row, cfg, mode_info)
    cls = str(row.get("candidate_class", candidate_boundary_class(row, cfg)))
    is_rowgap = _graph_evidence_is_row_gap(row) or "rowgap" in cls.lower()
    if cls.startswith("A_"):
        reward += float(cfg.get("GRAPH_CLASS_A_BONUS", 2.0))
        if cls == "A_observed_gap":
            reward += float(cfg.get("OBSERVED_SEQUENCE_OBS_GAP_EXTRA_REWARD", 4.0))
    elif cls.startswith("B_"):
        reward += float(cfg.get("GRAPH_CLASS_B_BONUS", 0.85))
    elif cls == "C_weak_evidence":
        reward -= float(cfg.get("GRAPH_CLASS_WEAK_PENALTY", 0.65))
    elif cls == "C_row_gap_aux":
        reward -= float(cfg.get("GRAPH_CLASS_ROWGAP_PENALTY", 2.8))
    if is_rowgap:
        reward -= float(cfg.get("GRAPH_B_ROWGAP_PENALTY", 2.25))
    if str(row.get("selected_stage", "")):
        reward += float(cfg.get("GRAPH_BOUNDARY_STAGE_BONUS", 0.35))
    return float(reward)


def _select_candidate_pool_for_graph(cands: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return pd.DataFrame()
    cc = classify_boundary_candidates(cands, cfg).copy()
    cc = cc[cc["is_graph_usable"].astype(bool)].copy()
    if len(cc) == 0:
        return cc
    max_keep = int(cfg.get("GRAPH_MAX_CANDIDATES", 520))
    rowgap = (
        cc[cc.get("is_row_gap_candidate", 0).astype(int) == 1]
        .copy()
        .sort_values("score", ascending=False)
    )
    non_rowgap = cc[cc.get("is_row_gap_candidate", 0).astype(int) != 1].copy()
    strong = non_rowgap[non_rowgap["candidate_class_rank"].astype(int) >= 3].copy()
    weak = (
        non_rowgap[non_rowgap["candidate_class_rank"].astype(int) < 3]
        .copy()
        .sort_values("score", ascending=False)
    )
    max_rowgap = min(
        int(cfg.get("GRAPH_MAX_ROWGAP_CANDIDATES", 70)),
        int(max_keep * float(cfg.get("GRAPH_MAX_ROWGAP_FRACTION", 0.18))),
    )
    parts = []
    if len(strong):
        parts.append(strong.sort_values("score", ascending=False).head(max_keep))
    remain = max_keep - sum((len(p) for p in parts))
    if remain > 0 and len(weak):
        parts.append(weak.head(remain))
    remain = max_keep - sum((len(p) for p in parts))
    if remain > 0 and len(rowgap):
        parts.append(rowgap.head(min(max_rowgap, remain)))
    if not parts:
        out = cc.sort_values("score", ascending=False).head(max_keep).copy()
    else:
        out = pd.concat(parts, ignore_index=True, sort=False)
    return (
        out.drop_duplicates(subset=["center_s"], keep="first")
        .sort_values("center_s")
        .reset_index(drop=True)
    )


def graph_segment_quality_cost(
    ctx: Dict[str, Any],
    a: float,
    b: float,
    typical_len: float,
    mode_info: Dict[str, Any],
    cfg: Dict,
    is_edge: bool = False,
) -> float:
    """Score a segment using extent, row coverage, point count and residual anchors."""
    lo, hi = (min(float(a), float(b)), max(float(a), float(b)))
    length = hi - lo
    if length <= 0:
        return 1000000000000.0
    if length < float(cfg.get("SEGMENT_SHORT_ABS_FORBID", 7000.0)):
        return 100000000.0 + (float(cfg.get("SEGMENT_SHORT_ABS_FORBID", 7000.0)) - length)
    cost = 0.0
    short_abs = float(cfg.get("SEGMENT_SHORT_ABS_STRONG_PENALTY", 13500.0))
    if not is_edge and length < short_abs:
        cost += (
            float(cfg.get("SEGMENT_SHORT_ABS_PENALTY", 10.0))
            * (short_abs - length)
            / max(short_abs, 1.0)
        )
    if bool(cfg.get("SEGMENT_USE_LENGTH_COST", True)):
        old_w = float(mode_info.get("length_weight", 0.0))
        mode2 = dict(mode_info)
        mode2["length_weight"] = old_w * float(cfg.get("SEGMENT_LENGTH_WEIGHT_SCALE", 0.55))
        cost += segment_cost(length, typical_len, mode2, cfg, is_edge=is_edge)
    point_count = _count_sorted_between(ctx.get("s_all", np.zeros((0,), dtype=float)), lo, hi)
    rows_present, row_counts = _row_support_between_ctx(ctx, lo, hi, cfg)
    min_points = (
        int(cfg.get("SEGMENT_MIN_POINTS_EDGE", 12))
        if is_edge
        else int(cfg.get("SEGMENT_MIN_POINTS_ABS", 30))
    )
    if point_count < min_points:
        cost += (
            float(cfg.get("SEGMENT_LOW_POINT_PENALTY", 5.5))
            * (min_points - point_count)
            / max(min_points, 1)
        )
    min_rows = int(cfg.get("SEGMENT_MIN_ROWS_PRESENT", 2))
    if not is_edge and rows_present < min_rows:
        cost += float(cfg.get("SEGMENT_ROW_MISSING_PENALTY", 6.5)) * (min_rows - rows_present)
    arr = np.asarray(row_counts, dtype=float)
    if not is_edge and point_count > 0 and (np.sum(arr > 0) >= 2):
        med_pos = np.median(arr[arr > 0])
        if med_pos > 0:
            imbalance = float(np.max(np.abs(arr - med_pos)) / max(med_pos, 1.0))
            cost += float(cfg.get("SEGMENT_ROW_BALANCE_PENALTY", 1.3)) * max(
                0.0, imbalance - 1.2
            )
    aw, asw, an = _internal_anchor_weight_between_ctx(ctx, lo, hi, cfg)
    if an > 0:
        anchor_cost = float(cfg.get("SEGMENT_INTERNAL_ANCHOR_PENALTY_A", 10.5)) * min(
            aw, 3.0
        ) + float(cfg.get("SEGMENT_INTERNAL_ANCHOR_PENALTY_B", 4.5)) * min(asw, 3.0)
        cost += min(float(cfg.get("SEGMENT_INTERNAL_ANCHOR_MAX_PENALTY", 28.0)), anchor_cost)
    return float(cost)


def _local_objective_for_boundary_set(
    work: pd.DataFrame,
    cands_for_ctx: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
    a: float,
    b: float,
) -> float:
    ctx = _make_graph_context(work, classify_boundary_candidates(cands_for_ctx, cfg), cfg)
    typical_len = _safe_float(mode_info.get("typical_len", np.nan), np.nan)
    centers = _existing_center_array(selected)
    centers = centers[(centers > min(a, b)) & (centers < max(a, b))]
    cuts = [float(a)] + centers.tolist() + [float(b)]
    obj = 0.0
    for i in range(len(cuts) - 1):
        obj -= graph_segment_quality_cost(
            ctx,
            cuts[i],
            cuts[i + 1],
            typical_len,
            mode_info,
            cfg,
            is_edge=i == 0 or i == len(cuts) - 2,
        )
    if selected is not None and len(selected):
        for _, row in classify_boundary_candidates(selected, cfg).iterrows():
            obj += graph_candidate_reward(row, cfg, mode_info)
    return float(obj)


def reoptimize_segments_with_residual_candidates(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Accept a local boundary sequence only when its objective improves."""
    current = (
        classify_boundary_candidates(selected, cfg)
        .copy()
        .sort_values("center_s")
        .reset_index(drop=True)
        if selected is not None and len(selected)
        else pd.DataFrame()
    )
    cands_cls = classify_boundary_candidates(cands, cfg)
    logs: List[Dict[str, Any]] = []
    if (
        not bool(cfg.get("ENABLE_SEGMENT_REOPTIMIZATION", True))
        or cands_cls is None
        or len(cands_cls) == 0
    ):
        return (current, pd.DataFrame([{"decision": "disabled_or_empty"}]))
    max_rounds = int(cfg.get("SEGMENT_REOPT_MAX_ROUNDS", 3))
    typical_len = _safe_float(mode_info.get("typical_len", np.nan), np.nan)
    for rd in range(max_rounds):
        changed = False
        centers = _existing_center_array(current)
        cuts = [float(work["s_rot"].min())] + centers.tolist() + [float(work["s_rot"].max())]
        new_boundaries = []
        remove_ranges = []
        for si, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
            width = float(b - a)
            if width < float(cfg.get("SEGMENT_REOPT_MIN_WIDTH_ABS", 12000.0)):
                continue
            inside = cands_cls[
                (cands_cls["center_s"].astype(float) > a)
                & (cands_cls["center_s"].astype(float) < b)
            ].copy()
            if len(inside) == 0:
                continue
            residual = inside[
                (
                    inside["candidate_class_rank"].astype(int)
                    >= int(cfg.get("SEGMENT_REOPT_MIN_INTERNAL_CLASS_RANK", 3))
                )
                & (
                    inside["score"].astype(float)
                    >= float(cfg.get("SEGMENT_REOPT_MIN_INTERNAL_SCORE", 10.2))
                )
            ].copy()
            if len(residual) == 0:
                continue
            if len(inside) > int(cfg.get("SEGMENT_REOPT_MAX_CANDS_PER_SEGMENT", 80)):
                strong = inside[inside["candidate_class_rank"].astype(int) >= 3]
                weak = inside[inside["candidate_class_rank"].astype(int) < 3].sort_values(
                    "score", ascending=False
                )
                inside = pd.concat(
                    [
                        strong,
                        weak.head(
                            max(
                                0,
                                int(cfg.get("SEGMENT_REOPT_MAX_CANDS_PER_SEGMENT", 80))
                                - len(strong),
                            )
                        ),
                    ],
                    ignore_index=True,
                    sort=False,
                )
            local_sel, local_trace = run_graph_dp_interval(
                work,
                inside,
                cfg,
                mode_info,
                float(a),
                float(b),
                typical_len,
                stage=f"segment_reopt_graph_evidence_round{rd}_seg{si}",
            )
            old_inside = current[
                (current["center_s"].astype(float) > a)
                & (current["center_s"].astype(float) < b)
            ].copy()
            old_obj = _local_objective_for_boundary_set(
                work, inside, old_inside, mode_info, cfg, a, b
            )
            new_obj = _local_objective_for_boundary_set(
                work, inside, local_sel, mode_info, cfg, a, b
            )
            accept = new_obj > old_obj + float(
                cfg.get("SEGMENT_REOPT_ACCEPT_MARGIN", 0.1)
            ) and (
                len(local_sel) > 0 or bool(cfg.get("SEGMENT_REOPT_ALLOW_EMPTY_REPLACE", False))
            )
            logs.append(
                {
                    "round": int(rd),
                    "segment_index": int(si),
                    "s_start": float(a),
                    "s_end": float(b),
                    "width": float(width),
                    "n_residual_strong": int(len(residual)),
                    "n_old_inside": int(len(old_inside)),
                    "n_new_inside": int(len(local_sel)),
                    "best_residual_center_s": float(
                        residual.sort_values("score", ascending=False).iloc[0]["center_s"]
                    ),
                    "best_residual_score": float(
                        residual.sort_values("score", ascending=False).iloc[0]["score"]
                    ),
                    "old_obj": float(old_obj),
                    "new_obj": float(new_obj),
                    "delta": float(new_obj - old_obj),
                    "decision": (
                        "replace_segment_boundaries_by_local_graph_dp_graph_evidence"
                        if accept
                        else "keep_current_segment"
                    ),
                }
            )
            if accept:
                remove_ranges.append((float(a), float(b)))
                if len(local_sel):
                    new_boundaries.append(local_sel)
                changed = True
        if remove_ranges:
            keep = np.ones(len(current), dtype=bool)
            for a, b in remove_ranges:
                keep &= ~(
                    (current["center_s"].astype(float) > a)
                    & (current["center_s"].astype(float) < b)
                )
            current = current.loc[keep].copy()
        if new_boundaries:
            current = (
                pd.concat([current] + new_boundaries, ignore_index=True, sort=False)
                .sort_values("center_s")
                .reset_index(drop=True)
            )
        if not changed:
            break
    if not logs:
        logs.append({"decision": "no_residual_segments_need_reoptimization"})
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def validate_boundary_sequence_locally(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Review selected boundaries for removal or replacement by nearby candidates."""
    current = (
        classify_boundary_candidates(selected, cfg)
        .copy()
        .sort_values("center_s")
        .reset_index(drop=True)
        if selected is not None and len(selected)
        else pd.DataFrame()
    )
    cands_cls = classify_boundary_candidates(cands, cfg)
    logs: List[Dict[str, Any]] = []
    if (
        not bool(cfg.get("ENABLE_FINAL_BOUNDARY_SEQUENCE_VALIDATION", True))
        or len(current) == 0
    ):
        return (current, pd.DataFrame([{"decision": "disabled_or_empty"}]))
    s0_global = float(work["s_rot"].min())
    s1_global = float(work["s_rot"].max())
    max_iter = int(cfg.get("FINAL_VALIDATE_MAX_ITER", 24))
    for it in range(max_iter):
        best = None
        for bi in range(len(current)):
            row = current.iloc[bi]
            c = float(row["center_s"])
            prev_s = s0_global if bi == 0 else float(current.iloc[bi - 1]["center_s"])
            next_s = (
                s1_global if bi == len(current) - 1 else float(current.iloc[bi + 1]["center_s"])
            )
            local_sel = current[
                (current["center_s"].astype(float) > prev_s)
                & (current["center_s"].astype(float) < next_s)
            ].copy()
            keep_obj = _local_objective_for_boundary_set(
                work, cands_cls, local_sel, mode_info, cfg, prev_s, next_s
            )
            drop_sel = local_sel[local_sel["center_s"].astype(float) != c].copy()
            drop_obj = _local_objective_for_boundary_set(
                work, cands_cls, drop_sel, mode_info, cfg, prev_s, next_s
            )
            rank = _safe_int(
                row.get(
                    "candidate_class_rank",
                    candidate_class_rank(str(row.get("candidate_class", ""))),
                ),
                0,
            )
            margin = (
                float(cfg.get("FINAL_VALIDATE_DROP_MARGIN_STRONG", 2.2))
                if rank >= int(cfg.get("FINAL_VALIDATE_STRONG_CLASS_RANK", 3))
                else float(cfg.get("FINAL_VALIDATE_DROP_MARGIN_WEAK", 0.15))
            )
            improve = drop_obj - keep_obj
            entry = {
                "iter": int(it),
                "boundary_index": int(bi),
                "center_s": float(c),
                "score": _safe_float(row.get("score", np.nan)),
                "candidate_class": str(row.get("candidate_class", "")),
                "candidate_class_rank": int(rank),
                "prev_s": float(prev_s),
                "next_s": float(next_s),
                "keep_obj": float(keep_obj),
                "drop_obj": float(drop_obj),
                "drop_improve": float(improve),
                "required_margin": float(margin),
                "action": "drop",
                "decision": "candidate_drop" if improve > margin else "keep",
            }
            if improve > margin and (best is None or improve > best[0]):
                best = (improve, "drop", bi, None, entry)
            logs.append(entry)
            if bool(cfg.get("FINAL_VALIDATE_USE_NEIGHBOR_REPLACEMENT", True)):
                win = max(
                    float(cfg.get("FINAL_VALIDATE_REPLACE_WINDOW_ABS", 9000.0)),
                    float(cfg.get("FINAL_VALIDATE_REPLACE_WINDOW_RATIO", 0.16))
                    * max(float(mode_info.get("typical_len", 1.0)), 1.0),
                )
                cand_local = cands_cls[
                    (cands_cls["center_s"].astype(float) > max(prev_s, c - win))
                    & (cands_cls["center_s"].astype(float) < min(next_s, c + win))
                ].copy()
                cand_local = cand_local[cand_local["is_graph_usable"].astype(bool)].copy()
                if len(cand_local):
                    cand_local = cand_local[
                        ~(
                            (cand_local.get("is_row_gap_candidate", 0).astype(int) == 1)
                            & (cand_local["candidate_class_rank"].astype(int) < 3)
                        )
                    ].copy()
                if len(cand_local):
                    cand_local = cand_local[
                        np.abs(cand_local["center_s"].astype(float) - c)
                        > float(cfg.get("CAND_CLUSTER_TOL", 900.0))
                    ].copy()
                if len(cand_local):
                    cand_local = cand_local.sort_values(
                        ["candidate_class_rank", "score"], ascending=False
                    ).head(16)
                    best_rep_obj = -1e18
                    best_rep_row = None
                    for _, rr in cand_local.iterrows():
                        rep_sel = drop_sel.copy()
                        rep_row = rr.to_dict()
                        rep_row["selected_stage"] = "final_sequence_replace_graph_evidence"
                        rep_sel = (
                            pd.concat(
                                [rep_sel, pd.DataFrame([rep_row])],
                                ignore_index=True,
                                sort=False,
                            )
                            .sort_values("center_s")
                            .reset_index(drop=True)
                        )
                        obj = _local_objective_for_boundary_set(
                            work, cands_cls, rep_sel, mode_info, cfg, prev_s, next_s
                        )
                        if obj > best_rep_obj:
                            best_rep_obj = obj
                            best_rep_row = rep_row
                    rep_improve = float(best_rep_obj - keep_obj)
                    rep_margin = float(cfg.get("FINAL_VALIDATE_REPLACE_MARGIN", 0.25))
                    entry2 = {
                        "iter": int(it),
                        "boundary_index": int(bi),
                        "center_s": float(c),
                        "action": "replace",
                        "best_replace_center_s": (
                            float(best_rep_row.get("center_s", np.nan))
                            if best_rep_row is not None
                            else np.nan
                        ),
                        "keep_obj": float(keep_obj),
                        "replace_obj": float(best_rep_obj),
                        "replace_improve": float(rep_improve),
                        "required_margin": float(rep_margin),
                        "decision": "candidate_replace" if rep_improve > rep_margin else "keep",
                    }
                    logs.append(entry2)
                    if (
                        best_rep_row is not None
                        and rep_improve > rep_margin
                        and (best is None or rep_improve > best[0])
                    ):
                        best = (rep_improve, "replace", bi, best_rep_row, entry2)
        if best is None:
            break
        improve, action, bi, rep_row, entry = best
        if action == "drop":
            entry = dict(entry)
            entry["decision"] = "drop_after_final_sequence_validation_graph_evidence"
            logs.append(entry)
            current = (
                current.drop(current.index[bi])
                .reset_index(drop=True)
                .sort_values("center_s")
                .reset_index(drop=True)
            )
        elif action == "replace" and rep_row is not None:
            entry = dict(entry)
            entry["decision"] = "replace_after_final_sequence_validation_graph_evidence"
            logs.append(entry)
            current = current.drop(current.index[bi]).reset_index(drop=True)
            current = (
                pd.concat([current, pd.DataFrame([rep_row])], ignore_index=True, sort=False)
                .sort_values("center_s")
                .reset_index(drop=True)
            )
    if not logs:
        logs.append({"decision": "no_action"})
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def is_strong_boundary_for_final_keep(row: pd.Series, cfg: Dict) -> bool:
    """Is strong boundary for final keep."""
    if _graph_evidence_is_row_gap(row):
        return bool(
            _graph_evidence_row_gap_strict_promotable(row, cfg)
            and _safe_int(row.get("continuous_rows", 999), 999) == 0
        )
    cls = str(row.get("candidate_class", candidate_boundary_class(row, cfg)))
    return bool(
        cls.startswith("A_")
        or cls.startswith("B_")
        or str(row.get("selected_stage", ""))
        in {"dp_local_refined", "final_sequence_replace_graph_evidence"}
    )


def diagnose_unselected_strong_candidates(
    work: pd.DataFrame, selected: pd.DataFrame, cands: pd.DataFrame, cfg: Dict
) -> pd.DataFrame:
    cands_cls = classify_boundary_candidates(cands, cfg)
    if cands_cls is None or len(cands_cls) == 0:
        return pd.DataFrame()
    selected_centers = _existing_center_array(selected)
    strong = cands_cls[cands_cls["candidate_class_rank"].astype(int) >= 3].copy()
    rows = []
    for _, row in strong.iterrows():
        c = float(row["center_s"])
        dist = float(np.min(np.abs(selected_centers - c))) if len(selected_centers) else np.inf
        if dist <= float(cfg.get("FINAL_RESIDUAL_STRONG_MIN_DIST", 2500.0)):
            continue
        rows.append(
            {
                "center_s": c,
                "score": _safe_float(row.get("score", np.nan)),
                "candidate_class": str(row.get("candidate_class", "")),
                "kind": str(row.get("kind", "")),
                "nearest_selected_dist": dist,
                "joint_drop": _safe_float(row.get("joint_drop", np.nan)),
                "n_strong_break": _safe_int(row.get("n_strong_break", 0)),
                "n_weak_break": _safe_int(row.get("n_weak_break", 0)),
                "left_recover": _safe_int(row.get("left_recover", 0)),
                "right_recover": _safe_int(row.get("right_recover", 0)),
                "continuous_rows": _safe_int(row.get("continuous_rows", 999)),
                "row_gap_demoted": int(_graph_evidence_is_row_gap(row)),
                "diagnosis": "unselected_strong_candidate_need_review",
            }
        )
    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def fit_boundary_edges(df: pd.DataFrame, selected: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if len(selected) == 0:
        return selected.copy()
    if not bool(cfg.get("ENABLE_EDGE_FIT", True)):
        out = selected.copy()
        out["upper_edge_s"] = out["center_s"].astype(float)
        out["lower_edge_s"] = out["center_s"].astype(float)
        out["gap_width"] = 0.0
        out["edgefit_score"] = 0.0
        out["edgefit_fallback"] = 1
        return out
    work = df[df["keep_for_partition"]].copy().reset_index(drop=True)
    out_rows: List[Dict[str, Any]] = []
    bin_size = float(cfg["EDGEFIT_BIN"])
    win = int(cfg["EDGEFIT_SMOOTH_WIN"])
    for _, rr in selected.iterrows():
        row = rr.to_dict()
        if _safe_int(row.get("narrow_gap_narrow_gap", 0), 0) == 1:
            row.update(
                upper_edge_s=float(row["s0"]),
                lower_edge_s=float(row["s1"]),
                gap_width=float(row["s1"] - row["s0"]),
                edgefit_score=1.0,
                edgefit_fallback=0,
                narrow_gap_edge_source="observed_point_interval",
            )
            out_rows.append(row)
            continue
        s0 = float(row.get("s0", row["center_s"] - 0.5 * row.get("width", 2400.0)))
        s1 = float(row.get("s1", row["center_s"] + 0.5 * row.get("width", 2400.0)))
        center = float(row["center_s"])
        width = max(float(s1 - s0), 1.0)
        n_bins = max(8, int(np.ceil(width / max(bin_size, 1e-06))))
        edges = np.linspace(s0, s1, n_bins + 1, dtype=float)
        joint = np.zeros(n_bins, dtype=float)
        for r in range(int(cfg["MAIN_ROWS"])):
            svals = work.loc[work["row_main"].astype(int) == r, "s_rot"].to_numpy(dtype=float)
            local = svals[(svals >= s0) & (svals <= s1)]
            hist, _ = np.histogram(local, bins=edges)
            hist = smooth1d(hist.astype(float), win)
            ref = max(float(np.quantile(hist, 0.9)) if len(hist) else 1.0, 1.0)
            joint += hist / ref
        joint /= max(float(cfg["MAIN_ROWS"]), 1.0)
        inside_min = float(np.min(joint))
        inside_max = float(np.max(joint))
        dyn = max(inside_max - inside_min, 1e-06)
        thr = inside_min + float(cfg["EDGEFIT_ENTER_RATIO"]) * dyn
        mask = joint <= thr
        segs = find_true_segments(mask)
        best = None
        for a, b in segs:
            upper = float(edges[a])
            lower = float(edges[b + 1])
            gap_width = float(lower - upper)
            if gap_width < float(cfg["EDGEFIT_MIN_GAP_RATIO"]) * width:
                continue
            if upper <= s0 + float(cfg["EDGEFIT_BORDER_MARGIN"]) or lower >= s1 - float(
                cfg["EDGEFIT_BORDER_MARGIN"]
            ):
                continue
            left_peak = float(np.max(joint[:a])) if a > 0 else inside_min
            right_peak = float(np.max(joint[b + 1 :])) if b + 1 < len(joint) else inside_min
            valley_mean = float(np.mean(joint[a : b + 1]))
            depth = 1.0 - valley_mean / max(max(left_peak, right_peak), 1e-06)
            score = depth + 0.25 * (left_peak > thr) + 0.25 * (right_peak > thr)
            cand = {
                "upper_edge_s": upper,
                "lower_edge_s": lower,
                "gap_width": gap_width,
                "edgefit_score": float(score),
                "edgefit_fallback": 0,
            }
            if best is None or cand["edgefit_score"] > best["edgefit_score"]:
                best = cand
        if best is None:
            half = 0.35 * width
            best = {
                "upper_edge_s": center - half,
                "lower_edge_s": center + half,
                "gap_width": 2.0 * half,
                "edgefit_score": -5.0,
                "edgefit_fallback": 1,
            }
        row.update(best)
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("center_s").reset_index(drop=True)


def build_plot_boundaries_from_edges(
    df: pd.DataFrame, boundaries: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """while classifying points inside the separator interval into:
    - true_gap_point: central interval points, treated as inter-plot noise and not counted;
    - recovered_upper_edge_point / recovered_lower_edge_point: points close to one
      plot edge and supported by the same row structure, counted in the adjacent plot;
    - plot_edge_zone: normal plot-edge points outside the true gap.

    keep_for_partition is only used to build reliable boundary evidence. It must not
    be used as the final point-ownership filter, because curved outer rows can be
    marked as keep_for_partition=False by the light outer-range filter and then
    disappear from 06/11/13/18 outputs. Therefore, final plot assignment can use all
    rectified points while the boundary s-range is still estimated from the reliable
    partition subset."""
    work = df[df["keep_for_partition"]].copy().reset_index(drop=True)
    if len(work) == 0:
        work = df.copy().reset_index(drop=True)
    if len(work) == 0:
        return (pd.DataFrame(), pd.DataFrame())
    smin = float(work["s_rot"].min()) - 1e-06
    smax = float(work["s_rot"].max()) + 1e-06
    if bool(CONFIG.get("ASSIGNMENT_SOURCE_FINAL_ASSIGN_USE_ALL_POINTS", True)):
        out = df.copy().reset_index(drop=True)
        if bool(CONFIG.get("ASSIGNMENT_SOURCE_FINAL_ASSIGN_DROP_OUTSIDE_S_RANGE", True)):
            out = (
                out[(out["s_rot"].astype(float) >= smin) & (out["s_rot"].astype(float) <= smax)]
                .copy()
                .reset_index(drop=True)
            )
    else:
        out = work.copy()
    out["plot_id"] = -1
    out["assigned_plot_id_fullcover"] = -1
    out["is_boundary_gap_point"] = False
    out["is_boundary_zone_point"] = False
    out["is_counted_plot_point"] = True
    out["boundary_zone_center_s"] = np.nan
    out["boundary_zone_assigned_side"] = ""
    out["boundary_zone_class"] = "plot_point"
    out["boundary_gap_reason"] = ""
    out["boundary_reassign_reason"] = ""
    out["structural_gap_visible_gap_assignment"] = True
    out["true_gap_gap_edge_reassign"] = True
    out["point_assignment_center_separator_assignment"] = True
    if len(boundaries) == 0:
        out["plot_id"] = 1
        out["assigned_plot_id_fullcover"] = 1
        return (
            pd.DataFrame(
                [
                    {
                        "plot_id": 1,
                        "s_start": smin,
                        "s_end": smax,
                        "s_start_fullcover": smin,
                        "s_end_fullcover": smax,
                        "count": int(len(out)),
                        "length": float(smax - smin),
                        "point_assignment_assignment_mode": "single_plot_no_boundary",
                        "structural_gap_visible_gap_assignment": True,
                    }
                ]
            ),
            out,
        )
    b = boundaries.copy().sort_values("center_s").reset_index(drop=True)
    centers = b["center_s"].astype(float).to_numpy()
    centers = np.asarray([c for c in centers if np.isfinite(c)], dtype=float)
    centers = np.unique(np.clip(centers, smin, smax))
    centers.sort()
    centers_original = centers.copy()

    def _density_recentering_smooth1d_local(x: np.ndarray, win: int) -> np.ndarray:
        win = max(1, int(win))
        if win <= 1 or len(x) == 0:
            return x.astype(float)
        k = np.ones(win, dtype=float) / float(win)
        return np.convolve(x.astype(float), k, mode="same")

    def _density_recentering_low_segments(mask: np.ndarray):
        segs = []
        i = 0
        n = len(mask)
        while i < n:
            if not bool(mask[i]):
                i += 1
                continue
            j = i
            while j + 1 < n and bool(mask[j + 1]):
                j += 1
            segs.append((i, j))
            i = j + 1
        return segs

    def _density_recentering_recenter_one_boundary(
        center: float, rr: pd.Series, ref_len_local: float
    ) -> Tuple[float, Dict[str, Any]]:
        info: Dict[str, Any] = {
            "density_recentering_center_original": float(center),
            "density_recentering_center_relocated": float(center),
            "density_recentering_center_shift": 0.0,
            "density_recentering_center_reloc_reason": "disabled_or_no_change",
        }
        if _safe_int(rr.get("narrow_gap_narrow_gap", 0), 0) == 1:
            info["density_recentering_center_reloc_reason"] = (
                "narrow_gap_preserve_observed_narrow_gap"
            )
            return (float(center), info)
        if not bool(CONFIG.get("DENSITY_RECENTERING_ENABLE_DENSITY_VALLEY_RECENTER", True)):
            return (float(center), info)
        search_half = float(CONFIG.get("DENSITY_RECENTERING_RECENTER_SEARCH_HALF_ABS", 2600.0))
        search_half = min(
            search_half,
            float(CONFIG.get("DENSITY_RECENTERING_RECENTER_SEARCH_HALF_RATIO", 0.11))
            * max(ref_len_local, 1.0),
        )
        search_half = max(
            search_half,
            float(CONFIG.get("DENSITY_RECENTERING_RECENTER_SEARCH_HALF_MIN_ABS", 1800.0)),
        )
        max_shift = min(
            float(CONFIG.get("DENSITY_RECENTERING_RECENTER_MAX_SHIFT_ABS", 950.0)),
            float(CONFIG.get("DENSITY_RECENTERING_RECENTER_MAX_SHIFT_RATIO", 0.035))
            * max(ref_len_local, 1.0),
        )
        try:
            s0_rr = float(rr.get("s0", np.nan))
            s1_rr = float(rr.get("s1", np.nan))
        except Exception:
            s0_rr, s1_rr = (np.nan, np.nan)
        lo = float(center) - search_half
        hi = float(center) + search_half
        if np.isfinite(s0_rr) and np.isfinite(s1_rr) and (s1_rr > s0_rr):
            pad = float(CONFIG.get("DENSITY_RECENTERING_RECENTER_CANDIDATE_PAD_ABS", 900.0))
            lo = min(lo, s0_rr - pad)
            hi = max(hi, s1_rr + pad)
        lo = max(smin, lo)
        hi = min(smax, hi)
        if hi <= lo + 10:
            info["density_recentering_center_reloc_reason"] = "invalid_search_window"
            return (float(center), info)
        bin_size = float(CONFIG.get("DENSITY_RECENTERING_RECENTER_BIN", 70.0))
        smooth_win = int(CONFIG.get("DENSITY_RECENTERING_RECENTER_SMOOTH_WIN", 5))
        edges = np.arange(lo, hi + bin_size, bin_size, dtype=float)
        if len(edges) < 8:
            edges = np.linspace(lo, hi, 9, dtype=float)
        svals = work["s_rot"].to_numpy(dtype=float)
        hist, _ = np.histogram(svals, bins=edges)
        sm = _density_recentering_smooth1d_local(hist.astype(float), smooth_win)
        if len(sm) == 0 or not np.any(np.isfinite(sm)):
            info["density_recentering_center_reloc_reason"] = "empty_density_profile"
            return (float(center), info)
        q = float(CONFIG.get("DENSITY_RECENTERING_RECENTER_LOW_Q", 0.3))
        thr_q = float(np.quantile(sm, q))
        mn, mx = (float(np.min(sm)), float(np.max(sm)))
        dyn = max(mx - mn, 1e-06)
        thr_dyn = (
            mn + float(CONFIG.get("DENSITY_RECENTERING_RECENTER_LOW_DYN_RATIO", 0.32)) * dyn
        )
        thr = min(thr_q, thr_dyn)
        mask = sm <= thr
        segs = _density_recentering_low_segments(mask)
        if not segs:
            info["density_recentering_center_reloc_reason"] = "no_low_density_segment"
            return (float(center), info)
        best = None
        min_seg_width = float(
            CONFIG.get("DENSITY_RECENTERING_RECENTER_MIN_SEG_WIDTH_ABS", 210.0)
        )
        for a, bidx in segs:
            seg_lo = float(edges[a])
            seg_hi = float(edges[bidx + 1])
            seg_w = seg_hi - seg_lo
            if seg_w < min_seg_width:
                continue
            seg_c = 0.5 * (seg_lo + seg_hi)
            shift = seg_c - float(center)
            if abs(shift) > max_shift:
                continue
            valley_mean = float(np.mean(sm[a : bidx + 1]))
            score = (
                seg_w / max(bin_size, 1e-06)
                - 0.45 * valley_mean
                - 0.18 * abs(shift) / max(max_shift, 1.0)
            )
            if best is None or score > best[0]:
                best = (score, seg_c, seg_w, valley_mean, seg_lo, seg_hi, shift)
        if best is None:
            info["density_recentering_center_reloc_reason"] = (
                "no_valid_low_segment_after_shift_guard"
            )
            return (float(center), info)
        _, new_c, seg_w, valley_mean, seg_lo, seg_hi, shift = best
        min_shift = float(CONFIG.get("DENSITY_RECENTERING_RECENTER_MIN_ACCEPT_SHIFT_ABS", 80.0))
        if abs(shift) < min_shift:
            info.update(
                {
                    "density_recentering_center_relocated": float(center),
                    "density_recentering_center_shift": 0.0,
                    "density_recentering_center_reloc_reason": "low_segment_found_but_shift_small",
                    "density_recentering_valley_seg_lo": float(seg_lo),
                    "density_recentering_valley_seg_hi": float(seg_hi),
                    "density_recentering_valley_width": float(seg_w),
                    "density_recentering_valley_mean_density": float(valley_mean),
                }
            )
            return (float(center), info)
        info.update(
            {
                "density_recentering_center_relocated": float(new_c),
                "density_recentering_center_shift": float(shift),
                "density_recentering_center_reloc_reason": "density_valley_recentering",
                "density_recentering_valley_seg_lo": float(seg_lo),
                "density_recentering_valley_seg_hi": float(seg_hi),
                "density_recentering_valley_width": float(seg_w),
                "density_recentering_valley_mean_density": float(valley_mean),
            }
        )
        return (float(new_c), info)

    if len(centers) >= 2:
        prelim_ref_len = float(np.median(np.diff(centers)))
    else:
        prelim_ref_len = float((smax - smin) / max(len(centers) + 1, 1))
    if not np.isfinite(prelim_ref_len) or prelim_ref_len <= 1:
        prelim_ref_len = float((smax - smin) / max(len(centers) + 1, 1))
    b_centers_for_reloc = (
        b["center_s"].astype(float).to_numpy() if len(b) else np.zeros((0,), dtype=float)
    )
    reloc_rows: List[Dict[str, Any]] = []
    relocated_centers: List[float] = []
    for c0 in centers:
        if len(b_centers_for_reloc):
            ridx0 = int(np.nanargmin(np.abs(b_centers_for_reloc - c0)))
            rr0 = b.iloc[ridx0]
        else:
            rr0 = pd.Series({"center_s": c0})
        c_new, info = _density_recentering_recenter_one_boundary(float(c0), rr0, prelim_ref_len)
        relocated_centers.append(c_new)
        reloc_rows.append(info)
    centers = np.asarray(relocated_centers, dtype=float)
    centers = np.clip(centers, smin, smax)
    centers.sort()
    if len(centers) == 0:
        out["plot_id"] = 1
        out["assigned_plot_id_fullcover"] = 1
        return (
            pd.DataFrame(
                [
                    {
                        "plot_id": 1,
                        "s_start": smin,
                        "s_end": smax,
                        "s_start_fullcover": smin,
                        "s_end_fullcover": smax,
                        "count": int(len(out)),
                        "length": float(smax - smin),
                        "point_assignment_assignment_mode": "single_plot_invalid_centers",
                        "structural_gap_visible_gap_assignment": True,
                    }
                ]
            ),
            out,
        )
    full_starts = [smin] + centers.tolist()
    full_ends = centers.tolist() + [smax]
    n_plots = len(centers) + 1
    for pid, (a, c) in enumerate(zip(full_starts, full_ends), start=1):
        a = float(a)
        c = float(c)
        if pid == 1:
            m = (out["s_rot"] >= a) & (out["s_rot"] <= c)
        else:
            m = (out["s_rot"] > a) & (out["s_rot"] <= c)
        out.loc[m, "plot_id"] = pid
        out.loc[m, "assigned_plot_id_fullcover"] = pid
    miss = out["plot_id"].astype(int) < 0
    if miss.any():
        ss = out.loc[miss, "s_rot"].astype(float).to_numpy()
        pid_arr = np.searchsorted(centers, ss, side="left") + 1
        out.loc[miss, "plot_id"] = pid_arr.astype(int)
        out.loc[miss, "assigned_plot_id_fullcover"] = pid_arr.astype(int)
    if len(centers) >= 2:
        ref_len = float(np.median(np.diff(centers)))
    else:
        ref_len = float((smax - smin) / max(n_plots, 1))
    if not np.isfinite(ref_len) or ref_len <= 1:
        ref_len = float((smax - smin) / max(n_plots, 1))
    zone_max_abs = float(
        CONFIG.get(
            "STRUCTURAL_GAP_ZONE_HALF_MAX_ABS", CONFIG.get("TRUE_GAP_ZONE_HALF_MAX_ABS", 3000.0)
        )
    )
    zone_max_ratio = float(
        CONFIG.get(
            "STRUCTURAL_GAP_ZONE_HALF_MAX_RATIO",
            CONFIG.get("TRUE_GAP_ZONE_HALF_MAX_RATIO", 0.05),
        )
    )
    zone_min_abs = float(
        CONFIG.get(
            "STRUCTURAL_GAP_ZONE_HALF_MIN_ABS", CONFIG.get("TRUE_GAP_ZONE_HALF_MIN_ABS", 700.0)
        )
    )
    zone_default = float(
        CONFIG.get(
            "STRUCTURAL_GAP_ZONE_HALF_DEFAULT_ABS",
            CONFIG.get("TRUE_GAP_ZONE_HALF_DEFAULT_ABS", 1500.0),
        )
    )
    zone_cap = max(zone_min_abs, min(zone_max_abs, zone_max_ratio * ref_len))
    gap_max_abs = float(CONFIG.get("STRUCTURAL_GAP_TRUE_GAP_HALF_MAX_ABS", 3000.0))
    gap_max_ratio = float(CONFIG.get("STRUCTURAL_GAP_TRUE_GAP_HALF_MAX_RATIO", 0.05))
    gap_min_abs = float(CONFIG.get("STRUCTURAL_GAP_TRUE_GAP_HALF_MIN_ABS", 800.0))
    gap_default = float(CONFIG.get("STRUCTURAL_GAP_TRUE_GAP_HALF_DEFAULT_ABS", 1500.0))
    gap_scale = float(CONFIG.get("STRUCTURAL_GAP_TRUE_GAP_HALF_SCALE_FROM_EDGEFIT", 0.95))
    gap_zone_ratio = float(CONFIG.get("STRUCTURAL_GAP_TRUE_GAP_MAX_ZONE_RATIO", 0.92))
    recover_band_abs = float(
        CONFIG.get(
            "STRUCTURAL_GAP_EDGE_RECOVER_BAND_ABS",
            CONFIG.get("TRUE_GAP_EDGE_RECOVER_BAND_ABS", 700.0),
        )
    )
    recover_band_ratio = float(
        CONFIG.get(
            "STRUCTURAL_GAP_EDGE_RECOVER_BAND_RATIO",
            CONFIG.get("TRUE_GAP_EDGE_RECOVER_BAND_RATIO", 0.22),
        )
    )
    anchor_win = float(
        CONFIG.get(
            "STRUCTURAL_GAP_EDGE_ANCHOR_WINDOW",
            CONFIG.get("TRUE_GAP_EDGE_ANCHOR_WINDOW", 2800.0),
        )
    )
    min_anchor_points = int(
        CONFIG.get(
            "STRUCTURAL_GAP_EDGE_ANCHOR_MIN_POINTS",
            CONFIG.get("TRUE_GAP_EDGE_ANCHOR_MIN_POINTS", 2),
        )
    )

    def _same_row_anchor_count(row_id: int, a: float, c: float) -> int:
        if c < a:
            a, c = (c, a)
        mm = (
            (out["row_main"].astype(int) == int(row_id))
            & (out["s_rot"].astype(float) >= float(a))
            & (out["s_rot"].astype(float) <= float(c))
        )
        if "is_boundary_gap_point" in out.columns:
            mm &= ~out["is_boundary_gap_point"].fillna(False).astype(bool)
        return int(mm.sum())

    gap_los: List[float] = []
    gap_his: List[float] = []
    boundary_gap_info: List[Dict[str, Any]] = []
    b_centers = b["center_s"].astype(float).to_numpy()
    for bi, center in enumerate(centers):
        if len(b_centers):
            ridx = int(np.nanargmin(np.abs(b_centers - center)))
            rr = b.iloc[ridx]
        else:
            rr = pd.Series({"center_s": center})
        raw_half = np.nan
        try:
            ue = float(rr.get("upper_edge_s", np.nan))
            le = float(rr.get("lower_edge_s", np.nan))
            if np.isfinite(ue) and np.isfinite(le):
                raw_half = 0.5 * abs(le - ue)
        except Exception:
            raw_half = np.nan
        if not np.isfinite(raw_half) or raw_half <= 0:
            raw_half = gap_default
        zone_half = float(min(max(raw_half, zone_min_abs), zone_cap))
        if not np.isfinite(zone_half) or zone_half <= 0:
            zone_half = zone_default
        visible_gap_half_candidate = (
            raw_half * gap_scale if np.isfinite(raw_half) and raw_half > 0 else gap_default
        )
        visible_gap_half = float(
            min(
                max(visible_gap_half_candidate, gap_min_abs),
                min(gap_max_abs, gap_max_ratio * ref_len, gap_zone_ratio * zone_half),
            )
        )
        if not np.isfinite(visible_gap_half) or visible_gap_half <= 0:
            visible_gap_half = min(gap_default, 0.85 * zone_half)
        visible_gap_half = min(visible_gap_half, max(1.0, gap_zone_ratio * zone_half))
        core_gap_max_abs = float(
            CONFIG.get(
                "GAP_NOISE_GAP_POINT_HALF_MAX_ABS",
                CONFIG.get("TRUE_GAP_TRUE_GAP_HALF_MAX_ABS", 1200.0),
            )
        )
        core_gap_max_ratio = float(
            CONFIG.get(
                "GAP_NOISE_GAP_POINT_HALF_MAX_RATIO",
                CONFIG.get("TRUE_GAP_TRUE_GAP_HALF_MAX_RATIO", 0.02),
            )
        )
        core_gap_min_abs = float(
            CONFIG.get(
                "GAP_NOISE_GAP_POINT_HALF_MIN_ABS",
                CONFIG.get("TRUE_GAP_TRUE_GAP_HALF_MIN_ABS", 250.0),
            )
        )
        core_gap_default = float(
            CONFIG.get(
                "GAP_NOISE_GAP_POINT_HALF_DEFAULT_ABS",
                CONFIG.get("TRUE_GAP_TRUE_GAP_HALF_DEFAULT_ABS", 500.0),
            )
        )
        core_gap_scale = float(
            CONFIG.get(
                "GAP_NOISE_GAP_POINT_HALF_SCALE_FROM_EDGEFIT",
                CONFIG.get("TRUE_GAP_TRUE_GAP_HALF_SCALE_FROM_EDGEFIT", 0.45),
            )
        )
        core_gap_zone_ratio = float(
            CONFIG.get(
                "GAP_NOISE_GAP_POINT_MAX_ZONE_RATIO",
                CONFIG.get("TRUE_GAP_TRUE_GAP_MAX_ZONE_RATIO", 0.55),
            )
        )
        core_gap_visible_ratio = float(
            CONFIG.get("GAP_NOISE_GAP_POINT_MAX_VISIBLE_RATIO", 0.72)
        )
        core_gap_half_candidate = (
            raw_half * core_gap_scale
            if np.isfinite(raw_half) and raw_half > 0
            else core_gap_default
        )
        core_gap_half = float(
            min(
                max(core_gap_half_candidate, core_gap_min_abs),
                min(
                    core_gap_max_abs,
                    core_gap_max_ratio * ref_len,
                    core_gap_zone_ratio * zone_half,
                    core_gap_visible_ratio * visible_gap_half,
                ),
            )
        )
        if not np.isfinite(core_gap_half) or core_gap_half <= 0:
            core_gap_half = min(core_gap_default, 0.55 * visible_gap_half, 0.55 * zone_half)
        core_gap_half = min(core_gap_half, max(1.0, visible_gap_half))
        zone_lo = max(smin, float(center) - zone_half)
        zone_hi = min(smax, float(center) + zone_half)
        vis_gap_lo = max(smin, float(center) - visible_gap_half)
        vis_gap_hi = min(smax, float(center) + visible_gap_half)
        if vis_gap_hi <= vis_gap_lo:
            vis_gap_lo = max(smin, float(center) - gap_min_abs)
            vis_gap_hi = min(smax, float(center) + gap_min_abs)
        gap_lo = max(smin, float(center) - core_gap_half)
        gap_hi = min(smax, float(center) + core_gap_half)
        if gap_hi <= gap_lo:
            gap_lo = max(smin, float(center) - core_gap_min_abs)
            gap_hi = min(smax, float(center) + core_gap_min_abs)
        if _safe_int(rr.get("narrow_gap_narrow_gap", 0), 0) == 1:
            vis_gap_lo = gap_lo = max(smin, float(rr["s0"]))
            vis_gap_hi = gap_hi = min(smax, float(rr["s1"]))
        gap_los.append(float(vis_gap_lo))
        gap_his.append(float(vis_gap_hi))
        boundary_gap_info.append(
            {
                "center_s": float(center),
                "density_recentering_center_original": (
                    float(centers_original[bi]) if bi < len(centers_original) else float(center)
                ),
                "density_recentering_center_shift": (
                    float(center - centers_original[bi]) if bi < len(centers_original) else 0.0
                ),
                "structural_gap_gap_lo": float(vis_gap_lo),
                "structural_gap_gap_hi": float(vis_gap_hi),
                "structural_gap_gap_half": float(0.5 * (vis_gap_hi - vis_gap_lo)),
                "gap_noise_core_gap_lo": float(gap_lo),
                "gap_noise_core_gap_hi": float(gap_hi),
                "gap_noise_core_gap_half": float(0.5 * (gap_hi - gap_lo)),
                "structural_gap_zone_lo": float(zone_lo),
                "structural_gap_zone_hi": float(zone_hi),
                "structural_gap_raw_edgefit_half": float(raw_half),
            }
        )
        recover_band_min = float(
            CONFIG.get("DENSITY_RECENTERING_EDGE_RECOVER_BAND_MIN_ABS", 180.0)
        )
        recover_band = max(
            recover_band_min,
            min(recover_band_abs, recover_band_ratio * max(gap_hi - gap_lo, 1.0)),
        )
        upper_pid = int(np.searchsorted(centers, center, side="left") + 1)
        lower_pid = min(upper_pid + 1, n_plots)
        mz = (out["s_rot"].astype(float) > zone_lo) & (out["s_rot"].astype(float) <= zone_hi)
        out.loc[mz, "is_boundary_zone_point"] = True
        out.loc[mz, "boundary_zone_center_s"] = center
        out.loc[mz & (out["s_rot"].astype(float) <= center), "boundary_zone_assigned_side"] = (
            "upper_plot"
        )
        out.loc[mz & (out["s_rot"].astype(float) > center), "boundary_zone_assigned_side"] = (
            "lower_plot"
        )
        out.loc[mz & (out["boundary_zone_class"] == "plot_point"), "boundary_zone_class"] = (
            "plot_edge_zone"
        )
        mg = (out["s_rot"].astype(float) > gap_lo) & (out["s_rot"].astype(float) <= gap_hi)
        if not mg.any():
            continue
        out.loc[mg, "is_boundary_gap_point"] = True
        out.loc[mg, "is_counted_plot_point"] = False
        out.loc[mg, "boundary_zone_class"] = "true_gap_point"
        out.loc[mg, "boundary_gap_reason"] = "inside_structural_gap_visible_interplot_gap"
        out.loc[mg, "plot_id"] = -1
        idxs = out.index[mg].tolist()
        for idx in idxs:
            s_val = float(out.at[idx, "s_rot"])
            row_id = (
                int(out.at[idx, "row_main"]) if not pd.isna(out.at[idx, "row_main"]) else -1
            )
            if row_id < 0:
                continue
            upper_dist = abs(s_val - gap_lo)
            lower_dist = abs(gap_hi - s_val)
            upper_anchor = (
                _same_row_anchor_count(row_id, gap_lo - anchor_win, gap_lo) >= min_anchor_points
            )
            lower_anchor = (
                _same_row_anchor_count(row_id, gap_hi, gap_hi + anchor_win) >= min_anchor_points
            )
            recover_side = ""
            if (
                upper_dist <= recover_band
                and upper_anchor
                and (upper_dist <= lower_dist or not lower_anchor)
            ):
                recover_side = "upper"
            elif lower_dist <= recover_band and lower_anchor:
                recover_side = "lower"
            elif upper_dist <= recover_band and upper_anchor:
                recover_side = "upper"
            if recover_side == "upper":
                out.at[idx, "plot_id"] = upper_pid
                out.at[idx, "is_boundary_gap_point"] = False
                out.at[idx, "is_counted_plot_point"] = True
                out.at[idx, "boundary_zone_class"] = "recovered_upper_edge_point"
                out.at[idx, "boundary_reassign_reason"] = (
                    "structural_gap_near_upper_gap_edge_and_row_anchor_exists"
                )
                out.at[idx, "boundary_gap_reason"] = ""
                out.at[idx, "boundary_zone_assigned_side"] = "upper_plot"
            elif recover_side == "lower":
                out.at[idx, "plot_id"] = lower_pid
                out.at[idx, "is_boundary_gap_point"] = False
                out.at[idx, "is_counted_plot_point"] = True
                out.at[idx, "boundary_zone_class"] = "recovered_lower_edge_point"
                out.at[idx, "boundary_reassign_reason"] = (
                    "structural_gap_near_lower_gap_edge_and_row_anchor_exists"
                )
                out.at[idx, "boundary_gap_reason"] = ""
                out.at[idx, "boundary_zone_assigned_side"] = "lower_plot"
    base_vis_starts = [smin] + gap_his
    base_vis_ends = gap_los + [smax]
    vis_starts = list(base_vis_starts)
    vis_ends = list(base_vis_ends)
    edge_margin = float(CONFIG.get("COUNTED_EXTENT_PLOT_BOX_EDGE_MARGIN_ABS", 180.0))
    min_visible_gap = float(CONFIG.get("COUNTED_EXTENT_MIN_VISIBLE_GAP_ABS", 350.0))
    counted_mask_all = (
        out.get("is_counted_plot_point", pd.Series(True, index=out.index))
        .fillna(True)
        .astype(bool)
    )
    use_counted_extent_box = bool(
        CONFIG.get("BOX_LAYOUT_USE_COUNTED_EXTENT_FOR_PLOT_BOX", False)
    )
    counted_s_mins = [np.nan for _ in range(n_plots)]
    counted_s_maxs = [np.nan for _ in range(n_plots)]
    for pid in range(1, n_plots + 1):
        pm_counted = (out["plot_id"].astype(int) == int(pid)) & counted_mask_all
        if not pm_counted.any():
            continue
        pmin = float(out.loc[pm_counted, "s_rot"].astype(float).min())
        pmax = float(out.loc[pm_counted, "s_rot"].astype(float).max())
        counted_s_mins[pid - 1] = pmin
        counted_s_maxs[pid - 1] = pmax
        if use_counted_extent_box:
            if np.isfinite(pmin):
                vis_starts[pid - 1] = min(float(vis_starts[pid - 1]), pmin - edge_margin)
            if np.isfinite(pmax):
                vis_ends[pid - 1] = max(float(vis_ends[pid - 1]), pmax + edge_margin)
    for pid in range(1, n_plots + 1):
        fs0 = float(full_starts[pid - 1])
        fs1 = float(full_ends[pid - 1])
        vis_starts[pid - 1] = max(fs0, min(float(vis_starts[pid - 1]), fs1))
        vis_ends[pid - 1] = min(fs1, max(float(vis_ends[pid - 1]), fs0))
    for bi, center in enumerate(centers):
        upper_i = bi
        lower_i = bi + 1
        if upper_i >= len(vis_ends) or lower_i >= len(vis_starts):
            continue
        ue = float(vis_ends[upper_i])
        ls = float(vis_starts[lower_i])
        if not (np.isfinite(ue) and np.isfinite(ls)):
            continue
        if ue + min_visible_gap > ls:
            mid = 0.5 * (ue + ls)
            half = 0.5 * min_visible_gap
            vis_ends[upper_i] = min(ue, mid - half)
            vis_starts[lower_i] = max(ls, mid + half)
            if vis_ends[upper_i] + min_visible_gap > vis_starts[lower_i]:
                vis_ends[upper_i] = float(center) - 0.5 * min_visible_gap
                vis_starts[lower_i] = float(center) + 0.5 * min_visible_gap
    rows2 = []
    for pid in range(1, n_plots + 1):
        fs0 = float(full_starts[pid - 1])
        fs1 = float(full_ends[pid - 1])
        vs0 = float(vis_starts[pid - 1])
        vs1 = float(vis_ends[pid - 1])
        if not np.isfinite(vs0) or not np.isfinite(vs1) or vs1 <= vs0:
            vs0, vs1 = (fs0, fs1)
        pm = out["plot_id"].astype(int) == pid
        rr = {
            "plot_id": int(pid),
            "s_start": float(vs0),
            "s_end": float(vs1),
            "s_start_fullcover": float(fs0),
            "s_end_fullcover": float(fs1),
            "s_start_structural_gap_gap_edge": float(base_vis_starts[pid - 1]),
            "s_end_structural_gap_gap_edge": float(base_vis_ends[pid - 1]),
            "s_start_counted_extent": (
                float(counted_s_mins[pid - 1])
                if np.isfinite(counted_s_mins[pid - 1])
                else np.nan
            ),
            "s_end_counted_extent": (
                float(counted_s_maxs[pid - 1])
                if np.isfinite(counted_s_maxs[pid - 1])
                else np.nan
            ),
            "box_layout_plot_box_extent_mode": (
                "counted_extent" if use_counted_extent_box else "structural_gap_edge"
            ),
            "count": int(pm.sum()),
            "length": float(vs1 - vs0),
            "length_fullcover": float(fs1 - fs0),
            "point_assignment_assignment_mode": "center_separator_full_cover_then_box_layout_structural_gap_box",
            "true_gap_gap_edge_reassign": True,
            "structural_gap_visible_gap_assignment": True,
            "counted_extent_counted_extent_box_correction": bool(use_counted_extent_box),
            "boundary_zone_assigned_count": int(
                (
                    pm
                    & out.get("is_boundary_zone_point", pd.Series(False, index=out.index))
                    .fillna(False)
                    .astype(bool)
                ).sum()
            ),
            "boundary_gap_noise_count_excluded": int(
                (
                    (
                        out.get(
                            "assigned_plot_id_fullcover", pd.Series(-1, index=out.index)
                        ).astype(int)
                        == pid
                    )
                    & out["is_boundary_gap_point"].fillna(False).astype(bool)
                ).sum()
            ),
            "recovered_boundary_edge_count": int(
                (pm & out["boundary_zone_class"].astype(str).str.startswith("recovered_")).sum()
            ),
        }
        if pid > 1 and pid - 2 < len(boundary_gap_info):
            rr["prev_gap_hi"] = float(boundary_gap_info[pid - 2]["structural_gap_gap_hi"])
            rr["prev_gap_center_s"] = float(boundary_gap_info[pid - 2]["center_s"])
        if pid <= len(boundary_gap_info):
            rr["next_gap_lo"] = float(boundary_gap_info[pid - 1]["structural_gap_gap_lo"])
            rr["next_gap_center_s"] = float(boundary_gap_info[pid - 1]["center_s"])
        rows2.append(rr)
    plot_df = pd.DataFrame(rows2).sort_values("plot_id").reset_index(drop=True)
    return (plot_df, out.reset_index(drop=True))


def build_evidence_curve(cands_all: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if len(cands_all) == 0:
        return pd.DataFrame(
            columns=["center_s", "evidence_score", "max_score", "mean_score", "n_candidates"]
        )
    df = cands_all.copy()
    step = float(cfg["SCAN_STEP"])
    df["center_bin"] = np.round(df["center_s"].astype(float) / step).astype(int)
    rows = []
    for b, g in df.groupby("center_bin"):
        rows.append(
            {
                "center_s": float(b * step),
                "evidence_score": float(np.max(g["score"].astype(float))),
                "max_score": float(np.max(g["score"].astype(float))),
                "mean_score": float(np.mean(g["score"].astype(float))),
                "n_candidates": int(len(g)),
                "max_n_strong": int(np.max(g["n_strong_break"].astype(float))),
                "max_n_weak": int(np.max(g["n_weak_break"].astype(float))),
                "min_joint_drop": float(np.min(g["joint_drop"].astype(float))),
            }
        )
    return pd.DataFrame(rows).sort_values("center_s").reset_index(drop=True)


def draw_evidence_curve_png(
    evidence: pd.DataFrame, selected: pd.DataFrame, out_path: Path, cfg: Dict
):
    W, H = (1600, 420)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)
    if len(evidence) == 0:
        _write_image_unicode(str(out_path), img)
        return
    x = evidence["center_s"].to_numpy(dtype=float)
    y = evidence["evidence_score"].to_numpy(dtype=float)
    xmin, xmax = (float(x.min()), float(x.max()))
    ymin, ymax = (
        min(0.0, float(np.nanmin(y))),
        max(float(np.nanmax(y)), float(cfg["CAND_MIN_SCORE"]) + 1.0),
    )
    margin_l, margin_r, margin_t, margin_b = (80, 30, 30, 55)

    def map_xy(xx, yy):
        px = int(
            round(margin_l + (xx - xmin) / max(xmax - xmin, 1e-06) * (W - margin_l - margin_r))
        )
        py = int(
            round(
                H - margin_b - (yy - ymin) / max(ymax - ymin, 1e-06) * (H - margin_t - margin_b)
            )
        )
        return (px, py)

    cv2.line(img, (margin_l, H - margin_b), (W - margin_r, H - margin_b), (180, 180, 180), 1)
    cv2.line(img, (margin_l, margin_t), (margin_l, H - margin_b), (180, 180, 180), 1)
    pts = np.array([map_xy(xx, yy) for xx, yy in zip(x, y)], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
    _, th_y = map_xy(xmin, float(cfg["CAND_MIN_SCORE"]))
    cv2.line(img, (margin_l, th_y), (W - margin_r, th_y), (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "candidate threshold",
        (margin_l + 10, max(20, th_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )
    if selected is not None and len(selected):
        for _, rr in selected.iterrows():
            sx = float(rr["center_s"])
            px, _ = map_xy(sx, ymin)
            cv2.line(img, (px, margin_t), (px, H - margin_b), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "1D boundary evidence curve + DP selected boundaries",
        (margin_l, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    _write_image_unicode(str(out_path), img)


def map_point(t: float, s: float, tfm: Dict):
    px = int(
        round(
            (float(t) - tfm["x_min"] + tfm.get("x_offset", CONFIG["POINT_MARGIN"]))
            * tfm["scale"]
        )
    )
    py = int(
        round(
            (float(s) - tfm["y_min"] + tfm.get("y_offset", CONFIG["POINT_MARGIN"]))
            * tfm["scale"]
        )
    )
    return (px, py)


def draw_points_and_plots(
    plot_points: pd.DataFrame, plot_df: pd.DataFrame, boundaries: pd.DataFrame, cfg: Dict
) -> np.ndarray:
    base = plot_points.copy()
    if len(base) == 0:
        return np.zeros((1000, 800, 3), dtype=np.uint8)
    tfm = compute_canvas_transform(base, cfg)
    canvas = np.zeros((tfm["H"], tfm["W"], 3), dtype=np.uint8)
    radius = int(cfg["POINT_RADIUS"])
    for _, row in base.iterrows():
        px, py = map_point(row["t_rot"], row["s_rot"], tfm)
        if bool(row.get("is_boundary_gap_point", False)) and (
            bool(cfg.get("BOUNDARY_VISUAL_LEGACY_ORANGE_GAP_POINTS", False))
            or bool(cfg.get("TRUE_GAP_VIS_SHOW_TRUE_GAP_POINTS", True))
        ):
            color = cfg["COLOR_GAP_POINT"]
        else:
            color = cfg["COLOR_KEEP"]
        cv2.circle(canvas, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
    if int(cfg.get("POINT_DILATE", 0)) > 0:
        kernel = np.ones((3, 3), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=int(cfg.get("POINT_DILATE", 0)))
    if boundaries is not None and len(boundaries):
        t_lo = float(np.quantile(base["t_rot"], 0.005)) - 30
        t_hi = float(np.quantile(base["t_rot"], 0.995)) + 30
        draw_records = []
        if (
            bool(cfg.get("DENSITY_RECENTERING_DRAW_BOUNDARIES_FROM_PLOT_DF", True))
            and plot_df is not None
            and (len(plot_df) > 1)
            and ("next_gap_center_s" in plot_df.columns)
        ):
            pdf = plot_df.sort_values("plot_id").reset_index(drop=True)
            for i in range(len(pdf) - 1):
                c = (
                    pd.to_numeric(
                        pd.Series([pdf.loc[i, "next_gap_center_s"]]), errors="coerce"
                    ).iloc[0]
                    if "next_gap_center_s" in pdf.columns
                    else np.nan
                )
                lo = (
                    pd.to_numeric(pd.Series([pdf.loc[i, "next_gap_lo"]]), errors="coerce").iloc[
                        0
                    ]
                    if "next_gap_lo" in pdf.columns
                    else np.nan
                )
                hi = pd.to_numeric(
                    pd.Series([pdf.loc[i + 1, "s_start"]]), errors="coerce"
                ).iloc[0]
                if np.isfinite(c) and np.isfinite(lo) and np.isfinite(hi):
                    draw_records.append((float(c), float(lo), float(hi)))
        if not draw_records:
            for _, rr in boundaries.iterrows():
                c = float(rr.get("center_s", np.nan))
                lo = float(rr.get("upper_edge_s", c))
                hi = float(rr.get("lower_edge_s", c))
                if np.isfinite(c):
                    draw_records.append((c, lo, hi))
        for c, lo, hi in draw_records:
            for ss, col in [(float(lo), (200, 200, 200)), (float(hi), (200, 200, 200))]:
                x1, y1 = map_point(t_lo, ss, tfm)
                x2, y2 = map_point(t_hi, ss, tfm)
                cv2.line(canvas, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
            x1, y1 = map_point(t_lo, float(c), tfm)
            x2, y2 = map_point(t_hi, float(c), tfm)
            cv2.line(canvas, (x1, y1), (x2, y2), cfg["COLOR_BOUNDARY"], 2, cv2.LINE_AA)
    if plot_df is not None and len(plot_df):
        extra = float(cfg["PLOT_BOX_EXTRA_MARGIN_PX"])
        side_ratio = float(cfg.get("PLOT_BOX_SIDE_EXTRA_RATIO", 0.1))
        min_side_extra = float(cfg.get("PLOT_BOX_MIN_SIDE_EXTRA", 20.0))
        for _, row in plot_df.iterrows():
            pid = int(row["plot_id"])
            gg = base[base["plot_id"].astype(int) == pid]
            if len(gg) == 0:
                continue
            ql = float(cfg["PLOT_BOX_QUANTILE_LOW"])
            qh = float(cfg["PLOT_BOX_QUANTILE_HIGH"])
            t_q_lo = float(np.quantile(gg["t_rot"], ql))
            t_q_hi = float(np.quantile(gg["t_rot"], qh))
            t_raw_lo = float(gg["t_rot"].min())
            t_raw_hi = float(gg["t_rot"].max())
            t_span = max(t_raw_hi - t_raw_lo, 1.0)
            side_extra = max(min_side_extra, side_ratio * t_span)
            t_lo = min(t_q_lo, t_raw_lo) - extra - side_extra
            t_hi = max(t_q_hi, t_raw_hi) + extra + side_extra
            s0 = float(row["s_start"])
            s1 = float(row["s_end"])
            x1, y1 = map_point(t_lo, s0, tfm)
            x2, y2 = map_point(t_hi, s1, tfm)
            xa, xb = sorted([x1, x2])
            ya, yb = sorted([y1, y2])
            color = cfg["PLOT_RECT_COLORS"][(pid - 1) % len(cfg["PLOT_RECT_COLORS"])]
            cv2.rectangle(canvas, (xa, ya), (xb, yb), color, 2, lineType=cv2.LINE_AA)
            tx = min(canvas.shape[1] - 80, xb + 8)
            ty = int(0.5 * (ya + yb))
            cv2.putText(
                canvas,
                f"P{pid}",
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
    return canvas


def _sequence_guard_sorted_centers(selected: pd.DataFrame) -> np.ndarray:
    if selected is None or len(selected) == 0 or "center_s" not in selected.columns:
        return np.zeros((0,), dtype=float)
    vals = pd.to_numeric(selected["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    return np.sort(vals)


def _sequence_guard_segments_from_selected(
    work: pd.DataFrame, selected: pd.DataFrame
) -> List[Tuple[float, float]]:
    centers = _sequence_guard_sorted_centers(selected)
    if len(work) == 0:
        return []
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    cuts = [smin] + centers.tolist() + [smax]
    return [(float(cuts[i]), float(cuts[i + 1])) for i in range(len(cuts) - 1)]


def _sequence_guard_expected_len_from_segments(
    segments: List[Tuple[float, float]], cfg: Dict
) -> float:
    lens = np.array([max(b - a, 1.0) for a, b in segments], dtype=float)
    lens = lens[np.isfinite(lens) & (lens > 1.0)]
    if len(lens) == 0:
        return np.nan
    qlo = np.quantile(lens, 0.2) if len(lens) >= 5 else np.min(lens)
    qhi = np.quantile(lens, 0.8) if len(lens) >= 5 else np.max(lens)
    core = lens[(lens >= qlo) & (lens <= qhi)]
    if len(core) == 0:
        core = lens
    return float(np.median(core))


def _sequence_guard_count_points_rows(
    work: pd.DataFrame, a: float, b: float, cfg: Dict
) -> Tuple[int, int]:
    if a > b:
        a, b = (b, a)
    gg = work[
        (work["s_rot"].astype(float) >= float(a)) & (work["s_rot"].astype(float) <= float(b))
    ]
    if len(gg) == 0:
        return (0, 0)
    row_min = int(cfg.get("SEGMENT_ROW_MIN_POINTS", 5))
    rows = 0
    for r in range(int(cfg.get("MAIN_ROWS", 3))):
        if int((gg["row_main"].astype(int) == r).sum()) >= row_min:
            rows += 1
    return (int(len(gg)), int(rows))


def _sequence_guard_is_ultra_protected(row: pd.Series, cfg: Dict) -> bool:
    cls = str(row.get("candidate_class", ""))
    try:
        jd = float(row.get("joint_drop", np.inf))
        cont = int(row.get("continuous_rows", 99))
        score = float(row.get("score", 0.0))
        strong = int(row.get("n_strong_break", 0))
        weak = int(row.get("n_weak_break", 0))
    except Exception:
        return False
    if (
        cls == "A_ultra_empty"
        and jd <= float(cfg.get("ONE_SIDE_PROTECT_ULTRA_JOINT_DROP", 0.09))
        and (cont <= int(cfg.get("ONE_SIDE_PROTECT_CONT_ROWS_MAX", 0)))
    ):
        return True
    if (
        score >= float(cfg.get("ONE_SIDE_PROTECT_SCORE", 14.6))
        and jd <= 0.12
        and (cont == 0)
        and (strong >= 3)
        and (weak >= 3)
    ):
        return True
    return False


def _sequence_guard_boundary_cleanliness(row: pd.Series, cfg: Dict) -> float:
    """Score empty-gap evidence with a penalty for row-gap-only support."""
    score = float(row.get("score", 0.0))
    jd = float(row.get("joint_drop", 1.0))
    cont = float(row.get("continuous_rows", 3.0))
    strong = float(row.get("n_strong_break", 0.0))
    weak = float(row.get("n_weak_break", 0.0))
    both = (
        1.0
        if int(row.get("left_recover", 0)) == 1 and int(row.get("right_recover", 0)) == 1
        else 0.0
    )
    cls = str(row.get("candidate_class", ""))
    kind = str(row.get("kind", ""))
    val = (
        0.35 * score
        + 4.0 * (1.0 - min(jd, 1.0))
        + 0.8 * strong
        + 0.25 * weak
        + 0.9 * both
        - 1.8 * cont
    )
    if cls == "A_ultra_empty":
        val += 5.0
    if kind == "row_gap_proposal" or int(row.get("is_row_gap_candidate", 0)) == 1:
        val -= 3.0
    return float(val)


def _sequence_guard_pick_best_internal_candidate(
    cands: pd.DataFrame, left: float, right: float, selected_centers: np.ndarray, cfg: Dict
) -> Optional[pd.Series]:
    if cands is None or len(cands) == 0:
        return None
    margin = max(1200.0, 0.06 * max(right - left, 1.0))
    cand = cands[
        (pd.to_numeric(cands["center_s"], errors="coerce") > left + margin)
        & (pd.to_numeric(cands["center_s"], errors="coerce") < right - margin)
    ].copy()
    if len(cand) == 0:
        return None
    cand = cand[
        pd.to_numeric(cand["score"], errors="coerce")
        >= float(cfg.get("CLUSTER_COMPRESS_MIN_CAND_SCORE", 10.8))
    ].copy()
    if len(cand) == 0:
        return None
    if len(selected_centers):
        dist = np.min(
            np.abs(cand["center_s"].to_numpy(dtype=float)[:, None] - selected_centers[None, :]),
            axis=1,
        )
        cand = cand[dist >= 1000.0].copy()
    if len(cand) == 0:
        return None
    cand["_clean"] = cand.apply(lambda r: _sequence_guard_boundary_cleanliness(r, cfg), axis=1)
    cand["_dist_mid_pen"] = np.abs(cand["center_s"].astype(float) - 0.5 * (left + right)) / max(
        right - left, 1.0
    )
    cand["_rank"] = cand["_clean"] - 1.2 * cand["_dist_mid_pen"]
    if (
        bool(cfg.get("CLUSTER_COMPRESS_PREFER_NON_ROWGAP", True))
        and "is_row_gap_candidate" in cand.columns
    ):
        non_rg = cand[cand["is_row_gap_candidate"].astype(int) == 0].copy()
        if len(non_rg):
            cand = non_rg
    cand = cand.sort_values(["_rank", "score"], ascending=False)
    return cand.iloc[0]


def _sequence_guard_local_sequence_objective(
    work: pd.DataFrame,
    centers: List[float],
    cands: pd.DataFrame,
    cfg: Dict,
    a: float,
    b: float,
    expected_len: float,
) -> float:
    centers = sorted([float(x) for x in centers if a < float(x) < b])
    cuts = [float(a)] + centers + [float(b)]
    obj = 0.0
    for i in range(len(cuts) - 1):
        s0, s1 = (cuts[i], cuts[i + 1])
        L = max(s1 - s0, 1.0)
        r = L / max(expected_len, 1.0)
        if r < 0.55:
            obj += 8.0 * (0.55 - r) / 0.55
        if r < 0.38:
            obj += 18.0 * (0.38 - r) / 0.38
        if r > 2.25:
            obj += 2.5 * (r - 2.25)
        npts, nrows = _sequence_guard_count_points_rows(work, s0, s1, cfg)
        if nrows < int(cfg.get("SEGMENT_MIN_ROWS_PRESENT", 2)):
            obj += 7.5
        if npts < int(cfg.get("SEGMENT_MIN_POINTS_ABS", 30)):
            obj += 6.0
        if cands is not None and len(cands):
            margin = max(2500.0, 0.08 * L)
            inside = cands[
                (cands["center_s"].astype(float) > s0 + margin)
                & (cands["center_s"].astype(float) < s1 - margin)
            ].copy()
            if len(inside):
                inside["_clean"] = inside.apply(
                    lambda rr: _sequence_guard_boundary_cleanliness(rr, cfg), axis=1
                )
                if "is_row_gap_candidate" in inside.columns:
                    non_rowgap = inside["is_row_gap_candidate"].fillna(0).astype(int) == 0
                else:
                    non_rowgap = pd.Series(True, index=inside.index)
                clean_empty = (inside["candidate_class"].astype(str) == "A_ultra_empty") | (
                    inside["continuous_rows"].fillna(99).astype(int) <= 0
                ) & non_rowgap & (inside["_clean"] >= 10.5)
                strong_inside = inside[clean_empty].copy()
                if len(strong_inside):
                    obj += min(
                        18.0,
                        4.0 * len(strong_inside) + 0.4 * float(strong_inside["_clean"].max()),
                    )
    for c in centers:
        near = (
            cands.iloc[np.abs(cands["center_s"].astype(float) - c).argsort()[:1]]
            if cands is not None and len(cands)
            else pd.DataFrame()
        )
        if len(near):
            obj -= 0.65 * _sequence_guard_boundary_cleanliness(near.iloc[0], cfg)
        obj += 1.5
    return float(obj)


def repair_boundary_clusters_by_local_quality(
    work: pd.DataFrame, cands: pd.DataFrame, selected: pd.DataFrame, cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve nearby cuts using local segment quality and candidate evidence."""
    if selected is None or len(selected) == 0:
        return (selected, pd.DataFrame())
    if not bool(cfg.get("ENABLE_FINAL_CLUSTER_COMPRESSION_RECHECK", True)):
        return (selected, pd.DataFrame())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    logs = []
    for it in range(int(cfg.get("CLUSTER_COMPRESS_MAX_ITER", 10))):
        centers = _sequence_guard_sorted_centers(current)
        segments = _sequence_guard_segments_from_selected(work, current)
        expected = _sequence_guard_expected_len_from_segments(segments, cfg)
        if not np.isfinite(expected) or expected <= 1:
            break
        changed = False
        best_action = None
        for i in range(len(centers) - 1):
            c1, c2 = (float(centers[i]), float(centers[i + 1]))
            mid_len = c2 - c1
            if not mid_len <= max(
                float(cfg.get("CLUSTER_COMPRESS_MIDDLE_MAX_ABS", 32000.0)),
                float(cfg.get("CLUSTER_COMPRESS_MIDDLE_MAX_RATIO", 0.68)) * expected,
            ):
                continue
            prev_s = float(work["s_rot"].min()) if i == 0 else float(centers[i - 1])
            next_s = (
                float(work["s_rot"].max()) if i + 2 >= len(centers) else float(centers[i + 2])
            )
            row1 = current.iloc[i]
            row2 = current.iloc[i + 1]
            if _sequence_guard_is_ultra_protected(
                row1, cfg
            ) and _sequence_guard_is_ultra_protected(row2, cfg):
                continue
            cand = _sequence_guard_pick_best_internal_candidate(cands, c1, c2, centers, cfg)
            if cand is None:
                continue
            k = float(cand["center_s"])
            left_new = k - prev_s
            right_new = next_s - k
            if left_new < float(
                cfg.get("CLUSTER_COMPRESS_NEW_CHILD_MIN_ABS", 18000.0)
            ) or right_new < float(cfg.get("CLUSTER_COMPRESS_NEW_CHILD_MIN_ABS", 18000.0)):
                continue
            if (
                left_new
                < float(cfg.get("CLUSTER_COMPRESS_NEW_CHILD_MIN_RATIO", 0.34)) * expected
                or right_new
                < float(cfg.get("CLUSTER_COMPRESS_NEW_CHILD_MIN_RATIO", 0.34)) * expected
            ):
                continue
            old_obj = _sequence_guard_local_sequence_objective(
                work, [c1, c2], cands, cfg, prev_s, next_s, expected
            )
            new_obj = _sequence_guard_local_sequence_objective(
                work, [k], cands, cfg, prev_s, next_s, expected
            )
            improve = old_obj - new_obj
            if improve >= float(cfg.get("CLUSTER_COMPRESS_MIN_IMPROVE", 0.35)):
                if best_action is None or improve > best_action[0]:
                    best_action = (
                        improve,
                        i,
                        cand,
                        old_obj,
                        new_obj,
                        expected,
                        prev_s,
                        next_s,
                        mid_len,
                    )
        if best_action is None:
            break
        improve, i, cand, old_obj, new_obj, expected, prev_s, next_s, mid_len = best_action
        c1 = float(current.iloc[i]["center_s"])
        c2 = float(current.iloc[i + 1]["center_s"])
        k = float(cand["center_s"])
        new_row = cand.drop(
            labels=[x for x in cand.index if str(x).startswith("_")], errors="ignore"
        ).to_dict()
        new_row["kind"] = (
            str(new_row.get("kind", "dense_scan")) + "|sequence_guard_cluster_replace"
        )
        cur2 = current.drop(current.index[[i, i + 1]]).reset_index(drop=True)
        cur2 = (
            pd.concat([cur2, pd.DataFrame([new_row])], ignore_index=True, sort=False)
            .sort_values("center_s")
            .reset_index(drop=True)
        )
        logs.append(
            {
                "stage": "cluster_compress_pair_to_one",
                "iter": it,
                "drop_left_center_s": c1,
                "drop_right_center_s": c2,
                "insert_center_s": k,
                "insert_score": float(cand.get("score", np.nan)),
                "insert_class": str(cand.get("candidate_class", "")),
                "middle_len": float(mid_len),
                "expected_len": float(expected),
                "old_obj": float(old_obj),
                "new_obj": float(new_obj),
                "improve": float(improve),
                "decision": "replace_two_boundaries_with_one_candidate",
            }
        )
        current = cur2
        changed = True
        if not changed:
            break
    if bool(cfg.get("ENABLE_ONE_SIDED_SHORT_RECHECK", True)):
        for it in range(int(cfg.get("ONE_SIDE_MAX_ITER", 12))):
            centers = _sequence_guard_sorted_centers(current)
            if len(centers) <= 1:
                break
            segments = _sequence_guard_segments_from_selected(work, current)
            expected = _sequence_guard_expected_len_from_segments(segments, cfg)
            if not np.isfinite(expected) or expected <= 1:
                break
            best_drop = None
            for i, c in enumerate(centers):
                if i == 0 or i == len(centers) - 1:
                    continue
                prev_s = centers[i - 1]
                next_s = centers[i + 1]
                left_len = c - prev_s
                right_len = next_s - c
                merged_len = next_s - prev_s
                row = current.iloc[i]
                if _sequence_guard_is_ultra_protected(row, cfg):
                    continue
                short_left = left_len <= max(
                    float(cfg.get("ONE_SIDE_SHORT_MAX_ABS", 28500.0)),
                    float(cfg.get("ONE_SIDE_SHORT_MAX_RATIO", 0.56)) * expected,
                )
                short_right = right_len <= max(
                    float(cfg.get("ONE_SIDE_SHORT_MAX_ABS", 28500.0)),
                    float(cfg.get("ONE_SIDE_SHORT_MAX_RATIO", 0.56)) * expected,
                )
                if not (short_left or short_right):
                    continue
                mr = merged_len / max(expected, 1.0)
                if mr < float(cfg.get("ONE_SIDE_MERGED_MIN_RATIO", 0.82)) or mr > float(
                    cfg.get("ONE_SIDE_MERGED_MAX_RATIO", 1.85)
                ):
                    continue
                npts, nrows = _sequence_guard_count_points_rows(work, prev_s, next_s, cfg)
                if npts < int(cfg.get("ONE_SIDE_MIN_MERGED_POINTS", 160)) or nrows < int(
                    cfg.get("ONE_SIDE_MIN_MERGED_ROWS", 2)
                ):
                    continue
                old_obj = _sequence_guard_local_sequence_objective(
                    work, [c], cands, cfg, prev_s, next_s, expected
                )
                new_obj = _sequence_guard_local_sequence_objective(
                    work, [], cands, cfg, prev_s, next_s, expected
                )
                improve = old_obj - new_obj
                cont = int(row.get("continuous_rows", 99))
                jd = float(row.get("joint_drop", 1.0))
                boundary_not_clear = (
                    cont >= 1
                    or jd > 0.16
                    or str(row.get("candidate_class", "")) not in ["A_ultra_empty"]
                )
                strong_short = min(left_len, right_len) <= max(
                    float(cfg.get("ONE_SIDE_SHORT_MAX_ABS", 28500.0)),
                    float(cfg.get("ONE_SIDE_SHORT_MAX_RATIO", 0.65)) * expected,
                )
                min_ratio_lr = min(left_len, right_len) / max(expected, 1.0)
                very_short_child = min(left_len, right_len) <= float(
                    cfg.get("ONE_SIDE_SHORT_MAX_ABS", 28500.0)
                )
                ratio_short_child = min_ratio_lr <= min(
                    0.58, float(cfg.get("ONE_SIDE_SHORT_MAX_RATIO", 0.65))
                )
                accept_drop = (
                    boundary_not_clear
                    and strong_short
                    and (improve >= -6.0)
                    and (very_short_child or ratio_short_child)
                )
                if accept_drop:
                    severity = max(
                        0.0, float(cfg.get("ONE_SIDE_SHORT_MAX_RATIO", 0.65)) - min_ratio_lr
                    ) + max(improve, 0.0)
                    if best_drop is None or severity > best_drop[0]:
                        best_drop = (
                            severity,
                            i,
                            old_obj,
                            new_obj,
                            expected,
                            left_len,
                            right_len,
                            merged_len,
                            npts,
                            nrows,
                            improve,
                        )
            if best_drop is None:
                break
            (
                severity,
                i,
                old_obj,
                new_obj,
                expected,
                left_len,
                right_len,
                merged_len,
                npts,
                nrows,
                improve,
            ) = best_drop
            c = float(current.iloc[i]["center_s"])
            logs.append(
                {
                    "stage": "one_sided_short_drop",
                    "iter": it,
                    "drop_center_s": c,
                    "drop_score": float(current.iloc[i].get("score", np.nan)),
                    "drop_class": str(current.iloc[i].get("candidate_class", "")),
                    "left_len": float(left_len),
                    "right_len": float(right_len),
                    "merged_len": float(merged_len),
                    "expected_len": float(expected),
                    "merged_points": int(npts),
                    "merged_rows": int(nrows),
                    "old_obj": float(old_obj),
                    "new_obj": float(new_obj),
                    "improve": float(improve),
                    "severity": float(severity),
                    "decision": "drop_one_sided_short_false_boundary",
                }
            )
            current = (
                current.drop(current.index[i])
                .reset_index(drop=True)
                .sort_values("center_s")
                .reset_index(drop=True)
            )
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def _anchor_recall_get_typical_len(
    mode_info: Dict[str, Any], work: pd.DataFrame, selected: pd.DataFrame, cfg: Dict
) -> float:
    vals = []
    try:
        v = float(mode_info.get("typical_len", np.nan))
        if np.isfinite(v) and v > 1000:
            vals.append(v)
    except Exception:
        pass
    try:
        v = float(mode_info.get("acf_peak_len", np.nan))
        if np.isfinite(v) and v > 1000:
            vals.append(v)
    except Exception:
        pass
    if selected is not None and len(selected) >= 2:
        cs = np.sort(
            pd.to_numeric(selected["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
        )
        ds = np.diff(cs)
        ds = ds[np.isfinite(ds) & (ds > 1000)]
        if len(ds):
            vals.append(float(np.median(ds)))
    if vals:
        return float(np.median(vals))
    smin, smax = (float(work["s_rot"].min()), float(work["s_rot"].max()))
    return max((smax - smin) / 4.0, 1000.0)


def _anchor_recall_is_rowgap(row: pd.Series) -> bool:
    return (
        str(row.get("kind", "")).lower() == "row_gap_proposal"
        or _safe_int(row.get("is_row_gap_candidate", 0), 0) == 1
    )


def _anchor_recall_is_ultra_rowgap(row: pd.Series, cfg: Dict) -> bool:
    """Require an empty, synchronous multirow break without continuous plants."""
    if not _anchor_recall_is_rowgap(row):
        return False
    return bool(
        _safe_float(row.get("score", 0.0), 0.0)
        >= float(cfg.get("ANCHOR_RECALL_ROWGAP_ULTRA_SCORE_MIN", 12.0))
        and _safe_int(row.get("n_strong_break", 0), 0)
        >= int(cfg.get("ANCHOR_RECALL_ROWGAP_ULTRA_MIN_STRONG", 3))
        and (
            _safe_int(row.get("n_weak_break", 0), 0)
            >= int(cfg.get("ANCHOR_RECALL_ROWGAP_ULTRA_MIN_WEAK", 3))
        )
        and (
            _safe_float(row.get("joint_drop", 9.0), 9.0)
            <= float(cfg.get("ANCHOR_RECALL_ROWGAP_ULTRA_MAX_JOINT_DROP", 0.12))
        )
        and (_safe_int(row.get("continuous_rows", 99), 99) == 0)
        and (
            _safe_int(row.get("row_gap_support_row_count", 0), 0)
            >= int(cfg.get("ANCHOR_RECALL_ROWGAP_ULTRA_MIN_SUPPORT_ROWS", 2))
        )
    )


def _anchor_recall_is_recall_anchor(row: pd.Series, cfg: Dict, terminal: bool = False) -> bool:
    """Identify recall anchors; subsequent false-split checks still apply."""
    cls = str(row.get("candidate_class", ""))
    score = _safe_float(row.get("score", 0.0), 0.0)
    jd = _safe_float(row.get("joint_drop", 9.0), 9.0)
    cont = _safe_int(row.get("continuous_rows", 99), 99)
    ns = _safe_int(row.get("n_strong_break", 0), 0)
    nw = _safe_int(row.get("n_weak_break", 0), 0)
    lr = _safe_int(row.get("left_recover", 0), 0)
    rr = _safe_int(row.get("right_recover", 0), 0)
    if _anchor_recall_is_ultra_rowgap(row, cfg):
        return True
    if not _anchor_recall_is_rowgap(row):
        if (
            cls.startswith("A_")
            and score >= float(cfg.get("ANCHOR_RECALL_DENSE_A_SCORE_MIN", 12.0))
            and (cont <= 1)
            and (ns >= 2)
            and (nw >= 3)
        ):
            return True
        if (
            cls.startswith("B_")
            and score >= float(cfg.get("ANCHOR_RECALL_DENSE_B_SCORE_MIN", 11.5))
            and (cont == 0)
            and (ns >= 2)
            and (nw >= 3)
        ):
            return True
        if (
            terminal
            and score >= float(cfg.get("ANCHOR_RECALL_TERMINAL_DENSE_SCORE_MIN", 10.8))
            and (cont == 0)
            and (ns >= 2)
            and (nw >= 3)
            and (jd <= float(cfg.get("ANCHOR_RECALL_TERMINAL_MAX_JOINT_DROP", 0.38)))
        ):
            return True
    if terminal and _anchor_recall_is_rowgap(row):
        if (
            score >= float(cfg.get("ANCHOR_RECALL_TERMINAL_ROWGAP_SCORE_MIN", 12.0))
            and cont == 0
            and (ns >= 2)
            and (nw >= 3)
            and (jd <= float(cfg.get("ANCHOR_RECALL_TERMINAL_ROWGAP_MAX_JOINT_DROP", 0.25)))
        ):
            return True
    return False


def _anchor_recall_recall_rank(
    row: pd.Series, a: float, b: float, typical_len: float, cfg: Dict
) -> float:
    c = _safe_float(row.get("center_s", np.nan), np.nan)
    if not np.isfinite(c):
        return -1e18
    left = max(c - a, 1.0)
    right = max(b - c, 1.0)
    typical = max(float(typical_len), 1.0)
    len_cost = abs(math.log(left / typical)) + abs(math.log(right / typical))
    score = _safe_float(row.get("score", 0.0), 0.0)
    cls = str(row.get("candidate_class", ""))
    jd = _safe_float(row.get("joint_drop", 1.0), 1.0)
    cont = _safe_int(row.get("continuous_rows", 99), 99)
    cluster_size = max(_safe_int(row.get("cluster_size", 1), 1), 1)
    bonus = 0.0
    if cls == "A_ultra_empty":
        bonus += 5.0
    elif cls.startswith("A_"):
        bonus += 3.0
    elif cls.startswith("B_"):
        bonus += 2.0
    if _anchor_recall_is_ultra_rowgap(row, cfg):
        bonus += 5.2
    elif _anchor_recall_is_rowgap(row):
        bonus -= 1.4
    if cont == 0:
        bonus += 2.2
    elif cont == 1:
        bonus -= 0.8
    else:
        bonus -= 2.0
    bonus += 2.8 * max(0.0, 0.35 - jd) / 0.35
    bonus += float(cfg.get("ANCHOR_RECALL_CLUSTER_SIZE_BONUS", 0.75)) * math.log1p(cluster_size)
    length_weight = float(cfg.get("ANCHOR_RECALL_RECALL_LENGTH_WEIGHT", 2.4))
    return float(score + bonus - length_weight * len_cost)


def _anchor_recall_selected_centers(selected: pd.DataFrame) -> List[float]:
    if selected is None or len(selected) == 0 or "center_s" not in selected.columns:
        return []
    vals = pd.to_numeric(selected["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    return sorted([float(v) for v in vals if np.isfinite(v)])


def _anchor_recall_append_boundary(
    current: pd.DataFrame, row: pd.Series, stage: str
) -> pd.DataFrame:
    new_row = row.to_dict()
    new_row["selected_stage"] = stage
    if "accepted" not in new_row:
        new_row["accepted"] = 1
    if current is None or len(current) == 0:
        return pd.DataFrame([new_row]).sort_values("center_s").reset_index(drop=True)
    out = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True, sort=False)
    out = (
        out.drop_duplicates(subset=["center_s"]).sort_values("center_s").reset_index(drop=True)
    )
    return out


def recall_residual_boundaries_in_long_segments(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Insert one supported cut at a time into long segments and recheck the sequence."""
    if not bool(cfg.get("ENABLE_ANCHOR_RECALL_LONG_SEGMENT_RECALL", True)):
        return (selected, pd.DataFrame([{"decision": "disabled"}]))
    if cands is None or len(cands) == 0:
        return (selected, pd.DataFrame([{"decision": "no_candidates"}]))
    current = (
        selected.copy().sort_values("center_s").reset_index(drop=True)
        if selected is not None
        else pd.DataFrame()
    )
    smin, smax = (float(work["s_rot"].min()), float(work["s_rot"].max()))
    typical = _anchor_recall_get_typical_len(mode_info, work, current, cfg)
    max_iter = int(cfg.get("ANCHOR_RECALL_LONG_RECALL_MAX_ITER", 10))
    seg_ratio = float(cfg.get("ANCHOR_RECALL_LONG_RECALL_SEG_RATIO", 1.28))
    min_child_ratio = float(cfg.get("ANCHOR_RECALL_LONG_RECALL_MIN_CHILD_RATIO", 0.32))
    edge_margin = float(
        cfg.get("ANCHOR_RECALL_LONG_RECALL_EDGE_MARGIN", min(9000.0, 0.16 * typical))
    )
    logs = []
    for it in range(max_iter):
        centers = _anchor_recall_selected_centers(current)
        cuts = [smin] + centers + [smax]
        inserted = False
        segs = []
        for i in range(len(cuts) - 1):
            a, b = (float(cuts[i]), float(cuts[i + 1]))
            segs.append((b - a, a, b, i))
        for length, a, b, seg_i in sorted(segs, key=lambda z: z[0], reverse=True):
            if length < seg_ratio * typical:
                continue
            lo = a + edge_margin
            hi = b - edge_margin
            if hi <= lo:
                continue
            sub = cands[
                (pd.to_numeric(cands["center_s"], errors="coerce") > lo)
                & (pd.to_numeric(cands["center_s"], errors="coerce") < hi)
            ].copy()
            if len(sub) == 0:
                continue
            if centers:
                cc = sub["center_s"].astype(float).to_numpy()
                dist = np.min(
                    np.abs(cc[:, None] - np.asarray(centers, dtype=float)[None, :]), axis=1
                )
                sub = sub[
                    dist > float(cfg.get("ANCHOR_RECALL_RECALL_EXISTING_TOL", 1200.0))
                ].copy()
            if len(sub) == 0:
                continue
            sub = sub[
                sub.apply(
                    lambda r: _anchor_recall_is_recall_anchor(r, cfg, terminal=False), axis=1
                )
            ].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "iter": it,
                        "seg_index": seg_i,
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": length,
                        "decision": "no_recall_anchor",
                    }
                )
                continue
            sub["left_len_after_split"] = sub["center_s"].astype(float) - a
            sub["right_len_after_split"] = b - sub["center_s"].astype(float)
            sub = sub[
                (sub["left_len_after_split"] >= min_child_ratio * typical)
                & (sub["right_len_after_split"] >= min_child_ratio * typical)
            ].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "iter": it,
                        "seg_index": seg_i,
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": length,
                        "decision": "no_candidate_child_length_ok",
                    }
                )
                continue
            sub["anchor_recall_recall_rank"] = [
                _anchor_recall_recall_rank(row, a, b, typical, cfg) for _, row in sub.iterrows()
            ]
            best = sub.sort_values(
                ["anchor_recall_recall_rank", "score"], ascending=False
            ).iloc[0]
            current = _anchor_recall_append_boundary(
                current,
                best.drop(labels=["anchor_recall_recall_rank"], errors="ignore"),
                "anchor_recall_long_segment_recall",
            )
            logs.append(
                {
                    "iter": it,
                    "seg_index": int(seg_i),
                    "seg_start": float(a),
                    "seg_end": float(b),
                    "seg_len": float(length),
                    "typical_len_used": float(typical),
                    "candidate_center_s": float(best["center_s"]),
                    "candidate_score": _safe_float(best.get("score", np.nan), np.nan),
                    "candidate_class": str(best.get("candidate_class", "")),
                    "candidate_kind": str(best.get("kind", "")),
                    "candidate_joint_drop": _safe_float(best.get("joint_drop", np.nan), np.nan),
                    "candidate_continuous_rows": _safe_int(best.get("continuous_rows", -1), -1),
                    "left_len_after_split": float(best["left_len_after_split"]),
                    "right_len_after_split": float(best["right_len_after_split"]),
                    "rank": float(best["anchor_recall_recall_rank"]),
                    "decision": "insert_residual_boundary_in_long_segment",
                }
            )
            inserted = True
            break
        if not inserted:
            logs.append(
                {
                    "iter": it,
                    "typical_len_used": float(typical),
                    "decision": "stop_no_long_segment_inserted",
                }
            )
            break
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def recall_terminal_edge_partial_boundary(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test a terminal cut against the main segment and the partial tail."""
    if not bool(cfg.get("ENABLE_ANCHOR_RECALL_TERMINAL_EDGE_RECALL", True)):
        return (selected, pd.DataFrame([{"decision": "disabled"}]))
    if cands is None or len(cands) == 0:
        return (selected, pd.DataFrame([{"decision": "no_candidates"}]))
    current = (
        selected.copy().sort_values("center_s").reset_index(drop=True)
        if selected is not None
        else pd.DataFrame()
    )
    centers = _anchor_recall_selected_centers(current)
    smin, smax = (float(work["s_rot"].min()), float(work["s_rot"].max()))
    typical = _anchor_recall_get_typical_len(mode_info, work, current, cfg)
    a = centers[-1] if centers else smin
    b = smax
    seg_len = b - a
    if seg_len < float(cfg.get("ANCHOR_RECALL_TERMINAL_MIN_PARENT_RATIO", 0.75)) * typical:
        return (
            current,
            pd.DataFrame(
                [
                    {
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": seg_len,
                        "typical_len_used": typical,
                        "decision": "skip_parent_too_short",
                    }
                ]
            ),
        )
    sub = cands[
        (
            pd.to_numeric(cands["center_s"], errors="coerce")
            > a + float(cfg.get("ANCHOR_RECALL_TERMINAL_LEFT_MIN_RATIO", 0.55)) * typical
        )
        & (
            pd.to_numeric(cands["center_s"], errors="coerce")
            < b - float(cfg.get("ANCHOR_RECALL_TERMINAL_MIN_TAIL_LEN", 2500.0))
        )
    ].copy()
    if len(sub) == 0:
        return (
            current,
            pd.DataFrame(
                [
                    {
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": seg_len,
                        "decision": "no_terminal_candidate_in_range",
                    }
                ]
            ),
        )
    sub = sub[
        sub.apply(lambda r: _anchor_recall_is_recall_anchor(r, cfg, terminal=True), axis=1)
    ].copy()
    if len(sub) == 0:
        return (
            current,
            pd.DataFrame(
                [
                    {
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": seg_len,
                        "decision": "no_terminal_anchor",
                    }
                ]
            ),
        )
    sub["left_len_after_split"] = sub["center_s"].astype(float) - a
    sub["right_tail_len_after_split"] = b - sub["center_s"].astype(float)
    sub = sub[
        (
            sub["left_len_after_split"]
            >= float(cfg.get("ANCHOR_RECALL_TERMINAL_LEFT_MIN_RATIO", 0.55)) * typical
        )
        & (
            sub["left_len_after_split"]
            <= float(cfg.get("ANCHOR_RECALL_TERMINAL_LEFT_MAX_RATIO", 1.35)) * typical
        )
        & (
            sub["right_tail_len_after_split"]
            >= float(cfg.get("ANCHOR_RECALL_TERMINAL_MIN_TAIL_LEN", 2500.0))
        )
        & (
            sub["right_tail_len_after_split"]
            <= float(cfg.get("ANCHOR_RECALL_TERMINAL_MAX_TAIL_RATIO", 0.35)) * typical
        )
    ].copy()
    if len(sub) == 0:
        return (
            current,
            pd.DataFrame(
                [
                    {
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": seg_len,
                        "typical_len_used": typical,
                        "decision": "no_terminal_candidate_child_length_ok",
                    }
                ]
            ),
        )
    sub["anchor_recall_terminal_rank"] = [
        _anchor_recall_recall_rank(row, a, b, typical, cfg) + 1.5 for _, row in sub.iterrows()
    ]
    best = sub.sort_values(["anchor_recall_terminal_rank", "score"], ascending=False).iloc[0]
    current = _anchor_recall_append_boundary(
        current,
        best.drop(labels=["anchor_recall_terminal_rank"], errors="ignore"),
        "anchor_recall_terminal_edge_partial_recall",
    )
    log = pd.DataFrame(
        [
            {
                "seg_start": float(a),
                "seg_end": float(b),
                "seg_len": float(seg_len),
                "typical_len_used": float(typical),
                "candidate_center_s": float(best["center_s"]),
                "candidate_score": _safe_float(best.get("score", np.nan), np.nan),
                "candidate_class": str(best.get("candidate_class", "")),
                "candidate_kind": str(best.get("kind", "")),
                "left_len_after_split": float(best["left_len_after_split"]),
                "right_tail_len_after_split": float(best["right_tail_len_after_split"]),
                "decision": "insert_terminal_edge_partial_boundary",
            }
        ]
    )
    return (current.sort_values("center_s").reset_index(drop=True), log)


def _recall_gate_sequence_centers(selected: pd.DataFrame) -> List[float]:
    if selected is None or len(selected) == 0 or "center_s" not in selected.columns:
        return []
    vals = pd.to_numeric(selected["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    return sorted([float(v) for v in vals if np.isfinite(v)])


def _recall_gate_expected_len_for_gate(
    work: pd.DataFrame, selected: pd.DataFrame, cfg: Dict
) -> float:
    segs = _sequence_guard_segments_from_selected(work, selected)
    exp = _sequence_guard_expected_len_from_segments(segs, cfg)
    if np.isfinite(exp) and exp > 1000:
        return float(exp)
    if work is not None and len(work):
        return max((float(work["s_rot"].max()) - float(work["s_rot"].min())) / 4.0, 1000.0)
    return 1000.0


def _recall_gate_segment_stats(
    work: pd.DataFrame, selected: pd.DataFrame, expected: float, cfg: Dict
) -> Dict[str, Any]:
    segs = _sequence_guard_segments_from_selected(work, selected)
    lens = (
        np.array([max(b - a, 1.0) for a, b in segs], dtype=float)
        if segs
        else np.zeros((0,), dtype=float)
    )
    exp = max(float(expected), 1.0)
    ratios = lens / exp if len(lens) else np.zeros((0,), dtype=float)
    short_thr = float(cfg.get("RECALL_GATE_GATE_SHORT_RATIO", 0.5))
    severe_thr = float(cfg.get("RECALL_GATE_GATE_SEVERE_SHORT_RATIO", 0.36))
    long_thr = float(cfg.get("RECALL_GATE_GATE_LONG_RATIO", 2.35))
    return {
        "n_segments": int(len(segs)),
        "min_len": float(np.min(lens)) if len(lens) else np.nan,
        "median_len": float(np.median(lens)) if len(lens) else np.nan,
        "max_len": float(np.max(lens)) if len(lens) else np.nan,
        "n_short": int(np.sum(ratios < short_thr)) if len(ratios) else 0,
        "n_severe_short": int(np.sum(ratios < severe_thr)) if len(ratios) else 0,
        "n_long": int(np.sum(ratios > long_thr)) if len(ratios) else 0,
    }


def _recall_gate_selected_rowgap_count(selected: pd.DataFrame, cfg: Dict) -> int:
    if selected is None or len(selected) == 0:
        return 0
    cnt = 0
    for _, r in selected.iterrows():
        if _anchor_recall_is_rowgap(r) and (not _anchor_recall_is_ultra_rowgap(r, cfg)):
            cnt += 1
    return int(cnt)


def _recall_gate_residual_anchor_count(
    work: pd.DataFrame, selected: pd.DataFrame, cands: pd.DataFrame, cfg: Dict
) -> int:
    try:
        diag = diagnose_final_partition(work, selected, cands, cfg)
        if diag is None or len(diag) == 0 or "residual_local_anchor_count" not in diag.columns:
            return 0
        return int(
            pd.to_numeric(diag["residual_local_anchor_count"], errors="coerce").fillna(0).sum()
        )
    except Exception:
        return 0


def _recall_gate_sequence_cost(
    work: pd.DataFrame, cands: pd.DataFrame, selected: pd.DataFrame, expected: float, cfg: Dict
) -> Tuple[float, Dict[str, Any]]:
    centers = _recall_gate_sequence_centers(selected)
    if work is None or len(work) == 0:
        return (1e18, {})
    smin, smax = (float(work["s_rot"].min()), float(work["s_rot"].max()))
    exp = max(float(expected), 1.0)
    base = (
        _sequence_guard_local_sequence_objective(work, centers, cands, cfg, smin, smax, exp)
        if cands is not None and len(cands)
        else 0.0
    )
    segs = _sequence_guard_segments_from_selected(work, selected)
    length_pen = 0.0
    short_thr = float(cfg.get("RECALL_GATE_GATE_SHORT_RATIO", 0.5))
    severe_thr = float(cfg.get("RECALL_GATE_GATE_SEVERE_SHORT_RATIO", 0.36))
    long_thr = float(cfg.get("RECALL_GATE_GATE_LONG_RATIO", 2.35))
    for a, b in segs:
        r = max(float(b - a), 1.0) / exp
        if r < short_thr:
            length_pen += (
                float(cfg.get("RECALL_GATE_GATE_SHORT_W", 16.0))
                * (short_thr - r)
                / max(short_thr, 1e-06)
            )
        if r < severe_thr:
            length_pen += (
                float(cfg.get("RECALL_GATE_GATE_SEVERE_SHORT_W", 36.0))
                * (severe_thr - r)
                / max(severe_thr, 1e-06)
            )
        if r > long_thr:
            length_pen += float(cfg.get("RECALL_GATE_GATE_LONG_W", 4.0)) * (r - long_thr)
    rowgap_pen = float(
        cfg.get("RECALL_GATE_GATE_ROWGAP_W", 1.8)
    ) * _recall_gate_selected_rowgap_count(selected, cfg)
    residual = _recall_gate_residual_anchor_count(work, selected, cands, cfg)
    residual_pen = float(cfg.get("RECALL_GATE_GATE_RESIDUAL_W", 4.0)) * residual
    total = float(base + length_pen + rowgap_pen + residual_pen)
    st = _recall_gate_segment_stats(work, selected, exp, cfg)
    st.update(
        {
            "expected_len": float(exp),
            "base_obj": float(base),
            "length_penalty": float(length_pen),
            "rowgap_penalty": float(rowgap_pen),
            "residual_anchor_count": int(residual),
            "residual_penalty": float(residual_pen),
            "total_cost": total,
            "n_boundaries": int(len(selected)) if selected is not None else 0,
            "n_rowgap_selected": int(_recall_gate_selected_rowgap_count(selected, cfg)),
        }
    )
    return (total, st)


def _recall_gate_accept_update_with_gate(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    cfg: Dict,
    stage: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not bool(cfg.get("ENABLE_RECALL_GATE_SAFE_RECALL_GATE", True)):
        return (after, pd.DataFrame([{"stage": stage, "decision": "accepted_gate_disabled"}]))
    before = (
        before.copy().sort_values("center_s").reset_index(drop=True)
        if before is not None and len(before)
        else pd.DataFrame()
    )
    after = (
        after.copy().sort_values("center_s").reset_index(drop=True)
        if after is not None and len(after)
        else pd.DataFrame()
    )
    if len(after) == len(before) and np.allclose(
        _recall_gate_sequence_centers(before), _recall_gate_sequence_centers(after), atol=1e-06
    ):
        return (before, pd.DataFrame([{"stage": stage, "decision": "no_change"}]))
    expected = _recall_gate_expected_len_for_gate(work, before, cfg)
    cost_b, stat_b = _recall_gate_sequence_cost(work, cands, before, expected, cfg)
    cost_a, stat_a = _recall_gate_sequence_cost(work, cands, after, expected, cfg)
    obj_delta = float(cost_a - cost_b)
    added = int(len(after) - len(before))
    residual_reduce = int(
        stat_b.get("residual_anchor_count", 0) - stat_a.get("residual_anchor_count", 0)
    )
    short_increase = int(stat_a.get("n_short", 0) - stat_b.get("n_short", 0))
    severe_short_increase = int(
        stat_a.get("n_severe_short", 0) - stat_b.get("n_severe_short", 0)
    )
    boundary_ratio_increase = float((len(after) - len(before)) / max(len(before), 1))
    max_added = int(cfg.get("RECALL_GATE_GATE_MAX_EXTRA_BOUNDARIES", 4))
    max_rel_inc = float(cfg.get("RECALL_GATE_GATE_MAX_REL_BOUNDARY_INCREASE", 0.45))
    slack = float(cfg.get("RECALL_GATE_GATE_OBJ_SLACK", 4.0))
    if "terminal" in stage:
        slack = float(cfg.get("RECALL_GATE_GATE_TERMINAL_MAX_OBJ_WORSE", 2.5))
    accept = False
    reason = "reject_default"
    min_base_for_rel = int(
        cfg.get("FALLBACK_RECOVERY_GATE_MIN_BASE_BOUNDARIES_FOR_REL_LIMIT", 6)
    )
    rel_limit_applicable = len(before) >= min_base_for_rel
    rel_limit_bad = boundary_ratio_increase > max_rel_inc and rel_limit_applicable
    if (
        bool(cfg.get("FALLBACK_RECOVERY_GATE_IGNORE_REL_LIMIT_IF_COST_IMPROVED", True))
        and obj_delta <= 0
    ):
        rel_limit_bad = False
    if (
        bool(cfg.get("FALLBACK_RECOVERY_GATE_IGNORE_REL_LIMIT_IF_RESIDUAL_REDUCED", True))
        and residual_reduce >= int(cfg.get("RECALL_GATE_GATE_RESIDUAL_REDUCE_MIN", 1))
        and (obj_delta <= slack)
    ):
        rel_limit_bad = False
    terminal_partial_ok = False
    terminal_new_center = np.nan
    terminal_tail_len = np.nan
    terminal_left_len = np.nan
    if (
        "terminal" in str(stage).lower()
        and added == 1
        and bool(cfg.get("FALLBACK_RECOVERY_GATE_ALLOW_TERMINAL_PARTIAL_SHORT", True))
    ):
        before_centers = _recall_gate_sequence_centers(before)
        after_centers = _recall_gate_sequence_centers(after)
        new_centers = [
            c for c in after_centers if all((abs(c - b) > 1e-06 for b in before_centers))
        ]
        if len(new_centers) == 1 and work is not None and len(work):
            terminal_new_center = float(new_centers[0])
            smin_gate = float(work["s_rot"].min())
            smax_gate = float(work["s_rot"].max())
            prev_last = before_centers[-1] if before_centers else smin_gate
            terminal_left_len = float(terminal_new_center - prev_last)
            terminal_tail_len = float(smax_gate - terminal_new_center)
            terminal_partial_ok = (
                terminal_new_center > prev_last
                and terminal_tail_len
                >= float(cfg.get("ANCHOR_RECALL_TERMINAL_MIN_TAIL_LEN", 2500.0))
                and (
                    terminal_tail_len
                    <= float(cfg.get("ANCHOR_RECALL_TERMINAL_MAX_TAIL_RATIO", 0.35))
                    * max(expected, 1.0)
                )
                and (
                    terminal_left_len
                    >= float(cfg.get("ANCHOR_RECALL_TERMINAL_LEFT_MIN_RATIO", 0.55))
                    * max(expected, 1.0)
                )
                and (
                    terminal_left_len
                    <= float(cfg.get("ANCHOR_RECALL_TERMINAL_LEFT_MAX_RATIO", 1.35))
                    * max(expected, 1.0)
                )
            )
    if added > max_added:
        accept = False
        reason = "reject_added_boundaries_too_many"
    elif (
        terminal_partial_ok
        and severe_short_increase <= 1
        and (short_increase <= 1)
        and (residual_reduce >= 0)
    ):
        accept = True
        reason = "accept_terminal_partial_recall"
    elif (
        bool(cfg.get("RECALL_GATE_GATE_REJECT_IF_SEVERE_SHORT_INCREASE", True))
        and severe_short_increase > 0
    ):
        accept = False
        reason = "reject_severe_short_increase"
    elif short_increase > int(cfg.get("RECALL_GATE_GATE_REJECT_IF_SHORT_INCREASE_GT", 1)):
        accept = False
        reason = "reject_short_segment_increase"
    elif obj_delta <= 0:
        accept = True
        reason = "accept_cost_improved"
    elif (
        residual_reduce >= int(cfg.get("RECALL_GATE_GATE_RESIDUAL_REDUCE_MIN", 1))
        and obj_delta <= slack
    ):
        accept = True
        reason = "accept_residual_reduced_within_slack"
    elif obj_delta <= min(1.0, slack * 0.25) and severe_short_increase <= 0:
        accept = True
        reason = "accept_nearly_neutral"
    elif rel_limit_bad:
        accept = False
        reason = "reject_boundary_ratio_increase_after_quality_check"
    else:
        accept = False
        reason = "reject_cost_or_quality_worse"
    row = {
        "stage": stage,
        "decision": reason,
        "accepted": int(accept),
        "n_boundaries_before": int(len(before)),
        "n_boundaries_after": int(len(after)),
        "added_boundaries": int(added),
        "boundary_ratio_increase": boundary_ratio_increase,
        "expected_len_gate": float(expected),
        "cost_before": float(cost_b),
        "cost_after": float(cost_a),
        "cost_delta": obj_delta,
        "residual_before": int(stat_b.get("residual_anchor_count", 0)),
        "residual_after": int(stat_a.get("residual_anchor_count", 0)),
        "residual_reduce": int(residual_reduce),
        "short_before": int(stat_b.get("n_short", 0)),
        "short_after": int(stat_a.get("n_short", 0)),
        "short_increase": int(short_increase),
        "severe_short_before": int(stat_b.get("n_severe_short", 0)),
        "severe_short_after": int(stat_a.get("n_severe_short", 0)),
        "severe_short_increase": int(severe_short_increase),
        "long_before": int(stat_b.get("n_long", 0)),
        "long_after": int(stat_a.get("n_long", 0)),
        "rowgap_before": int(stat_b.get("n_rowgap_selected", 0)),
        "rowgap_after": int(stat_a.get("n_rowgap_selected", 0)),
        "rel_limit_applicable": int(rel_limit_applicable),
        "rel_limit_bad_after_quality_check": int(rel_limit_bad),
        "min_base_for_rel_limit": int(min_base_for_rel),
        "terminal_partial_ok": int(terminal_partial_ok),
        "terminal_new_center": (
            float(terminal_new_center) if np.isfinite(terminal_new_center) else np.nan
        ),
        "terminal_left_len": (
            float(terminal_left_len) if np.isfinite(terminal_left_len) else np.nan
        ),
        "terminal_tail_len": (
            float(terminal_tail_len) if np.isfinite(terminal_tail_len) else np.nan
        ),
    }
    return (after if accept else before, pd.DataFrame([row]))


def _adaptive_guard_sorted_centers(selected: pd.DataFrame) -> np.ndarray:
    if selected is None or len(selected) == 0 or "center_s" not in selected.columns:
        return np.zeros((0,), dtype=float)
    vals = pd.to_numeric(selected["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    return np.sort(vals)


def _adaptive_guard_segments(
    work: pd.DataFrame, selected: pd.DataFrame
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    centers = _adaptive_guard_sorted_centers(selected)
    if len(work) == 0:
        return ([], centers)
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    cuts = [smin] + centers.tolist() + [smax]
    segs = [(float(cuts[i]), float(cuts[i + 1])) for i in range(len(cuts) - 1)]
    return (segs, centers)


def _adaptive_guard_get_reference_len(
    mode_info: Dict[str, Any], selected: pd.DataFrame, work: pd.DataFrame, cfg: Dict
) -> Tuple[float, str]:
    vals = []
    for key in [
        "acf_peak_len",
        "typical_len_before_harmonic",
        "typical_len_raw_before_harmonic",
    ]:
        try:
            v = float(mode_info.get(key, np.nan))
            if np.isfinite(v) and v > 1:
                vals.append((v, key))
        except Exception:
            pass
    try:
        typ = float(mode_info.get("typical_len", np.nan))
        if np.isfinite(typ) and typ > 1:
            vals.append((typ, "typical_len"))
    except Exception:
        pass
    if len(vals) == 0:
        segs, _ = _adaptive_guard_segments(work, selected)
        lens = np.array([b - a for a, b in segs], dtype=float)
        lens = lens[np.isfinite(lens) & (lens > 1)]
        if len(lens):
            return (float(np.median(lens)), "selected_median_fallback")
        return (np.nan, "none")
    vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)
    return (float(vals_sorted[0][0]), str(vals_sorted[0][1]))


def _adaptive_guard_boundary_is_rowgap(row: pd.Series) -> bool:
    try:
        return (
            str(row.get("kind", "")) == "row_gap_proposal"
            or int(row.get("is_row_gap_candidate", 0)) == 1
        )
    except Exception:
        return False


def _adaptive_guard_boundary_is_protected(row: pd.Series, cfg: Dict) -> bool:
    try:
        cls = str(row.get("candidate_class", ""))
        score = float(row.get("score", 0.0))
        jd = float(row.get("joint_drop", np.inf))
        cont = int(row.get("continuous_rows", 99))
        strong = int(row.get("n_strong_break", 0))
        weak = int(row.get("n_weak_break", 0))
        rec = int(row.get("left_recover", 0)) + int(row.get("right_recover", 0))
    except Exception:
        return False
    if cls == "A_ultra_empty" and cont <= int(
        cfg.get("ADAPTIVE_GUARD_PROTECT_CONT_ROWS_MAX", 0)
    ):
        return True
    if (
        score >= float(cfg.get("ADAPTIVE_GUARD_PROTECT_SCORE", 14.75))
        and jd <= float(cfg.get("ADAPTIVE_GUARD_PROTECT_JOINT_DROP", 0.13))
        and (cont <= int(cfg.get("ADAPTIVE_GUARD_PROTECT_CONT_ROWS_MAX", 0)))
        and (strong >= int(cfg.get("ADAPTIVE_GUARD_PROTECT_STRONG_BREAK_MIN", 3)))
        and (weak >= int(cfg.get("ADAPTIVE_GUARD_PROTECT_WEAK_BREAK_MIN", 3)))
        and (rec >= 1)
        and (not _adaptive_guard_boundary_is_rowgap(row))
    ):
        return True
    return False


def _adaptive_guard_boundary_is_weak(row: pd.Series, cfg: Dict) -> bool:
    try:
        cls = str(row.get("candidate_class", ""))
        score = float(row.get("score", 0.0))
        jd = float(row.get("joint_drop", 0.0))
        cont = int(row.get("continuous_rows", 0))
        rec = int(row.get("left_recover", 0)) + int(row.get("right_recover", 0))
    except Exception:
        return True
    if _adaptive_guard_boundary_is_rowgap(row):
        return True
    if cls.startswith("C_"):
        return True
    if cls.startswith("B_") and rec < 2:
        return True
    if score < float(cfg.get("ADAPTIVE_GUARD_WEAK_SCORE_TH", 12.2)):
        return True
    if cont >= int(cfg.get("ADAPTIVE_GUARD_WEAK_CONT_ROWS_MIN", 1)) and jd >= float(
        cfg.get("ADAPTIVE_GUARD_WEAK_JOINT_DROP_TH", 0.42)
    ):
        return True
    return False


def _adaptive_guard_length_cost(L: float, ref_len: float, edge: bool = False) -> float:
    if not np.isfinite(ref_len) or ref_len <= 1 or (not np.isfinite(L)) or (L <= 1):
        return 0.0
    r = float(L) / float(ref_len)
    cost = abs(math.log(max(r, 1e-06)))
    if r < 0.55:
        cost += 5.5 * ((0.55 - r) / 0.55) ** 2
    if r < 0.38:
        cost += 14.0 * ((0.38 - r) / 0.38) ** 2
    if r > 1.9:
        cost += 0.55 * (r - 1.9) ** 2
    if edge:
        cost *= 0.65
    return float(cost)


def _adaptive_guard_boundary_delete_penalty(row: pd.Series, cfg: Dict) -> float:
    try:
        score = float(row.get("score", 0.0))
        jd = float(row.get("joint_drop", 0.5))
        cont = int(row.get("continuous_rows", 0))
        cls = str(row.get("candidate_class", ""))
    except Exception:
        return 0.0
    if _adaptive_guard_boundary_is_protected(row, cfg):
        return 999.0
    pen = 0.05 * max(score - 10.0, 0.0)
    if cls == "A_strong":
        pen += 0.65
    if jd <= 0.14 and cont == 0:
        pen += 0.85
    if _adaptive_guard_boundary_is_rowgap(row):
        pen -= 0.9
    if _adaptive_guard_boundary_is_weak(row, cfg):
        pen -= 0.45
    return float(pen)


def diagnose_partition_state(
    work: pd.DataFrame, selected: pd.DataFrame, mode_info: Dict[str, Any], cfg: Dict
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    segs, centers = _adaptive_guard_segments(work, selected)
    lens = np.array([b - a for a, b in segs], dtype=float)
    lens = lens[np.isfinite(lens) & (lens > 1)]
    ref_len, ref_source = _adaptive_guard_get_reference_len(mode_info, selected, work, cfg)
    n_bound = int(len(centers))
    n_plots = int(n_bound + 1)
    med_len = float(np.median(lens)) if len(lens) else np.nan
    short_ratio = (
        float(
            np.mean(
                lens
                < float(cfg.get("ADAPTIVE_GUARD_OVERSPLIT_MEDIAN_VS_ACF_RATIO", 0.72)) * ref_len
            )
        )
        if len(lens) and np.isfinite(ref_len)
        else np.nan
    )
    severe_short_ratio = (
        float(np.mean(lens < 0.45 * ref_len)) if len(lens) and np.isfinite(ref_len) else np.nan
    )
    weak_ratio = np.nan
    rowgap_ratio = np.nan
    protected_ratio = np.nan
    if selected is not None and len(selected):
        weak_flags = selected.apply(
            lambda r: _adaptive_guard_boundary_is_weak(r, cfg), axis=1
        ).astype(bool)
        rowgap_flags = selected.apply(
            lambda r: _adaptive_guard_boundary_is_rowgap(r), axis=1
        ).astype(bool)
        prot_flags = selected.apply(
            lambda r: _adaptive_guard_boundary_is_protected(r, cfg), axis=1
        ).astype(bool)
        weak_ratio = float(weak_flags.mean())
        rowgap_ratio = float(rowgap_flags.mean())
        protected_ratio = float(prot_flags.mean())
    med_vs_ref = (
        float(med_len / ref_len)
        if np.isfinite(med_len) and np.isfinite(ref_len) and (ref_len > 1)
        else np.nan
    )
    flags = []
    if (
        n_bound >= int(cfg.get("ADAPTIVE_GUARD_OVERSPLIT_MIN_BOUNDARIES", 8))
        and np.isfinite(med_vs_ref)
        and (med_vs_ref < float(cfg.get("ADAPTIVE_GUARD_OVERSPLIT_MEDIAN_VS_ACF_RATIO", 0.72)))
    ):
        flags.append("selected_median_much_shorter_than_reference")
    if np.isfinite(short_ratio) and short_ratio > float(
        cfg.get("ADAPTIVE_GUARD_OVERSPLIT_SHORT_RATIO_TH", 0.34)
    ):
        flags.append("many_short_segments_vs_reference")
    if np.isfinite(weak_ratio) and weak_ratio > float(
        cfg.get("ADAPTIVE_GUARD_OVERSPLIT_WEAK_BOUNDARY_RATIO_TH", 0.28)
    ):
        flags.append("weak_boundary_ratio_high")
    if np.isfinite(rowgap_ratio) and rowgap_ratio > float(
        cfg.get("ADAPTIVE_GUARD_OVERSPLIT_ROWGAP_RATIO_TH", 0.18)
    ):
        flags.append("rowgap_boundary_ratio_high")
    try:
        if (
            bool(cfg.get("ADAPTIVE_GUARD_HARMONIC_HALF_OVERSPLIT_CHECK", True))
            and int(mode_info.get("harmonic_corrected", 0)) == 1
        ):
            typ = float(mode_info.get("typical_len", np.nan))
            acf = float(mode_info.get("acf_peak_len", np.nan))
            if np.isfinite(typ) and np.isfinite(acf) and (typ < 0.6 * acf):
                flags.append("typical_len_harmonic_corrected_to_half_acf")
    except Exception:
        pass
    over_split_risk = int(len(flags) >= int(cfg.get("ADAPTIVE_GUARD_OVERSPLIT_MIN_FLAGS", 2)))
    diag = {
        "stage": "adaptive_guard_partition_state_diagnosis",
        "n_boundaries": n_bound,
        "n_plots": n_plots,
        "reference_len": float(ref_len) if np.isfinite(ref_len) else np.nan,
        "reference_len_source": ref_source,
        "selected_median_len": med_len,
        "selected_median_vs_reference": med_vs_ref,
        "short_segment_ratio_vs_reference": short_ratio,
        "severe_short_segment_ratio_vs_reference": severe_short_ratio,
        "weak_boundary_ratio": weak_ratio,
        "rowgap_boundary_ratio": rowgap_ratio,
        "protected_boundary_ratio": protected_ratio,
        "over_split_risk": over_split_risk,
        "risk_flags": ";".join(flags) if flags else "none",
        "mode_typical_len": (
            float(mode_info.get("typical_len", np.nan))
            if isinstance(mode_info, dict)
            else np.nan
        ),
        "mode_acf_peak_len": (
            float(mode_info.get("acf_peak_len", np.nan))
            if isinstance(mode_info, dict)
            else np.nan
        ),
        "mode_harmonic_corrected": (
            int(mode_info.get("harmonic_corrected", 0)) if isinstance(mode_info, dict) else 0
        ),
    }
    return (diag, pd.DataFrame([diag]))


def merge_oversplit_boundaries(
    work: pd.DataFrame, selected: pd.DataFrame, mode_info: Dict[str, Any], cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected is None or len(selected) == 0:
        return (
            selected,
            pd.DataFrame(
                [{"stage": "adaptive_guard_safe_merge", "decision": "skip_no_boundaries"}]
            ),
        )
    if not bool(cfg.get("ADAPTIVE_GUARD_ENABLE_SAFE_MERGE_WEAK_BOUNDARIES", True)):
        return (
            selected,
            pd.DataFrame([{"stage": "adaptive_guard_safe_merge", "decision": "disabled"}]),
        )
    diag, _ = diagnose_partition_state(work, selected, mode_info, cfg)
    if int(diag.get("over_split_risk", 0)) != 1:
        return (
            selected,
            pd.DataFrame(
                [
                    {
                        **diag,
                        "stage": "adaptive_guard_safe_merge",
                        "decision": "skip_no_oversplit_risk",
                    }
                ]
            ),
        )
    ref_len = float(diag.get("reference_len", np.nan))
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    start_n = len(current)
    max_remove = max(
        1,
        int(
            math.floor(
                float(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_MAX_REMOVE_FRACTION", 0.45)) * start_n
            )
        ),
    )
    logs = []
    removed = 0
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    for it in range(int(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_MAX_ITER", 80))):
        centers = _adaptive_guard_sorted_centers(current)
        if len(centers) == 0 or removed >= max_remove:
            break
        cuts = [smin] + centers.tolist() + [smax]
        lens = np.array([cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)], dtype=float)
        med_vs_ref = (
            float(np.median(lens) / ref_len)
            if np.isfinite(ref_len) and ref_len > 1 and len(lens)
            else np.nan
        )
        short_ratio = (
            float(np.mean(lens < 0.72 * ref_len))
            if np.isfinite(ref_len) and ref_len > 1 and len(lens)
            else np.nan
        )
        if (
            np.isfinite(med_vs_ref)
            and med_vs_ref
            >= float(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_STOP_MEDIAN_VS_ACF_RATIO", 0.82))
            and np.isfinite(short_ratio)
            and (
                short_ratio
                <= float(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_SHORT_RATIO_STOP", 0.22))
            )
        ):
            logs.append(
                {
                    "stage": "adaptive_guard_safe_merge",
                    "iter": it,
                    "decision": "stop_risk_reduced",
                    "median_vs_ref": med_vs_ref,
                    "short_ratio": short_ratio,
                }
            )
            break
        best = None
        for j, c in enumerate(centers):
            dist_series = np.abs(current["center_s"].astype(float) - float(c))
            idx0 = int(dist_series.to_numpy().argmin())
            row = current.iloc[idx0]
            if _adaptive_guard_boundary_is_protected(row, cfg):
                continue
            L1 = float(cuts[j + 1] - cuts[j])
            L2 = float(cuts[j + 2] - cuts[j + 1])
            Lm = L1 + L2
            rmerged = Lm / max(ref_len, 1.0) if np.isfinite(ref_len) else np.nan
            max_ratio = float(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_MAX_MERGED_RATIO", 1.75))
            if _adaptive_guard_boundary_is_weak(row, cfg):
                max_ratio = float(
                    cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_ALLOW_LONGER_IF_WEAK", 2.15)
                )
            if np.isfinite(rmerged) and (
                rmerged < float(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_MIN_MERGED_RATIO", 0.58))
                or rmerged > max_ratio
            ):
                continue
            before = _adaptive_guard_length_cost(
                L1, ref_len, edge=j == 0
            ) + _adaptive_guard_length_cost(L2, ref_len, edge=j + 1 == len(lens) - 1)
            after = _adaptive_guard_length_cost(
                Lm, ref_len, edge=j == 0 or j + 1 == len(lens) - 1
            )
            del_pen = _adaptive_guard_boundary_delete_penalty(row, cfg)
            improve = before - after - del_pen
            has_short_child = (
                L1 < 0.72 * ref_len or L2 < 0.72 * ref_len if np.isfinite(ref_len) else False
            )
            weak = _adaptive_guard_boundary_is_weak(row, cfg)
            if not (has_short_child or weak):
                continue
            if improve > float(cfg.get("ADAPTIVE_GUARD_SAFE_MERGE_MIN_IMPROVE", 0.18)):
                rec = {
                    "stage": "adaptive_guard_safe_merge",
                    "iter": it,
                    "decision": "candidate_remove_boundary",
                    "remove_center_s": float(c),
                    "candidate_class": str(row.get("candidate_class", "")),
                    "kind": str(row.get("kind", "")),
                    "score": float(row.get("score", np.nan)),
                    "joint_drop": float(row.get("joint_drop", np.nan)),
                    "continuous_rows": (
                        int(row.get("continuous_rows", -1))
                        if pd.notna(row.get("continuous_rows", np.nan))
                        else -1
                    ),
                    "left_len": L1,
                    "right_len": L2,
                    "merged_len": Lm,
                    "merged_ratio_ref": rmerged,
                    "before_cost": before,
                    "after_cost": after,
                    "delete_penalty": del_pen,
                    "improve": improve,
                    "is_weak_boundary": int(weak),
                    "is_rowgap_boundary": int(_adaptive_guard_boundary_is_rowgap(row)),
                }
                if best is None or rec["improve"] > best["improve"]:
                    best = rec
        if best is None:
            logs.append(
                {
                    "stage": "adaptive_guard_safe_merge",
                    "iter": it,
                    "decision": "stop_no_safe_merge_found",
                }
            )
            break
        center = float(best["remove_center_s"])
        idx = int(np.argmin(np.abs(current["center_s"].to_numpy(dtype=float) - center)))
        current = current.drop(current.index[idx]).reset_index(drop=True)
        removed += 1
        best["decision"] = "accept_remove_boundary"
        best["removed_so_far"] = removed
        logs.append(best)
    final_diag, _ = diagnose_partition_state(work, current, mode_info, cfg)
    logs.append(
        {
            **final_diag,
            "stage": "adaptive_guard_safe_merge",
            "decision": "final_state_after_safe_merge",
            "removed_total": removed,
            "start_boundaries": start_n,
            "end_boundaries": len(current),
        }
    )
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def _observed_proposal_min_observed_gap_width(ref_len: float, cfg: Dict) -> float:
    """Observed proposal min observed gap width."""
    if not np.isfinite(ref_len) or ref_len <= 1:
        ref_len = 70000.0
    raw = max(
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_MIN_ABS", 7600.0)),
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_MIN_RATIO", 0.105)) * float(ref_len),
    )
    return float(min(raw, float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_MIN_MAX", 15000.0))))


def _observed_proposal_reference_len_for_observed_recall(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict
) -> Tuple[float, Dict[str, Any]]:
    """Estimate a reference scale for observed-gap candidate generation."""
    try:
        typical_len, meta = robust_typical_length_from_candidates(cands, work, cfg)
        typical_len2, hmeta = correct_typical_length_by_harmonics(typical_len, cands, cfg, meta)
        out = dict(meta)
        out.update(hmeta)
        out["observed_proposal_ref_len_before_harmonic"] = (
            float(typical_len) if np.isfinite(typical_len) else np.nan
        )
        out["observed_proposal_ref_len"] = (
            float(typical_len2) if np.isfinite(typical_len2) else np.nan
        )
        if np.isfinite(typical_len2) and typical_len2 > 1:
            return (float(typical_len2), out)
    except Exception as e:
        out = {"observed_proposal_ref_len_error": str(e)}
    try:
        acf_len, acf_meta = estimate_typical_length_from_density_acf(work, cfg)
        out.update({"observed_proposal_ref_len_fallback": "density_acf", **acf_meta})
        if np.isfinite(acf_len) and acf_len > 1:
            return (float(acf_len), out)
    except Exception as e:
        out["observed_proposal_acf_error"] = str(e)
    span = (
        float(work["s_rot"].max() - work["s_rot"].min())
        if work is not None and len(work)
        else np.nan
    )
    fallback = (
        70000.0
        if not np.isfinite(span) or span <= 1
        else max(30000.0, min(90000.0, span / 20.0))
    )
    out["observed_proposal_ref_len_fallback"] = "span_based"
    out["observed_proposal_ref_len"] = float(fallback)
    return (float(fallback), out)


def build_observed_gap_proposals(
    work: pd.DataFrame, classified_cands: pd.DataFrame, cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge row-level low-density intervals into observed-gap candidates."""
    if not bool(cfg.get("ENABLE_OBSERVED_PROPOSAL_OBSERVED_GAP_INTERVAL_PROPOSALS", True)):
        return (
            classified_cands,
            pd.DataFrame(
                [
                    {
                        "stage": "observed_proposal_observed_gap_interval_proposals",
                        "decision": "disabled",
                    }
                ]
            ),
        )
    if work is None or len(work) == 0:
        return (
            classified_cands,
            pd.DataFrame(
                [
                    {
                        "stage": "observed_proposal_observed_gap_interval_proposals",
                        "decision": "skip_empty_work",
                    }
                ]
            ),
        )
    cands = classified_cands.copy() if classified_cands is not None else pd.DataFrame()
    ref_len, ref_meta = _observed_proposal_reference_len_for_observed_recall(work, cands, cfg)
    min_w = _observed_proposal_min_observed_gap_width(ref_len, cfg)
    weak_w = float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_WEAK_FACTOR", 0.72)) * min_w
    cluster_tol = min(
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_CLUSTER_TOL_ABS", 9000.0)),
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_CLUSTER_TOL_RATIO", 0.13)) * max(ref_len, 1.0),
    )
    spread_max = min(
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_CLUSTER_SPREAD_ABS", 18000.0)),
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_CLUSTER_SPREAD_RATIO", 0.26))
        * max(ref_len, 1.0),
    )
    edge_guard = min(
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_EDGE_GUARD_ABS", 12000.0)),
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_EDGE_GUARD_RATIO", 0.18)) * max(ref_len, 1.0),
    )
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    row_col = (
        "row_main"
        if "row_main" in work.columns
        else "row_id" if "row_id" in work.columns else None
    )
    gap_records: List[Dict[str, Any]] = []
    bin_w = min(
        max(
            float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_BIN_ABS", 1000.0)),
            float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_BIN_RATIO", 0.016)) * max(ref_len, 1.0),
        ),
        float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_BIN_MAX", 1800.0)),
    )
    low_frac = float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_LOW_DENSITY_FRAC", 0.45))
    low_abs = float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_LOW_DENSITY_ABS", 2.0))
    min_run_w = float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_MIN_RUN_FACTOR", 0.68)) * min_w
    if row_col is None:
        grouped = [(0, work)]
    else:
        grouped = [(rid, sub) for rid, sub in work.groupby(row_col)]
    edges = np.arange(smin, smax + bin_w * 1.5, bin_w, dtype=float)
    for rid, sub in grouped:
        ss = (
            pd.to_numeric(sub.get("s_rot", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        ss = np.sort(ss)
        if ss.size < 5 or edges.size < 4:
            continue
        counts, _ = np.histogram(ss, bins=edges)
        pos = counts[counts > 0]
        dense_ref = float(np.quantile(pos, 0.75)) if pos.size else 1.0
        low_thr = max(low_abs, low_frac * dense_ref)
        low = counts <= low_thr
        close_bins = int(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_CLOSE_HIGH_HOLES_BINS", 1))
        if close_bins > 0 and len(low) > 2:
            low2 = low.copy()
            i = 0
            while i < len(low2):
                if low2[i]:
                    i += 1
                    continue
                j = i
                while j < len(low2) and (not low2[j]):
                    j += 1
                if i > 0 and j < len(low2) and (j - i <= close_bins):
                    low2[i:j] = True
                i = j
            low = low2
        start_i = None
        for bi, is_low in enumerate(low):
            if is_low and start_i is None:
                start_i = bi
            at_end = bi == len(low) - 1
            if (not is_low or at_end) and start_i is not None:
                end_i = bi if not is_low else bi + 1
                left_s = float(edges[start_i])
                right_s = float(edges[end_i])
                run_w = right_s - left_s
                run_count = int(counts[start_i:end_i].sum())
                if (
                    run_w >= min_run_w
                    and 0.5 * (left_s + right_s) > smin + edge_guard
                    and (0.5 * (left_s + right_s) < smax - edge_guard)
                ):
                    gap_records.append(
                        {
                            "row_id": int(rid) if str(rid).lstrip("-").isdigit() else rid,
                            "left_s": left_s,
                            "right_s": right_s,
                            "center_s": float(0.5 * (left_s + right_s)),
                            "gap_width": float(run_w),
                            "run_point_count": int(run_count),
                            "row_dense_ref": float(dense_ref),
                            "low_density_threshold": float(low_thr),
                            "bin_w": float(bin_w),
                            "strong_support": int(float(run_w) >= min_w),
                            "weak_support": int(float(run_w) >= weak_w),
                        }
                    )
                start_i = None
    logs: List[Dict[str, Any]] = []
    if len(gap_records) == 0:
        meta_row = {
            "stage": "observed_proposal_observed_gap_interval_proposals",
            "decision": "no_row_gaps",
            "ref_len": ref_len,
            "min_gap_width": min_w,
            "weak_gap_width": weak_w,
        }
        meta_row.update(ref_meta)
        return (cands, pd.DataFrame([meta_row]))
    gaps_df = pd.DataFrame(gap_records).sort_values("center_s").reset_index(drop=True)
    clusters = []
    current = []
    for _, r in gaps_df.iterrows():
        if not current:
            current = [r]
            continue
        cur_centers = [float(x["center_s"]) for x in current]
        if abs(float(r["center_s"]) - float(np.median(cur_centers))) <= cluster_tol:
            current.append(r)
        else:
            clusters.append(current)
            current = [r]
    if current:
        clusters.append(current)
    proposal_rows: List[Dict[str, Any]] = []
    template_cols = (
        list(cands.columns)
        if len(cands.columns)
        else ["s0", "s1", "center_s", "width", "score", "kind"]
    )
    max_props = int(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_MAX_PROPOSALS", 260))
    for ci, cl in enumerate(clusters):
        dfc = pd.DataFrame(cl)
        dfc = (
            dfc.sort_values("gap_width", ascending=False)
            .drop_duplicates(subset=["row_id"], keep="first")
            .sort_values("center_s")
        )
        support_rows = int(dfc["strong_support"].sum())
        weak_rows = int(dfc["weak_support"].sum())
        row_count = int(dfc["row_id"].nunique())
        center = float(np.median(dfc["center_s"].to_numpy(dtype=float)))
        spread = (
            float(np.max(dfc["center_s"]) - np.min(dfc["center_s"])) if len(dfc) else np.nan
        )
        med_gap = float(np.median(dfc["gap_width"].to_numpy(dtype=float)))
        min_gap = float(np.min(dfc["gap_width"].to_numpy(dtype=float)))
        max_gap = float(np.max(dfc["gap_width"].to_numpy(dtype=float)))
        decision = "candidate"
        if support_rows < int(
            cfg.get("OBSERVED_PROPOSAL_OBS_GAP_MIN_SUPPORT_ROWS", 2)
        ) or weak_rows < int(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_WEAK_SUPPORT_ROWS", 2)):
            decision = "reject_insufficient_row_support"
        elif not np.isfinite(spread) or spread > spread_max:
            decision = "reject_row_gap_centers_too_scattered"
        elif med_gap < 0.8 * min_w:
            decision = "reject_median_gap_too_small"
        logs.append(
            {
                "stage": "observed_proposal_observed_gap_interval_proposals",
                "cluster_id": int(ci),
                "decision": decision,
                "center_s": center,
                "support_rows": support_rows,
                "weak_support_rows": weak_rows,
                "row_count": row_count,
                "median_gap_width": med_gap,
                "min_gap_width_observed": min_gap,
                "max_gap_width_observed": max_gap,
                "center_spread": spread,
                "required_gap_width": min_w,
                "weak_gap_width": weak_w,
                "cluster_tol": cluster_tol,
                "spread_max": spread_max,
                "row_gap_details": ";".join(
                    [
                        f"r{rr}:{lw:.1f}-{rw:.1f},w={gw:.1f}"
                        for rr, lw, rw, gw in zip(
                            dfc["row_id"], dfc["left_s"], dfc["right_s"], dfc["gap_width"]
                        )
                    ]
                ),
                "ref_len": ref_len,
                **ref_meta,
            }
        )
        if decision != "candidate":
            continue
        width_for_band = float(
            max(
                float(cfg.get("MIN_BAND_WIDTH", 1000.0)),
                min(med_gap, float(cfg.get("MAX_BAND_WIDTH", 6200.0))),
            )
        )
        score = (
            float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_SCORE_BASE", 11.0))
            + float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_SCORE_PER_SUPPORT_ROW", 2.1))
            * support_rows
            + float(cfg.get("OBSERVED_PROPOSAL_OBS_GAP_SCORE_WIDTH_BONUS", 2.2))
            * min(1.8, med_gap / max(min_w, 1.0))
        )
        row = {col: np.nan for col in template_cols}
        row.update(
            {
                "s0": center - 0.5 * width_for_band,
                "s1": center + 0.5 * width_for_band,
                "center_s": center,
                "width": width_for_band,
                "score": score,
                "kind": "observed_gap_interval",
                "strict_high_conf": 1,
                "n_strong_break": min(int(cfg.get("MAIN_ROWS", 3)), support_rows),
                "n_weak_break": min(int(cfg.get("MAIN_ROWS", 3)), max(weak_rows, support_rows)),
                "joint_inside": 0.0,
                "joint_ref": 1.0,
                "joint_drop": max(0.005, min(0.12, 1.0 - med_gap / max(ref_len, 1.0))),
                "left_recover": 1,
                "right_recover": 1,
                "left_joint": 1.0,
                "right_joint": 1.0,
                "left_recover_rows": min(int(cfg.get("MAIN_ROWS", 3)), row_count),
                "right_recover_rows": min(int(cfg.get("MAIN_ROWS", 3)), row_count),
                "mean_top2_gap_ratio": min(1.5, med_gap / max(width_for_band, 1.0)),
                "dominant_single_row_penalty": 0.0,
                "continuous_rows": 0,
                "continuity_penalty": 0.0,
                "row_gap_support_rows": ",".join([str(x) for x in dfc["row_id"].tolist()]),
                "row_gap_support_count": int(row_count),
                "row_gap_support_row_count": int(row_count),
                "cluster_rank": 999,
                "cluster_size": int(row_count),
                "cluster_center_min": float(dfc["center_s"].min()),
                "cluster_center_max": float(dfc["center_s"].max()),
                "candidate_class": "A_observed_gap",
                "candidate_class_rank": 4,
                "is_row_gap_candidate": 0,
                "is_graph_strong": True,
                "is_graph_usable": True,
                "observed_proposal_observed_gap_support_rows": support_rows,
                "observed_proposal_observed_gap_weak_rows": weak_rows,
                "observed_proposal_observed_gap_median_width": med_gap,
                "observed_proposal_observed_gap_min_width": min_gap,
                "observed_proposal_observed_gap_max_width": max_gap,
                "observed_proposal_observed_gap_center_spread": spread,
                "selected_stage": "observed_proposal_observed_gap_interval_proposal",
            }
        )
        proposal_rows.append(row)
    if proposal_rows:
        proposals = pd.DataFrame(proposal_rows)
        proposals = proposals.sort_values(["center_s", "score"]).reset_index(drop=True)
        keep = []
        last_center = None
        for idx, r in proposals.sort_values("score", ascending=False).iterrows():
            c = float(r["center_s"])
            if last_center is None:
                keep.append(idx)
                continue
            kept_centers = [float(proposals.loc[k, "center_s"]) for k in keep]
            if all((abs(c - kc) > 0.5 * cluster_tol for kc in kept_centers)):
                keep.append(idx)
        proposals = (
            proposals.loc[keep]
            .copy()
            .sort_values("center_s")
            .head(max_props)
            .reset_index(drop=True)
        )
        out = pd.concat([cands, proposals], ignore_index=True, sort=False)
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        near_tol = max(300.0, 0.1 * cluster_tol)
        out["_observed_proposal_center_bin"] = np.round(
            pd.to_numeric(out["center_s"], errors="coerce") / near_tol
        ).astype("Int64")
        out = (
            out.drop_duplicates(subset=["_observed_proposal_center_bin"], keep="first")
            .drop(columns=["_observed_proposal_center_bin"])
            .sort_values("center_s")
            .reset_index(drop=True)
        )
        logs.append(
            {
                "stage": "observed_proposal_observed_gap_interval_proposals",
                "decision": "final_state",
                "n_row_gap_records": int(len(gap_records)),
                "n_clusters": int(len(clusters)),
                "n_accepted_proposals": int(len(proposals)),
                "n_candidates_before": int(len(cands)),
                "n_candidates_after": int(len(out)),
                "ref_len": ref_len,
                "required_gap_width": min_w,
            }
        )
        return (out, pd.DataFrame(logs))
    logs.append(
        {
            "stage": "observed_proposal_observed_gap_interval_proposals",
            "decision": "final_state",
            "n_row_gap_records": int(len(gap_records)),
            "n_clusters": int(len(clusters)),
            "n_accepted_proposals": 0,
            "n_candidates_before": int(len(cands)),
            "n_candidates_after": int(len(cands)),
            "ref_len": ref_len,
            "required_gap_width": min_w,
        }
    )
    return (cands, pd.DataFrame(logs))


def _observed_sequence_ref_len(mode_info: Dict[str, Any], cfg: Dict) -> float:
    """Choose a reference scale while checking candidate-spacing harmonics."""
    keys = [
        "acf_peak_len",
        "typical_len_before_harmonic",
        "typical_len_raw_before_harmonic",
        "typical_len_high_conf",
        "typical_len",
    ]
    vals = []
    for k in keys:
        try:
            v = float(mode_info.get(k, np.nan))
            if np.isfinite(v) and v > 5000:
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return 60000.0
    return float(max(vals[:3]) if len(vals) >= 2 else vals[0])


def _observed_sequence_len_cost(
    length: float, ref_len: float, cfg: Dict, is_edge: bool = False
) -> float:
    if not np.isfinite(length) or length <= 0 or (not np.isfinite(ref_len)) or (ref_len <= 1):
        return 1000000.0
    r = float(length) / float(ref_len)
    cost = abs(np.log(max(r, 1e-06)))
    short_floor = 0.35 if is_edge else 0.45
    if r < short_floor:
        cost += 4.0 * (short_floor - r) / short_floor
    if r > 2.2:
        cost += 1.5 * (r - 2.2)
    return float(cost)


def _observed_sequence_get_observed_candidates(cands: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return pd.DataFrame()
    cc = classify_boundary_candidates(cands, cfg).copy()
    if "kind" not in cc.columns:
        return pd.DataFrame()
    m = cc["kind"].astype(str).eq("observed_gap_interval")
    support = pd.to_numeric(
        cc.get("observed_proposal_observed_gap_support_rows", 0), errors="coerce"
    ).fillna(0)
    med_w = pd.to_numeric(
        cc.get("observed_proposal_observed_gap_median_width", cc.get("width", 0)),
        errors="coerce",
    ).fillna(0)
    m &= support >= int(cfg.get("OBSERVED_SEQUENCE_OBS_GAP_MIN_SUPPORT_ROWS", 2))
    m &= med_w >= float(cfg.get("OBSERVED_SEQUENCE_OBS_GAP_MIN_WIDTH_ABS", 6500.0))
    out = cc.loc[m].copy()
    if len(out):
        out = out.sort_values(["center_s", "score"]).drop_duplicates(
            subset=["center_s"], keep="last"
        )
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


def _observed_sequence_pair_cost(
    prev_s: float, cut_s: float, next_s: float, ref_len: float, cfg: Dict
) -> float:
    return _observed_sequence_len_cost(
        cut_s - prev_s, ref_len, cfg, is_edge=False
    ) + _observed_sequence_len_cost(next_s - cut_s, ref_len, cfg, is_edge=False)


def refine_observed_gap_sequence(
    work: pd.DataFrame,
    cands: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Refine or insert boundaries using observed intervals and adjacent segment costs."""
    if not bool(cfg.get("ENABLE_OBSERVED_SEQUENCE_OBSERVED_GAP_SEQUENCE_FIX", True)):
        return (
            selected.copy() if selected is not None else pd.DataFrame(),
            pd.DataFrame(
                [
                    {
                        "stage": "observed_sequence_observed_gap_sequence_fix",
                        "decision": "disabled",
                    }
                ]
            ),
        )
    cur = (
        selected.copy().sort_values("center_s").reset_index(drop=True)
        if selected is not None and len(selected)
        else pd.DataFrame()
    )
    obs = _observed_sequence_get_observed_candidates(cands, cfg)
    logs: List[Dict[str, Any]] = []
    if len(cur) == 0 or len(obs) == 0 or work is None or (len(work) == 0):
        logs.append(
            {
                "stage": "observed_sequence_observed_gap_sequence_fix",
                "decision": "skip_empty_selected_or_observed",
                "n_selected": int(len(cur)),
                "n_observed": int(len(obs)),
            }
        )
        return (cur, pd.DataFrame(logs))
    ref_len = _observed_sequence_ref_len(mode_info, cfg)
    smin = float(pd.to_numeric(work["s_rot"], errors="coerce").min())
    smax = float(pd.to_numeric(work["s_rot"], errors="coerce").max())
    search_w = min(
        float(cfg.get("OBSERVED_SEQUENCE_RELOCATE_SEARCH_ABS", 32000.0)),
        float(cfg.get("OBSERVED_SEQUENCE_RELOCATE_SEARCH_RATIO", 0.55)) * max(ref_len, 1.0),
    )
    min_improve = float(cfg.get("OBSERVED_SEQUENCE_RELOCATE_MIN_IMPROVE", 0.08))
    changed = True
    iter_no = 0
    while changed and iter_no < 6:
        changed = False
        iter_no += 1
        cur = cur.sort_values("center_s").reset_index(drop=True)
        centers = cur["center_s"].to_numpy(dtype=float)
        for i in range(len(cur)):
            old_c = float(cur.loc[i, "center_s"])
            prev_s = smin if i == 0 else float(cur.loc[i - 1, "center_s"])
            next_s = smax if i == len(cur) - 1 else float(cur.loc[i + 1, "center_s"])
            local = obs[
                (obs["center_s"].astype(float) > prev_s + 1.0)
                & (obs["center_s"].astype(float) < next_s - 1.0)
            ].copy()
            local = local[np.abs(local["center_s"].astype(float) - old_c) <= search_w].copy()
            if len(local) == 0:
                continue
            before = _observed_sequence_pair_cost(prev_s, old_c, next_s, ref_len, cfg)
            best = None
            for _, r in local.iterrows():
                c = float(r["center_s"])
                if abs(c - old_c) <= float(cfg.get("OBSERVED_SEQUENCE_SAME_CENTER_TOL", 500.0)):
                    continue
                if (
                    c - prev_s
                    < float(cfg.get("OBSERVED_SEQUENCE_INSERT_MIN_SEG_RATIO", 0.48)) * ref_len
                ):
                    continue
                if (
                    next_s - c
                    < float(cfg.get("OBSERVED_SEQUENCE_INSERT_MIN_SEG_RATIO", 0.48)) * ref_len
                ):
                    continue
                after = _observed_sequence_pair_cost(prev_s, c, next_s, ref_len, cfg)
                support = _safe_float(
                    r.get("observed_proposal_observed_gap_support_rows", 0), 0.0
                )
                med_w = _safe_float(
                    r.get("observed_proposal_observed_gap_median_width", r.get("width", 0.0)),
                    0.0,
                )
                obs_bonus = 0.05 * support + 0.05 * min(
                    2.0,
                    med_w
                    / max(
                        float(cfg.get("OBSERVED_SEQUENCE_OBS_GAP_MIN_WIDTH_ABS", 6500.0)), 1.0
                    ),
                )
                improve = before - after + obs_bonus
                rec = {
                    "stage": "observed_sequence_observed_gap_relocate",
                    "iter": int(iter_no),
                    "boundary_index": int(i),
                    "decision": "candidate",
                    "old_center_s": old_c,
                    "new_center_s": c,
                    "prev_s": prev_s,
                    "next_s": next_s,
                    "old_left_len": old_c - prev_s,
                    "old_right_len": next_s - old_c,
                    "new_left_len": c - prev_s,
                    "new_right_len": next_s - c,
                    "cost_before": before,
                    "cost_after": after,
                    "improve": improve,
                    "ref_len": ref_len,
                    "obs_support_rows": support,
                    "obs_median_width": med_w,
                    "obs_score": _safe_float(r.get("score", np.nan), np.nan),
                }
                if best is None or rec["improve"] > best["improve"]:
                    best = (rec, r)
            if best is not None and best[0]["improve"] >= min_improve:
                rec, r = best
                new_row = cur.loc[i].copy()
                for col in r.index:
                    new_row[col] = r[col]
                cur.loc[i] = new_row
                rec["decision"] = "accept_replace_selected_boundary_with_observed_gap"
                logs.append(rec)
                changed = True
                break
            elif best is not None:
                rec, _ = best
                rec["decision"] = "reject_improve_too_small"
                logs.append(rec)
    cur = cur.sort_values("center_s").reset_index(drop=True)
    inserted = 0
    max_insert = int(cfg.get("OBSERVED_SEQUENCE_MAX_INSERTS", 8))
    for _ in range(max_insert):
        centers = cur["center_s"].to_numpy(dtype=float)
        cuts = np.concatenate([[smin], centers, [smax]])
        best = None
        for si in range(len(cuts) - 1):
            a, b = (float(cuts[si]), float(cuts[si + 1]))
            seg_len = b - a
            if (
                seg_len
                < float(cfg.get("OBSERVED_SEQUENCE_INSERT_MIN_MERGED_RATIO", 1.35)) * ref_len
            ):
                continue
            local = obs[
                (
                    obs["center_s"].astype(float)
                    > a
                    + float(cfg.get("OBSERVED_SEQUENCE_INSERT_MIN_SEG_RATIO", 0.48)) * ref_len
                )
                & (
                    obs["center_s"].astype(float)
                    < b
                    - float(cfg.get("OBSERVED_SEQUENCE_INSERT_MIN_SEG_RATIO", 0.48)) * ref_len
                )
            ].copy()
            if len(local) == 0:
                continue
            before = _observed_sequence_len_cost(
                seg_len, ref_len, cfg, is_edge=si == 0 or si == len(cuts) - 2
            )
            for _, r in local.iterrows():
                c = float(r["center_s"])
                after = _observed_sequence_len_cost(
                    c - a, ref_len, cfg
                ) + _observed_sequence_len_cost(b - c, ref_len, cfg)
                support = _safe_float(
                    r.get("observed_proposal_observed_gap_support_rows", 0), 0.0
                )
                med_w = _safe_float(
                    r.get("observed_proposal_observed_gap_median_width", r.get("width", 0.0)),
                    0.0,
                )
                improve = (
                    before
                    - after
                    + 0.05 * support
                    + 0.05
                    * min(
                        2.0,
                        med_w
                        / max(
                            float(cfg.get("OBSERVED_SEQUENCE_OBS_GAP_MIN_WIDTH_ABS", 6500.0)),
                            1.0,
                        ),
                    )
                )
                rec = {
                    "stage": "observed_sequence_observed_gap_insert",
                    "decision": "candidate",
                    "segment_index": int(si),
                    "insert_center_s": c,
                    "seg_start": a,
                    "seg_end": b,
                    "seg_len": seg_len,
                    "left_len": c - a,
                    "right_len": b - c,
                    "cost_before": before,
                    "cost_after": after,
                    "improve": improve,
                    "ref_len": ref_len,
                    "obs_support_rows": support,
                    "obs_median_width": med_w,
                    "obs_score": _safe_float(r.get("score", np.nan), np.nan),
                }
                if best is None or improve > best[0]["improve"]:
                    best = (rec, r)
        if best is not None and best[0]["improve"] >= float(
            cfg.get("OBSERVED_SEQUENCE_INSERT_MIN_IMPROVE", 0.12)
        ):
            rec, r = best
            rec["decision"] = "accept_insert_observed_gap_boundary"
            logs.append(rec)
            cur = (
                pd.concat([cur, pd.DataFrame([r])], ignore_index=True, sort=False)
                .sort_values("center_s")
                .drop_duplicates(subset=["center_s"], keep="last")
                .reset_index(drop=True)
            )
            inserted += 1
        else:
            if best is not None:
                rec, _ = best
                rec["decision"] = "reject_best_insert_improve_too_small"
                logs.append(rec)
            break
    logs.append(
        {
            "stage": "observed_sequence_observed_gap_sequence_fix",
            "decision": "final_state",
            "n_selected_final": int(len(cur)),
            "n_observed_candidates": int(len(obs)),
            "inserted": int(inserted),
            "ref_len": float(ref_len),
        }
    )
    return (cur.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def _diagnostics_num(x, default=np.nan):
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _diagnostics_read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _diagnostics_nearest_selected_distance(center_s: float, selected: pd.DataFrame) -> float:
    if selected is None or len(selected) == 0 or "center_s" not in selected.columns:
        return float("inf")
    arr = selected["center_s"].to_numpy(dtype=float)
    if len(arr) == 0:
        return float("inf")
    return float(np.min(np.abs(arr - float(center_s))))


def _diagnostics_plot_length_stats(plot_df: pd.DataFrame) -> Dict[str, float]:
    if plot_df is None or len(plot_df) == 0 or "length" not in plot_df.columns:
        return {"median": np.nan, "q15": np.nan, "q85": np.nan, "min": np.nan, "max": np.nan}
    lens = plot_df["length"].to_numpy(dtype=float)
    lens = lens[np.isfinite(lens)]
    if len(lens) == 0:
        return {"median": np.nan, "q15": np.nan, "q85": np.nan, "min": np.nan, "max": np.nan}
    return {
        "median": float(np.nanmedian(lens)),
        "q15": float(np.nanquantile(lens, 0.15)),
        "q85": float(np.nanquantile(lens, 0.85)),
        "min": float(np.nanmin(lens)),
        "max": float(np.nanmax(lens)),
    }


def _diagnostics_boundary_label(row: pd.Series) -> Tuple[str, str]:
    cls = str(row.get("candidate_class", ""))
    kind = str(row.get("kind", ""))
    obs_rows = _diagnostics_num(
        row.get("observed_proposal_observed_gap_support_rows", np.nan), np.nan
    )
    obs_width = _diagnostics_num(
        row.get("observed_proposal_observed_gap_median_width", np.nan), np.nan
    )
    gap_width = _diagnostics_num(row.get("gap_width", np.nan), np.nan)
    joint_drop = _diagnostics_num(row.get("joint_drop", np.nan), np.nan)
    cont = _diagnostics_num(row.get("continuous_rows", np.nan), np.nan)
    if (
        np.isfinite(obs_rows)
        and obs_rows >= 2
        and np.isfinite(obs_width)
        and (obs_width >= 5000)
    ):
        return ("Wide gap", "Multirow observed-gap support; confirm against the image")
    if (
        np.isfinite(gap_width)
        and gap_width < 1600
        and (not (np.isfinite(obs_rows) and obs_rows >= 2))
    ):
        return ("Narrow gap", "Narrow fitted gap with limited multirow observed-gap support")
    if "observed" in kind.lower() or "observed" in cls.lower():
        return ("Observed gap candidate", "Candidate from a continuous low-density interval")
    if "weak" in cls.lower() or "row_gap" in cls.lower():
        return ("Weak boundary", "Weak candidate evidence; inspect the image")
    if np.isfinite(joint_drop) and joint_drop < 0.04 and np.isfinite(cont) and (cont <= 0):
        return ("Strong empty-gap candidate", "Low local density with few continuous rows")
    return ("Retained boundary", "Retained in the selected sequence")


def _diagnostics_plot_from_center(center_s: float, plot_df: pd.DataFrame) -> str:
    if plot_df is None or len(plot_df) == 0:
        return ""
    try:
        left = plot_df[plot_df["s_end"].astype(float) <= float(center_s)]
        right = plot_df[plot_df["s_start"].astype(float) >= float(center_s)]
        lp = int(left.iloc[-1]["plot_id"]) if len(left) else None
        rp = int(right.iloc[0]["plot_id"]) if len(right) else None
        if lp is not None and rp is not None:
            return f"P{lp}-P{rp}between"
        if lp is not None:
            return f"P{lp}after"
        if rp is not None:
            return f"P{rp}before"
    except Exception:
        pass
    return "unknown"


def _diagnostics_draw_local_boundary_tile(
    points: pd.DataFrame,
    selected: pd.DataFrame,
    center_s: float,
    out_path: Path,
    title: str,
    s_window: float,
    cfg: Dict,
    observed_centers: Optional[List[float]] = None,
):
    if points is None or len(points) == 0:
        return False
    if "s_rot" not in points.columns or "t_rot" not in points.columns:
        return False
    s0 = float(center_s) - float(s_window) * 0.5
    s1 = float(center_s) + float(s_window) * 0.5
    pts = points[
        (points["s_rot"].astype(float) >= s0) & (points["s_rot"].astype(float) <= s1)
    ].copy()
    if len(pts) == 0:
        return False
    t_lo = float(np.nanquantile(pts["t_rot"].astype(float), 0.005)) - 80.0
    t_hi = float(np.nanquantile(pts["t_rot"].astype(float), 0.995)) + 80.0
    s_lo, s_hi = (s0, s1)
    content_w = max(t_hi - t_lo, 1.0)
    content_h = max(s_hi - s_lo, 1.0)
    max_w = int(cfg.get("DIAGNOSTICS_TILE_MAX_WIDTH", 1050))
    max_h = int(cfg.get("DIAGNOSTICS_TILE_MAX_HEIGHT", 2300))
    header_h = 150
    margin = 45
    scale = min((max_w - 2 * margin) / content_w, (max_h - header_h - 2 * margin) / content_h)
    scale = max(scale, 0.01)
    W = max(500, int(round(content_w * scale + 2 * margin)))
    H = max(800, int(round(content_h * scale + 2 * margin + header_h)))
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (0, 0, 0)
    img[:header_h, :] = (30, 30, 30)

    def mp(t, s):
        x = int(round(margin + (float(t) - t_lo) * scale))
        y = int(round(header_h + margin + (float(s) - s_lo) * scale))
        return (x, y)

    cv2.putText(
        img,
        title[:95],
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "green=counted points; orange=boundary-gap points; red=selected center; cyan=near observed-gap proposal",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"s window: {s0:.0f} to {s1:.0f}",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    rad = max(1, int(cfg.get("POINT_RADIUS", 2)))
    for _, p in pts.iterrows():
        x, y = mp(p["t_rot"], p["s_rot"])
        gap = bool(p.get("is_boundary_gap_point", False))
        color = (0, 180, 255) if gap else (0, 255, 0)
        cv2.circle(img, (x, y), rad, color, -1, cv2.LINE_AA)
    if selected is not None and len(selected) and ("center_s" in selected.columns):
        for _, br in selected.iterrows():
            cs = float(br["center_s"])
            if s0 <= cs <= s1:
                _, y = mp(t_lo, cs)
                col = (0, 0, 255) if abs(cs - center_s) < 1e-06 else (120, 120, 120)
                cv2.line(
                    img,
                    (margin // 2, y),
                    (W - margin // 2, y),
                    col,
                    2 if abs(cs - center_s) < 1e-06 else 1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"B {cs:.0f}",
                    (W - 190, max(header_h + 20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    col,
                    1,
                    cv2.LINE_AA,
                )
    if observed_centers:
        for cs in observed_centers:
            if s0 <= float(cs) <= s1:
                _, y = mp(t_lo, float(cs))
                cv2.line(
                    img, (margin // 2, y), (W - margin // 2, y), (255, 255, 0), 1, cv2.LINE_AA
                )
                cv2.putText(
                    img,
                    f"OBS {cs:.0f}",
                    (20, max(header_h + 20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
    _write_image_unicode(str(out_path), img)
    return True


def write_boundary_diagnostics(
    out_dir: Path,
    rect_df: pd.DataFrame,
    work: pd.DataFrame,
    clustered: pd.DataFrame,
    selected: pd.DataFrame,
    plot_df: pd.DataFrame,
    plot_points: pd.DataFrame,
    mode_info: Dict,
    cfg: Dict,
):
    if not bool(cfg.get("ENABLE_DIAGNOSTICS_READABLE_DIAGNOSTICS", True)):
        return
    diag_dir = Path(out_dir) / "00_human_readable_diagnostics"
    tile_dir = diag_dir / "boundary_tiles"
    obs_dir = diag_dir / "unselected_observed_gap_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _diagnostics_read_csv_safe(Path(out_dir) / "10_partition_summary.csv")
    obs_prop = _diagnostics_read_csv_safe(
        Path(out_dir) / "03d_observed_proposal_observed_gap_interval_proposals.csv"
    )
    obs_seq_log = _diagnostics_read_csv_safe(
        Path(out_dir) / "04f9m_observed_sequence_observed_gap_sequence_fix_log.csv"
    )
    obs_diag_after = _diagnostics_read_csv_safe(
        Path(out_dir) / "04f9l_observed_guard_observed_gap_diagnosis_after.csv"
    )
    width_diag_after = _diagnostics_read_csv_safe(
        Path(out_dir) / "04f9h_gap_width_guard_gap_width_diagnosis_after.csv"
    )
    stats = _diagnostics_plot_length_stats(plot_df)
    ref_len = _diagnostics_num(mode_info.get("acf_peak_len", np.nan), np.nan)
    typical_len = _diagnostics_num(mode_info.get("typical_len", np.nan), np.nan)
    if not np.isfinite(ref_len):
        ref_len = typical_len
    rows = []
    if selected is not None and len(selected):
        sel_sorted = selected.sort_values("center_s").reset_index(drop=True)
        for i, r in sel_sorted.iterrows():
            label, reason = _diagnostics_boundary_label(r)
            center_s = float(r["center_s"])
            rows.append(
                {
                    "boundary_id": f"B{i + 1:02d}",
                    "location": _diagnostics_plot_from_center(center_s, plot_df),
                    "center_s": round(center_s, 1),
                    "review_priority": (
                        "high"
                        if label in ["Narrow gap", "Weak boundary", "Wide gap"]
                        else "medium"
                    ),
                    "review_label": label,
                    "reason": reason,
                    "candidate_type": str(r.get("candidate_class", "")),
                    "source": str(r.get("kind", "")),
                    "observed_support_rows": (
                        ""
                        if pd.isna(r.get("observed_proposal_observed_gap_support_rows", np.nan))
                        else int(
                            _diagnostics_num(
                                r.get("observed_proposal_observed_gap_support_rows"), 0
                            )
                        )
                    ),
                    "observed_gap_width": (
                        ""
                        if pd.isna(r.get("observed_proposal_observed_gap_median_width", np.nan))
                        else round(
                            _diagnostics_num(
                                r.get("observed_proposal_observed_gap_median_width"), 0
                            ),
                            1,
                        )
                    ),
                    "edge_gap_width": (
                        ""
                        if pd.isna(r.get("gap_width", np.nan))
                        else round(_diagnostics_num(r.get("gap_width"), 0), 1)
                    ),
                    "joint_drop": (
                        ""
                        if pd.isna(r.get("joint_drop", np.nan))
                        else round(_diagnostics_num(r.get("joint_drop"), 0), 4)
                    ),
                }
            )
    review_df = pd.DataFrame(rows)
    review_df.to_csv(
        diag_dir / "01_selected_boundary_review.csv", index=False, encoding="utf-8-sig"
    )
    obs_review_rows = []
    obs_centers_for_tiles: List[float] = []
    if obs_prop is not None and len(obs_prop) and ("decision" in obs_prop.columns):
        cand = obs_prop[
            obs_prop["decision"].astype(str).str.contains("candidate", case=False, na=False)
        ].copy()
        if len(cand):
            cand["nearest_selected_distance"] = (
                cand["center_s"]
                .astype(float)
                .apply(lambda x: _diagnostics_nearest_selected_distance(x, selected))
            )
            min_far = float(cfg.get("DIAGNOSTICS_UNSELECTED_OBS_MIN_DISTANCE", 3500.0))
            cand = cand.sort_values(
                ["nearest_selected_distance", "median_gap_width"], ascending=[False, False]
            )
            for _, r in cand.head(
                int(cfg.get("DIAGNOSTICS_MAX_UNSELECTED_OBS_REPORT", 25))
            ).iterrows():
                far = float(r.get("nearest_selected_distance", np.inf)) >= min_far
                obs_review_rows.append(
                    {
                        "center_s": round(float(r["center_s"]), 1),
                        "far_from_selected": "yes" if far else "no",
                        "nearest_selected_distance": round(
                            float(r.get("nearest_selected_distance", np.nan)), 1
                        ),
                        "support_rows": int(_diagnostics_num(r.get("support_rows", 0), 0)),
                        "weak_support_rows": int(
                            _diagnostics_num(r.get("weak_support_rows", 0), 0)
                        ),
                        "low_density_width": round(float(r.get("median_gap_width", np.nan)), 1),
                        "center_dispersion": round(float(r.get("center_spread", np.nan)), 1),
                        "interpretation": (
                            "Possible missed gap" if far else "Near a selected boundary"
                        ),
                    }
                )
                if far:
                    obs_centers_for_tiles.append(float(r["center_s"]))
    obs_review_df = pd.DataFrame(obs_review_rows)
    obs_review_df.to_csv(
        diag_dir / "02_unselected_observed_gaps.csv", index=False, encoding="utf-8-sig"
    )
    observed_centers = []
    if obs_prop is not None and len(obs_prop) and ("center_s" in obs_prop.columns):
        observed_centers = [
            float(x)
            for x in obs_prop.loc[
                obs_prop.get("decision", "")
                .astype(str)
                .str.contains("candidate", case=False, na=False),
                "center_s",
            ].tolist()
        ]
    s_win = float(
        cfg.get(
            "DIAGNOSTICS_BOUNDARY_TILE_S_WINDOW",
            max(65000.0, 1.25 * ref_len if np.isfinite(ref_len) else 65000.0),
        )
    )
    max_tiles = int(cfg.get("DIAGNOSTICS_MAX_BOUNDARY_TILES", 80))
    if selected is not None and len(selected):
        sel_sorted = selected.sort_values("center_s").reset_index(drop=True)
        made = 0
        for i, r in sel_sorted.iterrows():
            if made >= max_tiles:
                break
            label, reason = _diagnostics_boundary_label(r)
            important = label in [
                "Narrow gap",
                "Weak boundary",
                "Wide gap",
                "Observed gap candidate",
            ]
            if not important and (
                not bool(cfg.get("DIAGNOSTICS_EXPORT_ALL_BOUNDARY_TILES", False))
            ):
                continue
            center_s = float(r["center_s"])
            title = f"B{i + 1:02d} {label} center={center_s:.0f} class={r.get('candidate_class', '')}"
            outp = tile_dir / f"B{i + 1:02d}_{label}_s{center_s:.0f}.png"
            if _diagnostics_draw_local_boundary_tile(
                plot_points,
                selected,
                center_s,
                outp,
                title,
                s_win,
                cfg,
                observed_centers=observed_centers,
            ):
                made += 1
    for j, cs in enumerate(
        obs_centers_for_tiles[: int(cfg.get("DIAGNOSTICS_MAX_UNSELECTED_OBS_TILES", 20))],
        start=1,
    ):
        title = f"UNSELECTED_OBS_{j:02d} possible missed wide gap center={cs:.0f}"
        outp = obs_dir / f"OBS_{j:02d}_possible_missed_gap_s{cs:.0f}.png"
        _diagnostics_draw_local_boundary_tile(
            plot_points, selected, cs, outp, title, s_win, cfg, observed_centers=[cs]
        )
    n_bound = int(len(selected)) if selected is not None else 0
    n_plots = int(len(plot_df)) if plot_df is not None else 0
    n_obs_cand = (
        int(
            obs_prop["decision"]
            .astype(str)
            .str.contains("candidate", case=False, na=False)
            .sum()
        )
        if obs_prop is not None and len(obs_prop) and ("decision" in obs_prop.columns)
        else 0
    )
    n_obs_far = int(sum((1 for r in obs_review_rows if r.get("far_from_selected") == "yes")))
    n_narrow = int((review_df["review_label"] == "Narrow gap").sum()) if len(review_df) else 0
    n_weak = int((review_df["review_label"] == "Weak boundary").sum()) if len(review_df) else 0
    n_wide = int((review_df["review_label"] == "Wide gap").sum()) if len(review_df) else 0
    accepted_seq = 0
    if obs_seq_log is not None and len(obs_seq_log):
        for c in obs_seq_log.columns:
            if c.lower() == "decision":
                accepted_seq = int(
                    obs_seq_log[c]
                    .astype(str)
                    .str.contains("accept", case=False, na=False)
                    .sum()
                )
                break
    lines = []
    lines.append("Plot partition diagnostics")
    lines.append("=" * 42)
    lines.append("")
    lines.append("Summary")
    lines.append(f"- Selected boundaries: {n_bound}; proposed plots: {n_plots}。")
    lines.append(
        f"- Median plot extent: {stats['median']:.0f}; extent range: {stats['min']:.0f} ~ {stats['max']:.0f}。"
    )
    if np.isfinite(ref_len):
        lines.append(
            f"- Estimated repetition scale / ACF scale: {ref_len:.0f}; typical_len: {typical_len:.0f}。"
        )
    lines.append(
        f"- Observed low-density gap candidates: {n_obs_cand}; distant candidates to review: {n_obs_far}."
    )
    lines.append(
        f"- Selected boundaries: wide gaps: {n_wide}; narrow gaps to review: {n_narrow}; weak boundaries: {n_weak}."
    )
    lines.append(f"- Accepted observed-gap sequence updates: {accepted_seq}。")
    lines.append("")
    lines.append("Review files")
    lines.append(
        "1. Inspect selected boundaries in 00_human_readable_diagnostics/boundary_tiles/."
    )
    lines.append(
        "2. Inspect possible missed gaps in 00_human_readable_diagnostics/unselected_observed_gap_tiles/."
    )
    lines.append("3. Check the labels and measurements in 01_selected_boundary_review.csv.")
    lines.append(
        "4. Check 02_unselected_observed_gaps.csv for gaps omitted by boundary selection."
    )
    lines.append("")
    lines.append("Review labels")
    lines.append(
        "- Wide gap: supported by several rows; inspect the image before confirming the boundary."
    )
    lines.append(
        "- Narrow gap: limited fitted width and little support from observed gaps; inspect for a false split."
    )
    lines.append(
        "- Weak boundary: local missing plants or sparse detections may explain the gap."
    )
    lines.append(
        "- Unselected observed gap: a wide sparse interval without a nearby selected boundary; inspect for a missed split."
    )
    lines.append("")
    if len(review_df):
        lines.append("Selected boundaries for priority review")
        pr = review_df[review_df["review_priority"].astype(str) == "high"].head(30)
        if len(pr) == 0:
            lines.append("- No selected boundary is marked for priority review.")
        else:
            for _, r in pr.iterrows():
                lines.append(
                    f"- {r['boundary_id']} | {r['location']} | {r['review_label']} | center_s={r['center_s']} | {r['reason']}"
                )
    lines.append("")
    if len(obs_review_df):
        lines.append("Possible missed gaps")
        pr = obs_review_df[obs_review_df["far_from_selected"].astype(str) == "yes"].head(30)
        if len(pr) == 0:
            lines.append("- No distant observed gaps require priority review.")
        else:
            for _, r in pr.iterrows():
                lines.append(
                    f"- center_s={r['center_s']} | supporting rows={r['support_rows']} | width={r['low_density_width']} | distance to nearest selected boundary={r['nearest_selected_distance']} | {r['interpretation']}"
                )
    lines.append("")
    lines.append("Interpreting discrepancies")
    lines.append(
        "- For a narrow retained gap, check observed-gap support to assess a possible false split."
    )
    lines.append(
        "- A true gap in unselected_observed_gap_tiles indicates that candidate generation found it but boundary selection omitted it."
    )
    lines.append(
        "- If a true gap is absent from the candidate report, inspect observed-interval generation."
    )
    (diag_dir / "00_boundary_diagnostics.txt").write_text("\n".join(lines), encoding="utf-8")
    try:
        html = [
            "<html><head><meta charset='utf-8'><title>Boundary diagnostics</title></head><body style='font-family:Arial, sans-serif;background:#111;color:#eee;'>"
        ]
        html.append("<h2>Plot boundary diagnostics</h2>")
        html.append("<p>See 00_boundary_diagnostics.txt for the review summary.</p>")
        html.append("<h3>Selected boundaries for review</h3>")
        for p in sorted(tile_dir.glob("*.png"))[:200]:
            rel = p.relative_to(diag_dir).as_posix()
            html.append(
                f"<div style='margin:20px;border-top:1px solid #555;'><p>{p.name}</p><img src='{rel}' style='max-width:100%;'></div>"
            )
        html.append("<h3>Possible missed gaps</h3>")
        for p in sorted(obs_dir.glob("*.png"))[:200]:
            rel = p.relative_to(diag_dir).as_posix()
            html.append(
                f"<div style='margin:20px;border-top:1px solid #555;'><p>{p.name}</p><img src='{rel}' style='max-width:100%;'></div>"
            )
        html.append("</body></html>")
        (diag_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    except Exception:
        pass


def _periodic_sequence_get_acf_reference_len(
    work: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Use the density-autocorrelation scale unless spacing evidence supports an alternative."""
    checked_ref = cfg.get("_STRICT_ROWGAP_CHECKED_REFERENCE_LEN")
    if checked_ref is not None and np.isfinite(float(checked_ref)) and (float(checked_ref) > 1):
        return (float(checked_ref), dict(cfg["_STRICT_ROWGAP_CHECKED_REFERENCE_META"]))
    meta: Dict[str, Any] = {"stage": "periodic_sequence_reference_length"}
    acf_len = np.nan
    try:
        acf_len, acf_meta = estimate_typical_length_from_density_acf(work, cfg)
        if isinstance(acf_meta, dict):
            meta.update(
                {
                    f"acf_{k}": v
                    for k, v in acf_meta.items()
                    if isinstance(v, (int, float, str)) or v is None
                }
            )
    except Exception as e:
        meta["acf_error"] = repr(e)
    min_len = float(cfg.get("PERIODIC_SEQUENCE_REF_MIN_LEN", 18000.0))
    max_len = float(cfg.get("PERIODIC_SEQUENCE_REF_MAX_LEN", 120000.0))
    if np.isfinite(acf_len) and min_len <= float(acf_len) <= max_len:
        meta.update(
            {
                "ref_len": float(acf_len),
                "ref_source": "density_acf_primary",
                "harmonic_half_used": 0,
            }
        )
        return (float(acf_len), meta)
    try:
        dummy = pd.DataFrame()
        base_len, base_meta = robust_typical_length_from_candidates(dummy, work, cfg)
        if np.isfinite(base_len) and min_len <= float(base_len) <= max_len:
            meta.update(
                {
                    "ref_len": float(base_len),
                    "ref_source": "robust_fallback_no_harmonic",
                    "harmonic_half_used": 0,
                }
            )
            return (float(base_len), meta)
    except Exception as e:
        meta["fallback_error"] = repr(e)
    s_min, s_max = (float(work["s_rot"].min()), float(work["s_rot"].max()))
    span = max(s_max - s_min, 1.0)
    fallback = float(np.clip(span / max(round(span / 50000.0), 1), min_len, max_len))
    meta.update(
        {"ref_len": fallback, "ref_source": "span_rough_fallback", "harmonic_half_used": 0}
    )
    return (fallback, meta)


def _periodic_sequence_class_rank(row: pd.Series) -> int:
    if "candidate_class_rank" in row.index:
        try:
            return int(row.get("candidate_class_rank"))
        except Exception:
            pass
    cls = str(row.get("candidate_class", ""))
    if cls == "A_observed_gap":
        return 5
    if cls.startswith("A"):
        return 4
    if cls.startswith("B"):
        return 3
    if cls.startswith("C"):
        return 2
    return 1


def _periodic_sequence_candidate_reward(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> float:
    score = _safe_float(row.get("score", 0.0), 0.0)
    cls = str(row.get("candidate_class", ""))
    kind = str(row.get("kind", ""))
    reward = float(score) * 0.72
    rank = _periodic_sequence_class_rank(row)
    if cls == "A_observed_gap" or kind == "observed_gap_interval":
        reward += float(cfg.get("PERIODIC_SEQUENCE_OBSERVED_GAP_BONUS", 5.0))
        support = _safe_int(
            row.get(
                "observed_proposal_observed_gap_support_rows",
                row.get("row_gap_support_row_count", 0),
            ),
            0,
        )
        med_w = _safe_float(
            row.get("observed_proposal_observed_gap_median_width", row.get("width", 0.0)), 0.0
        )
        reward += 0.75 * max(0, support - 1)
        if np.isfinite(ref_len) and ref_len > 1:
            reward += min(2.2, max(0.0, med_w / max(ref_len, 1.0) * 4.0))
    elif rank >= 4:
        reward += float(cfg.get("PERIODIC_SEQUENCE_A_STRONG_BONUS", 2.6))
    elif rank == 3:
        reward += float(cfg.get("PERIODIC_SEQUENCE_B_BONUS", 1.1))
    else:
        reward -= float(cfg.get("PERIODIC_SEQUENCE_WEAK_PENALTY", 1.2))
    if "row_gap" in kind or "rowgap" in cls.lower():
        reward -= float(cfg.get("PERIODIC_SEQUENCE_ROWGAP_PENALTY", 2.0))
    med_w = _safe_float(
        row.get("observed_proposal_observed_gap_median_width", row.get("width", 0.0)), 0.0
    )
    if (
        med_w > 0
        and np.isfinite(ref_len)
        and (ref_len > 1)
        and (med_w < 0.055 * ref_len)
        and (cls != "A_observed_gap")
    ):
        reward -= float(cfg.get("PERIODIC_SEQUENCE_NARROW_NO_OBS_PENALTY", 3.2))
    cont = _safe_int(row.get("continuous_rows", 0), 0)
    reward -= 0.9 * max(0, cont)
    return float(reward)


def _periodic_sequence_segment_penalty(
    length: float, ref_len: float, cfg: Dict[str, Any], is_edge: bool = False
) -> float:
    if not np.isfinite(ref_len) or ref_len <= 1:
        return 0.0
    length = float(length)
    if length <= 0:
        return 1000000.0
    ratio = length / float(ref_len)
    edge_factor = (
        float(cfg.get("PERIODIC_SEQUENCE_EDGE_SEGMENT_COST_FACTOR", 0.35)) if is_edge else 1.0
    )
    edge_min = float(cfg.get("PERIODIC_SEQUENCE_EDGE_MIN_RATIO", 0.18))
    hard_short = float(cfg.get("PERIODIC_SEQUENCE_SHORT_RATIO_HARD", 0.52))
    if is_edge:
        if ratio < edge_min:
            return 120.0 * (edge_min - ratio + 0.05)
    elif ratio < hard_short:
        return 100000.0 + 1000.0 * (hard_short - ratio)
    good_lo = float(cfg.get("PERIODIC_SEQUENCE_GOOD_RATIO_LOW", 0.78))
    good_hi = float(cfg.get("PERIODIC_SEQUENCE_GOOD_RATIO_HIGH", 1.38))
    long_soft = float(cfg.get("PERIODIC_SEQUENCE_LONG_RATIO_SOFT", 1.62))
    w = float(cfg.get("PERIODIC_SEQUENCE_SEGMENT_COST_WEIGHT", 9.0)) * edge_factor
    if good_lo <= ratio <= good_hi:
        return 0.1 * w * (ratio - 1.0) ** 2
    if ratio < good_lo:
        return w * ((good_lo - ratio) / max(good_lo - hard_short, 1e-06)) ** 2 * 2.5
    if ratio <= long_soft:
        return w * ((ratio - good_hi) / max(long_soft - good_hi, 1e-06)) ** 2 * 1.1
    return w * (1.1 + (ratio - long_soft) ** 2 * 2.4)


def _periodic_sequence_prepare_candidate_pool(
    cands: pd.DataFrame, ref_len: float, cfg: Dict[str, Any]
) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return pd.DataFrame()
    cc = cands.copy()
    cc = cc[np.isfinite(pd.to_numeric(cc["center_s"], errors="coerce"))].copy()
    if len(cc) == 0:
        return cc
    cc["center_s"] = cc["center_s"].astype(float)
    cc["periodic_sequence_reward"] = cc.apply(
        lambda r: _periodic_sequence_candidate_reward(r, ref_len, cfg), axis=1
    )
    rank = cc.apply(_periodic_sequence_class_rank, axis=1)
    keep = (
        (rank >= 3)
        | (cc["periodic_sequence_reward"] >= float(cfg.get("DP_SCORE_BASE", 5.8)))
        | (cc.get("kind", pd.Series("", index=cc.index)).astype(str) == "observed_gap_interval")
    )
    cc = cc[keep].copy()
    max_keep = int(cfg.get("PERIODIC_SEQUENCE_MAX_CANDIDATES", 650))
    if len(cc) > max_keep:
        cc = cc.sort_values("periodic_sequence_reward", ascending=False).head(max_keep).copy()
    min_dup = max(800.0, 0.025 * float(ref_len) if np.isfinite(ref_len) else 1200.0)
    cc = cc.sort_values(
        ["center_s", "periodic_sequence_reward"], ascending=[True, False]
    ).reset_index(drop=True)
    kept = []
    for _, r in cc.iterrows():
        c = float(r["center_s"])
        if not kept or abs(c - float(kept[-1]["center_s"])) > min_dup:
            kept.append(r)
        elif float(r["periodic_sequence_reward"]) > float(
            kept[-1].get("periodic_sequence_reward", -1000000000.0)
        ):
            kept[-1] = r
    out = (
        pd.DataFrame(kept).sort_values("center_s").reset_index(drop=True)
        if kept
        else pd.DataFrame(columns=cc.columns)
    )
    return out


def select_periodic_boundary_sequence(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Select a sequence using local gap evidence and segment-scale consistency."""
    ref_len, ref_meta = _periodic_sequence_get_acf_reference_len(work, cfg)
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    pool = _periodic_sequence_prepare_candidate_pool(cands, ref_len, cfg)
    logs: List[Dict[str, Any]] = []
    mode_info: Dict[str, Any] = {
        "mode": "periodic_sequence_sequence_dp",
        "typical_len": float(ref_len),
        "periodic_sequence_ref_len": float(ref_len),
        **ref_meta,
    }
    if len(pool) == 0:
        mode_info["periodic_sequence_reason"] = "no_candidate_pool"
        return (
            pd.DataFrame(columns=cands.columns if cands is not None else []),
            pd.DataFrame([{"decision": "no_candidates", **mode_info}]),
            mode_info,
        )
    centers = pool["center_s"].astype(float).to_numpy()
    rewards = pool["periodic_sequence_reward"].astype(float).to_numpy()
    n = len(pool)
    min_spacing = float(cfg.get("PERIODIC_SEQUENCE_MIN_CENTER_SPACING_RATIO", 0.56)) * float(
        ref_len
    )
    boundary_penalty = float(cfg.get("PERIODIC_SEQUENCE_BOUNDARY_PENALTY", 5.8))
    dp = np.full(n, -1e18, dtype=float)
    prev = np.full(n, -1, dtype=int)
    for i in range(n):
        left_len = centers[i] - s_start
        start_score = (
            rewards[i]
            - boundary_penalty
            - _periodic_sequence_segment_penalty(left_len, ref_len, cfg, is_edge=True)
        )
        dp[i] = start_score
        prev[i] = -1
        for j in range(i):
            seg_len = centers[i] - centers[j]
            if seg_len < min_spacing:
                continue
            cand_score = (
                dp[j]
                + rewards[i]
                - boundary_penalty
                - _periodic_sequence_segment_penalty(seg_len, ref_len, cfg, is_edge=False)
            )
            if cand_score > dp[i]:
                dp[i] = cand_score
                prev[i] = j
    no_boundary_score = -_periodic_sequence_segment_penalty(
        s_end - s_start, ref_len, cfg, is_edge=True
    )
    best_score = no_boundary_score
    best_i = -1
    for i in range(n):
        end_len = s_end - centers[i]
        sc = dp[i] - _periodic_sequence_segment_penalty(end_len, ref_len, cfg, is_edge=True)
        if sc > best_score:
            best_score = sc
            best_i = i
    selected_idx: List[int] = []
    cur = int(best_i)
    while cur >= 0:
        selected_idx.append(cur)
        cur = int(prev[cur])
    selected_idx = selected_idx[::-1]
    selected = (
        pool.iloc[selected_idx].copy().reset_index(drop=True)
        if selected_idx
        else pool.iloc[[]].copy()
    )
    selected["selected_stage"] = "periodic_sequence_sequence_dp"
    selected["periodic_sequence_ref_len"] = float(ref_len)
    selected["periodic_sequence_reward"] = selected.get("periodic_sequence_reward", np.nan)
    selected_centers = (
        set(np.round(selected["center_s"].astype(float).to_numpy(), 6))
        if len(selected)
        else set()
    )
    cuts = (
        [s_start]
        + (selected["center_s"].astype(float).tolist() if len(selected) else [])
        + [s_end]
    )
    seg_lengths = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
    for i, r in pool.iterrows():
        c = float(r["center_s"])
        logs.append(
            {
                "center_s": c,
                "candidate_class": r.get("candidate_class", ""),
                "kind": r.get("kind", ""),
                "score": _safe_float(r.get("score", np.nan), np.nan),
                "periodic_sequence_reward": _safe_float(
                    r.get("periodic_sequence_reward", np.nan), np.nan
                ),
                "selected": int(round(c, 6) in selected_centers),
                "ref_len": float(ref_len),
                "min_spacing": float(min_spacing),
            }
        )
    summary = {
        "decision": "periodic_sequence_sequence_dp_done",
        "n_pool_candidates": int(len(pool)),
        "n_selected": int(len(selected)),
        "best_score": float(best_score),
        "no_boundary_score": float(no_boundary_score),
        "ref_len": float(ref_len),
        "segment_lengths": ";".join((f"{x:.1f}" for x in seg_lengths)),
        "segment_len_median": float(np.median(seg_lengths)) if seg_lengths else np.nan,
        **{k: v for k, v in ref_meta.items() if isinstance(v, (int, float, str)) or v is None},
    }
    trace = pd.concat(
        [pd.DataFrame([summary]), pd.DataFrame(logs)], ignore_index=True, sort=False
    )
    return (selected, trace, mode_info)


def recover_observed_gaps_in_long_segments(
    work: pd.DataFrame,
    clustered: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recover observed gaps inside long segments subject to sequence constraints."""
    logs: List[Dict[str, Any]] = []
    if selected is None:
        selected = pd.DataFrame()
    current = (
        selected.copy().sort_values("center_s").reset_index(drop=True)
        if len(selected)
        else selected.copy()
    )
    if clustered is None or len(clustered) == 0:
        return (
            current,
            pd.DataFrame(
                [
                    {
                        "stage": "periodic_sequence_observed_gap_recall",
                        "decision": "no_candidates",
                    }
                ]
            ),
        )
    ref_len = _safe_float(
        mode_info.get("periodic_sequence_ref_len", mode_info.get("typical_len", np.nan)), np.nan
    )
    if not np.isfinite(ref_len) or ref_len <= 1:
        return (
            current,
            pd.DataFrame(
                [{"stage": "periodic_sequence_observed_gap_recall", "decision": "no_ref_len"}]
            ),
        )
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    max_iter = int(cfg.get("PERIODIC_SEQUENCE_OBS_RECALL_MAX_ITER", 20))
    long_ratio = float(cfg.get("PERIODIC_SEQUENCE_OBS_RECALL_LONG_RATIO", 1.55))
    min_child = float(cfg.get("PERIODIC_SEQUENCE_OBS_RECALL_MIN_CHILD_RATIO", 0.55)) * ref_len
    obs = clustered.copy()
    kind = obs.get("kind", pd.Series("", index=obs.index)).astype(str)
    cls = obs.get("candidate_class", pd.Series("", index=obs.index)).astype(str)
    obs = obs[(kind == "observed_gap_interval") | (cls == "A_observed_gap")].copy()
    if len(obs) == 0:
        return (
            current,
            pd.DataFrame(
                [
                    {
                        "stage": "periodic_sequence_observed_gap_recall",
                        "decision": "no_observed_gap_candidates",
                    }
                ]
            ),
        )
    obs["center_s"] = obs["center_s"].astype(float)
    obs["periodic_sequence_reward"] = obs.apply(
        lambda r: _periodic_sequence_candidate_reward(r, ref_len, cfg), axis=1
    )
    for it in range(max_iter):
        centers = current["center_s"].astype(float).tolist() if len(current) else []
        cuts = [s_start] + centers + [s_end]
        inserted = False
        best = None
        for si in range(len(cuts) - 1):
            a, b = (float(cuts[si]), float(cuts[si + 1]))
            seg_len = b - a
            if seg_len < long_ratio * ref_len:
                continue
            inside = obs[
                (obs["center_s"] > a + min_child) & (obs["center_s"] < b - min_child)
            ].copy()
            if len(inside) == 0:
                logs.append(
                    {
                        "iter": it,
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": seg_len,
                        "decision": "long_segment_no_observed_gap",
                    }
                )
                continue

            def rank_row(r: pd.Series) -> float:
                c = float(r["center_s"])
                l1, l2 = (c - a, b - c)
                before = _periodic_sequence_segment_penalty(
                    seg_len, ref_len, cfg, is_edge=si == 0 or si == len(cuts) - 2
                )
                after = _periodic_sequence_segment_penalty(
                    l1, ref_len, cfg, is_edge=si == 0
                ) + _periodic_sequence_segment_penalty(
                    l2, ref_len, cfg, is_edge=si == len(cuts) - 2
                )
                return (
                    before
                    - after
                    + float(r.get("periodic_sequence_reward", 0.0))
                    - float(cfg.get("PERIODIC_SEQUENCE_BOUNDARY_PENALTY", 5.8))
                )

            inside["periodic_sequence_insert_gain"] = inside.apply(rank_row, axis=1)
            rr = inside.sort_values("periodic_sequence_insert_gain", ascending=False).iloc[0]
            if best is None or float(rr["periodic_sequence_insert_gain"]) > float(
                best["periodic_sequence_insert_gain"]
            ):
                best = rr
        if best is not None and float(best["periodic_sequence_insert_gain"]) > 0.0:
            new = best.copy()
            new["selected_stage"] = "periodic_sequence_observed_gap_recall_in_long_segment"
            current = (
                pd.concat([current, pd.DataFrame([new])], ignore_index=True, sort=False)
                .sort_values("center_s")
                .reset_index(drop=True)
            )
            logs.append(
                {
                    "iter": it,
                    "center_s": float(new["center_s"]),
                    "gain": float(best["periodic_sequence_insert_gain"]),
                    "decision": "accept_insert_observed_gap_in_long_segment",
                }
            )
            inserted = True
        if not inserted:
            logs.append({"iter": it, "decision": "stop_no_positive_insert"})
            break
    return (current, pd.DataFrame(logs))


def periodic_sequence_write_sequence_summary(
    out_dir: Path,
    work: pd.DataFrame,
    selected: pd.DataFrame,
    mode_info: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    centers = (
        selected["center_s"].astype(float).tolist()
        if selected is not None and len(selected)
        else []
    )
    cuts = [s_start] + centers + [s_end]
    rows = []
    ref_len = _safe_float(
        mode_info.get("periodic_sequence_ref_len", mode_info.get("typical_len", np.nan)), np.nan
    )
    for i in range(len(cuts) - 1):
        length = cuts[i + 1] - cuts[i]
        rows.append(
            {
                "segment_id": i + 1,
                "s_start": cuts[i],
                "s_end": cuts[i + 1],
                "length": length,
                "length_ratio_to_periodic_sequence_ref": (
                    length / ref_len if np.isfinite(ref_len) and ref_len > 1 else np.nan
                ),
                "is_edge_segment": int(i == 0 or i == len(cuts) - 2),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "04_periodic_sequence_sequence_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _feature_gap_aux_reference_len(
    work: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """Feature gap aux reference len."""
    try:
        ref, meta = _periodic_sequence_get_acf_reference_len(work, cfg)
        if np.isfinite(ref) and ref > 1:
            meta = dict(meta)
            meta["feature_gap_aux_len_source"] = "density_acf_auxiliary"
            return (float(ref), meta)
    except Exception as e:
        meta = {"feature_gap_aux_len_error": repr(e)}
    try:
        s = pd.to_numeric(work["s_rot"], errors="coerce").dropna().to_numpy(dtype=float)
        span = float(np.nanmax(s) - np.nanmin(s)) if len(s) else np.nan
        fallback = (
            span / max(4.0, min(24.0, math.sqrt(max(len(s), 1)) / 2.5))
            if np.isfinite(span)
            else np.nan
        )
        return (float(fallback), {"feature_gap_aux_len_source": "span_fallback", **meta})
    except Exception:
        return (np.nan, {"feature_gap_aux_len_source": "none"})


def _feature_gap_gap_min_width(ref_len: float, cfg: Dict[str, Any]) -> float:
    if np.isfinite(ref_len) and ref_len > 1:
        raw = max(
            float(cfg.get("FEATURE_GAP_GAP_MIN_WIDTH_ABS", 3800.0)),
            float(cfg.get("FEATURE_GAP_GAP_MIN_WIDTH_RATIO", 0.075)) * float(ref_len),
        )
        return float(min(raw, float(cfg.get("FEATURE_GAP_GAP_MIN_WIDTH_MAX", 9000.0))))
    return float(cfg.get("FEATURE_GAP_GAP_MIN_WIDTH_ABS", 3800.0))


def _feature_gap_max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask.astype(bool).tolist():
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _start_block_row_setup(work: pd.DataFrame, cfg: Dict[str, Any]):
    row_col = (
        "row_main"
        if "row_main" in work.columns
        else "row_id" if "row_id" in work.columns else None
    )
    n_rows = int(cfg.get("MAIN_ROWS", 3))
    if row_col is None:
        tmp = work.copy()
        tmp["_start_block_row"] = 0
        return (tmp, "_start_block_row", [0])
    vals = pd.to_numeric(work[row_col], errors="coerce").dropna().astype(int).unique().tolist()
    row_ids = sorted(vals)[:n_rows]
    if not row_ids:
        tmp = work.copy()
        tmp["_start_block_row"] = 0
        return (tmp, "_start_block_row", [0])
    return (work, row_col, row_ids)


def _start_block_interval_stats(
    counts_by_row: np.ndarray,
    edges: np.ndarray,
    row_dense_ref: np.ndarray,
    start_i: int,
    end_i: int,
    prepost_bins: int,
    start_bins: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    nb = counts_by_row.shape[1]
    idx = np.arange(start_i, end_i, dtype=int)
    if len(idx) == 0:
        return {}
    total_counts = counts_by_row.sum(axis=0)
    active_rows = (counts_by_row > 0).sum(axis=0).astype(int)
    gap_counts = counts_by_row[:, idx]
    ar = active_rows[idx]
    tc = total_counts[idx]
    gap_start = float(edges[start_i])
    gap_end = float(edges[end_i])
    gap_width = gap_end - gap_start
    gap_density_per_bin = float(tc.sum() / max(1, len(idx)))
    pre0 = max(0, start_i - prepost_bins)
    pre1 = start_i
    post0 = end_i
    post1 = min(nb, end_i + prepost_bins)
    start0 = end_i
    start1 = min(nb, end_i + start_bins)
    pre_counts = (
        counts_by_row[:, pre0:pre1] if pre1 > pre0 else np.zeros((counts_by_row.shape[0], 0))
    )
    post_counts = (
        counts_by_row[:, post0:post1]
        if post1 > post0
        else np.zeros((counts_by_row.shape[0], 0))
    )
    start_counts = (
        counts_by_row[:, start0:start1]
        if start1 > start0
        else np.zeros((counts_by_row.shape[0], 0))
    )

    def dense_rows_for(mat: np.ndarray, frac: float, min_count: float) -> int:
        if mat.size == 0:
            return 0
        row_sum = mat.sum(axis=1)
        thr = np.maximum(
            min_count, frac * np.maximum(row_dense_ref, 1.0) * max(1, mat.shape[1])
        )
        return int(np.sum(row_sum >= thr))

    side_frac = float(cfg.get("START_BLOCK_SIDE_DENSE_FRAC_OF_ROW_REF", 0.25))
    side_min = float(cfg.get("START_BLOCK_SIDE_MIN_COUNT_PER_ROW", 2))
    pre_dense_rows = dense_rows_for(pre_counts, side_frac, side_min)
    post_dense_rows = dense_rows_for(post_counts, side_frac, side_min)
    start_frac = float(cfg.get("START_BLOCK_START_DENSE_FRAC_OF_ROW_REF", 0.44))
    start_min = float(cfg.get("START_BLOCK_START_MIN_COUNT_PER_ROW", 3))
    start_dense_rows = dense_rows_for(start_counts, start_frac, start_min)
    start_total = float(start_counts.sum()) if start_counts.size else 0.0
    start_density_per_bin = (
        start_total / max(1, start_counts.shape[1]) if start_counts.size else 0.0
    )
    start_active = (
        (start_counts > 0).sum(axis=0).astype(int)
        if start_counts.size
        else np.zeros(0, dtype=int)
    )
    pre_density = (
        float(pre_counts.sum() / max(1, pre_counts.shape[1])) if pre_counts.size else 0.0
    )
    post_density = (
        float(post_counts.sum() / max(1, post_counts.shape[1])) if post_counts.size else 0.0
    )
    side_density = max(pre_density, post_density, 1e-06)
    gap_density_ratio = float(gap_density_per_bin / side_density)
    start_contrast = float((start_density_per_bin + 1.0) / (gap_density_per_bin + 1.0))
    active_le1_frac = float(np.mean(ar <= 1)) if len(ar) else 0.0
    active_ge2_frac = float(np.mean(ar >= 2)) if len(ar) else 0.0
    active3_frac = float(np.mean(ar >= 3)) if len(ar) else 0.0
    max_consec_ge2 = _feature_gap_max_consecutive(ar >= 2)
    max_consec_active3 = _feature_gap_max_consecutive(ar >= 3)
    start_active_ge2_frac = float(np.mean(start_active >= 2)) if len(start_active) else 0.0
    start_active3_frac = float(np.mean(start_active >= 3)) if len(start_active) else 0.0
    return {
        "gap_start": gap_start,
        "gap_end": gap_end,
        "center": 0.5 * (gap_start + gap_end),
        "gap_width": gap_width,
        "gap_bin_count": int(len(idx)),
        "gap_point_count": int(tc.sum()),
        "gap_density_per_bin": gap_density_per_bin,
        "pre_density_per_bin": pre_density,
        "post_density_per_bin": post_density,
        "gap_density_ratio": gap_density_ratio,
        "active_le1_frac": active_le1_frac,
        "active_ge2_frac": active_ge2_frac,
        "active3_frac": active3_frac,
        "max_consecutive_active_ge2_bins": int(max_consec_ge2),
        "max_consecutive_active3_bins": int(max_consec_active3),
        "pre_dense_rows": int(pre_dense_rows),
        "post_dense_rows": int(post_dense_rows),
        "start_dense_rows": int(start_dense_rows),
        "start_density_per_bin": start_density_per_bin,
        "start_contrast": start_contrast,
        "start_active_ge2_frac": start_active_ge2_frac,
        "start_active3_frac": start_active3_frac,
    }


def _start_block_prepare_pool(
    cands: pd.DataFrame, aux_len: float, cfg: Dict[str, Any]
) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return pd.DataFrame()
    cc = cands.copy()
    cc = cc[
        np.isfinite(
            pd.to_numeric(
                cc.get("center_s", pd.Series(np.nan, index=cc.index)), errors="coerce"
            )
        )
    ].copy()
    if len(cc) == 0:
        return cc
    cc["center_s"] = pd.to_numeric(cc["center_s"], errors="coerce").astype(float)
    cc["feature_gap_reward"] = cc.apply(
        lambda r: _feature_gap_candidate_reward(r, aux_len, cfg), axis=1
    )
    cc = cc[
        cc["feature_gap_reward"] >= float(cfg.get("START_BLOCK_SELECTION_MIN_SCORE", 13.2))
    ].copy()
    if len(cc) == 0:
        return cc
    min_w = _feature_gap_gap_min_width(aux_len, cfg)
    try:
        gw = (
            pd.to_numeric(
                cc.get("start_block_gap_width", cc.get("feature_gap_gap_width")),
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(gw):
            min_w = max(min_w, float(np.nanmedian(gw)))
    except Exception:
        pass
    nms = max(
        float(cfg.get("START_BLOCK_NMS_ABS", 9000.0)),
        float(cfg.get("START_BLOCK_NMS_RATIO_OF_GAP", 1.5)) * max(1.0, min_w),
    )
    cc = cc.sort_values("feature_gap_reward", ascending=False).reset_index(drop=True)
    keep = []
    for i, r in cc.iterrows():
        c = float(r["center_s"])
        if all((abs(c - float(cc.loc[j, "center_s"])) > nms for j in keep)):
            keep.append(int(i))
    return cc.loc[keep].copy().sort_values("center_s").reset_index(drop=True)


def _start_block_short_segment_recheck(
    selected: pd.DataFrame, work: pd.DataFrame, aux_len: float, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (
        selected is None
        or len(selected) <= 1
        or (not bool(cfg.get("START_BLOCK_ENABLE_SHORT_SEGMENT_RECHECK", True)))
    ):
        return (
            selected,
            pd.DataFrame([{"stage": "start_block_short_segment_recheck", "decision": "skip"}]),
        )
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    cur = selected.copy().sort_values("center_s").reset_index(drop=True)
    logs = []
    short_thr = float(cfg.get("START_BLOCK_SHORT_SEGMENT_ABS", 9500.0))
    if np.isfinite(aux_len) and aux_len > 1:
        short_thr = max(
            short_thr,
            float(cfg.get("START_BLOCK_SHORT_SEGMENT_RATIO_OF_AUX", 0.2)) * float(aux_len),
        )
    protect = float(cfg.get("START_BLOCK_PROTECT_SCORE_FROM_SHORT_MERGE", 18.5))
    changed = True
    while changed and len(cur) > 1:
        changed = False
        centers = cur["center_s"].astype(float).tolist()
        cuts = [s_start] + centers + [s_end]
        segs = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
        internal = [(i, segs[i]) for i in range(1, len(segs) - 1) if segs[i] < short_thr]
        if not internal:
            break
        seg_i, seg_len = sorted(internal, key=lambda x: x[1])[0]
        left_b = seg_i - 1
        right_b = seg_i
        left_score = _safe_float(
            cur.loc[left_b].get(
                "feature_gap_reward", cur.loc[left_b].get("start_block_boundary_likeness", 0.0)
            ),
            0.0,
        )
        right_score = _safe_float(
            cur.loc[right_b].get(
                "feature_gap_reward", cur.loc[right_b].get("start_block_boundary_likeness", 0.0)
            ),
            0.0,
        )
        if left_score >= protect and right_score >= protect:
            logs.append(
                {
                    "stage": "start_block_short_segment_recheck",
                    "decision": "keep_short_segment_both_boundaries_protected",
                    "segment_index": seg_i,
                    "segment_length": seg_len,
                    "left_score": left_score,
                    "right_score": right_score,
                }
            )
            break
        del_idx = left_b if left_score <= right_score else right_b
        logs.append(
            {
                "stage": "start_block_short_segment_recheck",
                "decision": "remove_weaker_boundary_creating_extreme_short_segment",
                "segment_index": seg_i,
                "segment_length": seg_len,
                "removed_center_s": float(cur.loc[del_idx, "center_s"]),
                "left_score": left_score,
                "right_score": right_score,
                "short_threshold": short_thr,
            }
        )
        cur = cur.drop(index=del_idx).reset_index(drop=True)
        changed = True
    if not logs:
        logs.append(
            {
                "stage": "start_block_short_segment_recheck",
                "decision": "no_extreme_short_internal_segment",
                "short_threshold": short_thr,
            }
        )
    return (cur, pd.DataFrame(logs))


def write_feature_gap_segment_summary(
    out_dir: Path, work: pd.DataFrame, selected: pd.DataFrame, mode_info: Dict[str, Any]
) -> None:
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    centers = (
        selected["center_s"].astype(float).tolist()
        if selected is not None and len(selected)
        else []
    )
    cuts = [s_start] + centers + [s_end]
    rows = []
    aux_len = (
        float(mode_info.get("feature_gap_aux_len", np.nan))
        if isinstance(mode_info, dict)
        else np.nan
    )
    for i in range(len(cuts) - 1):
        length = float(cuts[i + 1] - cuts[i])
        rows.append(
            {
                "plot_index": i + 1,
                "s_start": cuts[i],
                "s_end": cuts[i + 1],
                "length": length,
                "length_ratio_to_aux": (
                    length / aux_len if np.isfinite(aux_len) and aux_len > 1 else np.nan
                ),
                "mode": (
                    mode_info.get("mode", "start_block_feature_gap_structure_primary")
                    if isinstance(mode_info, dict)
                    else "start_block_feature_gap_structure_primary"
                ),
                "aux_len": aux_len,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        out_dir / "04_feature_gap_feature_gap_segments.csv", index=False, encoding="utf-8-sig"
    )
    df.to_csv(
        out_dir / "04_start_block_feature_gap_segments.csv", index=False, encoding="utf-8-sig"
    )


def _density_valley_close_holes(
    base_mask: np.ndarray, hole_mask: np.ndarray, max_hole: int
) -> np.ndarray:
    """Close short interruptions in a low-density interval."""
    out = np.asarray(base_mask, dtype=bool).copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and (not out[j]):
            j += 1
        if j - i <= int(max_hole) and np.all(hole_mask[i:j]):
            out[i:j] = True
        i = j
    return out


def _density_interval_stats_base(
    counts_by_row: np.ndarray,
    edges: np.ndarray,
    row_dense_ref: np.ndarray,
    start_i: int,
    end_i: int,
    prepost_bins: int,
    start_bins: int,
    low_thr: float,
    relaxed_thr: float,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Density interval stats base."""
    base = _start_block_interval_stats(
        counts_by_row, edges, row_dense_ref, start_i, end_i, prepost_bins, start_bins, cfg
    )
    if not base:
        return {}
    total_counts = counts_by_row.sum(axis=0)
    active_rows = (counts_by_row > 0).sum(axis=0).astype(int)
    idx = np.arange(start_i, end_i, dtype=int)
    tc = total_counts[idx]
    ar = active_rows[idx]
    high_multirow = (ar >= 2) & (tc > relaxed_thr)
    high_three = (ar >= 3) & (tc > relaxed_thr)
    base.update(
        {
            "density_valley_low_thr": float(low_thr),
            "density_valley_relaxed_thr": float(relaxed_thr),
            "density_valley_core_low_frac": float(np.mean(tc <= low_thr)) if len(tc) else 0.0,
            "density_valley_relaxed_low_frac": (
                float(np.mean(tc <= relaxed_thr)) if len(tc) else 0.0
            ),
            "density_valley_high_multirow_frac": (
                float(np.mean(high_multirow)) if len(tc) else 0.0
            ),
            "density_valley_high_three_frac": float(np.mean(high_three)) if len(tc) else 0.0,
            "density_valley_max_consecutive_high_multirow_bins": (
                int(_feature_gap_max_consecutive(high_multirow)) if len(tc) else 0
            ),
            "density_valley_mean_total_count": float(np.mean(tc)) if len(tc) else 0.0,
            "density_valley_median_total_count": float(np.median(tc)) if len(tc) else 0.0,
        }
    )
    return base


def _density_score_base(
    stats: Dict[str, Any], min_w: float, cfg: Dict[str, Any]
) -> Tuple[float, str, str, bool]:
    """Density score base."""
    w = float(stats.get("gap_width", 0.0))
    core_low = float(stats.get("density_valley_core_low_frac", 0.0))
    relaxed_low = float(stats.get("density_valley_relaxed_low_frac", 0.0))
    high_multi = float(stats.get("density_valley_high_multirow_frac", 1.0))
    high_multi_consec = int(stats.get("density_valley_max_consecutive_high_multirow_bins", 999))
    dens_ratio = float(stats.get("gap_density_ratio", 999.0))
    le1 = float(stats.get("active_le1_frac", 0.0))
    a3 = float(stats.get("active3_frac", 0.0))
    start_rows = int(stats.get("start_dense_rows", 0))
    start_contrast = float(stats.get("start_contrast", 0.0))
    start_ge2 = float(stats.get("start_active_ge2_frac", 0.0))
    pre_rows = int(stats.get("pre_dense_rows", 0))
    post_rows = int(stats.get("post_dense_rows", 0))
    width_score = min(4.5, max(0.0, (w / max(min_w, 1.0) - 0.75) * 3.0))
    valley_score = 6.5 * relaxed_low + 4.0 * core_low + 2.8 * max(0.0, 1.0 - dens_ratio)
    through_penalty = (
        4.2 * high_multi + 0.9 * max(0, high_multi_consec - 2) + 2.8 * max(0.0, a3 - 0.18)
    )
    start_score = 1.7 * min(3, start_rows) + 1.2 * min(2.8, start_contrast) + 2.0 * start_ge2
    side_score = 0.6 * min(2, pre_rows) + 0.9 * min(2, post_rows)
    score = float(width_score + valley_score + start_score + side_score - through_penalty)
    if w < min_w:
        return (score, "D_low", "reject_gap_width_too_small", False)
    if relaxed_low < float(cfg.get("DENSITY_VALLEY_MIN_RELAXED_LOW_FRAC", 0.55)):
        return (score, "D_low", "reject_not_continuous_low_density_interval", False)
    if core_low < float(cfg.get("DENSITY_VALLEY_MIN_CORE_LOW_FRAC", 0.25)):
        return (score, "D_low", "reject_lacks_enough_core_low_density_bins", False)
    if dens_ratio > float(cfg.get("DENSITY_VALLEY_GAP_DENSITY_RATIO_MAX", 0.82)):
        return (score, "D_low", "reject_gap_density_not_low_vs_sides", False)
    if high_multi > float(cfg.get("DENSITY_VALLEY_MAX_HIGH_MULTIROW_FRAC", 0.42)):
        return (score, "D_low", "reject_too_many_high_density_multirow_bins", False)
    if high_multi_consec > int(cfg.get("DENSITY_VALLEY_MAX_HIGH_MULTIROW_CONSEC", 4)):
        return (score, "D_low", "reject_multirow_high_density_through_gap", False)
    strong = score >= 13.0 and (start_rows >= 2 or w >= 1.45 * min_w or core_low >= 0.45)
    weak = score >= 8.5
    if strong:
        return (
            score,
            "A_feature_gap",
            "candidate_density_valley_soft_density_gap_strong",
            True,
        )
    if weak:
        return (score, "B_feature_gap", "candidate_density_valley_soft_density_gap_weak", True)
    return (score, "D_low", "reject_density_valley_score_too_low", False)


def build_density_valley_proposals(
    work: pd.DataFrame, classified_cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build density valley proposals."""
    if not bool(cfg.get("ENABLE_DENSITY_VALLEY_SOFT_DENSITY_GAP_CORE", True)):
        return (
            classified_cands,
            pd.DataFrame([{"stage": "density_valley_feature_gap", "decision": "disabled"}]),
        )
    cands = classified_cands.copy() if classified_cands is not None else pd.DataFrame()
    if work is None or len(work) == 0:
        return (
            cands,
            pd.DataFrame(
                [{"stage": "density_valley_feature_gap", "decision": "skip_empty_work"}]
            ),
        )
    ref_len, ref_meta = _feature_gap_aux_reference_len(work, cfg)
    s = pd.to_numeric(work["s_rot"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(s) < 10:
        return (
            cands,
            pd.DataFrame(
                [{"stage": "density_valley_feature_gap", "decision": "skip_too_few_points"}]
            ),
        )
    smin, smax = (float(np.nanmin(s)), float(np.nanmax(s)))
    min_w = max(
        float(cfg.get("DENSITY_VALLEY_GAP_MIN_WIDTH_ABS", 3800.0)),
        (
            float(cfg.get("DENSITY_VALLEY_GAP_MIN_WIDTH_RATIO", 0.06)) * float(ref_len)
            if np.isfinite(ref_len)
            else 0.0
        ),
    )
    min_w = min(min_w, float(cfg.get("DENSITY_VALLEY_GAP_MIN_WIDTH_MAX", 7600.0)))
    bin_w = float(cfg.get("DENSITY_VALLEY_BIN_ABS", 800.0))
    if np.isfinite(ref_len) and ref_len > 1:
        bin_w = min(
            max(bin_w, float(cfg.get("DENSITY_VALLEY_BIN_RATIO", 0.012)) * float(ref_len)),
            float(cfg.get("DENSITY_VALLEY_BIN_MAX", 1300.0)),
        )
    bin_w = max(300.0, bin_w)
    edge_guard = max(3000.0, 0.65 * min_w)
    work2, row_col, row_ids = _start_block_row_setup(work, cfg)
    edges = np.arange(smin, smax + bin_w * 1.5, bin_w, dtype=float)
    nb = len(edges) - 1
    if nb < 5:
        return (
            cands,
            pd.DataFrame(
                [
                    {
                        "stage": "density_valley_feature_gap",
                        "decision": "too_few_bins",
                        "bin_w": bin_w,
                    }
                ]
            ),
        )
    counts_by_row = np.zeros((len(row_ids), nb), dtype=float)
    row_dense_ref = []
    for ri, rid in enumerate(row_ids):
        sub = work2[pd.to_numeric(work2[row_col], errors="coerce").astype("Int64") == int(rid)]
        ss = (
            pd.to_numeric(sub.get("s_rot", pd.Series(dtype=float)), errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        hist, _ = np.histogram(ss, bins=edges)
        counts_by_row[ri, :] = hist.astype(float)
        pos = hist[hist > 0]
        row_dense_ref.append(
            float(np.quantile(pos, float(cfg.get("START_BLOCK_ROW_DENSE_REF_Q", 0.72))))
            if len(pos)
            else 1.0
        )
    row_dense_ref = np.asarray(row_dense_ref, dtype=float)
    total_counts = counts_by_row.sum(axis=0)
    active_rows = (counts_by_row > 0).sum(axis=0).astype(int)
    low_thr = float(
        np.quantile(total_counts, float(cfg.get("DENSITY_VALLEY_LOW_DENSITY_Q", 0.3)))
    )
    relaxed_thr = float(
        np.quantile(total_counts, float(cfg.get("DENSITY_VALLEY_RELAXED_LOW_DENSITY_Q", 0.42)))
    )
    if relaxed_thr < low_thr:
        relaxed_thr = low_thr
    core_low = total_counts <= low_thr
    relaxed_hole = (total_counts <= relaxed_thr) | (active_rows <= 1) & (
        total_counts <= relaxed_thr + 2
    )
    low = _density_valley_close_holes(
        core_low, relaxed_hole, int(cfg.get("DENSITY_VALLEY_ALLOW_HOLE_BINS", 2))
    )
    side_w = max(float(cfg.get("START_BLOCK_SIDE_WINDOW_ABS", 8500.0)), min_w * 1.15)
    if np.isfinite(ref_len) and ref_len > 1:
        side_w = max(
            side_w, float(cfg.get("START_BLOCK_SIDE_WINDOW_RATIO", 0.15)) * float(ref_len)
        )
    side_w = min(side_w, float(cfg.get("START_BLOCK_SIDE_WINDOW_MAX", 22000.0)))
    prepost_bins = max(1, int(round(side_w / bin_w)))
    start_w = float(cfg.get("START_BLOCK_START_BLOCK_WINDOW_ABS", 5600.0))
    if np.isfinite(ref_len) and ref_len > 1:
        start_w = max(
            start_w,
            float(cfg.get("START_BLOCK_START_BLOCK_WINDOW_RATIO", 0.085)) * float(ref_len),
        )
    start_w = min(start_w, float(cfg.get("START_BLOCK_START_BLOCK_WINDOW_MAX", 12000.0)))
    start_bins = max(1, int(round(start_w / bin_w)))
    logs: List[Dict[str, Any]] = []
    proposal_rows: List[Dict[str, Any]] = []
    start_i = None
    for bi, is_low in enumerate(low):
        if is_low and start_i is None:
            start_i = bi
        at_end = bi == len(low) - 1
        if (not is_low or at_end) and start_i is not None:
            end_i = bi if not is_low else bi + 1
            center = 0.5 * (float(edges[start_i]) + float(edges[end_i]))
            if center <= smin + edge_guard or center >= smax - edge_guard:
                start_i = None
                continue
            stats = _density_interval_stats(
                counts_by_row,
                edges,
                row_dense_ref,
                start_i,
                end_i,
                prepost_bins,
                start_bins,
                low_thr,
                relaxed_thr,
                cfg,
            )
            if not stats:
                start_i = None
                continue
            score, cls, decision, accept = _density_score_with_continuity(stats, min_w, cfg)
            log = {
                "stage": "density_valley_soft_density_gap_interval",
                "decision": decision,
                "candidate_class": cls,
                "center_s": stats.get("center"),
                "gap_start_s": stats.get("gap_start"),
                "gap_end_s": stats.get("gap_end"),
                "gap_width": stats.get("gap_width"),
                "density_valley_boundary_likeness": score,
                "min_gap_width_required": min_w,
                "bin_w": bin_w,
                "low_thr": low_thr,
                "relaxed_thr": relaxed_thr,
                **{
                    k: v
                    for k, v in stats.items()
                    if k not in {"gap_start", "gap_end", "center"}
                },
                **{
                    k: v
                    for k, v in ref_meta.items()
                    if isinstance(v, (int, float, str)) or v is None
                },
            }
            logs.append(log)
            if accept:
                row = {
                    col: np.nan for col in (list(cands.columns) if len(cands.columns) else [])
                }
                row.update(
                    {
                        "s0": stats["gap_start"],
                        "s1": stats["gap_end"],
                        "center_s": stats["center"],
                        "width": stats["gap_width"],
                        "score": score,
                        "kind": "feature_gap_interval",
                        "candidate_class": cls,
                        "selected_stage": "density_valley_soft_density_gap_interval_proposal",
                        "feature_gap_gap_start_s": stats["gap_start"],
                        "feature_gap_gap_end_s": stats["gap_end"],
                        "feature_gap_gap_width": stats["gap_width"],
                        "feature_gap_boundary_likeness": score,
                        "feature_gap_active_le1_frac": stats["active_le1_frac"],
                        "feature_gap_active_ge2_frac": stats["active_ge2_frac"],
                        "feature_gap_active3_frac": stats["active3_frac"],
                        "feature_gap_max_consecutive_active_ge2_bins": stats[
                            "max_consecutive_active_ge2_bins"
                        ],
                        "feature_gap_gap_density_ratio": stats["gap_density_ratio"],
                        "feature_gap_pre_dense_rows": stats["pre_dense_rows"],
                        "feature_gap_post_dense_rows": stats["post_dense_rows"],
                        "feature_calibration_post_start_dense_rows": stats["start_dense_rows"],
                        "feature_calibration_post_start_contrast": stats["start_contrast"],
                        "feature_calibration_post_start_active_ge2_frac": stats[
                            "start_active_ge2_frac"
                        ],
                        "feature_calibration_post_start_active3_frac": stats[
                            "start_active3_frac"
                        ],
                        "start_block_gap_start_s": stats["gap_start"],
                        "start_block_gap_end_s": stats["gap_end"],
                        "start_block_gap_width": stats["gap_width"],
                        "start_block_boundary_likeness": score,
                        "start_block_decision": decision,
                        "start_block_active_le1_frac": stats["active_le1_frac"],
                        "start_block_active_ge2_frac": stats["active_ge2_frac"],
                        "start_block_active3_frac": stats["active3_frac"],
                        "start_block_max_consecutive_active_ge2_bins": stats[
                            "max_consecutive_active_ge2_bins"
                        ],
                        "start_block_gap_density_ratio": stats["gap_density_ratio"],
                        "start_block_pre_dense_rows": stats["pre_dense_rows"],
                        "start_block_post_dense_rows": stats["post_dense_rows"],
                        "start_block_start_dense_rows": stats["start_dense_rows"],
                        "start_block_start_contrast": stats["start_contrast"],
                        "start_block_start_active_ge2_frac": stats["start_active_ge2_frac"],
                        "start_block_start_active3_frac": stats["start_active3_frac"],
                        "density_valley_boundary_likeness": score,
                        "density_valley_decision": decision,
                        "density_valley_core_low_frac": stats["density_valley_core_low_frac"],
                        "density_valley_relaxed_low_frac": stats[
                            "density_valley_relaxed_low_frac"
                        ],
                        "density_valley_high_multirow_frac": stats[
                            "density_valley_high_multirow_frac"
                        ],
                        "density_valley_max_consecutive_high_multirow_bins": stats[
                            "density_valley_max_consecutive_high_multirow_bins"
                        ],
                        "density_valley_mean_total_count": stats[
                            "density_valley_mean_total_count"
                        ],
                        "density_valley_median_total_count": stats[
                            "density_valley_median_total_count"
                        ],
                        "density_valley_low_thr": low_thr,
                        "density_valley_relaxed_thr": relaxed_thr,
                        "feature_gap_ref_len_aux": ref_len,
                    }
                )
                proposal_rows.append(row)
            start_i = None
    props = pd.DataFrame(proposal_rows)
    if len(props) > 0:
        merge_tol = max(
            float(cfg.get("START_BLOCK_CENTER_MERGE_ABS", 4200.0)),
            float(cfg.get("START_BLOCK_CENTER_MERGE_RATIO_OF_GAP", 0.65)) * min_w,
        )
        props = props.sort_values(
            "density_valley_boundary_likeness", ascending=False
        ).reset_index(drop=True)
        keep = []
        for i, r in props.iterrows():
            c = float(r["center_s"])
            if all((abs(c - float(props.loc[j, "center_s"])) > merge_tol for j in keep)):
                keep.append(int(i))
        props = props.loc[keep].copy().sort_values("center_s").reset_index(drop=True)
    aux_parts = []
    if bool(cfg.get("START_BLOCK_USE_OBSERVED_GAP_AUX", True)) and len(cands):
        kind = cands.get("kind", pd.Series("", index=cands.index)).astype(str)
        obs = cands[kind.eq("observed_gap_interval")].copy()
        if len(obs):
            aux_parts.append(obs)
    aux_parts.append(props)
    out = (
        pd.concat(
            [x for x in aux_parts if x is not None and len(x)], ignore_index=True, sort=False
        )
        if any((len(x) for x in aux_parts if x is not None))
        else props
    )
    logs.append(
        {
            "stage": "density_valley_soft_density_gap_interval",
            "decision": "final_state",
            "n_accepted_feature_gap": int(len(props)),
            "n_observed_aux": int(sum((1 for _ in []))),
            "n_candidates_before": int(len(cands)),
            "n_candidates_after": int(len(out)),
            "ref_len_aux": float(ref_len) if np.isfinite(ref_len) else np.nan,
            "min_gap_width_required": float(min_w),
            "bin_w": float(bin_w),
            "low_thr": low_thr,
            "relaxed_thr": relaxed_thr,
        }
    )
    return (out.reset_index(drop=True), pd.DataFrame(logs))


def select_density_valley_sequence(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    aux_len, aux_meta = _feature_gap_aux_reference_len(work, cfg)
    pool = _start_block_prepare_pool(cands, aux_len, cfg)
    mode_info = {
        "mode": "density_valley_soft_density_gap_feature_primary",
        "typical_len": float(aux_len) if np.isfinite(aux_len) else np.nan,
        "feature_gap_aux_len": float(aux_len) if np.isfinite(aux_len) else np.nan,
        **aux_meta,
    }
    if len(pool) == 0:
        return (
            pd.DataFrame(columns=cands.columns if cands is not None else []),
            pd.DataFrame(
                [{"decision": "no_density_valley_soft_density_gap_pool", **mode_info}]
            ),
            mode_info,
        )
    selected = pool.copy().sort_values("center_s").reset_index(drop=True)
    selected["selected_stage"] = "density_valley_soft_density_gap_feature_nms"
    selected["feature_gap_aux_len"] = float(aux_len) if np.isfinite(aux_len) else np.nan
    selected, short_log = _start_block_short_segment_recheck(selected, work, aux_len, cfg)
    selected["selected_stage"] = "density_valley_soft_density_gap_feature_nms_rechecked"
    s_start = float(work["s_rot"].min())
    s_end = float(work["s_rot"].max())
    centers = selected["center_s"].astype(float).tolist() if len(selected) else []
    cuts = [s_start] + centers + [s_end]
    seg_lengths = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
    logs = [
        {
            "decision": "density_valley_soft_density_gap_feature_done",
            "n_pool_candidates": int(len(pool)),
            "n_selected": int(len(selected)),
            "aux_len": float(aux_len) if np.isfinite(aux_len) else np.nan,
            "segment_lengths": ";".join((f"{x:.1f}" for x in seg_lengths)),
            "segment_len_median": float(np.median(seg_lengths)) if seg_lengths else np.nan,
            **{
                k: v
                for k, v in aux_meta.items()
                if isinstance(v, (int, float, str)) or v is None
            },
        }
    ]
    for _, r in pool.iterrows():
        logs.append(
            {
                "center_s": float(r["center_s"]),
                "candidate_class": r.get("candidate_class", ""),
                "kind": r.get("kind", ""),
                "feature_gap_reward": _safe_float(r.get("feature_gap_reward", np.nan), np.nan),
                "density_valley_boundary_likeness": _safe_float(
                    r.get("density_valley_boundary_likeness", np.nan), np.nan
                ),
                "density_valley_decision": r.get(
                    "density_valley_decision", r.get("start_block_decision", "")
                ),
                "start_block_gap_width": _safe_float(
                    r.get("start_block_gap_width", r.get("feature_gap_gap_width", np.nan)),
                    np.nan,
                ),
                "density_valley_core_low_frac": _safe_float(
                    r.get("density_valley_core_low_frac", np.nan), np.nan
                ),
                "density_valley_relaxed_low_frac": _safe_float(
                    r.get("density_valley_relaxed_low_frac", np.nan), np.nan
                ),
                "density_valley_high_multirow_frac": _safe_float(
                    r.get("density_valley_high_multirow_frac", np.nan), np.nan
                ),
                "start_block_start_dense_rows": _safe_int(
                    r.get(
                        "start_block_start_dense_rows",
                        r.get("feature_calibration_post_start_dense_rows", 0),
                    ),
                    0,
                ),
                "selected_by_nms": int(
                    any((abs(float(r["center_s"]) - cc) < 1e-06 for cc in centers))
                ),
            }
        )
    logs.extend(
        short_log.to_dict("records") if short_log is not None and len(short_log) else []
    )
    return (selected.reset_index(drop=True), pd.DataFrame(logs), mode_info)


def _continuity_max_consecutive_bool(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    best = cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _density_interval_stats(
    counts_by_row: np.ndarray,
    edges: np.ndarray,
    row_dense_ref: np.ndarray,
    start_i: int,
    end_i: int,
    prepost_bins: int,
    start_bins: int,
    low_thr: float,
    relaxed_thr: float,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Density interval stats."""
    base = _density_interval_stats_base(
        counts_by_row,
        edges,
        row_dense_ref,
        start_i,
        end_i,
        prepost_bins,
        start_bins,
        low_thr,
        relaxed_thr,
        cfg,
    )
    if not base:
        return base
    idx = np.arange(start_i, end_i, dtype=int)
    gap_bins = max(1, len(idx))
    nb = counts_by_row.shape[1]
    n_rows = counts_by_row.shape[0]
    gap_counts = counts_by_row[:, idx] if len(idx) else np.zeros((n_rows, 0), dtype=float)
    pre0 = max(0, start_i - prepost_bins)
    pre1 = start_i
    post0 = end_i
    post1 = min(nb, end_i + prepost_bins)
    pre_counts = (
        counts_by_row[:, pre0:pre1] if pre1 > pre0 else np.zeros((n_rows, 0), dtype=float)
    )
    post_counts = (
        counts_by_row[:, post0:post1] if post1 > post0 else np.zeros((n_rows, 0), dtype=float)
    )
    row_active_frac = []
    row_high_frac = []
    row_max_consec_active = []
    row_max_consec_active_frac = []
    row_gap_density = []
    row_side_density = []
    row_density_ratio = []
    row_pre_has = []
    row_post_has = []
    through_flags = []
    active_frac_thr = float(cfg.get("CONTINUITY_ROW_ACTIVE_FRAC_THROUGH", 0.43))
    consec_frac_thr = float(cfg.get("CONTINUITY_ROW_CONSEC_FRAC_THROUGH", 0.36))
    density_ratio_thr = float(cfg.get("CONTINUITY_ROW_DENSITY_RATIO_THROUGH", 0.48))
    min_gap_points = float(cfg.get("CONTINUITY_ROW_MIN_GAP_POINTS_THROUGH", 5))
    require_prepost = bool(cfg.get("CONTINUITY_REQUIRE_PRE_POST_FOR_THROUGH", True))
    prepost_min = float(cfg.get("CONTINUITY_PREPOST_ROW_MIN_POINTS", 3))
    for r in range(n_rows):
        g = gap_counts[r, :] if gap_counts.size else np.zeros(0, dtype=float)
        active = g > 0
        high = g > max(1.0, relaxed_thr / max(1, n_rows))
        af = float(np.mean(active)) if len(active) else 0.0
        hf = float(np.mean(high)) if len(high) else 0.0
        mc = _continuity_max_consecutive_bool(active)
        mcf = float(mc / max(1, gap_bins))
        gd = float(np.sum(g) / max(1, gap_bins)) if len(g) else 0.0
        pre_sum = float(np.sum(pre_counts[r, :])) if pre_counts.size else 0.0
        post_sum = float(np.sum(post_counts[r, :])) if post_counts.size else 0.0
        pre_bins = max(1, pre_counts.shape[1]) if pre_counts.size else 1
        post_bins = max(1, post_counts.shape[1]) if post_counts.size else 1
        sd = max(pre_sum / pre_bins, post_sum / post_bins, 1e-06)
        ratio = float(gd / sd)
        pre_has = bool(pre_sum >= prepost_min)
        post_has = bool(post_sum >= prepost_min)
        continuity_core = (
            af >= active_frac_thr
            or mcf >= consec_frac_thr
            or (ratio >= density_ratio_thr and np.sum(g) >= min_gap_points)
        )
        if require_prepost:
            through = bool(continuity_core and pre_has and post_has)
        else:
            through = bool(continuity_core)
        row_active_frac.append(af)
        row_high_frac.append(hf)
        row_max_consec_active.append(int(mc))
        row_max_consec_active_frac.append(mcf)
        row_gap_density.append(gd)
        row_side_density.append(sd)
        row_density_ratio.append(ratio)
        row_pre_has.append(int(pre_has))
        row_post_has.append(int(post_has))
        through_flags.append(int(through))
    through_rows = int(np.sum(through_flags))
    max_consec_frac = (
        float(np.max(row_max_consec_active_frac)) if row_max_consec_active_frac else 0.0
    )
    mean_active_frac = float(np.mean(row_active_frac)) if row_active_frac else 0.0
    max_active_frac = float(np.max(row_active_frac)) if row_active_frac else 0.0
    mean_density_ratio = float(np.mean(row_density_ratio)) if row_density_ratio else 0.0
    max_density_ratio = float(np.max(row_density_ratio)) if row_density_ratio else 0.0
    base.update(
        {
            "continuity_through_rows": through_rows,
            "continuity_through_flags": ";".join((str(x) for x in through_flags)),
            "continuity_row_active_frac": ";".join((f"{x:.3f}" for x in row_active_frac)),
            "continuity_row_density_ratio": ";".join((f"{x:.3f}" for x in row_density_ratio)),
            "continuity_row_max_consec_active_bins": ";".join(
                (str(x) for x in row_max_consec_active)
            ),
            "continuity_max_row_active_frac": max_active_frac,
            "continuity_mean_row_active_frac": mean_active_frac,
            "continuity_max_consec_active_frac": max_consec_frac,
            "continuity_mean_row_density_ratio": mean_density_ratio,
            "continuity_max_row_density_ratio": max_density_ratio,
            "continuity_pre_has_rows": int(np.sum(row_pre_has)),
            "continuity_post_has_rows": int(np.sum(row_post_has)),
        }
    )
    return base


def _continuity_has_clear_start(stats: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    return bool(
        int(stats.get("start_dense_rows", 0)) >= int(cfg.get("CONTINUITY_CLEAR_START_ROWS", 2))
        and float(stats.get("start_contrast", 0.0))
        >= float(cfg.get("CONTINUITY_CLEAR_START_CONTRAST", 1.28))
        and (
            float(stats.get("start_active_ge2_frac", 0.0))
            >= float(cfg.get("CONTINUITY_CLEAR_START_GE2_FRAC", 0.14))
        )
    )


def _density_score_with_continuity(
    stats: Dict[str, Any], min_w: float, cfg: Dict[str, Any]
) -> Tuple[float, str, str, bool]:
    """Density score with continuity."""
    score, cls, decision, accept = _density_score_base(stats, min_w, cfg)
    if not bool(cfg.get("ENABLE_CONTINUITY_CONTINUITY_VETO", True)):
        return (score, cls, decision, accept)
    through_rows = int(stats.get("continuity_through_rows", 0))
    max_consec_frac = float(stats.get("continuity_max_consec_active_frac", 0.0))
    high_multi = float(stats.get("density_valley_high_multirow_frac", 0.0))
    dens_ratio = float(stats.get("gap_density_ratio", 999.0))
    relaxed_low = float(stats.get("density_valley_relaxed_low_frac", 0.0))
    core_low = float(stats.get("density_valley_core_low_frac", 0.0))
    has_clear_start = _continuity_has_clear_start(stats, cfg)
    score -= float(cfg.get("CONTINUITY_THROUGH_ROW_PENALTY", 2.9)) * max(0, through_rows)
    score -= float(cfg.get("CONTINUITY_THROUGH_CONSEC_PENALTY", 7.0)) * max(
        0.0, max_consec_frac - 0.3
    )
    if through_rows >= int(
        cfg.get("CONTINUITY_REJECT_THROUGH_ROWS_HARD", 3)
    ) and max_consec_frac >= float(cfg.get("CONTINUITY_REJECT_MAX_CONSEC_FRAC", 0.5)):
        if not (has_clear_start and relaxed_low >= 0.68 and (core_low >= 0.33)):
            return (
                score,
                "D_low",
                "reject_continuity_three_rows_continuously_through_gap",
                False,
            )
    if through_rows >= int(cfg.get("CONTINUITY_REJECT_THROUGH_ROWS_SOFT", 2)):
        bad_continuity = (
            max_consec_frac >= float(cfg.get("CONTINUITY_SOFT_REJECT_MAX_CONSEC_FRAC", 0.38))
            or high_multi >= float(cfg.get("CONTINUITY_SOFT_REJECT_MIN_HIGH_MULTI_FRAC", 0.26))
            or dens_ratio >= float(cfg.get("CONTINUITY_SOFT_REJECT_MIN_DENS_RATIO", 0.58))
        )
        if bad_continuity and (not has_clear_start):
            return (
                score,
                "D_low",
                "reject_continuity_within_plot_continuity_no_start_restart",
                False,
            )
    if accept and score < float(cfg.get("CONTINUITY_SELECTION_MIN_SCORE_AFTER_VETO", 10.2)):
        return (score, "D_low", "reject_continuity_score_low_after_continuity_veto", False)
    if accept:
        cls = "A_feature_gap" if score >= 13.2 else "B_feature_gap"
        return (score, cls, "candidate_continuity_gap_passed_continuity_veto", True)
    return (score, cls, decision, accept)


def _feature_gap_candidate_reward(row: pd.Series, aux_len: float, cfg: Dict[str, Any]) -> float:
    """Feature gap candidate reward."""
    kind = str(row.get("kind", ""))
    if kind == "feature_gap_interval":
        decision = str(row.get("density_valley_decision", row.get("start_block_decision", "")))
        if decision.startswith("reject_continuity"):
            return float(-1000000.0)
        return float(
            _safe_float(
                row.get(
                    "density_valley_boundary_likeness",
                    row.get(
                        "start_block_boundary_likeness",
                        row.get("feature_gap_boundary_likeness", row.get("score", 0.0)),
                    ),
                ),
                0.0,
            )
        )
    if (
        kind == "observed_gap_interval"
        or str(row.get("candidate_class", "")) == "A_observed_gap"
    ):
        support = _safe_float(
            row.get(
                "observed_proposal_observed_gap_support_rows", row.get("support_rows", 0.0)
            ),
            0.0,
        )
        med_w = _safe_float(
            row.get(
                "observed_proposal_observed_gap_median_width",
                row.get("observed_proposal_observed_gap_width", row.get("width", 0.0)),
            ),
            0.0,
        )
        if support < float(cfg.get("CONTINUITY_OBS_MIN_SUPPORT_ROWS", 2)) or med_w < float(
            cfg.get("CONTINUITY_OBS_MIN_MEDIAN_WIDTH", 4200.0)
        ):
            return float(-1000000.0)
        return float(
            float(cfg.get("CONTINUITY_OBS_REWARD_BASE", 7.0))
            + 1.35 * support
            + min(3.0, med_w / 3800.0)
        )
    return float(-1000000.0)


def select_sequence_with_continuity_check(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selected, logs, mode_info = select_density_valley_sequence(work, cands, cfg)
    if selected is not None and len(selected):
        selected = selected.copy()
        selected["selected_stage"] = "continuity_feature_gap_continuity_veto_nms_rechecked"
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info["mode"] = "continuity_feature_gap_with_continuity_veto"
    if logs is None:
        logs = pd.DataFrame()
    extra = pd.DataFrame(
        [
            {
                "decision": "continuity_continuity_veto_selection_wrapper_done",
                "n_selected_after_continuity": (
                    int(len(selected)) if selected is not None else 0
                ),
                "mode": mode_info["mode"],
            }
        ]
    )
    logs = pd.concat([extra, logs], ignore_index=True, sort=False)
    return (selected, logs, mode_info)


def _structural_recall_float(
    row: pd.Series, names: List[str], default: float = np.nan
) -> float:
    for n in names:
        if n in row.index:
            v = _safe_float(row.get(n), np.nan)
            if np.isfinite(v):
                return float(v)
    return float(default)


def _structural_recall_str(row: pd.Series, names: List[str], default: str = "") -> str:
    for n in names:
        if n in row.index:
            v = row.get(n)
            if pd.notna(v):
                return str(v)
    return default


def _structural_recall_aux_len(
    work: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    try:
        aux_len, meta = _feature_gap_aux_reference_len(work, cfg)
        if np.isfinite(aux_len) and aux_len > 1:
            return (float(aux_len), dict(meta))
    except Exception:
        pass
    sspan = float(work["s_rot"].max() - work["s_rot"].min()) if len(work) else np.nan
    return (
        float(max(sspan / 20.0, 1.0)),
        {"structural_recall_aux_len_source": "fallback_span_div20"},
    )


def _structural_recall_candidate_quality(row: pd.Series, cfg: Dict[str, Any]) -> float:
    """Score low density and flank recovery, with a continuity penalty."""
    q0 = _structural_recall_float(
        row,
        [
            "density_valley_boundary_likeness",
            "start_block_boundary_likeness",
            "feature_gap_boundary_likeness",
            "feature_gap_reward",
            "score",
        ],
        0.0,
    )
    dens = _structural_recall_float(
        row,
        ["gap_density_ratio", "start_block_gap_density_ratio", "feature_gap_gap_density_ratio"],
        999.0,
    )
    start_contrast = _structural_recall_float(
        row,
        [
            "start_contrast",
            "start_block_start_contrast",
            "feature_calibration_post_start_contrast",
        ],
        0.0,
    )
    start_rows = _structural_recall_float(
        row,
        [
            "start_dense_rows",
            "start_block_start_dense_rows",
            "feature_calibration_post_start_dense_rows",
        ],
        0.0,
    )
    active3 = _structural_recall_float(
        row, ["active3_frac", "start_block_active3_frac", "feature_gap_active3_frac"], 1.0
    )
    through_rows = _structural_recall_float(row, ["continuity_through_rows"], 0.0)
    max_consec = _structural_recall_float(row, ["continuity_max_consec_active_frac"], 0.0)
    decision = _structural_recall_str(
        row, ["density_valley_decision", "start_block_decision", "decision"], ""
    )
    q = float(q0)
    if dens <= 0.25:
        q += 3.0
    elif dens <= 0.4:
        q += 2.0
    elif dens <= 0.55:
        q += 0.9
    elif dens <= 0.72:
        q += 0.2
    else:
        q -= 2.6
    if start_rows >= 3 and start_contrast >= 3.0:
        q += 2.4
    elif start_rows >= 3 and start_contrast >= 1.75:
        q += 1.4
    elif start_rows >= 2 and start_contrast >= 1.45:
        q += 0.8
    if active3 <= 0.35:
        q += 0.7
    elif active3 >= 0.85 and dens > 0.5 and (start_contrast < 1.45):
        q -= 1.3
    if through_rows >= 3 and dens > 0.6 and (start_contrast < 1.35):
        q -= 4.0
    elif through_rows >= 3 and dens > 0.5 and (start_contrast < 1.6):
        q -= 2.0
    elif through_rows >= 2 and dens > 0.62 and (start_contrast < 1.3):
        q -= 1.5
    if max_consec >= 0.8 and dens > 0.58 and (start_contrast < 1.35):
        q -= 1.8
    if "reject_gap_width_too_small" in decision:
        q -= 0.2
    if "reject_continuity_three_rows" in decision and (
        not (dens <= 0.48 or start_contrast >= 1.75)
    ):
        q -= 0.8
    try:
        if _structural_gap_is_strong_dense_scan_candidate(row, cfg):
            q = max(
                q,
                _structural_gap_num(row, ["score"], q)
                + float(cfg.get("STRUCTURAL_GAP_DENSE_QUALITY_BONUS", 2.6)),
            )
    except NameError:
        pass
    try:
        if _structural_gap_is_bad_feature_gap_through(row, cfg):
            q -= float(cfg.get("STRUCTURAL_GAP_FEATURE_BAD_QUALITY_PENALTY", 6.0))
    except NameError:
        pass
    return float(q)


def _filter_dense_near_feature_base(
    dense: pd.DataFrame, cands_out: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Suppress dense-scan duplicates near existing feature or observed gaps."""
    if dense is None or len(dense) == 0 or cands_out is None or (len(cands_out) == 0):
        return dense
    if "center_s" not in dense.columns or "center_s" not in cands_out.columns:
        return dense
    tol = float(cfg.get("CANDIDATE_FILTER_DENSE_SUPPRESS_NEAR_FEATURE_PX", 1800.0))
    if tol <= 0:
        return dense
    ref = cands_out.copy()
    ref = ref[pd.to_numeric(ref["center_s"], errors="coerce").notna()].copy()
    if len(ref) == 0:
        return dense

    def _is_ref_gap_row(r: pd.Series) -> bool:
        k = str(r.get("kind", "")).lower()
        cls = str(r.get("candidate_class", "")).lower()
        st = str(r.get("selected_stage", "")).lower()
        return (
            "feature_gap" in k
            or "observed_gap" in k
            or "feature_gap" in st
            or ("observed_gap" in st)
            or ("feature" in cls)
            or ("observed" in cls)
        )

    ref = ref[ref.apply(_is_ref_gap_row, axis=1)].copy()
    if len(ref) == 0:
        return dense
    ref_centers = pd.to_numeric(ref["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref_centers) == 0:
        return dense
    out = dense.copy()
    keep_mask = []
    for _, row in out.iterrows():
        c = _structural_gap_num(row, ["center_s"], np.nan)
        if not np.isfinite(c):
            keep_mask.append(False)
            continue
        dmin = float(np.min(np.abs(ref_centers - c)))
        keep_mask.append(dmin > tol)
    out = out.loc[keep_mask].copy()
    return out


def _filter_dense_near_feature(
    dense: pd.DataFrame, cands_out: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    if not bool(cfg.get("_RUNTIME_REPAIR_ACTIVE", False)):
        return _filter_dense_near_feature_base(dense, cands_out, cfg)
    if dense is None or len(dense) == 0 or cands_out is None or (len(cands_out) == 0):
        return dense
    if "center_s" not in dense.columns or "center_s" not in cands_out.columns:
        return dense
    tol = float(cfg.get("CANDIDATE_FILTER_DENSE_SUPPRESS_NEAR_FEATURE_PX", 1800.0))
    if tol <= 0:
        return dense
    ref = cands_out.copy()
    ref = ref[pd.to_numeric(ref["center_s"], errors="coerce").notna()].copy()
    if len(ref) == 0:
        return dense

    def _is_ref_gap_row(r: pd.Series) -> bool:
        k = str(r.get("kind", "")).lower()
        cls = str(r.get("candidate_class", "")).lower()
        st = str(r.get("selected_stage", "")).lower()
        is_gap = (
            "feature_gap" in k
            or "observed_gap" in k
            or "feature_gap" in st
            or ("observed_gap" in st)
            or ("feature" in cls)
            or ("observed" in cls)
        )
        if not is_gap or cls.startswith(("c_", "d_")):
            return False
        decisions = [
            str(r.get(col, "")).lower()
            for col in ("density_valley_decision", "start_block_decision", "decision")
        ]
        if any(("reject" in decision for decision in decisions)):
            return False
        return not _structural_gap_is_bad_feature_gap_through(r, cfg)

    ref = ref[ref.apply(_is_ref_gap_row, axis=1)].copy()
    if len(ref) == 0:
        return dense
    ref_centers = pd.to_numeric(ref["center_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref_centers) == 0:
        return dense
    out = dense.copy()
    keep_mask = []
    for _, row in out.iterrows():
        c = _structural_gap_num(row, ["center_s"], np.nan)
        if not np.isfinite(c):
            keep_mask.append(False)
            continue
        dmin = float(np.min(np.abs(ref_centers - c)))
        keep_mask.append(dmin > tol)
    out = out.loc[keep_mask].copy()
    return out


def _structural_gap_append_strong_dense_to_candidates(
    cands_out: pd.DataFrame, original_cands: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if cands_out is not None and len(cands_out):
        parts.append(cands_out.copy())
    if (
        bool(cfg.get("ENABLE_STRUCTURAL_GAP_DENSE_SCAN_PRESERVE", True))
        and original_cands is not None
        and len(original_cands)
        and ("center_s" in original_cands.columns)
    ):
        base = original_cands.copy()
        base = base[pd.to_numeric(base["center_s"], errors="coerce").notna()].copy()
        if len(base):
            keep = base.apply(
                lambda r: _structural_gap_is_strong_dense_scan_candidate(r, cfg), axis=1
            )
            dense = base[keep].copy()
            if len(dense):
                dense = _filter_dense_near_feature(dense, cands_out, cfg)
            if len(dense):
                dense["structural_gap_dense_preserve"] = 1
                dense["structural_gap_bad_feature_gap_through"] = 0
                dense["selected_stage"] = dense.get(
                    "selected_stage", "structural_gap_preserved_from_original_dense_scan"
                )
                parts.append(dense)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out[
        pd.to_numeric(
            out.get("center_s", pd.Series(np.nan, index=out.index)), errors="coerce"
        ).notna()
    ].copy()
    if len(out):
        out["center_s"] = out["center_s"].astype(float)
    return out.reset_index(drop=True)


def _structural_gap_fill_feature_gap_diag_from_log(
    cands_out: pd.DataFrame, log: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Restore missing diagnostic fields from the nearest matching interval record."""
    if cands_out is None or len(cands_out) == 0 or log is None or (len(log) == 0):
        return cands_out
    if "center_s" not in cands_out.columns or "center_s" not in log.columns:
        return cands_out
    out = cands_out.copy()
    lg = log.copy()
    out = out[pd.to_numeric(out["center_s"], errors="coerce").notna()].copy()
    lg = lg[pd.to_numeric(lg["center_s"], errors="coerce").notna()].copy()
    if len(out) == 0 or len(lg) == 0:
        return out
    out["center_s"] = out["center_s"].astype(float)
    lg["center_s"] = lg["center_s"].astype(float)
    diag_cols = [
        "gap_density_ratio",
        "gap_density_per_bin",
        "pre_density_per_bin",
        "post_density_per_bin",
        "active3_frac",
        "start_dense_rows",
        "start_contrast",
        "start_active_ge2_frac",
        "continuity_through_rows",
        "continuity_max_consec_active_frac",
        "continuity_mean_row_density_ratio",
        "start_block_gap_density_ratio",
        "start_block_start_contrast",
        "feature_gap_gap_density_ratio",
        "feature_gap_boundary_likeness",
        "density_valley_decision",
        "start_block_decision",
        "decision",
    ]
    available = [c for c in diag_cols if c in lg.columns]
    if not available:
        return out
    text_diag_cols = {"density_valley_decision", "start_block_decision", "decision"}
    for col in available:
        if col in text_diag_cols and col in out.columns:
            out[col] = out[col].astype(object)
    tol = max(600.0, float(cfg.get("SCAN_STEP", 120.0)) * 6.0)
    lg_centers = lg["center_s"].to_numpy(dtype=float)
    for idx, row in out.iterrows():
        c = float(row["center_s"])
        j = int(np.argmin(np.abs(lg_centers - c)))
        if abs(float(lg_centers[j]) - c) > tol:
            continue
        src = lg.iloc[j]
        for col in available:
            if col not in out.columns:
                dtype = object if col in text_diag_cols else float
                out[col] = pd.Series(np.nan, index=out.index, dtype=dtype)
            cur = out.at[idx, col]
            if pd.isna(cur) or str(cur).strip() == "":
                out.at[idx, col] = src.get(col, cur)
    return out.reset_index(drop=True)


def build_feature_gap_proposals(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cands2, log = build_density_valley_proposals(work, cands, cfg)
    if log is None or len(log) == 0 or "center_s" not in log.columns:
        return (cands2, log)
    lg = log.copy()
    lg = lg[pd.to_numeric(lg["center_s"], errors="coerce").notna()].copy()
    if len(lg) == 0:
        return (cands2, log)
    lg["center_s"] = lg["center_s"].astype(float)
    lg["structural_recall_quality"] = lg.apply(
        lambda r: _structural_recall_candidate_quality(r, cfg), axis=1
    )
    lg["structural_recall_gap_density_ratio"] = lg.apply(
        lambda r: _structural_recall_float(
            r,
            [
                "gap_density_ratio",
                "start_block_gap_density_ratio",
                "feature_gap_gap_density_ratio",
            ],
            999.0,
        ),
        axis=1,
    )
    lg["structural_recall_start_contrast"] = lg.apply(
        lambda r: _structural_recall_float(
            r,
            [
                "start_contrast",
                "start_block_start_contrast",
                "feature_calibration_post_start_contrast",
            ],
            0.0,
        ),
        axis=1,
    )
    min_q = float(cfg.get("STRUCTURAL_RECALL_RECALL_MIN_QUALITY", 7.6))
    dens_max = float(cfg.get("STRUCTURAL_RECALL_RECALL_MAX_DENSITY_RATIO", 0.72))
    start_min = float(cfg.get("STRUCTURAL_RECALL_RECALL_MIN_START_CONTRAST", 1.18))
    strong_q = float(cfg.get("STRUCTURAL_RECALL_RECALL_STRONG_QUALITY", 10.2))
    strong_start = float(cfg.get("STRUCTURAL_RECALL_RECALL_STRONG_START_CONTRAST", 1.75))
    low_dens = float(cfg.get("STRUCTURAL_RECALL_RECALL_LOW_DENSITY_RATIO", 0.48))
    inject = lg[
        (lg["structural_recall_quality"] >= min_q)
        & (
            (lg["structural_recall_gap_density_ratio"] <= dens_max)
            | (lg["structural_recall_start_contrast"] >= start_min)
            | (lg["structural_recall_quality"] >= strong_q)
            | (lg["structural_recall_gap_density_ratio"] <= low_dens)
            & (lg["structural_recall_start_contrast"] >= strong_start)
        )
    ].copy()
    if len(inject) == 0:
        log = lg
        log["structural_recall_injected_to_candidate_table"] = 0
        cands_keep = _structural_gap_append_strong_dense_to_candidates(cands2, cands, cfg)
        cands_keep = _structural_gap_fill_feature_gap_diag_from_log(cands_keep, lg, cfg)
        return (cands_keep, log)
    rows: List[Dict[str, Any]] = []
    for _, r in inject.iterrows():
        c = float(r["center_s"])
        gs = _structural_recall_float(
            r, ["gap_start_s", "start_block_gap_start_s", "feature_gap_gap_start_s"], np.nan
        )
        ge = _structural_recall_float(
            r, ["gap_end_s", "start_block_gap_end_s", "feature_gap_gap_end_s"], np.nan
        )
        gw = _structural_recall_float(
            r, ["gap_width", "start_block_gap_width", "feature_gap_gap_width"], np.nan
        )
        if not np.isfinite(gs) or not np.isfinite(ge) or ge <= gs:
            if not np.isfinite(gw) or gw <= 0:
                gw = max(800.0, float(cfg.get("SCAN_STEP", 120.0)) * 8.0)
            gs = c - 0.5 * gw
            ge = c + 0.5 * gw
        gw = float(max(ge - gs, 800.0))
        q = float(r["structural_recall_quality"])
        cls = "A_feature_gap" if q >= 12.8 else "B_feature_gap"
        row = {
            "s0": float(gs),
            "s1": float(ge),
            "center_s": float(c),
            "width": float(gw),
            "score": float(q),
            "kind": "feature_gap_interval",
            "candidate_class": cls,
            "candidate_class_rank": 4 if cls.startswith("A") else 3,
            "is_graph_strong": 1,
            "is_graph_usable": 1,
            "selected_stage": "structural_recall_injected_from_feature_gap_log",
            "structural_recall_quality": q,
            "structural_recall_gap_density_ratio": float(
                r.get("structural_recall_gap_density_ratio", np.nan)
            ),
            "structural_recall_start_contrast": float(
                r.get("structural_recall_start_contrast", np.nan)
            ),
            "density_valley_decision": str(r.get("decision", "")),
            "density_valley_boundary_likeness": _structural_recall_float(
                r, ["density_valley_boundary_likeness"], q
            ),
            "density_valley_core_low_frac": _structural_recall_float(
                r, ["density_valley_core_low_frac"], np.nan
            ),
            "density_valley_relaxed_low_frac": _structural_recall_float(
                r, ["density_valley_relaxed_low_frac"], np.nan
            ),
            "density_valley_high_multirow_frac": _structural_recall_float(
                r, ["density_valley_high_multirow_frac"], np.nan
            ),
            "start_block_gap_start_s": float(gs),
            "start_block_gap_end_s": float(ge),
            "start_block_gap_width": float(gw),
            "start_block_boundary_likeness": q,
            "start_block_gap_density_ratio": float(
                r.get("structural_recall_gap_density_ratio", np.nan)
            ),
            "start_block_start_contrast": float(
                r.get("structural_recall_start_contrast", np.nan)
            ),
            "start_block_decision": str(r.get("decision", "")),
            "feature_gap_gap_start_s": float(gs),
            "feature_gap_gap_end_s": float(ge),
            "feature_gap_gap_width": float(gw),
            "feature_gap_boundary_likeness": q,
            "feature_gap_gap_density_ratio": float(
                r.get("structural_recall_gap_density_ratio", np.nan)
            ),
        }
        for cc in [
            "gap_density_ratio",
            "gap_density_per_bin",
            "pre_density_per_bin",
            "post_density_per_bin",
            "active3_frac",
            "start_dense_rows",
            "start_contrast",
            "start_active_ge2_frac",
            "continuity_through_rows",
            "continuity_max_consec_active_frac",
            "continuity_mean_row_density_ratio",
        ]:
            if cc in r.index:
                row[cc] = r.get(cc)
        rows.append(row)
    injected = pd.DataFrame(rows)
    if len(cands2):
        cands_out = pd.concat([cands2, injected], ignore_index=True, sort=False)
    else:
        cands_out = injected
    cands_out = _structural_gap_append_strong_dense_to_candidates(cands_out, cands, cfg)
    cands_out = _structural_gap_fill_feature_gap_diag_from_log(cands_out, lg, cfg)
    bin_w = max(1600.0, float(cfg.get("SCAN_STEP", 120.0)) * 10.0)
    cands_out["_structural_recall_center_bin"] = np.round(
        cands_out["center_s"].astype(float) / bin_w
    ).astype(int)
    cands_out["_structural_recall_q_for_dedup"] = cands_out.apply(
        lambda rr: (
            _structural_recall_float(
                rr, ["structural_recall_quality"], _structural_recall_candidate_quality(rr, cfg)
            )
            if cfg.get("_RUNTIME_REPAIR_ACTIVE", False)
            else float(
                rr.get(
                    "structural_recall_quality", _structural_recall_candidate_quality(rr, cfg)
                )
            )
        ),
        axis=1,
    )
    cands_out = (
        cands_out.sort_values(["_structural_recall_q_for_dedup", "score"], ascending=False)
        .groupby("_structural_recall_center_bin", as_index=False)
        .head(1)
    )
    cands_out = (
        cands_out.drop(
            columns=["_structural_recall_center_bin", "_structural_recall_q_for_dedup"],
            errors="ignore",
        )
        .sort_values("center_s")
        .reset_index(drop=True)
    )
    cands_out = _structural_gap_fill_feature_gap_diag_from_log(cands_out, lg, cfg)
    lg["structural_recall_injected_to_candidate_table"] = (
        lg["center_s"].round(6).isin(injected["center_s"].round(6)).astype(int)
    )
    lg["structural_recall_n_injected_candidates"] = int(len(injected))
    return (cands_out, lg)


def _structural_recall_build_recall_pool(
    cands: pd.DataFrame, ref_len: float, cfg: Dict[str, Any]
) -> pd.DataFrame:
    if cands is None or len(cands) == 0:
        return pd.DataFrame()
    pool = cands.copy()
    if "center_s" not in pool.columns:
        return pd.DataFrame()
    pool = pool[pd.to_numeric(pool["center_s"], errors="coerce").notna()].copy()
    pool["center_s"] = pool["center_s"].astype(float)
    kind = pool.get("kind", pd.Series([""] * len(pool), index=pool.index)).astype(str)
    cls = pool.get("candidate_class", pd.Series([""] * len(pool), index=pool.index)).astype(str)
    dec = pool.get(
        "density_valley_decision",
        pool.get(
            "start_block_decision",
            pool.get("decision", pd.Series([""] * len(pool), index=pool.index)),
        ),
    ).astype(str)
    has_feat_cols = (
        pool[
            [
                c
                for c in [
                    "density_valley_boundary_likeness",
                    "start_block_boundary_likeness",
                    "feature_gap_boundary_likeness",
                ]
                if c in pool.columns
            ]
        ]
        .notna()
        .any(axis=1)
        if any(
            (
                c in pool.columns
                for c in [
                    "density_valley_boundary_likeness",
                    "start_block_boundary_likeness",
                    "feature_gap_boundary_likeness",
                ]
            )
        )
        else pd.Series(False, index=pool.index)
    )
    strong_dense = (
        pool.apply(lambda r: _structural_gap_is_strong_dense_scan_candidate(r, cfg), axis=1)
        if bool(cfg.get("ENABLE_STRUCTURAL_GAP_DENSE_SCAN_PRESERVE", True))
        else pd.Series(False, index=pool.index)
    )
    m = (
        kind.str.contains("feature_gap|observed_gap", case=False, na=False)
        | cls.str.contains("observed_gap|feature", case=False, na=False)
        | dec.str.contains(
            "candidate_continuity|reject_gap_width|reject_continuity", case=False, na=False
        )
        | has_feat_cols
        | strong_dense
    )
    pool = pool[m].copy()
    if len(pool):
        pool["structural_gap_dense_preserve"] = pool.apply(
            lambda r: int(_structural_gap_is_strong_dense_scan_candidate(r, cfg)), axis=1
        )
        pool["structural_gap_bad_feature_gap_through"] = pool.apply(
            lambda r: int(_structural_gap_is_bad_feature_gap_through(r, cfg)), axis=1
        )
    if len(pool) == 0:
        return pool
    pool["structural_recall_quality"] = pool.apply(
        lambda r: _structural_recall_candidate_quality(r, cfg), axis=1
    )
    pool["structural_recall_gap_density_ratio"] = pool.apply(
        lambda r: _structural_recall_float(
            r,
            [
                "gap_density_ratio",
                "start_block_gap_density_ratio",
                "feature_gap_gap_density_ratio",
            ],
            999.0,
        ),
        axis=1,
    )
    pool["structural_recall_start_contrast"] = pool.apply(
        lambda r: _structural_recall_float(
            r,
            [
                "start_contrast",
                "start_block_start_contrast",
                "feature_calibration_post_start_contrast",
            ],
            0.0,
        ),
        axis=1,
    )
    pool["structural_recall_gap_width"] = pool.apply(
        lambda r: _structural_recall_float(
            r, ["gap_width", "start_block_gap_width", "feature_gap_gap_width", "width"], 0.0
        ),
        axis=1,
    )
    if "structural_gap_dense_preserve" not in pool.columns:
        pool["structural_gap_dense_preserve"] = pool.apply(
            lambda r: int(_structural_gap_is_strong_dense_scan_candidate(r, cfg)), axis=1
        )
    if "structural_gap_bad_feature_gap_through" not in pool.columns:
        pool["structural_gap_bad_feature_gap_through"] = pool.apply(
            lambda r: int(_structural_gap_is_bad_feature_gap_through(r, cfg)), axis=1
        )
    pool["structural_gap_dense_preserve"] = (
        pd.to_numeric(pool["structural_gap_dense_preserve"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    pool["structural_gap_bad_feature_gap_through"] = (
        pd.to_numeric(pool["structural_gap_bad_feature_gap_through"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    min_q = float(cfg.get("STRUCTURAL_RECALL_RECALL_MIN_QUALITY", 7.6))
    dens_max = float(cfg.get("STRUCTURAL_RECALL_RECALL_MAX_DENSITY_RATIO", 0.72))
    start_min = float(cfg.get("STRUCTURAL_RECALL_RECALL_MIN_START_CONTRAST", 1.18))
    strong_q = float(cfg.get("STRUCTURAL_RECALL_RECALL_STRONG_QUALITY", 10.2))
    strong_start = float(cfg.get("STRUCTURAL_RECALL_RECALL_STRONG_START_CONTRAST", 1.75))
    low_dens = float(cfg.get("STRUCTURAL_RECALL_RECALL_LOW_DENSITY_RATIO", 0.48))
    pass_evidence = (pool["structural_recall_quality"] >= min_q) & (
        (pool["structural_recall_gap_density_ratio"] <= dens_max)
        | (pool["structural_recall_start_contrast"] >= start_min)
        | (pool["structural_recall_quality"] >= strong_q)
        | (
            pd.to_numeric(
                pool.get("structural_gap_dense_preserve", pd.Series(0, index=pool.index)),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
    )
    pass_strong = (pool["structural_recall_gap_density_ratio"] <= low_dens) & (
        pool["structural_recall_start_contrast"] >= strong_start
    )
    pass_dense = (
        pd.to_numeric(
            pool.get("structural_gap_dense_preserve", pd.Series(0, index=pool.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        == 1
    )
    bad_feat = (
        pd.to_numeric(
            pool.get("structural_gap_bad_feature_gap_through", pd.Series(0, index=pool.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        == 1
    )
    pool = pool[(pass_evidence | pass_strong | pass_dense) & (~bad_feat | pass_dense)].copy()
    if len(pool) == 0:
        return pool
    bin_w = max(float(ref_len) * 0.08, 1600.0)
    pool["structural_recall_center_bin"] = np.round(
        pool["center_s"].astype(float) / bin_w
    ).astype(int)
    pool = (
        pool.sort_values(["structural_recall_quality", "score"], ascending=False)
        .groupby("structural_recall_center_bin", as_index=False)
        .head(1)
    )
    return pool.sort_values("center_s").reset_index(drop=True)


def _structural_recall_center_quality_lookup(
    pool: pd.DataFrame, selected: pd.DataFrame, cfg: Dict[str, Any]
) -> Dict[float, float]:
    q: Dict[float, float] = {}
    for _, r in pool.iterrows() if pool is not None and len(pool) else []:
        q[round(float(r["center_s"]), 6)] = (
            _structural_recall_float(
                r, ["structural_recall_quality"], _structural_recall_candidate_quality(r, cfg)
            )
            if cfg.get("_RUNTIME_REPAIR_ACTIVE", False)
            else float(
                r.get("structural_recall_quality", _structural_recall_candidate_quality(r, cfg))
            )
        )
    for _, r in selected.iterrows() if selected is not None and len(selected) else []:
        c = round(float(r["center_s"]), 6)
        q[c] = max(q.get(c, -1000000000.0), _structural_recall_candidate_quality(r, cfg))
    return q


def _structural_recall_make_selected_from_centers(
    centers: List[float], base_selected: pd.DataFrame, pool: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    rows: List[pd.Series] = []
    base = pd.concat(
        [
            base_selected.copy() if base_selected is not None else pd.DataFrame(),
            pool.copy() if pool is not None else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    if len(base) == 0 or len(centers) == 0:
        empty = base.iloc[:0].copy()
        if "center_s" not in empty.columns:
            empty["center_s"] = pd.Series(dtype=float)
        return empty.reset_index(drop=True)
    base["_center_round"] = base["center_s"].astype(float).round(6)
    base["_structural_recall_quality_tmp"] = base.apply(
        lambda r: (
            _structural_recall_float(
                r, ["structural_recall_quality"], _structural_recall_candidate_quality(r, cfg)
            )
            if cfg.get("_RUNTIME_REPAIR_ACTIVE", False)
            else float(
                r.get("structural_recall_quality", _structural_recall_candidate_quality(r, cfg))
            )
        ),
        axis=1,
    )
    for c in sorted(set((float(x) for x in centers))):
        cr = round(c, 6)
        g = base[np.abs(base["center_s"].astype(float) - c) <= 1e-05]
        if len(g) == 0:
            g = base.iloc[(base["center_s"].astype(float) - c).abs().argsort()[:1]]
        row = g.sort_values("_structural_recall_quality_tmp", ascending=False).iloc[0].copy()
        row["center_s"] = float(c)
        if "s0" not in row.index or not np.isfinite(_safe_float(row.get("s0"), np.nan)):
            w = _structural_recall_float(
                row,
                [
                    "width",
                    "structural_recall_gap_width",
                    "start_block_gap_width",
                    "feature_gap_gap_width",
                ],
                2400.0,
            )
            row["s0"] = float(c - 0.5 * max(w, 800.0))
            row["s1"] = float(c + 0.5 * max(w, 800.0))
            row["width"] = float(max(w, 800.0))
        if "score" not in row.index or not np.isfinite(_safe_float(row.get("score"), np.nan)):
            row["score"] = float(
                row.get(
                    "_structural_recall_quality_tmp",
                    _structural_recall_candidate_quality(row, cfg),
                )
            )
        row["structural_recall_quality"] = float(
            row.get(
                "_structural_recall_quality_tmp", _structural_recall_candidate_quality(row, cfg)
            )
        )
        row["selected_stage"] = "structural_recall_fullcover_evidence_recall"
        rows.append(
            row.drop(
                labels=[
                    x
                    for x in ["_center_round", "_structural_recall_quality_tmp"]
                    if x in row.index
                ],
                errors="ignore",
            )
        )
    out = pd.DataFrame(rows)
    return out.sort_values("center_s").reset_index(drop=True)


def _structural_recall_add_recall_boundaries(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not bool(cfg.get("ENABLE_STRUCTURAL_RECALL_EVIDENCE_RECALL", True)) or len(pool) == 0:
        return (selected, pd.DataFrame())
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    centers = (
        sorted(selected["center_s"].astype(float).tolist())
        if selected is not None and len(selected)
        else []
    )
    min_spacing = float(cfg.get("STRUCTURAL_RECALL_MIN_CENTER_SPACING_RATIO", 0.34)) * ref_len
    target_window = float(cfg.get("STRUCTURAL_RECALL_TARGET_WINDOW_RATIO", 0.46)) * ref_len
    balance_w = float(cfg.get("STRUCTURAL_RECALL_RECALL_TARGET_BALANCE_WEIGHT", 18.0))
    logs: List[Dict[str, Any]] = []
    for cycle in range(int(cfg.get("STRUCTURAL_RECALL_RECALL_MAX_CYCLES", 4))):
        added_this_cycle: List[float] = []
        cuts = [smin] + sorted(centers) + [smax]
        for seg_i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            length = float(b - a)
            is_edge = seg_i == 0 or seg_i == len(cuts) - 2
            ratio_thr = (
                float(cfg.get("STRUCTURAL_RECALL_EDGE_LONG_SEG_RECALL_RATIO", 1.75))
                if is_edge
                else float(cfg.get("STRUCTURAL_RECALL_LONG_SEG_RECALL_RATIO", 1.45))
            )
            if length < ratio_thr * ref_len:
                continue
            n_units = int(round(length / ref_len))
            n_add = max(0, n_units - 1)
            if n_add <= 0:
                continue
            targets = [a + length * (j + 1) / (n_add + 1) for j in range(n_add)]
            for target in targets:
                cand = pool[
                    (pool["center_s"].astype(float) > a + 0.12 * ref_len)
                    & (pool["center_s"].astype(float) < b - 0.12 * ref_len)
                ].copy()
                if len(cand) == 0:
                    logs.append(
                        {
                            "decision": "structural_recall_recall_no_candidate_in_long_segment",
                            "cycle": cycle,
                            "seg_start": a,
                            "seg_end": b,
                            "target_s": target,
                            "seg_len": length,
                        }
                    )
                    continue
                all_centers = centers + added_this_cycle
                if all_centers:
                    dist_existing = np.min(
                        np.abs(
                            cand["center_s"].astype(float).to_numpy()[:, None]
                            - np.asarray(all_centers, dtype=float)[None, :]
                        ),
                        axis=1,
                    )
                    cand = cand[dist_existing >= min_spacing].copy()
                if len(cand) == 0:
                    logs.append(
                        {
                            "decision": "structural_recall_recall_blocked_by_spacing",
                            "cycle": cycle,
                            "seg_start": a,
                            "seg_end": b,
                            "target_s": target,
                            "seg_len": length,
                        }
                    )
                    continue
                cand["_target_dist"] = np.abs(cand["center_s"].astype(float) - target)
                cand = cand[cand["_target_dist"] <= target_window].copy()
                if len(cand) == 0:
                    logs.append(
                        {
                            "decision": "structural_recall_recall_no_candidate_near_target",
                            "cycle": cycle,
                            "seg_start": a,
                            "seg_end": b,
                            "target_s": target,
                            "seg_len": length,
                        }
                    )
                    continue
                cand["_target_score"] = cand["structural_recall_quality"].astype(
                    float
                ) - balance_w * (cand["_target_dist"] / max(ref_len, 1.0))
                best = cand.sort_values("_target_score", ascending=False).iloc[0]
                min_q = float(cfg.get("STRUCTURAL_RECALL_RECALL_MIN_QUALITY", 7.6))
                if float(best["structural_recall_quality"]) < min_q:
                    logs.append(
                        {
                            "decision": "structural_recall_recall_best_quality_too_low",
                            "cycle": cycle,
                            "seg_start": a,
                            "seg_end": b,
                            "target_s": target,
                            "best_s": float(best["center_s"]),
                            "best_quality": float(best["structural_recall_quality"]),
                        }
                    )
                    continue
                trial_centers = sorted(centers + added_this_cycle + [float(best["center_s"])])
                trial_cuts = [smin] + trial_centers + [smax]
                trial_lens = np.diff(np.asarray(trial_cuts, dtype=float))
                if (
                    float(np.min(trial_lens))
                    < float(cfg.get("STRUCTURAL_RECALL_MIN_SEG_AFTER_RECALL_RATIO", 0.42))
                    * ref_len
                ):
                    logs.append(
                        {
                            "decision": "structural_recall_recall_would_make_too_short_segment",
                            "cycle": cycle,
                            "seg_start": a,
                            "seg_end": b,
                            "target_s": target,
                            "best_s": float(best["center_s"]),
                            "best_quality": float(best["structural_recall_quality"]),
                            "min_trial_len": float(np.min(trial_lens)),
                        }
                    )
                    continue
                added_this_cycle.append(float(best["center_s"]))
                logs.append(
                    {
                        "decision": "structural_recall_recall_added_boundary_in_long_segment",
                        "cycle": cycle,
                        "seg_start": a,
                        "seg_end": b,
                        "seg_len": length,
                        "target_s": target,
                        "added_s": float(best["center_s"]),
                        "quality": float(best["structural_recall_quality"]),
                        "gap_density_ratio": float(
                            best.get("structural_recall_gap_density_ratio", np.nan)
                        ),
                        "start_contrast": float(
                            best.get("structural_recall_start_contrast", np.nan)
                        ),
                    }
                )
        if not added_this_cycle:
            break
        centers = sorted(set([round(float(x), 6) for x in centers + added_this_cycle]))
    out = _structural_recall_make_selected_from_centers(centers, selected, pool, cfg)
    return (out, pd.DataFrame(logs))


def _structural_recall_prune_short_segments(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (
        not bool(cfg.get("ENABLE_STRUCTURAL_RECALL_SHORT_SEGMENT_PRUNE", True))
        or selected is None
        or len(selected) == 0
    ):
        return (selected, pd.DataFrame())
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    centers = sorted(selected["center_s"].astype(float).tolist())
    qmap = _structural_recall_center_quality_lookup(pool, selected, cfg)
    logs: List[Dict[str, Any]] = []
    min_len = float(cfg.get("STRUCTURAL_RECALL_MIN_SEG_AFTER_PRUNE_RATIO", 0.5)) * ref_len
    for it in range(int(cfg.get("STRUCTURAL_RECALL_SHORT_SEGMENT_MAX_ITERS", 8))):
        cuts = [smin] + centers + [smax]
        lens = np.diff(np.asarray(cuts, dtype=float))
        if len(lens) == 0 or float(np.min(lens)) >= min_len:
            break
        idx = int(np.argmin(lens))
        short_len = float(lens[idx])
        candidates_to_delete = []
        if idx - 1 >= 0 and idx - 1 < len(centers):
            candidates_to_delete.append(centers[idx - 1])
        if idx >= 0 and idx < len(centers):
            candidates_to_delete.append(centers[idx])
        if not candidates_to_delete:
            break
        if idx == 0:
            del_c = centers[0]
        elif idx == len(lens) - 1:
            del_c = centers[-1]
        else:
            del_c = min(
                candidates_to_delete,
                key=lambda c: qmap.get(
                    round(float(c), 6),
                    _structural_recall_candidate_quality(
                        selected.iloc[
                            (selected["center_s"].astype(float) - c).abs().argsort()[:1][0]
                        ],
                        cfg,
                    ),
                ),
            )
        logs.append(
            {
                "decision": "structural_recall_prune_delete_boundary_causing_short_segment",
                "iter": it,
                "short_seg_index": idx + 1,
                "short_len": short_len,
                "min_len_required": min_len,
                "deleted_s": float(del_c),
                "deleted_quality": float(qmap.get(round(float(del_c), 6), np.nan)),
            }
        )
        centers = [c for c in centers if abs(c - del_c) > 1e-05]
    out = _structural_recall_make_selected_from_centers(centers, selected, pool, cfg)
    return (out, pd.DataFrame(logs))


def select_sequence_with_structural_recall(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base_selected, base_logs, mode_info = select_sequence_with_continuity_check(
        work, cands, cfg
    )
    ref_len, ref_meta = _structural_recall_aux_len(work, cfg)
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info.update(ref_meta)
    mode_info["mode"] = "structural_recall_fullcover_evidence_recall"
    mode_info["typical_len"] = float(ref_len)
    mode_info["feature_gap_aux_len"] = float(ref_len)
    if base_selected is None:
        base_selected = pd.DataFrame()
    pool = _structural_recall_build_recall_pool(cands, ref_len, cfg)
    selected = (
        base_selected.copy().sort_values("center_s").reset_index(drop=True)
        if len(base_selected)
        else pd.DataFrame()
    )
    recall_log = pd.DataFrame()
    prune_log = pd.DataFrame()
    if len(selected) and len(pool):
        recall_logs_all: List[pd.DataFrame] = []
        prune_logs_all: List[pd.DataFrame] = []
        for _structural_recall_outer in range(2):
            selected, _rlog = _structural_recall_add_recall_boundaries(
                work, selected, pool, ref_len, cfg
            )
            if _rlog is not None and len(_rlog):
                _rlog = _rlog.copy()
                _rlog["structural_recall_outer_cycle"] = int(_structural_recall_outer)
                recall_logs_all.append(_rlog)
            selected, _plog = _structural_recall_prune_short_segments(
                work, selected, pool, ref_len, cfg
            )
            if _plog is not None and len(_plog):
                _plog = _plog.copy()
                _plog["structural_recall_outer_cycle"] = int(_structural_recall_outer)
                prune_logs_all.append(_plog)
        recall_log = (
            pd.concat(recall_logs_all, ignore_index=True, sort=False)
            if recall_logs_all
            else pd.DataFrame()
        )
        prune_log = (
            pd.concat(prune_logs_all, ignore_index=True, sort=False)
            if prune_logs_all
            else pd.DataFrame()
        )
        selected["selected_stage"] = "structural_recall_fullcover_evidence_recall_pruned"
    elif len(pool):
        strong = pool[
            pool["structural_recall_quality"]
            >= float(cfg.get("STRUCTURAL_RECALL_RECALL_STRONG_QUALITY", 10.2))
        ].copy()
        if len(strong):
            seed = strong.sort_values("structural_recall_quality", ascending=False).head(1)
            selected = _structural_recall_make_selected_from_centers(
                seed["center_s"].astype(float).tolist(), seed, pool, cfg
            )
            selected, recall_log = _structural_recall_add_recall_boundaries(
                work, selected, pool, ref_len, cfg
            )
            selected, prune_log = _structural_recall_prune_short_segments(
                work, selected, pool, ref_len, cfg
            )
            selected["selected_stage"] = "structural_recall_fullcover_evidence_recall_seeded"
    logs: List[pd.DataFrame] = []
    logs.append(
        pd.DataFrame(
            [
                {
                    "decision": "structural_recall_selection_summary",
                    "n_base_selected": int(len(base_selected)),
                    "n_recall_pool": int(len(pool)),
                    "n_selected_after_structural_recall": int(len(selected)),
                    "ref_len": float(ref_len),
                    "mode": mode_info["mode"],
                }
            ]
        )
    )
    if base_logs is not None and len(base_logs):
        logs.append(base_logs)
    if recall_log is not None and len(recall_log):
        logs.append(recall_log)
    if prune_log is not None and len(prune_log):
        logs.append(prune_log)
    if len(pool):
        pool_log_cols = [
            c
            for c in [
                "center_s",
                "structural_recall_quality",
                "structural_recall_gap_density_ratio",
                "structural_recall_start_contrast",
                "structural_recall_gap_width",
                "density_valley_decision",
                "start_block_decision",
                "kind",
                "candidate_class",
            ]
            if c in pool.columns
        ]
        plog = pool[pool_log_cols].copy()
        plog["decision"] = "structural_recall_recall_pool_candidate"
        logs.append(plog)
    out_logs = pd.concat(logs, ignore_index=True, sort=False) if logs else pd.DataFrame()
    return (selected.reset_index(drop=True), out_logs, mode_info)


def _local_relocation_get_float(
    row: pd.Series, names: List[str], default: float = np.nan
) -> float:
    return _structural_recall_float(row, names, default)


def _local_relocation_get_str(row: pd.Series, names: List[str], default: str = "") -> str:
    return _structural_recall_str(row, names, default)


def _local_relocation_gap_width(row: pd.Series) -> float:
    return _local_relocation_get_float(
        row,
        [
            "structural_recall_gap_width",
            "gap_width",
            "edge_gap_width",
            "start_block_gap_width",
            "feature_gap_gap_width",
            "width",
        ],
        np.nan,
    )


def _local_relocation_density(row: pd.Series) -> float:
    return _local_relocation_get_float(
        row,
        [
            "structural_recall_gap_density_ratio",
            "gap_density_ratio",
            "start_block_gap_density_ratio",
            "feature_gap_gap_density_ratio",
        ],
        np.nan,
    )


def _local_relocation_start_contrast(row: pd.Series) -> float:
    return _local_relocation_get_float(
        row,
        [
            "structural_recall_start_contrast",
            "start_contrast",
            "start_block_start_contrast",
            "feature_calibration_post_start_contrast",
        ],
        0.0,
    )


def _local_relocation_boundary_evidence_score(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> float:
    """Score gap evidence for local boundary relocation."""
    q28 = _local_relocation_get_float(row, ["structural_recall_quality"], np.nan)
    if not np.isfinite(q28):
        q28 = _structural_recall_candidate_quality(row, cfg)
    dens = _local_relocation_density(row)
    start = _local_relocation_start_contrast(row)
    width = _local_relocation_gap_width(row)
    through_rows = _local_relocation_get_float(row, ["continuity_through_rows"], 0.0)
    max_consec = _local_relocation_get_float(row, ["continuity_max_consec_active_frac"], 0.0)
    active3 = _local_relocation_get_float(
        row, ["active3_frac", "start_block_active3_frac", "feature_gap_active3_frac"], np.nan
    )
    decision = _local_relocation_get_str(
        row, ["density_valley_decision", "start_block_decision", "decision"], ""
    )
    kind = _local_relocation_get_str(row, ["kind"], "")
    cls = _local_relocation_get_str(row, ["candidate_class"], "")
    score = 0.52 * float(q28)
    if np.isfinite(dens):
        if dens <= 0.18:
            score += 4.8
        elif dens <= 0.3:
            score += 3.7
        elif dens <= 0.42:
            score += 2.2
        elif dens <= 0.55:
            score += 0.9
        elif dens <= 0.68:
            score -= 0.6
        else:
            score -= 3.0
    else:
        score -= 0.3
    if start >= 5.0:
        score += 3.1
    elif start >= 3.2:
        score += 2.2
    elif start >= 2.0:
        score += 1.1
    elif start >= 1.45:
        score += 0.35
    elif start < 1.15:
        score -= 0.8
    if np.isfinite(width):
        if width >= max(0.075 * ref_len, 3600.0):
            score += 1.6
        elif width >= max(0.05 * ref_len, 2400.0):
            score += 0.9
        elif width >= 1700.0:
            score += 0.2
        elif width <= float(cfg.get("LOCAL_RELOCATION_VERY_NARROW_GAP_WIDTH_ABS", 1000.0)):
            score -= 2.2
        elif width <= float(cfg.get("LOCAL_RELOCATION_NARROW_GAP_WIDTH_ABS", 1700.0)):
            score -= 1.2
    else:
        score -= 0.5
    if "observed_gap" in kind.lower() or "observed_gap" in cls.lower():
        if np.isfinite(width) and width >= 3000.0:
            score += 1.5
        else:
            score += 0.6
    if through_rows >= 3 and (not np.isfinite(dens) or dens > 0.5) and (start < 1.8):
        score -= 3.2
    elif through_rows >= 3 and np.isfinite(dens) and (dens > 0.42) and (start < 2.2):
        score -= 1.4
    if max_consec >= 0.78 and np.isfinite(dens) and (dens > 0.48) and (start < 1.75):
        score -= 1.4
    if (
        np.isfinite(active3)
        and active3 >= 0.85
        and np.isfinite(dens)
        and (dens > 0.48)
        and (start < 1.75)
    ):
        score -= 1.2
    if "reject_gap_width_too_small" in decision:
        if np.isfinite(width) and width <= float(
            cfg.get("LOCAL_RELOCATION_VERY_NARROW_GAP_WIDTH_ABS", 1000.0)
        ):
            score -= 1.2
        elif np.isfinite(width) and width <= float(
            cfg.get("LOCAL_RELOCATION_NARROW_GAP_WIDTH_ABS", 1700.0)
        ):
            score -= 0.6
    if _structural_gap_is_strong_dense_scan_candidate(row, cfg):
        raw_score = _structural_gap_num(row, ["score"], 0.0)
        joint = _structural_gap_num(row, ["joint_drop"], 1.0)
        score += (
            float(cfg.get("STRUCTURAL_GAP_DENSE_EVIDENCE_BONUS", 4.8))
            + 0.18 * max(0.0, raw_score - 9.5)
            + 1.2 * max(0.0, 0.22 - joint)
        )
    if _structural_gap_is_bad_feature_gap_through(row, cfg):
        score -= float(cfg.get("STRUCTURAL_GAP_FEATURE_BAD_EVIDENCE_PENALTY", 7.0))
    return float(score)


def _local_relocation_is_suspicious_selected(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> Tuple[bool, str]:
    width = _local_relocation_gap_width(row)
    edge_width = _local_relocation_get_float(row, ["gap_width", "edge_gap_width"], np.nan)
    dens = _local_relocation_density(row)
    start = _local_relocation_start_contrast(row)
    decision = _local_relocation_get_str(
        row, ["density_valley_decision", "start_block_decision", "decision"], ""
    )
    through_rows = _local_relocation_get_float(row, ["continuity_through_rows"], 0.0)
    reasons: List[str] = []
    if np.isfinite(width) and width <= float(
        cfg.get("LOCAL_RELOCATION_VERY_NARROW_GAP_WIDTH_ABS", 1000.0)
    ):
        reasons.append("very_narrow_gap_width")
    elif np.isfinite(width) and width <= float(
        cfg.get("LOCAL_RELOCATION_NARROW_GAP_WIDTH_ABS", 1700.0)
    ):
        reasons.append("narrow_gap_width")
    if np.isfinite(edge_width) and edge_width <= float(
        cfg.get("LOCAL_RELOCATION_NARROW_EDGE_WIDTH_ABS", 1350.0)
    ):
        reasons.append("narrow_edgefit_visual_gap")
    if "reject_gap_width_too_small" in decision:
        reasons.append("old_rule_rejected_as_too_small_gap")
    if np.isfinite(dens) and dens > 0.45 and (start < 1.6):
        reasons.append("weak_valley_and_weak_restart")
    if through_rows >= 3 and (not np.isfinite(dens) or dens > 0.42) and (start < 2.0):
        reasons.append("multirow_continuity_through_gap")
    if _structural_gap_is_bad_feature_gap_through(row, cfg):
        reasons.append("structural_gap_bad_feature_gap_through")
    return (len(reasons) > 0, ";".join(reasons))


def _local_relocation_build_relocation_pool(
    cands: pd.DataFrame, selected: pd.DataFrame, ref_len: float, cfg: Dict[str, Any]
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if cands is not None and len(cands):
        cc = cands.copy()
        cc = (
            cc[pd.to_numeric(cc.get("center_s"), errors="coerce").notna()].copy()
            if "center_s" in cc.columns
            else pd.DataFrame()
        )
        if len(cc):
            kind = cc.get("kind", pd.Series([""] * len(cc), index=cc.index)).astype(str)
            cls = cc.get("candidate_class", pd.Series([""] * len(cc), index=cc.index)).astype(
                str
            )
            dec = cc.get(
                "density_valley_decision",
                cc.get(
                    "start_block_decision",
                    cc.get("decision", pd.Series([""] * len(cc), index=cc.index)),
                ),
            ).astype(str)
            has_feature_cols = (
                cc[
                    [
                        c
                        for c in [
                            "density_valley_boundary_likeness",
                            "start_block_boundary_likeness",
                            "feature_gap_boundary_likeness",
                            "structural_recall_quality",
                        ]
                        if c in cc.columns
                    ]
                ]
                .notna()
                .any(axis=1)
                if any(
                    (
                        c in cc.columns
                        for c in [
                            "density_valley_boundary_likeness",
                            "start_block_boundary_likeness",
                            "feature_gap_boundary_likeness",
                            "structural_recall_quality",
                        ]
                    )
                )
                else pd.Series(False, index=cc.index)
            )
            rank = pd.to_numeric(
                cc.get("candidate_class_rank", pd.Series([0] * len(cc), index=cc.index)),
                errors="coerce",
            ).fillna(0)
            strong_dense = (
                cc.apply(
                    lambda r: _structural_gap_is_strong_dense_scan_candidate(r, cfg), axis=1
                )
                if bool(cfg.get("ENABLE_STRUCTURAL_GAP_DENSE_SCAN_PRESERVE", True))
                else pd.Series(False, index=cc.index)
            )
            m = (
                kind.str.contains("feature_gap|observed_gap", case=False, na=False)
                | cls.str.contains("feature|observed_gap", case=False, na=False)
                | dec.str.contains(
                    "reject_gap_width|candidate_continuity|reject_continuity",
                    case=False,
                    na=False,
                )
                | has_feature_cols
                | (rank >= 3)
                | strong_dense
            )
            cc = cc[m].copy()
            if len(cc):
                cc["structural_gap_dense_preserve"] = cc.apply(
                    lambda r: int(_structural_gap_is_strong_dense_scan_candidate(r, cfg)),
                    axis=1,
                )
                cc["structural_gap_bad_feature_gap_through"] = cc.apply(
                    lambda r: int(_structural_gap_is_bad_feature_gap_through(r, cfg)), axis=1
                )
            if len(cc):
                parts.append(cc)
    if selected is not None and len(selected):
        parts.append(selected.copy())
    if not parts:
        return pd.DataFrame()
    pool = pd.concat(parts, ignore_index=True, sort=False)
    pool = pool[pd.to_numeric(pool["center_s"], errors="coerce").notna()].copy()
    if len(pool) == 0:
        return pool
    pool["center_s"] = pool["center_s"].astype(float)
    pool["structural_recall_quality"] = pool.apply(
        lambda r: _local_relocation_get_float(
            r, ["structural_recall_quality"], _structural_recall_candidate_quality(r, cfg)
        ),
        axis=1,
    )
    pool["local_relocation_gap_width"] = pool.apply(_local_relocation_gap_width, axis=1)
    pool["local_relocation_gap_density_ratio"] = pool.apply(_local_relocation_density, axis=1)
    pool["local_relocation_start_contrast"] = pool.apply(
        _local_relocation_start_contrast, axis=1
    )
    pool["local_relocation_evidence_score"] = pool.apply(
        lambda r: _local_relocation_boundary_evidence_score(r, ref_len, cfg), axis=1
    )
    min_pool = float(cfg.get("LOCAL_RELOCATION_RELOC_MIN_POOL_SCORE", 4.2))
    if "structural_gap_dense_preserve" not in pool.columns:
        pool["structural_gap_dense_preserve"] = pool.apply(
            lambda r: int(_structural_gap_is_strong_dense_scan_candidate(r, cfg)), axis=1
        )
    if "structural_gap_bad_feature_gap_through" not in pool.columns:
        pool["structural_gap_bad_feature_gap_through"] = pool.apply(
            lambda r: int(_structural_gap_is_bad_feature_gap_through(r, cfg)), axis=1
        )
    pool["structural_gap_dense_preserve"] = (
        pd.to_numeric(pool["structural_gap_dense_preserve"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    pool["structural_gap_bad_feature_gap_through"] = (
        pd.to_numeric(pool["structural_gap_bad_feature_gap_through"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    keep = (
        (pool["local_relocation_evidence_score"] >= min_pool)
        | (pool["local_relocation_gap_density_ratio"].astype(float) <= 0.5)
        | (pool["local_relocation_start_contrast"].astype(float) >= 1.6)
        | pool.get("candidate_class", pd.Series([""] * len(pool), index=pool.index))
        .astype(str)
        .str.contains("observed_gap", case=False, na=False)
        | (
            pd.to_numeric(pool["structural_gap_dense_preserve"], errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        )
    )
    pool = pool[keep].copy()
    if len(pool) == 0:
        return pool
    bin_w = max(1200.0, min(2600.0, 0.045 * float(ref_len)))
    pool["_local_relocation_center_bin"] = np.round(
        pool["center_s"].astype(float) / bin_w
    ).astype(int)
    sort_cols = ["local_relocation_evidence_score", "structural_recall_quality"]
    pool = (
        pool.sort_values(sort_cols, ascending=False)
        .groupby("_local_relocation_center_bin", as_index=False)
        .head(1)
    )
    return pool.sort_values("center_s").reset_index(drop=True)


def _local_relocation_local_objective(
    row: pd.Series, c: float, prev_s: float, next_s: float, ref_len: float, cfg: Dict[str, Any]
) -> float:
    ev = _local_relocation_boundary_evidence_score(row, ref_len, cfg)
    left_len = float(c - prev_s)
    right_len = float(next_s - c)
    min_child = max(
        float(cfg.get("LOCAL_RELOCATION_RELOC_MIN_CHILD_ABS", 12000.0)),
        float(cfg.get("LOCAL_RELOCATION_RELOC_MIN_CHILD_RATIO", 0.32)) * ref_len,
    )
    if left_len < min_child or right_len < min_child:
        return -1000000000.0 + min(left_len, right_len)
    balance = -(abs(left_len - ref_len) + abs(right_len - ref_len)) / max(ref_len, 1.0)
    return (
        float(cfg.get("LOCAL_RELOCATION_RELOC_EVIDENCE_WEIGHT", 1.0)) * ev
        + float(cfg.get("LOCAL_RELOCATION_RELOC_BALANCE_WEIGHT", 2.2)) * balance
    )


def _local_relocation_relocate_selected_boundaries(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    cands: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if (
        not bool(cfg.get("ENABLE_LOCAL_RELOCATION_BOUNDARY_RELOCATION", True))
        or selected is None
        or len(selected) == 0
    ):
        return (selected, pd.DataFrame(), pd.DataFrame())
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    pool = _local_relocation_build_relocation_pool(cands, current, ref_len, cfg)
    if len(pool) == 0:
        return (
            current,
            pd.DataFrame([{"decision": "local_relocation_no_relocation_pool"}]),
            pool,
        )
    logs: List[Dict[str, Any]] = []
    search_win = float(cfg.get("LOCAL_RELOCATION_RELOC_SEARCH_RATIO", 0.68)) * float(ref_len)
    same_tol = float(cfg.get("LOCAL_RELOCATION_RELOC_SAME_CENTER_TOL", 1800.0))
    for it in range(int(cfg.get("LOCAL_RELOCATION_RELOC_MAX_ITERS", 3))):
        changed = False
        current = current.sort_values("center_s").reset_index(drop=True)
        for bi in range(len(current)):
            c_old = float(current.loc[bi, "center_s"])
            prev_s = smin if bi == 0 else float(current.loc[bi - 1, "center_s"])
            next_s = smax if bi == len(current) - 1 else float(current.loc[bi + 1, "center_s"])
            old_row = current.loc[bi]
            old_susp, old_reason = _local_relocation_is_suspicious_selected(
                old_row, ref_len, cfg
            )
            old_obj = _local_relocation_local_objective(
                old_row, c_old, prev_s, next_s, ref_len, cfg
            )
            old_ev = _local_relocation_boundary_evidence_score(old_row, ref_len, cfg)
            lo = max(
                prev_s
                + max(
                    float(cfg.get("LOCAL_RELOCATION_RELOC_MIN_CHILD_ABS", 12000.0)),
                    0.3 * ref_len,
                ),
                c_old - search_win,
            )
            hi = min(
                next_s
                - max(
                    float(cfg.get("LOCAL_RELOCATION_RELOC_MIN_CHILD_ABS", 12000.0)),
                    0.3 * ref_len,
                ),
                c_old + search_win,
            )
            cand = pool[
                (pool["center_s"].astype(float) >= lo) & (pool["center_s"].astype(float) <= hi)
            ].copy()
            if len(cand) == 0:
                logs.append(
                    {
                        "decision": "local_relocation_keep_no_local_pool",
                        "iter": it,
                        "boundary_index": bi,
                        "old_center_s": c_old,
                        "old_suspicious": int(old_susp),
                        "old_reason": old_reason,
                    }
                )
                continue
            cand["_local_relocation_obj"] = cand.apply(
                lambda r: _local_relocation_local_objective(
                    r, float(r["center_s"]), prev_s, next_s, ref_len, cfg
                ),
                axis=1,
            )
            cand["_local_relocation_ev"] = cand.apply(
                lambda r: _local_relocation_boundary_evidence_score(r, ref_len, cfg), axis=1
            )
            cand = cand[np.isfinite(cand["_local_relocation_obj"].astype(float))].copy()
            if len(cand) == 0:
                continue
            best = cand.sort_values(
                ["_local_relocation_obj", "_local_relocation_ev", "structural_recall_quality"],
                ascending=False,
            ).iloc[0]
            c_new = float(best["center_s"])
            best_obj = float(best["_local_relocation_obj"])
            best_ev = float(best["_local_relocation_ev"])
            delta = best_obj - old_obj
            move_dist = abs(c_new - c_old)
            very_narrow = False
            w_old = _local_relocation_gap_width(old_row)
            if np.isfinite(w_old) and w_old <= float(
                cfg.get("LOCAL_RELOCATION_VERY_NARROW_GAP_WIDTH_ABS", 1000.0)
            ):
                very_narrow = True
            if move_dist <= same_tol:
                action = "keep_same_position_best"
                do_replace = False
                margin = np.inf
            else:
                if very_narrow:
                    margin = float(cfg.get("LOCAL_RELOCATION_RELOC_VERY_NARROW_MARGIN", 0.25))
                elif old_susp:
                    margin = float(cfg.get("LOCAL_RELOCATION_RELOC_SUSPICIOUS_MARGIN", 0.75))
                else:
                    margin = float(cfg.get("LOCAL_RELOCATION_RELOC_GENERAL_MARGIN", 2.4))
                do_replace = delta >= margin
                action = "replace" if do_replace else "keep"
            log = {
                "decision": f"local_relocation_relocation_{action}",
                "iter": int(it),
                "boundary_index": int(bi),
                "old_center_s": float(c_old),
                "new_center_s": float(c_new) if do_replace else float(c_old),
                "best_candidate_center_s": float(c_new),
                "move_dist": float(move_dist),
                "old_obj": float(old_obj),
                "best_obj": float(best_obj),
                "delta_obj": float(delta),
                "required_margin": float(margin) if np.isfinite(margin) else np.nan,
                "old_ev": float(old_ev),
                "best_ev": float(best_ev),
                "old_suspicious": int(old_susp),
                "old_reason": old_reason,
                "old_gap_width": (
                    float(_local_relocation_gap_width(old_row))
                    if np.isfinite(_local_relocation_gap_width(old_row))
                    else np.nan
                ),
                "best_gap_width": float(best.get("local_relocation_gap_width", np.nan)),
                "old_density": (
                    float(_local_relocation_density(old_row))
                    if np.isfinite(_local_relocation_density(old_row))
                    else np.nan
                ),
                "best_density": float(best.get("local_relocation_gap_density_ratio", np.nan)),
                "old_start_contrast": float(_local_relocation_start_contrast(old_row)),
                "best_start_contrast": float(
                    best.get("local_relocation_start_contrast", np.nan)
                ),
                "prev_s": float(prev_s),
                "next_s": float(next_s),
            }
            logs.append(log)
            if do_replace:
                new_row = best.copy()
                new_row["center_s"] = float(c_new)
                new_row["selected_stage"] = "local_relocation_boundary_relocated"
                new_row["local_relocation_relocated_from_center_s"] = float(c_old)
                new_row["local_relocation_relocation_reason"] = old_reason
                for col in current.columns:
                    if col not in new_row.index:
                        new_row[col] = np.nan
                for col, value in new_row.items():
                    if isinstance(value, str) and col in current.columns:
                        current[col] = current[col].astype(object)
                current.loc[bi, list(new_row.index)] = new_row
                changed = True
                break
        if not changed:
            break
    current = current.sort_values("center_s").reset_index(drop=True)
    try:
        current2, prune_log = _structural_recall_prune_short_segments(
            work, current, pool, ref_len, cfg
        )
        if len(current2) == 0 and len(current):
            logs.append(
                {"decision": "runtime_prune_all_rollback", "n_preserved": int(len(current))}
            )
            return (
                current.sort_values("center_s").reset_index(drop=True),
                pd.DataFrame(logs),
                pool,
            )
        if prune_log is not None and len(prune_log):
            pl = prune_log.copy()
            pl["decision"] = (
                pl.get("decision", "structural_recall_prune").astype(str)
                + "_after_local_relocation_relocation"
            )
            logs.append(
                {
                    "decision": "local_relocation_after_relocation_prune_summary",
                    "n_prune_rows": int(len(pl)),
                }
            )
            log_df = pd.concat([pd.DataFrame(logs), pl], ignore_index=True, sort=False)
            return (current2.sort_values("center_s").reset_index(drop=True), log_df, pool)
    except Exception as e:
        logs.append(
            {"decision": "local_relocation_after_relocation_prune_failed", "error": repr(e)}
        )
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs), pool)


def select_sequence_with_local_relocation(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base_selected, base_logs, mode_info = select_sequence_with_structural_recall(
        work, cands, cfg
    )
    ref_len = None
    if isinstance(mode_info, dict):
        for key in ["typical_len", "feature_gap_aux_len", "ref_len"]:
            v = mode_info.get(key)
            try:
                if v is not None and np.isfinite(float(v)) and (float(v) > 1):
                    ref_len = float(v)
                    break
            except Exception:
                pass
    if ref_len is None:
        ref_len, ref_meta = _structural_recall_aux_len(work, cfg)
    else:
        ref_meta = {"local_relocation_ref_len_source": "mode_info"}
    selected = (
        base_selected.copy().sort_values("center_s").reset_index(drop=True)
        if base_selected is not None and len(base_selected)
        else pd.DataFrame()
    )
    reloc_log = pd.DataFrame()
    reloc_pool = pd.DataFrame()
    if len(selected):
        selected, reloc_log, reloc_pool = _local_relocation_relocate_selected_boundaries(
            work, selected, cands, float(ref_len), cfg
        )
        if len(selected):
            selected["selected_stage"] = selected.get(
                "selected_stage", "local_relocation_fullcover_boundary_relocation"
            )
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info.update(ref_meta)
    mode_info["mode"] = "local_relocation_fullcover_boundary_relocation"
    mode_info["typical_len"] = float(ref_len)
    mode_info["feature_gap_aux_len"] = float(ref_len)
    mode_info["local_relocation_n_selected_after_relocation"] = int(len(selected))
    logs: List[pd.DataFrame] = []
    logs.append(
        pd.DataFrame(
            [
                {
                    "decision": "local_relocation_selection_summary",
                    "n_base_selected_from_structural_recall": (
                        int(len(base_selected)) if base_selected is not None else 0
                    ),
                    "n_selected_after_local_relocation": int(len(selected)),
                    "n_relocation_pool": int(len(reloc_pool)),
                    "ref_len": float(ref_len),
                    "mode": mode_info["mode"],
                }
            ]
        )
    )
    if base_logs is not None and len(base_logs):
        logs.append(base_logs)
    if reloc_log is not None and len(reloc_log):
        logs.append(reloc_log)
    if (
        bool(cfg.get("LOCAL_RELOCATION_EXPORT_RELOCATION_POOL_IN_TRACE", True))
        and reloc_pool is not None
        and len(reloc_pool)
    ):
        cols = [
            c
            for c in [
                "center_s",
                "local_relocation_evidence_score",
                "structural_recall_quality",
                "local_relocation_gap_density_ratio",
                "local_relocation_start_contrast",
                "local_relocation_gap_width",
                "candidate_class",
                "kind",
                "structural_gap_dense_preserve",
                "structural_gap_bad_feature_gap_through",
                "start_block_decision",
                "density_valley_decision",
            ]
            if c in reloc_pool.columns
        ]
        plog = reloc_pool[cols].copy()
        plog["decision"] = "local_relocation_relocation_pool_candidate"
        logs.append(plog)
    out_logs = pd.concat(logs, ignore_index=True, sort=False) if logs else pd.DataFrame()
    return (selected.reset_index(drop=True), out_logs, mode_info)


def _structural_gap_num(row: pd.Series, names: List[str], default: float = np.nan) -> float:
    """Read the first available numeric field from a candidate record."""
    return _local_relocation_get_float(row, names, default)


def _structural_gap_text(row: pd.Series, names: List[str], default: str = "") -> str:
    return _local_relocation_get_str(row, names, default)


def _structural_gap_is_feature_gap(row: pd.Series) -> bool:
    kind = _structural_gap_text(row, ["kind"], "").lower()
    cls = _structural_gap_text(row, ["candidate_class"], "").lower()
    stage = _structural_gap_text(row, ["selected_stage"], "").lower()
    return "feature_gap" in kind or "feature" in cls or "feature_gap" in stage


def _structural_gap_is_strong_dense_scan_candidate(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Require synchronous row breaks, a strong density drop and limited continuity."""
    if not bool(cfg.get("ENABLE_STRUCTURAL_GAP_DENSE_SCAN_PRESERVE", True)):
        return False
    kind = _structural_gap_text(row, ["kind"], "").lower()
    cls = _structural_gap_text(row, ["candidate_class"], "").lower()
    looks_dense = (
        kind == "dense_scan" or "dense" in kind or ("strong" in cls and "feature" not in cls)
    )
    if not looks_dense:
        return False
    if bool(cfg.get("CANDIDATE_FILTER_DENSE_REJECT_C_WEAK", True)):
        if "c_weak" in cls or cls.strip() == "c_weak_evidence":
            return False
    score = _structural_gap_num(
        row,
        ["score", "structural_recall_quality", "local_relocation_evidence_score"],
        -1000000000.0,
    )
    n_strong = _structural_gap_num(row, ["n_strong_break"], 0.0)
    n_weak = _structural_gap_num(row, ["n_weak_break"], 0.0)
    joint = _structural_gap_num(row, ["joint_drop"], 999.0)
    cont = _structural_gap_num(row, ["continuous_rows"], 999.0)
    cluster_size = _structural_gap_num(row, ["cluster_size"], 999.0)
    if not np.isfinite(cluster_size):
        cluster_size = 999.0
    return bool(
        np.isfinite(score)
        and score >= float(cfg.get("STRUCTURAL_GAP_DENSE_MIN_SCORE", 9.5))
        and (n_strong >= float(cfg.get("STRUCTURAL_GAP_DENSE_MIN_STRONG", 2)))
        and (n_weak >= float(cfg.get("STRUCTURAL_GAP_DENSE_MIN_WEAK", 3)))
        and (joint <= float(cfg.get("STRUCTURAL_GAP_DENSE_MAX_JOINT_DROP", 0.22)))
        and (cont <= float(cfg.get("STRUCTURAL_GAP_DENSE_MAX_CONTINUOUS_ROWS", 1)))
        and (cluster_size >= float(cfg.get("STRUCTURAL_GAP_DENSE_MIN_CLUSTER_SIZE", 3)))
    )


def _structural_gap_is_bad_feature_gap_through(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Identify sparse within-plot intervals that retain multirow continuity."""
    if not bool(cfg.get("ENABLE_STRUCTURAL_GAP_FEATURE_THROUGH_VETO", True)):
        return False
    if not _structural_gap_is_feature_gap(row):
        return False
    dens = _structural_gap_num(
        row,
        [
            "structural_recall_gap_density_ratio",
            "local_relocation_gap_density_ratio",
            "balanced_relocation_gap_density_ratio",
            "gap_density_ratio",
            "start_block_gap_density_ratio",
            "feature_gap_gap_density_ratio",
        ],
        np.nan,
    )
    active3 = _structural_gap_num(
        row, ["active3_frac", "start_block_active3_frac", "feature_gap_active3_frac"], np.nan
    )
    through = _structural_gap_num(row, ["continuity_through_rows"], 0.0)
    max_consec = _structural_gap_num(row, ["continuity_max_consec_active_frac"], 0.0)
    start = _structural_gap_num(
        row,
        [
            "structural_recall_start_contrast",
            "local_relocation_start_contrast",
            "balanced_relocation_start_contrast",
            "start_contrast",
            "start_block_start_contrast",
            "feature_calibration_post_start_contrast",
        ],
        0.0,
    )
    width = _structural_gap_num(
        row,
        [
            "structural_recall_gap_width",
            "local_relocation_gap_width",
            "balanced_relocation_gap_width",
            "gap_width",
            "start_block_gap_width",
            "feature_gap_gap_width",
            "width",
        ],
        np.inf,
    )
    dens_bad = np.isfinite(dens) and dens >= float(
        cfg.get("STRUCTURAL_GAP_FEATURE_BAD_DENSITY_RATIO", 0.52)
    )
    active_bad = np.isfinite(active3) and active3 >= float(
        cfg.get("STRUCTURAL_GAP_FEATURE_BAD_ACTIVE3_FRAC", 0.85)
    )
    through_bad = through >= float(cfg.get("STRUCTURAL_GAP_FEATURE_BAD_THROUGH_ROWS", 3))
    consec_bad = max_consec >= float(
        cfg.get("STRUCTURAL_GAP_FEATURE_BAD_MAX_CONSEC_ACTIVE_FRAC", 0.8)
    )
    start_weak = start < float(cfg.get("STRUCTURAL_GAP_FEATURE_BAD_START_CONTRAST", 1.75))
    width_ok = not np.isfinite(width) or width >= float(
        cfg.get("STRUCTURAL_GAP_FEATURE_BAD_MIN_GAP_WIDTH", 0.0)
    )
    return bool(
        width_ok and start_weak and dens_bad and (active_bad or through_bad or consec_bad)
    )


def _balanced_relocation_child_lengths(
    prev_s: float, c: float, next_s: float
) -> Tuple[float, float]:
    return (float(c - prev_s), float(next_s - c))


def _balanced_relocation_balance_penalty(
    prev_s: float, c: float, next_s: float, ref_len: float
) -> float:
    l1, l2 = _balanced_relocation_child_lengths(prev_s, c, next_s)
    return (abs(l1 - ref_len) + abs(l2 - ref_len)) / max(ref_len, 1.0)


def _balanced_relocation_is_pair_imbalanced(
    prev_s: float, c: float, next_s: float, ref_len: float, cfg: Dict[str, Any]
) -> Tuple[bool, str]:
    l1, l2 = _balanced_relocation_child_lengths(prev_s, c, next_s)
    short = min(l1, l2)
    long = max(l1, l2)
    reasons: List[str] = []
    if short < float(cfg.get("BALANCED_RELOCATION_IMBALANCED_SHORT_RATIO", 0.76)) * ref_len:
        reasons.append("one_child_too_short")
    if long > float(cfg.get("BALANCED_RELOCATION_IMBALANCED_LONG_RATIO", 1.24)) * ref_len:
        reasons.append("one_child_too_long")
    if short > 1 and long / short > float(
        cfg.get("BALANCED_RELOCATION_IMBALANCE_RATIO_TRIGGER", 1.55)
    ):
        reasons.append("child_length_ratio_imbalanced")
    return (len(reasons) > 0, ";".join(reasons))


def _balanced_relocation_local_objective(
    row: pd.Series, c: float, prev_s: float, next_s: float, ref_len: float, cfg: Dict[str, Any]
) -> float:
    l1, l2 = _balanced_relocation_child_lengths(prev_s, c, next_s)
    min_child = max(
        float(cfg.get("BALANCED_RELOCATION_RELOC_MIN_CHILD_ABS", 11500.0)),
        float(cfg.get("BALANCED_RELOCATION_RELOC_MIN_CHILD_RATIO", 0.3)) * ref_len,
    )
    if l1 < min_child or l2 < min_child:
        return -1000000000.0 + min(l1, l2)
    ev = _local_relocation_boundary_evidence_score(row, ref_len, cfg)
    bal = -_balanced_relocation_balance_penalty(prev_s, c, next_s, ref_len)
    return (
        float(cfg.get("BALANCED_RELOCATION_OBJECTIVE_EVIDENCE_WEIGHT", 1.0)) * ev
        + float(cfg.get("BALANCED_RELOCATION_OBJECTIVE_BALANCE_WEIGHT", 5.6)) * bal
    )


def _balanced_relocation_pool_candidate_ok(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    if bool(
        cfg.get("STRUCTURAL_GAP_FEATURE_BAD_FORBID_AS_RELOC_TARGET", True)
    ) and _structural_gap_is_bad_feature_gap_through(row, cfg):
        return False
    if _structural_gap_is_strong_dense_scan_candidate(row, cfg):
        return True
    cls = _local_relocation_get_str(row, ["candidate_class"], "")
    if cls == "D_low":
        return False
    return True


def _balanced_relocation_relocate_all_boundaries(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    cands: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if (
        selected is None
        or len(selected) == 0
        or (not bool(cfg.get("ENABLE_BALANCED_RELOCATION_COMPLETE_BOUNDARY_RELOCATION", True)))
    ):
        return (selected, pd.DataFrame(), pd.DataFrame())
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    pool = _local_relocation_build_relocation_pool(cands, current, ref_len, cfg)
    if pool is None or len(pool) == 0:
        return (
            current,
            pd.DataFrame([{"decision": "balanced_relocation_no_relocation_pool"}]),
            pd.DataFrame(),
        )
    pool = pool.copy()
    pool["balanced_relocation_evidence_score"] = pool.apply(
        lambda r: _local_relocation_boundary_evidence_score(r, ref_len, cfg), axis=1
    )
    pool["balanced_relocation_gap_width"] = pool.apply(_local_relocation_gap_width, axis=1)
    pool["balanced_relocation_gap_density_ratio"] = pool.apply(
        _local_relocation_density, axis=1
    )
    pool["balanced_relocation_start_contrast"] = pool.apply(
        _local_relocation_start_contrast, axis=1
    )
    logs: List[Dict[str, Any]] = []
    search_win = float(cfg.get("BALANCED_RELOCATION_RELOC_SEARCH_RATIO", 0.82)) * ref_len
    same_tol = float(cfg.get("BALANCED_RELOCATION_RELOC_SAME_CENTER_TOL", 1200.0))
    min_child = max(
        float(cfg.get("BALANCED_RELOCATION_RELOC_MIN_CHILD_ABS", 11500.0)),
        float(cfg.get("BALANCED_RELOCATION_RELOC_MIN_CHILD_RATIO", 0.3)) * ref_len,
    )
    for pass_id in range(int(cfg.get("BALANCED_RELOCATION_RELOC_MAX_PASSES", 8))):
        changed = False
        current = current.sort_values("center_s").reset_index(drop=True)
        centers = current["center_s"].astype(float).to_numpy()
        for bi in range(len(current)):
            centers = current["center_s"].astype(float).to_numpy()
            c_old = float(centers[bi])
            prev_s = smin if bi == 0 else float(centers[bi - 1])
            next_s = smax if bi == len(centers) - 1 else float(centers[bi + 1])
            if next_s - prev_s < 2 * min_child:
                logs.append(
                    {
                        "decision": "balanced_relocation_skip_neighbor_span_too_short",
                        "pass": int(pass_id),
                        "boundary_index": int(bi),
                        "old_center_s": c_old,
                        "prev_s": prev_s,
                        "next_s": next_s,
                    }
                )
                continue
            old_row = current.loc[bi]
            old_susp, old_reason = _local_relocation_is_suspicious_selected(
                old_row, ref_len, cfg
            )
            old_imb, old_imb_reason = _balanced_relocation_is_pair_imbalanced(
                prev_s, c_old, next_s, ref_len, cfg
            )
            old_ev = _local_relocation_boundary_evidence_score(old_row, ref_len, cfg)
            old_obj = _balanced_relocation_local_objective(
                old_row, c_old, prev_s, next_s, ref_len, cfg
            )
            old_bal_pen = _balanced_relocation_balance_penalty(prev_s, c_old, next_s, ref_len)
            old_width = _local_relocation_gap_width(old_row)
            very_narrow = np.isfinite(old_width) and old_width <= float(
                cfg.get("LOCAL_RELOCATION_VERY_NARROW_GAP_WIDTH_ABS", 1000.0)
            )
            lo = max(prev_s + min_child, c_old - search_win)
            hi = min(next_s - min_child, c_old + search_win)
            sub = pool[
                (pool["center_s"].astype(float) >= lo) & (pool["center_s"].astype(float) <= hi)
            ].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "decision": "balanced_relocation_keep_no_local_pool",
                        "pass": int(pass_id),
                        "boundary_index": int(bi),
                        "old_center_s": c_old,
                        "old_suspicious": int(old_susp),
                        "old_imbalanced_pair": int(old_imb),
                        "old_reason": old_reason,
                        "old_imbalance_reason": old_imb_reason,
                    }
                )
                continue
            sub = sub[
                sub.apply(
                    lambda r: _balanced_relocation_pool_candidate_ok(r, cfg)
                    or abs(float(r["center_s"]) - c_old) <= same_tol,
                    axis=1,
                )
            ].copy()
            if len(sub) == 0:
                continue
            sub["_balanced_relocation_obj"] = sub.apply(
                lambda r: _balanced_relocation_local_objective(
                    r, float(r["center_s"]), prev_s, next_s, ref_len, cfg
                ),
                axis=1,
            )
            sub["_balanced_relocation_ev"] = sub.apply(
                lambda r: _local_relocation_boundary_evidence_score(r, ref_len, cfg), axis=1
            )
            sub["_balanced_relocation_bal_pen"] = (
                sub["center_s"]
                .astype(float)
                .apply(
                    lambda cc: _balanced_relocation_balance_penalty(
                        prev_s, float(cc), next_s, ref_len
                    )
                )
            )
            sub = sub[np.isfinite(sub["_balanced_relocation_obj"].astype(float))].copy()
            if len(sub) == 0:
                continue
            best = sub.sort_values(
                [
                    "_balanced_relocation_obj",
                    "_balanced_relocation_ev",
                    "balanced_relocation_gap_width",
                ],
                ascending=False,
            ).iloc[0]
            c_new = float(best["center_s"])
            move_dist = abs(c_new - c_old)
            best_obj = float(best["_balanced_relocation_obj"])
            best_ev = float(best["_balanced_relocation_ev"])
            best_bal_pen = float(best["_balanced_relocation_bal_pen"])
            best_width = (
                float(best.get("balanced_relocation_gap_width", np.nan))
                if np.isfinite(best.get("balanced_relocation_gap_width", np.nan))
                else np.nan
            )
            delta_obj = best_obj - old_obj
            bal_gain = old_bal_pen - best_bal_pen
            if move_dist <= same_tol:
                logs.append(
                    {
                        "decision": "balanced_relocation_keep_same_position_best",
                        "pass": int(pass_id),
                        "boundary_index": int(bi),
                        "old_center_s": c_old,
                        "best_candidate_center_s": c_new,
                        "old_obj": old_obj,
                        "best_obj": best_obj,
                        "old_ev": old_ev,
                        "best_ev": best_ev,
                        "old_balance_penalty": old_bal_pen,
                        "best_balance_penalty": best_bal_pen,
                        "old_suspicious": int(old_susp),
                        "old_imbalanced_pair": int(old_imb),
                        "old_reason": old_reason,
                        "old_imbalance_reason": old_imb_reason,
                    }
                )
                continue
            if very_narrow:
                margin = float(cfg.get("BALANCED_RELOCATION_VERY_NARROW_MARGIN", 0.05))
            elif old_imb:
                margin = float(cfg.get("BALANCED_RELOCATION_IMBALANCED_MARGIN", 0.05))
            elif old_susp:
                margin = float(cfg.get("BALANCED_RELOCATION_SUSPICIOUS_MARGIN", 0.35))
            else:
                margin = float(cfg.get("BALANCED_RELOCATION_GENERAL_MARGIN", 2.2))
            reason_accept = ""
            do_replace = False
            if delta_obj >= margin:
                do_replace = True
                reason_accept = "objective_improved"
            elif old_imb and bal_gain >= float(
                cfg.get("BALANCED_RELOCATION_BALANCE_GAIN_ACCEPT", 0.22)
            ):
                max_loss = float(
                    cfg.get("BALANCED_RELOCATION_MAX_EVIDENCE_LOSS_FOR_BALANCE_FIX", 3.2)
                )
                if _structural_gap_is_strong_dense_scan_candidate(best, cfg):
                    max_loss = max(
                        max_loss,
                        float(
                            cfg.get("STRUCTURAL_GAP_DENSE_BALANCE_FIX_MAX_EVIDENCE_LOSS", 5.5)
                        ),
                    )
                min_alt_w = float(
                    cfg.get("BALANCED_RELOCATION_MIN_ALT_WIDTH_FOR_BALANCE_FIX", 1600.0)
                )
                if best_ev >= old_ev - max_loss and (
                    not np.isfinite(best_width)
                    or best_width >= min_alt_w
                    or _structural_gap_is_strong_dense_scan_candidate(best, cfg)
                ):
                    do_replace = True
                    reason_accept = "fix_short_long_pair_balance"
            log = {
                "decision": (
                    "balanced_relocation_relocation_replace"
                    if do_replace
                    else "balanced_relocation_relocation_keep"
                ),
                "pass": int(pass_id),
                "boundary_index": int(bi),
                "old_center_s": c_old,
                "new_center_s": c_new if do_replace else c_old,
                "best_candidate_center_s": c_new,
                "move_dist": move_dist,
                "old_obj": old_obj,
                "best_obj": best_obj,
                "delta_obj": delta_obj,
                "required_margin": margin,
                "old_ev": old_ev,
                "best_ev": best_ev,
                "old_balance_penalty": old_bal_pen,
                "best_balance_penalty": best_bal_pen,
                "balance_gain": bal_gain,
                "old_suspicious": int(old_susp),
                "old_imbalanced_pair": int(old_imb),
                "old_reason": old_reason,
                "old_imbalance_reason": old_imb_reason,
                "old_gap_width": float(old_width) if np.isfinite(old_width) else np.nan,
                "best_gap_width": best_width,
                "old_density": (
                    float(_local_relocation_density(old_row))
                    if np.isfinite(_local_relocation_density(old_row))
                    else np.nan
                ),
                "best_density": (
                    float(best.get("balanced_relocation_gap_density_ratio", np.nan))
                    if np.isfinite(best.get("balanced_relocation_gap_density_ratio", np.nan))
                    else np.nan
                ),
                "old_start_contrast": float(_local_relocation_start_contrast(old_row)),
                "best_start_contrast": (
                    float(best.get("balanced_relocation_start_contrast", np.nan))
                    if np.isfinite(best.get("balanced_relocation_start_contrast", np.nan))
                    else np.nan
                ),
                "prev_s": prev_s,
                "next_s": next_s,
                "accept_reason": reason_accept,
            }
            logs.append(log)
            if do_replace:
                new_row = current.loc[bi].copy()
                for col in current.columns:
                    if col in best.index:
                        new_row[col] = best[col]
                new_row["center_s"] = c_new
                new_row["selected_stage"] = "balanced_relocation_boundary_relocated"
                new_row["local_relocation_relocated_from_center_s"] = c_old
                new_row["local_relocation_relocation_reason"] = str(old_reason) + (
                    ";" + old_imb_reason if old_imb_reason else ""
                )
                new_row["balanced_relocation_relocated_from_center_s"] = c_old
                new_row["balanced_relocation_relocation_accept_reason"] = reason_accept
                for col, value in new_row.items():
                    if isinstance(value, str) and col in current.columns:
                        current[col] = current[col].astype(object)
                current.loc[bi, :] = new_row
                changed = True
        if not changed:
            break
    current = current.sort_values("center_s").reset_index(drop=True)
    return (current, pd.DataFrame(logs), pool.sort_values("center_s").reset_index(drop=True))


def select_sequence_with_balanced_relocation(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base_selected, base_logs, mode_info = select_sequence_with_local_relocation(
        work, cands, cfg
    )
    ref_len = None
    if isinstance(mode_info, dict):
        for key in ["typical_len", "feature_gap_aux_len", "ref_len"]:
            v = mode_info.get(key)
            try:
                if v is not None and np.isfinite(float(v)) and (float(v) > 1):
                    ref_len = float(v)
                    break
            except Exception:
                pass
    if ref_len is None:
        ref_len, ref_meta = _structural_recall_aux_len(work, cfg)
    else:
        ref_meta = {"balanced_relocation_ref_len_source": "mode_info"}
    selected = (
        base_selected.copy().sort_values("center_s").reset_index(drop=True)
        if base_selected is not None and len(base_selected)
        else pd.DataFrame()
    )
    balanced_relocation_log = pd.DataFrame()
    balanced_relocation_pool = pd.DataFrame()
    if len(selected):
        selected, balanced_relocation_log, balanced_relocation_pool = (
            _balanced_relocation_relocate_all_boundaries(
                work, selected, cands, float(ref_len), cfg
            )
        )
        if len(selected):
            selected["selected_stage"] = selected.get(
                "selected_stage", "balanced_relocation_fullcover_relocation_complete_scan"
            )
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info.update(ref_meta)
    mode_info["mode"] = "balanced_relocation_fullcover_relocation_complete_scan"
    mode_info["typical_len"] = float(ref_len)
    mode_info["feature_gap_aux_len"] = float(ref_len)
    mode_info["balanced_relocation_n_selected_after_complete_relocation"] = int(len(selected))
    logs: List[pd.DataFrame] = []
    logs.append(
        pd.DataFrame(
            [
                {
                    "decision": "balanced_relocation_selection_summary",
                    "n_base_selected_from_local_relocation": (
                        int(len(base_selected)) if base_selected is not None else 0
                    ),
                    "n_selected_after_balanced_relocation": int(len(selected)),
                    "n_relocation_pool": (
                        int(len(balanced_relocation_pool))
                        if balanced_relocation_pool is not None
                        else 0
                    ),
                    "ref_len": float(ref_len),
                    "mode": mode_info["mode"],
                }
            ]
        )
    )
    if base_logs is not None and len(base_logs):
        logs.append(base_logs)
    if balanced_relocation_log is not None and len(balanced_relocation_log):
        logs.append(balanced_relocation_log)
    if (
        bool(cfg.get("BALANCED_RELOCATION_EXPORT_RELOCATION_POOL_IN_TRACE", True))
        and balanced_relocation_pool is not None
        and len(balanced_relocation_pool)
    ):
        cols = [
            c
            for c in [
                "center_s",
                "balanced_relocation_evidence_score",
                "structural_recall_quality",
                "balanced_relocation_gap_density_ratio",
                "balanced_relocation_start_contrast",
                "balanced_relocation_gap_width",
                "candidate_class",
                "kind",
                "structural_gap_dense_preserve",
                "structural_gap_bad_feature_gap_through",
                "start_block_decision",
                "density_valley_decision",
            ]
            if c in balanced_relocation_pool.columns
        ]
        plog = balanced_relocation_pool[cols].copy()
        plog["decision"] = "balanced_relocation_relocation_pool_candidate"
        logs.append(plog)
    out_logs = pd.concat(logs, ignore_index=True, sort=False) if logs else pd.DataFrame()
    return (selected.reset_index(drop=True), out_logs, mode_info)


def run_partition_pipeline():
    out_dir = ensure_dir(Path(CONFIG["OUTPUT_DIR"]))
    df = load_points(CONFIG["INPUT_CSV"])
    rect_df, meta = coarse_rectify_and_rows(df, CONFIG)
    rect_df.to_csv(out_dir / "01_rectified_points_all.csv", index=False, encoding="utf-8-sig")
    work = rect_df[rect_df["keep_for_partition"]].copy().reset_index(drop=True)
    work.to_csv(
        out_dir / "02_rectified_points_for_partition.csv", index=False, encoding="utf-8-sig"
    )
    all_cands, clustered = build_all_candidates(work, CONFIG)
    all_cands.to_csv(
        out_dir / "03_boundary_candidates_all.csv", index=False, encoding="utf-8-sig"
    )
    clustered = classify_boundary_candidates(clustered, CONFIG)
    clustered.to_csv(
        out_dir / "03b_boundary_candidates_clustered.csv", index=False, encoding="utf-8-sig"
    )
    CONFIG["_RUNTIME_ORIGINAL_CANDIDATES"] = clustered.copy()
    clustered, observed_proposal_obs_gap_log = build_observed_gap_proposals(
        work, clustered, CONFIG
    )
    observed_proposal_obs_gap_log.to_csv(
        out_dir / "03d_observed_proposal_observed_gap_interval_proposals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clustered = classify_boundary_candidates(clustered, CONFIG)
    clustered, feature_gap_feature_gap_log = build_feature_gap_proposals(
        work, clustered, CONFIG
    )
    feature_gap_feature_gap_log.to_csv(
        out_dir / "03e_feature_gap_feature_gap_interval_proposals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clustered = classify_boundary_candidates(clustered, CONFIG)
    clustered.to_csv(
        out_dir / "03c_boundary_candidates_classified.csv", index=False, encoding="utf-8-sig"
    )
    if bool(CONFIG.get("ENABLE_FEATURE_GAP_FEATURE_GAP_CORE", True)):
        selected_center, dp_trace, mode_info = select_boundaries(work, clustered, CONFIG)
        dp_trace.to_csv(
            out_dir / "04a_feature_gap_feature_gap_primary_dp_trace.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_feature_gap_segment_summary(out_dir, work, selected_center, mode_info)
    elif bool(CONFIG.get("ENABLE_PERIODIC_SEQUENCE_SEQUENCE_DP", True)):
        selected_center, dp_trace, mode_info = select_periodic_boundary_sequence(
            work, clustered, CONFIG
        )
        dp_trace.to_csv(
            out_dir / "04a_periodic_sequence_sequence_dp_trace.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selected_center, periodic_sequence_recall_log = recover_observed_gaps_in_long_segments(
            work, clustered, selected_center, mode_info, CONFIG
        )
        periodic_sequence_recall_log.to_csv(
            out_dir / "04b_periodic_sequence_observed_gap_recall_in_long_segments.csv",
            index=False,
            encoding="utf-8-sig",
        )
        periodic_sequence_write_sequence_summary(
            out_dir, work, selected_center, mode_info, CONFIG
        )
    else:
        selected_center, dp_trace, mode_info = run_graph_dp_select_boundaries(
            work, clustered, CONFIG
        )
        dp_trace.to_csv(out_dir / "04a_graph_dp_trace.csv", index=False, encoding="utf-8-sig")
        selected_center, refine_log = refine_selected_boundaries_locally(
            work, clustered, selected_center, mode_info, CONFIG
        )
        if refine_log is not None and len(refine_log):
            refine_log.to_csv(
                out_dir / "04b_graph_dp_local_refine_log.csv", index=False, encoding="utf-8-sig"
            )
        selected_center, reopt_log1 = reoptimize_segments_with_residual_candidates(
            work, clustered, selected_center, mode_info, CONFIG
        )
        if reopt_log1 is not None and len(reopt_log1):
            reopt_log1.to_csv(
                out_dir / "04c_segment_reoptimization_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, validation_log = validate_boundary_sequence_locally(
            work, clustered, selected_center, mode_info, CONFIG
        )
        if validation_log is not None and len(validation_log):
            validation_log.to_csv(
                out_dir / "04d_final_boundary_sequence_validation_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, reopt_log2 = reoptimize_segments_with_residual_candidates(
            work, clustered, selected_center, mode_info, CONFIG
        )
        if reopt_log2 is not None and len(reopt_log2):
            reopt_log2.to_csv(
                out_dir / "04e_segment_reoptimization_after_validation_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, weak_merge_log = merge_weak_oversplit_boundaries(
            work, selected_center, CONFIG
        )
        if weak_merge_log is not None and len(weak_merge_log):
            weak_merge_log.to_csv(
                out_dir / "04f_final_weak_oversplit_merge_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, false_split_log = prune_false_splits_by_child_segment_quality(
            work, selected_center, CONFIG
        )
        if false_split_log is not None and len(false_split_log):
            false_split_log.to_csv(
                out_dir / "04f2_final_false_split_recheck_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, cluster_repair_log = repair_boundary_clusters_by_local_quality(
            work, clustered, selected_center, CONFIG
        )
        if cluster_repair_log is not None and len(cluster_repair_log):
            cluster_repair_log.to_csv(
                out_dir / "04f3_final_cluster_compression_recheck_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, false_split_log2 = prune_false_splits_by_child_segment_quality(
            work, selected_center, CONFIG
        )
        if false_split_log2 is not None and len(false_split_log2):
            false_split_log2.to_csv(
                out_dir / "04f4_final_false_split_recheck_after_cluster_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        _before_residual_recall = (
            selected_center.copy().sort_values("center_s").reset_index(drop=True)
            if selected_center is not None
            else pd.DataFrame()
        )
        _cand_after_recall, residual_recall_log = recall_residual_boundaries_in_long_segments(
            work, clustered, selected_center, mode_info, CONFIG
        )
        if residual_recall_log is not None and len(residual_recall_log):
            residual_recall_log.to_csv(
                out_dir / "04f5_residual_long_segment_recall_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        _cand_after_recall, false_split_log3 = prune_false_splits_by_child_segment_quality(
            work, _cand_after_recall, CONFIG
        )
        if false_split_log3 is not None and len(false_split_log3):
            false_split_log3.to_csv(
                out_dir / "04f6_false_split_after_residual_recall_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, recall_gate_log = _recall_gate_accept_update_with_gate(
            work,
            clustered,
            _before_residual_recall,
            _cand_after_recall,
            CONFIG,
            stage="anchor_recall_residual_recall_plus_false_split",
        )
        if recall_gate_log is not None and len(recall_gate_log):
            recall_gate_log.to_csv(
                out_dir / "04f6b_safe_gate_after_residual_recall.csv",
                index=False,
                encoding="utf-8-sig",
            )
        _before_terminal_recall = (
            selected_center.copy().sort_values("center_s").reset_index(drop=True)
            if selected_center is not None
            else pd.DataFrame()
        )
        _cand_after_terminal, terminal_recall_log = recall_terminal_edge_partial_boundary(
            work, clustered, selected_center, mode_info, CONFIG
        )
        if terminal_recall_log is not None and len(terminal_recall_log):
            terminal_recall_log.to_csv(
                out_dir / "04f7_terminal_edge_partial_recall_log.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selected_center, terminal_gate_log = _recall_gate_accept_update_with_gate(
            work,
            clustered,
            _before_terminal_recall,
            _cand_after_terminal,
            CONFIG,
            stage="anchor_recall_terminal_edge_partial_recall",
        )
        if terminal_gate_log is not None and len(terminal_gate_log):
            terminal_gate_log.to_csv(
                out_dir / "04f7b_safe_gate_after_terminal_recall.csv",
                index=False,
                encoding="utf-8-sig",
            )
        adaptive_guard_diag, adaptive_guard_diag_df = diagnose_partition_state(
            work, selected_center, mode_info, CONFIG
        )
        adaptive_guard_diag_df.to_csv(
            out_dir / "04f8a_partition_state_diagnosis.csv", index=False, encoding="utf-8-sig"
        )
        if (
            bool(CONFIG.get("ENABLE_ADAPTIVE_GUARD_CONSERVATIVE_ADAPTIVE_GUARD", True))
            and int(adaptive_guard_diag.get("over_split_risk", 0)) == 1
        ):
            selected_center, adaptive_guard_merge_log = merge_oversplit_boundaries(
                work, selected_center, mode_info, CONFIG
            )
        else:
            adaptive_guard_merge_log = pd.DataFrame(
                [
                    {
                        **adaptive_guard_diag,
                        "stage": "adaptive_guard_safe_merge",
                        "decision": "skip_guard_not_triggered",
                    }
                ]
            )
        adaptive_guard_merge_log.to_csv(
            out_dir / "04f8e_conservative_oversplit_merge_gate.csv",
            index=False,
            encoding="utf-8-sig",
        )
        selected_center, observed_sequence_obs_seq_log = refine_observed_gap_sequence(
            work, clustered, selected_center, mode_info, CONFIG
        )
        observed_sequence_obs_seq_log.to_csv(
            out_dir / "04f9m_observed_sequence_observed_gap_sequence_fix_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
    residual_diag = diagnose_final_partition(work, selected_center, clustered, CONFIG)
    if residual_diag is not None and len(residual_diag):
        residual_diag.to_csv(
            out_dir / "04g_final_partition_residual_anchor_diagnosis.csv",
            index=False,
            encoding="utf-8-sig",
        )
    unselected_strong = diagnose_unselected_strong_candidates(
        work, selected_center, clustered, CONFIG
    )
    if unselected_strong is not None and len(unselected_strong):
        unselected_strong.to_csv(
            out_dir / "04h_unselected_strong_candidates_need_review.csv",
            index=False,
            encoding="utf-8-sig",
        )
    selected = fit_boundary_edges(rect_df, selected_center, CONFIG)
    selected.to_csv(
        out_dir / "04_dp_selected_boundaries.csv", index=False, encoding="utf-8-sig"
    )
    plot_df, plot_points = build_plot_boundaries_from_edges(rect_df, selected)
    edge_report = refine_partition_edges(out_dir, plot_df, plot_points, CONFIG)
    plot_df.to_csv(out_dir / "05_plot_boundaries.csv", index=False, encoding="utf-8-sig")
    plot_points.to_csv(out_dir / "06_plot_points.csv", index=False, encoding="utf-8-sig")
    evidence = build_evidence_curve(all_cands, CONFIG)
    evidence.to_csv(out_dir / "07_evidence_curve.csv", index=False, encoding="utf-8-sig")
    draw_evidence_curve_png(evidence, selected, out_dir / "08_evidence_curve.png", CONFIG)
    partition_stats = export_partition_outputs(out_dir, plot_df, plot_points, CONFIG)
    if not (out_dir / "09_points_with_dp_boundaries.png").exists():
        vis = draw_points_and_plots(plot_points, plot_df, selected, CONFIG)
        _write_image_unicode(str(out_dir / "09_points_with_dp_boundaries.png"), vis)
    summary = {
        "n_total_points": int(len(rect_df)),
        "n_used_for_partition": int(len(work)),
        "n_candidates_all": int(len(all_cands)),
        "n_candidates_clustered": int(len(clustered)),
        "n_graph_strong_candidates": (
            int((clustered["candidate_class_rank"].astype(int) >= 3).sum())
            if "candidate_class_rank" in clustered.columns
            else 0
        ),
        "n_selected_boundaries": int(len(selected)),
        "n_plots": int(len(plot_df)),
        "used_existing_row_id": int(meta.get("used_existing_row_id", 0)),
        "n_partition_assigned_points": partition_stats["assigned_points"],
        "n_partition_gap_points": partition_stats["gap_points"],
        "partition_edge_refinement_enabled": int(edge_report["enabled"]),
        "edge_plots_changed": edge_report["plots_changed"],
        "edge_plots_needing_review": edge_report["plots_needing_edge_review"],
    }
    for k, v in mode_info.items():
        if isinstance(v, (int, float, str)) or v is None:
            summary[k] = v
    pd.DataFrame([summary]).to_csv(
        out_dir / "10_partition_summary.csv", index=False, encoding="utf-8-sig"
    )
    try:
        write_boundary_diagnostics(
            out_dir=out_dir,
            rect_df=rect_df,
            work=work,
            clustered=clustered,
            selected=selected,
            plot_df=plot_df,
            plot_points=plot_points,
            mode_info=mode_info,
            cfg=CONFIG,
        )
    except Exception as e:
        print("[WARN] diagnostics readable diagnostics failed:", repr(e))
    print("[DONE] saved to", out_dir)
    print(
        "[INFO] DP mode:",
        mode_info.get("mode"),
        "typical_len:",
        mode_info.get("typical_len"),
        "n_plots:",
        len(plot_df),
    )


def _long_segment_num(row: pd.Series, keys: List[str], default: float = np.nan) -> float:
    for k in keys:
        if k in row.index:
            try:
                v = float(row.get(k))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return default


def _long_segment_candidate_evidence(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> float:
    try:
        ev = float(_local_relocation_boundary_evidence_score(row, ref_len, cfg))
        if np.isfinite(ev):
            return ev
    except Exception:
        pass
    return _long_segment_num(
        row,
        [
            "balanced_relocation_evidence_score",
            "local_relocation_evidence_score",
            "structural_recall_quality",
            "score",
            "feature_gap_reward",
            "feature_gap_boundary_likeness",
        ],
        0.0,
    )


def _long_segment_candidate_class_ok(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    cls = str(row.get("candidate_class", ""))
    kind = str(row.get("kind", ""))
    reject_prefix = tuple(cfg.get("LONG_SEGMENT_INSERT_REJECT_CLASSES_PREFIX", ("C_", "D_")))
    allow_prefix = tuple(cfg.get("LONG_SEGMENT_INSERT_ALLOW_CLASSES_PREFIX", ("A_", "B_")))
    if cls.startswith(reject_prefix):
        return False
    if kind == "observed_gap_interval":
        return True
    if cls.startswith(allow_prefix):
        return True
    try:
        if _structural_gap_is_strong_dense_scan_candidate(row, cfg):
            return True
    except Exception:
        pass
    return False


def _long_segment_insert_missing_boundaries_in_long_segments(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    cands: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long segment only when internal evidence and both child extents support it."""
    logs: List[Dict[str, Any]] = []
    if selected is None or len(selected) == 0 or cands is None or (len(cands) == 0):
        return (
            selected,
            pd.DataFrame([{"decision": "long_segment_skip_no_selected_or_candidates"}]),
        )
    if not bool(cfg.get("ENABLE_LONG_SEGMENT_LONG_SEGMENT_INTERNAL_INSERT", True)):
        return (selected, pd.DataFrame([{"decision": "long_segment_disabled"}]))
    try:
        ref_len = float(ref_len)
    except Exception:
        ref_len = np.nan
    if not np.isfinite(ref_len) or ref_len <= 1:
        return (selected, pd.DataFrame([{"decision": "long_segment_skip_bad_ref_len"}]))
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    cand_pool = cands.copy()
    cand_pool["center_s"] = pd.to_numeric(cand_pool["center_s"], errors="coerce")
    cand_pool = cand_pool.dropna(subset=["center_s"]).copy()
    ok_mask = cand_pool.apply(lambda r: _long_segment_candidate_class_ok(r, cfg), axis=1)
    cand_pool = cand_pool[ok_mask].copy()
    if len(cand_pool) == 0:
        return (current, pd.DataFrame([{"decision": "long_segment_skip_no_usable_candidates"}]))
    min_child = float(cfg.get("LONG_SEGMENT_INSERT_MIN_CHILD_RATIO", 0.55)) * ref_len
    max_child = float(cfg.get("LONG_SEGMENT_INSERT_MAX_CHILD_RATIO", 1.45)) * ref_len
    long_thr = float(cfg.get("LONG_SEGMENT_LONG_SEGMENT_INSERT_RATIO", 1.55)) * ref_len
    min_existing = float(cfg.get("LONG_SEGMENT_INSERT_MIN_DIST_EXISTING_ABS", 2200.0))
    min_ev = float(cfg.get("LONG_SEGMENT_INSERT_MIN_EVIDENCE", 10.5))
    max_passes = int(cfg.get("LONG_SEGMENT_INSERT_MAX_PASSES", 3))
    for pass_id in range(max_passes):
        changed = False
        current = current.sort_values("center_s").reset_index(drop=True)
        centers = current["center_s"].astype(float).to_numpy()
        cuts = np.concatenate([[smin], centers, [smax]])
        for seg_i in range(len(cuts) - 1):
            left = float(cuts[seg_i])
            right = float(cuts[seg_i + 1])
            seg_len = right - left
            if seg_len <= long_thr:
                continue
            lo = left + min_child
            hi = right - min_child
            if hi <= lo:
                continue
            sub = cand_pool[
                (cand_pool["center_s"].astype(float) >= lo)
                & (cand_pool["center_s"].astype(float) <= hi)
            ].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "decision": "long_segment_long_segment_no_internal_candidate",
                        "pass": pass_id,
                        "seg_index": seg_i,
                        "seg_left": left,
                        "seg_right": right,
                        "seg_len": seg_len,
                        "ref_len": ref_len,
                    }
                )
                continue
            if len(centers):
                sub["_dist_existing"] = (
                    sub["center_s"]
                    .astype(float)
                    .apply(lambda c: float(np.min(np.abs(centers - c))))
                )
                sub = sub[sub["_dist_existing"] >= min_existing].copy()
            if len(sub) == 0:
                continue
            sub["_ev"] = sub.apply(
                lambda r: _long_segment_candidate_evidence(r, ref_len, cfg), axis=1
            )
            sub["_left_len"] = sub["center_s"].astype(float) - left
            sub["_right_len"] = right - sub["center_s"].astype(float)
            sub = sub[
                (sub["_ev"] >= min_ev)
                & (sub["_left_len"] >= min_child)
                & (sub["_right_len"] >= min_child)
                & (sub["_left_len"] <= max_child)
                & (sub["_right_len"] <= max_child)
            ].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "decision": "long_segment_long_segment_candidates_failed_length_or_evidence",
                        "pass": pass_id,
                        "seg_index": seg_i,
                        "seg_left": left,
                        "seg_right": right,
                        "seg_len": seg_len,
                        "ref_len": ref_len,
                    }
                )
                continue
            ideal = left + ref_len
            sub["_balance_pen"] = (
                np.abs(sub["_left_len"] - ref_len) + np.abs(sub["_right_len"] - ref_len)
            ) / max(ref_len, 1.0)
            sub["_ideal_pen"] = np.abs(sub["center_s"].astype(float) - ideal) / max(
                ref_len, 1.0
            )
            sub["_long_segment_obj"] = (
                sub["_ev"]
                - float(cfg.get("LONG_SEGMENT_INSERT_BALANCE_WEIGHT", 8.0))
                * sub["_balance_pen"]
                - float(cfg.get("LONG_SEGMENT_INSERT_IDEAL_SPLIT_WEIGHT", 1.6))
                * sub["_ideal_pen"]
            )
            best = sub.sort_values(["_long_segment_obj", "_ev"], ascending=False).iloc[0]
            c_new = float(best["center_s"])
            new_row = best.copy()
            new_row["selected_stage"] = "long_segment_long_segment_internal_insert"
            new_row["long_segment_insert_from_long_segment"] = 1
            new_row["long_segment_insert_segment_left"] = left
            new_row["long_segment_insert_segment_right"] = right
            new_row["long_segment_insert_segment_len"] = seg_len
            new_row["long_segment_insert_ref_len"] = ref_len
            new_row["long_segment_insert_left_len"] = float(best["_left_len"])
            new_row["long_segment_insert_right_len"] = float(best["_right_len"])
            new_row["long_segment_insert_evidence"] = float(best["_ev"])
            new_row["long_segment_insert_objective"] = float(best["_long_segment_obj"])
            current = pd.concat(
                [current, pd.DataFrame([new_row])], ignore_index=True, sort=False
            )
            logs.append(
                {
                    "decision": "long_segment_insert_internal_boundary_in_long_segment",
                    "pass": pass_id,
                    "seg_index": seg_i,
                    "insert_center_s": c_new,
                    "candidate_class": str(best.get("candidate_class", "")),
                    "kind": str(best.get("kind", "")),
                    "evidence": float(best["_ev"]),
                    "objective": float(best["_long_segment_obj"]),
                    "seg_left": left,
                    "seg_right": right,
                    "seg_len": seg_len,
                    "left_len": float(best["_left_len"]),
                    "right_len": float(best["_right_len"]),
                    "ref_len": ref_len,
                }
            )
            changed = True
            break
        if not changed:
            break
    current = current.sort_values("center_s").reset_index(drop=True)
    return (current, pd.DataFrame(logs))


def select_sequence_with_long_segment_recovery(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selected, logs, mode_info = select_sequence_with_balanced_relocation(work, cands, cfg)
    ref_len = np.nan
    if isinstance(mode_info, dict):
        for key in ["typical_len", "feature_gap_aux_len", "ref_len"]:
            try:
                v = float(mode_info.get(key, np.nan))
                if np.isfinite(v) and v > 1:
                    ref_len = v
                    break
            except Exception:
                pass
    if not np.isfinite(ref_len):
        try:
            ref_len, _ = _structural_recall_aux_len(work, cfg)
        except Exception:
            ref_len = np.nan
    long_segment_log = pd.DataFrame()
    if selected is not None and len(selected) and np.isfinite(ref_len):
        selected, long_segment_log = _long_segment_insert_missing_boundaries_in_long_segments(
            work, selected, cands, float(ref_len), cfg
        )
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info["mode"] = "long_segment_long_segment_insert_visfix"
    mode_info["typical_len"] = (
        float(ref_len) if np.isfinite(ref_len) else mode_info.get("typical_len", np.nan)
    )
    mode_info["long_segment_n_selected_after_long_segment_insert"] = (
        int(len(selected)) if selected is not None else 0
    )
    out_logs = []
    if logs is not None and len(logs):
        out_logs.append(logs)
    if long_segment_log is not None and len(long_segment_log):
        out_logs.append(long_segment_log)
    out_logs = (
        pd.concat(out_logs, ignore_index=True, sort=False) if out_logs else pd.DataFrame()
    )
    return (selected.reset_index(drop=True), out_logs, mode_info)


def compute_canvas_transform(df: pd.DataFrame, cfg: Dict):
    x = df["t_rot"].to_numpy(dtype=float)
    y = df["s_rot"].to_numpy(dtype=float)
    x_min, x_max = (float(x.min()), float(x.max()))
    y_min, y_max = (float(y.min()), float(y.max()))
    x_span0 = max(1.0, x_max - x_min)
    pad_x = max(
        float(cfg.get("LONG_SEGMENT_VIS_EXTRA_T_MARGIN_ABS", 0.0)),
        float(cfg.get("LONG_SEGMENT_VIS_EXTRA_T_MARGIN_RATIO", 0.0)) * x_span0,
    )
    x_min -= pad_x
    x_max += pad_x
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)
    margin = float(cfg["POINT_MARGIN"])
    long_target = float(cfg["POINT_TARGET_LONG_SIDE"])
    short_target = float(cfg["POINT_MIN_SHORT_SIDE"])
    content_w = x_span + 2 * margin
    content_h = y_span + 2 * margin
    scale = long_target / content_h
    scale = max(scale, short_target / content_w)
    W = max(int(round(content_w * scale)), 600)
    H = max(int(round(content_h * scale)), 1000)
    x_offset = margin
    y_offset = margin
    if bool(cfg.get("POINT_CENTER_CONTENT", True)):
        extra_x = max(0.0, W / scale - content_w)
        extra_y = max(0.0, H / scale - content_h)
        x_offset += 0.5 * extra_x
        y_offset += 0.5 * extra_y
    return {
        "x_min": x_min,
        "y_min": y_min,
        "scale": scale,
        "W": W,
        "H": H,
        "x_offset": x_offset,
        "y_offset": y_offset,
    }


def _candidate_pool_load_original_clustered_candidates(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Candidate pool load original clustered candidates."""
    if not bool(cfg.get("ENABLE_CANDIDATE_POOL_REPAIR_POOL_WITH_ORIGINAL_CLUSTERED", True)):
        return pd.DataFrame()
    try:
        out_dir = Path(str(cfg.get("OUTPUT_DIR", "")))
        csv_name = str(
            cfg.get(
                "CANDIDATE_POOL_ORIGINAL_CLUSTERED_CSV", "03b_boundary_candidates_clustered.csv"
            )
        )
        p = out_dir / csv_name
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        if "center_s" not in df.columns:
            return pd.DataFrame()
        df["center_s"] = pd.to_numeric(df["center_s"], errors="coerce")
        df = df.dropna(subset=["center_s"]).copy()
        df["candidate_pool_from_original_clustered"] = 1
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _candidate_pool_merge_repair_candidate_pool(
    cands: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Combine current and original candidates for post-selection recovery."""
    parts: List[pd.DataFrame] = []
    if cands is not None and len(cands):
        cc = cands.copy()
        if "center_s" in cc.columns:
            cc["center_s"] = pd.to_numeric(cc["center_s"], errors="coerce")
            cc = cc.dropna(subset=["center_s"]).copy()
            cc["candidate_pool_from_current_pool"] = 1
            parts.append(cc)
    orig = _candidate_pool_load_original_clustered_candidates(cfg)
    if orig is not None and len(orig):
        parts.append(orig)
    if not parts:
        return pd.DataFrame()
    pool = pd.concat(parts, ignore_index=True, sort=False)
    pool["center_s"] = pd.to_numeric(pool["center_s"], errors="coerce")
    pool = pool.dropna(subset=["center_s"]).copy()
    if len(pool) == 0:
        return pool
    return pool.sort_values("center_s").reset_index(drop=True)


def select_sequence_with_high_evidence_insertion(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selected, logs, mode_info = select_sequence_with_long_segment_recovery(work, cands, cfg)
    ref_len = np.nan
    if isinstance(mode_info, dict):
        for key in ["typical_len", "feature_gap_aux_len", "ref_len"]:
            try:
                v = float(mode_info.get(key, np.nan))
                if np.isfinite(v) and v > 1:
                    ref_len = v
                    break
            except Exception:
                pass
    if not np.isfinite(ref_len):
        try:
            ref_len, _ = _structural_recall_aux_len(work, cfg)
        except Exception:
            ref_len = np.nan
    evidence_insert_log = pd.DataFrame()
    if selected is not None and len(selected) and np.isfinite(ref_len):
        selected, evidence_insert_log = (
            _evidence_insert_insert_missing_high_evidence_boundaries(
                work, selected, cands, float(ref_len), cfg
            )
        )
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info["mode"] = "safe_repair_safe_post_repair_strong_dense"
    mode_info["typical_len"] = (
        float(ref_len) if np.isfinite(ref_len) else mode_info.get("typical_len", np.nan)
    )
    mode_info["evidence_insert_n_selected_after_safe_repair"] = (
        int(len(selected)) if selected is not None else 0
    )
    out_logs = []
    if logs is not None and len(logs):
        out_logs.append(logs)
    if evidence_insert_log is not None and len(evidence_insert_log):
        out_logs.append(evidence_insert_log)
    out_logs = (
        pd.concat(out_logs, ignore_index=True, sort=False) if out_logs else pd.DataFrame()
    )
    return (selected.reset_index(drop=True), out_logs, mode_info)


def _terminal_recovery_tail_support(work: pd.DataFrame, a: float, b: float) -> Tuple[int, int]:
    """Count points and covered rows in a terminal interval."""
    if work is None or len(work) == 0 or b <= a:
        return (0, 0)
    sub = work[
        (pd.to_numeric(work["s_rot"], errors="coerce") >= a)
        & (pd.to_numeric(work["s_rot"], errors="coerce") <= b)
    ].copy()
    n = int(len(sub))
    rows = 0
    if n > 0 and "row_main" in sub.columns:
        rows = int(
            pd.to_numeric(sub["row_main"], errors="coerce").dropna().astype(int).nunique()
        )
    return (n, rows)


def _terminal_recovery_is_terminal_candidate(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> bool:
    """Check terminal gap evidence before testing tail and main-segment support."""
    cls = str(row.get("candidate_class", ""))
    kind = str(row.get("kind", ""))
    try:
        if _structural_gap_is_bad_feature_gap_through(row, cfg):
            return False
    except Exception:
        pass
    raw_score = _structural_gap_num(row, ["score"], -1000000000.0)
    ev = _long_segment_candidate_evidence(row, ref_len, cfg)
    strong_dense = False
    try:
        strong_dense = bool(_structural_gap_is_strong_dense_scan_candidate(row, cfg))
    except Exception:
        strong_dense = False
    if cls.startswith(("A_", "B_")):
        return bool(
            ev >= float(cfg.get("TERMINAL_RECOVERY_TERMINAL_MIN_EVIDENCE", 8.8))
            or raw_score >= float(cfg.get("TERMINAL_RECOVERY_TERMINAL_MIN_RAW_SCORE", 11.8))
        )
    if "observed_gap" in kind:
        return bool(
            ev >= float(cfg.get("TERMINAL_RECOVERY_TERMINAL_MIN_EVIDENCE", 8.8))
            or raw_score >= float(cfg.get("TERMINAL_RECOVERY_TERMINAL_MIN_RAW_SCORE", 11.8))
        )
    if strong_dense:
        return bool(
            raw_score >= float(cfg.get("SAFE_REPAIR_STRONG_DENSE_INSERT_MIN_RAW_SCORE", 10.5))
        )
    if "row_gap" in kind:
        return bool(
            raw_score >= float(cfg.get("TERMINAL_RECOVERY_TERMINAL_ROWGAP_MIN_RAW_SCORE", 12.0))
        )
    return False


def _terminal_recovery_insert_terminal_tail_boundaries(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    cands: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Test partial terminal plots without changing internal segments."""
    logs: List[Dict[str, Any]] = []
    if not bool(cfg.get("ENABLE_TERMINAL_RECOVERY_TERMINAL_TAIL_INSERT", True)):
        return (selected, pd.DataFrame([{"decision": "terminal_recovery_disabled"}]))
    if selected is None or len(selected) == 0 or cands is None or (len(cands) == 0):
        return (
            selected,
            pd.DataFrame([{"decision": "terminal_recovery_skip_no_selected_or_candidates"}]),
        )
    try:
        ref_len = float(ref_len)
    except Exception:
        ref_len = np.nan
    if not np.isfinite(ref_len) or ref_len <= 1:
        return (selected, pd.DataFrame([{"decision": "terminal_recovery_skip_bad_ref_len"}]))
    smin = float(pd.to_numeric(work["s_rot"], errors="coerce").min())
    smax = float(pd.to_numeric(work["s_rot"], errors="coerce").max())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    centers = current["center_s"].astype(float).to_numpy()
    if len(centers) == 0:
        return (current, pd.DataFrame([{"decision": "terminal_recovery_skip_no_centers"}]))
    pool = _candidate_pool_merge_repair_candidate_pool(cands, cfg)
    if "center_s" not in pool.columns:
        return (current, pd.DataFrame([{"decision": "terminal_recovery_skip_no_center_s"}]))
    pool["center_s"] = pd.to_numeric(pool["center_s"], errors="coerce")
    pool = pool.dropna(subset=["center_s"]).copy()
    if len(pool) == 0:
        return (current, pd.DataFrame([{"decision": "terminal_recovery_skip_empty_pool"}]))
    ok_mask = pool.apply(
        lambda r: _terminal_recovery_is_terminal_candidate(r, ref_len, cfg), axis=1
    )
    pool = pool[ok_mask].copy()
    if len(pool) == 0:
        return (
            current,
            pd.DataFrame([{"decision": "terminal_recovery_skip_no_terminal_candidate"}]),
        )
    min_tail_len = float(cfg.get("TERMINAL_RECOVERY_TERMINAL_TAIL_MIN_LEN_ABS", 350.0))
    max_tail_len = float(cfg.get("TERMINAL_RECOVERY_TERMINAL_TAIL_MAX_RATIO", 0.55)) * ref_len
    min_tail_points = int(cfg.get("TERMINAL_RECOVERY_TERMINAL_TAIL_MIN_POINTS", 18))
    min_tail_rows = int(cfg.get("TERMINAL_RECOVERY_TERMINAL_TAIL_MIN_ROWS", 2))
    full_min = float(cfg.get("TERMINAL_RECOVERY_TERMINAL_FULL_MIN_RATIO", 0.62)) * ref_len
    full_max = float(cfg.get("TERMINAL_RECOVERY_TERMINAL_FULL_MAX_RATIO", 1.55)) * ref_len
    min_existing = float(cfg.get("TERMINAL_RECOVERY_TERMINAL_MIN_DIST_EXISTING_ABS", 1800.0))
    inserts: List[pd.Series] = []

    def _evaluate_edge(edge_side: str, left: float, right: float):
        nonlocal pool, centers, inserts, logs
        seg_len = right - left
        if seg_len <= 0:
            return
        sub = pool[
            (pool["center_s"].astype(float) > left) & (pool["center_s"].astype(float) < right)
        ].copy()
        if len(sub) == 0:
            logs.append(
                {
                    "decision": "terminal_recovery_no_candidate_in_edge_segment",
                    "edge_side": edge_side,
                    "seg_left": left,
                    "seg_right": right,
                    "seg_len": seg_len,
                    "ref_len": ref_len,
                }
            )
            return
        sub["_dist_existing"] = (
            sub["center_s"].astype(float).apply(lambda c: float(np.min(np.abs(centers - c))))
        )
        sub = sub[sub["_dist_existing"] >= min_existing].copy()
        if len(sub) == 0:
            return
        rows = []
        for _, r in sub.iterrows():
            c = float(r["center_s"])
            if edge_side == "left":
                tail_len = c - left
                full_len = right - c
                tail_a, tail_b = (left, c)
            else:
                full_len = c - left
                tail_len = right - c
                tail_a, tail_b = (c, right)
            tail_pts, tail_rows = _terminal_recovery_tail_support(work, tail_a, tail_b)
            ev = _long_segment_candidate_evidence(r, ref_len, cfg)
            raw_score = _structural_gap_num(r, ["score"], 0.0)
            ok = (
                tail_len >= min_tail_len
                and tail_len <= max_tail_len
                and (tail_pts >= min_tail_points)
                and (tail_rows >= min_tail_rows)
                and (full_len >= full_min)
                and (full_len <= full_max)
            )
            rows.append(
                {
                    "_idx": _,
                    "_center_s": c,
                    "_ok": bool(ok),
                    "_ev": float(ev) if np.isfinite(ev) else 0.0,
                    "_raw_score": float(raw_score) if np.isfinite(raw_score) else 0.0,
                    "_tail_len": float(tail_len),
                    "_full_len": float(full_len),
                    "_tail_points": int(tail_pts),
                    "_tail_rows": int(tail_rows),
                    "_edge_side": edge_side,
                }
            )
        diag = pd.DataFrame(rows)
        good = diag[diag["_ok"]].copy()
        if len(good) == 0:
            logs.append(
                {
                    "decision": "terminal_recovery_edge_candidates_failed_tail_or_full_length_support",
                    "edge_side": edge_side,
                    "seg_left": left,
                    "seg_right": right,
                    "seg_len": seg_len,
                    "ref_len": ref_len,
                    "n_checked": int(len(diag)),
                    "best_raw_score": float(diag["_raw_score"].max()) if len(diag) else np.nan,
                    "best_tail_points": int(diag["_tail_points"].max()) if len(diag) else 0,
                }
            )
            return
        good["_obj"] = (
            good["_ev"]
            + 0.2 * good["_raw_score"]
            - 2.5 * (np.abs(good["_full_len"] - ref_len) / max(ref_len, 1.0))
        )
        best_diag = good.sort_values(["_obj", "_ev", "_raw_score"], ascending=False).iloc[0]
        best = sub.loc[best_diag["_idx"]].copy()
        best["selected_stage"] = "terminal_recovery_terminal_tail_boundary"
        best["terminal_recovery_terminal_tail_boundary"] = 1
        best["terminal_recovery_terminal_edge_side"] = edge_side
        best["terminal_recovery_terminal_tail_len"] = float(best_diag["_tail_len"])
        best["terminal_recovery_terminal_full_len"] = float(best_diag["_full_len"])
        best["terminal_recovery_terminal_tail_points"] = int(best_diag["_tail_points"])
        best["terminal_recovery_terminal_tail_rows"] = int(best_diag["_tail_rows"])
        best["terminal_recovery_terminal_evidence"] = float(best_diag["_ev"])
        best["terminal_recovery_terminal_raw_score"] = float(best_diag["_raw_score"])
        best["terminal_recovery_terminal_objective"] = float(best_diag["_obj"])
        inserts.append(best)
        logs.append(
            {
                "decision": "terminal_recovery_insert_terminal_tail_boundary",
                "edge_side": edge_side,
                "insert_center_s": float(best_diag["_center_s"]),
                "candidate_class": str(best.get("candidate_class", "")),
                "kind": str(best.get("kind", "")),
                "tail_len": float(best_diag["_tail_len"]),
                "full_len": float(best_diag["_full_len"]),
                "tail_points": int(best_diag["_tail_points"]),
                "tail_rows": int(best_diag["_tail_rows"]),
                "evidence": float(best_diag["_ev"]),
                "raw_score": float(best_diag["_raw_score"]),
                "ref_len": ref_len,
                "seg_left": left,
                "seg_right": right,
            }
        )

    _evaluate_edge("left", smin, float(centers[0]))
    _evaluate_edge("right", float(centers[-1]), smax)
    if inserts:
        current = pd.concat([current, pd.DataFrame(inserts)], ignore_index=True, sort=False)
        current = current.sort_values("center_s").reset_index(drop=True)
    elif not logs:
        logs.append(
            {"decision": "terminal_recovery_no_terminal_tail_insert_needed", "ref_len": ref_len}
        )
    return (current, pd.DataFrame(logs))


def select_sequence_with_terminal_recovery(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selected, logs, mode_info = select_sequence_with_high_evidence_insertion(work, cands, cfg)
    ref_len = np.nan
    if isinstance(mode_info, dict):
        for key in ["typical_len", "feature_gap_aux_len", "ref_len"]:
            try:
                v = float(mode_info.get(key, np.nan))
                if np.isfinite(v) and v > 1:
                    ref_len = v
                    break
            except Exception:
                pass
    if not np.isfinite(ref_len):
        try:
            ref_len, _ = _structural_recall_aux_len(work, cfg)
        except Exception:
            ref_len = np.nan
    terminal_recovery_log = pd.DataFrame()
    if selected is not None and len(selected) and np.isfinite(ref_len):
        selected, terminal_recovery_log = _terminal_recovery_insert_terminal_tail_boundaries(
            work, selected, cands, float(ref_len), cfg
        )
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info["mode"] = "terminal_policy_internal_repair_only_no_terminal_tail_auto_insert"
    mode_info["typical_len"] = (
        float(ref_len) if np.isfinite(ref_len) else mode_info.get("typical_len", np.nan)
    )
    mode_info["terminal_recovery_n_selected_after_terminal_tail_insert"] = (
        int(len(selected)) if selected is not None else 0
    )
    out_logs = []
    if logs is not None and len(logs):
        out_logs.append(logs)
    if terminal_recovery_log is not None and len(terminal_recovery_log):
        out_logs.append(terminal_recovery_log)
    out_logs = (
        pd.concat(out_logs, ignore_index=True, sort=False) if out_logs else pd.DataFrame()
    )
    return (selected.reset_index(drop=True), out_logs, mode_info)


def _dense_replacement_num(row: pd.Series, names: List[str], default: float = np.nan) -> float:
    for n in names:
        if n in row.index:
            try:
                v = float(row.get(n))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return float(default)


def _dense_replacement_str(row: pd.Series, names: List[str], default: str = "") -> str:
    for n in names:
        if n in row.index:
            try:
                v = row.get(n)
                if pd.notna(v):
                    return str(v)
            except Exception:
                pass
    return default


def _dense_replacement_length_cost_two(
    left_len: float, right_len: float, ref_len: float
) -> float:
    """Use log-ratio costs for the two adjacent segment extents."""
    if not np.isfinite(ref_len) or ref_len <= 1:
        return np.inf
    arr = np.asarray([left_len, right_len], dtype=float) / max(ref_len, 1e-09)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        return np.inf
    arr = np.clip(arr, 1e-06, 1000000.0)
    return float(np.sum(np.log(arr) ** 2))


def _dense_replacement_is_weak_or_suspicious_selected(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> Tuple[bool, str]:
    """Identify narrow, weak or continuous intervals among selected boundaries."""
    cls = _dense_replacement_str(row, ["candidate_class"], "")
    kind = _dense_replacement_str(row, ["kind"], "")
    decision = _dense_replacement_str(
        row, ["density_valley_decision", "start_block_decision", "decision"], ""
    )
    reasons: List[str] = []
    if cls.startswith(("C_", "D_")):
        reasons.append("weak_class")
    if "feature_gap" in kind.lower() and cls.startswith(("C_", "D_")):
        reasons.append("weak_feature_gap")
    if "reject_gap_width_too_small" in decision or "reject_gap_width" in decision:
        reasons.append("rejected_narrow_gap")
    width = _dense_replacement_num(
        row, ["local_relocation_gap_width", "gap_width", "edge_gap_width"], np.nan
    )
    if np.isfinite(width) and width <= float(
        cfg.get("LOCAL_RELOCATION_NARROW_GAP_WIDTH_ABS", 1700.0)
    ):
        reasons.append("narrow_gap")
    dens = _dense_replacement_num(
        row,
        [
            "local_relocation_gap_density_ratio",
            "gap_density_ratio",
            "continuity_gap_density_ratio",
            "start_block_gap_density_ratio",
        ],
        np.nan,
    )
    start = _dense_replacement_num(
        row,
        [
            "local_relocation_start_contrast",
            "start_contrast",
            "continuity_start_contrast",
            "start_block_start_contrast",
        ],
        np.nan,
    )
    through_rows = _dense_replacement_num(row, ["continuity_through_rows"], 0.0)
    if (
        through_rows >= 3
        and (not np.isfinite(dens) or dens > 0.42)
        and (not np.isfinite(start) or start < 2.0)
    ):
        reasons.append("multirow_continuity_through_gap")
    try:
        if _structural_gap_is_bad_feature_gap_through(row, cfg):
            reasons.append("bad_feature_gap_through")
    except Exception:
        pass
    return (len(reasons) > 0, ";".join(reasons))


def _dense_replacement_is_strong_dense_replace_candidate(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> bool:
    """Check whether a dense-scan candidate can replace a weaker selected boundary."""
    kind = _dense_replacement_str(row, ["kind"], "")
    if "dense_scan" not in kind.lower():
        return False
    cls = _dense_replacement_str(row, ["candidate_class"], "")
    if cls.startswith(("C_", "D_")):
        return False
    raw = _dense_replacement_num(row, ["score"], -1000000000.0)
    joint = _dense_replacement_num(row, ["joint_drop"], 1.0)
    cont = _dense_replacement_num(row, ["continuous_rows"], 999.0)
    nweak = _dense_replacement_num(row, ["n_weak_break"], 0.0)
    nstrong = _dense_replacement_num(row, ["n_strong_break"], 0.0)
    ev = _long_segment_candidate_evidence(row, ref_len, cfg)
    if raw < float(cfg.get("DENSE_REPLACEMENT_STRONG_DENSE_MIN_RAW_SCORE", 10.5)):
        return False
    if np.isfinite(joint) and joint > float(
        cfg.get("DENSE_REPLACEMENT_STRONG_DENSE_MAX_JOINT_DROP", 0.28)
    ):
        return False
    if np.isfinite(cont) and cont > float(
        cfg.get("DENSE_REPLACEMENT_STRONG_DENSE_MAX_CONT_ROWS", 1)
    ):
        return False
    if nweak < float(cfg.get("DENSE_REPLACEMENT_STRONG_DENSE_MIN_WEAK_BREAK", 3)):
        return False
    if nstrong < float(cfg.get("DENSE_REPLACEMENT_STRONG_DENSE_MIN_STRONG_BREAK", 2)):
        return False
    if not np.isfinite(ev) or ev < float(
        cfg.get("DENSE_REPLACEMENT_STRONG_DENSE_MIN_EVIDENCE", 8.2)
    ):
        return False
    return True


def _dense_replacement_local_strong_dense_replace(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    cands: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare a selected cut with nearby strong dense-scan candidates."""
    logs: List[Dict[str, Any]] = []
    if not bool(cfg.get("ENABLE_DENSE_REPLACEMENT_LOCAL_STRONG_DENSE_REPLACE", True)):
        return (selected, pd.DataFrame([{"decision": "dense_replacement_disabled"}]))
    if selected is None or len(selected) == 0:
        return (selected, pd.DataFrame([{"decision": "dense_replacement_skip_no_selected"}]))
    try:
        ref_len = float(ref_len)
    except Exception:
        ref_len = np.nan
    if not np.isfinite(ref_len) or ref_len <= 1:
        return (selected, pd.DataFrame([{"decision": "dense_replacement_skip_bad_ref_len"}]))
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    smin = float(pd.to_numeric(work["s_rot"], errors="coerce").min())
    smax = float(pd.to_numeric(work["s_rot"], errors="coerce").max())
    pool = _candidate_pool_merge_repair_candidate_pool(cands, cfg)
    if pool is None or len(pool) == 0 or "center_s" not in pool.columns:
        return (current, pd.DataFrame([{"decision": "dense_replacement_skip_empty_pool"}]))
    pool = pool.copy()
    pool["center_s"] = pd.to_numeric(pool["center_s"], errors="coerce")
    pool = pool.dropna(subset=["center_s"]).copy()
    if len(pool) == 0:
        return (
            current,
            pd.DataFrame([{"decision": "dense_replacement_skip_empty_pool_after_numeric"}]),
        )
    strong_mask = pool.apply(
        lambda r: _dense_replacement_is_strong_dense_replace_candidate(r, ref_len, cfg), axis=1
    )
    strong_pool = pool[strong_mask].copy()
    if len(strong_pool) == 0:
        return (
            current,
            pd.DataFrame([{"decision": "dense_replacement_skip_no_strong_dense_candidate"}]),
        )
    strong_pool["_dense_replacement_ev"] = strong_pool.apply(
        lambda r: _long_segment_candidate_evidence(r, ref_len, cfg), axis=1
    )
    strong_pool["_dense_replacement_raw"] = pd.to_numeric(
        strong_pool.get("score", 0), errors="coerce"
    ).fillna(0)
    search_win = max(
        float(cfg.get("DENSE_REPLACEMENT_REPLACE_SEARCH_ABS", 12000.0)),
        float(cfg.get("DENSE_REPLACEMENT_REPLACE_SEARCH_RATIO", 0.46)) * ref_len,
    )
    same_tol = float(cfg.get("DENSE_REPLACEMENT_REPLACE_SAME_TOL", 900.0))
    min_child = max(
        float(cfg.get("DENSE_REPLACEMENT_REPLACE_MIN_CHILD_ABS", 9000.0)),
        float(cfg.get("DENSE_REPLACEMENT_REPLACE_MIN_CHILD_RATIO", 0.5)) * ref_len,
    )
    max_child = float(cfg.get("DENSE_REPLACEMENT_REPLACE_MAX_CHILD_RATIO", 1.75)) * ref_len
    max_passes = int(cfg.get("DENSE_REPLACEMENT_MAX_PASSES", 3))
    max_replace = int(cfg.get("DENSE_REPLACEMENT_MAX_REPLACE_PER_RUN", 8))
    n_replace = 0
    for pass_id in range(max_passes):
        changed = False
        current = current.sort_values("center_s").reset_index(drop=True)
        centers = current["center_s"].astype(float).to_numpy()
        cuts = np.concatenate([[smin], centers, [smax]])
        for bi in range(len(current)):
            if n_replace >= max_replace:
                break
            c_old = float(current.loc[bi, "center_s"])
            prev_s = float(cuts[bi])
            next_s = float(cuts[bi + 2])
            old_left = c_old - prev_s
            old_right = next_s - c_old
            if old_left <= 0 or old_right <= 0:
                continue
            old_row = current.loc[bi].copy()
            old_weak, old_reason = _dense_replacement_is_weak_or_suspicious_selected(
                old_row, ref_len, cfg
            )
            old_cost = _dense_replacement_length_cost_two(old_left, old_right, ref_len)
            old_ratio_short = min(old_left, old_right) / max(ref_len, 1e-09)
            old_ratio_long = max(old_left, old_right) / max(ref_len, 1e-09)
            old_imbalance = max(old_left, old_right) / max(min(old_left, old_right), 1e-09)
            imbalanced = (
                old_imbalance
                >= float(cfg.get("DENSE_REPLACEMENT_IMBALANCE_RATIO_TRIGGER", 1.38))
                or old_ratio_short
                <= float(cfg.get("DENSE_REPLACEMENT_SHORT_RATIO_TRIGGER", 0.82))
                or old_ratio_long
                >= float(cfg.get("DENSE_REPLACEMENT_LONG_RATIO_TRIGGER", 1.18))
            )
            old_is_strong_dense = False
            try:
                old_is_strong_dense = bool(
                    _dense_replacement_is_strong_dense_replace_candidate(old_row, ref_len, cfg)
                )
            except Exception:
                old_is_strong_dense = False
            if old_is_strong_dense and (not imbalanced):
                continue
            lo = max(prev_s + min_child, c_old - search_win)
            hi = min(next_s - min_child, c_old + search_win)
            if hi <= lo:
                logs.append(
                    {
                        "decision": "dense_replacement_no_valid_replace_range",
                        "pass": pass_id,
                        "boundary_index": bi,
                        "old_center_s": c_old,
                        "prev_s": prev_s,
                        "next_s": next_s,
                        "old_left_len": old_left,
                        "old_right_len": old_right,
                        "ref_len": ref_len,
                    }
                )
                continue
            sub = strong_pool[
                (strong_pool["center_s"].astype(float) >= lo)
                & (strong_pool["center_s"].astype(float) <= hi)
            ].copy()
            sub = sub[np.abs(sub["center_s"].astype(float) - c_old) > same_tol].copy()
            if len(sub) == 0:
                continue
            rows: List[Dict[str, Any]] = []
            for idx, r in sub.iterrows():
                c_alt = float(r["center_s"])
                alt_left = c_alt - prev_s
                alt_right = next_s - c_alt
                if alt_left < min_child or alt_right < min_child:
                    continue
                if alt_left > max_child or alt_right > max_child:
                    continue
                alt_cost = _dense_replacement_length_cost_two(alt_left, alt_right, ref_len)
                gain = old_cost - alt_cost
                rows.append(
                    {
                        "_idx": idx,
                        "_center_s": c_alt,
                        "_alt_left": alt_left,
                        "_alt_right": alt_right,
                        "_alt_cost": alt_cost,
                        "_gain": gain,
                        "_ev": float(
                            r.get(
                                "_dense_replacement_ev",
                                _long_segment_candidate_evidence(r, ref_len, cfg),
                            )
                        ),
                        "_raw": float(
                            r.get(
                                "_dense_replacement_raw",
                                _dense_replacement_num(r, ["score"], 0.0),
                            )
                        ),
                    }
                )
            if not rows:
                continue
            diag = pd.DataFrame(rows)
            min_gain = (
                float(cfg.get("DENSE_REPLACEMENT_MIN_LENGTH_COST_GAIN_IF_OLD_WEAK", 0.018))
                if old_weak
                else float(cfg.get("DENSE_REPLACEMENT_MIN_LENGTH_COST_GAIN", 0.03))
            )
            if not old_weak and imbalanced:
                min_gain = max(
                    min_gain, float(cfg.get("DENSE_REPLACEMENT_MIN_LENGTH_COST_GAIN", 0.03))
                )
            good = diag[diag["_gain"] >= min_gain].copy()
            if len(good) == 0 or not (old_weak or imbalanced):
                best_diag = diag.sort_values(["_gain", "_ev", "_raw"], ascending=False).iloc[0]
                logs.append(
                    {
                        "decision": "dense_replacement_keep_no_sufficient_local_gain",
                        "pass": pass_id,
                        "boundary_index": bi,
                        "old_center_s": c_old,
                        "best_alt_center_s": float(best_diag["_center_s"]),
                        "old_cost": old_cost,
                        "best_alt_cost": float(best_diag["_alt_cost"]),
                        "gain": float(best_diag["_gain"]),
                        "required_gain": min_gain,
                        "old_weak": int(old_weak),
                        "imbalanced": int(imbalanced),
                        "old_reason": old_reason,
                    }
                )
                continue
            good["_obj"] = good["_gain"] + 0.01 * good["_ev"] + 0.002 * good["_raw"]
            best_diag = good.sort_values(
                ["_obj", "_gain", "_ev", "_raw"], ascending=False
            ).iloc[0]
            best = strong_pool.loc[best_diag["_idx"]].copy()
            c_new = float(best_diag["_center_s"])
            best["center_s"] = c_new
            best["selected_stage"] = "dense_replacement_strong_dense_local_replace"
            best["dense_replacement_replaced_from_center_s"] = c_old
            best["dense_replacement_replace_old_reason"] = old_reason
            best["dense_replacement_replace_old_left_len"] = float(old_left)
            best["dense_replacement_replace_old_right_len"] = float(old_right)
            best["dense_replacement_replace_new_left_len"] = float(best_diag["_alt_left"])
            best["dense_replacement_replace_new_right_len"] = float(best_diag["_alt_right"])
            best["dense_replacement_replace_old_cost"] = float(old_cost)
            best["dense_replacement_replace_new_cost"] = float(best_diag["_alt_cost"])
            best["dense_replacement_replace_gain"] = float(best_diag["_gain"])
            best["dense_replacement_replace_ref_len"] = float(ref_len)
            for col in best.index:
                if col not in current.columns:
                    current[col] = np.nan
            for col in current.columns:
                if col not in best.index:
                    best[col] = np.nan
            for col in current.columns:
                if current[col].dtype != object:
                    current[col] = current[col].astype(object)
            current.loc[bi, current.columns] = best[current.columns]
            logs.append(
                {
                    "decision": "dense_replacement_replace_weak_boundary_with_strong_dense",
                    "pass": int(pass_id),
                    "boundary_index": int(bi),
                    "old_center_s": float(c_old),
                    "new_center_s": float(c_new),
                    "prev_s": float(prev_s),
                    "next_s": float(next_s),
                    "old_left_len": float(old_left),
                    "old_right_len": float(old_right),
                    "new_left_len": float(best_diag["_alt_left"]),
                    "new_right_len": float(best_diag["_alt_right"]),
                    "old_cost": float(old_cost),
                    "new_cost": float(best_diag["_alt_cost"]),
                    "gain": float(best_diag["_gain"]),
                    "required_gain": float(min_gain),
                    "old_weak": int(old_weak),
                    "imbalanced": int(imbalanced),
                    "old_reason": old_reason,
                    "candidate_class": str(best.get("candidate_class", "")),
                    "kind": str(best.get("kind", "")),
                    "raw_score": float(best_diag["_raw"]),
                    "evidence": float(best_diag["_ev"]),
                    "ref_len": float(ref_len),
                }
            )
            changed = True
            n_replace += 1
            break
        if not changed or n_replace >= max_replace:
            break
    if not logs:
        logs.append(
            {"decision": "dense_replacement_no_replacement_needed", "ref_len": float(ref_len)}
        )
    current = current.sort_values("center_s").reset_index(drop=True)
    return (current, pd.DataFrame(logs))


def select_sequence_with_dense_replacement(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selected, logs, mode_info = select_sequence_with_terminal_recovery(work, cands, cfg)
    ref_len = np.nan
    if isinstance(mode_info, dict):
        for key in ["typical_len", "feature_gap_aux_len", "ref_len"]:
            try:
                v = float(mode_info.get(key, np.nan))
                if np.isfinite(v) and v > 1:
                    ref_len = v
                    break
            except Exception:
                pass
    if not np.isfinite(ref_len):
        try:
            ref_len, _ = _structural_recall_aux_len(work, cfg)
        except Exception:
            ref_len = np.nan
    dense_replacement_log = pd.DataFrame()
    if selected is not None and len(selected) and np.isfinite(ref_len):
        selected, dense_replacement_log = _dense_replacement_local_strong_dense_replace(
            work, selected, cands, float(ref_len), cfg
        )
    mode_info = dict(mode_info) if isinstance(mode_info, dict) else {}
    mode_info["mode"] = "dense_replacement_local_strong_dense_replacement"
    mode_info["typical_len"] = (
        float(ref_len) if np.isfinite(ref_len) else mode_info.get("typical_len", np.nan)
    )
    mode_info["dense_replacement_n_selected_after_local_dense_replace"] = (
        int(len(selected)) if selected is not None else 0
    )
    out_logs: List[pd.DataFrame] = []
    if logs is not None and len(logs):
        out_logs.append(logs)
    if dense_replacement_log is not None and len(dense_replacement_log):
        out_logs.append(dense_replacement_log)
    out_logs = (
        pd.concat(out_logs, ignore_index=True, sort=False) if out_logs else pd.DataFrame()
    )
    return (selected.reset_index(drop=True), out_logs, mode_info)


def _strict_rowgap_is_strict_rowgap_insert_candidate(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> bool:
    """Require strong row-gap evidence for an internal recovery cut."""
    if not bool(cfg.get("ENABLE_STRICT_ROWGAP_STRICT_ROWGAP_INTERNAL_INSERT", True)):
        return False
    kind = str(row.get("kind", "")).lower()
    cls = str(row.get("candidate_class", ""))
    if "row_gap" not in kind:
        return False
    raw = _structural_gap_num(row, ["score"], -1000000000.0)
    ev = _long_segment_candidate_evidence(row, ref_len, cfg)
    joint = _structural_gap_num(row, ["joint_drop"], 1.0)
    cont = _structural_gap_num(row, ["continuous_rows"], 999.0)
    nweak = _structural_gap_num(row, ["n_weak_break"], 0.0)
    nstrong = _structural_gap_num(row, ["n_strong_break"], 0.0)
    left_rec = _structural_gap_num(row, ["left_recover"], 0.0)
    right_rec = _structural_gap_num(row, ["right_recover"], 0.0)
    cluster_size = _structural_gap_num(row, ["cluster_size"], 0.0)
    if not np.isfinite(raw) or raw < float(cfg.get("STRICT_ROWGAP_ROWGAP_MIN_RAW_SCORE", 13.0)):
        return False
    if np.isfinite(joint) and joint > float(
        cfg.get("STRICT_ROWGAP_ROWGAP_MAX_JOINT_DROP", 0.2)
    ):
        return False
    if np.isfinite(cont) and cont > float(cfg.get("STRICT_ROWGAP_ROWGAP_MAX_CONT_ROWS", 1)):
        return False
    if nweak < float(cfg.get("STRICT_ROWGAP_ROWGAP_MIN_WEAK_BREAK", 3)):
        return False
    if nstrong < float(cfg.get("STRICT_ROWGAP_ROWGAP_MIN_STRONG_BREAK", 2)):
        return False
    if bool(cfg.get("STRICT_ROWGAP_ROWGAP_REQUIRE_BOTH_RECOVER", True)) and (
        not (left_rec >= 1 and right_rec >= 1)
    ):
        return False
    if np.isfinite(cluster_size) and cluster_size < float(
        cfg.get("STRICT_ROWGAP_ROWGAP_MIN_CLUSTER_SIZE", 5)
    ):
        return False
    if cls.startswith("D_"):
        return False
    return True


def _evidence_insert_is_safe_insert_candidate(
    row: pd.Series, ref_len: float, cfg: Dict[str, Any]
) -> bool:
    cls = str(row.get("candidate_class", ""))
    kind = str(row.get("kind", ""))
    strict_rowgap = False
    try:
        strict_rowgap = bool(
            _strict_rowgap_is_strict_rowgap_insert_candidate(row, ref_len, cfg)
        )
    except Exception:
        strict_rowgap = False
    strong_dense = False
    try:
        strong_dense = bool(_structural_gap_is_strong_dense_scan_candidate(row, cfg))
    except Exception:
        strong_dense = False
    if cls.startswith(("C_", "D_")):
        if not (strong_dense or strict_rowgap):
            return False
    if not strict_rowgap:
        try:
            if _structural_gap_is_bad_feature_gap_through(row, cfg):
                return False
        except Exception:
            pass
    ev = _long_segment_candidate_evidence(row, ref_len, cfg)
    if not np.isfinite(ev):
        return False
    if strict_rowgap:
        return bool(ev >= float(cfg.get("STRICT_ROWGAP_ROWGAP_MIN_EVIDENCE", 8.2)))
    if strong_dense:
        raw_score = _structural_gap_num(row, ["score"], -1000000000.0)
        return bool(
            np.isfinite(raw_score)
            and raw_score
            >= float(cfg.get("SAFE_REPAIR_STRONG_DENSE_INSERT_MIN_RAW_SCORE", 10.5))
            and (ev >= float(cfg.get("SAFE_REPAIR_STRONG_DENSE_INSERT_MIN_EVIDENCE", 9.6)))
        )
    ok = False
    if cls.startswith(("A_", "B_")):
        ok = True
    if "observed_gap" in kind:
        ok = True
    if not ok:
        return False
    return bool(ev >= float(cfg.get("EVIDENCE_INSERT_INSERT_MIN_EVIDENCE", 10.8)))


def _evidence_insert_insert_missing_high_evidence_boundaries(
    work: pd.DataFrame,
    selected: pd.DataFrame,
    cands: pd.DataFrame,
    ref_len: float,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logs: List[Dict[str, Any]] = []
    if not bool(cfg.get("ENABLE_EVIDENCE_INSERT_SAFE_HIGH_EVIDENCE_INSERT", True)):
        return (selected, pd.DataFrame([{"decision": "evidence_insert_disabled"}]))
    if selected is None or len(selected) == 0 or cands is None or (len(cands) == 0):
        return (
            selected,
            pd.DataFrame([{"decision": "evidence_insert_skip_no_selected_or_candidates"}]),
        )
    try:
        ref_len = float(ref_len)
    except Exception:
        ref_len = np.nan
    if not np.isfinite(ref_len) or ref_len <= 1:
        return (selected, pd.DataFrame([{"decision": "evidence_insert_skip_bad_ref_len"}]))
    smin = float(work["s_rot"].min())
    smax = float(work["s_rot"].max())
    current = selected.copy().sort_values("center_s").reset_index(drop=True)
    pool = _candidate_pool_merge_repair_candidate_pool(cands, cfg)
    if "center_s" not in pool.columns:
        return (
            current,
            pd.DataFrame([{"decision": "evidence_insert_skip_no_center_s_in_candidates"}]),
        )
    pool["center_s"] = pd.to_numeric(pool["center_s"], errors="coerce")
    pool = pool.dropna(subset=["center_s"]).copy()
    if len(pool) == 0:
        return (
            current,
            pd.DataFrame([{"decision": "evidence_insert_skip_empty_candidate_pool"}]),
        )
    pool["_strict_rowgap_strict_rowgap"] = pool.apply(
        lambda r: bool(_strict_rowgap_is_strict_rowgap_insert_candidate(r, ref_len, cfg)),
        axis=1,
    )
    ok_mask = pool.apply(
        lambda r: _evidence_insert_is_safe_insert_candidate(r, ref_len, cfg), axis=1
    )
    pool = pool[ok_mask].copy()
    if len(pool) == 0:
        return (current, pd.DataFrame([{"decision": "evidence_insert_skip_no_safe_candidate"}]))
    seg_min_ratio = float(cfg.get("EVIDENCE_INSERT_INSERT_SEGMENT_MIN_RATIO", 1.42))
    min_child_ratio = float(cfg.get("EVIDENCE_INSERT_INSERT_MIN_CHILD_RATIO", 0.45))
    edge_min_child_ratio = float(cfg.get("EVIDENCE_INSERT_INSERT_EDGE_MIN_CHILD_RATIO", 0.32))
    max_child_ratio = float(cfg.get("EVIDENCE_INSERT_INSERT_MAX_CHILD_RATIO", 1.6))
    min_existing = float(cfg.get("EVIDENCE_INSERT_INSERT_MIN_DIST_EXISTING_ABS", 2200.0))
    max_passes = int(cfg.get("EVIDENCE_INSERT_INSERT_MAX_PASSES", 4))
    max_insert = int(cfg.get("EVIDENCE_INSERT_INSERT_MAX_PER_RUN", 12))
    rowgap_parent_min = float(cfg.get("STRICT_ROWGAP_ROWGAP_PARENT_MIN_RATIO", 1.72)) * ref_len
    rowgap_min_child = float(cfg.get("STRICT_ROWGAP_ROWGAP_MIN_CHILD_RATIO", 0.68)) * ref_len
    rowgap_max_child = float(cfg.get("STRICT_ROWGAP_ROWGAP_MAX_CHILD_RATIO", 1.38)) * ref_len
    n_insert = 0
    for pass_id in range(max_passes):
        changed = False
        current = current.sort_values("center_s").reset_index(drop=True)
        centers = current["center_s"].astype(float).to_numpy()
        cuts = np.concatenate([[smin], centers, [smax]])
        for seg_i in range(len(cuts) - 1):
            left = float(cuts[seg_i])
            right = float(cuts[seg_i + 1])
            seg_len = right - left
            if seg_len < seg_min_ratio * ref_len:
                continue
            is_left_edge = seg_i == 0
            is_right_edge = seg_i == len(cuts) - 2
            min_left = (edge_min_child_ratio if is_left_edge else min_child_ratio) * ref_len
            min_right = (edge_min_child_ratio if is_right_edge else min_child_ratio) * ref_len
            max_child = max_child_ratio * ref_len
            lo = left + min_left
            hi = right - min_right
            if hi <= lo:
                logs.append(
                    {
                        "decision": "strict_rowgap_long_segment_no_valid_insert_range",
                        "pass": pass_id,
                        "seg_index": seg_i,
                        "seg_left": left,
                        "seg_right": right,
                        "seg_len": seg_len,
                        "ref_len": ref_len,
                    }
                )
                continue
            sub = pool[
                (pool["center_s"].astype(float) >= lo) & (pool["center_s"].astype(float) <= hi)
            ].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "decision": "strict_rowgap_long_segment_no_safe_internal_candidate",
                        "pass": pass_id,
                        "seg_index": seg_i,
                        "seg_left": left,
                        "seg_right": right,
                        "seg_len": seg_len,
                        "ref_len": ref_len,
                    }
                )
                continue
            if "_strict_rowgap_strict_rowgap" not in sub.columns:
                sub["_strict_rowgap_strict_rowgap"] = sub.apply(
                    lambda r: bool(
                        _strict_rowgap_is_strict_rowgap_insert_candidate(r, ref_len, cfg)
                    ),
                    axis=1,
                )
            rowgap_mask = sub["_strict_rowgap_strict_rowgap"].fillna(False).astype(bool)
            if seg_len < rowgap_parent_min:
                sub = sub[~rowgap_mask].copy()
                rowgap_mask = (
                    sub["_strict_rowgap_strict_rowgap"].fillna(False).astype(bool)
                    if len(sub)
                    else pd.Series([], dtype=bool)
                )
            if len(sub) == 0:
                logs.append(
                    {
                        "decision": "strict_rowgap_only_rowgap_but_parent_not_long_enough",
                        "pass": pass_id,
                        "seg_index": seg_i,
                        "seg_left": left,
                        "seg_right": right,
                        "seg_len": seg_len,
                        "ref_len": ref_len,
                        "rowgap_parent_min": rowgap_parent_min,
                    }
                )
                continue
            if len(centers):
                sub["_dist_existing"] = (
                    sub["center_s"]
                    .astype(float)
                    .apply(lambda c: float(np.min(np.abs(centers - c))))
                )
                sub = sub[sub["_dist_existing"] >= min_existing].copy()
            if len(sub) == 0:
                continue
            sub["_ev"] = sub.apply(
                lambda r: _long_segment_candidate_evidence(r, ref_len, cfg), axis=1
            )
            sub["_left_len"] = sub["center_s"].astype(float) - left
            sub["_right_len"] = right - sub["center_s"].astype(float)
            if "_strict_rowgap_strict_rowgap" not in sub.columns:
                sub["_strict_rowgap_strict_rowgap"] = sub.apply(
                    lambda r: bool(
                        _strict_rowgap_is_strict_rowgap_insert_candidate(r, ref_len, cfg)
                    ),
                    axis=1,
                )
            sub["_strict_rowgap_strict_rowgap"] = (
                sub["_strict_rowgap_strict_rowgap"].fillna(False).astype(bool)
            )
            normal_child_ok = (
                (sub["_left_len"] >= min_left)
                & (sub["_right_len"] >= min_right)
                & (sub["_left_len"] <= max_child)
                & (sub["_right_len"] <= max_child)
            )
            rowgap_child_ok = ~sub["_strict_rowgap_strict_rowgap"] | (
                sub["_left_len"] >= rowgap_min_child
            ) & (sub["_right_len"] >= rowgap_min_child) & (
                sub["_left_len"] <= rowgap_max_child
            ) & (
                sub["_right_len"] <= rowgap_max_child
            )
            sub = sub[normal_child_ok & rowgap_child_ok].copy()
            if len(sub) == 0:
                logs.append(
                    {
                        "decision": "strict_rowgap_candidates_failed_child_length",
                        "pass": pass_id,
                        "seg_index": seg_i,
                        "seg_left": left,
                        "seg_right": right,
                        "seg_len": seg_len,
                        "ref_len": ref_len,
                    }
                )
                continue
            ideal = left + ref_len
            sub["_balance_pen"] = (
                np.abs(sub["_left_len"] - ref_len) + np.abs(sub["_right_len"] - ref_len)
            ) / max(ref_len, 1.0)
            sub["_ideal_pen"] = np.abs(sub["center_s"].astype(float) - ideal) / max(
                ref_len, 1.0
            )
            sub["_raw_score"] = pd.to_numeric(sub.get("score", 0), errors="coerce").fillna(0.0)
            sub["_rowgap_bonus"] = np.where(
                sub["_strict_rowgap_strict_rowgap"], 0.35 * sub["_raw_score"], 0.0
            )
            sub["_evidence_insert_obj"] = (
                sub["_ev"]
                + sub["_rowgap_bonus"]
                - float(cfg.get("EVIDENCE_INSERT_INSERT_BALANCE_WEIGHT", 7.0))
                * sub["_balance_pen"]
                - float(cfg.get("EVIDENCE_INSERT_INSERT_IDEAL_SPLIT_WEIGHT", 1.2))
                * sub["_ideal_pen"]
            )
            best = sub.sort_values(["_evidence_insert_obj", "_ev"], ascending=False).iloc[0]
            c_new = float(best["center_s"])
            new_row = best.copy()
            new_row["selected_stage"] = (
                "strict_rowgap_strict_rowgap_internal_insert"
                if bool(best.get("_strict_rowgap_strict_rowgap", False))
                else "evidence_insert_safe_high_evidence_insert"
            )
            new_row["evidence_insert_insert_from_long_segment"] = 1
            new_row["strict_rowgap_strict_rowgap_insert"] = int(
                bool(best.get("_strict_rowgap_strict_rowgap", False))
            )
            new_row["evidence_insert_insert_segment_left"] = left
            new_row["evidence_insert_insert_segment_right"] = right
            new_row["evidence_insert_insert_segment_len"] = seg_len
            new_row["evidence_insert_insert_ref_len"] = ref_len
            new_row["evidence_insert_insert_left_len"] = float(best["_left_len"])
            new_row["evidence_insert_insert_right_len"] = float(best["_right_len"])
            new_row["evidence_insert_insert_evidence"] = float(best["_ev"])
            new_row["evidence_insert_insert_objective"] = float(best["_evidence_insert_obj"])
            current = pd.concat(
                [current, pd.DataFrame([new_row])], ignore_index=True, sort=False
            )
            n_insert += 1
            changed = True
            logs.append(
                {
                    "decision": (
                        "strict_rowgap_insert_strict_rowgap_boundary"
                        if bool(best.get("_strict_rowgap_strict_rowgap", False))
                        else "evidence_insert_insert_missing_high_evidence_boundary"
                    ),
                    "pass": pass_id,
                    "seg_index": seg_i,
                    "insert_center_s": c_new,
                    "candidate_class": str(best.get("candidate_class", "")),
                    "kind": str(best.get("kind", "")),
                    "is_strict_rowgap": int(
                        bool(best.get("_strict_rowgap_strict_rowgap", False))
                    ),
                    "evidence": float(best["_ev"]),
                    "objective": float(best["_evidence_insert_obj"]),
                    "seg_left": left,
                    "seg_right": right,
                    "seg_len": seg_len,
                    "left_len": float(best["_left_len"]),
                    "right_len": float(best["_right_len"]),
                    "ref_len": ref_len,
                }
            )
            break
        if n_insert >= max_insert or not changed:
            break
    current = current.sort_values("center_s").reset_index(drop=True)
    if not logs:
        logs.append({"decision": "strict_rowgap_no_insert_needed", "ref_len": ref_len})
    return (current, pd.DataFrame(logs))


def _recover_narrow_gaps(work, selected, cfg):
    """Propose an observed narrow gap only inside an anomalously long segment.

    This conservative supplement is called only in the unresolved-reference path.
    Segment lengths limit the search; actual row occupancy decides acceptance.
    It cannot infer a boundary from length or an expected number of plots alone.
    """
    logs = []
    if len(selected) == 0 or len(work) == 0:
        return (selected, pd.DataFrame(logs))
    current = selected.sort_values("center_s").copy()
    centers = current.center_s.to_numpy(dtype=float)
    cuts = np.r_[float(work.s_rot.min()), centers, float(work.s_rot.max())]
    lengths = np.diff(cuts)
    nrows = int(cfg.get("MAIN_ROWS", 3))
    samples = [
        np.sort(work.loc[work.row_main.astype(int) == r, "s_rot"].to_numpy(dtype=float))
        for r in range(nrows)
    ]
    if any((len(s) < 8 for s in samples)):
        return (current, pd.DataFrame(logs))
    min_band = min((float(v) for v in cfg["BAND_WIDTHS"]))
    min_width = min_band / 4
    side = float(cfg.get("SIDE_WINDOW", 1600.0))
    ordered = np.sort(np.concatenate(samples))
    intervals = []
    for interior in (0, 1):
        for a, b in zip(ordered[: -(interior + 1)], ordered[interior + 1 :]):
            a, b = (float(np.nextafter(a, b)), float(np.nextafter(b, a)))
            if min_width <= b - a < min_band * 0.8:
                intervals.append((a, b))
    count = lambda s, a, b: int(
        np.searchsorted(s, b, side="left") - np.searchsorted(s, a, side="left")
    )
    additions = []
    for seg, (start, end, length) in enumerate(zip(cuts[:-1], cuts[1:], lengths)):
        others = np.delete(lengths, seg)
        reference = float(np.median(others))
        if reference <= 0 or not 1.7 <= length / reference <= 2.5:
            continue
        if len(others) > 1 and np.max(np.abs(others / reference - 1)) > 0.25:
            continue
        lower = max(start + 0.7 * reference, end - 1.3 * reference)
        upper = min(start + 1.3 * reference, end - 0.7 * reference)
        if upper <= lower:
            continue
        logs.append(
            dict(
                stage="narrow_gap_narrow_gap",
                decision="inspect_long_segment",
                segment=seg + 1,
                segment_start=start,
                segment_end=end,
                reference_other_segments=reference,
                search_lo=lower,
                search_hi=upper,
            )
        )
        hits = []
        for a, b in intervals:
            center, width = ((a + b) / 2, b - a)
            if not lower <= center <= upper:
                continue
            inside = np.array([count(s, a, b) for s in samples])
            if int(inside.sum()) > 1 or int((inside == 0).sum()) < nrows - 1:
                continue
            left = np.array([count(s, a - side, a) for s in samples])
            right = np.array([count(s, b, b + side) for s in samples])
            if np.any(np.minimum(left, right) < 4):
                continue
            ratio = inside / width / (np.minimum(left, right) / side)
            joint = inside.sum() / width / (np.minimum(left, right).sum() / side)
            if float(joint) > 0.15 or float(ratio.max()) > 0.5:
                continue
            block = min(side / 4, width)
            near_left = np.array([count(s, a - block, a) for s in samples]) / block
            far_left = np.array([count(s, a - side - block, a - block) for s in samples]) / side
            near_right = np.array([count(s, b, b + block) for s in samples]) / block
            far_right = (
                np.array([count(s, b + block, b + block + side) for s in samples]) / side
            )
            left_peak = near_left / np.maximum(far_left, 1 / side)
            right_peak = near_right / np.maximum(far_right, 1 / side)
            peak = (
                max(float(np.sort(left_peak)[-2]), float(np.sort(right_peak)[-2]))
                if nrows >= 2
                else max(left_peak[0], right_peak[0])
            )
            if peak < 1.5:
                continue
            quality = width * (1 - joint) * min(peak, 3.0)
            hits.append(
                dict(
                    center_s=float(center),
                    s0=float(a),
                    s1=float(b),
                    width=float(width),
                    raw_gap_count=int(inside.sum()),
                    empty_rows=int((inside == 0).sum()),
                    local_rate_ratio=float(joint),
                    max_row_rate_ratio=float(ratio.max()),
                    edge_peak_two_rows=float(peak),
                    narrow_quality=float(quality),
                    segment=seg + 1,
                    reference_other_segments=reference,
                )
            )
        ranked = []
        for r in sorted(hits, key=lambda r: r["narrow_quality"], reverse=True):
            if all((abs(r["center_s"] - q["center_s"]) > min_band for q in ranked)):
                ranked.append(r)
        for rank, r in enumerate(ranked):
            logs.append(
                dict(
                    stage="narrow_gap_narrow_gap",
                    decision="narrow_gap_candidate",
                    rank=rank + 1,
                    **r,
                )
            )
        if not ranked:
            continue
        if (
            len(ranked) > 1
            and ranked[1]["narrow_quality"] >= 0.85 * ranked[0]["narrow_quality"]
        ):
            logs.append(
                dict(
                    stage="narrow_gap_narrow_gap",
                    decision="ambiguous_narrow_gaps_no_change",
                    segment=seg + 1,
                )
            )
            continue
        best = ranked[0]
        row = evaluate_candidate_band(
            work, row_ref_counts(work, side), best["s0"], best["s1"], cfg
        )
        row.update(best)
        row.update(
            kind="narrow_multiline_gap_recovery",
            selected_stage="narrow_gap_narrow_gap_needs_review",
            runtime_boundary_needs_review=1,
            narrow_gap_narrow_gap=1,
            runtime_evidence="narrow_gap_and_two_row_edge_block",
            runtime_strength=1,
        )
        additions.append(row)
        logs.append(
            dict(stage="narrow_gap_narrow_gap", decision="insert_observed_narrow_gap", **best)
        )
    if additions:
        current = pd.concat([current, pd.DataFrame(additions)], ignore_index=True, sort=False)
    return (current.sort_values("center_s").reset_index(drop=True), pd.DataFrame(logs))


def _get_original_candidates(cands, cfg):
    raw = cfg.get("_RUNTIME_ORIGINAL_CANDIDATES")
    if isinstance(raw, pd.DataFrame):
        return raw.copy()
    raw = _candidate_pool_load_original_clustered_candidates(cfg)
    return raw if len(raw) else cands.copy()


def _collect_gap_evidence(raw, cfg, include_one_side=False):
    """Use existing row-gap evidence; do not use the unreliable reference to reject gaps."""
    rows = []
    for _, r in raw.iterrows():
        f = lambda key, default=0.0: _safe_float(r.get(key), default)
        ns, nw = (f("n_strong_break"), f("n_weak_break"))
        left, right = (f("left_recover") >= 1, f("right_recover") >= 1)
        if not (
            np.isfinite(f("center_s", np.nan))
            and f("cluster_size") >= 3
            and (f("score") >= float(cfg.get("STRUCTURAL_GAP_DENSE_MIN_SCORE", 9.5)))
            and (f("joint_drop", 1) <= 0.35)
            and (nw >= max(2, int(cfg.get("MAIN_ROWS", 3)) - 1))
        ):
            continue
        strong = ns >= 2 and f("continuous_rows", 3) <= 1 and left and right
        supported = left and right and (ns >= 1)
        one_side = include_one_side and ns >= 2 and (left or right)
        if not (strong or supported or one_side):
            continue
        item = r.copy()
        item["runtime_evidence"] = (
            "strong_two_sides"
            if strong
            else "supported_two_sides" if supported else "one_side_needs_review"
        )
        item["runtime_strength"] = 2 if strong else 1 if supported else 0
        rows.append(item)
    if not rows:
        return raw.iloc[:0].copy()
    candidates = pd.DataFrame(rows).sort_values(["runtime_strength", "score"], ascending=False)
    spacing = float(cfg.get("FEATURE_GAP_DP_MIN_SPACING_ABS", 6500.0))
    if cfg.get("_RUNTIME_LOCAL_GAP_SPACING") is not None:
        spacing = max(spacing, 0.5 * float(cfg["_RUNTIME_LOCAL_GAP_SPACING"]))
    kept = []
    for _, row in candidates.iterrows():
        if all((abs(float(row.center_s) - float(prev.center_s)) >= spacing for prev in kept)):
            kept.append(row)
    return pd.DataFrame(kept).sort_values("center_s").reset_index(drop=True)


def _audit_reference_scale(work, evidence, cfg):
    ref, meta = _periodic_sequence_get_acf_reference_len(work, cfg)
    audit = dict(
        decision="runtime_reference_kept",
        reference_before=float(ref),
        reference_after=float(ref),
        reference_source_before=meta.get("ref_source", ""),
        repair_needed=False,
        reference_check_reason="no_conflicting_gap_sequence",
    )
    fallback = meta.get("ref_source") != "density_acf_primary"
    centers = evidence.center_s.to_numpy(dtype=float) if len(evidence) else np.array([])
    strong_count = int((evidence.runtime_strength >= 2).sum()) if len(evidence) else 0
    audit.update(
        reference_support_centers=";".join(map(str, centers)),
        reference_support_count=len(centers),
        reference_strong_support_count=strong_count,
    )
    verified = None
    if len(centers) >= 2 and strong_count >= 2:
        distances = np.diff(centers)
        candidate = float(np.median(distances))
        spread = float(np.max(np.abs(distances / candidate - 1)))
        edge_lens = np.array(
            [centers[0] - float(work.s_rot.min()), float(work.s_rot.max()) - centers[-1]]
        )
        enough = len(centers) >= 3 or (
            fallback
            and np.all((edge_lens >= 0.65 * candidate) & (edge_lens <= 1.5 * candidate))
        )
        coherent = enough and spread <= 0.2 and np.all(edge_lens <= 1.75 * candidate)
        audit.update(
            reference_candidate_len=candidate, reference_interval_max_relative_error=spread
        )
        if coherent and ref >= 1.6 * candidate:
            verified = candidate
            audit.update(
                decision="runtime_reference_recovered_from_gap_evidence",
                reference_after=candidate,
                repair_needed=True,
                reference_check_reason="reference_conflicts_with_independent_gaps",
            )
        elif len(centers) >= 3 and spread <= 0.2 and (ref >= 1.6 * candidate):
            audit.update(
                repair_needed=True,
                reference_check_reason="internal_gaps_conflict_with_reference_but_terminal_coverage_uncertain",
            )
    if fallback and verified is None:
        audit.update(
            repair_needed=True,
            reference_check_reason="rough_span_reference_is_not_a_measured_plot_length",
        )
    return (audit, verified, meta)


def _select_supported_gap_sequence(work, raw, cfg):
    selected = _collect_gap_evidence(raw, cfg, include_one_side=True)
    if len(selected):
        selected["selected_stage"] = "runtime_provisional_local_gap_evidence"
        selected["runtime_boundary_needs_review"] = 1
    elif "center_s" not in selected:
        selected["center_s"] = pd.Series(dtype=float)
    return selected


def select_sequence_with_reference_audit(
    work: pd.DataFrame, cands: pd.DataFrame, cfg: Dict[str, Any]
):
    raw = _get_original_candidates(cands, cfg)
    evidence = _collect_gap_evidence(raw, cfg)
    audit, checked_ref, ref_meta = _audit_reference_scale(work, evidence, cfg)
    selected, logs, info = select_sequence_with_dense_replacement(work, cands, cfg)
    info = dict(info) if isinstance(info, dict) else {}
    has_prune = (
        len(logs)
        and "decision" in logs
        and logs.decision.fillna("")
        .astype(str)
        .str.contains("prune_delete_boundary", regex=False)
        .any()
    )
    collapsed = len(selected) == 0 and bool(has_prune)
    recovery = bool(audit["repair_needed"]) or collapsed
    path = "baseline"
    if recovery and checked_ref is not None:
        run_cfg = dict(
            cfg,
            _RUNTIME_REPAIR_ACTIVE=True,
            _STRICT_ROWGAP_CHECKED_REFERENCE_LEN=checked_ref,
            _STRICT_ROWGAP_CHECKED_REFERENCE_META=dict(
                ref_meta, ref_len=checked_ref, ref_source="independent_gap_sequence"
            ),
        )
        repaired_cands, feature_log = build_feature_gap_proposals(work, raw.copy(), run_cfg)
        repaired_cands = classify_boundary_candidates(repaired_cands, run_cfg)
        selected, retry_log, retry_info = select_sequence_with_dense_replacement(
            work, repaired_cands, run_cfg
        )
        logs = pd.concat(
            [
                logs.assign(runtime_attempt="baseline"),
                retry_log.assign(runtime_attempt="recovery"),
            ],
            ignore_index=True,
            sort=False,
        )
        info.update(retry_info)
        path = "checked_reference_recovery"
        missing = [
            float(c)
            for c in evidence.center_s
            if len(selected) == 0
            or np.min(np.abs(selected.center_s.astype(float) - c))
            > float(cfg.get("FEATURE_GAP_DP_MIN_SPACING_ABS", 6500.0))
        ]
        if missing:
            selected = evidence.copy()
            selected["selected_stage"] = "runtime_provisional_consistent_gap_sequence"
            path = "consistent_gap_sequence_needs_review"
    elif recovery:
        provisional_cfg = dict(cfg)
        if (
            "reference_candidate_len" in audit
            and audit.get("reference_interval_max_relative_error", 1) <= 0.2
        ):
            provisional_cfg["_RUNTIME_LOCAL_GAP_SPACING"] = audit["reference_candidate_len"]
        selected = _select_supported_gap_sequence(work, raw, provisional_cfg)
        path = "local_gap_evidence_needs_review"
        if bool(cfg.get("ENABLE_NARROW_GAP_NARROW_GAP_RECOVERY", True)):
            selected, narrow_log = _recover_narrow_gaps(work, selected, cfg)
            if len(narrow_log):
                logs = pd.concat([logs, narrow_log], ignore_index=True, sort=False)
            if (
                "narrow_gap_narrow_gap" in selected
                and selected["narrow_gap_narrow_gap"].fillna(0).astype(int).any()
            ):
                path = "local_gap_with_narrow_recovery_needs_review"
        info["typical_len"] = np.nan
        audit["reference_after"] = np.nan
    if recovery and len(selected):
        selected["runtime_boundary_needs_review"] = 1
    missing_strong = []
    for _, r in evidence.iterrows():
        if r.runtime_strength >= 2 and (
            len(selected) == 0
            or np.min(np.abs(selected.center_s.astype(float) - float(r.center_s)))
            > float(cfg.get("FEATURE_GAP_DP_MIN_SPACING_ABS", 6500.0))
        ):
            missing_strong.append(float(r.center_s))
    review = recovery or len(selected) == 0 or bool(missing_strong)
    audit.update(
        runtime_path=path,
        runtime_baseline_collapsed=int(collapsed),
        runtime_missing_strong_gaps=";".join(map(str, missing_strong)),
    )
    info.update({k: v for k, v in audit.items() if k != "decision"})
    info.update(
        mode="reference_audit_and_gap_recovery",
        partition_review_required=int(review),
        partition_review_reason=(
            path if recovery else "no_boundary_or_missing_strong_evidence" if review else ""
        ),
        n_plots_is_provisional=int(review),
    )
    logs = pd.concat([pd.DataFrame([audit]), logs], ignore_index=True, sort=False)
    return (selected.reset_index(drop=True), logs, info)


def _select_repeated_gap_sequence(work, raw, cfg):
    """Recover repeated short plots only when independent gap positions support a short scale.

    No physical length, folder name, expected count, or synthetic boundary is used."""
    logs = []

    def f(r, k, d=0.0):
        return _safe_float(r.get(k), d)

    if len(raw) < 6:
        return (None, pd.DataFrame(logs), {})
    strong = []
    for _, r in raw.iterrows():
        if (
            f(r, "score") >= 9.5
            and f(r, "cluster_size") >= 3
            and (f(r, "n_strong_break") >= 2)
            and (f(r, "n_weak_break") >= 2)
            and (f(r, "continuous_rows", 3) <= 1)
            and (f(r, "joint_drop", 1) <= 0.4)
            and (max(f(r, "left_recover"), f(r, "right_recover")) >= 1)
        ):
            strong.append(r)
    if len(strong) < 6:
        return (None, pd.DataFrame(logs), {})
    anchors = pd.DataFrame(strong).sort_values("center_s").reset_index(drop=True)
    gaps = np.diff(anchors.center_s.to_numpy(dtype=float))
    period = float(np.median(gaps))
    spacing = float(cfg.get("FEATURE_GAP_DP_MIN_SPACING_ABS", 6500.0))
    single = (gaps >= 0.75 * period) & (gaps <= 1.25 * period)
    multiples = np.rint(gaps / period).astype(int)
    consistent = (
        (multiples >= 1) & (multiples <= 3) & (np.abs(gaps / period - multiples) <= 0.25)
    )
    if not (
        period < spacing
        and period >= 3 * float(cfg.get("CAND_CLUSTER_TOL", 900.0))
        and (int(single.sum()) >= 5)
        and (single.mean() >= 0.7)
        and (consistent.mean() >= 0.85)
    ):
        return (None, pd.DataFrame(logs), {})
    lo, hi = (float(work.s_rot.min()), float(work.s_rot.max()))
    if anchors.center_s.iloc[-1] - anchors.center_s.iloc[0] < 0.7 * (hi - lo):
        return (None, pd.DataFrame(logs), {})

    def edge_ok(c):
        return c - lo >= 0.35 * period and hi - c >= 0.35 * period

    kept = []
    for _, r in anchors.iterrows():
        if edge_ok(float(r.center_s)):
            item = r.copy()
            item["repeated_sequence_evidence"] = "independent_multiline_gap"
            kept.append(item)
        else:
            logs.append(
                dict(
                    stage="repeated_sequence_short_sequence",
                    decision="reject_terminal_fragment",
                    center_s=float(r.center_s),
                )
            )
    if len(kept) < 6:
        return (None, pd.DataFrame(logs), {})
    for _, r in raw.sort_values("score", ascending=False).iterrows():
        c = float(r.center_s)
        centers = np.sort([float(x.center_s) for x in kept])
        if not edge_ok(c) or np.min(abs(centers - c)) < 0.65 * period:
            continue
        if not (
            f(r, "score") >= float(cfg.get("CAND_MIN_SCORE", 5.8))
            and f(r, "cluster_size") >= 3
            and (f(r, "n_strong_break") >= 1)
            and (f(r, "n_weak_break") >= 2)
            and (f(r, "continuous_rows", 3) <= 1)
            and (f(r, "joint_drop", 1) <= 0.2)
            and (max(f(r, "left_recover"), f(r, "right_recover")) >= 1)
        ):
            continue
        left = centers[centers < c]
        right = centers[centers > c]
        intervals = []
        if len(left):
            intervals.append(c - left[-1])
        if len(right):
            intervals.append(right[0] - c)
        if not intervals or not all((0.75 * period <= v <= 1.25 * period for v in intervals)):
            continue
        item = r.copy()
        item["repeated_sequence_evidence"] = "weaker_gap_supported_by_sequence"
        kept.append(item)
        logs.append(
            dict(
                stage="repeated_sequence_short_sequence",
                decision="recover_observed_gap",
                center_s=c,
            )
        )
    selected = pd.DataFrame(kept).sort_values("center_s").reset_index(drop=True)
    distances = np.diff(selected.center_s)
    unresolved = int(np.sum((distances < 0.65 * period) | (distances > 1.35 * period)))
    selected["selected_stage"] = "repeated_sequence_short_sequence_needs_review"
    selected["repeated_sequence_short_sequence"] = 1
    selected["repeated_sequence_reference_period"] = period
    selected["runtime_boundary_needs_review"] = 1
    audit = dict(
        stage="repeated_sequence_short_sequence",
        decision="short_period_supported_by_observed_gaps",
        period=period,
        original_min_spacing=spacing,
        strong_support=len(anchors),
        adjacent_interval_support=int(single.sum()),
        interval_consistency=float(consistent.mean()),
        unresolved_interval_count=unresolved,
    )
    logs.insert(0, audit)
    info = dict(
        mode="observed_gap_sequence",
        typical_len=period,
        runtime_path="short_gap_sequence_needs_review",
        partition_review_required=1,
        partition_review_reason="short_period_recovered_from_observed_gaps",
        n_plots_is_provisional=1,
        repeated_sequence_period=period,
        repeated_sequence_strong_support=len(anchors),
        repeated_sequence_unresolved_intervals=unresolved,
    )
    return (selected, pd.DataFrame(logs), info)


def _select_local_gap_sequence(work, raw, cfg):
    """Require repeated measured gaps and check support against their local rows."""
    empty = (None, pd.DataFrame(), {})
    if len(raw) < 2 or len(work) == 0:
        return empty
    candidates = raw.loc[
        (raw.score >= 9.5)
        & (raw.cluster_size >= 3)
        & (raw.n_strong_break >= 2)
        & (raw.n_weak_break >= 2)
        & (raw.continuous_rows <= 1)
        & (raw.joint_drop <= 0.45)
    ].sort_values("center_s")
    if len(candidates) < 2:
        return empty
    distances = np.diff(candidates.center_s.to_numpy(dtype=float))
    period = float(np.median(distances))
    spacing = float(cfg.get("FEATURE_GAP_DP_MIN_SPACING_ABS", 6500.0))
    if not 3 * float(cfg.get("CAND_CLUSTER_TOL", 900.0)) <= period < spacing:
        return empty
    lo, hi = (float(work.s_rot.min()), float(work.s_rot.max()))
    row_positions = [
        np.sort(group.s_rot.to_numpy(dtype=float)) for _, group in work.groupby("row_main")
    ]
    side = 0.3 * period

    def counts(start, end):
        return np.array(
            [np.searchsorted(s, end) - np.searchsorted(s, start) for s in row_positions]
        )

    def local_support(start, end):
        flanks = np.minimum(counts(start - side, start), counts(end, end + side))
        ratios = counts(start, end) * side / (np.maximum(flanks, 1) * (end - start))
        return ratios[(flanks >= 5) & (ratios <= 0.6)]

    kept = []
    logs = []
    for _, row in candidates.iterrows():
        center, start, end = (float(row.center_s), float(row.s0), float(row.s1))
        if min(center - lo, hi - center) < 0.35 * period:
            logs.append(
                dict(
                    stage="local_sequence_short_local",
                    decision="reject_terminal_fragment",
                    center_s=center,
                )
            )
            continue
        item = row.copy()
        support = local_support(start, end)
        if len(support) < 2:
            best = None
            for shifted in center + period * np.linspace(-0.18, 0.18, 19):
                for width in period * np.array([0.12, 0.16, 0.2]):
                    a, b = (shifted - width / 2, shifted + width / 2)
                    nearby = local_support(a, b)
                    if len(nearby) < 2 or np.mean(nearby) > 0.35:
                        continue
                    quality = width * (1.0 - np.mean(nearby)) - 0.35 * abs(shifted - center)
                    if best is None or quality > best[0]:
                        best = (quality, a, b, nearby)
            if best is None:
                logs.append(
                    dict(
                        stage="local_sequence_short_local",
                        decision="insufficient_local_rows",
                        center_s=center,
                    )
                )
                continue
            _, start, end, support = best
            item["center_s"], item["s0"], item["s1"], item["width"] = (
                (start + end) / 2,
                start,
                end,
                end - start,
            )
            logs.append(
                dict(
                    stage="local_sequence_short_local",
                    decision="refit_candidate_to_local_valley",
                    center_s=center,
                    refined_center_s=float(item.center_s),
                )
            )
        item["local_sequence_local_rows"] = len(support)
        kept.append(item)
    if len(kept) < 2:
        return empty
    selected = pd.DataFrame(kept).reset_index(drop=True)
    gaps = np.diff(selected.center_s.to_numpy(dtype=float))
    single = (gaps >= 0.75 * period) & (gaps <= 1.25 * period)
    multiples = np.rint(gaps / period)
    consistent = (
        (multiples >= 1) & (multiples <= 3) & (np.abs(gaps / period - multiples) <= 0.25)
    )
    end_lengths = (
        np.array([selected.center_s.iloc[0] - lo, hi - selected.center_s.iloc[-1]]) / period
    )
    if len(selected) <= 5:
        if not single.all() or np.any((end_lengths < 0.55) | (end_lengths > 1.5)):
            return empty
    elif single.mean() < 0.7 or consistent.mean() < 0.85 or np.any(end_lengths > 1.65):
        return empty
    unresolved = int(np.sum(~single))
    selected["selected_stage"] = "local_gap_sequence_needs_review"
    selected["repeated_sequence_reference_period"] = period
    selected["runtime_boundary_needs_review"] = 1
    logs.insert(
        0,
        dict(
            stage="local_sequence_short_local",
            decision="repeated_locally_supported_gaps",
            period=period,
            candidate_count=len(candidates),
            kept_count=len(selected),
            unresolved_interval_count=unresolved,
        ),
    )
    info = dict(
        mode="local_gap_sequence",
        typical_len=period,
        runtime_path="short_local_sequence_needs_review",
        partition_review_required=1,
        partition_review_reason="short_period_and_local_row_support",
        n_plots_is_provisional=1,
        repeated_sequence_period=period,
        repeated_sequence_unresolved_intervals=unresolved,
    )
    return (selected, pd.DataFrame(logs), info)


def select_sequence_with_local_gap_support(work, cands, cfg):
    raw = _get_original_candidates(cands, cfg)
    previous, previous_log, previous_info = _select_repeated_gap_sequence(work, raw, cfg)
    if previous is not None and previous_info["repeated_sequence_unresolved_intervals"] == 0:
        return (previous, previous_log, previous_info)
    selected, logs, info = _select_local_gap_sequence(work, raw, cfg)
    if selected is not None:
        if previous is None:
            return (selected, logs, info)
        prior_centers = set(previous.center_s)
        if len(selected) > len(previous) and prior_centers.issubset(set(selected.center_s)):
            return (selected, logs, info)
    if previous is not None:
        return (previous, previous_log, previous_info)
    return select_sequence_with_reference_audit(work, cands, cfg)


def _measured_gap_candidates(raw):
    if len(raw) == 0:
        return raw
    return raw.loc[
        (raw.score >= 9.5)
        & (raw.cluster_size >= 3)
        & (raw.n_strong_break >= 2)
        & (raw.n_weak_break >= 2)
        & (raw.continuous_rows <= 1)
        & (raw.joint_drop <= 0.45)
    ].sort_values("center_s")


def _recover_periodic_gap_sequence(work, raw, previous):
    """Resolve missing intervals using local gaps and a measured repeating scale."""
    empty = (None, pd.DataFrame(), {})
    if len(raw) < 3 or len(work) == 0:
        return empty
    candidates = _measured_gap_candidates(raw)
    if len(candidates) < 3:
        return empty
    centers = candidates.center_s.to_numpy(dtype=float)
    period = float(np.median(np.diff(centers)))
    if not np.isfinite(period) or period <= 0:
        return empty
    s = np.sort(work.s_rot.to_numpy(dtype=float))
    rows = [np.sort(group.s_rot.to_numpy(dtype=float)) for _, group in work.groupby("row_main")]
    side = 0.3 * period

    def count(positions, start, end):
        return np.searchsorted(positions, end) - np.searchsorted(positions, start)

    def has_same_row_support(start, end):
        supported = 0
        for positions in rows:
            flank = min(
                count(positions, start - side, start), count(positions, end, end + side)
            )
            if flank >= 5 and count(positions, start, end) * side <= 0.6 * flank * (
                end - start
            ):
                supported += 1
        return supported >= 2

    def find_common_valley(center):
        best = None
        for offset in np.linspace(-0.18, 0.18, 37):
            for fraction in (0.08, 0.1, 0.12, 0.16, 0.2):
                location, width = (center + offset * period, fraction * period)
                start, end = (location - width / 2, location + width / 2)
                flank = min(count(s, start - side, start), count(s, end, end + side))
                if flank < 12:
                    continue
                ratio = count(s, start, end) * side / (flank * width)
                if ratio > 0.22:
                    continue
                quality = fraction * (1 - ratio) - 0.35 * abs(offset)
                if best is None or quality > best[0]:
                    best = (quality, start, end, ratio)
        return best

    selected = []
    logs = []
    same_row_centers = []
    for _, row in candidates.iterrows():
        center, start, end = (float(row.center_s), float(row.s0), float(row.s1))
        if min(center - s[0], s[-1] - center) < 0.35 * period:
            continue
        item = row.copy()
        if has_same_row_support(start, end):
            same_row_centers.append(center)
            item["measured_sequence_local_evidence"] = "same_row_flank_support"
        else:
            valley = find_common_valley(center)
            if valley is None:
                continue
            _, start, end, ratio = valley
            item["measured_sequence_local_evidence"] = "common_low_occupancy_valley"
            item["center_s"], item["s0"], item["s1"], item["width"] = (
                (start + end) / 2,
                start,
                end,
                end - start,
            )
            logs.append(
                dict(
                    stage="measured_sequence_period_recovery",
                    decision="locate_common_valley",
                    center_s=center,
                    refined_center_s=item.center_s,
                    local_density_ratio=ratio,
                )
            )
        selected.append(item)
    if len(same_row_centers) < 3 or len(selected) <= len(previous):
        return empty
    selected = pd.DataFrame(selected).sort_values("center_s").reset_index(drop=True)
    new_centers = selected.center_s.to_numpy(dtype=float)
    gaps = np.diff(new_centers) / period
    edges = np.array([new_centers[0] - s[0], s[-1] - new_centers[-1]]) / period
    if np.any((gaps < 0.72) | (gaps > 1.28)) or np.any((edges < 0.35) | (edges > 1.6)):
        return empty
    if len(previous) and any(
        (np.min(np.abs(new_centers - c)) > 0.24 * period for c in previous.center_s)
    ):
        return empty
    selected["selected_stage"] = "measured_sequence_period_recovery_needs_review"
    selected["repeated_sequence_reference_period"] = period
    selected["runtime_boundary_needs_review"] = 1
    logs.insert(
        0,
        dict(
            stage="measured_sequence_period_recovery",
            decision="verified_local_gap_sequence",
            measured_period=period,
            row_supported_gaps=len(same_row_centers),
            previous_boundary_count=len(previous),
            selected_boundary_count=len(selected),
        ),
    )
    info = dict(
        mode="measured_gap_sequence",
        typical_len=period,
        runtime_path="measured_gap_sequence_needs_review",
        partition_review_required=1,
        partition_review_reason="measured_period_and_observed_local_valleys",
        n_plots_is_provisional=1,
        repeated_sequence_period=period,
        repeated_sequence_unresolved_intervals=0,
    )
    return (selected, pd.DataFrame(logs), info)


def _rescan_at_measured_scale(work, raw, previous, cfg):
    """Rescan with band widths proportional to an independently observed period."""
    empty = (None, pd.DataFrame(), {})
    if len(work) == 0:
        return empty
    centers = (
        _measured_gap_candidates(raw).center_s.to_numpy(dtype=float)
        if len(raw)
        else np.array([])
    )
    if len(centers) < 6 or len(centers) <= len(previous) + 1:
        return empty
    distances = np.diff(centers)
    period = float(np.median(distances))
    if not np.isfinite(period) or period <= 0:
        return empty
    regular = (distances >= 0.75 * period) & (distances <= 1.25 * period)
    lo, hi = (float(work.s_rot.min()), float(work.s_rot.max()))
    if regular.mean() < 0.75 or centers[-1] - centers[0] < 0.65 * (hi - lo):
        return empty
    scale = min(cfg["BAND_WIDTHS"]) / (0.22 * period)
    if 0.9 <= scale <= 1.1:
        return empty
    normalized = work.assign(s_rot=(work.s_rot - lo) * scale)
    earlier = (
        previous.assign(center_s=(previous.center_s - lo) * scale)
        if len(previous)
        else previous
    )
    _, measured = build_all_candidates(normalized, cfg)
    measured = classify_boundary_candidates(measured, cfg)
    selected, logs, info = _recover_periodic_gap_sequence(normalized, measured, earlier)
    if selected is None:
        return empty
    for column in ("center_s", "s0", "s1", "cluster_center_min", "cluster_center_max"):
        if column in selected:
            selected[column] = selected[column] / scale + lo
    selected["width"] /= scale
    selected["repeated_sequence_reference_period"] /= scale
    for column in ("center_s", "refined_center_s"):
        if column in logs:
            logs[column] = logs[column] / scale + lo
    logs["measured_period"] /= scale
    logs["candidate_scan_scale"] = scale
    info["typical_len"] /= scale
    info["repeated_sequence_period"] /= scale
    info["runtime_path"] = "scale_adjusted_gap_sequence_needs_review"
    return (selected, logs, info)


def select_boundaries(work, cands, cfg):
    selected, logs, info = select_sequence_with_local_gap_support(work, cands, cfg)
    if info.get("repeated_sequence_unresolved_intervals") == 0:
        return (selected, logs, info)
    raw = _get_original_candidates(cands, cfg)
    recovered, recovery_log, recovery_info = _recover_periodic_gap_sequence(work, raw, selected)
    if recovered is None:
        recovered, recovery_log, recovery_info = _rescan_at_measured_scale(
            work, raw, selected, cfg
        )
    if recovered is None:
        return (selected, logs, info)
    return (recovered, pd.concat([recovery_log, logs], ignore_index=True), recovery_info)


def _write_image_unicode(path, image, params=None):
    """Write PNG/JPEG to Unicode Windows paths and fail explicitly on encoder errors."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".png", image, params or [])
    if not ok:
        raise OSError("Image encoding failed: {}".format(path))
    encoded.tofile(str(path))
    return True


def _export_boundary_review(out_dir, summary):
    rect = pd.read_csv(out_dir / "01_rectified_points_all.csv", low_memory=False)
    try:
        selected = pd.read_csv(out_dir / "04_dp_selected_boundaries.csv")
    except pd.errors.EmptyDataError:
        selected = pd.DataFrame(columns=["center_s"])
    raw = pd.read_csv(out_dir / "03b_boundary_candidates_clustered.csv")
    uncertain = bool(int(summary.get("partition_review_required", 0)))
    review = raw.copy()
    if "narrow_gap_narrow_gap" in selected:
        narrow = selected[selected["narrow_gap_narrow_gap"].fillna(0).astype(int) == 1]
        if len(narrow):
            review = pd.concat([review, narrow], ignore_index=True, sort=False)
    centers = selected.center_s.to_numpy(dtype=float) if len(selected) else np.array([])
    tol = float(CONFIG.get("FEATURE_GAP_DP_MIN_SPACING_ABS", 6500.0))
    review["near_selected_boundary"] = [
        int(len(centers) > 0 and np.min(np.abs(centers - c)) <= tol) for c in review.center_s
    ]
    review["result_is_provisional"] = int(uncertain)
    review.to_csv(out_dir / "00_boundary_review.csv", index=False, encoding="utf-8-sig")
    width, height = (2000, 600)
    canvas = np.full((height, width, 3), 255, np.uint8)
    smin, smax = (float(rect.s_rot.min()), float(rect.s_rot.max()))
    tmin, tmax = (float(rect.t_rot.min()), float(rect.t_rot.max()))
    sx = lambda s: int(80 + (float(s) - smin) / max(smax - smin, 1) * 1840)
    ty = lambda t: int(135 + (float(t) - tmin) / max(tmax - tmin, 1) * 340)
    palette = [(80, 130, 40), (185, 115, 45), (50, 145, 205)]
    for r in rect.itertuples():
        color = palette[int(r.row_main) % 3] if bool(r.keep_for_partition) else (180, 180, 180)
        cv2.circle(canvas, (sx(r.s_rot), ty(r.t_rot)), 1, color, -1)
    color = (0, 135, 235) if uncertain else (45, 135, 35)
    for index, c in enumerate(centers):
        x = sx(c)
        cv2.line(canvas, (x, 110), (x, 485), color, 2)
        cv2.putText(
            canvas,
            "B{}: {:.0f}".format(index + 1, c),
            (max(1, x - 65), 515 + 22 * (index % 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    cuts = [smin] + centers.tolist() + [smax]
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        cv2.putText(
            canvas,
            "P{}".format(i + 1),
            (sx((a + b) / 2) - 15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    title = (
        "PROVISIONAL - NEEDS MANUAL REVIEW"
        if uncertain
        else "BASELINE RESULT - NO RECOVERY TRIGGERED"
    )
    cv2.putText(canvas, title, (80, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "{} proposed plots | s_rot coordinates | {}".format(
            len(cuts) - 1, summary.get("runtime_path", "baseline")
        ),
        (80, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "All input points shown; grey points were not used for partitioning. Boundaries are not manual ground truth.",
        (80, 575),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    _write_image_unicode(out_dir / "00_partition_review.png", canvas)


def main():
    import datetime
    import hashlib
    import json
    import sys

    source = Path(CONFIG["INPUT_CSV"])
    if not source.is_file():
        raise FileNotFoundError("Input CSV does not exist: {}".format(source))
    out = Path(CONFIG["OUTPUT_DIR"])
    if out.exists() and any(out.iterdir()):
        out = out / ("run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=True)
    CONFIG["OUTPUT_DIR"] = str(out)
    for key in (
        "_RUNTIME_ORIGINAL_CANDIDATES",
        "_STRICT_ROWGAP_CHECKED_REFERENCE_LEN",
        "_STRICT_ROWGAP_CHECKED_REFERENCE_META",
    ):
        CONFIG.pop(key, None)
    status = dict(
        status="running",
        stage="partitioning",
        code_revision="partition_edges_1",
        python=sys.executable,
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        input_csv=str(source.resolve()),
        output_dir=str(out.resolve()),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(
            Path(__file__).with_name("partition_config.py").read_bytes()
        ).hexdigest(),
        edge_code_sha256=hashlib.sha256(
            Path(__file__).with_name("partition_edges.py").read_bytes()
        ).hexdigest(),
        point_quality_filter_applied=False,
        plot_row_fitting_applied=False,
        phenotype_analysis_applied=False,
    )

    def write_status():
        (out / "run_status.json").write_text(
            json.dumps(status, indent=2, allow_nan=False), encoding="utf-8"
        )

    write_status()
    print("[OUTPUT]", out.resolve(), flush=True)
    try:
        run_partition_pipeline()
        summary = pd.read_csv(out / "10_partition_summary.csv").iloc[0].to_dict()
        _export_boundary_review(out, summary)
        edgefile = out / "partition_edges.json"
        edge = json.loads(edgefile.read_text("utf-8")) if edgefile.exists() else {}
        partition_review = bool(int(summary.get("partition_review_required", 0)))
        edge_review = bool(edge.get("plots_needing_edge_review", 0))
        status.update(
            status="completed_needs_review" if partition_review or edge_review else "completed",
            n_proposed_plots=int(summary["n_plots"]),
            partition_review_required=partition_review,
            edges_refined=bool(edge.get("enabled", False)),
            edge_plots_changed=int(edge.get("plots_changed", 0)),
            edge_plots_needing_review=int(edge.get("plots_needing_edge_review", 0)),
            partition_assigned_points=int(summary["n_partition_assigned_points"]),
        )
        if partition_review or edge_review:
            (out / "00_review_required.txt").write_text(
                "Inspect separator evidence and 30_partition_edge_refinement.csv before accepting plot edges.\n",
                encoding="utf-8",
            )
        write_status()
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        write_status()
        raise
    return out


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive crop-plot partitioning from global seedling coordinates"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", help="One dedup_keep_global CSV")
    group.add_argument("--batch-root", help="Root directory for batch processing")
    parser.add_argument(
        "--pattern",
        default="dedup_keep_global_*_safe_light_clean.csv",
        help="CSV filename pattern used by --batch-root",
    )
    parser.add_argument(
        "--output",
        help="New output directory; an occupied directory receives a new run subfolder",
    )
    parser.add_argument(
        "--no-edge-refinement", action="store_true", help="Export initial partition edges"
    )
    args = parser.parse_args()
    if args.no_edge_refinement:
        CONFIG["ENABLE_PARTITION_EDGE_REFINEMENT"] = False
    if args.batch_root:
        root = Path(args.batch_root)
        inputs = sorted(root.rglob(args.pattern))
        if not inputs:
            parser.error("No input CSV found under --batch-root")
        destination = Path(args.output) if args.output else root / "partition_results"
        failures = []
        for inp in inputs:
            name = "__".join(inp.relative_to(root).parts[:-2]) or inp.parent.name
            CONFIG.update(INPUT_CSV=str(inp), OUTPUT_DIR=str(destination / name))
            try:
                main()
            except Exception as exc:
                print("[FAILED]", inp, repr(exc), flush=True)
                failures.append(str(inp))
        if failures:
            raise SystemExit(
                "{} dataset(s) failed; inspect their run_status.json".format(len(failures))
            )
    else:
        if args.input:
            CONFIG["INPUT_CSV"] = args.input
        if args.output:
            CONFIG["OUTPUT_DIR"] = args.output
        main()


def export_partition_outputs(out_dir, plot_df, plot_points, cfg):
    """Save partition-stage point groups and figures for subsequent row analysis."""
    assigned = plot_points.loc[partition_point_mask(plot_points)].copy()
    gap_points = plot_points.loc[~partition_point_mask(plot_points)].copy()
    assigned.to_csv(
        out_dir / "06a_partition_assigned_points.csv", index=False, encoding="utf-8-sig"
    )
    gap_points.to_csv(
        out_dir / "06b_partition_gap_points.csv", index=False, encoding="utf-8-sig"
    )
    groups = out_dir / "partitioned_points"
    groups.mkdir(exist_ok=True)
    for row in plot_df.itertuples():
        assigned.loc[assigned.plot_id == row.plot_id].to_csv(
            groups / "plot_{:04d}.csv".format(int(row.plot_id)),
            index=False,
            encoding="utf-8-sig",
        )
    canvas = draw_partition_result(plot_points, plot_df, cfg)
    _write_image_unicode(out_dir / "09_points_with_plot_boundaries.png", canvas)
    _write_image_unicode(out_dir / "09_points_with_dp_boundaries.png", canvas)
    handoff = dict(
        stage="partitioning",
        boundaries="05_plot_boundaries.csv",
        all_partition_points="06_plot_points.csv",
        assigned_points="06a_partition_assigned_points.csv",
        plot_directory="partitioned_points",
        coarse_row_column="row_main",
        coarse_row_role="partition evidence only; downstream row estimation is separate",
        point_quality_filter_applied=False,
        plot_row_fitting_applied=False,
        phenotype_analysis_applied=False,
    )
    (out_dir / "partition_handoff.json").write_text(
        json.dumps(handoff, indent=2), encoding="utf-8"
    )
    return dict(assigned_points=len(assigned), gap_points=len(gap_points))


def draw_partition_result(points, boundaries, cfg):
    """Draw raw partition points and boxes using the exported interval endpoints."""
    frame = points[["s_rot", "t_rot"]].copy()
    tmid = float(frame.t_rot.median())
    markers = pd.DataFrame(
        {"s_rot": boundaries[["s_start", "s_end"]].to_numpy().ravel(), "t_rot": tmid}
    )
    tfm = compute_canvas_transform(pd.concat([frame, markers], ignore_index=True), cfg)
    canvas = np.zeros((tfm["H"], tfm["W"], 3), dtype=np.uint8)
    assigned_mask = partition_point_mask(points)
    radius = int(cfg.get("POINT_RADIUS", 2))
    for index, row in points.iterrows():
        x, y = map_point(row.t_rot, row.s_rot, tfm)
        color = (
            cfg.get("COLOR_KEEP", (0, 255, 0))
            if bool(assigned_mask.loc[index])
            else cfg.get("COLOR_GAP_POINT", (0, 180, 255))
        )
        cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    palette = cfg.get("PLOT_RECT_COLORS", [(0, 0, 255), (0, 255, 255), (255, 255, 255)])
    for row in boundaries.itertuples():
        q = points.loc[(points.plot_id == row.plot_id) & assigned_mask]
        if len(q):
            lo, hi = (float(q.t_rot.min()), float(q.t_rot.max()))
        else:
            lo, hi = (float(frame.t_rot.min()), float(frame.t_rot.max()))
        margin = float(cfg.get("PLOT_BOX_EXTRA_MARGIN_PX", 26.0)) + max(
            float(cfg.get("PLOT_BOX_MIN_SIDE_EXTRA", 20.0)),
            float(cfg.get("PLOT_BOX_SIDE_EXTRA_RATIO", 0.1)) * (hi - lo),
        )
        x1, y1 = map_point(lo - margin, row.s_start, tfm)
        x2, y2 = map_point(hi + margin, row.s_end, tfm)
        color = palette[(int(row.plot_id) - 1) % len(palette)]
        cv2.rectangle(
            canvas,
            (min(x1, x2), min(y1, y2)),
            (max(x1, x2), max(y1, y2)),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "P{}".format(int(row.plot_id)),
            (min(canvas.shape[1] - 70, max(x1, x2) + 8), int((y1 + y2) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


if __name__ == "__main__":
    cli()
