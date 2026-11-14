"""Refine plot edges from the points and coarse rows used at partitioning."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Parameters:
    local_intervals: int = 20
    support_band: float = 2.0
    isolated_step: float = 3.0
    spacing_margin: float = 0.5
    min_support_rows: int = 2
    min_reference_gaps: int = 3
    row_ids: tuple = (0, 1, 2)
    uncertain_policy: str = "visible_extent"


def endpoint_evidence(points, side, cfg):
    """Outermost accepted point plus multirow and local-spacing evidence."""
    row_records = []
    for row in cfg.row_ids:
        v = np.sort(points.loc[points.row_main == row, "s_rot"].to_numpy(float))
        if not len(v):
            continue
        v = v[: cfg.local_intervals + 1] if side == "start" else v[-cfg.local_intervals - 1 :]
        d = np.diff(v)
        d = d[d > 1e-07]
        edge = v[0] if side == "start" else v[-1]
        outer_step = (
            (v[1] - v[0] if side == "start" else v[-1] - v[-2]) if len(v) > 1 else np.inf
        )
        row_records.append(
            dict(
                row=int(row),
                edge=float(edge),
                n=len(v),
                median=float(np.median(d)) if len(d) else np.nan,
                q90=float(np.quantile(d, 0.9)) if len(d) else np.nan,
                outer_step=float(outer_step),
                intervals=len(d),
            )
        )
    if not row_records:
        return dict(
            anchor=None,
            spacing=0.0,
            tolerance=0.0,
            support=0,
            strong=False,
            candidate=None,
            rows=row_records,
        )
    anchor = (min if side == "start" else max)((r["edge"] for r in row_records))
    usable = [r for r in row_records if r["intervals"] >= 4]
    spacing = float(np.median([r["median"] for r in usable])) if usable else 0.0
    tolerance = float(np.median([r["q90"] for r in usable])) if usable else 0.0
    for r in row_records:
        r["supports_edge"] = bool(
            tolerance > 0
            and r["intervals"] >= 4
            and (abs(r["edge"] - anchor) <= cfg.support_band * tolerance)
            and (r["outer_step"] <= cfg.isolated_step * tolerance)
        )
    support = sum((r["supports_edge"] for r in row_records))
    candidate = anchor + (-1 if side == "start" else 1) * cfg.spacing_margin * spacing
    return dict(
        anchor=float(anchor),
        spacing=spacing,
        tolerance=tolerance,
        support=int(support),
        strong=bool(support >= cfg.min_support_rows),
        candidate=float(candidate),
        rows=row_records,
    )


def gap_reference(junctions, target, cfg):
    """Same-strip, leave-one-junction-out reference; never pool different strips."""
    refs = [
        (j["index"], j["point_gap"])
        for j in junctions
        if j["index"] != target
        and j["upper"]["strong"]
        and j["lower"]["strong"]
        and (j["point_gap"] > 0)
    ]
    if len(refs) < cfg.min_reference_gaps:
        return None
    values = np.array([x[1] for x in refs], float)
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if median <= 0:
        return None
    q10, q90 = np.quantile(values, [0.1, 0.9])
    return dict(
        median=median,
        mad_scale=scale,
        low=float(q10),
        high=float(q90),
        ids=[int(x[0]) for x in refs],
        n=len(refs),
    )


def refine(boundaries, counted, cfg=Parameters()):
    b = boundaries.sort_values("s_start").reset_index(drop=True).copy()
    if b.plot_id.duplicated().any() or (b.s_end <= b.s_start).any():
        raise ValueError("Invalid plot identifiers or original intervals")
    if not set(counted.plot_id).issubset(set(b.plot_id)):
        raise ValueError("Assigned partition points refer to unknown plots")
    if not np.isfinite(counted[["s_rot", "t_rot"]].to_numpy(float)).all():
        raise ValueError("Assigned partition point coordinates must be finite")
    records = []
    for r in b.itertuples():
        p = counted[counted.plot_id == r.plot_id]
        if not len(p):

            def empty_evidence(value):
                return dict(
                    anchor=float(value),
                    candidate=float(value),
                    spacing=0.0,
                    tolerance=0.0,
                    support=0,
                    strong=False,
                    rows=[],
                )

            a, z = (empty_evidence(r.s_start), empty_evidence(r.s_end))
        else:
            a = endpoint_evidence(p, "start", cfg)
            z = endpoint_evidence(p, "end", cfg)
        records.append(
            dict(
                plot_id=int(r.plot_id),
                old_start=float(r.s_start),
                old_end=float(r.s_end),
                start=min(float(r.s_start), a["anchor"]),
                end=max(float(r.s_end), z["anchor"]),
                start_evidence=a,
                end_evidence=z,
                start_method="retained_needs_review",
                end_method="retained_needs_review",
                start_review=True,
                end_review=True,
                containment_fixed=bool(
                    a["anchor"] < r.s_start - 1e-06 or z["anchor"] > r.s_end + 1e-06
                ),
            )
        )
    junctions = []
    for index, (u, d) in enumerate(zip(records[:-1], records[1:]), 1):
        eu, ed = (u["end_evidence"], d["start_evidence"])
        junctions.append(
            dict(
                index=index,
                upper_id=u["plot_id"],
                lower_id=d["plot_id"],
                upper=eu,
                lower=ed,
                point_gap=ed["candidate"] - eu["candidate"],
                observed_blank=ed["anchor"] - eu["anchor"],
            )
        )
    for j, u, d in zip(junctions, records[:-1], records[1:]):
        eu, ed = (j["upper"], j["lower"])
        ref = gap_reference(junctions, j["index"], cfg)
        j["reference"] = ref
        p, q = (eu["anchor"], ed["anchor"])
        if q <= p:
            raise ValueError(
                f"Longitudinal overlap between plots {u['plot_id']} and {d['plot_id']}; review assignment"
            )
        u["end"] = min(max(u["end"], p), q)
        d["start"] = min(max(d["start"], p), q)
        if u["end"] > d["start"]:
            middle = (u["end"] + d["start"]) / 2
            u["end"] = d["start"] = middle
        status = "insufficient_row_or_gap_evidence"
        uc, dc = (eu["candidate"], ed["candidate"])
        independent_side = ""
        # A supported edge can move without inventing the opposite weak edge.
        if eu["strong"] and not ed["strong"] and p <= uc <= d["start"] and uc < q:
            u["end"] = uc
            u["end_method"] = "independent_multirow_edge"
            u["end_review"] = False
            independent_side = "upper"
        elif ed["strong"] and not eu["strong"] and u["end"] <= dc <= q and dc > p:
            d["start"] = dc
            d["start_method"] = "independent_multirow_edge"
            d["start_review"] = False
            independent_side = "lower"
        if eu["strong"] and ed["strong"] and (uc < dc):
            wide = bool(
                ref
                and j["point_gap"]
                > ref["high"] + max(3 * ref["mad_scale"], eu["tolerance"], ed["tolerance"])
            )
            if wide:
                status = "wide_blank_ambiguous"
            else:
                u["end"], d["start"] = (uc, dc)
                u["end_method"] = d["start_method"] = "multirow_edge"
                u["end_review"] = d["start_review"] = False
                status = "two_sided_multirow"
        elif eu["strong"] != ed["strong"] and ref:
            width = ref["median"]
            if eu["strong"]:
                uc, dc = (eu["candidate"], eu["candidate"] + width)
                feasible = p <= uc and dc <= q and (eu["candidate"] + ref["high"] <= q)
            else:
                dc, uc = (ed["candidate"], ed["candidate"] - width)
                feasible = p <= uc and dc <= q and (ed["candidate"] - ref["high"] >= p)
            if feasible and uc < dc:
                u["end"], d["start"] = (uc, dc)
                u["end_method"] = (
                    "multirow_edge" if eu["strong"] else "same_strip_gap_inference"
                )
                d["start_method"] = (
                    "multirow_edge" if ed["strong"] else "same_strip_gap_inference"
                )
                u["end_review"] = d["start_review"] = True
                if eu["strong"]:
                    d["start_gap_reference_range"] = [uc + ref["low"], uc + ref["high"]]
                else:
                    u["end_gap_reference_range"] = [dc - ref["high"], dc - ref["low"]]
                status = "one_sided_gap_inference"
            else:
                status = "gap_reference_conflicts_with_points"
        elif eu["strong"] and ed["strong"]:
            status = "insufficient_room_for_edge_margins"
        j["evidence_status"] = status
        j["independent_side"] = independent_side
        if independent_side and status != "one_sided_gap_inference":
            status = independent_side + "_supported_other_retained"
        j["status"] = status
        j["new_upper_end"], j["new_lower_start"] = (u["end"], d["start"])
        j["old_upper_end"], j["old_lower_start"] = (u["old_end"], d["old_start"])
    for r, side in [(records[0], "start"), (records[-1], "end")]:
        ev = r[side + "_evidence"]
        if ev["strong"]:
            r[side] = ev["candidate"]
            r[side + "_method"] = "outer_observed_support"
        else:
            r[side + "_method"] = "outer_weak_retained"
        r[side + "_review"] = True
    # In visible-extent mode, an unresolved edge may shrink inward to the
    # assigned points. It never removes points or expands an uncertain gap.
    if cfg.uncertain_policy == "visible_extent":
        for r in records:
            for side in ("start", "end"):
                evidence = r[side + "_evidence"]
                method = r[side + "_method"]
                eligible = method in (
                    "retained_needs_review",
                    "outer_weak_retained",
                    "same_strip_gap_inference",
                )
                if not eligible or not evidence["rows"]:
                    continue
                candidate = evidence["candidate"]
                proposed = (
                    max(r[side], candidate) if side == "start" else min(r[side], candidate)
                )
                if abs(proposed - r[side]) > 1e-7:
                    r[side] = proposed
                    r[side + "_method"] = "visible_extent_fallback"
                    r[side + "_review"] = True
        for j, upper, lower in zip(junctions, records[:-1], records[1:]):
            if (
                upper["end_method"] == "visible_extent_fallback"
                or lower["start_method"] == "visible_extent_fallback"
            ):
                j["status"] = "visible_extent_needs_review"
            j["new_upper_end"], j["new_lower_start"] = upper["end"], lower["start"]
    for j, upper, lower in zip(junctions, records[:-1], records[1:]):
        j["upper_method"], j["lower_method"] = upper["end_method"], lower["start_method"]
    for r in records:
        if (
            r["start"] > r["start_evidence"]["anchor"] + 1e-06
            or r["end"] < r["end_evidence"]["anchor"] - 1e-06
        ):
            raise AssertionError("Containment invariant failed")
        if r["end"] <= r["start"]:
            raise AssertionError("Positive length invariant failed")
    for a, z in zip(records[:-1], records[1:]):
        if a["end"] > z["start"] + 1e-06:
            raise AssertionError("Non-overlap invariant failed")
    return (records, junctions)


def parameters_from_config(config):
    rows = int(config.get("MAIN_ROWS", 3))
    p = Parameters(
        local_intervals=int(config.get("EDGE_LOCAL_INTERVALS", 20)),
        support_band=float(config.get("EDGE_SUPPORT_BAND", 2.0)),
        isolated_step=float(config.get("EDGE_ISOLATED_STEP", 3.0)),
        spacing_margin=float(config.get("EDGE_SPACING_MARGIN", 0.5)),
        min_support_rows=int(config.get("EDGE_MIN_SUPPORT_ROWS", 2)),
        min_reference_gaps=int(config.get("EDGE_MIN_REFERENCE_GAPS", 3)),
        row_ids=tuple(range(rows)),
        uncertain_policy=str(config.get("EDGE_UNCERTAIN_POLICY", "visible_extent")),
    )
    if (
        p.uncertain_policy not in ("visible_extent", "retain_initial")
        or rows < 1
        or p.local_intervals < 4
        or (not 1 <= p.min_support_rows <= rows)
        or (p.min_reference_gaps < 1)
        or (p.support_band <= 0)
        or (p.isolated_step <= 0)
        or (p.spacing_margin < 0)
        or (
            not all(
                (math.isfinite(v) for v in (p.support_band, p.isolated_step, p.spacing_margin))
            )
        )
    ):
        raise ValueError("Invalid endpoint refinement parameters")
    return p


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def partition_point_mask(points):
    """Use initial plot ownership and gap flags, without applying a point-quality filter."""
    mask = pd.to_numeric(points["plot_id"], errors="coerce").fillna(-1) > 0
    if "is_counted_plot_point" in points:
        mask &= (
            points["is_counted_plot_point"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "1.0"])
        )
    if "is_boundary_gap_point" in points:
        mask &= (
            ~points["is_boundary_gap_point"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "1.0"])
        )
    return mask


def refine_partition_edges(out_dir, boundaries, partition_points, config):
    """Refine provisional plot intervals before partition tables and figures are saved."""
    required = {"plot_id", "row_main", "s_rot", "t_rot"}
    if not required.issubset(partition_points):
        raise ValueError(
            "Partition edge refinement requires plot ownership, coordinates and coarse row_main labels"
        )
    assigned = partition_points.loc[partition_point_mask(partition_points)].copy()
    enabled = bool(config.get("ENABLE_PARTITION_EDGE_REFINEMENT", True))
    if not enabled:
        return dict(
            enabled=False,
            plots=len(boundaries),
            partition_assigned_points=len(assigned),
            plots_changed=0,
            plots_needing_edge_review=0,
        )
    params = parameters_from_config(config)
    if not assigned.row_main.isin(params.row_ids).all():
        raise ValueError("An assigned point has an invalid coarse partition-row label")
    if "tmp_id" in partition_points and (
        partition_points.tmp_id.isna().any() or partition_points.tmp_id.duplicated().any()
    ):
        raise ValueError("Partition points require unique, nonmissing tmp_id values")
    original = boundaries.copy(deep=True)
    records, junctions = refine(original, assigned, params)
    audits = []
    for r in records:
        pid = r["plot_id"]
        index = boundaries.index[boundaries.plot_id == pid]
        if len(index) != 1:
            raise ValueError("Duplicate or missing plot boundary")
        updates = dict(
            s_start_initial=r["old_start"],
            s_end_initial=r["old_end"],
            s_start=r["start"],
            s_end=r["end"],
            length=r["end"] - r["start"],
            partition_point_count=int((assigned.plot_id == pid).sum()),
            edge_start_method=r["start_method"],
            edge_end_method=r["end_method"],
            edge_review_required=bool(r["start_review"] or r["end_review"]),
        )
        for key, value in updates.items():
            if key not in boundaries:
                boundaries[key] = pd.Series(
                    index=boundaries.index,
                    dtype=object if isinstance(value, (str, bool)) else float,
                )
            boundaries.loc[index, key] = value
        audits.append(
            dict(
                plot_id=pid,
                **updates,
                shift_start=r["start"] - r["old_start"],
                shift_end=r["end"] - r["old_end"],
                start_support=r["start_evidence"]["support"],
                end_support=r["end_evidence"]["support"],
                initial_containment_fixed=r["containment_fixed"],
            )
        )
    joint_rows = []
    for j in junctions:
        ref = j["reference"] or {}
        upper = boundaries.index[boundaries.plot_id == j["upper_id"]]
        lower = boundaries.index[boundaries.plot_id == j["lower_id"]]
        if "next_gap_lo" in boundaries:
            boundaries.loc[upper, "next_gap_lo"] = j["new_upper_end"]
        if "prev_gap_hi" in boundaries:
            boundaries.loc[lower, "prev_gap_hi"] = j["new_lower_start"]
        joint_rows.append(
            dict(
                upper_plot=j["upper_id"],
                lower_plot=j["lower_id"],
                status=j["status"],
                evidence_status=j["evidence_status"],
                independent_side=j["independent_side"],
                upper_method=j["upper_method"],
                lower_method=j["lower_method"],
                old_upper_end=j["old_upper_end"],
                old_lower_start=j["old_lower_start"],
                new_upper_end=j["new_upper_end"],
                new_lower_start=j["new_lower_start"],
                upper_support=j["upper"]["support"],
                lower_support=j["lower"]["support"],
                reference_gaps=ref.get("n", 0),
                reference_width=ref.get("median"),
            )
        )
    out_dir = Path(out_dir)
    original.to_csv(
        out_dir / "05a_plot_boundaries_before_edge_refinement.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(audits).to_csv(
        out_dir / "30_partition_edge_refinement.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        joint_rows,
        columns=[
            "upper_plot",
            "lower_plot",
            "status",
            "evidence_status",
            "independent_side",
            "upper_method",
            "lower_method",
            "old_upper_end",
            "old_lower_start",
            "new_upper_end",
            "new_lower_start",
            "upper_support",
            "lower_support",
            "reference_gaps",
            "reference_width",
        ],
    ).to_csv(out_dir / "31_partition_junctions.csv", index=False, encoding="utf-8-sig")
    report = dict(
        enabled=True,
        stage="partitioning",
        parameters=asdict(params),
        plots=len(records),
        partition_assigned_points=len(assigned),
        point_membership_changed=False,
        row_hint_column="row_main",
        point_source="partition assignment before plot-level filtering",
        plots_changed=sum(
            abs(r["start"] - r["old_start"]) > 1e-6 or abs(r["end"] - r["old_end"]) > 1e-6
            for r in records
        ),
        plots_needing_edge_review=sum(r["start_review"] or r["end_review"] for r in records),
        coordinate_units="reconstructed input coordinates",
        edges=records,
    )
    (out_dir / "partition_edges.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
