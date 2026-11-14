"""Checks for partition-stage edge refinement and the downstream handoff."""

import ast
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import partition_edges as edges
import plot_partition as program


def fixture(n=6):
    bounds, points = [], []
    for pid in range(1, n + 1):
        start = (pid - 1) * 1100.0
        bounds.append(dict(plot_id=pid, s_start=start, s_end=start + 1000))
        for row in range(3):
            for s in np.arange(start + 20, start + 1000, 20):
                points.append(
                    dict(
                        plot_id=pid,
                        row_main=row,
                        s_rot=s,
                        t_rot=row * 100.0,
                        tmp_id=len(points),
                        is_counted_plot_point=True,
                        is_boundary_gap_point=False,
                    )
                )
    return pd.DataFrame(bounds), pd.DataFrame(points)


class PartitionEdges(unittest.TestCase):
    def test_common_edges_contain_partition_points(self):
        bounds, points = fixture()
        records, junctions = edges.refine(bounds, points)
        self.assertTrue(all(j["status"] == "two_sided_multirow" for j in junctions))
        for r in records:
            q = points[points.plot_id == r["plot_id"]]
            self.assertLessEqual(r["start"], q.s_rot.min())
            self.assertGreaterEqual(r["end"], q.s_rot.max())

    def test_refinement_does_not_change_partition_points(self):
        bounds, points = fixture(2)
        snapshot = points.copy(deep=True)
        with tempfile.TemporaryDirectory() as temp:
            report = edges.refine_partition_edges(temp, bounds, points, {})
        pd.testing.assert_frame_equal(points, snapshot)
        self.assertEqual(report["partition_assigned_points"], len(points))
        self.assertEqual(report["stage"], "partitioning")
        self.assertNotIn("measurement_s_start", bounds)

    def test_final_row_and_noise_fields_do_not_control_refinement(self):
        bounds, points = fixture(2)
        extra = points.assign(nearest_col_id=-99, is_noise=True, plot_keep_main3=False)
        a, b = bounds.copy(), bounds.copy()
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            edges.refine_partition_edges(left, a, points, {})
            edges.refine_partition_edges(right, b, extra, {})
        pd.testing.assert_frame_equal(a, b)

    def test_gap_assignment_is_distinct_from_quality_filtering(self):
        bounds, points = fixture(1)
        points.loc[0, "is_boundary_gap_point"] = True
        points.loc[0, "is_counted_plot_point"] = False
        before = points.copy()
        with tempfile.TemporaryDirectory() as temp:
            report = edges.refine_partition_edges(temp, bounds, points, {})
        self.assertEqual(report["partition_assigned_points"], len(points) - 1)
        pd.testing.assert_frame_equal(points, before)

    def test_conservative_mode_retains_common_edges(self):
        bounds, points = fixture()
        points = points[
            ~((points.plot_id == 2) & points.row_main.isin([1, 2]) & (points.s_rot < 1420))
        ]
        records, junctions = edges.refine(
            bounds, points, edges.Parameters(uncertain_policy="retain_initial")
        )
        self.assertEqual(junctions[0]["status"], "one_sided_gap_inference")
        self.assertLess(records[1]["start"], 1250)

    def test_shared_missingness_is_flagged(self):
        bounds, points = fixture()
        points = points[~((points.plot_id == 2) & (points.s_rot < 1320))]
        records, junctions = edges.refine(
            bounds, points, edges.Parameters(uncertain_policy="retain_initial")
        )
        self.assertEqual(junctions[0]["status"], "wide_blank_ambiguous")
        self.assertEqual(records[1]["start"], 1100)

    def test_short_strip_has_no_invented_reference(self):
        bounds, points = fixture(2)
        _, junctions = edges.refine(bounds, points)
        self.assertIsNone(junctions[0]["reference"])

    def test_reference_excludes_the_target_junction(self):
        bounds, points = fixture()
        _, junctions = edges.refine(bounds, points)
        for j in junctions:
            self.assertNotIn(j["index"], j["reference"]["ids"])

    def test_scale_and_translation(self):
        bounds, points = fixture()
        original, _ = edges.refine(bounds, points)
        bounds[["s_start", "s_end"]] = bounds[["s_start", "s_end"]] * 7.3 - 15300
        points.s_rot = points.s_rot * 7.3 - 15300
        moved, _ = edges.refine(bounds, points)
        for a, b in zip(original, moved):
            self.assertAlmostEqual(a["start"] * 7.3 - 15300, b["start"], places=7)
            self.assertAlmostEqual(a["end"] * 7.3 - 15300, b["end"], places=7)

    def test_empty_plot_keeps_initial_edges(self):
        bounds, points = fixture(1)
        records, _ = edges.refine(bounds, points.iloc[:0])
        self.assertEqual(records[0]["start"], 0)
        self.assertEqual(records[0]["end"], 1000)

    def test_conflicting_point_ownership_is_reported(self):
        bounds, points = fixture(2)
        points.loc[0, "s_rot"] = 1150
        with self.assertRaisesRegex(ValueError, "overlap"):
            edges.refine(bounds, points)

    def test_invalid_coarse_row_is_reported(self):
        bounds, points = fixture(1)
        points.loc[0, "row_main"] = -1
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "coarse partition-row"):
                edges.refine_partition_edges(temp, bounds, points, {})

    def test_disabled_refinement_preserves_initial_intervals(self):
        bounds, points = fixture(2)
        before = bounds.copy()
        with tempfile.TemporaryDirectory() as temp:
            result = edges.refine_partition_edges(
                temp, bounds, points, {"ENABLE_PARTITION_EDGE_REFINEMENT": False}
            )
            self.assertFalse(list(Path(temp).iterdir()))
        self.assertFalse(result["enabled"])
        pd.testing.assert_frame_equal(bounds, before)

    def test_edge_exports_contain_no_phenotype_tables(self):
        bounds, points = fixture(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            edges.refine_partition_edges(root, bounds, points, {})
            self.assertEqual(
                {p.name for p in root.iterdir()},
                {
                    "05a_plot_boundaries_before_edge_refinement.csv",
                    "30_partition_edge_refinement.csv",
                    "31_partition_junctions.csv",
                    "partition_edges.json",
                },
            )
            for p in root.glob("*.csv"):
                columns = set(pd.read_csv(p).columns)
                self.assertTrue(columns.isdisjoint({"MG", "NMG", "MG_row", "NMG_row"}))

    def test_partition_export_is_a_raw_point_handoff(self):
        import json

        bounds, points = fixture(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            edges.refine_partition_edges(root, bounds, points, {})
            result = program.export_partition_outputs(root, bounds, points, program.CONFIG)
            saved = pd.read_csv(root / "06a_partition_assigned_points.csv")
            pd.testing.assert_frame_equal(saved, points)
            handoff = json.loads((root / "partition_handoff.json").read_text("utf-8"))
            self.assertFalse(handoff["point_quality_filter_applied"])
            self.assertFalse(handoff["plot_row_fitting_applied"])
            self.assertFalse(handoff["phenotype_analysis_applied"])
            self.assertEqual(result["assigned_points"], len(points))
            self.assertFalse((root / "13_plot_points_clean.csv").exists())
            self.assertEqual(len(list((root / "partitioned_points").glob("*.csv"))), 2)

    def test_edge_hook_precedes_partition_csv_export(self):
        tree = ast.parse(inspect.getsource(program.run_partition_pipeline))
        statements = tree.body[0].body
        build = refine = export = None
        for i, node in enumerate(statements):
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
            if any(
                isinstance(c.func, ast.Name) and c.func.id == "build_plot_boundaries_from_edges"
                for c in calls
            ):
                build = i
            if any(
                isinstance(c.func, ast.Name) and c.func.id == "refine_partition_edges"
                for c in calls
            ):
                refine = i
            if any(
                isinstance(n, ast.Constant) and n.value == "05_plot_boundaries.csv"
                for n in ast.walk(node)
            ):
                export = i
        self.assertLess(build, refine)
        self.assertLess(refine, export)

    def test_post_partition_processing_is_not_part_of_the_program(self):
        for name in (
            "compat_filter_noise_by_plot_columns",
            "compat_build_column_models",
            "export_point_layout_outputs",
        ):
            self.assertFalse(hasattr(program, name))
        for name in ("compute_gap_metrics", "row_gap_metrics", "counted_points"):
            self.assertFalse(hasattr(edges, name))

    def test_supported_upper_edge_moves_without_a_gap_reference(self):
        bounds, points = fixture(2)
        points = points[
            ~((points.plot_id == 2) & points.row_main.isin([1, 2]) & (points.s_rot < 1420))
        ]
        records, junctions = edges.refine(
            bounds, points, edges.Parameters(uncertain_policy="retain_initial")
        )
        self.assertEqual(records[0]["end"], 990)
        self.assertEqual(records[0]["end_method"], "independent_multirow_edge")
        self.assertEqual(records[1]["start"], 1100)
        self.assertEqual(junctions[0]["independent_side"], "upper")

    def test_supported_lower_edge_moves_without_a_gap_reference(self):
        bounds, points = fixture(2)
        points = points[
            ~((points.plot_id == 1) & points.row_main.isin([1, 2]) & (points.s_rot > 800))
        ]
        records, junctions = edges.refine(
            bounds, points, edges.Parameters(uncertain_policy="retain_initial")
        )
        self.assertEqual(records[1]["start"], 1110)
        self.assertEqual(records[1]["start_method"], "independent_multirow_edge")
        self.assertEqual(records[0]["end"], 1000)
        self.assertEqual(junctions[0]["independent_side"], "lower")

    def test_visible_fallback_shrinks_a_weak_edge_and_preserves_points(self):
        bounds, points = fixture(2)
        points = points[
            ~((points.plot_id == 2) & points.row_main.isin([1, 2]) & (points.s_rot < 1420))
        ]
        snapshot = points.copy()
        records, _ = edges.refine(bounds, points)
        self.assertEqual(records[1]["start"], 1110)
        self.assertEqual(records[1]["start_method"], "visible_extent_fallback")
        self.assertTrue(records[1]["start_review"])
        pd.testing.assert_frame_equal(points, snapshot)

    def test_visible_policy_explicitly_allows_shared_missingness_to_shrink(self):
        bounds, points = fixture()
        points = points[~((points.plot_id == 2) & (points.s_rot < 1320))]
        records, junctions = edges.refine(bounds, points)
        self.assertEqual(records[1]["start"], 1310)
        self.assertTrue(records[1]["start_review"])
        self.assertEqual(junctions[0]["evidence_status"], "wide_blank_ambiguous")

    def test_invalid_edge_policy_is_reported(self):
        with self.assertRaises(ValueError):
            edges.parameters_from_config({"EDGE_UNCERTAIN_POLICY": "unknown"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
