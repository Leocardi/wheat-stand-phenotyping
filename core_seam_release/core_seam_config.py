"""Defaults for core/seam construction and ownership-based detection fusion."""

BUILD_PATHS = {
    "KEYFRAME_CSV": "results/strip/keyframes.csv",
    "KEYFRAME_IMG_DIR": None,
    "OUT_DIR": "results/core_seam",
}

CROP_GEOMETRY = {
    "STRIP_AXIS_HINT": "y",
    "SEAM_HALF_BAND": 56.0,
    "CORE_CONTEXT": 96.0,
    "SEAM_CONTEXT": 96.0,
    "SEAM_FEATHER": 18.0,
}

CROP_BACKGROUND = {"FILL_OUTSIDE": True, "FILL_VALUE": (0, 0, 0)}

BUILD_CONFIG = {**BUILD_PATHS, **CROP_GEOMETRY, **CROP_BACKGROUND}

MERGE_PATHS = {
    "KEYFRAME_CSV": "results/strip/keyframes.csv",
    "CORE_META_CSV": "results/core_seam/core_crops_meta.csv",
    "SEAM_META_CSV": "results/core_seam/seam_crops_meta.csv",
    "STRIP_META_JSON": "results/core_seam/strip_meta.json",
    "CORE_DET_CSV": "results/core_detections/detections.csv",
    "SEAM_DET_CSV": "results/seam_detections/detections.csv",
    "OUT_DIR": "results/merged",
}

OWNERSHIP_FUSION = {
    "MIN_SCORE": 0.05,
    "EXPECTED_ROWS_PER_STRIP": 3,
    "FALLBACK_MATCH_S": 36.0,
    "FALLBACK_MATCH_T": 20.0,
}

RESIDUAL_CLEANING = {
    "DO_FINAL_LIGHT_CLEAN": True,
    "FINAL_IOF_MIN": 0.92,
    "FINAL_IOU_MIN": 0.55,
    "FINAL_CENTER_DIST_MAX": 8.0,
    "FINAL_AREA_RATIO_MIN": 0.45,
    "FINAL_S_DIST_MAX": 12.0,
    "FINAL_T_DIST_MAX": 8.0,
    "FINAL_REQUIRE_SAME_ROW": True,
}

MERGE_CONFIG = {**MERGE_PATHS, **OWNERSHIP_FUSION, **RESIDUAL_CLEANING}
