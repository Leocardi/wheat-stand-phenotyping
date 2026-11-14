"""Settings for translation-based strip registration. Distances use image pixels."""

INPUT_OUTPUT = {
    "INPUT_DIR": "data/frames",
    "OUTPUT_DIR": "results/strip",
    "IMG_EXTS": {".jpg", ".tif", ".tiff", ".png", ".bmp", ".jpeg"},
    "ROTATE_DEG": 180,
}

REGISTRATION = {
    "REG_SCALE": 0.25,
    "MARGIN_RATIO": 0.08,
    "PHASE_MIN_RESP": 0.15,
    "DX_CLIP": 120,
    "MAX_DY_FRAC": 0.9,
}

PREPROCESSING = {
    "USE_CLAHE": True,
    "USE_VEG_MASK": True,
    "USE_GLARE_SOFT": True,
    "GLARE_V_TH": 245,
    "GLARE_S_TH": 60,
    "GLARE_REDUCE": 0.3,
}

NCC_SEARCH = {
    "USE_NCC_FALLBACK": True,
    "NCC_MIN_SCORE": 0.25,
    "SEARCH_DX": 30,
    "SEARCH_DY": 180,
    "TEMPLATE_W_RATIO": 0.7,
    "TEMPLATE_H_RATIO": 0.35,
}

KEYFRAME_SELECTION = {
    "KEY_OVERLAP_TARGET": 0.65,
    "MIN_OVERLAP_GUARD": 0.5,
    "Y_WINDOW_FRAC": 0.2,
    "CAND_MAX": 12,
    "USE_CONTINUITY_AWARE_DET_KEYFRAMES": True,
}

MOTION_CHECKS = {
    "VALIDATE_FRAME_SHIFT": True,
    "VALIDATE_ANCHOR_SHIFT": True,
    "MIN_FRAME_FORWARD_PX": 1.0,
    "MIN_ANCHOR_FORWARD_PX": 30.0,
    "MAX_ANCHOR_FORWARD_FRAC": 0.85,
    "ANCHOR_STEP_TOL_FRAC": 0.45,
    "ANCHOR_STEP_TOL_RATIO": 0.9,
    "ANCHOR_DX_MAX": 240.0,
}

IMAGE_EXPORT = {
    "EXPORT_KEYFRAME_IMAGES": True,
    "KEYFRAMES_DIRNAME": "keyframes_images",
    "GEOM_KEYFRAMES_DIRNAME": "geom_keyframes_images",
    "SAVE_ROTATED_KEYFRAMES": True,
    "JPEG_QUALITY": 95,
    "PNG_COMPRESSION": 3,
}

PREVIEW = {
    "RENDER_PREVIEW": True,
    "RENDER_SCALE": 0.2,
    "FEATHER_PX": 30,
    "CHUNK_HEIGHT_FULL": 60000,
    "EXPORT_DET_KEYFRAME_PREVIEW": True,
}

CONFIG = {
    **INPUT_OUTPUT,
    **REGISTRATION,
    **PREPROCESSING,
    **NCC_SEARCH,
    **KEYFRAME_SELECTION,
    **MOTION_CHECKS,
    **IMAGE_EXPORT,
    **PREVIEW,
}
