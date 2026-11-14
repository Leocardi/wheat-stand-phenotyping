"""Filesystem, input-contract and run-record utilities."""

from __future__ import annotations
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import cv2
import numpy as np
import pandas as pd


def read_image(path, flags=cv2.IMREAD_COLOR, expected=None):
    path = Path(path)
    try:
        image = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    except (OSError, cv2.error) as exc:
        raise OSError("Cannot read image: {}".format(path)) from exc
    if image is None:
        raise OSError("Cannot read image: {}".format(path))
    if expected is not None and image.shape[:2] != expected:
        raise ValueError(
            "Image dimensions differ from the keyframe table: {} has {}, expected {}".format(
                path, image.shape[:2], expected
            )
        )
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = (
        [cv2.IMWRITE_JPEG_QUALITY, 95]
        if path.suffix.lower() in (".jpg", ".jpeg")
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    ok, encoded = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise OSError("Cannot encode image: {}".format(path))
    encoded.tofile(path)


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def start_run(output, stage, config, program):
    out = Path(output).resolve()
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        out = out / ("run_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=True)
    status = dict(
        status="running",
        stage=stage,
        output_dir=str(out),
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        pandas=pd.__version__,
        opencv=cv2.__version__,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(program),
                Path(__file__),
                Path(__file__).with_name("core_seam_config.py"),
            ]
        },
    )
    effective = dict(config, OUT_DIR=str(out))
    save_json(out / "run_config.json", effective)
    save_json(out / "run_status.json", status)
    print("[OUTPUT] {}".format(out), flush=True)
    return out, status, effective


def finish_run(out, status, error=None, summary=None):
    status.update(
        status="failed" if error else "completed",
        finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    if error:
        status["error"] = "{}: {}".format(type(error).__name__, error)
    if summary:
        status.update(summary)
    save_json(out / "run_status.json", status)


def validate_keyframes(df, shape=None):
    """Require a uniform-size source-frame table for translation-only crops."""
    if len(df) < 2:
        raise ValueError("At least two keyframes are required")
    if df["file"].duplicated().any() or df["k"].duplicated().any():
        raise ValueError("Keyframe names and indices must be unique")
    if not np.isfinite(df[["fx1", "fy1", "img_w", "img_h"]].to_numpy(float)).all():
        raise ValueError("Non-finite keyframe coordinates or dimensions")
    if (df[["img_w", "img_h"]] <= 0).any().any():
        raise ValueError("Image dimensions must be positive")
    if df[["img_w", "img_h"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Core/seam construction requires equal-sized source frames")
    if (
        shape is not None
        and (int(df.iloc[0]["img_h"]), int(df.iloc[0]["img_w"])) != shape
    ):
        raise ValueError("Source image dimensions do not match metadata")


def validate_metadata(meta, kind):
    required = {"crop_file", "crop_global_x1", "crop_global_y1", "crop_w", "crop_h"}
    required.update(
        {"src_file", "core_left_s", "core_right_s"}
        if kind == "core"
        else {"file_left", "file_right", "seam_left_s", "seam_right_s"}
    )
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            "Missing {} metadata columns: {}".format(kind, ", ".join(sorted(missing)))
        )
    if meta["crop_file"].isna().any() or meta["crop_file"].duplicated().any():
        raise ValueError(
            "Crop filenames must be present and unique in {} metadata".format(kind)
        )
    if not np.isfinite(
        meta[["crop_global_x1", "crop_global_y1", "crop_w", "crop_h"]].to_numpy(float)
    ).all():
        raise ValueError("Non-finite crop origins or dimensions")
    if (meta[["crop_w", "crop_h"]] <= 0).any().any():
        raise ValueError("Crop dimensions must be positive")


def validate_detection_join(detections, metadata, kind):
    unknown = set(detections["file"]) - set(metadata["crop_file"])
    if unknown:
        raise ValueError(
            "{} detection filenames do not match metadata: {}".format(
                kind, ", ".join(sorted(unknown)[:5])
            )
        )
