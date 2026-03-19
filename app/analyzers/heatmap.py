from __future__ import annotations

import base64

import cv2
import numpy as np


def encode_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray) -> str:
    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be 2D")

    resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    hm_uint8 = np.clip(resized * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image_bgr, 0.65, colored, 0.35, 0)
    ok, buffer = cv2.imencode(".png", overlay)
    if not ok:
        raise ValueError("Failed to encode heatmap image")

    return base64.b64encode(buffer.tobytes()).decode("utf-8")
