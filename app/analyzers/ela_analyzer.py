from __future__ import annotations

import cv2
import numpy as np


class ELAAnalyzer:
    def __init__(self, jpeg_quality: int = 90) -> None:
        self.jpeg_quality = jpeg_quality

    def analyze(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
        if not success:
            raise ValueError("Failed to encode image for ELA")

        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(image_bgr, recompressed)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        gray_float = gray.astype(np.float32)
        normalized = cv2.normalize(gray_float, None, 0, 1, cv2.NORM_MINMAX)

        mean_energy = float(np.mean(normalized))
        p95 = float(np.percentile(normalized, 95))
        score = float(np.clip((0.65 * p95 + 0.35 * mean_energy), 0.0, 1.0))
        return score, normalized
