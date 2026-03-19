from __future__ import annotations

import cv2
import numpy as np


class NoiseAnalyzer:
    def analyze(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = cv2.absdiff(gray, blur)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        local_texture = cv2.GaussianBlur(np.abs(lap), (7, 7), 0)
        local_texture = cv2.normalize(local_texture, None, 0, 1, cv2.NORM_MINMAX)

        residual_norm = cv2.normalize(residual, None, 0, 1, cv2.NORM_MINMAX)
        inconsistency = cv2.absdiff(residual_norm, local_texture)

        score = float(np.clip(np.percentile(inconsistency, 90), 0.0, 1.0))
        return score, inconsistency
