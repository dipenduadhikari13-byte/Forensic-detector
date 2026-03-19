from __future__ import annotations

import cv2
import numpy as np


class FFTAnalyzer:
    def __init__(self, low_freq_radius_ratio: float = 0.08) -> None:
        self.low_freq_radius_ratio = low_freq_radius_ratio

    def analyze(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * self.low_freq_radius_ratio)

        yy, xx = np.ogrid[:h, :w]
        center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

        low_energy = float(np.mean(magnitude[center_mask])) if np.any(center_mask) else 0.0
        high_energy = float(np.mean(magnitude[~center_mask])) if np.any(~center_mask) else 0.0

        ratio = high_energy / (low_energy + 1e-6)
        score = float(np.clip((ratio - 0.5) / 1.8, 0.0, 1.0))

        mag_norm = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        return score, mag_norm
