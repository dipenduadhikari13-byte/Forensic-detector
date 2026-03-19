from __future__ import annotations

import cv2
import numpy as np


class AISignatureAnalyzer:
    def analyze(self, image_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        resized = cv2.resize(gray, (min(512, w), min(512, h)))
        fft = np.fft.fftshift(np.fft.fft2(resized))
        magnitude = np.log1p(np.abs(fft))

        yy, xx = np.indices(magnitude.shape)
        cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        r_max = int(radius.max())
        radial_profile = np.zeros(r_max + 1, dtype=np.float32)
        counts = np.zeros(r_max + 1, dtype=np.float32)

        r_int = np.clip(radius.astype(np.int32), 0, r_max)
        np.add.at(radial_profile, r_int, magnitude)
        np.add.at(counts, r_int, 1)
        radial_profile /= (counts + 1e-6)

        high_band = radial_profile[int(r_max * 0.35): int(r_max * 0.85)]
        low_band = radial_profile[: int(r_max * 0.15)]
        spectral_ratio = float(np.mean(high_band) / (np.mean(low_band) + 1e-6)) if len(high_band) else 0.0

        median = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
        residual = np.abs(gray - median)
        residual_entropy = self._entropy(residual)

        chans = cv2.split(image_bgr)
        channel_corr = float(np.mean([
            np.corrcoef(chans[0].ravel(), chans[1].ravel())[0, 1],
            np.corrcoef(chans[1].ravel(), chans[2].ravel())[0, 1],
            np.corrcoef(chans[0].ravel(), chans[2].ravel())[0, 1],
        ]))
        channel_corr = 0.0 if np.isnan(channel_corr) else channel_corr

        spectral_score = float(np.clip((spectral_ratio - 0.55) / 1.2, 0.0, 1.0))
        entropy_score = float(np.clip((3.8 - residual_entropy) / 2.4, 0.0, 1.0))
        corr_score = float(np.clip((channel_corr - 0.83) / 0.16, 0.0, 1.0))

        score = float(np.clip(
            0.45 * spectral_score +
            0.35 * entropy_score +
            0.20 * corr_score,
            0.0,
            1.0,
        ))

        heatmap = cv2.normalize(residual, None, 0, 1, cv2.NORM_MINMAX)
        return {
            "score": score,
            "components": {
                "spectral": round(spectral_score, 4),
                "residual_entropy": round(entropy_score, 4),
                "channel_corr": round(corr_score, 4),
            },
            "heatmap": heatmap,
        }

    @staticmethod
    def _entropy(arr: np.ndarray) -> float:
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        hist = cv2.calcHist([arr_u8], [0], None, [256], [0, 256]).flatten()
        p = hist / (hist.sum() + 1e-9)
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())
