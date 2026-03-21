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
        spectral_slope = self._spectral_slope(radial_profile)

        median = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
        residual = np.abs(gray - median)
        residual_entropy = self._entropy(residual)

        patch_noise_uniformity, luminance_noise_corr = self._noise_statistics(gray)

        chans = cv2.split(image_bgr)
        channel_corr = float(np.mean([
            np.corrcoef(chans[0].ravel(), chans[1].ravel())[0, 1],
            np.corrcoef(chans[1].ravel(), chans[2].ravel())[0, 1],
            np.corrcoef(chans[0].ravel(), chans[2].ravel())[0, 1],
        ]))
        channel_corr = 0.0 if np.isnan(channel_corr) else channel_corr

        spectral_score = float(np.clip((spectral_ratio - 0.55) / 1.2, 0.0, 1.0))
        slope_score = float(np.clip((spectral_slope - 0.55) / 1.4, 0.0, 1.0))
        entropy_score = float(np.clip((3.8 - residual_entropy) / 2.4, 0.0, 1.0))
        corr_score = float(np.clip((channel_corr - 0.83) / 0.16, 0.0, 1.0))
        uniformity_score = float(np.clip((patch_noise_uniformity - 0.42) / 0.42, 0.0, 1.0))
        luminance_coupling_score = float(np.clip((0.30 - luminance_noise_corr) / 0.30, 0.0, 1.0))

        score = float(np.clip(
            0.26 * spectral_score +
            0.20 * slope_score +
            0.20 * entropy_score +
            0.16 * corr_score +
            0.10 * uniformity_score +
            0.08 * luminance_coupling_score,
            0.0,
            1.0,
        ))

        heatmap = cv2.normalize(residual, None, 0, 1, cv2.NORM_MINMAX)
        return {
            "score": score,
            "components": {
                "spectral": round(spectral_score, 4),
                "spectral_slope": round(slope_score, 4),
                "residual_entropy": round(entropy_score, 4),
                "channel_corr": round(corr_score, 4),
                "noise_uniformity": round(uniformity_score, 4),
                "noise_luminance_coupling": round(luminance_coupling_score, 4),
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

    @staticmethod
    def _spectral_slope(radial_profile: np.ndarray) -> float:
        if radial_profile.size < 8:
            return 0.0
        start = max(1, int(radial_profile.size * 0.08))
        end = max(start + 4, int(radial_profile.size * 0.80))
        y = np.log1p(np.maximum(radial_profile[start:end], 1e-6))
        x = np.arange(y.size, dtype=np.float32)
        if y.size < 4:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(-slope)

    @staticmethod
    def _noise_statistics(gray: np.ndarray, patch_size: int = 32) -> tuple[float, float]:
        h, w = gray.shape
        if h < patch_size or w < patch_size:
            return 0.0, 0.0

        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        residual = np.abs(gray - blur)

        patch_std_values: list[float] = []
        patch_mean_values: list[float] = []

        step = patch_size
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                patch_res = residual[y:y + patch_size, x:x + patch_size]
                patch_gray = gray[y:y + patch_size, x:x + patch_size]
                patch_std_values.append(float(np.std(patch_res)))
                patch_mean_values.append(float(np.mean(patch_gray)))

        if len(patch_std_values) < 6:
            return 0.0, 0.0

        stds = np.asarray(patch_std_values, dtype=np.float32)
        means = np.asarray(patch_mean_values, dtype=np.float32)

        uniformity = float(np.std(stds) / (np.mean(stds) + 1e-6))
        corr = np.corrcoef(means, stds)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        return uniformity, float(abs(corr))
