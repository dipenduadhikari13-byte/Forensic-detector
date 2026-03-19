from __future__ import annotations

import cv2
import numpy as np

from .ela_analyzer import ELAAnalyzer
from .noise_analyzer import NoiseAnalyzer
from .fft_analyzer import FFTAnalyzer
from .copy_move_analyzer import CopyMoveAnalyzer


class EditAnalyzer:
    def __init__(self) -> None:
        self.ela_analyzer = ELAAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.fft_analyzer = FFTAnalyzer()
        self.copy_move_analyzer = CopyMoveAnalyzer()

    def analyze(self, image_bgr: np.ndarray) -> dict:
        ela_score, ela_map = self.ela_analyzer.analyze(image_bgr)
        noise_score, noise_map = self.noise_analyzer.analyze(image_bgr)
        fft_score, fft_map = self.fft_analyzer.analyze(image_bgr)
        copy_move_score, copy_move_map = self.copy_move_analyzer.analyze(image_bgr)

        combined_heatmap = (
            0.35 * ela_map +
            0.25 * noise_map +
            0.20 * fft_map +
            0.20 * copy_move_map
        )
        combined_heatmap = np.clip(combined_heatmap, 0.0, 1.0)

        edit_score = float(np.clip(
            0.35 * ela_score +
            0.25 * noise_score +
            0.20 * fft_score +
            0.20 * copy_move_score,
            0.0,
            1.0,
        ))

        return {
            "score": edit_score,
            "components": {
                "ela": round(ela_score, 4),
                "noise": round(noise_score, 4),
                "fft": round(fft_score, 4),
                "copy_move": round(copy_move_score, 4),
            },
            "heatmap": combined_heatmap,
        }
