from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.analyzers import AIModelAnalyzer, AISignatureAnalyzer, EditAnalyzer, encode_heatmap


@dataclass
class DetectionResult:
    score: float
    confidence: float
    label: str
    explanation: str
    ai_score: float
    edit_score: float
    details: dict
    heatmap_base64: str | None


class DetectionService:
    def __init__(self) -> None:
        self.edit_analyzer = EditAnalyzer()
        self.ai_analyzer = AIModelAnalyzer()
        self.ai_signature_analyzer = AISignatureAnalyzer()

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError("Image exceeds 20MB limit")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid or unsupported image file")
        if image.shape[0] < 96 or image.shape[1] < 96:
            raise ValueError("Image too small; minimum supported size is 96x96")
        return image

    @staticmethod
    def _decide_label(ai_score: float, edit_score: float) -> str:
        if ai_score >= 0.62 and ai_score >= (edit_score + 0.05):
            return "ai_generated"
        if edit_score >= 0.54:
            return "manipulated"
        return "real"

    @staticmethod
    def _compute_confidence(ai_score: float, edit_score: float, label: str) -> float:
        if label == "ai_generated":
            margin = ai_score - max(edit_score, 0.5)
        elif label == "manipulated":
            margin = edit_score - max(ai_score, 0.45)
        else:
            margin = 1.0 - max(ai_score, edit_score)
        return float(np.clip(0.55 + 0.8 * margin, 0.0, 1.0))

    @staticmethod
    def _build_explanation(label: str, ai_score: float, edit_score: float, confidence: float) -> str:
        if label == "ai_generated":
            return (
                "Synthetic-generation signals dominate the ensemble analysis. "
                f"AI score {ai_score:.2f} vs edit score {edit_score:.2f}, confidence {confidence:.2f}."
            )
        if label == "manipulated":
            return (
                "Artifact-level inconsistencies indicate likely local editing (splicing/copy-move/inpainting). "
                f"Edit score {edit_score:.2f}, AI score {ai_score:.2f}, confidence {confidence:.2f}."
            )
        return (
            "No dominant synthetic or manipulation signatures were found by the ensemble. "
            f"AI score {ai_score:.2f}, edit score {edit_score:.2f}, confidence {confidence:.2f}."
        )

    def analyze_image(self, image_bytes: bytes, include_heatmap: bool = False) -> DetectionResult:
        image_bgr = self._decode_image(image_bytes)

        edit_out = self.edit_analyzer.analyze(image_bgr)
        ai_model_out = self.ai_analyzer.analyze(image_bgr)
        ai_signature_out = self.ai_signature_analyzer.analyze(image_bgr)

        edit_score = float(edit_out["score"])
        ai_model_score = float(ai_model_out["score"])
        ai_signature_score = float(ai_signature_out["score"])

        model_weight = 0.65 if ai_model_out.get("model_ready") else 0.15
        ai_score = float(np.clip(
            model_weight * ai_model_score +
            (1.0 - model_weight) * ai_signature_score,
            0.0,
            1.0,
        ))
        final_score = float(np.clip(max(edit_score, ai_score), 0.0, 1.0))

        label = self._decide_label(ai_score=ai_score, edit_score=edit_score)
        confidence = self._compute_confidence(ai_score, edit_score, label)
        explanation = self._build_explanation(label, ai_score, edit_score, confidence)

        heatmap_base64 = None
        if include_heatmap:
            combined_hm = np.clip(0.75 * edit_out["heatmap"] + 0.25 * ai_signature_out["heatmap"], 0.0, 1.0)
            heatmap_base64 = encode_heatmap(image_bgr, combined_hm)

        return DetectionResult(
            score=final_score,
            confidence=confidence,
            label=label,
            explanation=explanation,
            ai_score=ai_score,
            edit_score=edit_score,
            details={
                "edit": edit_out["components"],
                "ai_model": ai_model_out["components"],
                "ai_signature": ai_signature_out["components"],
                "ai_model_ready": ai_model_out.get("model_ready", False),
                "model_loaded_from_checkpoint": ai_model_out.get("model_loaded_from_checkpoint", False),
                "model_error": ai_model_out.get("model_error"),
            },
            heatmap_base64=heatmap_base64,
        )
