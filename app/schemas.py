from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeImageResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Final manipulation likelihood score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Ensemble confidence")
    label: str = Field(..., description="real | manipulated | ai_generated")
    explanation: str
    ai_score: float = Field(..., ge=0.0, le=1.0)
    edit_score: float = Field(..., ge=0.0, le=1.0)
    details: dict[str, object]
    heatmap_base64: Optional[str] = Field(default=None, description="PNG image as base64 when requested")
