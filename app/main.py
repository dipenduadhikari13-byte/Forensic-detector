from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.schemas import AnalyzeImageResponse
from app.services.detection_service import DetectionService

app = FastAPI(
    title="Forensic Detector API",
    version="1.0.0",
    description="FastAPI service for AI-generation and image-manipulation detection",
)

service = DetectionService()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze/image", response_model=AnalyzeImageResponse)
async def analyze_image(
    file: UploadFile = File(...),
    include_heatmap: bool = Query(default=False, description="Include heatmap PNG as base64"),
) -> AnalyzeImageResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        result = await asyncio.to_thread(
            service.analyze_image,
            image_bytes,
            include_heatmap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {exc}") from exc

    return AnalyzeImageResponse(
        score=result.score,
        confidence=result.confidence,
        label=result.label,
        explanation=result.explanation,
        ai_score=result.ai_score,
        edit_score=result.edit_score,
        details=result.details,
        heatmap_base64=result.heatmap_base64,
    )
