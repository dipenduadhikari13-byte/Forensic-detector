from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def build_test_image() -> bytes:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (220, 220), (0, 120, 255), -1)
    cv2.circle(image, (128, 128), 60, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode test image")
    return encoded.tobytes()


def run() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200, health.text

    image_bytes = build_test_image()
    files = {"file": ("sample.png", image_bytes, "image/png")}

    response = client.post("/analyze/image?include_heatmap=true", files=files)
    assert response.status_code == 200, response.text

    payload = response.json()
    required = ["score", "confidence", "label", "explanation", "ai_score", "edit_score", "details"]
    for key in required:
        assert key in payload, f"missing key: {key}"

    assert 0.0 <= payload["score"] <= 1.0
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["label"] in {"real", "manipulated", "ai_generated"}

    print("Smoke test passed")


if __name__ == "__main__":
    run()
