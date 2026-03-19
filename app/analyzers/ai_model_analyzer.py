from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    torch = importlib.import_module("torch")
    models = importlib.import_module("torchvision.models")
    transforms = importlib.import_module("torchvision.transforms")
except Exception:
    torch = None
    models = None
    transforms = None


class AIModelAnalyzer:
    def __init__(self, checkpoint_path: str = "models/real_vs_ai.pt") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._torch_available = torch is not None and models is not None and transforms is not None
        self._device = torch.device("cpu") if self._torch_available else None
        self._model: Optional[Any] = None
        self._model_ready = False
        self._model_error: str | None = "torch/torchvision unavailable" if not self._torch_available else None
        self._preprocess = None
        if self._torch_available and transforms is not None:
            self._preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _build_model(self) -> Any:
        if not self._torch_available or models is None:
            raise RuntimeError("torch/torchvision unavailable")
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
        return model

    def _load_model_if_needed(self) -> None:
        if self._model is not None:
            return

        if not self._torch_available:
            self._model = None
            self._model_ready = False
            return

        try:
            model = self._build_model()
            if self.checkpoint_path.exists():
                state = torch.load(self.checkpoint_path, map_location=self._device)
                model.load_state_dict(state, strict=False)
                self._model_ready = True
            model.to(self._device)
            model.eval()
            self._model = model
        except Exception as exc:
            self._model = None
            self._model_ready = False
            self._model_error = str(exc)

    def analyze(self, image_bgr: np.ndarray) -> dict:
        self._load_model_if_needed()
        if self._model is None:
            return {
                "score": 0.5,
                "components": {
                    "real_prob": 0.5,
                    "ai_prob": 0.5,
                },
                "model_loaded_from_checkpoint": False,
                "model_ready": False,
                "model_error": self._model_error,
            }

        if not self._model_ready:
            return {
                "score": 0.5,
                "components": {
                    "real_prob": 0.5,
                    "ai_prob": 0.5,
                },
                "model_loaded_from_checkpoint": False,
                "model_ready": False,
                "model_error": self._model_error or "Checkpoint not found; using heuristic-only AI detection fallback",
            }

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        assert self._preprocess is not None
        tensor = self._preprocess(rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

        ai_prob = float(np.clip(probs[1], 0.0, 1.0))
        return {
            "score": ai_prob,
            "components": {
                "real_prob": round(float(probs[0]), 4),
                "ai_prob": round(ai_prob, 4),
            },
            "model_loaded_from_checkpoint": self.checkpoint_path.exists(),
            "model_ready": True,
            "model_error": None,
        }
