from __future__ import annotations

import cv2
import numpy as np


class CopyMoveAnalyzer:
    def __init__(self, max_features: int = 1200) -> None:
        self.max_features = max_features

    def analyze(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=self.max_features)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        heatmap = np.zeros(gray.shape, dtype=np.float32)
        if descriptors is None or len(keypoints) < 20:
            return 0.0, heatmap

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(descriptors, descriptors, k=2)

        good_pairs: list[tuple[int, int]] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.queryIdx == m.trainIdx:
                continue
            if m.distance < 0.78 * n.distance:
                pt1 = np.array(keypoints[m.queryIdx].pt)
                pt2 = np.array(keypoints[m.trainIdx].pt)
                dist = float(np.linalg.norm(pt1 - pt2))
                if dist > 14.0:
                    good_pairs.append((m.queryIdx, m.trainIdx))

        if not good_pairs:
            return 0.0, heatmap

        for q_idx, t_idx in good_pairs:
            qx, qy = map(int, keypoints[q_idx].pt)
            tx, ty = map(int, keypoints[t_idx].pt)
            cv2.circle(heatmap, (qx, qy), 10, 1.0, -1)
            cv2.circle(heatmap, (tx, ty), 10, 1.0, -1)

        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

        keypoint_count = max(len(keypoints), 1)
        ratio = min(len(good_pairs) / keypoint_count, 1.0)
        score = float(np.clip(ratio * 4.0, 0.0, 1.0))
        return score, heatmap
