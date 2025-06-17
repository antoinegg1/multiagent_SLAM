
"""Estimate 2‑D rigid transform between two top‑down RGB images using ORB + RANSAC."""
import cv2
import numpy as np

class ImagePoseEstimator:
    def __init__(self, nfeatures=500):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def estimate(self, img_a: np.ndarray, img_b: np.ndarray):
        """Return 3×3 homography H s.t. img_b ≈ H * img_a."""
        kp1, des1 = self.orb.detectAndCompute(img_a, None)
        kp2, des2 = self.orb.detectAndCompute(img_b, None)
        if des1 is None or des2 is None:
            return None

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
        # Convert 2×3 to 3×3
        if H is not None:
            H = np.vstack([H, [0, 0, 1]])
        return H
