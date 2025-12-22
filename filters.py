import cv2
import numpy as np

def apply_filter(frame, ftype):
    """Apply visual filters"""
    if ftype == "none":
        return frame

    f = frame.copy()

    if ftype == "cinematic":
        f = cv2.convertScaleAbs(f, alpha=1.2, beta=-15)
        f[:, :, 0] = np.clip(f[:, :, 0] * 0.9, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.1, 0, 255)
    elif ftype == "bright":
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        f = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    elif ftype == "vintage":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        f = cv2.transform(f, kernel)
    elif ftype == "dark":
        f = cv2.convertScaleAbs(f, alpha=0.7, beta=-30)
    elif ftype == "contrast":
        f = cv2.convertScaleAbs(f, alpha=1.5, beta=0)

    return np.clip(f, 0, 255).astype(np.uint8)
