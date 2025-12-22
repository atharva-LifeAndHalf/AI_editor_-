import cv2
import numpy as np 
def crossfade(a, b, n):
    out = []
    for i in range(n):
        alpha = 0.5 - 0.5 * np.cos(np.pi * i / n)
        out.append(cv2.addWeighted(a, 1 - alpha, b, alpha, 0))
    return out


def dissolve(a, b, n):
    out = []
    for i in range(n):
        alpha = i / n
        out.append(cv2.addWeighted(a, 1 - alpha, b, alpha, 0))
    return out


def fade_to_black(a, b, frames=10):
    out = []

    # Ensure same size
    h, w = a.shape[:2]
    b = cv2.resize(b, (w, h))

    black = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(frames):
        alpha = i / frames
        frame = cv2.addWeighted(black, 1 - alpha, b, alpha, 0)
        out.append(frame)

    return out
