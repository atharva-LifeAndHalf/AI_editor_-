import cv2
import numpy as np
import os
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import mediapipe as mp

    try:
        from mediapipe.python.solutions import selfie_segmentation as mp_selfie_seg
    except (ImportError, AttributeError):
        mp_selfie_seg = mp.solutions.selfie_segmentation
    segmentor = mp_selfie_seg.SelfieSegmentation(model_selection=1)
    print(" MediaPipe initialized")
except Exception as e:
    print(f"MediaPipe unavailable: {e}")
    segmentor = None

def apply_stabilization(prev_gray, curr_frame):
    """Video stabilization"""
    try:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

        if prev_pts is None:
            return curr_frame, curr_gray

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        idx = np.where(status == 1)[0]

        if len(idx) < 3:
            return curr_frame, curr_gray

        m, _ = cv2.estimateAffinePartial2D(prev_pts[idx], curr_pts[idx])

        if m is None:
            return curr_frame, curr_gray

        rows, cols = curr_frame.shape[:2]
        stabilized = cv2.warpAffine(curr_frame, m, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return stabilized, curr_gray
    except:
        return curr_frame, cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)


def add_logo_overlay(frame, logo_path, position=(20, 20), scale=0.15):
    """Add logo with transparency"""
    try:
        if not os.path.exists(logo_path):
            return frame

        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            return frame

        h, w = frame.shape[:2]
        logo_w = int(w * scale)
        logo_h = int(logo.shape[0] * (logo_w / logo.shape[1]))
        logo = cv2.resize(logo, (logo_w, logo_h))

        y_off = h - logo_h - position[1]
        x_off = w - logo_w - position[0]

        if y_off < 0 or x_off < 0:
            return frame

        roi = frame[y_off:y_off + logo_h, x_off:x_off + logo_w]

        if logo.shape[2] == 4:
            alpha = logo[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = (alpha * logo[:, :, c] + (1 - alpha) * roi[:, :, c])
        else:
            cv2.addWeighted(roi, 1, logo, 0.7, 0, roi)

        frame[y_off:y_off + logo_h, x_off:x_off + logo_w] = roi
        return frame
    except:
        return frame


def remove_background(frame):
    """Remove background using MediaPipe"""
    try:
        if segmentor is None:
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = segmentor.process(rgb)

        if res.segmentation_mask is None:
            return frame

        mask = res.segmentation_mask > 0.5
        bg = np.zeros(frame.shape, dtype=np.uint8)
        return np.where(np.stack((mask,) * 3, axis=-1), frame, bg)
    except:
        return frame


def remove_object(frame, roi):
    """Remove object using inpainting"""
    try:
        if roi is None or len(roi) != 4:
            return frame

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x, y, w, h = roi

        h_frame, w_frame = frame.shape[:2]
        if x < 0 or y < 0 or x + w > w_frame or y + h > h_frame:
            return frame

        mask[y:y + h, x:x + w] = 255
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
    except:
        return frame

