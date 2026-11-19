"""Region-of-interest (ROI) cropping and per-ROI analysis."""
import cv2
import numpy as np


def rect_roi(img, x, y, w, h):
    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    return img[y:y + h, x:x + w], (x, y, w, h)


def circle_roi(img, cx, cy, radius):
    """Crops the bounding box of the circle and masks out pixels outside the circle (set to 0)."""
    H, W = img.shape[:2]
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(W, cx + radius), min(H, cy + radius)
    crop = img[y0:y1, x0:x1].copy()
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    out = np.zeros_like(crop)
    out[mask] = crop[mask]
    return out, mask, (x0, y0, x1 - x0, y1 - y0)


def roi_stats(roi_gray, mask=None):
    vals = roi_gray[mask] if mask is not None else roi_gray.ravel()
    if vals.size == 0:
        return None
    return {
        "mean_intensity": float(np.mean(vals)),
        "std_intensity": float(np.std(vals)),
        "min_intensity": float(np.min(vals)),
        "max_intensity": float(np.max(vals)),
        "pixel_count": int(vals.size),
    }


def roi_edges(roi_gray, t1=50, t2=150):
    return cv2.Canny(roi_gray, t1, t2)
