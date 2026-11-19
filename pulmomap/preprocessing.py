import cv2
import numpy as np

SIZE = 512


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def resize_preserve_aspect(img, size=SIZE):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=img.dtype)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = r
    return canvas


def min_max_normalize(img):
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return ((img - lo) / (hi - lo) * 255).astype(np.uint8)
    return img.astype(np.uint8)


def hist_equalize(img):
    return cv2.equalizeHist(img)


def clahe(img, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def average_blur(img, k=5):
    return cv2.blur(img, (k, k))


def gaussian_blur(img, k=5, sigma=1.0):
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_blur(img, k=5):
    return cv2.medianBlur(img, k)


def image_stats(img):
    """Basic intensity statistics for a grayscale image."""
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "min": float(np.min(img)),
        "max": float(np.max(img)),
        "median": float(np.median(img)),
    }


def histogram(img, bins=256):
    hist, edges = np.histogram(img.ravel(), bins=bins, range=(0, 255))
    return hist, edges


FILTERS = {"Gaussian": gaussian_blur, "Average": average_blur, "Median": median_blur}
