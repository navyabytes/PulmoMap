import cv2
import numpy as np


def sobel(img, ksize=3):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def sobel_xy(img, ksize=3):
    """Returns (sobel_x, sobel_y) as normalized 8-bit images."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    norm = lambda a: cv2.normalize(np.abs(a), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm(gx), norm(gy)


_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
_PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)


def prewitt_xy(img):
    """Returns (prewitt_x, prewitt_y) as normalized 8-bit images."""
    gx = cv2.filter2D(img.astype(np.float32), -1, _PREWITT_X)
    gy = cv2.filter2D(img.astype(np.float32), -1, _PREWITT_Y)
    norm = lambda a: cv2.normalize(np.abs(a), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm(gx), norm(gy)


def canny(img, t1=50, t2=150):
    t1, t2 = min(t1, t2), max(t1, t2)
    return cv2.Canny(img, t1, t2)


def harris(img, block=2, ksize=3, k=0.04, thresh=0.01):
    """Returns a BGR overlay with Harris corners marked in red."""
    r = cv2.cornerHarris(np.float32(img), block, ksize, k)
    r = cv2.dilate(r, None)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out[r > thresh * r.max()] = (0, 0, 255)
    return out


def shi_tomasi(img, max_corners=100, quality=0.01, min_dist=10):
    """Returns a BGR overlay with Shi-Tomasi corners marked in red."""
    corners = cv2.goodFeaturesToTrack(img, max_corners, quality, min_dist)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        for c in corners:
            x, y = c.ravel()
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
    return out


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def morph_erode(binary, k=5):
    return cv2.erode(binary, _kernel(k))


def morph_dilate(binary, k=5):
    return cv2.dilate(binary, _kernel(k))


def morph_gradient(binary, k=5):
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, _kernel(k))
