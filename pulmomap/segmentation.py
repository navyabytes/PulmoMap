import cv2
import numpy as np
from sklearn.cluster import KMeans


def otsu(img):
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def adaptive(img, block=51, c=5):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def opening(img, k=5):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, _kernel(k))


def closing(img, k=5):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(k))


def kmeans_segment(img, k=3):
    """Intensity K-means; returns a mask of the darkest cluster (lung candidates)."""
    data = img.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(data)
    dark = int(np.argmin(km.cluster_centers_.ravel()))
    return (km.labels_.reshape(img.shape) == dark).astype(np.uint8) * 255


def _valid_region(img):
    """Mask of the non-padded image area (padding is exactly 0)."""
    m = (img > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(15))
    return cv2.erode(m, _kernel(9))


def lung_mask(img, min_area=1500):
    """Classical lung segmentation.
    Lungs are the DARK regions inside the thorax, so threshold inverted,
    drop components touching the image border/padding, keep the two largest."""
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    valid = _valid_region(img)
    vals = blur[valid > 0].reshape(-1, 1)
    # two-level Otsu: separate air/background vs lung vs tissue, use midpoint of the two cuts
    t1, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = vals[vals < t1].reshape(-1, 1).astype(np.uint8)
    t2, _ = cv2.threshold(low, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = (t1 + t2) / 2
    b = ((blur < t) & (valid > 0)).astype(np.uint8) * 255
    return largest_two(b, min_area)


def largest_two(binary, min_area=1500):
    """Clean a binary mask: open/close, drop border-touching blobs, keep 2 largest, fill holes."""
    b = closing(opening(binary, 5), 7)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    H, W = b.shape
    c = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area and not (x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1):
            c.append((a, i))
    out = np.zeros_like(b)
    for _, i in sorted(c, reverse=True)[:2]:
        out[lab == i] = 255
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f = np.zeros_like(out)
    cv2.drawContours(f, cnts, -1, 255, cv2.FILLED)
    return f
