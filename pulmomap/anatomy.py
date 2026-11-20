"""Whole-image feature extraction and anatomical region *estimation*.

IMPORTANT HONESTY NOTE: this module does not perform validated anatomical
segmentation of bones, the heart or the mediastinum. Every region produced
here is one of three kinds, and callers/UI must always say which:

  CLASSICAL_ESTIMATE   - a rough geometric ROI derived from simple heuristics
                         (e.g. the lung mask's bounding box). Useful as a
                         starting point for inspection, not a validated mask.
  REQUIRES_TRAINED_MODEL - PulmoMap has no trained model for this structure;
                         classical thresholding cannot reliably isolate it
                         (e.g. individual ribs, the vertebral column).
  NOT_RELIABLE         - even a rough estimate is not attempted because the
                         structure is too variable/ambiguous with classical
                         methods alone.
"""
import cv2
import numpy as np

from pulmomap import texture as T

CLASSICAL_ESTIMATE = "Classical estimate (geometry-based ROI, not a validated segmentation)"
REQUIRES_TRAINED_MODEL = "Requires a trained model (not implemented)"
NOT_RELIABLE = "Not reliably identifiable with classical image processing alone"

# Static catalogue describing what PulmoMap can and cannot do for each
# anatomical structure named in the brief.
ANATOMY_CATALOG = {
    "Heart / cardiac silhouette": dict(
        status=CLASSICAL_ESTIMATE,
        note="Estimated as a central-lower rectangle spanning the gap between the two lung masks. "
             "This approximates where the cardiac silhouette usually sits on a PA film; it is not "
             "a traced cardiac border and cardiothoracic ratio should not be computed from it."),
    "Mediastinal region": dict(
        status=CLASSICAL_ESTIMATE,
        note="Estimated as the vertical strip between the medial edges of the right and left lung "
             "masks. Width/position depend entirely on how well the lung mask was segmented."),
    "Tracheal region": dict(
        status=NOT_RELIABLE,
        note="The trachea is a thin, low-contrast air column; classical thresholding on a "
             "standard PA film does not reliably separate it from surrounding soft tissue."),
    "Hilar regions": dict(
        status=NOT_RELIABLE,
        note="Hilar structures (vessels/bronchi) overlap extensively in projection and need "
             "trained segmentation or multi-view imaging to localize reliably."),
    "Ribs": dict(
        status=REQUIRES_TRAINED_MODEL,
        note="Individual rib segmentation needs a model trained on labeled bone contours; plain "
             "thresholding cannot separate overlapping rib shadows from lung and soft tissue."),
    "Clavicles": dict(
        status=REQUIRES_TRAINED_MODEL,
        note="Same limitation as ribs — bone-vs-soft-tissue separation needs a trained model."),
    "Vertebral column": dict(
        status=REQUIRES_TRAINED_MODEL,
        note="The spine is partly obscured by the mediastinum on a PA film; reliable localization "
             "needs a trained model or a lateral view."),
    "Shoulder structures": dict(
        status=NOT_RELIABLE,
        note="Highly variable position/framing across films; no stable classical heuristic."),
    "Right hemidiaphragm": dict(
        status=CLASSICAL_ESTIMATE,
        note="Estimated as the lowest boundary row of the right lung mask. Only as reliable as "
             "the underlying lung segmentation."),
    "Left hemidiaphragm": dict(
        status=CLASSICAL_ESTIMATE,
        note="Estimated as the lowest boundary row of the left lung mask. Only as reliable as the "
             "underlying lung segmentation."),
    "Costophrenic angles": dict(
        status=CLASSICAL_ESTIMATE,
        note="Estimated as a small corner ROI at the lower-lateral extremity of each lung mask — "
             "a location estimate, not a validated detection of blunting or effusion."),
    "Subdiaphragmatic / upper abdominal region": dict(
        status=CLASSICAL_ESTIMATE,
        note="Estimated as the band below the lowest lung boundary, within the valid (non-padded) "
             "image area. May include the normal gastric air bubble, which is not an abnormality."),
}


# ---------------------------------------------------------------------------
# Whole-image feature extraction (independent of lung segmentation)
# ---------------------------------------------------------------------------

def intensity_features(gray):
    vals = gray.ravel().astype(np.float64)
    pct = np.percentile(vals, [5, 25, 50, 75, 95])
    return {
        "mean": float(vals.mean()), "median": float(np.median(vals)),
        "std": float(vals.std()), "min": float(vals.min()), "max": float(vals.max()),
        "p5": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
        "p75": float(pct[3]), "p95": float(pct[4]),
    }


def edge_features(gray, t1=50, t2=150):
    gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    canny = cv2.Canny(gray, t1, t2)
    density = float(np.count_nonzero(canny)) / canny.size
    return {"sobel_x": cv2.normalize(np.abs(gx), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            "sobel_y": cv2.normalize(np.abs(gy), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            "sobel_magnitude": sobel_mag, "canny": canny, "edge_density": density}


def texture_features(gray):
    _, lbp_hist = T.lbp(gray)
    glcm = T.glcm_features(gray)
    return {"lbp_hist": lbp_hist, **glcm}


def shape_features(gray, t1=50, t2=150):
    """Generic contour features on a Canny edge map of the whole image (not a segmentation)."""
    edges = cv2.Canny(gray, t1, t2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    feats = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        per = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        M = cv2.moments(c)
        cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2
        cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2
        feats.append({
            "area_px": float(area), "perimeter_px": float(per),
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
            "circularity": float(4 * np.pi * area / per ** 2) if per else 0.0,
            "solidity": float(area / ha) if ha else 0.0,
            "aspect_ratio": float(w / h) if h else 0.0,
        })
    return feats


# ---------------------------------------------------------------------------
# Per-pixel feature maps — these always render regardless of any downstream
# thresholding/candidate logic, so a blank candidate layer never implies a
# blank analysis: the maps below are the ground truth the candidate logic
# is derived from, and are worth looking at directly.
# ---------------------------------------------------------------------------

def local_mean_std(gray, k):
    """Local mean/std via box filtering — the shared primitive behind the contrast, texture
    and z-score maps below. k is the (odd) side length of the local window in pixels."""
    k = k if k % 2 == 1 else k + 1
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g ** 2, (k, k))
    var = np.clip(mean_sq - mean ** 2, 0, None)
    return mean, np.sqrt(var)


def local_contrast_map(gray, k=15):
    """Local standard deviation at every pixel, scaled to 0-255 — literally 'how much contrast
    is happening right here', independent of any candidate threshold."""
    _, std = local_mean_std(gray, k)
    return cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def texture_variation_map(gray, k=15):
    """Local variability of the LBP code at every pixel — a per-pixel texture-roughness map."""
    codes, _ = T.lbp(gray)
    _, std = local_mean_std(codes.astype(np.float32), k)
    return cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def edge_density_heatmap(gray, k=25, t1=50, t2=150):
    """Fraction of edge pixels in a sliding local window, at every pixel — where the image is
    'busy' with fine detail versus smooth."""
    k = k if k % 2 == 1 else k + 1
    edges = cv2.Canny(gray, t1, t2).astype(np.float32) / 255.0
    density = cv2.blur(edges, (k, k))
    return cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def z_score_map(gray, k=15):
    """Continuous local z-score at every pixel (how many local std-devs a pixel is from its own
    neighbourhood mean), scaled to 0-255 for display. The candidate thresholding in
    candidate_regions.py operates on the un-scaled version of this map."""
    mean, std = local_mean_std(gray, k)
    std_safe = np.where(std < 1e-3, 1e-3, std)
    z = np.abs(gray.astype(np.float32) - mean) / std_safe
    return cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def connected_components_overlay(binary, gray, min_area=1, max_components=50):
    """Draws every connected component's bounding box + centroid + id on top of `gray`, purely as
    a structural visualization — this is NOT a semantic segmentation, just 'here is every blob
    the thresholded binary image currently contains'."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    kept = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if kept >= max_components:
            break
        x, y, w, h = stats[i, :4]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 180, 0), 1)
        cx, cy = cents[i]
        cv2.circle(vis, (int(cx), int(cy)), 2, (255, 180, 0), -1)
        cv2.putText(vis, str(kept + 1), (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 0), 1)
        kept += 1
    return vis, n - 1, kept


# ---------------------------------------------------------------------------
# 3x3 anatomical grid
# ---------------------------------------------------------------------------

def grid_cells(shape, n=3):
    H, W = shape[:2]
    ys = np.linspace(0, H, n + 1).astype(int)
    xs = np.linspace(0, W, n + 1).astype(int)
    cells = {}
    for r in range(n):
        for c in range(n):
            label = f"R{r+1}C{c+1}"
            cells[label] = (xs[c], ys[r], xs[c + 1] - xs[c], ys[r + 1] - ys[r])
    return cells


def grid_stats(gray, n=3):
    cells = grid_cells(gray.shape, n)
    out = {}
    for label, (x, y, w, h) in cells.items():
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        out[label] = {"bbox": (x, y, w, h), "mean": float(crop.mean()), "std": float(crop.std())}
    return out


# ---------------------------------------------------------------------------
# Geometric (classical-estimate) regions derived from a lung mask
# ---------------------------------------------------------------------------

def _lung_bboxes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    if len(cnts) < 2:
        return None
    boxes = sorted([cv2.boundingRect(c) for c in cnts], key=lambda b: b[0])
    return boxes  # [image-left (patient right), image-right (patient left)]


def estimate_regions(gray, mask):
    """Returns dict of {label: (x, y, w, h)} for the CLASSICAL_ESTIMATE regions that can
    actually be derived from a (possibly two-lung) segmentation mask. Returns {} if the
    lung mask does not have two separate regions to anchor the estimate on."""
    boxes = _lung_bboxes(mask)
    if boxes is None:
        return {}
    (rx, ry, rw, rh), (lx, ly, lw, lh) = boxes
    H, W = gray.shape

    mediastinum = (rx + rw, min(ry, ly), max(0, lx - (rx + rw)), max(rh, lh))
    heart_y = ry + int(0.45 * rh)
    heart_h = int(0.5 * max(rh, lh))
    heart = (max(0, rx + rw - 20), heart_y, min(W, (lx + 20) - max(0, rx + rw - 20)), heart_h)
    diaphragm_r = (rx, ry + rh - max(10, rh // 12), rw, max(10, rh // 12))
    diaphragm_l = (lx, ly + lh - max(10, lh // 12), lw, max(10, lh // 12))
    cp_angle_r = (rx, ry + rh - max(15, rh // 8), max(15, rw // 6), max(15, rh // 8))
    cp_angle_l = (lx + lw - max(15, lw // 6), ly + lh - max(15, lh // 8), max(15, lw // 6), max(15, lh // 8))
    sub_y = max(ry + rh, ly + lh)
    subdiaphragm = (0, sub_y, W, max(0, H - sub_y))

    return {
        "Heart / cardiac silhouette (estimated)": heart,
        "Mediastinal region (estimated)": mediastinum,
        "Right hemidiaphragm (estimated)": diaphragm_r,
        "Left hemidiaphragm (estimated)": diaphragm_l,
        "Right costophrenic angle (estimated)": cp_angle_r,
        "Left costophrenic angle (estimated)": cp_angle_l,
        "Subdiaphragmatic region (estimated)": subdiaphragm,
    }
