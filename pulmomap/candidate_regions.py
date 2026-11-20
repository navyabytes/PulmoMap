"""Candidate region highlighting across the WHOLE image.

This is explicitly NOT disease detection. It surfaces regions whose local
intensity/contrast/texture/edge-density differs from their surroundings,
using several complementary classical techniques. Every candidate is just a
region whose appearance is locally unusual by one of these generic numerical
measures — dark is not automatically abnormal, bright is not automatically
abnormal, and normal anatomical structures (ribs, vessels, hila, the gastric
air bubble below the diaphragm) will often show up here too. Candidates
require expert interpretation and are not a diagnosis, a probability, or a
validated finding of any kind.

ROOT CAUSE OF THE PREVIOUS "always zero candidates" BUG (fixed here): the
z-score method's raw detections are naturally 1-2px-wide ridges along
intensity transitions, not blobs. The old code ran a 5x5 morphological
*opening* over that raw mask as a blanket cleanup step — opening erodes
before it dilates, and a structuring element wider than the detected ridges
erases them completely before the dilation step has anything left to grow
back. That is a generic bug (verified: a single 3x3 erosion alone already
collapsed a 2186px raw mask down to 18px), not a matter of sensitivity
tuning. The fix is to turn the continuous z-score field into genuine blobs
*before* thresholding (via Gaussian smoothing, which is scale-appropriate
for "a patch of unusual appearance" rather than "a single unusual pixel"),
and to use a much smaller, fixed cleanup kernel afterwards instead of a
blanket opening tied to the user's morphology-kernel slider.
"""
import cv2
import numpy as np

from pulmomap import anatomy as ANAT, texture as T

METHODS = [
    "Local intensity deviation (z-score)",
    "CLAHE local contrast",
    "Adaptive threshold (local darker/brighter)",
    "Local texture variation",
    "Edge-density variation",
]

METHOD_COLORS = {
    "Local intensity deviation (z-score)": (0, 200, 255),     # amber
    "CLAHE local contrast": (255, 120, 255),                   # magenta
    "Adaptive threshold (local darker/brighter)": (0, 255, 120),  # green
    "Local texture variation": (255, 200, 0),                  # cyan-ish
    "Edge-density variation": (120, 120, 255),                 # red-ish
}


def _raw_mask(gray, method, intensity_sensitivity, texture_sensitivity, contrast_sensitivity, kernel_size):
    """Each method returns an *un-cleaned* binary mask plus a short reason string describing
    the rule that produced it (for the debug panel)."""
    win = max(9, kernel_size * 3) | 1

    if method == "Local intensity deviation (z-score)":
        mean, std = ANAT.local_mean_std(gray, win)
        std_safe = np.where(std < 1e-3, 1e-3, std)
        z = np.abs(gray.astype(np.float32) - mean) / std_safe
        raw = (z > intensity_sensitivity).astype(np.uint8) * 255
        # Thicken the (naturally ridge-thin) detections into blobs BEFORE any cleanup step runs,
        # so a small opening doesn't erase them outright (this was the root cause of the old bug).
        mask = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        reason = f"local z-score > {intensity_sensitivity:.1f} (window {win}px), dilated to blob-form"

    elif method == "CLAHE local contrast":
        cl = cv2.createCLAHE(clipLimit=contrast_sensitivity, tileGridSize=(8, 8)).apply(gray)
        diff = cv2.absdiff(cl, gray)
        thresh = int(np.clip(255 * (intensity_sensitivity / 10.0), 5, 250))
        _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
        reason = f"|CLAHE(clip={contrast_sensitivity:.1f}) - original| > {thresh}"

    elif method == "Adaptive threshold (local darker/brighter)":
        block = max(3, (kernel_size * 4) | 1)
        C = max(1, int(contrast_sensitivity))
        # NOTE: darker/brighter must use opposite-signed C around the SAME local mean, leaving a
        # dead zone in between. Using the same +C for both (the original bug) makes them exact
        # complements, so their OR always covers 100% of the image regardless of settings.
        darker = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, block, C)
        brighter = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block, -C)
        mask = cv2.bitwise_or(darker, brighter)
        reason = f"locally darker OR brighter than neighbourhood mean by more than C={C} (block {block}px)"

    elif method == "Local texture variation":
        tex = ANAT.texture_variation_map(gray, win).astype(np.float32)
        thresh = float(np.clip(255 * (texture_sensitivity / 10.0), 5, 250))
        mask = (tex > thresh).astype(np.uint8) * 255
        reason = f"local LBP-code variation > {thresh:.0f}/255 (window {win}px)"

    else:  # Edge-density variation
        density = ANAT.edge_density_heatmap(gray, win).astype(np.float32)
        thresh = float(np.clip(255 * (intensity_sensitivity / 10.0), 5, 250))
        mask = (density > thresh).astype(np.uint8) * 255
        reason = f"local edge density > {thresh:.0f}/255 (window {win}px)"

    return mask, reason


def extract_candidates(gray, method="Local intensity deviation (z-score)",
                        min_area=80, max_area=20000, intensity_sensitivity=2.0,
                        texture_sensitivity=7.0, contrast_sensitivity=3.0, kernel_size=5,
                        valid_mask=None):
    """Returns (overlay_mask, candidates, debug). `valid_mask` restricts candidates to a region
    of interest (e.g. inside the lung fields) if provided; otherwise the whole image.

    `debug` always reports counts at every pipeline stage, so the UI can explain *why* a
    candidate count is zero (as opposed to silently showing nothing) — zero is a valid result
    when nothing in the image meets the selected criteria, but the cause should be visible."""
    raw, reason = _raw_mask(gray, method, intensity_sensitivity, texture_sensitivity,
                             contrast_sensitivity, kernel_size)
    raw_nonzero = int(np.count_nonzero(raw))

    # Small, fixed cleanup kernel: large enough to remove single-pixel speckle, small enough
    # to not erase genuine small candidate blobs regardless of the user's chosen kernel_size.
    cleanup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cleanup)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, kernel_size), max(3, kernel_size))))
    clean_nonzero = int(np.count_nonzero(clean))

    restricted_nonzero = clean_nonzero
    if valid_mask is not None:
        clean = clean & (valid_mask > 0).astype(np.uint8) * 255
        restricted_nonzero = int(np.count_nonzero(clean))

    n, lab, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
    components_total = n - 1
    candidates = []
    overlay_mask = np.zeros_like(gray)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        crop = gray[y:y + h, x:x + w]
        comp = (lab[y:y + h, x:x + w] == i)
        vals = crop[comp]
        if vals.size == 0:
            continue
        edges = cv2.Canny(crop, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size if edges.size else 0.0
        candidates.append({
            "id": len(candidates) + 1,
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cents[i][0]), float(cents[i][1])),
            "area_px": int(area),
            "mean_intensity": float(vals.mean()),
            "std_intensity": float(vals.std()),
            "local_contrast": float(vals.max() - vals.min()),
            "edge_density": edge_density,
        })
        overlay_mask[y:y + h, x:x + w][comp] = 255

    debug = {
        "method": method, "rule": reason,
        "raw_mask_nonzero_px": raw_nonzero,
        "after_cleanup_nonzero_px": clean_nonzero,
        "after_roi_restriction_nonzero_px": restricted_nonzero,
        "connected_components_found": components_total,
        "components_passing_area_filter": len(candidates),
        "min_area_px": min_area, "max_area_px": max_area,
    }
    return overlay_mask, candidates, debug


def draw_candidates(gray, candidates, color=(0, 200, 255)):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in candidates:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cx, cy = c["centroid"]
        cv2.circle(vis, (int(cx), int(cy)), 3, color, -1)
        label = f"#{c['id']} A={c['area_px']}"
        cv2.putText(vis, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


CANDIDATE_DISCLAIMER = (
    "These highlighted regions are generic image-processing candidates based on local intensity, "
    "contrast, texture or edge-density statistics — they are NOT disease detections. Dark regions "
    "are not automatically abnormal; bright regions are not automatically abnormal; normal anatomy "
    "(ribs, vessels, hilar structures, the gastric air bubble below the diaphragm) will often "
    "appear here too. No candidate is or implies a diagnosis of cancer, pneumonia, tuberculosis or "
    "any other condition. Interpretation requires an expert. Zero candidates is a valid result — "
    "it means nothing in the image met the selected threshold, not that the tool failed."
)
