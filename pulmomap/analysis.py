import cv2
import numpy as np


def split_left_right(mask):
    """Returns (left, right) as the patient's lungs.
    In a PA film the patient's RIGHT lung appears on the image LEFT."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])  # image-left first
    return cnts  # [image-left (patient right), image-right (patient left)]


def contour_stats(c):
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2
    cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2
    return {
        "area_px": float(area),
        "perimeter_px": float(per),
        "bbox_w": int(w), "bbox_h": int(h),
        "aspect_ratio": float(w / h) if h else 0.0,
        "circularity": float(4 * np.pi * area / per ** 2) if per else 0.0,
        "solidity": float(area / ha) if ha else 0.0,
        "centroid": (float(cx), float(cy)),
    }


def zone_distribution(mask_single, bbox):
    """Splits a lung into upper/middle/lower thirds; returns % of lung area."""
    x, y, w, h = bbox
    crop = mask_single[y:y + h, x:x + w]
    total = max(int(np.count_nonzero(crop)), 1)
    thirds = np.array_split(np.arange(h), 3)
    return {n: 100.0 * np.count_nonzero(crop[t[0]:t[-1] + 1]) / total
            for n, t in zip(["upper", "middle", "lower"], thirds)}


def analyse(mask):
    cnts = split_left_right(mask)
    if len(cnts) < 2:
        return None
    names = ["Right lung (image left)", "Left lung (image right)"]
    res, per_mask = {}, []
    for n, c in zip(names, cnts):
        s = contour_stats(c)
        m = np.zeros_like(mask)
        cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
        s["zones_%"] = zone_distribution(m, cv2.boundingRect(c))
        res[n] = s
        per_mask.append(m)
    a1, a2 = res[names[0]]["area_px"], res[names[1]]["area_px"]
    res["symmetry"] = {
        "area_ratio_R/L": a1 / a2 if a2 else 0.0,
        "symmetry_index": 1 - abs(a1 - a2) / max(a1 + a2, 1),
        # mirror one lung across the image midline, compare via IoU
        "mirror_iou": _mirror_iou(per_mask[0], per_mask[1]),
    }
    return res


def _mirror_iou(m1, m2):
    ys, xs = np.nonzero(m1 | m2)
    if len(xs) == 0:
        return 0.0
    mid = (xs.min() + xs.max()) / 2
    M = np.float32([[-1, 0, 2 * mid], [0, 1, 0]])
    flipped = cv2.warpAffine(m1, M, (m1.shape[1], m1.shape[0]))
    inter = np.count_nonzero(flipped & m2)
    union = np.count_nonzero(flipped | m2)
    return inter / union if union else 0.0


def background_removed(gray, mask):
    """Everything except the masked region, with the masked region blacked out."""
    out = gray.copy()
    out[mask > 0] = 0
    return out


def boundary_overlay(gray, mask, color=(0, 255, 255)):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, cnts, -1, color, 2)
    return bgr


def boundary_only(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, 2)
    return out


def contour_features(c):
    """Generic shape/contour features for any single contour (used in the Feature Extraction Lab)."""
    return contour_stats(c)


def overlay(gray, mask):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tint = bgr.copy()
    tint[mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(bgr, 0.65, tint, 0.35, 0)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    return out
