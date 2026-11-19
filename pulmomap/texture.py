"""Texture feature extraction: Local Binary Patterns and GLCM (Haralick) features."""
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def lbp(img, p=8, r=1):
    """Uniform LBP. Returns (lbp_image, 10-bin normalized histogram)."""
    codes = local_binary_pattern(img, p, r, method="uniform")
    n_bins = p + 2
    hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins), density=True)
    return codes, hist


def glcm_features(img, distances=(1,), angles=(0,), levels=256):
    """Returns dict of Haralick features averaged over given distances/angles, plus entropy."""
    glcm = graycomatrix(img, distances=list(distances), angles=list(angles),
                         levels=levels, symmetric=True, normed=True)
    feats = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        feats[prop] = float(np.mean(graycoprops(glcm, prop)))
    p = glcm[:, :, 0, 0]
    p_nonzero = p[p > 0]
    feats["entropy"] = float(-np.sum(p_nonzero * np.log2(p_nonzero))) if p_nonzero.size else 0.0
    return feats
