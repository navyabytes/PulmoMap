# PulmoMap: Chest X-Ray Image Processing & Anatomical Visualization Platform

IPCV Lab project, JIIT Noida. Classical image-processing and anatomical visualization only — not a
clinical diagnostic system.

## Three distinct capabilities (do not confuse them)

| Capability | What it does | Status |
|---|---|---|
| Image processing | Enhances/transforms X-rays (filters, contrast, edges, texture) | ✅ Implemented |
| Anatomical segmentation | Extracts lungs and other image regions with classical CV | ✅ Implemented |
| Abnormality / disease detection | Identifies pathological findings using a trained, validated model | 🚫 Not implemented — requires a separate ML/DL model with clinical validation |

PulmoMap only ever extracts descriptive, unvalidated features (shape, texture, intensity, edges) that
could be useful as input to a future trained model or to a clinician's own review — it never outputs a
disease label, probability or diagnosis.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## App sections

1. **Getting Started** — upload/sample image, dimensions, grayscale rationale, histogram, basic stats.
2. **Image Preprocessing** — grayscale, resize, normalization, histogram equalization, CLAHE, Gaussian /
   average / median filtering, run individually or as a full pipeline.
3. **Lung Segmentation** — Otsu + morphology or K-means; extract both/right/left lung, boundaries,
   background-removed image, binary masks; downloadable PNGs; failure warnings when two lungs aren't found.
4. **Whole-Image Feature Extraction** — intensity, edge, texture (LBP/GLCM) and shape features computed
   over the *entire* radiograph (not just the segmented lungs), plus a 3x3 anatomical analysis grid with
   per-cell statistics.
5. **Feature Extraction Lab** — Sobel/Prewitt/Canny edges, Harris/Shi-Tomasi corners, morphological
   operations, LBP + GLCM texture, shape/contour features (area, perimeter, circularity, solidity, hull).
6. **Anatomical Regions Explorer** — an honest catalogue of what classical image processing can and
   cannot localize beyond the lungs (heart/mediastinum as geometry-based estimates; ribs/clavicles/spine
   explicitly flagged as requiring a trained model), plus an advanced ROI picker (whole image, right/left
   lung, estimated heart/mediastinum/costophrenic regions, lung zones, grid cell, or a manual rectangle)
   with two-region side-by-side comparison.
7. **Candidate Region Explorer** — highlights locally unusual regions across the whole image using
   local intensity z-score, CLAHE contrast difference, or adaptive thresholding, with
   area/sensitivity/kernel controls. Explicitly **not** disease detection — see disclaimer in-app.
8. **Radiographic Findings Explorer** — educational reference on 10 common radiographic patterns
   (consolidation, nodules, effusion, pneumothorax, etc.) plus an interactive ROI workspace. Purely
   educational text + generic image-processing tools — no detection of any finding is performed.
9. **Anatomical Analysis** — per-lung area/perimeter/shape/zone distribution, right-vs-left symmetry
   index and mirror IoU, all pixel-based (not physical units without calibration).
10. **Comparison Dashboard** — side-by-side before/after and method-vs-method comparisons with
    downloadable figures.
11. **Detailed Report** — consolidated TXT and PDF report of everything run in the session, including
    limitations and the medical-use disclaimer.

## Modules

- `pulmomap/preprocessing.py` — grayscale, resize, normalize, hist-eq, CLAHE, filters, image stats.
- `pulmomap/segmentation.py` — Otsu, adaptive threshold, morphology, K-means, lung mask.
- `pulmomap/features.py` — Sobel, Prewitt, Canny, Harris, Shi-Tomasi, morphological features.
- `pulmomap/texture.py` — LBP and GLCM (Haralick) texture features.
- `pulmomap/roi.py` — rectangle/circle ROI cropping and per-ROI statistics.
- `pulmomap/analysis.py` — contour stats, zone distribution, left/right split, symmetry, overlays.
- `pulmomap/anatomy.py` — whole-image intensity/edge/texture/shape features, the 3x3 analysis grid, and
  the anatomical-region catalogue + geometry-based region estimates (heart, mediastinum, diaphragm,
  costophrenic angles) anchored on the lung mask.
- `pulmomap/candidate_regions.py` — whole-image candidate-region highlighting (local intensity deviation,
  CLAHE contrast, adaptive threshold) — generic image-processing output, not a disease detector.
- `pulmomap/findings.py` — educational reference content for radiographic patterns.
- `pulmomap/explanations.py` — the beginner/technical explanation registry used throughout the UI.
- `pulmomap/report.py` / `pulmomap/export.py` — TXT and PDF report generation.

## Limitations

- Classical segmentation can fail on atypical exposures, rotated films, hardware, or pathology that
  changes lung brightness — the app validates region count and warns rather than silently guessing.
- Pixel-based measurements are not physical measurements unless the image is calibrated.
- No deep-learning segmentation (U-Net/FCN) or disease classifier is included; adding either would
  require trained, validated weights, which are out of scope for this project.
- Anatomical regions beyond the lungs (heart, mediastinum, diaphragm, costophrenic angles) are
  **geometry-based estimates** anchored on the lung mask's bounding boxes, not traced/validated
  segmentations — see the Anatomical Regions Explorer for the full honesty catalogue.
- Ribs, clavicles and the vertebral column are explicitly **not** segmented — classical thresholding
  cannot reliably separate overlapping bone shadows from soft tissue; this would require a trained model.
- The Candidate Region Explorer surfaces generic statistical/contrast outliers across the whole image.
  These are not disease detections: normal anatomy (ribs, vessels, hila) commonly appears as a
  "candidate," and results require expert interpretation.
