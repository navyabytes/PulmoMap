import cv2
import numpy as np
import streamlit as st

from pulmomap import preprocessing as P, features as F, segmentation as S, analysis as A, report as R
from pulmomap import texture as T, roi as ROI, export as EXP
from pulmomap.explanations import explain, DISCLAIMER
from pulmomap.findings import FINDINGS, FINDINGS_DISCLAIMER

st.set_page_config(page_title="PulmoMap", page_icon="🫁", layout="wide")

# ---------------------------------------------------------------------------
# Visual language: flat, dense, dark "clinical tool" skin — near-black
# surfaces, crisp 1px borders instead of shadows, one accent color, muted
# gray body text, small tracked uppercase labels. Function over decoration:
# nothing here shrinks image/plot real estate or hides a control.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0e0f; --panel: #121617; --panel-2: #171c1e; --border: #262b2d;
  --text: #eef1f2; --muted: #8b969a;
  --accent: #19b89a; --accent-hover: #2bd6b6; --accent-light: rgba(25,184,154,.14);
  --warning: #f5a524; --warning-light: rgba(245,165,36,.12);
  --danger: #ff6b6b; --success: #22c55e;
}
@keyframes pmFadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.6rem; animation: pmFadeIn .35s ease-out; }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--text); }
p, label, .stMarkdown { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* eyebrow label used above every section title */
.pm-eyebrow { font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
              color: var(--accent); margin-bottom: .2rem; }
.pm-page-header { padding-bottom: .9rem; margin-bottom: 1.1rem; border-bottom: 1px solid var(--border); }
.pm-page-title { font-size: 1.6rem; font-weight: 700; color: var(--text); margin: 0; }
.pm-page-sub { font-size: .92rem; color: var(--muted); margin-top: .25rem; }

/* flat cards */
.pm-card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           padding: .9rem 1.1rem; margin-bottom: .6rem; font-size: .88rem; }
.pm-card b { color: var(--text); }
.pm-card.ok { border-left: 3px solid var(--success); }
.pm-card.no { border-left: 3px solid var(--danger); }
.pm-card.warn { border-left: 3px solid var(--warning); background: var(--warning-light); }

/* metrics */
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border);
                               border-radius: 8px; padding: .6rem .8rem; }
div[data-testid="stMetricValue"] { font-size: 1.3rem; color: var(--text); }
div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; letter-spacing: .05em;
                                    text-transform: uppercase; }

/* sidebar nav rendered as a flat vertical list */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  border-radius: 7px; padding: .4rem .6rem; margin-bottom: .1rem; transition: background .15s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: var(--panel-2); }

/* buttons */
.stButton button, .stDownloadButton button {
  border-radius: 7px; border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
}
.stButton button:hover, .stDownloadButton button:hover {
  border-color: var(--accent); color: var(--accent-hover);
}

/* expanders as subtle panels */
details[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border);
                                     border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def page_header(eyebrow, title, subtitle=None):
    html = f'<div class="pm-page-header"><div class="pm-eyebrow">{eyebrow}</div>' \
           f'<p class="pm-page-title">{title}</p>'
    if subtitle:
        html += f'<p class="pm-page-sub">{subtitle}</p>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


st.markdown('<div class="pm-eyebrow">🫁 PulmoMap</div>', unsafe_allow_html=True)
st.markdown('<p class="pm-page-title" style="font-size:2rem;">Chest X-Ray Image Processing & Feature Exploration</p>',
            unsafe_allow_html=True)
st.markdown('<p class="pm-page-sub">Intelligent chest X-ray processing, anatomical segmentation and '
            'radiographic feature exploration.</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pm-card warn">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="pm-card ok"><b>✅ Image processing</b> — enhances and transforms X-rays (filters, contrast, edges). Implemented below.</div>
<div class="pm-card ok"><b>✅ Anatomical segmentation</b> — extracts lungs and other image regions using classical algorithms. Implemented below.</div>
<div class="pm-card no"><b>🚫 Abnormality / disease detection</b> — identifying pathological findings needs a separately trained,
clinically validated ML/DL model. <u>Not implemented here</u> — PulmoMap only extracts descriptive features
(shape, texture, intensity, edges) that a trained model or a clinician could later make use of.</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
SECTIONS = ["Getting Started", "Image Preprocessing", "Lung Segmentation", "Feature Extraction Lab",
            "Radiographic Findings Explorer", "Anatomical Analysis", "Comparison Dashboard", "Detailed Report"]

with st.sidebar:
    st.markdown('<div class="pm-eyebrow">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Section", SECTIONS, label_visibility="collapsed")
    st.divider()
    st.caption("Upload once, explore all sections below.")
    up = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    use_sample = st.checkbox("Use the bundled sample image instead")

if use_sample:
    img = cv2.imread("data/samples/sample_cxr.png", cv2.IMREAD_COLOR)
    src_name = "sample_cxr.png"
elif up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    src_name = up.name
else:
    img = None
    src_name = None

if img is None and page != "Getting Started":
    st.info("Upload a chest X-ray (or check 'Use the bundled sample image') in the sidebar to use this section.")
    st.stop()
if img is not None and img.size == 0:
    st.error("Could not decode that image. Try a different file.")
    st.stop()

if img is not None:
    orig_rgb = img[:, :, ::-1]
    gray_full = P.to_grayscale(img)
    gray = P.resize_preserve_aspect(gray_full)
    norm = P.min_max_normalize(gray)

# ===========================================================================
# 1. GETTING STARTED
# ===========================================================================
if page == "Getting Started":
    page_header("Section 1 / 8", "Getting Started", "Upload an image and get oriented before diving into analysis.")
    st.markdown("""
PulmoMap lets you explore a chest X-ray through classical (non-AI) image-processing and computer-vision
techniques. It is built for two kinds of users:

- **Beginners** who want to understand *what each operation does* in plain English.
- **Technical users** who want to extract anatomical structures, image features, and build a feature
  report for further reference.

**How to use it**
1. Upload an image in the sidebar (or tick "Use the bundled sample image").
2. Pick a section from the sidebar: Preprocessing, Segmentation, Feature Extraction, Findings Explorer,
   Anatomical Analysis, Comparison, or the final Detailed Report.
3. Every slider and algorithm has an expander ("What does ... mean?") explaining it in simple and
   technical terms, and what happens as you move it up or down.

**Why grayscale?** Chest X-rays are radiation-intensity images — there is no real colour information,
just how much radiation passed through each part of the body. Any colour you may see in a photo of a
printed film is a display artifact, so all processing here works on the grayscale intensity data.
""")
    if img is not None:
        c1, c2 = st.columns(2)
        c1.image(orig_rgb, caption=f"Original ({src_name})", width="stretch")
        c2.image(gray, caption="Grayscale, resized to 512x512", clamp=True, width="stretch")
        with st.expander("Image information & statistics"):
            st.json({"format": src_name.split(".")[-1].upper() if src_name else "n/a",
                     "original_size": f"{img.shape[1]} x {img.shape[0]}", **P.image_stats(gray)})
        hist, edges = P.histogram(gray)
        st.caption("Intensity histogram (grayscale, resized image)")
        st.bar_chart(hist)

# ===========================================================================
# 2. IMAGE PREPROCESSING
# ===========================================================================
elif page == "Image Preprocessing":
    page_header("Section 2 / 8", "Image Preprocessing", "Grayscale, normalization, contrast and noise-reduction operations, individually or as a pipeline.")
    st.caption("Each operation can be inspected individually, or chained into a full pipeline below.")

    mode = st.radio("Mode", ["Individual operations", "Full pipeline"], horizontal=True)

    if mode == "Individual operations":
        op = st.selectbox("Operation", ["Grayscale", "Resizing", "Normalization", "Histogram equalization",
                                         "CLAHE", "Gaussian filter", "Average filter", "Median filter",
                                         "Noise reduction (median)", "Contrast enhancement (CLAHE)"])
        key_map = {"Grayscale": "grayscale", "Resizing": "resize", "Normalization": "normalize",
                   "Histogram equalization": "histeq", "CLAHE": "clahe", "Gaussian filter": "gaussian",
                   "Average filter": "average", "Median filter": "median",
                   "Noise reduction (median)": "median", "Contrast enhancement (CLAHE)": "clahe"}
        explain(key_map[op])

        if op in ("CLAHE", "Contrast enhancement (CLAHE)"):
            clip = st.slider("Clip limit", 1.0, 10.0, 2.0, 0.5, help="Higher = stronger local contrast")
            tile = st.select_slider("Tile grid size", [4, 8, 16, 32], 8)
            explain("clahe_tile")
            out = P.clahe(norm, clip, tile)
        elif op in ("Gaussian filter",):
            k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5)
            out = P.gaussian_blur(norm, k)
        elif op in ("Average filter",):
            k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5)
            out = P.average_blur(norm, k)
        elif op in ("Median filter", "Noise reduction (median)"):
            k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5)
            out = P.median_blur(norm, k)
        elif op == "Histogram equalization":
            out = P.hist_equalize(norm)
        elif op == "Normalization":
            out = norm
        elif op == "Resizing":
            out = gray
        else:
            out = gray

        c1, c2 = st.columns(2)
        c1.image(gray, caption="Before", clamp=True, width="stretch")
        c2.image(out, caption="After", clamp=True, width="stretch")

    else:
        st.subheader("Full preprocessing pipeline")
        clip = st.slider("CLAHE clip limit", 1.0, 6.0, 2.0, 0.5)
        tile = st.select_slider("CLAHE tile grid", [4, 8, 16], 8)
        flt = st.selectbox("Noise filter", ["Gaussian", "Average", "Median"])
        ksz = st.select_slider("Filter kernel", [3, 5, 7, 9], 5)
        explain("clahe")
        explain(flt.lower())

        heq = P.hist_equalize(norm)
        cl = P.clahe(norm, clip, tile)
        filt = P.FILTERS[flt](cl, ksz)

        st.session_state["preproc_filt"] = filt
        st.session_state["preproc_params"] = f"CLAHE(clip={clip}, tile={tile}) -> {flt} filter(k={ksz})"

        imgs = [gray, norm, heq, cl, filt]
        titles = ["Grayscale", "Normalized", "Hist. Equalized", "CLAHE", f"{flt} filtered"]
        fig = R.comparison_figure(imgs, titles, cols=5)
        st.pyplot(fig)
        st.download_button("Download comparison figure", R.fig_to_png(fig), "preprocessing_pipeline.png", "image/png")

# ===========================================================================
# 3. LUNG SEGMENTATION
# ===========================================================================
elif page == "Lung Segmentation":
    page_header("Section 3 / 8", "Lung Segmentation", "Classical (non-AI) extraction of lung regions, boundaries and masks.")
    st.caption("Classical (non-AI) segmentation. Always visually verify the mask — see warnings below if it fails.")

    seg_method = st.radio("Segmentation method", ["Classical (Otsu + morphology)", "K-means (3 clusters)"])
    min_area = st.slider("Minimum lung region area (px)", 300, 6000, 1500, 100,
                          help="Connected regions smaller than this are discarded as noise, not lung tissue.")
    explain("otsu"); explain("kmeans")

    filt = st.session_state.get("preproc_filt", P.clahe(norm, 2.0, 8))

    if seg_method.startswith("Classical"):
        mask = S.lung_mask(norm, min_area)
    else:
        km = S.kmeans_segment(filt)
        km = km * (S._valid_region(norm) > 0)
        mask = S.largest_two(km, min_area)

    n_regions = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)[0] - 1
    if n_regions < 2:
        st.warning("Segmentation did not find two separate lung regions. This can happen with unusual "
                    "exposure, rotated films, hardware overlapping the lungs, or pathology that changes "
                    "lung brightness. Try lowering 'Minimum lung region area' or switching segmentation method.")

    extraction_mode = st.selectbox("Extraction mode", [
        "Show segmented lungs (both)", "Show right lung only", "Show left lung only",
        "Show everything except lungs", "Show lung boundaries", "Show binary mask"])

    lr = A.split_left_right(mask) if n_regions >= 2 else []
    right_mask = left_mask = None
    if len(lr) == 2:
        right_mask = np.zeros_like(mask); cv2.drawContours(right_mask, [lr[0]], -1, 255, cv2.FILLED)
        left_mask = np.zeros_like(mask); cv2.drawContours(left_mask, [lr[1]], -1, 255, cv2.FILLED)

    c1, c2 = st.columns(2)
    c1.image(gray, caption="Input (grayscale, 512x512)", clamp=True, width="stretch")

    if extraction_mode == "Show segmented lungs (both)":
        c2.image(A.overlay(gray, mask)[:, :, ::-1], caption="Lung overlay", width="stretch")
    elif extraction_mode == "Show right lung only" and right_mask is not None:
        c2.image(A.overlay(gray, right_mask)[:, :, ::-1], caption="Right lung (patient's right)", width="stretch")
    elif extraction_mode == "Show left lung only" and left_mask is not None:
        c2.image(A.overlay(gray, left_mask)[:, :, ::-1], caption="Left lung (patient's left)", width="stretch")
    elif extraction_mode == "Show everything except lungs":
        c2.image(A.background_removed(gray, mask), caption="Background (lungs removed)", clamp=True, width="stretch")
    elif extraction_mode == "Show lung boundaries":
        c2.image(A.boundary_only(mask), caption="Lung boundaries", clamp=True, width="stretch")
    elif extraction_mode == "Show binary mask":
        c2.image(mask, caption="Binary lung mask", clamp=True, width="stretch")
    else:
        c2.info("That lung could not be isolated (fewer than two regions found).")

    st.subheader("All segmentation outputs")
    cols = st.columns(4)
    cols[0].image(S.otsu(filt), caption="Otsu thresholding", clamp=True, width="stretch")
    cols[1].image(S.adaptive(filt), caption="Adaptive thresholding", clamp=True, width="stretch")
    cols[2].image(mask, caption="Combined lung mask", clamp=True, width="stretch")
    cols[3].image(A.boundary_overlay(gray, mask)[:, :, ::-1], caption="Boundary overlay", width="stretch")

    st.subheader("Download extracted images / masks")
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("Download combined mask (PNG)", cv2.imencode(".png", mask)[1].tobytes(), "lung_mask.png")
    if right_mask is not None:
        dl2.download_button("Download right lung mask (PNG)", cv2.imencode(".png", right_mask)[1].tobytes(), "right_lung_mask.png")
    if left_mask is not None:
        dl3.download_button("Download left lung mask (PNG)", cv2.imencode(".png", left_mask)[1].tobytes(), "left_lung_mask.png")

    st.session_state["seg_mask"] = mask
    st.session_state["seg_method"] = seg_method

    with st.expander("How do the morphological cleanup steps work?"):
        explain("morph_open"); explain("morph_close")

# ===========================================================================
# 4. FEATURE EXTRACTION LAB
# ===========================================================================
elif page == "Feature Extraction Lab":
    page_header("Section 4 / 8", "Feature Extraction Lab", "Edges, corners, morphology, texture and shape descriptors — pick what you need.")
    category = st.selectbox("Feature category", ["Edge detection", "Corner detection",
                                                   "Morphological features", "Texture analysis",
                                                   "Shape & contour features"])
    filt = st.session_state.get("preproc_filt", P.clahe(norm, 2.0, 8))

    if category == "Edge detection":
        method = st.selectbox("Method", ["Sobel X", "Sobel Y", "Sobel magnitude",
                                          "Prewitt X", "Prewitt Y", "Canny"])
        if method.startswith("Sobel"):
            gx, gy = F.sobel_xy(filt)
            out = {"Sobel X": gx, "Sobel Y": gy, "Sobel magnitude": F.sobel(filt)}[method]
        elif method.startswith("Prewitt"):
            px, py = F.prewitt_xy(filt)
            out = px if method == "Prewitt X" else py
        else:
            c1_ = st.slider("Canny low threshold", 10, 290, 50)
            c2_ = st.slider("Canny high threshold", 20, 300, 150)
            explain("canny_low"); explain("canny_high")
            out = F.canny(filt, c1_, c2_)
        st.image(out, caption=method, clamp=True, width="stretch")

    elif category == "Corner detection":
        method = st.selectbox("Method", ["Harris", "Shi-Tomasi"])
        explain("harris" if method == "Harris" else "shi_tomasi")
        out = F.harris(filt) if method == "Harris" else F.shi_tomasi(filt)
        st.image(out[:, :, ::-1], caption=f"{method} corners (red)", width="stretch")

    elif category == "Morphological features":
        op = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Morphological gradient"])
        k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5)
        binary = S.otsu(filt)
        out = {"Erosion": F.morph_erode, "Dilation": F.morph_dilate, "Opening": S.opening,
               "Closing": S.closing, "Morphological gradient": F.morph_gradient}[op](binary, k)
        explain({"Erosion": "morph_open", "Dilation": "morph_close", "Opening": "morph_open",
                 "Closing": "morph_close", "Morphological gradient": "morph_gradient"}[op])
        c1, c2 = st.columns(2)
        c1.image(binary, caption="Input binary (Otsu)", clamp=True, width="stretch")
        c2.image(out, caption=op, clamp=True, width="stretch")

    elif category == "Texture analysis":
        explain("lbp"); explain("glcm")
        p = st.slider("LBP neighbours (P)", 4, 16, 8)
        r = st.slider("LBP radius (R)", 1, 3, 1)
        lbp_img, hist = T.lbp(filt, p, r)
        c1, c2 = st.columns(2)
        c1.image(cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                  caption="LBP code image", clamp=True, width="stretch")
        c2.bar_chart(hist)
        feats = T.glcm_features(filt)
        st.subheader("GLCM texture scores")
        a, b, c, d, e = st.columns(5)
        a.metric("Contrast", f"{feats['contrast']:.2f}")
        b.metric("Homogeneity", f"{feats['homogeneity']:.3f}")
        c.metric("Energy", f"{feats['energy']:.3f}")
        d.metric("Correlation", f"{feats['correlation']:.3f}")
        e.metric("Entropy", f"{feats['entropy']:.2f}")

    else:  # Shape & contour features
        mask = st.session_state.get("seg_mask")
        if mask is None:
            st.info("Run Lung Segmentation first to extract a mask for shape analysis.")
        else:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                st.warning("No contours found in the current mask.")
            else:
                idx = st.selectbox("Contour", list(range(len(cnts))), format_func=lambda i: f"Region {i+1}")
                s = A.contour_features(cnts[idx])
                st.json(s)
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vis, [cnts[idx]], -1, (0, 255, 0), 2)
                hull = cv2.convexHull(cnts[idx])
                cv2.drawContours(vis, [hull], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(cnts[idx])
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
                st.image(vis[:, :, ::-1], caption="Contour (green), convex hull (red), bounding box (yellow)", width="stretch")

# ===========================================================================
# 5. RADIOGRAPHIC FINDINGS EXPLORER
# ===========================================================================
elif page == "Radiographic Findings Explorer":
    page_header("Section 5 / 8", "Radiographic Findings Explorer", "Educational reference only — PulmoMap does not detect any of these findings.")
    st.markdown(f'<div class="pm-disclaimer">⚠️ {FINDINGS_DISCLAIMER}</div>', unsafe_allow_html=True)

    names = [f["name"] for f in FINDINGS]
    choice = st.selectbox("Finding", names)
    f = next(x for x in FINDINGS if x["name"] == choice)
    st.subheader(f["name"])
    st.markdown(f"**Definition:** {f['definition']}")
    st.markdown(f"**Typical appearance:** {f['appearance']}")
    st.markdown(f"**Anatomical location:** {f['location']}")
    st.markdown(f"**Why it matters:** {f['importance']}")
    with st.expander("How might image processing help visualize this (experimental, not detection)?"):
        st.write(f["ip_relevance"])
        st.caption("Experimental — these are generic image-processing tools, not a trained detector for this finding.")

    st.divider()
    st.subheader("Interactive ROI workspace")
    st.caption("Select a rectangular region to inspect intensity, texture, edges and shape locally.")
    H, W = gray.shape
    x = st.slider("ROI x (left edge)", 0, W - 10, W // 4)
    y = st.slider("ROI y (top edge)", 0, H - 10, H // 4)
    w = st.slider("ROI width", 10, W - x, min(150, W - x))
    h = st.slider("ROI height", 10, H - y, min(150, H - y))

    crop, box = ROI.rect_roi(gray, x, y, w, h)
    stats = ROI.roi_stats(crop)
    edges = ROI.roi_edges(crop)

    c1, c2, c3 = st.columns(3)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    c1.image(vis[:, :, ::-1], caption="Selected region (full image)", width="stretch")
    c2.image(crop, caption="Cropped ROI", clamp=True, width="stretch")
    c3.image(edges, caption="ROI edge map (Canny)", clamp=True, width="stretch")

    if stats:
        a, b, c, d = st.columns(4)
        a.metric("Mean intensity", f"{stats['mean_intensity']:.1f}")
        b.metric("Std deviation", f"{stats['std_intensity']:.1f}")
        c.metric("Min", f"{stats['min_intensity']:.0f}")
        d.metric("Max", f"{stats['max_intensity']:.0f}")
        hist, _ = P.histogram(crop)
        st.caption("Local intensity histogram")
        st.bar_chart(hist)
        feats = T.glcm_features(crop) if crop.size > 0 else {}
        if feats:
            st.caption(f"ROI texture — contrast {feats['contrast']:.2f}, homogeneity {feats['homogeneity']:.3f}, "
                        f"energy {feats['energy']:.3f}, entropy {feats['entropy']:.2f}")

# ===========================================================================
# 6. ANATOMICAL ANALYSIS
# ===========================================================================
elif page == "Anatomical Analysis":
    page_header("Section 6 / 8", "Anatomical Analysis", "Per-lung geometry and right/left symmetry, in pixels.")
    mask = st.session_state.get("seg_mask")
    if mask is None:
        st.info("Run Lung Segmentation first — this page analyses the resulting mask.")
    else:
        stats = A.analyse(mask)
        if stats is None:
            st.warning("No analysis available: two separate lung regions were not detected in the current mask.")
        else:
            st.caption("Pixel-based measurements only — not physical units unless the image is calibrated.")
            for name in [k for k in stats if k != "symmetry"]:
                v = stats[name]
                st.subheader(name)
                a, b, c, d = st.columns(4)
                a.metric("Area (px)", f"{v['area_px']:.0f}")
                b.metric("Perimeter (px)", f"{v['perimeter_px']:.0f}")
                c.metric("Circularity", f"{v['circularity']:.2f}")
                d.metric("Solidity", f"{v['solidity']:.2f}")
                st.caption(f"Bounding box {v['bbox_w']}x{v['bbox_h']} px · aspect ratio {v['aspect_ratio']:.2f} "
                            f"· centroid ({v['centroid'][0]:.0f}, {v['centroid'][1]:.0f})")
                st.caption("Zone distribution (% of lung area in upper/middle/lower thirds):")
                st.bar_chart(v["zones_%"], height=160)

            s = stats["symmetry"]
            st.subheader("Right vs. Left Symmetry")
            a, b, c = st.columns(3)
            a.metric("Area ratio R/L", f"{s['area_ratio_R/L']:.2f}")
            b.metric("Symmetry index", f"{s['symmetry_index']:.3f}")
            c.metric("Mirror IoU", f"{s['mirror_iou']:.3f}")
            with st.expander("What do these mean?"):
                st.markdown("""
- **Area ratio R/L**: right lung area divided by left lung area in pixels; 1.0 means equal area.
- **Symmetry index**: 1 minus the normalized area difference; closer to 1 means more symmetric.
- **Mirror IoU**: reflects the right lung across the image midline and measures overlap (Intersection
  over Union) with the left lung — a shape-symmetry measure, not just area.
""")
            st.caption("Differences here are descriptive geometry only and must not be interpreted as "
                        "evidence of disease.")
            st.session_state["anatomy_stats"] = stats

# ===========================================================================
# 7. COMPARISON DASHBOARD
# ===========================================================================
elif page == "Comparison Dashboard":
    page_header("Section 7 / 8", "Comparison Dashboard", "Side-by-side before/after and method-vs-method views.")
    comparison = st.selectbox("Comparison", [
        "Original vs. Grayscale", "Original vs. Normalized", "Original vs. CLAHE",
        "Filtering methods (Gaussian / Average / Median)", "Edge detection algorithms",
        "Segmentation methods (Otsu / Adaptive / K-means)", "Right vs. Left lung"])

    if comparison == "Original vs. Grayscale":
        imgs, titles = [orig_rgb, gray], ["Original", "Grayscale"]
    elif comparison == "Original vs. Normalized":
        imgs, titles = [gray, norm], ["Grayscale", "Normalized"]
    elif comparison == "Original vs. CLAHE":
        imgs, titles = [gray, P.clahe(norm, 2.0, 8)], ["Grayscale", "CLAHE"]
    elif comparison == "Filtering methods (Gaussian / Average / Median)":
        k = st.select_slider("Kernel size", [3, 5, 7, 9], 5)
        cl = P.clahe(norm, 2.0, 8)
        imgs = [gray, P.gaussian_blur(cl, k), P.average_blur(cl, k), P.median_blur(cl, k)]
        titles = ["Original", "Gaussian", "Average", "Median"]
    elif comparison == "Edge detection algorithms":
        cl = P.clahe(norm, 2.0, 8)
        imgs = [F.sobel(cl), F.canny(cl, 50, 150), F.harris(cl)[:, :, ::-1]]
        titles = ["Sobel magnitude", "Canny", "Harris corners"]
    elif comparison == "Segmentation methods (Otsu / Adaptive / K-means)":
        cl = P.clahe(norm, 2.0, 8)
        km = S.kmeans_segment(cl) * (S._valid_region(norm) > 0)
        imgs = [S.otsu(cl), S.adaptive(cl), km]
        titles = ["Otsu", "Adaptive", "K-means"]
    else:
        mask = st.session_state.get("seg_mask")
        if mask is None:
            st.info("Run Lung Segmentation first to compare left vs. right lung.")
            imgs, titles = [], []
        else:
            lr = A.split_left_right(mask)
            if len(lr) < 2:
                st.warning("Two lungs were not detected in the current mask.")
                imgs, titles = [], []
            else:
                rmask = np.zeros_like(mask); cv2.drawContours(rmask, [lr[0]], -1, 255, cv2.FILLED)
                lmask = np.zeros_like(mask); cv2.drawContours(lmask, [lr[1]], -1, 255, cv2.FILLED)
                imgs = [rmask, lmask]
                titles = ["Right lung mask", "Left lung mask"]

    if imgs:
        fig = R.comparison_figure(imgs, titles, cols=min(4, len(imgs)))
        st.pyplot(fig)
        st.download_button("Download comparison figure", R.fig_to_png(fig), "comparison.png", "image/png")

# ===========================================================================
# 8. DETAILED REPORT
# ===========================================================================
elif page == "Detailed Report":
    page_header("Section 8 / 8", "Detailed Report", "Everything run this session, consolidated into a downloadable report.")
    anatomy = st.session_state.get("anatomy_stats")
    info = {
        "input": {"filename": src_name, "size_px": f"{gray.shape[1]}x{gray.shape[0]}"},
        "preprocessing": [st.session_state.get("preproc_params", "No full pipeline run — defaults used.")],
        "segmentation": [st.session_state.get("seg_method", "Not yet run")],
        "anatomy": anatomy,
    }

    sections = R.detailed_sections(info)
    txt = R.sections_to_text(sections)
    st.code(txt)

    c1, c2 = st.columns(2)
    c1.download_button("Download report (.txt)", txt, "pulmomap_report.txt")

    pdf_images = []
    mask = st.session_state.get("seg_mask")
    if mask is not None:
        fig = R.comparison_figure([gray, mask, A.overlay(gray, mask)[:, :, ::-1]],
                                   ["Input", "Lung mask", "Overlay"], cols=3)
        pdf_images.append(("Segmentation visualization", R.fig_to_png(fig)))
    try:
        pdf_bytes = EXP.build_pdf(sections, pdf_images, title="PulmoMap Detailed Report")
        c2.download_button("Download report (.pdf)", pdf_bytes, "pulmomap_report.pdf", "application/pdf")
    except Exception as e:
        st.warning(f"PDF export unavailable: {e}")

    if anatomy is None:
        st.info("Run Lung Segmentation and Anatomical Analysis first for a complete report.")
