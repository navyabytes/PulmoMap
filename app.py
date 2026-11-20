import hashlib

import cv2
import numpy as np
import streamlit as st

from pulmomap import preprocessing as P, features as F, segmentation as S, analysis as A, report as R
from pulmomap import texture as T, roi as ROI, export as EXP
from pulmomap import anatomy as ANAT, candidate_regions as CAND
from pulmomap.explanations import explain, DISCLAIMER
from pulmomap.findings import FINDINGS, FINDINGS_DISCLAIMER

st.set_page_config(page_title="PulmoMap", page_icon="🫁", layout="wide")

# ---------------------------------------------------------------------------
# Visual language: clean dark dashboard. Minimal borders/cards — reserved for
# things that genuinely need to stand out (safety disclaimer, warnings).
# Everything else is plain text/headings so images stay the visual focus.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0e0f; --panel: #121617; --panel-2: #171c1e; --border: #262b2d;
  --text: #eef1f2; --muted: #8b969a;
  --accent: #19b89a; --accent-hover: #2bd6b6;
  --warning: #f5a524; --warning-light: rgba(245,165,36,.10);
  --danger: #ff6b6b; --success: #22c55e;
}
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top: 1.8rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--text); }
p, label, .stMarkdown { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

.pm-title { font-size: 2rem; font-weight: 700; margin: 0; }
.pm-subtitle { font-size: .95rem; color: var(--muted); margin-top: .2rem; margin-bottom: 1rem; }
.pm-warn { border-left: 3px solid var(--warning); background: var(--warning-light);
           border-radius: 8px; padding: .75rem 1rem; margin: .6rem 0; font-size: .88rem; }

div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border);
                               border-radius: 8px; padding: .5rem .7rem; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; color: var(--text); }
div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; letter-spacing: .05em;
                                    text-transform: uppercase; }

button[data-baseweb="tab"] { font-size: .95rem; }
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }

.stButton button, .stDownloadButton button {
  border-radius: 7px; border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
}
.stButton button:hover, .stDownloadButton button:hover { border-color: var(--accent); color: var(--accent-hover); }

details[data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def page_header(eyebrow, title):
    st.markdown(f'<p style="color:var(--accent); font-size:.75rem; font-weight:600; '
                f'letter-spacing:.1em; text-transform:uppercase; margin-bottom:.1rem;">{eyebrow}</p>'
                f'<p class="pm-title" style="font-size:1.4rem;">{title}</p>', unsafe_allow_html=True)


st.markdown('<p class="pm-title">🫁 PulmoMap</p>', unsafe_allow_html=True)
st.markdown('<p class="pm-subtitle">Chest X-ray image processing, anatomical segmentation and feature '
            'exploration — classical computer vision, explained in plain language.</p>', unsafe_allow_html=True)
st.markdown(f'<div class="pm-warn">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)
st.caption("✅ Image processing and anatomical segmentation are implemented with classical algorithms below. "
           "🚫 Abnormality/disease detection is **not** implemented — that needs a separately trained, "
           "clinically validated model. PulmoMap only extracts descriptive features for later reference.")

# ---------------------------------------------------------------------------
# Navigation & input
# ---------------------------------------------------------------------------
WORKSPACES = ["Getting Started", "Image Processing", "Anatomical Analysis", "Feature Extraction", "Results & Reports"]

with st.sidebar:
    st.markdown("**Presentation Mode**")
    presentation_mode = st.checkbox("Guided step-by-step demo (for presenting to others)", value=False,
                                     key="sidebar_presentation_mode")
    st.divider()
    if not presentation_mode:
        st.markdown("**Workspace**")
        workspace = st.radio("Workspace", WORKSPACES, label_visibility="collapsed", key="sidebar_workspace")
    else:
        workspace = None
    st.divider()
    st.caption("Upload once, used across every workspace below.")
    up = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                           key="sidebar_file_uploader")
    use_sample = st.checkbox("Use the bundled sample image instead", key="sidebar_use_sample")

if use_sample:
    img = cv2.imread("data/samples/sample_cxr.png", cv2.IMREAD_COLOR)
    src_name = "sample_cxr.png"
elif up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    src_name = up.name
else:
    img = None
    src_name = None

if img is not None and img.size == 0:
    st.error("Could not decode that image. Try a different file.")
    st.stop()

if not presentation_mode and img is None and workspace != "Getting Started":
    st.info("Upload a chest X-ray (or check 'Use the bundled sample image') in the sidebar to use this workspace.")
    st.stop()

IMAGE_DERIVED_KEYS = ["seg_mask", "seg_method", "seg_min_area_used", "preproc_filt", "preproc_params", "anatomy_stats"]

if img is not None:
    orig_rgb = img[:, :, ::-1]
    gray_full = P.to_grayscale(img)
    gray = P.resize_preserve_aspect(gray_full)
    norm = P.min_max_normalize(gray)

    # A different image (new upload, or switching sample<->upload) must invalidate every
    # previously-computed mask/measurement — otherwise a stale segmentation from the last
    # image could silently be reused on this one.
    image_id = f"{src_name}:{hashlib.md5(img.tobytes()).hexdigest()}"
    if st.session_state.get("image_id") != image_id:
        for k in IMAGE_DERIVED_KEYS:
            st.session_state.pop(k, None)
        st.session_state["image_id"] = image_id


def run_lung_segmentation(seg_method="Classical (Otsu + morphology)", min_area=1500):
    """Single source of truth for lung segmentation — used by the Lung Segmentation workspace
    itself and by every other workspace's "run it for me" fallback, so the algorithm is never
    duplicated. Stores the mask (and the settings used) in session state for every workspace."""
    filt = st.session_state.get("preproc_filt", P.clahe(norm, 2.0, 8))
    if seg_method.startswith("Classical"):
        mask = S.lung_mask(norm, min_area)
    else:
        km = S.kmeans_segment(filt)
        km = km * (S._valid_region(norm) > 0)
        mask = S.largest_two(km, min_area)
    st.session_state["seg_mask"] = mask
    st.session_state["seg_method"] = seg_method
    st.session_state["seg_min_area_used"] = min_area
    return mask


def ensure_lung_mask(reason, button_key):
    """Shows the existing mask if one is already in session state; otherwise explains why a
    mask is needed and offers a one-click button to compute it with sensible defaults — no
    manual trip to another workspace required. Returns the mask, or None if not yet computed."""
    mask = st.session_state.get("seg_mask")
    if mask is not None:
        return mask
    st.info(reason)
    if st.button("Run Lung Segmentation now (classical, default settings)", key=button_key):
        with st.spinner("Running lung segmentation..."):
            run_lung_segmentation("Classical (Otsu + morphology)", 1500)
        st.rerun()
    return None


# ===========================================================================
# VIEW FUNCTIONS — one per preserved capability. Reused by both the normal
# 5-workspace/tab navigation and by Presentation Mode's guided sequence.
# ===========================================================================

def view_getting_started():
    st.markdown("""
PulmoMap lets you explore a chest X-ray through classical (non-AI) image-processing and computer-vision
techniques. It is built for two kinds of users:

- **Beginners** who want to understand *what each operation does* in plain English.
- **Technical users** who want to extract anatomical structures, image features, and build a feature
  report for further reference.

**How to use it**
1. Upload an image in the sidebar (or tick "Use the bundled sample image").
2. Pick a workspace from the sidebar: Image Processing, Anatomical Analysis, Feature Extraction, or
   Results & Reports. Each workspace groups related tools into tabs.
3. Every control has a short plain-language explanation beside it; an optional "Technical explanation"
   expander has the formulas, for anyone who wants them.

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


OP_BLURBS = {
    "grayscale": "Keeps only brightness, discarding colour (X-rays carry no real colour anyway).",
    "resize": "Fits the image to a standard size so every image is processed the same way.",
    "normalize": "Stretches brightness to use the full black-to-white range. If the image already "
                 "spans 0-255, this has no visible effect — that's expected, not a bug.",
    "histeq": "Spreads out brightness so faint details become easier to see.",
    "clahe": "Boosts local contrast in small patches, revealing detail without washing out the whole image.",
    "gaussian": "Smooths the image gently, reducing general noise while keeping edges fairly sharp.",
    "average": "Smooths the image by simple averaging — a basic, slightly blurrier version of Gaussian.",
    "median": "Removes isolated noise specks by replacing each pixel with its neighbourhood's median.",
}


def view_preprocessing():
    st.caption("Each operation can be inspected individually, or chained into a full pipeline below.")
    mode = st.radio("Mode", ["Individual operations", "Full pipeline"], horizontal=True, key="preproc_mode")

    if mode == "Individual operations":
        op = st.selectbox("Operation", ["Grayscale", "Resizing", "Normalization", "Histogram equalization",
                                         "CLAHE", "Gaussian filter", "Average filter", "Median filter"],
                           key="preproc_op")
        key_map = {"Grayscale": "grayscale", "Resizing": "resize", "Normalization": "normalize",
                   "Histogram equalization": "histeq", "CLAHE": "clahe", "Gaussian filter": "gaussian",
                   "Average filter": "average", "Median filter": "median"}
        op_key = key_map[op]
        st.caption("💡 " + OP_BLURBS[op_key])

        if op == "CLAHE":
            clip = st.slider("Clip limit", 1.0, 10.0, 2.0, 0.5, help="Higher = stronger local contrast",
                              key="preproc_clahe_clip")
            with st.expander("Advanced settings"):
                tile = st.select_slider("Tile grid size", [4, 8, 16, 32], 8, key="preproc_clahe_tile")
            out = P.clahe(norm, clip, tile)
            params = f"CLAHE(clip={clip}, tile={tile})"
        elif op == "Gaussian filter":
            k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5, key="preproc_gaussian_k")
            out = P.gaussian_blur(norm, k)
            params = f"Gaussian(kernel={k})"
        elif op == "Average filter":
            k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5, key="preproc_average_k")
            out = P.average_blur(norm, k)
            params = f"Average(kernel={k})"
        elif op == "Median filter":
            k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5, key="preproc_median_k")
            out = P.median_blur(norm, k)
            params = f"Median(kernel={k})"
        elif op == "Histogram equalization":
            out = P.hist_equalize(norm)
            params = "Histogram equalization (no parameters)"
        elif op == "Normalization":
            out, params = norm, "Min-max normalization"
        else:
            out, params = gray, "Resize to 512x512"

        # Unrounded diagnostics — computed once, reused by every display below, so the
        # visualization and the numbers can never drift apart (same arrays, same math).
        before_stats, after_stats = P.image_stats(gray), P.image_stats(out)
        shapes_match = out.shape == gray.shape
        g64, o64 = gray.astype(np.float64), out.astype(np.float64)
        abs_diff = np.abs(g64 - o64) if shapes_match else None
        mean_abs_diff = float(abs_diff.mean()) if shapes_match else float("nan")
        max_abs_diff = float(abs_diff.max()) if shapes_match else float("nan")
        pct_changed = float(100.0 * np.count_nonzero(abs_diff > 0) / abs_diff.size) if shapes_match else float("nan")
        is_near_identical = shapes_match and mean_abs_diff < 0.5

        tab1, tab2, tab3 = st.tabs(["Visual Results", "Numerical Measurements", "How It Works"])
        with tab1:
            diff_map = cv2.normalize(abs_diff.astype(np.uint8), None, 0, 255,
                                      cv2.NORM_MINMAX, dtype=cv2.CV_8U) if shapes_match else None
            c1, c2, c3 = st.columns(3)
            c1.image(gray, caption="Original", clamp=True, width="stretch")
            c2.image(out, caption=f"Processed ({op})", clamp=True, width="stretch")
            if diff_map is not None:
                diff_caption = "Difference map (brighter = more change)"
                c3.image(diff_map, caption=diff_caption, clamp=True, width="stretch")
                if is_near_identical:
                    c3.caption("Appears black because the actual difference is genuinely ~0 here "
                               "(see Numerical Measurements) — not a rendering error.")
            else:
                c3.info("No difference map: this operation changes the image dimensions.")

        with tab2:
            st.caption("Computed directly from the Original and Processed arrays shown above — same data, no rounding "
                       "applied internally (displayed to 2-4 decimals for readability only).")
            a, b = st.columns(2)
            a.metric("Shape (before)", f"{gray.shape[1]}x{gray.shape[0]}")
            b.metric("Shape (after)", f"{out.shape[1]}x{out.shape[0]}")
            a.metric("Dtype (before)", str(gray.dtype)); b.metric("Dtype (after)", str(out.dtype))
            a.metric("Min (before)", f"{before_stats['min']:.0f}"); b.metric("Min (after)", f"{after_stats['min']:.0f}")
            a.metric("Max (before)", f"{before_stats['max']:.0f}"); b.metric("Max (after)", f"{after_stats['max']:.0f}")
            a.metric("Mean (before)", f"{before_stats['mean']:.4f}"); b.metric("Mean (after)", f"{after_stats['mean']:.4f}")
            a.metric("Std dev (before)", f"{before_stats['std']:.4f}"); b.metric("Std dev (after)", f"{after_stats['std']:.4f}")
            if shapes_match:
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean absolute difference", f"{mean_abs_diff:.6f} / 255")
                m2.metric("Max absolute difference", f"{max_abs_diff:.0f} / 255")
                m3.metric("Pixels changed", f"{pct_changed:.2f}%")
            st.caption(f"Parameters used: {params}")
            if is_near_identical:
                st.warning(f"**Why no visible change:** this operation ({op}) is mathematically a near no-op "
                           f"on this image. For normalization specifically: the input already spans "
                           f"{before_stats['min']:.0f}-{before_stats['max']:.0f}, so min-max stretching to "
                           f"0-255 changes nothing — the *algorithm* is still correct, this image just has "
                           f"nothing left to stretch. An image with a narrower intensity range would show a "
                           f"large, visible change from the same function.")
        with tab3:
            explain(op_key)

    else:
        st.subheader("Full preprocessing pipeline")
        st.caption("💡 Chains CLAHE (local contrast) and a noise filter — a typical preparation before "
                   "segmentation or feature extraction.")
        clip = st.slider("CLAHE clip limit", 1.0, 6.0, 2.0, 0.5, key="preproc_pipeline_clip")
        tile = st.select_slider("CLAHE tile grid", [4, 8, 16], 8, key="preproc_pipeline_tile")
        flt = st.selectbox("Noise filter", ["Gaussian", "Average", "Median"], key="preproc_pipeline_filter")
        ksz = st.select_slider("Filter kernel", [3, 5, 7, 9], 5, key="preproc_pipeline_kernel")
        explain("clahe"); explain(flt.lower())

        heq = P.hist_equalize(norm)
        cl = P.clahe(norm, clip, tile)
        filt = P.FILTERS[flt](cl, ksz)

        st.session_state["preproc_filt"] = filt
        st.session_state["preproc_params"] = f"CLAHE(clip={clip}, tile={tile}) -> {flt} filter(k={ksz})"

        imgs = [gray, norm, heq, cl, filt]
        titles = ["Grayscale", "Normalized", "Hist. Equalized", "CLAHE", f"{flt} filtered"]
        fig = R.comparison_figure(imgs, titles, cols=5)
        st.pyplot(fig)
        st.download_button("Download comparison figure", R.fig_to_png(fig), "preprocessing_pipeline.png", "image/png",
                            key="preproc_dl_pipeline_fig")


def view_segmentation():
    st.caption("Classical (non-AI) segmentation. Always visually verify the mask — see warnings below if it fails.")

    seg_method = st.radio("Segmentation method", ["Classical (Otsu + morphology)", "K-means (3 clusters)"],
                           key="seg_method_radio")
    st.caption("💡 Finds the two darkest large regions in the chest (air-filled lungs show up dark on X-ray) "
               "and treats them as the lung fields.")
    min_area = st.slider("Minimum lung region area (px)", 300, 6000, 1500, 100,
                          help="Connected regions smaller than this are discarded as noise, not lung tissue.",
                          key="seg_min_area")
    explain("otsu"); explain("kmeans")

    mask = run_lung_segmentation(seg_method, min_area)
    filt = st.session_state.get("preproc_filt", P.clahe(norm, 2.0, 8))

    n_regions = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)[0] - 1
    if n_regions < 2:
        st.warning("Segmentation did not find two separate lung regions. This can happen with unusual "
                    "exposure, rotated films, hardware overlapping the lungs, or pathology that changes "
                    "lung brightness. Try lowering 'Minimum lung region area' or switching segmentation method.")

    extraction_mode = st.selectbox("Extraction mode", [
        "Show segmented lungs (both)", "Show right lung only", "Show left lung only",
        "Show everything except lungs", "Show lung boundaries", "Show binary mask"],
        key="seg_extraction_mode")

    lr = A.split_left_right(mask) if n_regions >= 2 else []
    right_mask = left_mask = None
    if len(lr) == 2:
        right_mask = np.zeros_like(mask); cv2.drawContours(right_mask, [lr[0]], -1, 255, cv2.FILLED)
        left_mask = np.zeros_like(mask); cv2.drawContours(left_mask, [lr[1]], -1, 255, cv2.FILLED)

    tab1, tab2, tab3 = st.tabs(["Result", "All segmentation outputs", "Download"])
    with tab1:
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
        st.caption(f"Parameters used: {seg_method}, minimum area={min_area}px")

    with tab2:
        cols = st.columns(4)
        cols[0].image(S.otsu(filt), caption="Otsu thresholding", clamp=True, width="stretch")
        cols[1].image(S.adaptive(filt), caption="Adaptive thresholding", clamp=True, width="stretch")
        cols[2].image(mask, caption="Combined lung mask", clamp=True, width="stretch")
        cols[3].image(A.boundary_overlay(gray, mask)[:, :, ::-1], caption="Boundary overlay", width="stretch")
        with st.expander("How do the morphological cleanup steps work?"):
            explain("morph_open"); explain("morph_close")

    with tab3:
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("Combined mask (PNG)", cv2.imencode(".png", mask)[1].tobytes(), "lung_mask.png",
                             key="seg_dl_combined")
        if right_mask is not None:
            dl2.download_button("Right lung mask (PNG)", cv2.imencode(".png", right_mask)[1].tobytes(), "right_lung_mask.png",
                                 key="seg_dl_right")
        if left_mask is not None:
            dl3.download_button("Left lung mask (PNG)", cv2.imencode(".png", left_mask)[1].tobytes(), "left_lung_mask.png",
                                 key="seg_dl_left")


def view_whole_image_features():
    st.caption("These statistics describe the whole image as uploaded — not just the lungs.")
    tabs = st.tabs(["Intensity", "Edges & texture maps", "Contrast", "Structure", "3x3 grid"])

    with tabs[0]:
        ifeat = ANAT.intensity_features(gray)
        a, b, c, d, e = st.columns(5)
        a.metric("Mean", f"{ifeat['mean']:.1f}"); b.metric("Median", f"{ifeat['median']:.1f}")
        c.metric("Std dev", f"{ifeat['std']:.1f}"); d.metric("Min", f"{ifeat['min']:.0f}"); e.metric("Max", f"{ifeat['max']:.0f}")
        st.caption(f"Percentiles — p5 {ifeat['p5']:.0f}, p25 {ifeat['p25']:.0f}, p50 {ifeat['p50']:.0f}, "
                   f"p75 {ifeat['p75']:.0f}, p95 {ifeat['p95']:.0f}")
        hist, _ = P.histogram(gray)
        st.bar_chart(hist)

    with tabs[1]:
        t1_ = st.slider("Canny low threshold", 10, 290, 50, key="wif_canny_low")
        t2_ = st.slider("Canny high threshold", 20, 300, 150, key="wif_canny_high")
        st.caption("💡 Canny finds boundaries where brightness changes sharply — the edges of anatomy and hardware.")
        efeat = ANAT.edge_features(gray, t1_, t2_)
        cols = st.columns(3)
        cols[0].image(efeat["sobel_magnitude"], caption="Sobel magnitude", clamp=True, width="stretch")
        cols[1].image(efeat["canny"], caption="Canny edges", clamp=True, width="stretch")
        cols[2].metric("Edge density", f"{efeat['edge_density']*100:.2f}%",
                        help="Fraction of pixels classified as an edge — a rough 'how busy is this image' measure.")

        st.markdown("**Per-pixel variation maps** — always render, independent of any detection threshold.")
        k_map = st.select_slider("Local window size (px)", [9, 15, 21, 27, 31], 15, key="wif_local_window")
        mcols = st.columns(3)
        mcols[0].image(ANAT.local_contrast_map(gray, k_map), caption="Local contrast map", clamp=True, width="stretch")
        mcols[1].image(ANAT.texture_variation_map(gray, k_map), caption="Texture variation heatmap", clamp=True, width="stretch")
        mcols[2].image(ANAT.edge_density_heatmap(gray, k_map, t1_, t2_), caption="Edge-density heatmap", clamp=True, width="stretch")

        st.markdown("**Texture (LBP / GLCM)**")
        tfeat = ANAT.texture_features(gray)
        a, b, c, d, e = st.columns(5)
        a.metric("Contrast", f"{tfeat['contrast']:.2f}"); b.metric("Homogeneity", f"{tfeat['homogeneity']:.3f}")
        c.metric("Energy", f"{tfeat['energy']:.3f}"); d.metric("Correlation", f"{tfeat['correlation']:.3f}")
        e.metric("Entropy", f"{tfeat['entropy']:.2f}")
        st.bar_chart(tfeat["lbp_hist"])
        explain("lbp"); explain("glcm")

    with tabs[2]:
        clip = st.slider("CLAHE clip limit", 1.0, 10.0, 2.0, 0.5, key="whole_img_clahe_clip")
        adaptive_block = st.select_slider("Adaptive threshold block size", [11, 21, 31, 51, 71], 31, key="wif_adaptive_block")
        ccols = st.columns(3)
        ccols[0].image(P.clahe(norm, clip, 8), caption="CLAHE output", clamp=True, width="stretch")
        ccols[1].image(ANAT.z_score_map(gray, 15), caption="Local intensity variation (z-score map)", clamp=True, width="stretch")
        ccols[2].image(S.adaptive(gray, adaptive_block, 5), caption="Adaptive threshold output", clamp=True, width="stretch")

    with tabs[3]:
        st.caption("Every connected component of a simple Otsu threshold — a structural summary "
                   "(bounding boxes, centroids, ids), not a semantic segmentation of any organ.")
        cc_vis, cc_total, cc_kept = ANAT.connected_components_overlay(S.otsu(gray), gray, min_area=30, max_components=60)
        st.image(cc_vis[:, :, ::-1], caption=f"{cc_total} components total, {cc_kept} shown (min area 30px)", width="stretch")
        st.markdown("**Shape & contour features (top edge-based contours)**")
        shapes = ANAT.shape_features(gray)
        if shapes:
            st.dataframe(shapes, width="stretch")
        else:
            st.info("No sizeable contours found at the default Canny thresholds.")

    with tabs[4]:
        st.caption("Divides the image into a 3x3 grid so you can compare regions without assuming any "
                   "lung/organ boundary.")
        gstats = ANAT.grid_stats(gray, 3)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for label, s in gstats.items():
            x, y, w, h = s["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (25, 184, 154), 1)
            cv2.putText(vis, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 184, 154), 1)
        st.image(vis[:, :, ::-1], caption="3x3 grid overlay", width="stretch")
        cell = st.selectbox("Inspect a grid cell", list(gstats.keys()), key="wif_grid_cell")
        x, y, w, h = gstats[cell]["bbox"]
        crop = gray[y:y + h, x:x + w]
        cc1, cc2 = st.columns(2)
        cc1.image(crop, caption=f"Grid cell {cell}", clamp=True, width="stretch")
        cc2.metric("Mean intensity", f"{gstats[cell]['mean']:.1f}")
        cc2.metric("Std deviation", f"{gstats[cell]['std']:.1f}")
        st.table({label: {"mean": round(s["mean"], 1), "std": round(s["std"], 1)} for label, s in gstats.items()})


def view_feature_lab(default_category=None):
    categories = ["Edge detection", "Corner detection", "Morphological features",
                  "Texture analysis", "Shape & contour features"]
    idx = categories.index(default_category) if default_category in categories else 0
    category = st.selectbox("Feature category", categories, index=idx, key="flab_category")
    filt = st.session_state.get("preproc_filt", P.clahe(norm, 2.0, 8))

    if category == "Edge detection":
        method = st.selectbox("Method", ["Sobel X", "Sobel Y", "Sobel magnitude",
                                          "Prewitt X", "Prewitt Y", "Canny"], key="flab_edge_method")
        st.caption("💡 Finds boundaries where pixel intensity changes sharply — useful for visualizing "
                   "structural edges in the X-ray.")
        if method.startswith("Sobel"):
            gx, gy = F.sobel_xy(filt)
            out = {"Sobel X": gx, "Sobel Y": gy, "Sobel magnitude": F.sobel(filt)}[method]
        elif method.startswith("Prewitt"):
            px, py = F.prewitt_xy(filt)
            out = px if method == "Prewitt X" else py
        else:
            c1_ = st.slider("Canny low threshold", 10, 290, 50, key="flab_canny_low")
            c2_ = st.slider("Canny high threshold", 20, 300, 150, key="flab_canny_high")
            explain("canny_low"); explain("canny_high")
            out = F.canny(filt, c1_, c2_)
        st.image(out, caption=method, clamp=True, width="stretch")

    elif category == "Corner detection":
        method = st.selectbox("Method", ["Harris", "Shi-Tomasi"], key="flab_corner_method")
        st.caption("💡 Finds points where the image changes sharply in more than one direction — often "
                   "rib junctions or hardware edges, not a meaningful disease marker.")
        explain("harris" if method == "Harris" else "shi_tomasi")
        out = F.harris(filt) if method == "Harris" else F.shi_tomasi(filt)
        st.image(out[:, :, ::-1], caption=f"{method} corners (red)", width="stretch")

    elif category == "Morphological features":
        op = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing", "Morphological gradient"],
                           key="flab_morph_op")
        k = st.select_slider("Kernel size", [3, 5, 7, 9, 11], 5, key="flab_morph_kernel")
        binary = S.otsu(filt)
        out = {"Erosion": F.morph_erode, "Dilation": F.morph_dilate, "Opening": S.opening,
               "Closing": S.closing, "Morphological gradient": F.morph_gradient}[op](binary, k)
        explain({"Erosion": "morph_open", "Dilation": "morph_close", "Opening": "morph_open",
                 "Closing": "morph_close", "Morphological gradient": "morph_gradient"}[op])
        c1, c2 = st.columns(2)
        c1.image(binary, caption="Input binary (Otsu)", clamp=True, width="stretch")
        c2.image(out, caption=op, clamp=True, width="stretch")

    elif category == "Texture analysis":
        st.caption("💡 Describes how 'rough' or 'smooth' a region looks, numerically.")
        explain("lbp"); explain("glcm")
        p = st.slider("LBP neighbours (P)", 4, 16, 8, key="flab_lbp_p")
        r = st.slider("LBP radius (R)", 1, 3, 1, key="flab_lbp_r")
        lbp_img, hist = T.lbp(filt, p, r)
        c1, c2 = st.columns(2)
        c1.image(cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                  caption="LBP code image", clamp=True, width="stretch")
        c2.bar_chart(hist)
        feats = T.glcm_features(filt)
        a, b, c, d, e = st.columns(5)
        a.metric("Contrast", f"{feats['contrast']:.2f}"); b.metric("Homogeneity", f"{feats['homogeneity']:.3f}")
        c.metric("Energy", f"{feats['energy']:.3f}"); d.metric("Correlation", f"{feats['correlation']:.3f}")
        e.metric("Entropy", f"{feats['entropy']:.2f}")

    else:
        mask = ensure_lung_mask("Shape & contour analysis needs a lung mask.", "flab_run_seg_btn")
        if mask is not None:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                st.warning("No contours found in the current mask.")
            else:
                idx = st.selectbox("Contour", list(range(len(cnts))), format_func=lambda i: f"Region {i+1}",
                                    key="flab_contour_idx")
                s = A.contour_features(cnts[idx])
                st.json(s)
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vis, [cnts[idx]], -1, (0, 255, 0), 2)
                hull = cv2.convexHull(cnts[idx])
                cv2.drawContours(vis, [hull], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(cnts[idx])
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
                st.image(vis[:, :, ::-1], caption="Contour (green), convex hull (red), bounding box (yellow)", width="stretch")


def view_anatomy_regions():
    st.caption("Beyond the lungs: what PulmoMap can estimate, and what it honestly cannot, using classical methods.")
    tab1, tab2 = st.tabs(["Estimated Regions & Catalogue", "Advanced ROI Analysis"])

    with tab1:
        with st.expander("What can classical image processing actually identify? (click for the full catalogue)"):
            for name, info in ANAT.ANATOMY_CATALOG.items():
                icon = {"Classical estimate (geometry-based ROI, not a validated segmentation)": "🟢",
                        "Requires a trained model (not implemented)": "🔴",
                        "Not reliably identifiable with classical image processing alone": "🟡"}[info["status"]]
                st.markdown(f"{icon} **{name}** — {info['status']}  \n"
                            f"<span style='color:var(--muted);font-size:.85rem;'>{info['note']}</span>", unsafe_allow_html=True)

        mask = ensure_lung_mask("Region estimates (heart, mediastinum, diaphragm, etc.) are anchored on "
                                 "the lung mask's bounding boxes.", "anat_run_seg_btn")
        if mask is None:
            regions = {}
        else:
            regions = ANAT.estimate_regions(gray, mask)
            if not regions:
                st.warning("Two separate lung regions were not found in the current mask, so no region estimates could be anchored.")
            else:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for name, (x, y, w, h) in regions.items():
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (25, 184, 154), 2)
                    cv2.putText(vis, name.split(" (")[0], (x + 3, max(12, y + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (25, 184, 154), 1)
                st.image(vis[:, :, ::-1], caption="Estimated regions — geometry-based, NOT a validated segmentation", width="stretch")

    with tab2:
        st.caption("Pick a region preset (or draw a manual rectangle / pick a grid cell). Optionally compare two regions.")

        H, W = gray.shape
        presets = ["Complete image"]
        if mask is not None:
            lr = A.split_left_right(mask)
            if len(lr) == 2:
                presets += ["Right lung", "Left lung"]
        regions = ANAT.estimate_regions(gray, mask) if mask is not None else {}
        presets += list(regions.keys())
        presets += ["Upper lung zone", "Middle lung zone", "Lower lung zone", "Grid cell (3x3)", "Manual rectangle"]

        def resolve_roi(preset, key_prefix):
            if preset == "Complete image":
                return (0, 0, W, H)
            if preset in ("Right lung", "Left lung") and mask is not None:
                lr = A.split_left_right(mask)
                c = lr[0] if preset == "Right lung" else lr[1]
                return cv2.boundingRect(c)
            if preset in regions:
                return regions[preset]
            if preset in ("Upper lung zone", "Middle lung zone", "Lower lung zone") and mask is not None:
                ys = np.nonzero(mask)[0]
                if ys.size == 0:
                    return (0, 0, W, H)
                y0, y1 = ys.min(), ys.max()
                third = (y1 - y0) // 3
                idx = {"Upper lung zone": 0, "Middle lung zone": 1, "Lower lung zone": 2}[preset]
                y_start = y0 + idx * third
                y_end = y0 + (idx + 1) * third if idx < 2 else y1
                return (0, int(y_start), W, int(y_end - y_start))
            if preset == "Grid cell (3x3)":
                cells = ANAT.grid_cells((H, W), 3)
                cell = st.selectbox("Grid cell", list(cells.keys()), key=f"anatroi_{key_prefix}_cell")
                return cells[cell]
            if preset == "Manual rectangle":
                x = st.slider("x", 0, W - 10, W // 4, key=f"anatroi_{key_prefix}_x")
                y = st.slider("y", 0, H - 10, H // 4, key=f"anatroi_{key_prefix}_y")
                w = st.slider("width", 10, W - x, min(150, W - x), key=f"anatroi_{key_prefix}_w")
                h = st.slider("height", 10, H - y, min(150, H - y), key=f"anatroi_{key_prefix}_h")
                return (x, y, w, h)
            return (0, 0, W, H)

        def roi_full_stats(x, y, w, h):
            crop, _ = ROI.rect_roi(gray, x, y, w, h)
            if crop.size == 0:
                return None, None
            stats = ROI.roi_stats(crop)
            tfeat = T.glcm_features(crop) if min(crop.shape) >= 8 else {}
            edges = cv2.Canny(crop, 50, 150)
            stats["edge_density"] = float(np.count_nonzero(edges)) / edges.size if edges.size else 0.0
            return crop, {"stats": stats, "texture": tfeat}

        compare = st.checkbox("Compare two regions", key="anat_compare_two")
        cols = st.columns(2) if compare else [st.container()]
        labels = ["Region A", "Region B"] if compare else ["Region"]
        for col, label in zip(cols, labels):
            with col:
                st.markdown(f"**{label}**")
                preset = st.selectbox("Preset", presets, key=f"anatroi_preset_{label}")
                x, y, w, h = resolve_roi(preset, label)
                crop, result = roi_full_stats(x, y, w, h)
                if crop is None:
                    st.warning("Empty region.")
                    continue
                st.image(crop, caption=f"{preset} ({w}x{h}px)", clamp=True, width="stretch")
                s = result["stats"]
                a, b, c, d = st.columns(4)
                a.metric("Mean", f"{s['mean_intensity']:.1f}"); b.metric("Median", f"{s['median_intensity']:.1f}")
                c.metric("Std dev", f"{s['std_intensity']:.1f}"); d.metric("Edge density", f"{s['edge_density']*100:.1f}%")
                if result["texture"]:
                    t = result["texture"]
                    st.caption(f"GLCM — contrast {t['contrast']:.2f}, homogeneity {t['homogeneity']:.3f}, "
                               f"energy {t['energy']:.3f}, entropy {t['entropy']:.2f}")


def view_candidate_explorer():
    st.markdown(f'<div class="pm-warn">⚠️ {CAND.CANDIDATE_DISCLAIMER}</div>', unsafe_allow_html=True)
    st.caption("💡 Highlights regions whose brightness, contrast, texture or edges differ from their "
               "surroundings. This is a generic image-processing tool, not a disease detector — and "
               "zero candidates is a perfectly valid result.")

    method = st.selectbox("Detection method", CAND.METHODS, key="cand_method")
    default_sens = {"Local intensity deviation (z-score)": 2.0, "CLAHE local contrast": 2.0,
                     "Adaptive threshold (local darker/brighter)": 2.0,
                     "Local texture variation": 7.0, "Edge-density variation": 2.0}[method]

    c1, c2 = st.columns(2)
    sensitivity = c1.slider("Sensitivity", 0.5, 10.0, default_sens, 0.5, key=f"cand_sens_{method}",
                             help="How different a region must be from its surroundings to count. Higher = stricter (fewer candidates).")
    min_area = c2.slider("Minimum region size (px)", 20, 3000, 80, 10,
                          help="Regions smaller than this are treated as noise/artifacts and hidden.",
                          key="cand_min_area")
    show_overlay = st.checkbox("Show candidate overlay (boxes + ids)", value=True, key="cand_show_overlay")

    with st.expander("Advanced settings"):
        max_area = st.slider("Maximum region size (px)", 500, 60000, 20000, 500, key="cand_max_area")
        kernel_size = st.select_slider("Morphological kernel size", [3, 5, 7, 9, 11], 5, key="cand_kernel_size")
        contrast_sens = st.slider("Contrast strength (CLAHE clip / adaptive C)", 1.0, 20.0, 3.0, 0.5,
                                   key="cand_contrast_sens")
        show_heatmap = st.checkbox("Show local variation heatmap", value=False, key="cand_show_heatmap")
        restrict_to_lungs = st.checkbox("Restrict search to segmented lung area (if available)",
                                         key="cand_restrict_lungs")
        explain("candidate_method")
        explain("candidate_kernel")

    valid_mask = None
    if restrict_to_lungs:
        valid_mask = ensure_lung_mask("Restricting to the lung area needs a lung mask. Searching the "
                                       "whole image until then.", "cand_run_seg_btn")

    overlay_mask, candidates, dbg = CAND.extract_candidates(
        gray, method=method, min_area=min_area, max_area=max_area,
        intensity_sensitivity=sensitivity, texture_sensitivity=sensitivity,
        contrast_sensitivity=contrast_sens, kernel_size=kernel_size, valid_mask=valid_mask)

    st.markdown("**Legend:** ⬜ White = candidate pixels &nbsp;·&nbsp; ⬛ Black = background &nbsp;·&nbsp; "
                "🟧 Box = connected component &nbsp;·&nbsp; • = centroid &nbsp;·&nbsp; `#N A=area(px)` = id and size")

    panels = [("1. Original (grayscale)", gray), ("2. Candidate binary mask", overlay_mask)]
    if show_overlay:
        panels.append(("3. Candidate overlay", CAND.draw_candidates(gray, candidates, (32, 176, 255))[:, :, ::-1]))
    if show_heatmap:
        heatmap_fn = {"Local intensity deviation (z-score)": lambda g: ANAT.z_score_map(g, max(9, kernel_size * 3)),
                      "Edge-density variation": lambda g: ANAT.edge_density_heatmap(g, max(9, kernel_size * 3)),
                      "Local texture variation": lambda g: ANAT.texture_variation_map(g, max(9, kernel_size * 3)),
                      "CLAHE local contrast": lambda g: ANAT.local_contrast_map(g, max(9, kernel_size * 3)),
                      "Adaptive threshold (local darker/brighter)": lambda g: ANAT.local_contrast_map(g, max(9, kernel_size * 3))}[method]
        panels.append(("4. Local variation heatmap", heatmap_fn(gray)))

    cols = st.columns(len(panels))
    for col, (cap, im) in zip(cols, panels):
        col.image(im, caption=cap, clamp=True, width="stretch")

    st.markdown(f"**{len(candidates)} candidate region(s)** found using *{method}*.")
    if candidates:
        avg_area = sum(c["area_px"] for c in candidates) / len(candidates)
        if avg_area < 0.0015 * gray.size:
            st.info("Most detected regions are very small (tiny high-contrast spots such as vessel "
                    "crossings, rib edges, or noise) — consider raising 'Minimum region size' to focus "
                    "on larger, more meaningful regions.")
        st.dataframe(candidates, width="stretch")
    else:
        st.info("No candidates at these settings — a valid result, not an error.")

    with st.expander("Why this many candidates? (debug details)"):
        st.json(dbg)
        if len(candidates) == 0:
            if dbg["raw_mask_nonzero_px"] == 0:
                st.caption("The detection rule itself matched 0 pixels — try lowering the sensitivity slider.")
            elif dbg["connected_components_found"] == 0:
                st.caption("Matching pixels existed but cleanup removed them all — try a smaller morphological kernel.")
            else:
                st.caption(f"{dbg['connected_components_found']} component(s) were found, but none fell between "
                           f"the minimum ({min_area}px) and maximum ({max_area}px) size filters.")


def view_findings_explorer():
    st.markdown(f'<div class="pm-warn">⚠️ {FINDINGS_DISCLAIMER}</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Reference", "ROI Workspace"])

    with tab1:
        names = [f["name"] for f in FINDINGS]
        choice = st.selectbox("Finding", names, key="find_choice")
        f = next(x for x in FINDINGS if x["name"] == choice)
        st.subheader(f["name"])
        st.markdown(f"**Definition:** {f['definition']}")
        st.markdown(f"**Typical appearance:** {f['appearance']}")
        st.markdown(f"**Anatomical location:** {f['location']}")
        st.markdown(f"**Why it matters:** {f['importance']}")
        with st.expander("How might image processing help visualize this (experimental, not detection)?"):
            st.write(f["ip_relevance"])
            st.caption("Experimental — generic image-processing tools, not a trained detector for this finding.")

    with tab2:
        st.caption("Select a rectangular region to inspect intensity, texture, edges and shape locally.")
        H, W = gray.shape
        x = st.slider("ROI x (left edge)", 0, W - 10, W // 4, key="find_roi_x")
        y = st.slider("ROI y (top edge)", 0, H - 10, H // 4, key="find_roi_y")
        w = st.slider("ROI width", 10, W - x, min(150, W - x), key="find_roi_w")
        h = st.slider("ROI height", 10, H - y, min(150, H - y), key="find_roi_h")

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


def view_anatomical_analysis():
    mask = ensure_lung_mask("Anatomical geometry and symmetry are computed from a lung mask.", "anatstats_run_seg_btn")
    if mask is None:
        return
    stats = A.analyse(mask)
    if stats is None:
        st.warning("No analysis available: two separate lung regions were not detected in the current mask.")
        return

    st.caption("Pixel-based measurements only — not physical units unless the image is calibrated.")
    for name in [k for k in stats if k != "symmetry"]:
        v = stats[name]
        st.subheader(name)
        a, b, c, d = st.columns(4)
        a.metric("Area (px)", f"{v['area_px']:.0f}"); b.metric("Perimeter (px)", f"{v['perimeter_px']:.0f}")
        c.metric("Circularity", f"{v['circularity']:.2f}"); d.metric("Solidity", f"{v['solidity']:.2f}")
        st.caption(f"Bounding box {v['bbox_w']}x{v['bbox_h']} px · aspect ratio {v['aspect_ratio']:.2f} "
                    f"· centroid ({v['centroid'][0]:.0f}, {v['centroid'][1]:.0f})")
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
    st.caption("Differences here are descriptive geometry only and must not be interpreted as evidence of disease.")
    st.session_state["anatomy_stats"] = stats


def view_comparison_dashboard():
    comparison = st.selectbox("Comparison", [
        "Original vs. Grayscale", "Original vs. Normalized", "Original vs. CLAHE",
        "Filtering methods (Gaussian / Average / Median)", "Edge detection algorithms",
        "Segmentation methods (Otsu / Adaptive / K-means)", "Right vs. Left lung"],
        key="cmp_comparison")

    if comparison == "Original vs. Grayscale":
        imgs, titles = [orig_rgb, gray], ["Original", "Grayscale"]
    elif comparison == "Original vs. Normalized":
        imgs, titles = [gray, norm], ["Grayscale", "Normalized"]
    elif comparison == "Original vs. CLAHE":
        imgs, titles = [gray, P.clahe(norm, 2.0, 8)], ["Grayscale", "CLAHE"]
    elif comparison == "Filtering methods (Gaussian / Average / Median)":
        k = st.select_slider("Kernel size", [3, 5, 7, 9], 5, key="cmp_kernel")
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
        mask = ensure_lung_mask("Comparing right vs. left lung needs a lung mask.", "cmp_run_seg_btn")
        if mask is None:
            imgs, titles = [], []
        else:
            lr = A.split_left_right(mask)
            if len(lr) < 2:
                st.warning("Two lungs were not detected in the current mask.")
                imgs, titles = [], []
            else:
                rmask = np.zeros_like(mask); cv2.drawContours(rmask, [lr[0]], -1, 255, cv2.FILLED)
                lmask = np.zeros_like(mask); cv2.drawContours(lmask, [lr[1]], -1, 255, cv2.FILLED)
                imgs, titles = [rmask, lmask], ["Right lung mask", "Left lung mask"]

    if imgs:
        fig = R.comparison_figure(imgs, titles, cols=min(4, len(imgs)))
        st.pyplot(fig)
        st.download_button("Download comparison figure", R.fig_to_png(fig), "comparison.png", "image/png",
                            key="cmp_dl_fig")


def view_detailed_report():
    anatomy = st.session_state.get("anatomy_stats")
    if anatomy is None and st.session_state.get("seg_mask") is not None:
        # A mask already exists (computed in this or another workspace) but Anatomical Analysis
        # hasn't been visited yet this session — derive the same stats here instead of making
        # the user navigate over just to populate the report.
        anatomy = A.analyse(st.session_state["seg_mask"])
        if anatomy is not None:
            st.session_state["anatomy_stats"] = anatomy
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
    c1.download_button("Download report (.txt)", txt, "pulmomap_report.txt", key="report_dl_txt")

    pdf_images = []
    mask = st.session_state.get("seg_mask")
    if mask is not None:
        fig = R.comparison_figure([gray, mask, A.overlay(gray, mask)[:, :, ::-1]],
                                   ["Input", "Lung mask", "Overlay"], cols=3)
        pdf_images.append(("Segmentation visualization", R.fig_to_png(fig)))
    try:
        pdf_bytes = EXP.build_pdf(sections, pdf_images, title="PulmoMap Detailed Report")
        c2.download_button("Download report (.pdf)", pdf_bytes, "pulmomap_report.pdf", "application/pdf",
                            key="report_dl_pdf")
    except Exception as e:
        st.warning(f"PDF export unavailable: {e}")

    if anatomy is None:
        ensure_lung_mask("Run lung segmentation to include anatomical measurements in this report.",
                         "report_run_seg_btn")


# ===========================================================================
# ROUTING: normal 5-workspace/tab navigation, or Presentation Mode
# ===========================================================================

def _view_results_and_report():
    st.subheader("Anatomical measurements")
    view_anatomical_analysis()
    st.divider()
    st.subheader("Detailed report")
    view_detailed_report()


PRESENTATION_STEPS = [
    ("Introduction", None),
    ("Upload X-ray", None),
    ("Preprocessing", view_preprocessing),
    ("Lung Segmentation", view_segmentation),
    ("Whole-Image Feature Extraction", view_whole_image_features),
    ("Edge Detection", lambda: view_feature_lab(default_category="Edge detection")),
    ("Texture Analysis", lambda: view_feature_lab(default_category="Texture analysis")),
    ("Anatomical Region Analysis", view_anatomy_regions),
    ("Candidate Region Exploration", view_candidate_explorer),
    ("Results and Report", _view_results_and_report),
]

if presentation_mode:
    step_idx = st.session_state.get("pstep", 0)
    title, fn = PRESENTATION_STEPS[step_idx]
    page_header(f"Step {step_idx + 1} of {len(PRESENTATION_STEPS)}", title)
    nav1, nav2, nav3 = st.columns([1, 4, 1])
    if nav1.button("⬅ Back", disabled=step_idx == 0, width="stretch", key="pm_back_btn"):
        st.session_state["pstep"] = max(0, step_idx - 1)
        st.rerun()
    nav2.progress((step_idx + 1) / len(PRESENTATION_STEPS))
    if nav3.button("Next ➡", disabled=step_idx == len(PRESENTATION_STEPS) - 1, width="stretch", key="pm_next_btn"):
        st.session_state["pstep"] = min(len(PRESENTATION_STEPS) - 1, step_idx + 1)
        st.rerun()
    st.divider()

    if step_idx == 0:
        st.markdown("""
**PulmoMap** is a chest X-ray image-processing and anatomical-visualization platform built entirely on
classical (non-AI) computer vision — no trained model, no disease detection.

This walkthrough demonstrates, in order: preprocessing, lung segmentation, whole-image feature
extraction, edge detection, texture analysis, anatomical region analysis, candidate region
exploration, and results & report generation.

Click **Next** to begin.
""")
    elif step_idx == 1:
        if img is not None:
            st.success(f"Image loaded: {src_name}")
            st.image(gray, caption="Grayscale preview", width=420)
        else:
            st.info("Upload an image, or tick 'Use the bundled sample image', in the sidebar — then click Next.")
    else:
        if img is None:
            st.warning("No image loaded yet — go back to Step 2 and upload one (or use the sample image).")
        else:
            fn()

else:
    if workspace == "Getting Started":
        view_getting_started()

    elif workspace == "Image Processing":
        t1, t2 = st.tabs(["Preprocessing", "Lung Segmentation"])
        with t1: view_preprocessing()
        with t2: view_segmentation()

    elif workspace == "Anatomical Analysis":
        t1, t2, t3 = st.tabs(["Lung Geometry & Symmetry", "Anatomical Regions (Beyond Lungs)", "Radiographic Findings Reference"])
        with t1: view_anatomical_analysis()
        with t2: view_anatomy_regions()
        with t3: view_findings_explorer()

    elif workspace == "Feature Extraction":
        t1, t2, t3 = st.tabs(["Whole-Image Features", "Feature Extraction Lab", "Candidate Region Explorer"])
        with t1: view_whole_image_features()
        with t2: view_feature_lab()
        with t3: view_candidate_explorer()

    elif workspace == "Results & Reports":
        t1, t2 = st.tabs(["Comparison Dashboard", "Detailed Report"])
        with t1: view_comparison_dashboard()
        with t2: view_detailed_report()
