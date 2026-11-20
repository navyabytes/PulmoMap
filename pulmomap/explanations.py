"""Beginner + technical explanation helpers, used consistently across the app.

Every parameter/algorithm the UI exposes should have an entry here so
`explain_from_key()` can render a consistent expander: simple explanation,
technical explanation, and what happens when a value goes up/down.
"""
import streamlit as st

EXPLANATIONS = {
    "grayscale": dict(
        title="Grayscale conversion",
        simple="Turns the colour image into shades of grey (black to white) so only brightness is kept.",
        technical="Converts a 3-channel BGR image to a single-channel intensity image using a weighted "
                  "sum of the R, G and B channels (cv2.cvtColor, COLOR_BGR2GRAY).",
        note="Chest X-rays are captured as grayscale radiation-intensity data to begin with; any colour "
             "seen in a photo of a film is just a display artifact, so analysis is always done in grayscale.",
    ),
    "resize": dict(
        title="Resizing",
        simple="Shrinks or pads the image to a fixed size so every image is processed the same way.",
        technical="Resizes while preserving aspect ratio (area interpolation), then pads the shorter side "
                  "with zeros so the result is exactly 512x512.",
        up="A larger target size keeps more detail but is slower to process.",
        down="A smaller target size is faster but can lose fine detail.",
    ),
    "normalize": dict(
        title="Min-max normalization",
        simple="Stretches brightness so the darkest pixel becomes pure black and the brightest becomes pure white.",
        technical="Linear rescale: new = (old - min) / (max - min) * 255. Improves consistency across "
                  "images captured with different exposure settings.",
    ),
    "histeq": dict(
        title="Histogram equalization",
        simple="Spreads out brightness values so the image uses the full range from dark to light, making "
               "details easier to see.",
        technical="Computes the cumulative distribution function of pixel intensities and remaps values so "
                  "the output histogram is approximately uniform.",
        up="N/A — equalization is global and has no strength parameter.",
        down="Can over-amplify contrast uniformly, sometimes making already-bright areas look washed out.",
    ),
    "clahe": dict(
        title="CLAHE (Contrast Limited Adaptive Histogram Equalization)",
        simple="Like histogram equalization, but applied in small local patches so contrast improves "
               "locally without over-brightening the whole image.",
        technical="Divides the image into tiles, equalizes each tile's histogram, clips bins above "
                  "`clip_limit` before equalizing (to limit amplification), then blends tile edges.",
        up="Higher clip limit → stronger local contrast enhancement, but can amplify noise and graininess.",
        down="Lower clip limit → gentler, more restrained enhancement with less noise.",
    ),
    "clahe_tile": dict(
        title="CLAHE tile grid size",
        simple="Controls how small the local patches are that get their own contrast boost.",
        technical="Sets the (rows, cols) grid the image is divided into before per-tile histogram "
                  "equalization; each tile is processed independently and blended at the seams.",
        up="More tiles (smaller patches) → finer local detail enhancement, more sensitive to local noise.",
        down="Fewer tiles (larger patches) → smoother, closer to plain histogram equalization.",
    ),
    "gaussian": dict(
        title="Gaussian filtering",
        simple="Blurs the image smoothly, giving more weight to pixels near the center of the blur region.",
        technical="Convolves the image with a 2D Gaussian kernel of the given size and sigma, suppressing "
                  "high-frequency noise while preserving edges better than a plain average filter.",
        up="Larger kernel → stronger, smoother blur; removes more noise but also more fine detail.",
        down="Smaller kernel → light blur; keeps detail but removes less noise.",
        best_for="General sensor/Gaussian-type noise.",
    ),
    "average": dict(
        title="Average (box) filtering",
        simple="Replaces each pixel with the plain average of its neighbours.",
        technical="Convolves with a uniform kernel of all-equal weights (cv2.blur).",
        up="Larger kernel → heavier blur, can smear edges noticeably.",
        down="Smaller kernel → mild smoothing.",
        best_for="Simple uniform noise; tends to blur edges more than Gaussian for the same strength.",
    ),
    "median": dict(
        title="Median filtering",
        simple="Replaces each pixel with the middle value of its neighbourhood, which removes stray bright "
               "or dark specks without blurring edges much.",
        technical="Non-linear filter: sorts pixel values in the kernel window and outputs the median.",
        up="Larger kernel → removes larger specks of noise but can erode thin structures.",
        down="Smaller kernel → preserves thin structures but removes only small specks.",
        best_for="Salt-and-pepper (impulse) noise — specifically what it's designed for, unlike Gaussian/average.",
    ),
    "canny_low": dict(
        title="Canny low threshold",
        simple="The minimum edge strength needed for a pixel to be considered a *possible* edge.",
        technical="In hysteresis thresholding, gradient magnitudes above this value are considered weak "
                  "edge candidates, kept only if connected to a strong edge.",
        up="Higher → fewer weak edges get linked in, so some real but faint edges may be dropped.",
        down="Lower → more candidate edges, which can introduce noise as false edges.",
    ),
    "canny_high": dict(
        title="Canny high threshold",
        simple="The edge strength needed for a pixel to be accepted immediately as a strong, definite edge.",
        technical="Gradient magnitudes above this value are always kept as strong edges; they seed the "
                  "edge-linking (hysteresis) process.",
        up="Higher → only very strong edges are accepted directly, so fewer total edges are detected.",
        down="Lower → more edges accepted directly, which can introduce noise.",
        note="Canny requires low < high; the app enforces this automatically.",
    ),
    "otsu": dict(
        title="Otsu thresholding",
        simple="Automatically picks a single brightness cutoff that best separates the image into two "
               "groups: darker and lighter regions.",
        technical="Finds the threshold that minimizes intra-class intensity variance (equivalently "
                  "maximizes between-class variance) over the image histogram.",
    ),
    "adaptive": dict(
        title="Adaptive thresholding",
        simple="Instead of one cutoff for the whole image, calculates a *local* cutoff for each region, "
               "which helps when brightness varies across the image.",
        technical="Thresholds each pixel against the Gaussian-weighted mean of its local neighbourhood "
                  "(block size), minus a constant C.",
        up="Larger block size → cutoff reflects broader context, smoother regions.",
        down="Smaller block size → more locally sensitive, can be noisier.",
    ),
    "kmeans": dict(
        title="K-means segmentation",
        simple="Groups pixels into clusters of similar brightness, without needing a fixed threshold.",
        technical="Runs K-means clustering (k clusters) on pixel intensities; the darkest cluster is taken "
                  "as the lung candidate region.",
        note="Unsupervised and intensity-only — it has no anatomical knowledge, so it can fail on unusual "
             "exposures or pathology that changes lung brightness.",
    ),
    "morph_open": dict(
        title="Morphological opening",
        simple="Erosion followed by dilation — removes small bright specks/noise while keeping the overall "
               "shape of larger regions.",
        technical="Erosion shrinks foreground regions (removing small protrusions), dilation then regrows "
                  "the remaining regions back to roughly their original size.",
        up="Larger kernel → removes larger specks/protrusions, can also erode thin real structures.",
        down="Smaller kernel → gentler cleanup.",
    ),
    "morph_close": dict(
        title="Morphological closing",
        simple="Dilation followed by erosion — fills small holes/gaps inside regions while keeping overall shape.",
        technical="Dilation grows foreground regions (filling small gaps), erosion then shrinks back to "
                  "roughly the original size.",
        up="Larger kernel → fills bigger holes/gaps.",
        down="Smaller kernel → only fills very small gaps.",
    ),
    "morph_gradient": dict(
        title="Morphological gradient",
        simple="Highlights the outline of shapes by taking the difference between dilation and erosion.",
        technical="gradient = dilate(img) - erode(img); pixels that change under both operations mark the boundary.",
    ),
    "harris": dict(
        title="Harris corner detection",
        simple="Finds points where the image changes sharply in more than one direction at once — 'corners'.",
        technical="Uses the local structure tensor (second-moment matrix) of image gradients; a response "
                  "function based on its eigenvalues flags corner-like points.",
        note="On a chest X-ray, 'corners' usually correspond to rib junctions, scapula edges or hardware — "
             "not meaningful disease markers by themselves.",
    ),
    "shi_tomasi": dict(
        title="Shi-Tomasi corner detection ('Good Features to Track')",
        simple="A refinement of Harris corner detection that ranks corners by how reliable they are to track.",
        technical="Uses the minimum eigenvalue of the structure tensor directly as the corner score, rather "
                  "than Harris's combined response function.",
    ),
    "lbp": dict(
        title="Local Binary Patterns (LBP)",
        simple="Describes local texture by comparing each pixel to its neighbours (brighter or darker) and "
               "encoding the pattern as a number.",
        technical="For each pixel, thresholds `P` neighbours at radius `R` against the center value, forming "
                  "a binary pattern converted to a decimal label (uniform LBP reduces rotation sensitivity).",
        note="Useful for describing textural regularity/irregularity of lung fields, not for diagnosis.",
    ),
    "glcm": dict(
        title="Gray-Level Co-occurrence Matrix (GLCM)",
        simple="Counts how often pairs of pixel brightness values occur next to each other, then summarizes "
               "that into texture scores like contrast and smoothness.",
        technical="Builds a matrix of co-occurring intensity pairs at a given distance/angle, then computes "
                  "Haralick features (contrast, homogeneity, energy, correlation) from it.",
    ),
    "candidate_method": dict(
        title="Candidate-region detection method",
        simple="Picks which generic rule is used to flag a patch of the image as locally unusual.",
        technical="'Local intensity deviation' flags pixels far from their local neighbourhood "
                  "mean in standard-deviation units (a local z-score). 'CLAHE local contrast' "
                  "flags where adaptive histogram equalization changes the image the most. "
                  "'Adaptive threshold' flags regions separated from their local background by "
                  "a Gaussian-weighted local cutoff.",
        note="These are generic statistical/contrast rules, not a trained detector — results "
             "require expert interpretation and are never a diagnosis.",
    ),
    "candidate_min_area": dict(
        title="Minimum candidate area",
        simple="The smallest patch size (in pixels) that counts as a candidate region.",
        technical="Connected components smaller than this area (in px) after cleanup are discarded "
                  "as noise before being reported as candidates.",
        up="Fewer, larger candidates are kept; small specks are filtered out.",
        down="More, smaller candidates are kept; more noise/specks may appear.",
    ),
    "candidate_max_area": dict(
        title="Maximum candidate area",
        simple="The largest patch size (in pixels) that still counts as a single candidate region.",
        technical="Connected components larger than this area are discarded, since very large "
                  "regions usually represent broad anatomy or lighting rather than a localized "
                  "feature of interest.",
        up="Allows larger regions to be reported as single candidates.",
        down="Splits/excludes large regions, keeping the focus on smaller localized patches.",
    ),
    "intensity_sensitivity": dict(
        title="Intensity / contrast sensitivity",
        simple="Controls how different a patch has to be from its surroundings before it's flagged.",
        technical="Threshold applied to the chosen method's local deviation/contrast measure "
                  "(e.g. a z-score cutoff, or an absolute difference threshold after CLAHE).",
        up="Higher sensitivity threshold → stricter, fewer candidates (only strong local "
           "differences are flagged).",
        down="Lower threshold → more candidates, including subtler or noisier differences.",
    ),
    "candidate_kernel": dict(
        title="Morphological kernel size (candidate cleanup)",
        simple="Controls how aggressively small/noisy candidate blobs are cleaned up before reporting.",
        technical="Sets the structuring-element size used for opening (remove small specks) and "
                  "closing (fill small gaps) on the raw candidate mask.",
        up="Larger kernel → smoother, more consolidated candidate shapes; can merge nearby blobs.",
        down="Smaller kernel → finer-grained candidates; more likely to include small noise.",
    ),
}


def explain(key, extra=""):
    """Render a consistent explanation expander from the EXPLANATIONS registry."""
    e = EXPLANATIONS.get(key)
    if not e:
        return
    with st.expander(f"What does '{e['title']}' mean?"):
        st.markdown(f"**Simple explanation:** {e['simple']}")
        if e.get("technical"):
            st.markdown(f"**Technical explanation:** {e['technical']}")
        if e.get("up"):
            st.markdown(f"**Increasing it:** {e['up']}")
        if e.get("down"):
            st.markdown(f"**Decreasing it:** {e['down']}")
        if e.get("best_for"):
            st.markdown(f"**Best for:** {e['best_for']}")
        if e.get("note"):
            st.caption(e["note"])
        if extra:
            st.markdown(extra)


DISCLAIMER = (
    "PulmoMap is an educational image-processing and anatomical visualization tool. "
    "It is **not** a medical diagnostic system and must not be used to make clinical decisions. "
    "Classical segmentation masks are not clinically validated. No disease-detection model is included — "
    "this app does not and cannot identify pneumonia, cancer, TB or any other condition."
)
