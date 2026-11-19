import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def comparison_figure(images, titles, cols=4):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t)
    fig.tight_layout()
    return fig


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def detailed_sections(info):
    """Builds a list of (heading, [lines]) sections for the Detailed Report page.
    `info` is a dict assembled by app.py from whatever the user actually ran."""
    sections = [("Project", ["PulmoMap - Chest X-Ray Image Processing & Anatomical Visualization",
                             "IPCV Lab project, JIIT."])]

    if info.get("input"):
        sections.append(("Input image", [f"{k}: {v}" for k, v in info["input"].items()]))

    if info.get("preprocessing"):
        sections.append(("Preprocessing operations performed", info["preprocessing"]))

    if info.get("segmentation"):
        sections.append(("Segmentation", info["segmentation"]))

    if info.get("features"):
        sections.append(("Extracted features", info["features"]))

    if info.get("anatomy"):
        lines = []
        for k, v in info["anatomy"].items():
            if k == "symmetry":
                continue
            lines += [k,
                      f"  area (px): {v['area_px']:.0f}",
                      f"  perimeter (px): {v['perimeter_px']:.1f}",
                      f"  aspect ratio: {v['aspect_ratio']:.2f}",
                      f"  circularity: {v['circularity']:.2f}",
                      f"  solidity: {v['solidity']:.2f}",
                      "  zones: " + ", ".join(f"{a} {b:.1f}%" for a, b in v["zones_%"].items())]
        s = info["anatomy"].get("symmetry")
        if s:
            lines += ["Symmetry",
                      f"  area ratio R/L: {s['area_ratio_R/L']:.2f}",
                      f"  symmetry index: {s['symmetry_index']:.3f}",
                      f"  mirror IoU: {s['mirror_iou']:.3f}"]
        sections.append(("Anatomical measurements (pixel-based, not physical units)", lines))

    if info.get("roi"):
        sections.append(("ROI analysis", info["roi"]))

    sections.append(("Limitations", [
        "Pixel-based measurements are not physical/clinical measurements unless the image is calibrated.",
        "Classical segmentation (thresholding/morphology/K-means) can fail on atypical exposures, rotated "
        "films, hardware/tubes, or pathology that changes lung brightness - always visually verify the mask.",
        "No disease-detection or diagnostic model is included in this app.",
    ]))
    sections.append(("Disclaimer", [
        "PulmoMap is an educational image-processing and anatomical visualization tool. It is not a medical "
        "diagnostic system and must not be used to make clinical decisions.",
    ]))
    return sections


def sections_to_text(sections):
    lines = []
    for heading, body in sections:
        lines.append(heading)
        lines.append("-" * len(heading))
        lines += list(body)
        lines.append("")
    return "\n".join(lines)


def text_report(stats, name="image"):
    L = ["PulmoMap - Anatomical Analysis Report", f"Image: {name}", ""]
    for k, v in stats.items():
        if k == "symmetry":
            continue
        L += [k,
              f"  area (px): {v['area_px']:.0f}",
              f"  perimeter (px): {v['perimeter_px']:.1f}",
              f"  aspect ratio: {v['aspect_ratio']:.2f}",
              f"  circularity: {v['circularity']:.2f}",
              f"  solidity: {v['solidity']:.2f}",
              "  zones: " + ", ".join(f"{a} {b:.1f}%" for a, b in v["zones_%"].items()), ""]
    s = stats["symmetry"]
    L += ["Symmetry",
          f"  area ratio R/L: {s['area_ratio_R/L']:.2f}",
          f"  symmetry index: {s['symmetry_index']:.3f}",
          f"  mirror IoU: {s['mirror_iou']:.3f}", "",
          "Note: anatomical image analysis only; not a clinical diagnosis."]
    return "\n".join(L)
