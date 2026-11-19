"""Educational reference content for common radiographic patterns.

IMPORTANT: This module is purely educational text. PulmoMap has no trained
classifier and makes no attempt to detect any of these findings in an
uploaded image. Nothing here should be read as a detection or diagnosis.
"""

FINDINGS = [
    dict(
        name="Consolidation",
        definition="Alveolar air spaces fill with fluid, pus or other material instead of air.",
        appearance="A dense, fairly uniform area of increased whiteness (opacity) that can obscure "
                    "underlying vessels; may contain 'air bronchograms' (dark branching lines).",
        location="Can occur in any lobe; often lobar or segmental in distribution.",
        importance="One of the most common findings radiologists look for; can have infectious, "
                    "inflammatory or other causes.",
        ip_relevance="Thresholding/CLAHE can make bright regions stand out visually; texture features "
                      "(GLCM) can quantify how uniform a bright region is.",
    ),
    dict(
        name="Interstitial pattern",
        definition="Abnormal markings in the connective-tissue framework of the lung (the 'interstitium') "
                    "rather than the air spaces themselves.",
        appearance="Fine linear, net-like or hazy increased markings spread through the lung fields.",
        location="Often diffuse or bilateral, sometimes with a basal or apical predominance.",
        importance="Pattern recognition can narrow the range of underlying processes being considered.",
        ip_relevance="LBP/GLCM texture descriptors can numerically characterize 'fine vs coarse' or "
                      "'regular vs irregular' markings.",
    ),
    dict(
        name="Pulmonary nodule",
        definition="A small (typically <3cm), roughly round area of increased density within the lung.",
        appearance="A discrete round or oval white spot, well- or poorly-defined, surrounded by normal lung.",
        location="Can appear anywhere in the lung fields.",
        importance="Size, shape and growth over time are important factors in characterizing nodules.",
        ip_relevance="Blob-like bright-region candidates can be extracted with thresholding + contour "
                      "shape features (circularity, area) as unvalidated visual candidates only.",
    ),
    dict(
        name="Pulmonary mass",
        definition="Similar to a nodule but larger (typically >3cm).",
        appearance="A larger, often irregular area of increased density.",
        location="Anywhere in the lung fields.",
        importance="Larger lesions are tracked for size, margins and any associated findings.",
        ip_relevance="Same contour/shape-based candidate extraction as nodules, at a larger size threshold.",
    ),
    dict(
        name="Atelectasis",
        definition="Collapse or incomplete expansion of part of a lung.",
        appearance="Increased density with volume loss — displacement of fissures, ribs crowded together, "
                    "or shift of the mediastinum/diaphragm toward the affected side.",
        location="Can be segmental, lobar, or affect an entire lung.",
        importance="Volume loss is the distinguishing clue versus consolidation (which doesn't lose volume).",
        ip_relevance="Shape/contour analysis of lung boundary (area, aspect ratio) can describe volume "
                      "asymmetry, which is what PulmoMap's symmetry index is designed to surface.",
    ),
    dict(
        name="Pleural effusion",
        definition="Fluid accumulation in the pleural space surrounding the lung.",
        appearance="A dense opacity, often with a characteristic curved upper border (meniscus), typically "
                    "at the base of the lung and following gravity.",
        location="Usually at the lung bases; larger effusions can blunt or obscure the costophrenic angle.",
        importance="Can indicate a wide range of underlying conditions and may require drainage.",
        ip_relevance="Edge/contour analysis can highlight the characteristic meniscus-shaped boundary.",
    ),
    dict(
        name="Pneumothorax",
        definition="Air trapped in the pleural space, outside the lung itself.",
        appearance="A region with no lung markings (vessels), bounded by a thin visible pleural line; "
                    "the lung edge appears displaced away from the chest wall.",
        location="Typically apical (toward the top) when the patient is upright, but can be anywhere.",
        importance="Can be a medical emergency, particularly if under tension.",
        ip_relevance="Edge detection can highlight the thin pleural line boundary for visual exploration.",
    ),
    dict(
        name="Cavitation",
        definition="A gas-filled space within an area of lung disease (e.g. within a nodule, mass or "
                    "consolidation), usually from tissue necrosis.",
        appearance="A dark, air-filled area with a surrounding wall of varying thickness, inside an "
                    "otherwise denser region.",
        location="Can occur within nodules, masses or areas of consolidation.",
        importance="Wall thickness and contents are often examined.",
        ip_relevance="Contour/hole analysis (using morphological closing to compare with the filled region) "
                      "can visually outline internal dark regions within a bright region of interest.",
    ),
    dict(
        name="Fibrotic pattern",
        definition="Scarring of lung tissue, generally from chronic injury or disease.",
        appearance="Coarse linear/reticular markings, often with architectural distortion and volume loss; "
                    "can include 'honeycombing' (small cystic spaces) in advanced cases.",
        location="Pattern and distribution (e.g. basal vs apical) vary with the underlying cause.",
        importance="Distribution and pattern help narrow the differential.",
        ip_relevance="Texture features (GLCM contrast/homogeneity, LBP) can numerically describe coarse, "
                      "irregular textures in a selected region.",
    ),
    dict(
        name="Reticular pattern",
        definition="A net-like pattern of fine or coarse linear opacities.",
        appearance="Crisscrossing line-like markings forming a mesh-like appearance across part of the lung.",
        location="Can be diffuse or localized, often bilateral.",
        importance="One of several interstitial pattern subtypes used in pattern-based assessment.",
        ip_relevance="Edge-detection density and LBP uniformity scores can quantify 'how net-like' a "
                      "selected region's texture is.",
    ),
]

FINDINGS_DISCLAIMER = (
    "The descriptions below are general educational references about radiographic patterns. "
    "PulmoMap does not detect, flag or screen for any of these findings in uploaded images — there is no "
    "trained classification model in this app. Any bright-region or shape 'candidate' extraction offered "
    "elsewhere in the app is an unvalidated image-processing exercise, not a detection of disease."
)
