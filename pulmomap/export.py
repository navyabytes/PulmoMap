"""PDF export for the detailed report, built with fpdf2 (pure-Python, lightweight)."""
import io
from fpdf import FPDF

# fpdf2's built-in Helvetica font only supports Latin-1. The rest of the app is written with
# plain-English Unicode punctuation (em-dashes, smart quotes, bullets) for readability in the
# browser UI, so PDF export must transliterate that punctuation to ASCII rather than crash.
_PDF_SAFE = {
    "—": "-", "–": "-",       # em dash, en dash
    "‘": "'", "’": "'",       # smart single quotes
    "“": '"', "”": '"',       # smart double quotes
    "•": "-", "…": "...",     # bullet, ellipsis
}


def _pdf_safe(text):
    for uni, ascii_ in _PDF_SAFE.items():
        text = text.replace(uni, ascii_)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(sections, images=None, title="PulmoMap Report"):
    """sections: list of (heading, [lines]) tuples. images: list of (caption, png_bytes)."""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(pdf.epw, 10, _pdf_safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    for heading, lines in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(pdf.epw, 8, _pdf_safe(heading), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            pdf.multi_cell(pdf.epw, 5.5, _pdf_safe(line))
        pdf.ln(2)

    for caption, png_bytes in images or []:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _pdf_safe(caption), new_x="LMARGIN", new_y="NEXT")
        pdf.image(io.BytesIO(png_bytes), w=170)

    return bytes(pdf.output())
