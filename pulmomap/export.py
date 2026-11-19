"""PDF export for the detailed report, built with fpdf2 (pure-Python, lightweight)."""
import io
from fpdf import FPDF


def build_pdf(sections, images=None, title="PulmoMap Report"):
    """sections: list of (heading, [lines]) tuples. images: list of (caption, png_bytes)."""
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(pdf.epw, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    for heading, lines in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(pdf.epw, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in lines:
            pdf.multi_cell(pdf.epw, 5.5, line)
        pdf.ln(2)

    for caption, png_bytes in images or []:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, caption, new_x="LMARGIN", new_y="NEXT")
        pdf.image(io.BytesIO(png_bytes), w=170)

    return bytes(pdf.output())
