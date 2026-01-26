import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle
)
from reportlab.lib import colors


def calculate_ai_percentage(results):
    ai_count = sum(1 for r in results if r["prediction"] == "AI Generated")
    total = len(results)
    return round((ai_count / total) * 100, 2)

def generate_pie_chart(ai_percentage):
    human_percentage = 100 - ai_percentage

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [ai_percentage, human_percentage],
        labels=["AI Generated", "Real"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff6b6b", "#4ecdc4"]
    )
    ax.set_title("AI vs Real Image Distribution")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)

    return buffer

def tensor_to_rl_image(tensor, width=2.5 * inch):
    """
    Converts TF tensor (224,224,3) → ReportLab Image
    """
    img_np = (tensor.numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    return RLImage(buffer, width=width, height=width)

def create_pdf_report(results, output_path="report.pdf"):
    """
    results: output from predict_batch
    """

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    elements = []

    # ===== TITLE =====
    elements.append(
        Paragraph("<b>AI Image Detection Report</b>", styles["Title"])
    )
    elements.append(Spacer(1, 12))

    # ===== AI PERCENTAGE =====
    ai_percentage = calculate_ai_percentage(results)

    elements.append(
        Paragraph(
            f"<b>Total AI Generated Images:</b> {ai_percentage}%",
            styles["Heading2"]
        )
    )
    elements.append(Spacer(1, 12))

    # ===== PIE CHART =====
    pie_chart = generate_pie_chart(ai_percentage)
    elements.append(RLImage(pie_chart, width=3 * inch, height=3 * inch))
    elements.append(Spacer(1, 24))

    # ===== IMAGE RESULTS =====
    elements.append(
        Paragraph("<b>Individual Image Analysis</b>", styles["Heading2"])
    )
    elements.append(Spacer(1, 12))

    for idx, r in enumerate(results, start=1):
        img = tensor_to_rl_image(r["image_tensor"])

        info = Paragraph(
            f"""
            <b>Image {idx}</b><br/>
            Prediction: <b>{r["prediction"]}</b><br/>
            Confidence: <b>{r["confidence"]}%</b>
            """,
            styles["Normal"]
        )

        table = Table(
            [[img, info]],
            colWidths=[3 * inch, 3 * inch]
        )

        table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ])
        )

        elements.append(table)
        elements.append(Spacer(1, 18))

    # ===== DISCLAIMER =====
    elements.append(Spacer(1, 24))
    elements.append(
        Paragraph(
            "<i>Disclaimer: These images were analyzed using an AI-based system. "
            "AI predictions are probabilistic and may contain inaccuracies. "
            "This report should not be considered as definitive proof.</i>",
            styles["Italic"]
        )
    )

    # ===== BUILD PDF =====
    doc.build(elements)
