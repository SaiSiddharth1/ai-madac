"""
Report Export Service.
Generates downloadable PDF and JSON representations of multi-agent analysis reports.
"""

import json
import logging
import os
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors

from app.config import settings

logger = logging.getLogger(__name__)


def generate_pdf_report(report_data: dict, question: str, dataset_name: str) -> BytesIO:
    """
    Generate a styled PDF report for a query result.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#4f46e5"),
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#1e1e42"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        leftIndent=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("AI MADAC — Analysis Report", title_style))
    story.append(Paragraph(f"Dataset: {dataset_name} | Question: &quot;{question}&quot;", subtitle_style))
    story.append(Spacer(1, 10))

    # Executive Summary / Answer
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(report_data.get("answer", "No answer summary available."), body_style))

    # Key Insights
    insights = report_data.get("insights", [])
    if insights:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Key Insights & Findings", heading_style))
        for ins in insights:
            story.append(Paragraph(f"• {ins}", bullet_style))

    # Charts are exported as PNG files by the visualization agent so the PDF
    # tells the same visual story as the interactive report.
    chart_paths = report_data.get("chart_paths", []) or []
    rendered_charts = 0
    for chart_path in chart_paths[:2]:
        filename = os.path.basename(chart_path)
        local_path = os.path.join(settings.CHARTS_DIR, filename)
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")) or not os.path.isfile(local_path):
            continue
        if rendered_charts == 0:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Visual Analysis", heading_style))
        story.append(Image(local_path, width=510, height=287, kind="proportional"))
        story.append(Spacer(1, 8))
        rendered_charts += 1

    # Agents Invoked
    agents = report_data.get("agents_used", [])
    if agents:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Autonomous Multi-Agent Execution Path", heading_style))
        agent_names = ", ".join([a.replace("_", " ").title() for a in agents])
        story.append(Paragraph(f"Executed Agents: {agent_names}", body_style))

    # Data Table Preview (if available)
    data_used = report_data.get("data_used")
    if data_used and isinstance(data_used, dict) and data_used.get("rows"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("Data Table Preview", heading_style))

        cols = data_used.get("columns", [])[:6]  # Max 6 columns in PDF table
        rows = data_used.get("rows", [])[:10]   # Max 10 rows

        table_data = [[Paragraph(f"<b>{c}</b>", body_style) for c in cols]]
        for r in rows:
            table_data.append([Paragraph(str(r.get(c, "")), body_style) for c in cols])

        t = Table(table_data, colWidths=[80] * len(cols))
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer
