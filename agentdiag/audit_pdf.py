"""
PDF export for CAFT audit reports.

Generates a professional single-page PDF suitable for forwarding to
a VP of Engineering. Uses reportlab (no external dependencies beyond pip).

Usage:
    from agentdiag.audit_pdf import export_pdf
    export_pdf(audit_results, "audit_report.pdf", team_size=5, hourly_rate=75)

    # Or from CLI:
    caft audit /path/to/traces -o report.pdf
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


# Colors matching the dashboard theme
GREEN = colors.HexColor("#22c55e")
YELLOW = colors.HexColor("#eab308")
RED = colors.HexColor("#ef4444")
DARK_BG = colors.HexColor("#1a1d27")
LIGHT_TEXT = colors.HexColor("#e2e8f0")
DIM_TEXT = colors.HexColor("#94a3b8")
ACCENT = colors.HexColor("#3b82f6")


def _health_color(health: str) -> colors.Color:
    if health == "green":
        return GREEN
    elif health == "yellow":
        return YELLOW
    elif health == "red":
        return RED
    return DIM_TEXT


def _anomaly_explanation(signature: str) -> str:
    explanations = {
        "distributional_shift": "Agent behavior pattern changed significantly mid-session",
        "mechanical_repetition": "Agent stuck in a read/edit/test/fail loop without progress",
        "context_thrashing": "Agent rapidly switching between unrelated files",
        "progress_stall": "Agent reading extensively but producing no output",
        "premature_termination": "Agent delivered without verifying its work",
    }
    return explanations.get(signature, f"Anomalous behavior: {signature}")


def _estimate_wasted_minutes(session: dict) -> float:
    anomalies = session.get("anomaly_count", 0)
    events = session.get("events", 0)
    if anomalies == 0:
        return 0.0
    return anomalies * 0.5 + min(15.0, events * 0.1)


def export_pdf(
    audit_results: dict,
    output_path: str,
    team_size: int = 1,
    sessions_per_week: int = 0,
    hourly_rate: float = 75.0,
    company_name: str = "Agent Monitoring Audit",
) -> str:
    """Export audit results as a professional PDF.

    Returns the output path on success, or raises ImportError if
    reportlab is not installed.
    """
    if not HAS_REPORTLAB:
        raise ImportError(
            "PDF export requires reportlab: pip install reportlab"
        )

    results = audit_results.get("results", [])
    n_sessions = len(results)
    if n_sessions == 0:
        raise ValueError("No sessions to export")

    healthy = [r for r in results if r["health"] == "green"]
    degraded = [r for r in results if r["health"] == "yellow"]
    problematic = [r for r in results if r["health"] == "red"]
    total_anomalies = sum(r["anomaly_count"] for r in results)

    anomaly_rate = len(degraded) + len(problematic)
    anomaly_pct = (anomaly_rate / n_sessions * 100) if n_sessions > 0 else 0

    total_wasted = sum(_estimate_wasted_minutes(r) for r in results)
    avg_wasted = total_wasted / max(anomaly_rate, 1)

    if sessions_per_week == 0:
        sessions_per_week = max(10, n_sessions * 2)

    problem_per_week = int(sessions_per_week * anomaly_pct / 100)
    weekly_hours = problem_per_week * avg_wasted / 60
    monthly_cost = weekly_hours * 4.3 * hourly_rate * team_size

    # Build PDF
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CAFTTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=4,
        textColor=colors.HexColor("#1e293b"),
    )
    subtitle_style = ParagraphStyle(
        "CAFTSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "CAFTHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#1e293b"),
    )
    body_style = ParagraphStyle(
        "CAFTBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    stat_style = ParagraphStyle(
        "CAFTStat",
        parent=styles["Normal"],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1e293b"),
        alignment=1,  # center
    )
    stat_label_style = ParagraphStyle(
        "CAFTStatLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#94a3b8"),
        alignment=1,
        spaceAfter=4,
    )

    elements = []

    # Title
    elements.append(Paragraph(company_name, title_style))
    elements.append(Paragraph(
        f"CAFT Agent Monitoring Audit | {datetime.now().strftime('%B %d, %Y')} | "
        f"{n_sessions} sessions analyzed",
        subtitle_style,
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#e2e8f0"),
    ))
    elements.append(Spacer(1, 12))

    # Executive summary boxes
    summary_data = [
        [
            Paragraph(f"<b>{len(healthy)}</b>", stat_style),
            Paragraph(f"<b>{len(degraded)}</b>", stat_style),
            Paragraph(f"<b>{len(problematic)}</b>", stat_style),
            Paragraph(f"<b>{total_anomalies}</b>", stat_style),
        ],
        [
            Paragraph("Healthy", stat_label_style),
            Paragraph("Degraded", stat_label_style),
            Paragraph("Problematic", stat_label_style),
            Paragraph("Total Anomalies", stat_label_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[1.6 * inch] * 4)
    summary_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        # Color the healthy/problematic numbers
        ("TEXTCOLOR", (0, 0), (0, 0), GREEN),
        ("TEXTCOLOR", (1, 0), (1, 0), YELLOW if len(degraded) > 0 else DIM_TEXT),
        ("TEXTCOLOR", (2, 0), (2, 0), RED if len(problematic) > 0 else DIM_TEXT),
        ("TEXTCOLOR", (3, 0), (3, 0), RED if total_anomalies > 0 else DIM_TEXT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # Cost estimate
    if anomaly_rate > 0:
        elements.append(Paragraph("Estimated Impact", heading_style))
        cost_text = (
            f"Based on {n_sessions} sessions analyzed, <b>{anomaly_pct:.0f}%</b> showed anomalous "
            f"behavior with an average detection delay of <b>{avg_wasted:.0f} minutes</b>. "
            f"Projected at {sessions_per_week} sessions/week with {team_size} developers, "
            f"this represents approximately <b>{weekly_hours:.1f} wasted hours/week</b>, "
            f"or <font color='#ef4444'><b>${monthly_cost:,.0f}/month</b></font> "
            f"in unproductive agent time (at ${hourly_rate:.0f}/hr loaded cost)."
        )
        elements.append(Paragraph(cost_text, body_style))
        elements.append(Spacer(1, 8))

    # Session table
    elements.append(Paragraph("Session Details", heading_style))

    table_data = [["Status", "Session", "Events", "Anomalies", "Action MI", "Issue"]]
    for r in sorted(results, key=lambda x: x["anomaly_count"], reverse=True):
        health = r["health"]
        name = Path(r.get("path", "unknown")).stem[:10]
        events = str(r.get("events", 0))
        anomalies = str(r.get("anomaly_count", 0))
        mi = f"{r.get('metrics', {}).get('action_mi', 0):.2f}b"

        # Find top anomaly signature
        issue = ""
        if r["anomaly_count"] > 0:
            sigs = {}
            for a in r.get("anomalies", []):
                if isinstance(a, dict):
                    sig = a.get("signature", "unclassified")
                    sigs[sig] = sigs.get(sig, 0) + 1
            if sigs:
                top = max(sigs, key=sigs.get)
                issue = _anomaly_explanation(top)[:50]

        status = {"green": "OK", "yellow": "WARN", "red": "FAIL"}.get(health, "?")
        table_data.append([status, name, events, anomalies, mi, issue])

    session_table = Table(
        table_data,
        colWidths=[0.5 * inch, 1.0 * inch, 0.6 * inch, 0.7 * inch, 0.7 * inch, 3.0 * inch],
    )
    session_table.setStyle(TableStyle([
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        # Body
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#334155")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))

    # Color-code status column
    for i, r in enumerate(sorted(results, key=lambda x: x["anomaly_count"], reverse=True), 1):
        color = _health_color(r["health"])
        session_table.setStyle(TableStyle([
            ("TEXTCOLOR", (0, i), (0, i), color),
            ("FONTNAME", (0, i), (0, i), "Helvetica-Bold"),
        ]))

    elements.append(session_table)
    elements.append(Spacer(1, 16))

    # Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    if problematic:
        recs = [
            "Set up continuous CAFT monitoring to detect anomalies in real-time.",
            "Review problematic sessions — anomaly signatures indicate specific failure modes "
            "addressable through better prompting or task decomposition.",
        ]
        if any(r["anomaly_count"] > 10 for r in results):
            recs.append(
                "Multiple repetition anomalies suggest adding retry limits or verification "
                "steps to agent workflows."
            )
    else:
        recs = [
            "All sessions appear healthy. Consider running CAFT in continuous monitoring "
            "mode to catch issues as they arise."
        ]

    for i, rec in enumerate(recs, 1):
        elements.append(Paragraph(f"<b>{i}.</b> {rec}", body_style))
        elements.append(Spacer(1, 4))

    # Footer
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(
        width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"),
    ))
    footer_style = ParagraphStyle(
        "CAFTFooter", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#94a3b8"),
        alignment=1,
    )
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Generated by CAFT (Cognitive Agent Fault Taxonomy) | "
        "Zero-training anomaly detection for AI agents | "
        "github.com/sam-kos41/caft-agent-monitor",
        footer_style,
    ))

    doc.build(elements)
    return output_path
