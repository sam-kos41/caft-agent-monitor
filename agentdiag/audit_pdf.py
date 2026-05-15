"""
PDF export for CAFT session behavioral-profile reports.

Single-page PDF describing the information-theoretic shape of each
session (steady / phase_shifting / looping). DESCRIPTIVE, not a quality
or cost verdict — the dollar-impact estimate and health verdict were
removed (see docs/CONSTRUCT_REVISION.md). team_size/hourly_rate are
kept for signature compatibility but ignored.

Usage:
    from agentdiag.audit_pdf import export_pdf
    export_pdf(audit_results, "report.pdf")

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

    GREEN = colors.HexColor("#22c55e")
    YELLOW = colors.HexColor("#eab308")
    RED = colors.HexColor("#ef4444")
    DARK_BG = colors.HexColor("#1a1d27")
    LIGHT_TEXT = colors.HexColor("#e2e8f0")
    DIM_TEXT = colors.HexColor("#94a3b8")
    ACCENT = colors.HexColor("#3b82f6")
except ImportError:
    HAS_REPORTLAB = False
    GREEN = YELLOW = RED = DARK_BG = LIGHT_TEXT = DIM_TEXT = ACCENT = None


def _health_color(health):
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

    # Descriptive behavioral state, not a quality verdict, and NO dollar
    # estimate — see docs/CONSTRUCT_REVISION.md.
    def _state(r):
        return r.get("behavioral_state", r.get("health", "unknown"))

    steady = [r for r in results if _state(r) == "steady"]
    phase_shifting = [r for r in results if _state(r) == "phase_shifting"]
    looping = [r for r in results if _state(r) == "looping"]
    total_anomalies = sum(r["anomaly_count"] for r in results)

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
        f"CAFT Session Behavioral Profile | {datetime.now().strftime('%B %d, %Y')} | "
        f"{n_sessions} sessions analyzed",
        subtitle_style,
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#e2e8f0"),
    ))
    elements.append(Spacer(1, 12))

    # Summary boxes — behavioral states (descriptive)
    summary_data = [
        [
            Paragraph(f"<b>{len(steady)}</b>", stat_style),
            Paragraph(f"<b>{len(phase_shifting)}</b>", stat_style),
            Paragraph(f"<b>{len(looping)}</b>", stat_style),
            Paragraph(f"<b>{total_anomalies}</b>", stat_style),
        ],
        [
            Paragraph("Steady", stat_label_style),
            Paragraph("Phase-shifting", stat_label_style),
            Paragraph("Looping", stat_label_style),
            Paragraph("IT-anomaly windows", stat_label_style),
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
        # Neutral coloring — no state is "good" or "bad"
        ("TEXTCOLOR", (0, 0), (3, 0), ACCENT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # Honest caveat in place of the (removed) dollar-impact estimate
    elements.append(Paragraph("How to read this", heading_style))
    elements.append(Paragraph(
        "Behavioral state describes the information-theoretic shape of a "
        "session (how repetitive vs. how varied). It is <b>not</b> a "
        "quality, success, or cost verdict. 'Looping' may be a focused "
        "code search (fine) or a stuck agent (not fine) — the signature "
        "alone cannot distinguish these; that requires task-outcome "
        "context. Use the per-session signatures as a place to look, "
        "not as a conclusion.",
        body_style,
    ))
    elements.append(Spacer(1, 8))

    # Session table
    elements.append(Paragraph("Session Details", heading_style))

    table_data = [["State", "Session", "Events", "IT-anom", "Action MI",
                   "Dominant signature"]]
    for r in sorted(results, key=lambda x: x["anomaly_count"], reverse=True):
        state = r.get("behavioral_state", r.get("health", "unknown"))
        name = Path(r.get("path", "unknown")).stem[:10]
        events = str(r.get("events", 0))
        anomalies = str(r.get("anomaly_count", 0))
        mi = f"{r.get('metrics', {}).get('action_mi', 0):.2f}b"

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

        table_data.append([state, name, events, anomalies, mi, issue])

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

    # State column: bold + neutral accent (no good/bad coloring)
    for i, _r in enumerate(results, 1):
        session_table.setStyle(TableStyle([
            ("TEXTCOLOR", (0, i), (0, i), ACCENT),
            ("FONTNAME", (0, i), (0, i), "Helvetica-Bold"),
        ]))

    elements.append(session_table)
    elements.append(Spacer(1, 16))

    # No "recommendations" — that implied the states were verdicts.
    elements.append(Paragraph("Interpreting these states", heading_style))
    recs = [
        "steady: stable, varied flow — no strong repetition or shift.",
        "phase_shifting: the action mix changes across the session "
        "(common in long, legitimately multi-task work).",
        "looping: repetition dominates — could be a focused search OR a "
        "stuck agent; inspect the dominant signature and the trace.",
        "IT-anomaly windows count within-session deviation, not errors.",
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
