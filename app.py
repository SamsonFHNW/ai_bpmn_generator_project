"""AI-Augmented BPMN 2.0 Engine

4-Layer Pipeline (ABPMS Manifesto):
──────────────────────────────────
  Layer 1 — LLM       : Translate natural language → rich BPMN JSON
                         (Verb-Noun naming, Pool/Lane, message flows)
  Layer 2 — APIs      : Enrich with real-world context
                         (market trends, domain intelligence)
  Layer 3 — ML        : Graph-theoretic Lean Muda audit
                         (critical path, bottlenecks, waste detection)
  Layer 4 — Engine    : Render SVG + export BPMN 2.0 XML
"""
import os

import config  # loads .env; exposes ANTHROPIC_API_KEY, EXPORT_PATH, …
from nicegui import ui

from bpmn_engine import (
    translate_to_bpmn_schema,
    validate_bpmn_structure,
    analyze_process_health,
    generate_bpmn_xml,
    render_diagram,
)
from bpmn_engine.api_context import enrich_with_context


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_text: str) -> dict:
    """
    Execute the full 4-layer ABPMS pipeline.

    Returns
    ───────
    bpmn_json          dict   — enriched BPMN process model
    validation_ok      bool   — True if BPMN 2.0 structure is valid
    validation_errors  list   — human-readable error messages (may be empty)
    health_report      dict   — Lean Muda + graph analysis
    xml_file           str    — path to the exported .bpmn file
    """
    # ── Layer 1: LLM — Perceive ───────────────────────────────────────────────
    bpmn_json = translate_to_bpmn_schema(input_text)
    if not bpmn_json:
        raise ValueError("Translation failed — could not parse input text")

    # ── Layer 2: APIs — Enrich ────────────────────────────────────────────────
    bpmn_json = enrich_with_context(bpmn_json, input_text)

    # ── Layer 3: ML — Reason (validate + Muda audit) ──────────────────────────
    validation_ok, validation_errors = validate_bpmn_structure(bpmn_json)
    health_report                    = analyze_process_health(bpmn_json)

    # ── Layer 4: Engine — Enact (XML export) ──────────────────────────────────
    os.makedirs(config.EXPORT_PATH, exist_ok=True)
    xml_str  = generate_bpmn_xml(bpmn_json)
    xml_file = os.path.join(config.EXPORT_PATH, "diagram.bpmn")
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return {
        "bpmn_json":         bpmn_json,
        "validation_ok":     validation_ok,
        "validation_errors": validation_errors,
        "health_report":     health_report,
        "xml_file":          xml_file,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI Display Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_health_color(health: str) -> str:
    """Return CSS class based on health status."""
    if health == "good":
        return "text-green-700"
    if health == "needs_improvement":
        return "text-orange-600"
    return "text-red-600"


def _get_issue_color(severity: str) -> str:
    """Return CSS class based on issue severity."""
    if severity == "high":
        return "text-red-600"
    if severity == "medium":
        return "text-orange-500"
    return "text-gray-500"


def _display_result(result: dict) -> None:
    """Display the pipeline result in the UI."""
    bpmn_json = result["bpmn_json"]
    health    = result["health_report"]
    
    # Render SVG diagram
    svg = render_diagram(bpmn_json)
    
    out.clear()
    with out:
        ui.html(svg)

        # ── Parser mode badge ─────────────────────────────────────────────────
        mode = "LLM (Claude Haiku)" if config.ANTHROPIC_API_KEY else "Regex Fallback"
        ui.label(f"Parser: {mode}").classes("text-xs text-gray-400 mt-3")

        # ── Process domain (API Layer) ────────────────────────────────────────
        domain = bpmn_json.get("_process_type")
        if domain:
            ui.label(f"Domain detected: {domain}").classes("text-xs text-blue-500 mt-1")

        # ── API context sources ───────────────────────────────────────────────
        market = bpmn_json.get("_market_context")
        if market:
            trend = market.get("trend", "—")
            ui.label(
                f"Market ({market['source']}): trend={trend}"
            ).classes("text-xs text-amber-600 mt-1")

        # ── BPMN 2.0 validation (Semantic Logic Layer) ────────────────────────
        if result["validation_ok"]:
            ui.label("✔  Validation: BPMN 2.0 structure OK").classes(
                "text-xs text-green-700 mt-1"
            )
        else:
            first = result["validation_errors"][0] if result["validation_errors"] else ""
            count = len(result["validation_errors"])
            ui.label(
                f"⚠  Validation: {count} issue(s) — {first}"
            ).classes("text-xs text-yellow-700 mt-1")

        # ── Process health summary (ML Layer) ─────────────────────────────────
        h_color = _get_health_color(health["health"])
        ui.label(
            f"Process Health: {health['score']:.0%}  ·  "
            f"Waiting: {'⚠' if health['waiting_time_flag'] else '✔'}  ·  "
            f"Over-processing: {'⚠' if health['over_processing_flag'] else '✔'}  ·  "
            f"Tasks: {health['task_count']} ({health['manual_task_count']} manual)"
        ).classes(f"text-xs {h_color} mt-1")

        # ── Graph metrics (ML Layer) ──────────────────────────────────────────
        gm = health.get("graph_metrics")
        if gm:
            bottleneck_str = (
                f" · Bottlenecks: {', '.join(gm['bottleneck_nodes'][:3])}"
                if gm["bottleneck_nodes"] else ""
            )
            ui.label(
                f"Graph: {gm['node_count']} nodes · "
                f"{gm['edge_count']} edges · "
                f"Critical path: {gm['critical_path']}"
                f"{bottleneck_str}"
            ).classes("text-xs text-indigo-600 mt-1")

        # ── Muda issue details ───────────────────────────────────────────────
        for issue in health.get("issues", []):
            sev_color = _get_issue_color(issue["severity"])
            ui.label(
                f"  [{issue['type']}] {issue['description']}"
            ).classes(f"text-xs {sev_color}")

        # ── Export button ─────────────────────────────────────────────────────
        def export_xml() -> None:
            ui.notify(f"BPMN XML saved → {result['xml_file']}", type="positive")

        ui.button("Export BPMN XML", on_click=export_xml).classes("mt-3")


# ─────────────────────────────────────────────────────────────────────────────
# UI callbacks
# ─────────────────────────────────────────────────────────────────────────────

def generate() -> None:
    """Handle the Generate button click."""
    text = inp.value.strip()
    if not text:
        ui.notify("Please enter a process description", type="warning")
        return

    try:
        result = run_pipeline(text)
    except Exception as exc:
        ui.notify(f"Pipeline error: {exc}", type="negative")
        return

    _display_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# NiceGUI layout
# ─────────────────────────────────────────────────────────────────────────────

ui.add_head_html("""
<style>
  body { background: #ece4d2; font-family: Georgia, 'Times New Roman', serif; }
  .nicegui-content { max-width: 1200px; margin: 0 auto; padding: 24px; }
</style>
""")

ui.label("AI-Augmented BPMN 2.0 Engine").classes("text-2xl font-bold mb-1")
ui.label("LLM · APIs · ML · Engine").classes("text-sm text-gray-500 mb-4")

inp = ui.textarea(
    label="Process description",
    placeholder=(
        "e.g.  Customer submits invoice.  "
        "Finance team verifies invoice.  "
        "If approved, accounting records payment.  "
        "Otherwise, send rejection notice to vendor."
    ),
).classes("w-full mb-2").props("rows=3 outlined")

ui.button("Generate BPMN Diagram", on_click=generate).classes("mt-2")

out = ui.column().classes("w-full mt-4")

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(port=8001, title="AI BPMN Engine", reload=False)
