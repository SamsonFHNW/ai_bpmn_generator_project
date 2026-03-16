"""BPMN Engine — public API.

4-Layer pipeline importable from bpmn_engine:

    from bpmn_engine import (
        translate_to_bpmn_schema,   # Layer 1: LLM   (Semantic Parsing)
        validate_bpmn_structure,    # Layer 3: ML    (Semantic Logic)
        analyze_process_health,     # Layer 3: ML    (Muda + Graph Analysis)
        generate_bpmn_xml,          # Layer 4: Engine (XML export)
        render_diagram,             # Layer 4: Engine (SVG render)
    )

    # Layer 2: API enrichment — import directly:
    from bpmn_engine.api_context import enrich_with_context
"""
from .translator import translate_to_bpmn_schema
from .validator  import validate_bpmn_structure
from .analyzer   import analyze_process_health
from .exporter   import generate_bpmn_xml
from .renderer   import render_diagram

__all__ = [
    "translate_to_bpmn_schema",
    "validate_bpmn_structure",
    "analyze_process_health",
    "generate_bpmn_xml",
    "render_diagram",
]
