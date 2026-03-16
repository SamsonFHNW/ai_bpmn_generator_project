"""API Layer — External perception: enriches BPMN JSON with real-world context.

Pipeline position:
    LLM (translate) → **API (enrich)** → ML (analyze) → Engine (render)

Providers
─────────
MarketTrendProvider      : Simulated market-demand / trend data.
NullContextProvider      : No-op for process types with no useful API data.

Public API
──────────
    from bpmn_engine.api_context import enrich_with_context

    bpmn_json = enrich_with_context(bpmn_json, raw_description)
    # Adds _process_type, _market_context keys.
    # Injects context as BPMN text annotations on the first task.
"""
import json
import os
import urllib.request
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Process-domain detector
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "logistics":   ["ship", "deliver", "freight", "warehouse", "dispatch", "route",
                    "transport", "carrier", "shipment"],
    "finance":     ["invoice", "payment", "loan", "credit", "budget", "expense",
                    "ledger", "payroll", "bank", "fund"],
    "procurement": ["purchase", "vendor", "supplier", "contract", "rfq", "order",
                    "tender", "sourcing", "procurement"],
    "hr":          ["hire", "onboard", "employee", "recruitment", "leave",
                    "appraisal", "resignation", "headcount"],
    "inventory":   ["stock", "reorder", "inventory", "replenish", "supply",
                    "safety stock", "sku", "bin"],
}


def _detect_process_type(description: str) -> str:
    """Return the best-matching domain label for a process description."""
    text   = description.lower()
    scores = {
        domain: sum(1 for kw in kws if kw in text)
        for domain, kws in _DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "generic"


# ─────────────────────────────────────────────────────────────────────────────
# Provider base class
# ─────────────────────────────────────────────────────────────────────────────

class APIContextProvider:
    """Abstract provider: returns a context dict for injection into BPMN."""

    name: str = "base"

    def get_context(self, process_type: str, description: str) -> dict[str, Any]:
        return {}


# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# MarketTrendProvider
# ─────────────────────────────────────────────────────────────────────────────

_TREND_TABLE: dict[str, dict[str, Any]] = {
    "logistics":   {"trend": "supply_chain_stress", "backlog_days": 3,
                    "suggestion": "Supply chain stress detected (backlog: 3 days). "
                                  "Add a buffer-stock check task before dispatch."},
    "finance":     {"trend": "rising_rates",        "volatility": "medium",
                    "suggestion": "Rising interest rates. Add rate-validation "
                                  "step before issuing loan or credit approvals."},
    "procurement": {"trend": "vendor_consolidation","preferred_vendors": 2,
                    "suggestion": "Market consolidating — prefer approved vendor list "
                                  "and add dual-source approval gate."},
    "hr":          {"trend": "talent_shortage",     "avg_time_to_hire_days": 45,
                    "suggestion": "Avg. time-to-hire is 45 days. "
                                  "Parallelize screening and reference checks."},
    "inventory":   {"trend": "demand_rising",       "safety_stock_multiplier": 1.3,
                    "suggestion": "Demand trending up (+30%). "
                                  "Increase safety-stock reorder point in diagram."},
    "generic":     {"trend": "stable",
                    "suggestion": "Market conditions stable — no adaptive changes needed."},
}


class MarketTrendProvider(APIContextProvider):
    """
    Returns simulated market-trend intelligence for the detected process domain.

    In production this would call Alpha Vantage, Bloomberg, or a proprietary
    data feed.  The simulated values are intentionally realistic to demonstrate
    the injection mechanism.
    """

    name = "market"

    def get_context(self, process_type: str, description: str) -> dict[str, Any]:
        row = _TREND_TABLE.get(process_type, _TREND_TABLE["generic"])
        return {"source": "Market Trends (simulated)", **row}


# ─────────────────────────────────────────────────────────────────────────────
# NullContextProvider
# ─────────────────────────────────────────────────────────────────────────────

class NullContextProvider(APIContextProvider):
    """No-op provider — returns an empty context for unknown domains."""
    name = "none"


# ─────────────────────────────────────────────────────────────────────────────
# Provider registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, list[APIContextProvider]] = {
    "logistics":   [MarketTrendProvider()],
    "inventory":   [MarketTrendProvider()],
    "finance":     [MarketTrendProvider()],
    "procurement": [MarketTrendProvider()],
    "hr":          [MarketTrendProvider()],
    "generic":     [NullContextProvider()],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_context(
    bpmn_json:   dict[str, Any],
    description: str,
) -> dict[str, Any]:
    """
    API Layer entry point.

    1. Detect the process domain from the raw description.
    2. Query registered providers for real-world context.
    3. Inject context as BPMN text annotations on the first task.
    4. Return an enriched copy of bpmn_json (original is not mutated).

    Underscore-prefixed metadata keys (_process_type,
    _market_context) follow the existing _colors convention — they are
    ignored by the validator, exporter, and renderer.
    """
    process_type = _detect_process_type(description)
    providers    = _REGISTRY.get(process_type, [NullContextProvider()])

    # Collect non-null contexts
    contexts: list[dict[str, Any]] = [
        ctx
        for p in providers
        if p.name != "none"
        for ctx in [p.get_context(process_type, description)]
        if ctx
    ]

    # Shallow copy — never mutate the caller's dict
    result = dict(bpmn_json)
    result["_process_type"] = process_type

    # Store raw context blobs for UI display
    for ctx in contexts:
        src = ctx.get("source", "")
        if "market" in src.lower() or "Market" in src:
            result["_market_context"] = ctx

    if not contexts:
        return result

    # Find first task as annotation anchor
    first_task: str | None = next(
        (
            t["name"]
            for pool in bpmn_json.get("pools", [])
            for lane in pool.get("lanes", [])
            for t in lane.get("tasks", [])
        ),
        None,
    )
    if not first_task:
        return result

    # Inject suggestions as BPMN text annotations (≤10 words each)
    existing_anns: list[dict] = list(bpmn_json.get("annotations", []) or [])
    for ctx in contexts:
        suggestion = ctx.get("suggestion", "")
        if not suggestion:
            continue
        truncated = " ".join(suggestion.split()[:10])
        existing_anns.append({
            "text":        truncated,
            "attached_to": first_task,
            "source":      ctx.get("source", "API"),
        })

    result["annotations"] = existing_anns
    return result
