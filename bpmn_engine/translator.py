"""LLM Layer — semantic parsing: natural language → BPMN 2.0 JSON schema.

Pipeline position:
    **LLM (translate)** → APIs → ML → Engine

Translation modes
─────────────────
1. Anthropic Claude (claude-haiku-4-5-20251001) — when ANTHROPIC_API_KEY is set.
2. Regex fallback — always available, used on any LLM failure.

LLM conventions enforced
────────────────────────
Verb-Noun naming   : "Verify Invoice", not "Invoice verification"
Pools vs Lanes     : Pool = separate org; Lane = internal role within one org
Flow types         : Sequence Flow (within pool) vs Message Flow (cross-pool)
Message events     : Start/End events with subtype="message" for cross-pool comms

Canonical output schema
───────────────────────
{
  "title":          str,
  "pools":          [{name, lanes: [{name, tasks: [{name, type, assignee}]}]}],
  "gateways":       [{type, name, reasoning, yes_branch, no_branch}],
  "events":         [{type, subtype?, name}],
  "flows":          [{type: Sequence|Message, source, target}],
  "message_flows":  [{source, target, label?}],
  "message_events": [{type, subtype: message, name, pool?, lane?}],
  "annotations":    [{text, attached_to}],
  "data_objects":   [{name, type: input|output, attached_to}],
  "_colors":        {str: str}   (optional custom colours)
}
"""
import json
import re
from pathlib import Path
from typing import Any

import config
_API_KEY: str = config.ANTHROPIC_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def translate_to_bpmn_schema(input_text: str) -> dict[str, Any] | None:
    """
    LLM Layer entry point.

    Translate a natural-language process description into rich BPMN JSON.
    Falls back to the regex parser on any LLM failure or missing key.

    Returns a dict on success, or None if both paths fail (practically
    impossible — the regex fallback always returns something).
    """
    if _API_KEY:
        try:
            result = _llm_translate(input_text)
            if result:
                return result
        except Exception:
            pass  # fall through to regex

    return _regex_translate(input_text)


# ─────────────────────────────────────────────────────────────────────────────
# LLM path — Anthropic Claude
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a BPMN 2.0 expert. Convert the process description into a rich JSON object.

TASK NAMING — Verb-Noun standard (REQUIRED)
────────────────────────────────────────────
Every task name MUST use active-voice Verb-Noun form, max 4 words:
  ✓ "Verify Invoice"    ✗ "Invoice verification"
  ✓ "Process Payment"   ✗ "Payment is processed"
  ✓ "Ship Order"        ✗ "The order gets shipped"
  ✓ "Check Stock"       ✗ "System checks if product is in stock"

GATEWAY RULES (CRITICAL)
─────────────────────────
Count the decision words in the description: "if", "otherwise", "whether",
"depending on". Create EXACTLY ONE gateway per decision point found.
  • Exclusive (XOR) : one path chosen  — use for "if/otherwise"
  • Parallel  (AND) : all paths run    — use for "at the same time / simultaneously"
  • Inclusive (OR)  : one or more paths — use for "one or more of"
Gateway name: ≤ 3 words ending in "?" describing the condition (e.g. "In Stock?").
yes_branch : list of EXACT task names that execute when condition is TRUE.
no_branch  : list of EXACT task names that execute when condition is FALSE/otherwise.
IMPORTANT: YES branch = the positive / available / approved path.
           NO  branch = the negative / unavailable / rejected path.

FLOWS — MANDATORY COMPLETENESS
───────────────────────────────
The "flows" array MUST include a Sequence edge for EVERY connection:
  Start → first task
  each task → next task (in order)
  last pre-gateway task → gateway
  gateway → each yes_branch task (first task in each yes path)
  gateway → each no_branch task (first task in each no path)
  each branch task → next branch task (if multiple tasks in a branch)
  last task in each branch → End
Do NOT omit any edge. The diagram will break if an edge is missing.
NEVER add a task to flows that is not also in a pool/lane tasks list.

POOLS vs LANES
──────────────
Pool = a SEPARATE organisation. Create one Pool per organisation involved.
Lane = an internal role/department within one Pool.
ALL tasks (including branch tasks) MUST appear in a pool/lane tasks list.
A Pool MUST have at least one Lane.

FLOW TYPES
──────────
Sequence Flow: within the same Pool (solid line).
Message Flow: cross-pool only (dashed). Never within the same pool.

WORKED EXAMPLE
──────────────
Description: "Customer submits order. System checks credit.
If approved, system confirms order and warehouse ships goods.
Otherwise system rejects order."

{
  "title": "Order Processing",
  "pools": [{"name": "Company", "lanes": [
    {"name": "Sales System", "tasks": [
      {"name": "Submit Order",    "type": "User"},
      {"name": "Check Credit",    "type": "Service"},
      {"name": "Confirm Order",   "type": "Service"},
      {"name": "Ship Goods",      "type": "Manual"},
      {"name": "Reject Order",    "type": "Service"}
    ]}
  ]}],
  "gateways": [{"type": "Exclusive", "name": "Credit OK?", "reasoning": "credit check outcome",
    "yes_branch": ["Confirm Order", "Ship Goods"],
    "no_branch":  ["Reject Order"]}],
  "events": [{"type": "Start", "name": "Start"}, {"type": "End", "name": "End"}],
  "flows": [
    {"type": "Sequence", "source": "Start",         "target": "Submit Order"},
    {"type": "Sequence", "source": "Submit Order",  "target": "Check Credit"},
    {"type": "Sequence", "source": "Check Credit",  "target": "Credit OK?"},
    {"type": "Sequence", "source": "Credit OK?",    "target": "Confirm Order"},
    {"type": "Sequence", "source": "Confirm Order", "target": "Ship Goods"},
    {"type": "Sequence", "source": "Ship Goods",    "target": "End"},
    {"type": "Sequence", "source": "Credit OK?",    "target": "Reject Order"},
    {"type": "Sequence", "source": "Reject Order",  "target": "End"}
  ]
}

OTHER ELEMENT RULES
───────────────────
Events     : 1–3 word names. Intermediate events only for explicit timers/messages.
Annotations: Business rule ≤ 10 words. "attached_to" = exact task name.
Data objects: "type" = "input"|"output"; name ≤ 4 words; "attached_to" = exact task name.
Message events: only for explicit cross-pool communication triggers.

Return ONLY valid JSON — no explanation, no markdown, no code fences.

Schema:
{
  "title": "...",
  "pools": [{"name": "...", "lanes": [{"name": "...", "tasks": [{"name": "...", "type": "User|Service|Manual"}]}]}],
  "gateways": [{"type": "Exclusive|Parallel|Inclusive", "name": "Short?", "reasoning": "...", "yes_branch": ["task"], "no_branch": ["task"]}],
  "events": [{"type": "Start|End|Intermediate", "name": "..."}],
  "flows": [{"type": "Sequence|Message", "source": "...", "target": "..."}],
  "message_flows": [],
  "message_events": [],
  "annotations": [{"text": "...", "attached_to": "task name"}],
  "data_objects": [{"name": "...", "type": "input|output", "attached_to": "task name"}]
}"""


def _llm_translate(input_text: str) -> dict[str, Any] | None:
    import anthropic

    client  = anthropic.Anthropic(api_key=_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Process description: {input_text}"}],
    )
    raw = message.content[0].text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",         "", raw)

    result = json.loads(raw)

    # Validate minimum required keys
    if "events" not in result or "pools" not in result:
        return None

    # Sanitise message_events: discard entries missing required fields
    me_clean = [
        me for me in (result.get("message_events") or [])
        if me.get("type") and me.get("subtype") == "message" and me.get("name")
    ]
    result["message_events"] = me_clean

    # Gateway-count sanity check:
    # count "if" / "otherwise" keywords that START a sentence — each pair = 1 gateway
    sentence_ifs = len(re.findall(
        r"(?:^|[.!?]\s+)if\b", input_text, flags=re.I | re.MULTILINE
    ))
    expected_gw = max(1, sentence_ifs)
    actual_gw   = len(result.get("gateways", []))
    if actual_gw > expected_gw + 1:   # allow 1 extra (e.g. nested condition)
        return None  # too many gateways — fall back to regex

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Regex fallback parser
# ─────────────────────────────────────────────────────────────────────────────

_STRIP_ACTOR = re.compile(
    r"^(?:the\s+)?(?:system|customer|user|warehouse|manager|staff|employee)\s+",
    flags=re.I,
)


def _split_tasks(text: str) -> list[str]:
    """Split a comma/and-separated action string into Verb-Noun task names (≤4 words)."""
    raw = [t.strip() for t in re.split(r",\s*|\s+and\s+", text) if t.strip()]
    result = []
    for t in raw:
        # Strip leading actor phrases ("The system checks…" → "checks…")
        cleaned = _STRIP_ACTOR.sub("", t).strip()
        words   = cleaned.split()
        result.append(" ".join(words[:4]).capitalize())
    return result or ["Proceed"]


def _regex_translate(input_text: str) -> dict[str, Any]:
    """Regex-based fallback parser — produces the same rich BPMN format as the LLM path."""

    clean = re.sub(r"\s+", " ", input_text.strip())
    parts = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]

    title:    str            = parts[0].capitalize() if parts else "Process"
    tasks:    list[dict]     = []
    gateways: list[dict]     = []
    colors:   dict[str, str] = {}
    pools:    dict[str, dict] = {}
    lanes:    dict[str, dict] = {}
    warehouse_keywords = ["warehouse", "location", "store", "distribution center", "amsterdam", "hamburg"]
    pool_names = set()
    lane_names = set()
    message_flows = []


    # Detect pools/lanes for warehouses/locations and track context
    current_pool = None
    current_lane = None
    for s in parts:
        found_pool = None
        for kw in warehouse_keywords:
            if kw in s.lower():
                pool_name = kw.capitalize()
                lane_name = f"{pool_name} Lane"
                pool_names.add(pool_name)
                lane_names.add(lane_name)
                if pool_name not in pools:
                    pools[pool_name] = {"name": pool_name, "lanes": []}
                if lane_name not in lanes:
                    lanes[lane_name] = {"name": lane_name, "tasks": []}
                found_pool = pool_name
        if found_pool:
            current_pool = found_pool
            current_lane = f"{current_pool} Lane"

    # Fallback: if no warehouse/location found, use default pool/lane
    if not pool_names:
        pool_names.add("Process")
        lane_names.add("Participant")
        pools["Process"] = {"name": "Process", "lanes": []}
        lanes["Participant"] = {"name": "Participant", "tasks": []}
        current_pool = "Process"
        current_lane = "Participant"

    # Parse tasks and gateways, assign based on context
    for s in parts[1:]:
        m_color = re.match(r"color:(\w+)=(#[0-9a-fA-F]{6})", s)
        if m_color:
            colors[m_color.group(1)] = m_color.group(2)
            continue

        if re.match(r"otherwise\b", s, flags=re.I) and gateways:
            action          = re.sub(r"^otherwise[\s,]+", "", s, flags=re.I).strip()
            no_branch_tasks = _split_tasks(action)
            if no_branch_tasks:
                gateways[-1]["no_branch"] = no_branch_tasks
            continue

        if re.match(r"if\b", s.strip(), flags=re.I):
            dlow = s.lower()
            if any(w in dlow for w in ("parallel", "simultaneous", "concurrent", "in parallel")):
                gw_type = "Parallel"
            elif any(w in dlow for w in ("either", "any", "one or more", "inclusive")):
                gw_type = "Inclusive"
            else:
                gw_type = "Exclusive"

            m_yes = re.search(r"\bthen\s+(.+?)(?:\s+else|$)", s, flags=re.I)
            m_no  = re.search(r"\belse\s+(.+?)$",              s, flags=re.I)

            if m_yes:
                m_cond     = re.search(r"\bif\s+(.+?)\s+then\b", s, flags=re.I)
                condition  = m_cond.group(1).strip() if m_cond else "condition"
                yes_branch = _split_tasks(m_yes.group(1))
                no_branch  = _split_tasks(m_no.group(1)) if m_no else ["Reject"]
            else:
                m_comma = re.match(r"if\s+(.+?),\s+(.+?)(?:\s+else\s+(.+?))?$", s, flags=re.I)
                if m_comma:
                    condition  = m_comma.group(1).strip()
                    yes_branch = _split_tasks(m_comma.group(2))
                    no_branch  = _split_tasks(m_comma.group(3)) if m_comma.group(3) else ["Reject"]
                else:
                    m_cond     = re.search(r"\bif\s+(.+?)$", s, flags=re.I)
                    condition  = m_cond.group(1).strip() if m_cond else s
                    yes_branch = ["Approve"]
                    no_branch  = ["Reject"]

            label = " ".join(condition.split()[:4]).capitalize().rstrip(",") + "?"

            gateways.append({
                "type":       gw_type,
                "name":       label,
                "reasoning":  f"Decision point: '{s[:60]}'",
                "yes_branch": yes_branch,
                "no_branch":  no_branch,
            })
        else:
            # Detect warehouse context and extract actions
            warehouse_match = re.search(r"warehouse in (\w+)[^.,]*([\w\s]+)", s, flags=re.I)
            if warehouse_match:
                pool = warehouse_match.group(1).capitalize()
                actions = warehouse_match.group(2)
                lane_name = f"{pool} Lane"
                # Final robust extraction: match all verb-object pairs
                phrases = re.split(r"and|,|then", actions)
                for phrase in phrases:
                    phrase = phrase.strip()
                    # Match verb+noun (e.g., 'checks stock', 'ships electronics', 'ships furniture')
                    m = re.match(r"(checks?|ships?|validates?|verifies?|processes?|sends?|notifies?|generates?|calculates?|updates?)\s+(\w+)", phrase, flags=re.I)
                    if m:
                        verb = m.group(1)
                        noun = m.group(2)
                        task_name = f"{verb.capitalize()} {noun.capitalize()}"
                        task_type = "Service"
                        if lane_name in lanes:
                            lanes[lane_name]["tasks"].append({"name": task_name, "type": task_type, "assignee": lane_name})
                        tasks.append({"name": task_name, "type": task_type, "assignee": lane_name})
                    else:
                        # Fallback: match any 'verb noun' pattern
                        fallback = re.match(r"(\w+)\s+(\w+)", phrase)
                        if fallback:
                            task_name = f"{fallback.group(1).capitalize()} {fallback.group(2).capitalize()}"
                            task_type = "Service"
                            if lane_name in lanes:
                                lanes[lane_name]["tasks"].append({"name": task_name, "type": task_type, "assignee": lane_name})
                            tasks.append({"name": task_name, "type": task_type, "assignee": lane_name})
                continue
            # Also handle 'while <warehouse> ships <item>'
            while_match = re.search(r"while (\w+) ships ([\w\s]+)", s, flags=re.I)
            if while_match:
                pool = while_match.group(1).capitalize()
                action = f"Ship {while_match.group(2).strip()}"
                lane_name = f"{pool} Lane"
                task_type = "Service"
                if lane_name in lanes:
                    lanes[lane_name]["tasks"].append({"name": action, "type": task_type, "assignee": lane_name})
                tasks.append({"name": action, "type": task_type, "assignee": lane_name})
                continue
            # Fallback: split compound sentences by 'and', 'then', ','
            sub_tasks = re.split(r"\band\b|\bthen\b|,", s)
            for sub in sub_tasks:
                cleaned_s = _STRIP_ACTOR.sub("", sub).strip()
                words     = cleaned_s.split()
                task_name = " ".join(words[:4]).capitalize() or sub.capitalize()
                task_type = "Service" if re.match(
                    r"(check|validate|verify|process|send|notify|generate|calculate|update|ship)",
                    words[0] if words else "", flags=re.I
                ) else "User"
                if current_lane and current_lane in lanes:
                    lanes[current_lane]["tasks"].append({"name": task_name, "type": task_type, "assignee": current_lane})
                else:
                    # Default to first lane
                    first_lane = next(iter(lanes.values()))
                    first_lane["tasks"].append({"name": task_name, "type": task_type, "assignee": first_lane["name"]})
                tasks.append({"name": task_name, "type": task_type, "assignee": "Participant"})

    if not tasks:
        first_lane = next(iter(lanes.values()))
        first_lane["tasks"].append({"name": "Process Request", "type": "User", "assignee": first_lane["name"]})
        tasks = [{"name": "Process Request", "type": "User", "assignee": "Participant"}]
    if not gateways:
        gateways = [{
            "type":       "Exclusive",
            "name":       "Decision?",
            "reasoning":  "Default decision gateway",
            "yes_branch": ["Approve"],
            "no_branch":  ["Reject"],
        }]

    # Build ordered flows
    task_names = [t["name"] for t in tasks]
    gw         = gateways[0]
    start_name = title
    end_name   = "End"

    flows: list[dict] = []
    for pool_name in pool_names:
        lane_name = f"{pool_name} Lane"
        lane_tasks = lanes[lane_name]["tasks"] if lane_name in lanes else []
        if lane_tasks:
            flows.append({"type": "Sequence", "source": start_name, "target": lane_tasks[0]["name"]})
            for i in range(len(lane_tasks) - 1):
                flows.append({"type": "Sequence", "source": lane_tasks[i]["name"], "target": lane_tasks[i + 1]["name"]})
            flows.append({"type": "Sequence", "source": lane_tasks[-1]["name"], "target": gw["name"]})

    if not flows:
        flows.append({"type": "Sequence", "source": start_name, "target": gw["name"]})

    for branch in (gw["yes_branch"], gw["no_branch"]):
        if not branch:
            continue
        flows.append({"type": "Sequence", "source": gw["name"], "target": branch[0]})
        for i in range(len(branch) - 1):
            flows.append({"type": "Sequence", "source": branch[i], "target": branch[i + 1]})
        flows.append({"type": "Sequence", "source": branch[-1], "target": end_name})

    # Generate message flows for cross-pool communication
    if len(pool_names) > 1:
        pool_list = list(pool_names)
        for i, pool_a in enumerate(pool_list):
            for pool_b in pool_list[i+1:]:
                lane_a = lanes[f"{pool_a} Lane"]
                lane_b = lanes[f"{pool_b} Lane"]
                for task_a in lane_a["tasks"]:
                    for task_b in lane_b["tasks"]:
                        if "order" in task_a["name"].lower() and "order" in task_b["name"].lower():
                            message_flows.append({"source": task_a["name"], "target": task_b["name"], "label": "Order Transfer"})

    # Add branch tasks to all lanes for layout
    branch_task_names = {t["name"] for t in tasks}
    for gw in gateways:
        for t_name in gw.get("yes_branch", []) + gw.get("no_branch", []):
            if t_name not in branch_task_names:
                first_lane = next(iter(lanes.values()))
                first_lane["tasks"].append({"name": t_name, "type": "User", "assignee": first_lane["name"]})
                tasks.append({"name": t_name, "type": "User", "assignee": "Participant"})
                branch_task_names.add(t_name)

    # Assemble pools/lanes
    for pool_name in pool_names:
        lane_name = f"{pool_name} Lane"
        if lane_name in lanes:
            pools[pool_name]["lanes"].append(lanes[lane_name])

    return {
        "title":          title,
        "pools":          list(pools.values()),
        "gateways":       gateways,
        "events":         [
            {"type": "Start", "name": start_name},
            {"type": "End",   "name": end_name},
        ],
        "flows":          flows,
        "message_flows":  message_flows,
        "message_events": [],
        "_colors":        colors,
    }
