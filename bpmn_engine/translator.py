"""LLM Layer — semantic parsing: natural language → BPMN 2.0 JSON schema.

Pipeline position:
    **LLM (translate)** → APIs → ML → Engine

Translation modes
────────────────
1. Anthropic Claude (claude-haiku-4-5-20251001) — when ANTHROPIC_API_KEY is set.
2. Enhanced Regex fallback — always available, used on any LLM failure.

LLM conventions enforced
───────────────────────
Verb-Noun naming   : "Verify Invoice", not "Invoice verification"
Pools vs Lanes     : Pool = separate org; Lane = internal role within one org
Flow types         : Sequence Flow (within pool) vs Message Flow (cross-pool)
Message events     : Start/End events with subtype="message" for cross-pool comms
"""
import json
import re
from typing import Any

import config
_API_KEY: str = config.ANTHROPIC_API_KEY


def translate_to_bpmn_schema(input_text: str) -> dict[str, Any] | None:
    """LLM Layer entry point - translate natural language to BPMN JSON."""
    if _API_KEY:
        try:
            result = _llm_translate(input_text)
            if result and _validate_bpmn_output(result):
                return result
        except Exception as e:
            print(f"LLM translation failed: {e}")
            pass

    return _regex_translate(input_text)


def _validate_bpmn_output(bpmn_json: dict) -> bool:
    """Validate BPMN JSON has required structure and valid references."""
    if "events" not in bpmn_json or "pools" not in bpmn_json:
        return False
    
    node_names = set()
    for ev in bpmn_json.get("events", []):
        if ev.get("name"):
            node_names.add(ev["name"])
    
    for pool in bpmn_json.get("pools", []):
        for lane in pool.get("lanes", []):
            for task in lane.get("tasks", []):
                if task.get("name"):
                    node_names.add(task["name"])
    
    for gw in bpmn_json.get("gateways", []):
        if gw.get("name"):
            node_names.add(gw["name"])
        node_names.update(gw.get("yes_branch", []))
        node_names.update(gw.get("no_branch", []))
    
    for flow in bpmn_json.get("flows", []):
        if flow.get("source") not in node_names or flow.get("target") not in node_names:
            return False
    
    return True


_SYSTEM_PROMPT = """You are a BPMN 2.0 expert AI. Convert natural language process descriptions into valid BPMN 2.0 JSON.

TASK NAMING: Use Verb-Noun form, max 4 words. Examples: "Verify Invoice", "Process Payment", "Ship Order"
Task Types: User (human), Service (automated), Manual (physical work)

GATEWAYS: 
- Exclusive (XOR): "if/otherwise" decisions
- Parallel (AND): "simultaneously/in parallel" 
- Inclusive (OR): "one or more of"

FLOW RULES:
- Every flow source/target must exist
- Sequence Flow: within same pool (solid)
- Message Flow: cross-pool (dashed)
- Include ALL connections in flows array

EXAMPLE:
Input: "Customer places order. System checks inventory. If in stock, ship order. Otherwise backorder."
Output:
{
  "title": "Order Processing",
  "pools": [
    {"name": "Customer", "lanes": [{"name": "Customer", "tasks": [{"name": "Place Order", "type": "User"}]}]},
    {"name": "Company", "lanes": [
      {"name": "Sales", "tasks": [{"name": "Check Inventory", "type": "Service"}]},
      {"name": "Warehouse", "tasks": [{"name": "Ship Order", "type": "Manual"}, {"name": "Create Backorder", "type": "Service"}]}
    ]}
  ],
  "gateways": [{"type": "Exclusive", "name": "In Stock?", "reasoning": "Check availability", "yes_branch": ["Ship Order"], "no_branch": ["Create Backorder"]}],
  "events": [{"type": "Start", "name": "Start"}, {"type": "End", "name": "End"}],
  "flows": [
    {"type": "Sequence", "source": "Start", "target": "Place Order"},
    {"type": "Message", "source": "Place Order", "target": "Check Inventory"},
    {"type": "Sequence", "source": "Check Inventory", "target": "In Stock?"},
    {"type": "Sequence", "source": "In Stock?", "target": "Ship Order"},
    {"type": "Sequence", "source": "Ship Order", "target": "End"},
    {"type": "Sequence", "source": "In Stock?", "target": "Create Backorder"},
    {"type": "Sequence", "source": "Create Backorder", "target": "End"}
  ],
  "message_flows": [{"source": "Place Order", "target": "Check Inventory", "label": "Order"}]
}

Return ONLY valid JSON."""


def _llm_translate(input_text: str) -> dict[str, Any] | None:
    import anthropic

    client = anthropic.Anthropic(api_key=_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Process: {input_text}"}],
    )
    raw = message.content[0].text.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    if "events" not in result or "pools" not in result:
        return None

    me_clean = [
        me for me in (result.get("message_events") or [])
        if me.get("type") and me.get("subtype") == "message" and me.get("name")
    ]
    result["message_events"] = me_clean

    return result


_STRIP_ACTOR = re.compile(
    r"^(?:the\s+)?(?:system|customer|user|warehouse|manager|staff|employee)\s+",
    flags=re.I,
)


def _split_tasks(text: str) -> list[str]:
    """Split comma/and-separated actions into Verb-Noun tasks."""
    raw = [t.strip() for t in re.split(r",\s*|\s+and\s+", text) if t.strip()]
    result = []
    for t in raw:
        cleaned = _STRIP_ACTOR.sub("", t).strip()
        words = cleaned.split()
        result.append(" ".join(words[:4]).capitalize())
    return result or ["Proceed"]


def _infer_task_type(task_name: str) -> str:
    """Infer task type from common patterns."""
    task_lower = task_name.lower()
    
    service_patterns = ['check', 'validate', 'verify', 'process', 'send', 'notify',
        'generate', 'calculate', 'update', 'create', 'approve', 'reject', 'fetch']
    manual_patterns = ['ship', 'deliver', 'assemble', 'inspect', 'package', 'handle', 'move']
    
    for pattern in service_patterns:
        if pattern in task_lower:
            return "Service"
    for pattern in manual_patterns:
        if pattern in task_lower:
            return "Manual"
    return "User"


def _detect_organizations(text: str) -> dict:
    """Detect organizations in the description."""
    org_patterns = {
        'customer': r'\b(customer|client|buyer)\b',
        'vendor': r'\b(vendor|supplier)\b',
        'warehouse': r'\b(warehouse|distribution)\b',
    }
    found = {}
    text_lower = text.lower()
    for org, pattern in org_patterns.items():
        if re.search(pattern, text_lower):
            found[org] = True
    return found


def _regex_translate(input_text: str) -> dict[str, Any]:
    """Enhanced regex-based fallback parser."""

    clean = re.sub(r"\s+", " ", input_text.strip())
    parts = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]

    title = parts[0].capitalize() if parts else "Process"
    tasks, gateways, colors = [], [], {}
    pools, lanes = {}, {}
    message_flows = []

    detected_orgs = _detect_organizations(input_text)
    pool_names, lane_names = set(), set()
    
    pool_names.add("Company")
    lanes["Company"] = {"name": "Company", "tasks": []}
    
    for org in detected_orgs:
        pool_names.add(org.capitalize())
        lanes[org.capitalize()] = {"name": org.capitalize(), "tasks": []}

    current_lane = "Company"

    for s in parts[1:]:
        if not s.strip():
            continue
            
        m_color = re.match(r"color:(\w+)=(#[0-9a-fA-F]{6})", s)
        if m_color:
            colors[m_color.group(1)] = m_color.group(2)
            continue

        if re.match(r"otherwise\b", s, flags=re.I) and gateways:
            action = re.sub(r"^otherwise[\s,]+", "", s, flags=re.I).strip()
            no_branch = _split_tasks(action)
            if no_branch:
                gateways[-1]["no_branch"] = no_branch
            continue

        if re.match(r"(if|whether|depending on)\b", s.strip(), flags=re.I):
            dlow = s.lower()
            
            if any(w in dlow for w in ("parallel", "simultaneous", "in parallel")):
                gw_type = "Parallel"
            elif any(w in dlow for w in ("either", "any", "one or more")):
                gw_type = "Inclusive"
            else:
                gw_type = "Exclusive"

            m_cond = re.search(r"\b(?:if|whether)\s+(.+?)(?:\s+then|,\s*|$)", s, flags=re.I)
            condition = m_cond.group(1).strip() if m_cond else s
            
            m_yes = re.search(r"\b(?:then|do)\s+(.+?)(?:\s+else|\s+otherwise|$)", s, flags=re.I)
            m_no = re.search(r"\b(?:else|otherwise)\s+(.+?)$", s, flags=re.I)
            
            yes_branch = _split_tasks(m_yes.group(1)) if m_yes else ["Approve"]
            no_branch = _split_tasks(m_no.group(1)) if m_no else ["Reject"]
            
            label = " ".join(condition.split()[:4]).capitalize().rstrip(",") + "?"
            
            gateways.append({
                "type": gw_type,
                "name": label,
                "reasoning": f"Decision: '{s[:50]}'",
                "yes_branch": yes_branch,
                "no_branch": no_branch,
            })
            continue

        sub_tasks = re.split(r"\band\b|\bthen\b|,", s)
        for sub in sub_tasks:
            cleaned_s = _STRIP_ACTOR.sub("", sub).strip()
            if not cleaned_s:
                continue
            words = cleaned_s.split()
            task_name = " ".join(words[:4]).capitalize() or sub.capitalize()
            task_type = _infer_task_type(task_name)
            
            lanes[current_lane]["tasks"].append({
                "name": task_name, "type": task_type, "assignee": current_lane
            })
            tasks.append({"name": task_name, "type": task_type, "assignee": current_lane})

    if not tasks:
        lanes["Company"]["tasks"].append({"name": "Process Request", "type": "User", "assignee": "Company"})
        tasks = [{"name": "Process Request", "type": "User", "assignee": "Company"}]
    
    if not gateways:
        gateways = [{"type": "Exclusive", "name": "Decision?", "reasoning": "Default",
            "yes_branch": ["Approve"], "no_branch": ["Reject"]}]

    gw = gateways[0]
    start_name, end_name = title, "End"
    flows = []
    
    first_task = lanes[current_lane]["tasks"][0]["name"] if lanes[current_lane]["tasks"] else None
    if first_task:
        flows.append({"type": "Sequence", "source": start_name, "target": first_task})
        for lane in lanes.values():
            lt = lane["tasks"]
            for j in range(len(lt) - 1):
                flows.append({"type": "Sequence", "source": lt[j]["name"], "target": lt[j+1]["name"]})
            if lt:
                flows.append({"type": "Sequence", "source": lt[-1]["name"], "target": gw["name"]})
    else:
        flows.append({"type": "Sequence", "source": start_name, "target": gw["name"]})

    for branch_key in ["yes_branch", "no_branch"]:
        branch = gw.get(branch_key, [])
        if branch:
            flows.append({"type": "Sequence", "source": gw["name"], "target": branch[0]})
            for j in range(len(branch) - 1):
                flows.append({"type": "Sequence", "source": branch[j], "target": branch[j+1]})
            flows.append({"type": "Sequence", "source": branch[-1], "target": end_name})

    branch_task_names = {t["name"] for t in tasks}
    for gw in gateways:
        for t_name in gw.get("yes_branch", []) + gw.get("no_branch", []):
            if t_name not in branch_task_names:
                t_type = _infer_task_type(t_name)
                lanes["Company"]["tasks"].append({"name": t_name, "type": t_type, "assignee": "Company"})
                tasks.append({"name": t_name, "type": t_type, "assignee": "Company"})
                branch_task_names.add(t_name)

    for pool_name in pool_names:
        pool_lanes = [lanes[ln] for ln in lanes if ln.lower() == pool_name.lower() or pool_name == "Company"]
        if pool_lanes:
            pools[pool_name] = {"name": pool_name, "lanes": pool_lanes}

    if not pools:
        pools["Company"] = {"name": "Company", "lanes": [lanes.get("Company", {"name": "Company", "tasks": []})]}

    return {
        "title": title,
        "pools": list(pools.values()),
        "gateways": gateways,
        "events": [{"type": "Start", "name": start_name}, {"type": "End", "name": end_name}],
        "flows": flows,
        "message_flows": message_flows,
        "message_events": [],
        "_colors": colors,
    }
