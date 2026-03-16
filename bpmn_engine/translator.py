"""LLM Layer — semantic parsing: natural language → BPMN 2.0 JSON schema.

Pipeline position:
    **LLM (translate)** → APIs → ML → Engine
"""
import json
import re
from typing import Any

import config
_API_KEY: str = config.ANTHROPIC_API_KEY


def translate_to_bpmn_schema(input_text: str) -> dict[str, Any] | None:
    """LLM Layer entry point."""
    if _API_KEY:
        try:
            result = _llm_translate(input_text)
            if result and _validate_bpmn_output(result):
                return result
        except Exception as e:
            print(f"LLM translation failed: {e}")

    return _regex_translate(input_text)


def _validate_bpmn_output(bpmn_json: dict) -> bool:
    """Validate BPMN JSON has required structure."""
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


# Common action words for task extraction
ACTION_VERBS = [
    'place', 'submit', 'check', 'verify', 'validate', 'process', 'approve', 
    'reject', 'ship', 'deliver', 'create', 'send', 'notify', 'update',
    'calculate', 'generate', 'fetch', 'retrieve', 'review', 'assess',
    'complete', 'finish', 'close', 'archive', 'prepare', 'dispatch',
    'confirm', 'cancel', 'return', 'refund', 'inspect', 'pack'
]

# Words to strip from task names
STOP_WORDS = {'the', 'a', 'an', 'system', 'user', 'customer', 'employee', 'manager', 'it', 'then'}


def _extract_tasks(text: str) -> list[str]:
    """Extract verb-noun task patterns from text."""
    tasks = []
    text_lower = text.lower()
    
    # Split by common separators
    segments = re.split(r',| and | then |;', text)
    
    for seg in segments:
        seg = seg.strip()
        if not seg or len(seg) < 3:
            continue
            
        words = seg.split()
        cleaned_words = [w.rstrip('.,!?;:') for w in words if w.lower() not in STOP_WORDS]
        
        # Look for action verb as first word
        if cleaned_words:
            first_word = cleaned_words[0].lower()
            if first_word in ACTION_VERBS and len(cleaned_words) >= 2:
                # Verb + Object pattern
                noun = cleaned_words[1].capitalize()
                task_name = f"{first_word.capitalize()} {noun}"
                tasks.append(task_name)
            elif first_word.capitalize() not in STOP_WORDS:
                # Use first meaningful word as task
                task_name = first_word.capitalize()
                if len(cleaned_words) > 1:
                    task_name += " " + cleaned_words[1].capitalize()
                tasks.append(task_name)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tasks = []
    for t in tasks:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_tasks.append(t)
    
    return unique_tasks[:5]  # Max 5 tasks per segment


def _infer_task_type(task_name: str) -> str:
    """Infer task type from task name."""
    task_lower = task_name.lower()
    
    for pattern in ['check', 'validate', 'verify', 'process', 'send', 'notify', 'generate', 
                   'calculate', 'update', 'create', 'approve', 'reject', 'fetch', 'confirm']:
        if pattern in task_lower:
            return "Service"
    
    for pattern in ['ship', 'deliver', 'assemble', 'inspect', 'package', 'handle', 
                   'move', 'load', 'pick', 'pack', 'dispatch']:
        if pattern in task_lower:
            return "Manual"
    
    return "User"


def _regex_translate(input_text: str) -> dict[str, Any]:
    """Improved regex-based parser with better task extraction."""

    # Clean the text
    clean = re.sub(r"\s+", " ", input_text.strip())
    parts = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]

    title = parts[0].capitalize() if parts else "Process"
    all_tasks = []
    gateways = []
    
    # Detect organizations
    text_lower = input_text.lower()
    has_customer = any(kw in text_lower for kw in ['customer', 'client', 'buyer', 'user'])
    has_vendor = any(kw in text_lower for kw in ['vendor', 'supplier'])
    has_warehouse = any(kw in text_lower for kw in ['warehouse', 'distribution'])

    # Build pools
    pools = {}
    pools["Company"] = {"name": "Company", "lanes": {"Operations": {"name": "Operations", "tasks": []}}}
    
    if has_customer:
        pools["Customer"] = {"name": "Customer", "lanes": {"Customer": {"name": "Customer", "tasks": []}}}
    
    if has_vendor:
        pools["Vendor"] = {"name": "Vendor", "lanes": {"Vendor": {"name": "Vendor", "tasks": []}}}
    
    if has_warehouse:
        pools["Warehouse"] = {"name": "Warehouse", "lanes": {"Warehouse": {"name": "Warehouse", "tasks": []}}}

    # Find and process conditional statements (if/otherwise)
    # Pattern: "If [condition], [yes tasks]. Otherwise [no tasks]."
    conditional_pattern = r'if\s+(.+?)(?:\s*,\s*|\s+then\s+)(.+?)(?:\.?\s*otherwise\.?\s*(.+?))?(?:\.|$)'
    conditional_match = re.search(conditional_pattern, input_text, re.IGNORECASE | re.DOTALL)
    
    if conditional_match:
        condition = conditional_match.group(1).strip()
        yes_text = conditional_match.group(2).strip()
        no_text = conditional_match.group(3).strip() if conditional_match.group(3) else ""
        
        # Extract tasks from yes branch
        yes_tasks = _extract_tasks(yes_text)
        no_tasks = _extract_tasks(no_text) if no_text else []
        
        # Create gateway
        gw_name = condition.split()[0].capitalize() + "?"
        if len(condition.split()) > 1:
            gw_name = " ".join(condition.split()[:2]).rstrip(',') + "?"
        
        gateway = {
            "type": "Exclusive",
            "name": gw_name,
            "reasoning": f"Decision: {condition[:50]}",
            "yes_branch": yes_tasks if yes_tasks else ["Approve"],
            "no_branch": no_tasks if no_tasks else ["Reject"]
        }
        gateways.append(gateway)
        
        # Add branch tasks
        for t in gateway["yes_branch"] + gateway["no_branch"]:
            task_type = _infer_task_type(t)
            all_tasks.append({"name": t, "type": task_type, "pool": "Company", "lane": "Operations"})
    
    # Extract regular tasks from non-conditional sentences
    for i, s in enumerate(parts):
        s = s.strip()
        if not s:
            continue
        
        # Skip if this is part of a conditional we already processed
        if re.search(r'\bif\b', s, re.IGNORECASE) and conditional_match:
            continue
        if re.search(r'\botherwise\b', s, re.IGNORECASE) and conditional_match:
            continue
            
        # Skip very short segments
        if len(s) < 5:
            continue
            
        tasks = _extract_tasks(s)
        
        for task_name in tasks:
            # Skip if already added from gateway
            existing_names = [t["name"] for t in all_tasks]
            if task_name in existing_names:
                continue
            
            # Determine pool/lane based on context
            task_lower = task_name.lower()
            if has_warehouse and any(kw in task_lower for kw in ['ship', 'pack', 'pick', 'deliver', 'dispatch']):
                pool, lane = "Warehouse", "Warehouse"
            elif has_customer and i == 0:
                pool, lane = "Customer", "Customer"
            else:
                pool, lane = "Company", "Operations"
            
            all_tasks.append({
                "name": task_name, 
                "type": _infer_task_type(task_name), 
                "pool": pool, 
                "lane": lane
            })

    # Ensure at least one task
    if not all_tasks:
        all_tasks.append({"name": "Process Request", "type": "User", "pool": "Company", "lane": "Operations"})
    
    # Ensure gateway exists
    if not gateways:
        gateways.append({
            "type": "Exclusive",
            "name": "Approved?",
            "reasoning": "Default decision",
            "yes_branch": ["Approve"],
            "no_branch": ["Reject"]
        })
        all_tasks.append({"name": "Approve", "type": "Service", "pool": "Company", "lane": "Operations"})
        all_tasks.append({"name": "Reject", "type": "Service", "pool": "Company", "lane": "Operations"})

    # Assign tasks to pools
    for task in all_tasks:
        pool_name = task["pool"]
        lane_name = task["lane"]
        
        if pool_name in pools:
            if lane_name not in pools[pool_name]["lanes"]:
                pools[pool_name]["lanes"][lane_name] = {"name": lane_name, "tasks": []}
            pools[pool_name]["lanes"][lane_name]["tasks"].append({
                "name": task["name"],
                "type": task["type"],
                "assignee": lane_name
            })

    # Convert lanes dict to list
    for pool_name in pools:
        pools[pool_name]["lanes"] = list(pools[pool_name]["lanes"].values())

    # Build flows
    gw = gateways[0]
    start_name = title
    end_name = "End"

    flows = []
    
    # Get ordered task names (excluding gateway branch tasks that come from gateway definition)
    gateway_tasks = set(gw.get("yes_branch", []) + gw.get("no_branch", []))
    sequential_tasks = [t for t in all_tasks if t["name"] not in gateway_tasks]
    
    # Start -> first sequential task
    if sequential_tasks:
        flows.append({"type": "Sequence", "source": start_name, "target": sequential_tasks[0]["name"]})
        
        # Connect sequential tasks
        for j in range(len(sequential_tasks) - 1):
            flows.append({
                "type": "Sequence", 
                "source": sequential_tasks[j]["name"], 
                "target": sequential_tasks[j+1]["name"]
            })
        
        # Last sequential task -> gateway
        flows.append({"type": "Sequence", "source": sequential_tasks[-1]["name"], "target": gw["name"]})
    else:
        flows.append({"type": "Sequence", "source": start_name, "target": gw["name"]})

    # Gateway -> yes branch
    yes_branch = gw.get("yes_branch", [])
    if yes_branch:
        flows.append({"type": "Sequence", "source": gw["name"], "target": yes_branch[0]})
        for j in range(len(yes_branch) - 1):
            flows.append({"type": "Sequence", "source": yes_branch[j], "target": yes_branch[j+1]})
        flows.append({"type": "Sequence", "source": yes_branch[-1], "target": end_name})

    # Gateway -> no branch
    no_branch = gw.get("no_branch", [])
    if no_branch:
        flows.append({"type": "Sequence", "source": gw["name"], "target": no_branch[0]})
        for j in range(len(no_branch) - 1):
            flows.append({"type": "Sequence", "source": no_branch[j], "target": no_branch[j+1]})
        flows.append({"type": "Sequence", "source": no_branch[-1], "target": end_name})

    return {
        "title": title,
        "pools": list(pools.values()),
        "gateways": gateways,
        "events": [{"type": "Start", "name": start_name}, {"type": "End", "name": end_name}],
        "flows": flows,
        "message_flows": [],
        "message_events": [],
        "_colors": {},
    }
