"""BPMN 2.0 XML export.

Generates standard-compliant BPMN 2.0 XML from the rich process JSON
produced by the translator.  Uses only stdlib (uuid + string formatting)
so there is no dependency on the unmaintained bpmn-python library.
"""
import uuid
from typing import Any


def _uid(prefix: str = "el") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _esc(text: str) -> str:
    """Escape XML special characters in attribute/element values."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_bpmn_xml(process_json: dict[str, Any]) -> str:
    """
    Generate BPMN 2.0 compliant XML from a rich process JSON dict.

    The output starts with a proper XML declaration and uses the standard
    BPMN 2.0 namespaces (bpmn, bpmndi, dc, di).
    """
    proc_id = _uid("Process")
    element_ids: dict[str, str] = {}

    def get_id(name: str, prefix: str = "el") -> str:
        if name not in element_ids:
            element_ids[name] = _uid(prefix)
        return element_ids[name]

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn:definitions',
        '  xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"',
        '  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"',
        '  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"',
        '  xmlns:di="http://www.omg.org/spec/DD/20100524/DI"',
        '  targetNamespace="http://bpmn.io/schema/bpmn"',
        f'  id="Definitions_{uuid.uuid4().hex[:8]}">',
        f'  <bpmn:process id="{proc_id}"',
        f'    name="{_esc(process_json.get("title", "Process"))}"',
        '    isExecutable="true">',
    ]

    # ── Events ────────────────────────────────────────────────────────────────
    for event in process_json.get("events", []):
        eid  = get_id(event["name"], "ev")
        name = _esc(event["name"])
        etype = event.get("type", "Start")
        if etype == "Start":
            tag = "startEvent"
        elif etype == "End":
            tag = "endEvent"
        else:
            tag = "intermediateThrowEvent"
        lines.append(f'    <bpmn:{tag} id="{eid}" name="{name}"/>')

    # ── Tasks from pools / lanes ──────────────────────────────────────────────
    for pool in process_json.get("pools", []):
        for lane in pool.get("lanes", []):
            for task in lane.get("tasks", []):
                tid       = get_id(task["name"], "task")
                task_name = _esc(task["name"])
                task_type = task.get("type", "User")
                if task_type == "Service":
                    tag = "serviceTask"
                elif task_type == "Manual":
                    tag = "manualTask"
                else:
                    tag = "userTask"
                assignee = task.get("assignee", "")
                extra = f' activiti:assignee="{_esc(assignee)}"' if assignee else ""
                lines.append(f'    <bpmn:{tag} id="{tid}" name="{task_name}"{extra}/>')

    # ── Gateways + inline branch tasks ────────────────────────────────────────
    for gw in process_json.get("gateways", []):
        gid     = get_id(gw["name"], "gw")
        gw_name = _esc(gw.get("name", "Decision?"))
        gw_type = gw.get("type", "Exclusive")
        if gw_type == "Parallel":
            tag = "parallelGateway"
        elif gw_type == "Inclusive":
            tag = "inclusiveGateway"
        else:
            tag = "exclusiveGateway"
        lines.append(f'    <bpmn:{tag} id="{gid}" name="{gw_name}"/>')

        # Branch tasks that may not appear in any pool
        for t_name in gw.get("yes_branch", []) + gw.get("no_branch", []):
            if t_name not in element_ids:
                tid = get_id(t_name, "task")
                lines.append(
                    f'    <bpmn:userTask id="{tid}" name="{_esc(t_name)}"/>'
                )

    # ── Sequence flows ────────────────────────────────────────────────────────
    for flow in process_json.get("flows", []):
        src = flow.get("source", "")
        tgt = flow.get("target", "")
        if src in element_ids and tgt in element_ids:
            fid  = _uid("flow")
            ftype = flow.get("type", "Sequence")
            tag  = "sequenceFlow" if ftype == "Sequence" else "messageFlow"
            lines.append(
                f'    <bpmn:{tag} id="{fid}"'
                f' sourceRef="{element_ids[src]}"'
                f' targetRef="{element_ids[tgt]}"/>'
            )

    lines += [
        "  </bpmn:process>",
        "</bpmn:definitions>",
    ]

    return "\n".join(lines)
