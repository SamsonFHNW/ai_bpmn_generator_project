"""Semantic Logic Layer — strict BPMN 2.0 structural validation.

Pipeline position:
    LLM → APIs → **Validator + ML** → Engine

Checks
──────
  1. Exactly one Start event     (thin-border circle)
  2. At least one End event      (thick-border circle)
  3. Valid event subtypes        (message | timer | none)
  4. Valid gateway types         (Exclusive | Parallel | Inclusive)
  5. Gateway branch completeness (yes_branch or no_branch must be defined)
  6. Valid task types            (User | Service | Manual)
  7. Data object integrity       (attached_to task must exist; valid type field)
  8. Annotation completeness     (text required; attached_to task must exist)
  9. Message flow completeness   (source & target required)
 10. Message flow cross-pool     (message flows must cross pool boundaries)

The function signature is unchanged — returns (is_valid, [error, ...]).
"""
from typing import Any


_VALID_TASK_TYPES  = {"User", "Service", "Manual"}
_VALID_GW_TYPES    = {"Exclusive", "Parallel", "Inclusive"}
_VALID_EVENT_TYPES = {"Start", "Intermediate", "End"}
_VALID_SUBTYPES: dict[str, set[str | None]] = {
    "Start":        {"message", "timer", "none", None},
    "Intermediate": {"message", "timer", "none", None},
    "End":          {"message", "none",  None},
}


def validate_bpmn_structure(process_json: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    BPMN 2.0 compliance validation.

    Returns
    ───────
    (is_valid: bool, errors: list[str])
    """
    errors: list[str] = []

    # ── 1 + 2: Start / End event counts ──────────────────────────────────────
    events       = process_json.get("events", [])
    start_events = [e for e in events if e.get("type") == "Start"]
    end_events   = [e for e in events if e.get("type") == "End"]

    if len(start_events) != 1:
        errors.append(
            f"Expected exactly 1 Start event (thin-border circle), "
            f"found {len(start_events)}"
        )
    if len(end_events) < 1:
        errors.append(
            "No End event defined — every process must terminate (thick-border circle)"
        )

    # ── 3: Event subtype validation ───────────────────────────────────────────
    for ev in events:
        etype   = ev.get("type")
        subtype = ev.get("subtype")
        name    = ev.get("name", "<unnamed>")

        if etype not in _VALID_EVENT_TYPES:
            errors.append(
                f"Event '{name}' has invalid type '{etype}' "
                f"(must be Start, Intermediate, or End)"
            )
            continue  # no point checking subtype if type is wrong

        allowed = _VALID_SUBTYPES.get(etype, set())
        if subtype not in allowed:
            errors.append(
                f"Event '{name}' (type {etype}) has invalid subtype '{subtype}' "
                f"(allowed: {', '.join(str(s) for s in allowed if s)})"
            )

    # Also validate entries in the separate message_events list
    for me in process_json.get("message_events", []) or []:
        me_type    = me.get("type")
        me_subtype = me.get("subtype")
        me_name    = me.get("name", "<unnamed>")

        if me_type not in _VALID_EVENT_TYPES:
            errors.append(
                f"Message event '{me_name}' has invalid type '{me_type}'"
            )
        if me_subtype != "message":
            errors.append(
                f"Message event '{me_name}' must have subtype 'message', "
                f"got '{me_subtype}'"
            )
        if not me_name or me_name == "<unnamed>":
            errors.append("A message event is missing its name")

    # ── 4 + 5: Gateway type and branch validation ─────────────────────────────
    for gw in process_json.get("gateways", []):
        name    = gw.get("name", "<unnamed>")
        gw_type = gw.get("type")

        if gw_type not in _VALID_GW_TYPES:
            errors.append(
                f"Gateway '{name}' has invalid type '{gw_type}' "
                f"(must be Exclusive, Parallel, or Inclusive)"
            )
        if not gw.get("yes_branch") and not gw.get("no_branch"):
            errors.append(
                f"Gateway '{name}' has no branches — "
                "define at least yes_branch or no_branch"
            )

    # ── 6: Task type validation + collect task name set ───────────────────────
    all_task_names: set[str] = set()

    for pool in process_json.get("pools", []):
        pool_name = pool.get("name")
        if not pool_name:
            errors.append("A pool is missing its name")

        for lane in pool.get("lanes", []):
            lane_name = lane.get("name")
            if not lane_name:
                errors.append(
                    f"A lane in pool '{pool_name or '?'}' is missing its name"
                )

            for task in lane.get("tasks", []):
                task_name = task.get("name")
                if not task_name:
                    errors.append(
                        f"A task in lane '{lane_name or '?'}' is missing its name"
                    )
                else:
                    all_task_names.add(task_name)

                task_type = task.get("type")
                if task_type and task_type not in _VALID_TASK_TYPES:
                    errors.append(
                        f"Task '{task_name or '?'}' has invalid type '{task_type}' "
                        f"(must be User, Service, or Manual)"
                    )

    # ── 7: Data object integrity ──────────────────────────────────────────────
    for do in process_json.get("data_objects", []) or []:
        do_name  = do.get("name")
        do_type  = do.get("type")
        attached = do.get("attached_to")

        if not do_name:
            errors.append("A data object is missing its name")
        if do_type and do_type not in ("input", "output"):
            errors.append(
                f"Data object '{do_name or '?'}' has invalid type '{do_type}' "
                "(must be 'input' or 'output')"
            )
        if attached and attached not in all_task_names:
            errors.append(
                f"Data object '{do_name or '?'}' is attached to unknown task "
                f"'{attached}'"
            )

    # ── 8: Annotation completeness ────────────────────────────────────────────
    for ann in process_json.get("annotations", []) or []:
        ann_text = ann.get("text")
        attached = ann.get("attached_to")

        if not ann_text:
            errors.append(
                f"Annotation attached to '{attached or '?'}' is missing text"
            )
        if attached and attached not in all_task_names:
            errors.append(
                f"Annotation refers to unknown task '{attached}'"
            )

    # ── 9 + 10: Message flow completeness and cross-pool check ───────────────
    # Build task → pool lookup
    task_to_pool: dict[str, str] = {
        t["name"]: pool.get("name", "")
        for pool in process_json.get("pools", [])
        for lane in pool.get("lanes", [])
        for t in lane.get("tasks", [])
        if t.get("name")
    }

    for mf in process_json.get("message_flows", []) or []:
        src = mf.get("source")
        tgt = mf.get("target")

        if not src or not tgt:
            errors.append("A message flow is missing source or target")
            continue

        # Cross-pool check (only meaningful when both endpoints resolve to pools)
        src_pool = task_to_pool.get(src)
        tgt_pool = task_to_pool.get(tgt)
        if src_pool and tgt_pool and src_pool == tgt_pool:
            errors.append(
                f"Message flow from '{src}' to '{tgt}' connects tasks in the "
                f"same pool '{src_pool}' — use a Sequence flow instead"
            )

    return len(errors) == 0, errors
