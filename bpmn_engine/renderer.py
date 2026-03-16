"""Engine Layer — SVG renderer for BPMN 2.0 diagrams.

Pipeline position:
    LLM → APIs → ML → **Engine (render)**

Supported BPMN elements
───────────────────────
Events     : Start  (thin green circle)
             End    (thick red circle + filled disc)
             Intermediate (double-circle, blue)
             Message Start/End (envelope icon inside circle)
Tasks      : User / Service / Manual — rounded rectangles with role icon
             Collapsed Sub-process   — task with ⊕ marker
Gateways   : Exclusive (X), Parallel (+), Inclusive (O) — orange diamonds
Connectors : Orthogonal (90-degree only) sequence flows with arrowheads
             Message flows (dashed inter-pool, open-circle source)
Swimlanes  : Pools (left blue strip) + Lanes (alternating colour bands)
Annotations: BPMN text-annotation bracket [ with dashed connector
Data objects: Document-icon shapes (input ◀ / output ▶)

BPMNLayout
──────────
Flow-graph-based layout with topological column assignment.
Each node gets a column from topological longest-path over the flow graph,
and a lane row from pool/lane membership (or inherited for events/gateways).

    layout = BPMNLayout(process_json)
    cx, cy = layout.node_pos["Task Name"]

Supports arbitrary pools, lanes, and sequential gateways.

Gateway routing
───────────────
  YES exits TOP    vertex of split diamond → orthogonal up then right
  NO  exits BOTTOM vertex of split diamond → orthogonal down then right
  Main flow traverses LEFT ↔ RIGHT vertices
"""
from __future__ import annotations

from collections import deque, defaultdict
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_TASK_W     = 120    # task rectangle width
_TASK_H     = 52     # task rectangle height
_COL_W      = 190    # column stride
_LANE_H     = 250    # lane height (accommodates same-lane branch offsets)
_TITLE_H    = 46     # title bar height
_GW_R       = 24     # gateway diamond half-size
_EV_R       = 20     # event circle radius
_POOL_LW    = 34     # pool label strip width
_LANE_LW    = 52     # lane label strip width
_PAD_L      = 70     # left padding within first column
_BRANCH_GAP = 90     # vertical px offset for yes/no in same lane

# Backward-compatibility aliases used by tests
_POOL_W  = _POOL_LW
_LANE_W  = _LANE_LW

_LANE_COLORS = ("#eef2fb", "#f5f7ff")


# ─────────────────────────────────────────────────────────────────────────────
# Topological column assignment (Kahn's BFS + DP longest-path)
# ─────────────────────────────────────────────────────────────────────────────

def _longest_path_cols(
    nodes: set[str],
    edges: list[tuple[str, str]],
) -> dict[str, int]:
    """Return column index for each node via Kahn's BFS + DP longest-path.
    Handles cycles gracefully (cycle nodes stay at their last assigned col)."""
    in_deg:  dict[str, int]       = {n: 0 for n in nodes}
    out_adj: dict[str, list[str]] = defaultdict(list)

    for s, t in edges:
        if s in in_deg and t in in_deg:
            in_deg[t] += 1
            out_adj[s].append(t)

    col   = {n: 0 for n in nodes}
    queue = deque(n for n in nodes if in_deg[n] == 0)

    while queue:
        node = queue.popleft()
        for nbr in out_adj[node]:
            col[nbr] = max(col[nbr], col[node] + 1)
            in_deg[nbr] -= 1
            if in_deg[nbr] == 0:
                queue.append(nbr)

    return col


# ─────────────────────────────────────────────────────────────────────────────
# BPMNLayout — flow-graph-based coordinate calculator
# ─────────────────────────────────────────────────────────────────────────────

class BPMNLayout:
    """
    Flow-graph-based BPMN layout engine.

    Column assignment : topological longest-path over sequence-flow graph.
    Row assignment    : pool/lane membership for tasks; inherited for
                        events and gateways via adjacent nodes in flows.

    When yes/no branches all share the same lane as their split gateway,
    a ±_BRANCH_GAP pixel offset is applied so they don't overlap.

    Attributes
    ----------
    node_pos   : dict[str, (int,int)]   (cx, cy) per node name
    node_col   : dict[str, int]         column index per node
    node_lane  : dict[str, int]         flat lane index per node
    flat_lanes : list[(pool_name, lane_name)]
    header_w   : int   x-offset before first diagram element
    width      : int   total SVG canvas width
    height     : int   total SVG canvas height
    """

    def __init__(self, process_json: dict[str, Any]) -> None:
        pools    = process_json.get("pools",    [])
        gateways = process_json.get("gateways", [])
        events   = process_json.get("events",   [])
        flows    = process_json.get("flows",    [])

        # ── Flat lane list + task-to-lane map ─────────────────────────────────
        self.flat_lanes: list[tuple[str, str]] = []
        self._task_to_flat: dict[str, int]     = {}

        for p in pools:
            for ln in p.get("lanes", []):
                fi = len(self.flat_lanes)
                self.flat_lanes.append(
                    (p.get("name", "Pool"), ln.get("name", "Lane"))
                )
                for t in ln.get("tasks", []):
                    self._task_to_flat[t["name"]] = fi

        # ── Collect all node names ────────────────────────────────────────────
        all_nodes: set[str] = set()
        for e in events:
            all_nodes.add(e["name"])
        all_nodes.update(self._task_to_flat)
        for gw in gateways:
            all_nodes.add(gw["name"])
            for t in gw.get("yes_branch", []) + gw.get("no_branch", []):
                all_nodes.add(t)

        # ── Directed sequence edges ───────────────────────────────────────────
        edges: list[tuple[str, str]] = []
        flow_pairs: set[tuple[str, str]] = set()

        for f in flows:
            if f.get("type") == "Message":
                continue
            s, t = f.get("source", ""), f.get("target", "")
            if s and t and s in all_nodes and t in all_nodes and s != t:
                edges.append((s, t))
                flow_pairs.add((s, t))

        # Fill gateway→branch edges absent from flows
        for gw in gateways:
            for t in gw.get("yes_branch", []) + gw.get("no_branch", []):
                if t in all_nodes and (gw["name"], t) not in flow_pairs:
                    edges.append((gw["name"], t))

        # ── Column assignment ─────────────────────────────────────────────────
        self.node_col = _longest_path_cols(all_nodes, edges)

        # ── Lane (row) assignment ─────────────────────────────────────────────
        in_nb:  dict[str, list[str]] = defaultdict(list)
        out_nb: dict[str, list[str]] = defaultdict(list)
        for s, t in edges:
            in_nb[t].append(s)
            out_nb[s].append(t)

        self.node_lane: dict[str, int] = dict(self._task_to_flat)
        n_flat = len(self.flat_lanes)

        # Gateways inherit lane from incoming node
        for gw in gateways:
            gn = gw["name"]
            if gn not in self.node_lane:
                for src in in_nb.get(gn, []):
                    if src in self.node_lane:
                        self.node_lane[gn] = self.node_lane[src]
                        break
            if gn not in self.node_lane:
                self.node_lane[gn] = 0

            gw_lane = self.node_lane[gn]
            for t in gw.get("yes_branch", []):
                if t not in self.node_lane:
                    self.node_lane[t] = (
                        max(0, gw_lane - 1) if n_flat > 1 else gw_lane
                    )
            for t in gw.get("no_branch", []):
                if t not in self.node_lane:
                    self.node_lane[t] = (
                        min(n_flat - 1, gw_lane + 1) if n_flat > 1 else gw_lane
                    )

        # Events inherit lane from connected nodes
        for e in events:
            en = e["name"]
            if en not in self.node_lane:
                for src in in_nb.get(en, []):
                    if src in self.node_lane:
                        self.node_lane[en] = self.node_lane[src]
                        break
            if en not in self.node_lane:
                for tgt in out_nb.get(en, []):
                    if tgt in self.node_lane:
                        self.node_lane[en] = self.node_lane[tgt]
                        break
            if en not in self.node_lane:
                self.node_lane[en] = 0

        # Catch-all
        for n in all_nodes:
            if n not in self.node_lane:
                self.node_lane[n] = 0

        # ── Header width ──────────────────────────────────────────────────────
        if n_flat == 0:
            self.header_w = 0
        elif n_flat == 1:
            self.header_w = _POOL_LW
        else:
            self.header_w = _POOL_LW + _LANE_LW

        # ── Base pixel positions ──────────────────────────────────────────────
        max_col = max(self.node_col.values(), default=0)
        n_vis   = max(1, n_flat)

        self.node_pos: dict[str, tuple[int, int]] = {}
        for name in all_nodes:
            col  = self.node_col.get(name, 0)
            lane = self.node_lane.get(name, 0)
            cx   = self.header_w + _PAD_L + col * _COL_W
            cy   = _TITLE_H + lane * _LANE_H + _LANE_H // 2
            self.node_pos[name] = (cx, cy)

        # ── Same-lane branch vertical offset ─────────────────────────────────
        # When yes AND no branches all land in the same lane as the gateway,
        # offset them vertically so they don't overlap.
        for gw in gateways:
            gn    = gw["name"]
            yes_b = gw.get("yes_branch", [])
            no_b  = gw.get("no_branch",  [])
            if not yes_b or not no_b:
                continue
            gw_lane   = self.node_lane.get(gn, 0)
            yes_lanes = {self.node_lane.get(t, 0) for t in yes_b}
            no_lanes  = {self.node_lane.get(t, 0) for t in no_b}
            if (yes_lanes | no_lanes) == {gw_lane}:
                for t in yes_b:
                    if t in self.node_pos:
                        cx, cy = self.node_pos[t]
                        self.node_pos[t] = (cx, cy - _BRANCH_GAP)
                for t in no_b:
                    if t in self.node_pos:
                        cx, cy = self.node_pos[t]
                        self.node_pos[t] = (cx, cy + _BRANCH_GAP)

        self.width  = self.header_w + _PAD_L + (max_col + 1) * _COL_W + 80
        self.height = _TITLE_H + n_vis * _LANE_H + 30


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, max_len: int = 14, max_lines: int = 2) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) <= max_len:
            cur = cand
        else:
            if cur:
                lines.append(cur)
                if len(lines) >= max_lines:
                    break
            cur = w[:max_len]
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines or [str(text)[:max_len]]


def _esc(t: str) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _task_icon(task_type: str) -> str:
    return {"Service": "⚙", "Manual": "✍", "User": "👤"}.get(task_type, "")


def _draw_task(
    out: list[str],
    cx: int, cy: int,
    task_w: int, task_h: int,
    label: str,
    task_type: str      = "User",
    task_color: str     = "#ffffff",
    is_subprocess: bool = False,
) -> None:
    lx     = cx - task_w // 2
    border = {"Service": "#1565c0", "Manual": "#6d4c41"}.get(task_type, "#2c3e50")
    out.append(
        f'<rect x="{lx}" y="{cy - task_h // 2}" width="{task_w}" height="{task_h}"'
        f' rx="5" fill="{task_color}" stroke="{border}" stroke-width="1.5"/>'
    )
    if is_subprocess:
        mx, my = cx, cy + task_h // 2 - 8
        out += [
            f'<circle cx="{mx}" cy="{my}" r="7" fill="#fff"'
            f' stroke="{border}" stroke-width="1.2"/>',
            f'<text x="{mx}" y="{my + 4}" text-anchor="middle"'
            f' font-size="10" fill="{border}">+</text>',
        ]
    icon = _task_icon(task_type)
    if icon:
        out.append(
            f'<text x="{lx + 6}" y="{cy - task_h // 2 + 12}"'
            f' font-size="10" fill="{border}" opacity="0.7">{icon}</text>'
        )
    for j, part in enumerate(_wrap(_esc(label))):
        out.append(
            f'<text x="{cx}" y="{cy - 5 + j * 14}" text-anchor="middle"'
            f' font-size="10" fill="#111">{part}</text>'
        )


def _draw_envelope_icon(out: list[str], cx: int, cy: int, color: str) -> None:
    w, h = 14, 10
    lx   = cx - w // 2
    ty   = cy - h // 2
    out += [
        f'<rect x="{lx}" y="{ty}" width="{w}" height="{h}"'
        f' rx="1" fill="none" stroke="{color}" stroke-width="1.2"/>',
        f'<polyline points="{lx},{ty} {cx},{cy + 2} {lx + w},{ty}"'
        f' fill="none" stroke="{color}" stroke-width="1.2"/>',
    ]


def _draw_annotation(
    out: list[str],
    text: str,
    ann_x: int, ann_y: int,
    task_cx: int, task_cy: int,
) -> None:
    aw = max(70, min(len(text) * 6 + 16, 130))
    ah = 36
    out.append(
        f'<line x1="{task_cx}" y1="{task_cy}"'
        f' x2="{ann_x + aw // 2}" y2="{ann_y + ah}"'
        f' stroke="#999" stroke-width="1" stroke-dasharray="4,3"/>'
    )
    out += [
        f'<polyline points="{ann_x + 8},{ann_y} {ann_x},{ann_y}'
        f' {ann_x},{ann_y + ah} {ann_x + 8},{ann_y + ah}"'
        f' stroke="#555" stroke-width="1.4" fill="none"/>',
        f'<rect x="{ann_x}" y="{ann_y}" width="{aw}" height="{ah}"'
        f' rx="2" fill="#fffde7" fill-opacity="0.85" stroke="none"/>',
    ]
    for j, part in enumerate(_wrap(_esc(text), 16, 2)):
        out.append(
            f'<text x="{ann_x + 12}" y="{ann_y + 14 + j * 13}"'
            f' font-size="9" fill="#444">{part}</text>'
        )


def _draw_data_object(
    out: list[str],
    name: str,
    obj_type: str,
    task_cx: int, task_cy: int,
    obj_x: int, obj_y: int,
) -> None:
    dw, dh, fold = 26, 32, 8
    border = "#1565c0" if obj_type == "input"  else "#e67e22"
    fill   = "#e3f2fd" if obj_type == "input"  else "#fff3e0"
    out.append(
        f'<path d="M{obj_x},{obj_y} L{obj_x + dw - fold},{obj_y}'
        f' L{obj_x + dw},{obj_y + fold} L{obj_x + dw},{obj_y + dh}'
        f' L{obj_x},{obj_y + dh} Z"'
        f' fill="{fill}" stroke="{border}" stroke-width="1.4"/>'
    )
    out.append(
        f'<path d="M{obj_x + dw - fold},{obj_y}'
        f' L{obj_x + dw - fold},{obj_y + fold} L{obj_x + dw},{obj_y + fold}"'
        f' fill="none" stroke="{border}" stroke-width="1.2"/>'
    )
    arrow = "▶" if obj_type == "output" else "◀"
    out.append(
        f'<text x="{obj_x + dw // 2}" y="{obj_y + dh // 2 + 5}"'
        f' text-anchor="middle" font-size="9" fill="{border}">{arrow}</text>'
    )
    for j, part in enumerate(_wrap(_esc(name), 10, 2)):
        out.append(
            f'<text x="{obj_x + dw // 2}" y="{obj_y + dh + 10 + j * 11}"'
            f' text-anchor="middle" font-size="8" fill="{border}">{part}</text>'
        )
    line_y = obj_y + dh if obj_type == "output" else obj_y
    out.append(
        f'<line x1="{obj_x + dw // 2}" y1="{line_y}"'
        f' x2="{task_cx}" y2="{task_cy}"'
        f' stroke="{border}" stroke-width="1" stroke-dasharray="3,2"/>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Edge routing
# ─────────────────────────────────────────────────────────────────────────────

def _ortho_path(x1: int, y1: int, x2: int, y2: int) -> str:
    """Z-shape orthogonal path: H to midpoint → V → H.
    Used for task→task, task→gateway, and branch→End edges."""
    if abs(y1 - y2) < 4:
        return f"M{x1},{y1} H{x2}"
    mx = (x1 + x2) // 2
    return f"M{x1},{y1} H{mx} V{y2} H{x2}"


def _ortho_path_vfirst(x1: int, y1: int, x2: int, y2: int) -> str:
    """Γ-shape orthogonal path: V first → then H.
    Used for gateway YES/NO branch exits so the arrow visibly exits
    the diamond UP (yes) or DOWN (no) before turning right."""
    if abs(y1 - y2) < 4:
        return f"M{x1},{y1} H{x2}"
    return f"M{x1},{y1} V{y2} H{x2}"


def _draw_edge(
    out:       list[str],
    s:         str,
    t:         str,
    node_pos:  dict[str, tuple[int, int]],
    node_type: dict[str, tuple],
    gw_yes:    dict[str, set[str]],
    gw_no:     dict[str, set[str]],
) -> None:
    """Draw one sequence-flow edge with orthogonal routing."""
    sx, sy = node_pos[s]
    tx, ty = node_pos[t]
    s_info = node_type.get(s, ("task",))
    t_info = node_type.get(t, ("task",))

    # Determine if this is a YES or NO branch exit from a gateway
    is_yes_exit = s_info[0] == "gateway" and t in gw_yes.get(s, set())
    is_no_exit  = s_info[0] == "gateway" and t in gw_no.get(s, set())

    # Source exit point
    if s_info[0] == "gateway":
        if is_yes_exit:
            ep_sx, ep_sy = sx, sy - _GW_R      # top vertex (YES → up)
        elif is_no_exit:
            ep_sx, ep_sy = sx, sy + _GW_R      # bottom vertex (NO → down)
        else:
            ep_sx, ep_sy = sx + _GW_R, sy      # right vertex (pass-through)
    elif s_info[0] == "event":
        ep_sx, ep_sy = sx + _EV_R, sy
    else:
        ep_sx, ep_sy = sx + _TASK_W // 2, sy

    # Target entry point
    if t_info[0] == "gateway":
        ep_tx, ep_ty = tx - _GW_R, ty
    elif t_info[0] == "event":
        ep_tx, ep_ty = tx - _EV_R, ty
    else:
        ep_tx, ep_ty = tx - _TASK_W // 2, ty

    # YES/NO branch exits: V-first so arrow visibly leaves diamond up/down
    # All other edges: Z-shape (right to midpoint then vertical then right)
    if is_yes_exit or is_no_exit:
        path = _ortho_path_vfirst(ep_sx, ep_sy, ep_tx, ep_ty)
    else:
        path = _ortho_path(ep_sx, ep_sy, ep_tx, ep_ty)

    out.append(
        f'<path d="{path}" stroke="#34495e" stroke-width="1.8" fill="none"'
        f' marker-end="url(#arr)"/>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Swimlane renderer (multi-pool, multi-lane)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_swimlanes(
    out:        list[str],
    flat_lanes: list[tuple[str, str]],
    header_w:   int,
    width:      int,
    height:     int,  # noqa: ARG001 (reserved for future use)
) -> None:
    """Draw pool + lane background bands."""
    if not flat_lanes:
        return

    n_flat         = len(flat_lanes)
    has_multi_lane = n_flat > 1

    # Group consecutive lanes by pool name
    pool_groups: list[tuple[str, list[str], int]] = []
    for fi, (pool_name, lane_name) in enumerate(flat_lanes):
        if pool_groups and pool_groups[-1][0] == pool_name:
            pool_groups[-1][1].append(lane_name)
        else:
            pool_groups.append((pool_name, [lane_name], fi))

    for pool_name, lane_names, start_fi in pool_groups:
        n_pl    = len(lane_names)
        p_top   = _TITLE_H + start_fi * _LANE_H
        p_h     = n_pl * _LANE_H
        p_mid_y = p_top + p_h // 2

        out += [
            f'<rect x="0" y="{p_top}" width="{_POOL_LW}" height="{p_h}"'
            f' fill="#dce4f5" stroke="#4a6fa5" stroke-width="1.5"/>',
            f'<text x="{_POOL_LW // 2}" y="{p_mid_y}" text-anchor="middle"'
            f' font-size="11" font-weight="700" fill="#2c3e6b"'
            f' transform="rotate(-90 {_POOL_LW // 2} {p_mid_y})">'
            f'{_esc(pool_name)}</text>',
        ]

        for i, lane_name in enumerate(lane_names):
            l_top = p_top + i * _LANE_H
            bg    = _LANE_COLORS[(start_fi + i) % 2]
            out.append(
                f'<rect x="{header_w}" y="{l_top}" width="{width - header_w}"'
                f' height="{_LANE_H}" fill="{bg}" stroke="#b0bde8" stroke-width="0.8"/>'
            )
            if has_multi_lane:
                l_mid_y = l_top + _LANE_H // 2
                out += [
                    f'<rect x="{_POOL_LW}" y="{l_top}" width="{_LANE_LW}"'
                    f' height="{_LANE_H}" fill="#e4ecf9" stroke="#4a6fa5" stroke-width="1"/>',
                    f'<text x="{_POOL_LW + _LANE_LW // 2}" y="{l_mid_y}"'
                    f' text-anchor="middle" font-size="9" fill="#2c3e6b"'
                    f' transform="rotate(-90 {_POOL_LW + _LANE_LW // 2} {l_mid_y})">'
                    f'{_esc(lane_name)}</text>',
                ]

    out.append(
        f'<rect x="0" y="{_TITLE_H}" width="{width}" height="{n_flat * _LANE_H}"'
        f' fill="none" stroke="#4a6fa5" stroke-width="1.5"/>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_diagram(process_json: dict[str, Any]) -> str:
    """Render a full BPMN SVG from rich process JSON. Returns an SVG string."""

    pools          = process_json.get("pools",          [])
    gateways       = process_json.get("gateways",       [])
    events         = process_json.get("events",         [])
    flows          = process_json.get("flows",          [])
    annotations    = process_json.get("annotations",    [])
    data_objects   = process_json.get("data_objects",   [])
    message_flows  = process_json.get("message_flows",  [])
    message_events = process_json.get("message_events", [])
    colors         = process_json.get("_colors",        {})
    title          = process_json.get("title",          "Process Flow")

    task_color    = colors.get("task",    "#ffffff")
    gateway_color = colors.get("gateway", "#ffffff")
    event_color   = colors.get("event",   "#e8f5e9")

    # ── Layout ────────────────────────────────────────────────────────────────
    layout   = BPMNLayout(process_json)
    node_pos = layout.node_pos
    width    = layout.width
    height   = layout.height
    header_w = layout.header_w

    # ── Node type lookup ──────────────────────────────────────────────────────
    node_type: dict[str, tuple] = {}
    for e in events:
        node_type[e["name"]] = ("event", e.get("type", "Start"), e.get("subtype"))
    for p in pools:
        for ln in p.get("lanes", []):
            for t in ln.get("tasks", []):
                node_type[t["name"]] = ("task", t.get("type", "User"))
    for gw in gateways:
        node_type[gw["name"]] = ("gateway", gw.get("type", "Exclusive"))

    # Gateway branch sets
    gw_yes: dict[str, set[str]] = {}
    gw_no:  dict[str, set[str]] = {}
    for gw in gateways:
        gw_yes[gw["name"]] = set(gw.get("yes_branch", []))
        gw_no[gw["name"]]  = set(gw.get("no_branch",  []))
        for t in gw.get("yes_branch", []) + gw.get("no_branch", []):
            if t not in node_type:
                node_type[t] = ("task", "User")

    sp_names: set[str] = {
        sp["name"]
        for p in pools for ln in p.get("lanes", [])
        for sp in ln.get("subprocesses", [])
    }

    # Message event flags
    msg_start = any(
        me.get("type") == "Start" and me.get("subtype") == "message"
        for me in (message_events or [])
    )
    msg_end = any(
        me.get("type") == "End" and me.get("subtype") == "message"
        for me in (message_events or [])
    )

    # ── SVG open + defs ───────────────────────────────────────────────────────
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arr" markerWidth="8" markerHeight="6"'
        ' refX="7" refY="3" orient="auto">',
        '<path d="M0,0 L8,3 L0,6 z" fill="#34495e"/>',
        "</marker>",
        '<marker id="arr_msg" markerWidth="8" markerHeight="6"'
        ' refX="7" refY="3" orient="auto">',
        '<path d="M0,0 L8,3 L0,6" stroke="#1565c0" stroke-width="1" fill="none"/>',
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="6"'
        f' fill="#fafaf7" stroke="#d0d0c8" stroke-width="1.5"/>',
    ]

    # ── Swimlane backgrounds ──────────────────────────────────────────────────
    if layout.flat_lanes:
        _draw_swimlanes(out, layout.flat_lanes, header_w, width, height)

    # ── Title bar ─────────────────────────────────────────────────────────────
    out += [
        f'<rect x="0" y="0" width="{width}" height="{_TITLE_H}" rx="6" fill="#2c3e50"/>',
        f'<text x="{18 + header_w}" y="30" font-size="16" font-weight="700"'
        f' fill="#ecf0f1" font-family="Georgia, serif">{_esc(title)}</text>',
    ]

    # ── Sequence flow edges ───────────────────────────────────────────────────
    drawn_edges: set[tuple[str, str]] = set()

    for f in flows:
        if f.get("type") == "Message":
            continue
        s, t = f.get("source", ""), f.get("target", "")
        if s not in node_pos or t not in node_pos or (s, t) in drawn_edges:
            continue
        drawn_edges.add((s, t))
        _draw_edge(out, s, t, node_pos, node_type, gw_yes, gw_no)

    # Fill gateway→branch edges absent from flows
    for gw in gateways:
        gn = gw["name"]
        for t in gw.get("yes_branch", []) + gw.get("no_branch", []):
            if t in node_pos and (gn, t) not in drawn_edges:
                drawn_edges.add((gn, t))
                _draw_edge(out, gn, t, node_pos, node_type, gw_yes, gw_no)

    # ── BPMN nodes ────────────────────────────────────────────────────────────
    for name, (cx, cy) in node_pos.items():
        ntype = node_type.get(name)
        if ntype is None:
            continue
        kind = ntype[0]

        # ── Event ─────────────────────────────────────────────────────────────
        if kind == "event":
            ev_type = ntype[1] if len(ntype) > 1 else "Start"
            subtype = ntype[2] if len(ntype) > 2 else None

            if ev_type == "Start":
                out += [
                    f'<circle cx="{cx}" cy="{cy}" r="{_EV_R}"'
                    f' fill="#ffffff" stroke="#27ae60" stroke-width="2"/>',
                    f'<text x="{cx}" y="{cy + 5}" text-anchor="middle"'
                    f' font-size="11" font-weight="700" fill="#27ae60">S</text>',
                ]
                if msg_start or subtype == "message":
                    _draw_envelope_icon(out, cx, cy, "#27ae60")

            elif ev_type == "End":
                out += [
                    f'<circle cx="{cx}" cy="{cy}" r="22"'
                    f' fill="#ffffff" stroke="#e74c3c" stroke-width="4"/>',
                    f'<circle cx="{cx}" cy="{cy}" r="14"'
                    f' fill="#e74c3c" stroke="none"/>',
                ]
                if msg_end or subtype == "message":
                    _draw_envelope_icon(out, cx, cy, "#ffffff")

            else:  # Intermediate
                out += [
                    f'<circle cx="{cx}" cy="{cy}" r="16"'
                    f' fill="{event_color}" stroke="#3498db" stroke-width="1.5"/>',
                    f'<circle cx="{cx}" cy="{cy}" r="11"'
                    f' fill="none" stroke="#3498db" stroke-width="1"/>',
                ]
                if subtype == "message":
                    _draw_envelope_icon(out, cx, cy, "#3498db")

            for j, part in enumerate(_wrap(_esc(name), 12, 2)):
                out.append(
                    f'<text x="{cx}" y="{cy + 36 + j * 13}" text-anchor="middle"'
                    f' font-size="9" fill="#555">{part}</text>'
                )

        # ── Gateway ───────────────────────────────────────────────────────────
        elif kind == "gateway":
            gw_type = ntype[1] if len(ntype) > 1 else "Exclusive"
            gw_sym  = {"Parallel": "+", "Inclusive": "O"}.get(gw_type, "X")
            out += [
                f'<polygon points="{cx},{cy - _GW_R} {cx + _GW_R},{cy}'
                f' {cx},{cy + _GW_R} {cx - _GW_R},{cy}"'
                f' fill="{gateway_color}" stroke="#e67e22" stroke-width="2"/>',
                f'<text x="{cx}" y="{cy + 6}" text-anchor="middle"'
                f' font-size="14" font-weight="700" fill="#e67e22">{gw_sym}</text>',
            ]
            # Gateway name — rendered below the NO label to avoid overlap
            # YES / NO branch labels near gateway exit arrows
            yes_set = gw_yes.get(name, set())
            no_set  = gw_no.get(name, set())
            if yes_set:
                out += [
                    f'<rect x="{cx + 3}" y="{cy - _GW_R - 22}"'
                    f' width="22" height="14" rx="4" fill="#e6f9ef" stroke="none"/>',
                    f'<text x="{cx + 14}" y="{cy - _GW_R - 11}"'
                    f' text-anchor="middle" font-size="9" font-weight="700"'
                    f' fill="#27ae60">yes</text>',
                ]
            if no_set:
                out += [
                    f'<rect x="{cx + 3}" y="{cy + _GW_R + 8}"'
                    f' width="18" height="14" rx="4" fill="#fdecea" stroke="none"/>',
                    f'<text x="{cx + 12}" y="{cy + _GW_R + 19}"'
                    f' text-anchor="middle" font-size="9" font-weight="700"'
                    f' fill="#e74c3c">no</text>',
                ]
            # Name below no-label (or below diamond if no branches)
            name_y_start = cy + _GW_R + 28 if no_set else cy + _GW_R + 13
            for j, part in enumerate(_wrap(_esc(name), 12, 2)):
                out.append(
                    f'<text x="{cx}" y="{name_y_start + j * 11}"'
                    f' text-anchor="middle" font-size="8" fill="#555">{part}</text>'
                )

        # ── Task ──────────────────────────────────────────────────────────────
        elif kind == "task":
            task_type = ntype[1] if len(ntype) > 1 else "User"
            _draw_task(
                out, cx, cy, _TASK_W, _TASK_H, name,
                task_type, task_color, name in sp_names,
            )

    # ── Message flows (dashed inter-pool lines) ────────────────────────────────
    for mf in message_flows or []:
        src, tgt = mf.get("source", ""), mf.get("target", "")
        if src in node_pos and tgt in node_pos:
            sx, sy = node_pos[src]
            tx, ty = node_pos[tgt]
            out += [
                f'<circle cx="{sx}" cy="{sy - _TASK_H // 2 - 6}" r="5"'
                f' fill="#fff" stroke="#1565c0" stroke-width="1.5"/>',
                f'<line x1="{sx}" y1="{sy - _TASK_H // 2 - 11}"'
                f' x2="{tx}" y2="{ty + _TASK_H // 2 + 11}"'
                f' stroke="#1565c0" stroke-width="1.2" stroke-dasharray="5,3"'
                f' marker-end="url(#arr_msg)"/>',
            ]

    # ── Text annotations ──────────────────────────────────────────────────────
    ann_offset = 0
    for ann in annotations or []:
        attached = ann.get("attached_to", "")
        text     = ann.get("text", "")
        if not text:
            continue
        if attached in node_pos:
            tx, ty = node_pos[attached]
            # Preferred: above the task
            raw_y = ty - _TASK_H // 2 - 60 - ann_offset
            if raw_y >= _TITLE_H + 10:
                # Enough room above — place above task
                _draw_annotation(out, text, tx - 10, raw_y, tx, ty - _TASK_H // 2)
            else:
                # Too close to title bar — place to the right of the task
                _draw_annotation(
                    out, text,
                    tx + _TASK_W // 2 + 14, ty - 20 + ann_offset * 6,
                    tx + _TASK_W // 2, ty,
                )
        else:
            _draw_annotation(
                out, text,
                header_w + ann_offset, _TITLE_H + 8,
                header_w + _PAD_L, _TITLE_H + _LANE_H // 2,
            )
        ann_offset += 14

    # ── Data objects ──────────────────────────────────────────────────────────
    for do in data_objects or []:
        attached = do.get("attached_to", "")
        if attached in node_pos:
            tx, ty = node_pos[attached]
            _draw_data_object(
                out,
                do.get("name",  "Data"),
                do.get("type",  "output"),
                tx, ty + _TASK_H // 2,
                tx - 13, ty + _TASK_H // 2 + 18,
            )

    out.append("</svg>")
    return "".join(out)
