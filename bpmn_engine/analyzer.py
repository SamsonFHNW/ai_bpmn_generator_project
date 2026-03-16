"""ML Layer — Lean Muda Analysis with graph-theoretic waste detection.

Pipeline position:
    LLM → APIs → **ML (analyze)** → Engine

ProcessGraph
────────────
A directed graph built from BPMN flows.  Provides:
  - topological_order()     Kahn's BFS algorithm
  - longest_path_length()   DP over topological order (critical path)
  - bottleneck_nodes()      high in-degree convergence points
  - isolated_nodes()        tasks with degree zero (connectivity gaps)

Muda categories detected
────────────────────────
  Waiting Time     : critical path too long / sequential tasks with no parallel gw
  Over-processing  : excessive manual tasks
  Transport        : cross-pool message flow overuse
  Bottleneck       : high-convergence nodes in the flow graph
  Redundant Gateway: exclusive gateways with identical branches
  Defect           : isolated / disconnected nodes
"""
from collections import deque
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# ProcessGraph — directed graph for algorithmic analysis
# ─────────────────────────────────────────────────────────────────────────────

class ProcessGraph:
    """
    Lightweight directed graph for BPMN process analysis.

    Built from the flows list in a BPMN JSON dict.  All node names that
    appear in the graph but not in `edges` still exist as isolated nodes.
    """

    def __init__(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]],
    ) -> None:
        # Deduplicate while preserving order (Python 3.7+ dict guarantee)
        self.nodes: list[str]           = list(dict.fromkeys(nodes))
        self._node_set: set[str]        = set(self.nodes)
        self._out: dict[str, list[str]] = {n: [] for n in self.nodes}
        self._in:  dict[str, list[str]] = {n: [] for n in self.nodes}

        for src, tgt in edges:
            if src in self._node_set and tgt in self._node_set:
                self._out[src].append(tgt)
                self._in[tgt].append(src)

    def in_degree(self, node: str) -> int:
        return len(self._in.get(node, []))

    def out_degree(self, node: str) -> int:
        return len(self._out.get(node, []))

    def degree_centrality(self, node: str) -> int:
        return self.in_degree(node) + self.out_degree(node)

    def topological_order(self) -> list[str]:
        """
        Return nodes in topological order via Kahn's BFS algorithm.

        Gracefully handles cycles — returns the partial order found before
        the cycle is encountered, no exception is raised.
        """
        in_deg = {n: self.in_degree(n) for n in self.nodes}
        queue  = deque(sorted(n for n in self.nodes if in_deg[n] == 0))
        result: list[str] = []

        while queue:
            n = queue.popleft()
            result.append(n)
            for succ in sorted(self._out.get(n, [])):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        return result

    def longest_path_length(self) -> int:
        """
        Critical path — dynamic programming over topological order.

        Returns the number of edges on the longest path (0 for a single node).
        """
        order = self.topological_order()
        dist  = {n: 0 for n in self.nodes}

        for n in order:
            for succ in self._out.get(n, []):
                dist[succ] = max(dist[succ], dist[n] + 1)

        return max(dist.values(), default=0)

    def bottleneck_nodes(self, threshold: int = 3) -> list[str]:
        """
        Return nodes whose in-degree meets or exceeds `threshold`.

        High-convergence join points that serialise many parallel paths.
        """
        return [n for n in self.nodes if self.in_degree(n) >= threshold]

    def isolated_nodes(self) -> list[str]:
        """Return nodes with no incoming or outgoing edges (modelling errors)."""
        return [
            n for n in self.nodes
            if self.in_degree(n) == 0 and self.out_degree(n) == 0
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _collect_node_names(process_json: dict[str, Any]) -> list[str]:
    """Gather every named node: events + tasks + gateway names + branch tasks."""
    names: list[str] = []

    for ev in process_json.get("events", []):
        if ev.get("name"):
            names.append(ev["name"])

    for pool in process_json.get("pools", []):
        for lane in pool.get("lanes", []):
            names.extend(t["name"] for t in lane.get("tasks", []) if t.get("name"))

    for gw in process_json.get("gateways", []):
        if gw.get("name"):
            names.append(gw["name"])
        names.extend(gw.get("yes_branch", []))
        names.extend(gw.get("no_branch",  []))

    return names


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_process_health(process_json: dict[str, Any]) -> dict[str, Any]:
    """
    ML-layer Lean Muda health analysis.

    Builds a ProcessGraph from BPMN flows and applies both graph-theoretic
    metrics and heuristic Muda rules to detect and score waste.

    Returns
    ───────
    score                float   0.0–1.0 (1.0 = no waste detected)
    health               str     "good" | "needs_improvement" | "critical"
    waiting_time_flag    bool
    over_processing_flag bool
    issues               list[dict]  each has: type, description, severity
    task_count           int
    manual_task_count    int
    graph_metrics        dict    critical_path, bottleneck_nodes,
                                 isolated_nodes, node_count, edge_count,
                                 parallel_gateways
    """
    issues: list[dict] = []
    score  = 1.0

    # ── Collect elements ──────────────────────────────────────────────────────
    all_tasks: list[dict] = [
        t
        for pool in process_json.get("pools", [])
        for lane in pool.get("lanes", [])
        for t in lane.get("tasks", [])
    ]
    manual_tasks = [t for t in all_tasks if t.get("type") == "Manual"]
    gateways     = process_json.get("gateways", [])
    flows        = process_json.get("flows",    [])
    pools        = process_json.get("pools",    [])

    # ── Build process graph ───────────────────────────────────────────────────
    all_node_names = _collect_node_names(process_json)
    edge_pairs: list[tuple[str, str]] = [
        (f["source"], f["target"])
        for f in flows
        if f.get("source") and f.get("target")
    ]
    graph = ProcessGraph(all_node_names, edge_pairs)

    critical_path = graph.longest_path_length()
    bottlenecks   = graph.bottleneck_nodes(threshold=3)
    isolated      = graph.isolated_nodes()
    parallel_gws  = [gw for gw in gateways if gw.get("type") == "Parallel"]

    # ── Graph: critical path length ───────────────────────────────────────────
    if critical_path > 8:
        issues.append({
            "type":        "Waiting Time",
            "description": (
                f"Critical path spans {critical_path} hops. "
                "Introduce Parallel gateways to run independent steps concurrently."
            ),
            "severity": "high",
        })
        score -= 0.20
    elif len(all_tasks) > 4 and not parallel_gws:
        issues.append({
            "type":        "Waiting Time",
            "description": (
                f"{len(all_tasks)} sequential tasks with no Parallel gateway. "
                "Consider parallelising independent steps to reduce cycle time."
            ),
            "severity": "medium",
        })
        score -= 0.15

    # ── Graph: bottleneck nodes ───────────────────────────────────────────────
    if bottlenecks:
        issues.append({
            "type":        "Bottleneck",
            "description": (
                f"High-convergence nodes: {', '.join(bottlenecks[:3])}. "
                "These points serialise many paths — decompose or automate them."
            ),
            "severity": "medium",
        })
        score -= 0.10

    # ── Graph: isolated / disconnected nodes ──────────────────────────────────
    if isolated:
        issues.append({
            "type":        "Defect",
            "description": (
                f"{len(isolated)} task(s) not connected by any flow: "
                f"{', '.join(isolated[:3])}. "
                "Add Sequence flows to integrate them into the process."
            ),
            "severity": "medium",
        })
        score -= 0.05

    # ── Heuristic: over-processing (manual tasks) ─────────────────────────────
    if len(manual_tasks) > 3:
        issues.append({
            "type":        "Over-processing",
            "description": (
                f"{len(manual_tasks)} manual tasks detected. "
                "Automate repetitive steps (data entry, approvals) "
                "to eliminate unnecessary human effort."
            ),
            "severity": "high",
        })
        score -= 0.20

    # ── Heuristic: redundant exclusive gateways ───────────────────────────────
    for gw in gateways:
        if (
            gw.get("type") == "Exclusive"
            and gw.get("yes_branch") and gw.get("no_branch")
            and set(gw["yes_branch"]) == set(gw["no_branch"])
        ):
            issues.append({
                "type":        "Redundant Gateway",
                "description": (
                    f"Gateway '{gw.get('name', '?')}' leads to identical outcomes "
                    "on both branches — remove this gateway."
                ),
                "severity": "medium",
            })
            score -= 0.10
            break  # report once

    # ── Heuristic: transport (cross-pool message overuse) ─────────────────────
    if len(pools) > 1:
        msg_flows = [f for f in flows if f.get("type") == "Message"]
        if len(msg_flows) > 4:
            issues.append({
                "type":        "Transport",
                "description": (
                    f"{len(msg_flows)} cross-pool message flows. "
                    "Excessive handoffs increase latency; consolidate responsibilities."
                ),
                "severity": "low",
            })
            score -= 0.10

    score = max(round(score, 2), 0.0)

    return {
        "score":                score,
        "health":               (
            "good"             if score >= 0.75 else
            "needs_improvement" if score >= 0.50 else
            "critical"
        ),
        "waiting_time_flag":    any(i["type"] == "Waiting Time"    for i in issues),
        "over_processing_flag": any(i["type"] == "Over-processing" for i in issues),
        "issues":               issues,
        "task_count":           len(all_tasks),
        "manual_task_count":    len(manual_tasks),
        "graph_metrics": {
            "critical_path":     critical_path,
            "bottleneck_nodes":  bottlenecks,
            "isolated_nodes":    isolated,
            "node_count":        len(all_node_names),
            "edge_count":        len(edge_pairs),
            "parallel_gateways": len(parallel_gws),
        },
    }
