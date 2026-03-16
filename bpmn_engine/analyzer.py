"""ML Layer — Lean Muda Analysis with graph-theoretic waste detection.

Pipeline position:
    LLM → APIs → **ML (analyze)** → Engine

ProcessGraph
───────────
A directed graph built from BPMN flows providing:
  - topological_order()     Kahn's BFS algorithm
  - longest_path_length()   DP over topological order (critical path)
  - bottleneck_nodes()      high in-degree convergence points
  - isolated_nodes()        tasks with degree zero

Muda categories detected
───────────────────────
  Waiting Time     : critical path too long / sequential tasks with no parallel gw
  Over-processing  : excessive manual tasks
  Transport        : cross-pool message flow overuse
  Bottleneck       : high-convergence nodes
  Redundant Gateway: exclusive gateways with identical branches
  Defect           : isolated / disconnected nodes
"""
from collections import deque
from typing import Any


class ProcessGraph:
    """Lightweight directed graph for BPMN process analysis."""

    def __init__(self, nodes: list[str], edges: list[tuple[str, str]]) -> None:
        self.nodes = list(dict.fromkeys(nodes))
        self._node_set = set(self.nodes)
        self._out = {n: [] for n in self.nodes}
        self._in = {n: [] for n in self.nodes}

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
        in_deg = {n: self.in_degree(n) for n in self.nodes}
        queue = deque(sorted(n for n in self.nodes if in_deg[n] == 0))
        result = []

        while queue:
            n = queue.popleft()
            result.append(n)
            for succ in sorted(self._out.get(n, [])):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        return result

    def longest_path_length(self) -> int:
        order = self.topological_order()
        dist = {n: 0 for n in self.nodes}

        for n in order:
            for succ in self._out.get(n, []):
                dist[succ] = max(dist[succ], dist[n] + 1)

        return max(dist.values(), default=0)

    def bottleneck_nodes(self, threshold: int = 3) -> list[str]:
        return [n for n in self.nodes if self.in_degree(n) >= threshold]

    def isolated_nodes(self) -> list[str]:
        return [n for n in self.nodes if self.in_degree(n) == 0 and self.out_degree(n) == 0]


def _collect_node_names(process_json: dict[str, Any]) -> list[str]:
    """Gather every named node: events + tasks + gateway names + branch tasks."""
    names = []

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
        names.extend(gw.get("no_branch", []))

    return names


def analyze_process_health(process_json: dict[str, Any]) -> dict[str, Any]:
    """
    ML-layer Lean Muda health analysis with optimization recommendations.
    
    Returns:
        score (0.0-1.0): Health score (1.0 = no waste)
        health: "good" | "needs_improvement" | "critical"
        issues: List of detected issues with recommendations
        optimization_recommendations: AI-generated improvement suggestions
        graph_metrics: Critical path, bottlenecks, etc.
    """
    issues = []
    recommendations = []
    score = 1.0

    # Collect elements
    all_tasks = [
        t for pool in process_json.get("pools", [])
        for lane in pool.get("lanes", [])
        for t in lane.get("tasks", [])
    ]
    manual_tasks = [t for t in all_tasks if t.get("type") == "Manual"]
    service_tasks = [t for t in all_tasks if t.get("type") == "Service"]
    user_tasks = [t for t in all_tasks if t.get("type") == "User"]
    gateways = process_json.get("gateways", [])
    flows = process_json.get("flows", [])
    pools = process_json.get("pools", [])

    # Build process graph
    all_node_names = _collect_node_names(process_json)
    edge_pairs = [
        (f["source"], f["target"])
        for f in flows
        if f.get("source") and f.get("target")
    ]
    graph = ProcessGraph(all_node_names, edge_pairs)

    critical_path = graph.longest_path_length()
    bottlenecks = graph.bottleneck_nodes(threshold=3)
    isolated = graph.isolated_nodes()
    parallel_gws = [gw for gw in gateways if gw.get("type") == "Parallel"]
    exclusive_gws = [gw for gw in gateways if gw.get("type") == "Exclusive"]

    # ── Critical Path Analysis ───────────────────────────────────────────────
    if critical_path > 8:
        issues.append({
            "type": "Waiting Time",
            "description": f"Critical path spans {critical_path} hops - too long!",
            "severity": "high",
        })
        recommendations.append({
            "action": "Introduce Parallel gateways",
            "benefit": "Reduce cycle time by running independent steps concurrently",
            "impact": "high"
        })
        score -= 0.20
    elif len(all_tasks) > 4 and not parallel_gws:
        issues.append({
            "type": "Waiting Time",
            "description": f"{len(all_tasks)} sequential tasks with no Parallel gateway",
            "severity": "medium",
        })
        recommendations.append({
            "action": "Add Parallel gateway for independent tasks",
            "benefit": "Reduce overall process time",
            "impact": "medium"
        })
        score -= 0.10

    # ── Bottleneck Analysis ──────────────────────────────────────────────────
    if bottlenecks:
        issues.append({
            "type": "Bottleneck",
            "description": f"High-convergence nodes: {', '.join(bottlenecks[:3])}",
            "severity": "medium",
        })
        recommendations.append({
            "action": f"Decompose or automate '{bottlenecks[0]}'",
            "benefit": "Reduce serialization at convergence points",
            "impact": "medium"
        })
        score -= 0.10

    # ── Isolated Nodes (Disconnected) ─────────────────────────────────────────
    if isolated:
        issues.append({
            "type": "Defect",
            "description": f"{len(isolated)} task(s) not connected: {', '.join(isolated[:3])}",
            "severity": "medium",
        })
        recommendations.append({
            "action": "Add Sequence flows to integrate disconnected tasks",
            "benefit": "Ensure complete process coverage",
            "impact": "high"
        })
        score -= 0.05

    # ── Over-processing Analysis ─────────────────────────────────────────────
    if len(manual_tasks) > 3:
        issues.append({
            "type": "Over-processing",
            "description": f"{len(manual_tasks)} manual tasks - consider automation",
            "severity": "high",
        })
        recommendations.append({
            "action": "Automate repetitive manual tasks",
            "benefit": "Reduce labor costs and human error",
            "impact": "high",
            "specific": [t["name"] for t in manual_tasks[:3]]
        })
        score -= 0.15
    
    # Check for automation opportunities
    automatable = [t for t in manual_tasks if any(kw in t["name"].lower() 
        for kw in ["check", "verify", "validate", "update", "send", "notify"])]
    if automatable and len(automatable) > 1:
        recommendations.append({
            "action": "Consider RPA for: " + ", ".join([t["name"] for t in automatable[:2]]),
            "benefit": "Automate repetitive validation/notification tasks",
            "impact": "medium"
        })

    # ── Redundant Gateway Analysis ────────────────────────────────────────────
    for gw in gateways:
        if (gw.get("type") == "Exclusive" and gw.get("yes_branch") 
            and gw.get("no_branch") and set(gw["yes_branch"]) == set(gw["no_branch"])):
            issues.append({
                "type": "Redundant Gateway",
                "description": f"Gateway '{gw.get('name')}' has identical branches",
                "severity": "medium",
            })
            recommendations.append({
                "action": f"Remove redundant gateway '{gw.get('name')}'",
                "benefit": "Simplify process flow",
                "impact": "low"
            })
            score -= 0.10
            break

    # ── Cross-pool Communication Analysis ────────────────────────────────────
    if len(pools) > 1:
        msg_flows = [f for f in flows if f.get("type") == "Message"]
        if len(msg_flows) > 4:
            issues.append({
                "type": "Transport",
                "description": f"{len(msg_flows)} cross-pool message flows - potential handoff delays",
                "severity": "low",
            })
            recommendations.append({
                "action": "Consolidate cross-organizational handoffs",
                "benefit": "Reduce latency and communication overhead",
                "impact": "medium"
            })
            score -= 0.05

    # ── Process Efficiency Score ──────────────────────────────────────────────
    # Calculate automation ratio
    automation_ratio = len(service_tasks) / max(len(all_tasks), 1)
    if automation_ratio < 0.3 and len(all_tasks) > 5:
        recommendations.append({
            "action": "Increase automation ratio",
            "benefit": f"Currently {int(automation_ratio*100)}% automated - target 50%+",
            "impact": "high"
        })

    # ── Gateway Efficiency ───────────────────────────────────────────────────
    if len(exclusive_gws) > 3:
        recommendations.append({
            "action": "Consider consolidating decision points",
            "benefit": "Reduce gateway complexity for better traceability",
            "impact": "low"
        })

    # ── Process Complexity Score ──────────────────────────────────────────────
    complexity = len(all_tasks) + len(gateways) * 2
    if complexity > 15:
        recommendations.append({
            "action": "Consider breaking into subprocesses",
            "benefit": "Reduce complexity for easier maintenance",
            "impact": "medium"
        })

    score = max(round(score, 2), 0.0)

    return {
        "score": score,
        "health": (
            "good" if score >= 0.75 else
            "needs_improvement" if score >= 0.50 else
            "critical"
        ),
        "waiting_time_flag": any(i["type"] == "Waiting Time" for i in issues),
        "over_processing_flag": any(i["type"] == "Over-processing" for i in issues),
        "issues": issues,
        "optimization_recommendations": recommendations,
        "task_count": len(all_tasks),
        "manual_task_count": len(manual_tasks),
        "service_task_count": len(service_tasks),
        "user_task_count": len(user_tasks),
        "automation_ratio": round(automation_ratio, 2),
        "graph_metrics": {
            "critical_path": critical_path,
            "bottleneck_nodes": bottlenecks,
            "isolated_nodes": isolated,
            "node_count": len(all_node_names),
            "edge_count": len(edge_pairs),
            "parallel_gateways": len(parallel_gws),
            "exclusive_gateways": len(exclusive_gws),
            "process_complexity": complexity,
        },
    }
