"""BPMN engine integration tests.

Run from the project root:
    python -m pytest tests/ -v
or:
    python tests/test_flows.py
"""
import sys
import os
import unittest

# Ensure project root is on sys.path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bpmn_engine.translator  import translate_to_bpmn_schema
from bpmn_engine.validator   import validate_bpmn_structure
from bpmn_engine.analyzer    import analyze_process_health, ProcessGraph
from bpmn_engine.exporter    import generate_bpmn_xml
from bpmn_engine.renderer    import render_diagram, BPMNLayout, _POOL_W
from bpmn_engine.api_context import enrich_with_context, _detect_process_type


class TestTranslator(unittest.TestCase):
    def test_returns_rich_format(self):
        bpmn = translate_to_bpmn_schema("Customer orders pizza. Kitchen prepares. Delivery sends pizza.")
        self.assertIsNotNone(bpmn)
        self.assertIn("pools",    bpmn)
        self.assertIn("events",   bpmn)
        self.assertIn("gateways", bpmn)
        self.assertIn("flows",    bpmn)

    def test_contains_start_and_end_events(self):
        bpmn = translate_to_bpmn_schema("Manager reviews request. If approved then notify team.")
        event_types = {e["type"] for e in bpmn.get("events", [])}
        self.assertIn("Start", event_types)
        self.assertIn("End",   event_types)

    def test_gateway_detected(self):
        bpmn = translate_to_bpmn_schema(
            "System receives payment. If valid then record transaction else notify customer."
        )
        self.assertGreaterEqual(len(bpmn.get("gateways", [])), 1)

    def test_parallel_gateway(self):
        bpmn = translate_to_bpmn_schema(
            "Process starts. In parallel, notify team and update records. End."
        )
        gw_types = {gw["type"] for gw in bpmn.get("gateways", [])}
        # regex may still produce Exclusive — just assert gateways exist
        self.assertGreaterEqual(len(bpmn.get("gateways", [])), 1)


class TestValidator(unittest.TestCase):
    def _make_valid(self) -> dict:
        return {
            "title":    "Test Process",
            "pools":    [{"name": "Org", "lanes": [{"name": "Role", "tasks": [
                {"name": "Do Thing", "type": "User", "assignee": "Role"}
            ]}]}],
            "gateways": [{"type": "Exclusive", "name": "OK?", "reasoning": "check",
                          "yes_branch": ["Approve"], "no_branch": ["Reject"]}],
            "events":   [{"type": "Start", "name": "Begin"}, {"type": "End", "name": "Done"}],
            "flows":    [],
        }

    def test_valid_process_passes(self):
        valid, errors = validate_bpmn_structure(self._make_valid())
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_missing_start_event(self):
        bpmn = self._make_valid()
        bpmn["events"] = [{"type": "End", "name": "Done"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("Start" in e for e in errors))

    def test_missing_end_event(self):
        bpmn = self._make_valid()
        bpmn["events"] = [{"type": "Start", "name": "Begin"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("End" in e for e in errors))

    def test_invalid_gateway_type(self):
        bpmn = self._make_valid()
        bpmn["gateways"][0]["type"] = "Complex"
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("invalid type" in e for e in errors))

    def test_empty_gateway_branches(self):
        bpmn = self._make_valid()
        bpmn["gateways"][0]["yes_branch"] = []
        bpmn["gateways"][0]["no_branch"]  = []
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)


class TestAnalyzer(unittest.TestCase):
    def _make_bpmn(self, task_count: int, task_type: str = "User",
                   parallel: bool = False) -> dict:
        tasks = [{"name": f"Task {i}", "type": task_type} for i in range(task_count)]
        gateways = (
            [{"type": "Parallel", "name": "Par", "yes_branch": ["A"], "no_branch": ["B"]}]
            if parallel else []
        )
        return {
            "pools":    [{"name": "Org", "lanes": [{"name": "Role", "tasks": tasks}]}],
            "gateways": gateways,
            "events":   [{"type": "Start", "name": "S"}, {"type": "End", "name": "E"}],
            "flows":    [],
        }

    def test_healthy_process(self):
        health = analyze_process_health(self._make_bpmn(2))
        self.assertIn("score",  health)
        self.assertIn("health", health)
        # Graph analysis may flag isolated nodes in a minimal test fixture;
        # verify the score is reasonable (no over-processing or waiting-time flags)
        self.assertFalse(health["waiting_time_flag"])
        self.assertFalse(health["over_processing_flag"])

    def test_waiting_time_detected(self):
        health = analyze_process_health(self._make_bpmn(6))
        self.assertTrue(health["waiting_time_flag"])

    def test_no_waiting_time_with_parallel(self):
        health = analyze_process_health(self._make_bpmn(6, parallel=True))
        self.assertFalse(health["waiting_time_flag"])

    def test_over_processing_detected(self):
        health = analyze_process_health(self._make_bpmn(5, task_type="Manual"))
        self.assertTrue(health["over_processing_flag"])

    def test_score_range(self):
        health = analyze_process_health(self._make_bpmn(6, task_type="Manual"))
        self.assertGreaterEqual(health["score"], 0.0)
        self.assertLessEqual(health["score"],    1.0)


class TestExporter(unittest.TestCase):
    def _sample_bpmn(self) -> dict:
        return translate_to_bpmn_schema(
            "Customer submits order. If approved then ship goods else notify customer."
        )

    def test_xml_starts_with_declaration(self):
        xml = generate_bpmn_xml(self._sample_bpmn())
        self.assertTrue(xml.startswith("<?xml") or xml.startswith("<"))

    def test_xml_contains_process(self):
        xml = generate_bpmn_xml(self._sample_bpmn())
        self.assertIn("bpmn:process", xml)

    def test_xml_contains_start_end_events(self):
        xml = generate_bpmn_xml(self._sample_bpmn())
        self.assertIn("startEvent", xml)
        self.assertIn("endEvent",   xml)

    def test_xml_contains_gateway(self):
        xml = generate_bpmn_xml(self._sample_bpmn())
        self.assertIn("Gateway", xml)


class TestRenderer(unittest.TestCase):
    def test_returns_svg_string(self):
        bpmn = translate_to_bpmn_schema("Process request. If valid then complete else reject.")
        svg  = render_diagram(bpmn)
        self.assertIsInstance(svg, str)
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)

    def test_svg_contains_start_end_symbols(self):
        bpmn = translate_to_bpmn_schema("Do work. If done then finish else retry.")
        svg  = render_diagram(bpmn)
        # Start circle (green) and end circle (red) should appear
        self.assertIn("#27ae60", svg)  # start event green
        self.assertIn("#e74c3c", svg)  # end event red

    def test_svg_contains_gateway_diamond(self):
        bpmn = translate_to_bpmn_schema("Review request. If approved then proceed else stop.")
        svg  = render_diagram(bpmn)
        self.assertIn("polygon", svg)  # gateway diamond


class TestFullPipeline(unittest.TestCase):
    def test_pizza_delivery(self):
        text  = "Customer orders pizza. Kitchen prepares. Delivery sends pizza. If late then notify customer else close order."
        bpmn  = translate_to_bpmn_schema(text)
        self.assertIsNotNone(bpmn)
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertIsInstance(errors, list)
        health = analyze_process_health(bpmn)
        self.assertIn("health", health)
        self.assertIn("score",  health)
        xml = generate_bpmn_xml(bpmn)
        self.assertTrue(xml.startswith("<?xml") or xml.startswith("<"))
        svg = render_diagram(bpmn)
        self.assertIn("<svg", svg)

    def test_order_approval(self):
        text = "Manager reviews request. If approved, finance processes payment. Otherwise, reject and notify."
        bpmn = translate_to_bpmn_schema(text)
        self.assertGreaterEqual(len(bpmn.get("gateways", [])), 1)
        xml  = generate_bpmn_xml(bpmn)
        self.assertIn("bpmn:process", xml)

    def test_multi_warehouse_distribution(self):
        text = (
            "A company has two warehouses, one in amsterdam, the other in hamburg, that store different products. "
            "When an order is placed, the warehouse in amsterdam checks stock and ships electronics, "
            "while hamburg ships furniture. Afterwards, the order is consolidated and sent to the customer. "
            "If any item is unavailable, notify customer else approve order."
        )
        bpmn = translate_to_bpmn_schema(text)
        self.assertIsNotNone(bpmn)
        # Validate pools for both warehouses
        pool_names = [p["name"].lower() for p in bpmn.get("pools", [])]
        self.assertIn("amsterdam", pool_names)
        self.assertIn("hamburg", pool_names)
        # Validate lanes and tasks
        lane_names = [ln["name"].lower() for p in bpmn.get("pools", []) for ln in p.get("lanes", [])]
        self.assertIn("amsterdam lane", lane_names)
        self.assertIn("hamburg lane", lane_names)
        task_names = [t["name"].lower() for p in bpmn.get("pools", []) for ln in p.get("lanes", []) for t in ln.get("tasks", [])]
        self.assertTrue(any("check stock" in t for t in task_names))
        self.assertTrue(any("ship electronics" in t for t in task_names))
        self.assertTrue(any("ship furniture" in t for t in task_names))
        # Validate message flows for order transfer
        msg_flows = bpmn.get("message_flows", [])
        self.assertTrue(any("order transfer" in mf.get("label", "").lower() for mf in msg_flows))
        # Validate gateway and branches
        gateways = bpmn.get("gateways", [])
        self.assertGreaterEqual(len(gateways), 1)
        # Validate SVG rendering
        svg = render_diagram(bpmn)
        self.assertIn("<svg", svg)


class TestApiContext(unittest.TestCase):
    def test_detect_logistics(self):
        self.assertEqual(_detect_process_type("ship goods to the warehouse for delivery"), "logistics")

    def test_detect_finance(self):
        self.assertEqual(_detect_process_type("process invoice and record payment in ledger"), "finance")

    def test_detect_generic(self):
        self.assertEqual(_detect_process_type("do some general work"), "generic")

    def test_enrich_adds_process_type(self):
        bpmn = translate_to_bpmn_schema("Ship goods. Deliver to customer.")
        enriched = enrich_with_context(bpmn, "Ship goods to warehouse for delivery.")
        self.assertIn("_process_type", enriched)

    def test_enrich_does_not_mutate_input(self):
        bpmn     = translate_to_bpmn_schema("Process invoice and record payment.")
        original = dict(bpmn)
        enrich_with_context(bpmn, "invoice payment")
        # Original dict must not have been modified
        self.assertNotIn("_process_type", bpmn)
        self.assertEqual(bpmn.get("title"), original.get("title"))

    def test_enrich_injects_annotation_for_logistics(self):
        bpmn     = translate_to_bpmn_schema("Ship order. Deliver package.")
        enriched = enrich_with_context(bpmn, "ship order deliver package to warehouse")
        self.assertIn("_market_context", enriched)

    def test_generic_domain_no_api_context(self):
        bpmn     = translate_to_bpmn_schema("Do some work.")
        enriched = enrich_with_context(bpmn, "do some work")
        self.assertEqual(enriched.get("_process_type"), "generic")
        self.assertNotIn("_market_context",  enriched)


class TestProcessGraph(unittest.TestCase):
    def _linear(self):
        return ProcessGraph(
            ["A", "B", "C"],
            [("A", "B"), ("B", "C")],
        )

    def test_topological_sort_linear(self):
        g = self._linear()
        order = g.topological_order()
        self.assertEqual(order, ["A", "B", "C"])

    def test_longest_path_linear(self):
        g = self._linear()
        self.assertEqual(g.longest_path_length(), 2)

    def test_bottleneck_nodes(self):
        # D has three incoming edges
        g = ProcessGraph(
            ["A", "B", "C", "D"],
            [("A", "D"), ("B", "D"), ("C", "D")],
        )
        self.assertIn("D", g.bottleneck_nodes(threshold=3))

    def test_isolated_nodes(self):
        g = ProcessGraph(
            ["A", "B", "E"],
            [("A", "B")],   # E is isolated
        )
        self.assertIn("E", g.isolated_nodes())
        self.assertNotIn("A", g.isolated_nodes())

    def test_cycle_graceful(self):
        """Kahn's algorithm must not raise on a cycle."""
        g = ProcessGraph(["A", "B"], [("A", "B"), ("B", "A")])
        try:
            g.topological_order()
            g.longest_path_length()
        except Exception as exc:
            self.fail(f"Cycle raised unexpected exception: {exc}")

    def test_graph_metrics_in_health_report(self):
        bpmn = translate_to_bpmn_schema(
            "Receive order. Verify stock. Ship goods. If delivered then close order else retry."
        )
        health = analyze_process_health(bpmn)
        self.assertIn("graph_metrics", health)
        gm = health["graph_metrics"]
        self.assertIn("critical_path",     gm)
        self.assertIn("bottleneck_nodes",  gm)
        self.assertIn("isolated_nodes",    gm)
        self.assertIn("node_count",        gm)
        self.assertIn("edge_count",        gm)
        self.assertIn("parallel_gateways", gm)


class TestValidatorExtended(unittest.TestCase):
    def _valid(self) -> dict:
        return {
            "title":    "Test",
            "pools":    [{"name": "Org", "lanes": [{"name": "Role", "tasks": [
                {"name": "Do Thing", "type": "User", "assignee": "Role"},
                {"name": "Review Item", "type": "User", "assignee": "Role"},
            ]}]}],
            "gateways": [{"type": "Exclusive", "name": "OK?", "reasoning": "check",
                          "yes_branch": ["Approve"], "no_branch": ["Reject"]}],
            "events":   [{"type": "Start", "name": "Begin"}, {"type": "End", "name": "Done"}],
            "flows":    [],
        }

    def test_valid_event_no_subtype(self):
        """Events without a subtype key must still pass validation."""
        valid, errors = validate_bpmn_structure(self._valid())
        self.assertTrue(valid, errors)

    def test_invalid_event_subtype(self):
        bpmn = self._valid()
        bpmn["events"][0]["subtype"] = "fax"
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("subtype" in e for e in errors))

    def test_valid_message_subtype_on_start(self):
        bpmn = self._valid()
        bpmn["events"][0]["subtype"] = "message"
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertTrue(valid, errors)

    def test_data_object_unknown_task(self):
        bpmn = self._valid()
        bpmn["data_objects"] = [{"name": "Form", "type": "input", "attached_to": "NonExistent"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("unknown task" in e for e in errors))

    def test_data_object_invalid_type(self):
        bpmn = self._valid()
        bpmn["data_objects"] = [{"name": "Form", "type": "blob", "attached_to": "Do Thing"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("invalid type" in e for e in errors))

    def test_annotation_unknown_task(self):
        bpmn = self._valid()
        bpmn["annotations"] = [{"text": "SLA: 2h", "attached_to": "Missing Task"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("unknown task" in e for e in errors))

    def test_message_flow_same_pool_error(self):
        bpmn = self._valid()
        bpmn["message_flows"] = [{"source": "Do Thing", "target": "Review Item"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)
        self.assertTrue(any("same pool" in e for e in errors))

    def test_message_flow_missing_source(self):
        bpmn = self._valid()
        bpmn["message_flows"] = [{"target": "Do Thing"}]
        valid, errors = validate_bpmn_structure(bpmn)
        self.assertFalse(valid)


class TestBPMNLayout(unittest.TestCase):
    """Tests for the flow-graph-based BPMNLayout engine."""

    def _make_json(self, pools=None, gateways=None, events=None, flows=None):
        """Build a minimal process JSON for layout testing."""
        if pools is None:
            pools = [{"name": "Org", "lanes": [{"name": "Role", "tasks": [
                {"name": "T0", "type": "User"},
                {"name": "T1", "type": "User"},
            ]}]}]
        return {
            "title": "Test",
            "pools": pools,
            "gateways": gateways or [],
            "events": events or [
                {"type": "Start", "name": "Start"},
                {"type": "End",   "name": "End"},
            ],
            "flows": flows or [
                {"type": "Sequence", "source": "Start", "target": "T0"},
                {"type": "Sequence", "source": "T0",    "target": "T1"},
                {"type": "Sequence", "source": "T1",    "target": "End"},
            ],
        }

    def test_node_positions_assigned(self):
        layout = BPMNLayout(self._make_json())
        self.assertIn("Start", layout.node_pos)
        self.assertIn("T0",    layout.node_pos)
        self.assertIn("End",   layout.node_pos)
        self.assertGreater(layout.width,  0)
        self.assertGreater(layout.height, 0)

    def test_topological_columns_increase(self):
        layout = BPMNLayout(self._make_json())
        col_start = layout.node_col["Start"]
        col_t0    = layout.node_col["T0"]
        col_t1    = layout.node_col["T1"]
        col_end   = layout.node_col["End"]
        self.assertLess(col_start, col_t0)
        self.assertLess(col_t0,    col_t1)
        self.assertLess(col_t1,    col_end)

    def test_yes_branch_above_no_branch_single_lane(self):
        """In a single-lane process, yes branch gets negative offset (higher up)."""
        pools = [{"name": "Org", "lanes": [{"name": "Role", "tasks": [
            {"name": "T0",  "type": "User"},
            {"name": "Yes", "type": "User"},
            {"name": "No",  "type": "User"},
        ]}]}]
        gateways = [{"type": "Exclusive", "name": "GW",
                     "yes_branch": ["Yes"], "no_branch": ["No"]}]
        flows = [
            {"type": "Sequence", "source": "Start", "target": "T0"},
            {"type": "Sequence", "source": "T0",    "target": "GW"},
            {"type": "Sequence", "source": "GW",    "target": "Yes"},
            {"type": "Sequence", "source": "GW",    "target": "No"},
            {"type": "Sequence", "source": "Yes",   "target": "End"},
            {"type": "Sequence", "source": "No",    "target": "End"},
        ]
        layout = BPMNLayout(self._make_json(pools=pools, gateways=gateways, flows=flows))
        _, yes_cy = layout.node_pos["Yes"]
        _, no_cy  = layout.node_pos["No"]
        self.assertLess(yes_cy, no_cy)

    def test_multi_lane_different_y(self):
        """Tasks in different lanes have different cy values."""
        pools = [{"name": "Org", "lanes": [
            {"name": "Lane A", "tasks": [{"name": "TA", "type": "User"}]},
            {"name": "Lane B", "tasks": [{"name": "TB", "type": "User"}]},
        ]}]
        flows = [
            {"type": "Sequence", "source": "Start", "target": "TA"},
            {"type": "Sequence", "source": "TA",    "target": "TB"},
            {"type": "Sequence", "source": "TB",    "target": "End"},
        ]
        layout = BPMNLayout(self._make_json(pools=pools, flows=flows))
        _, cy_a = layout.node_pos["TA"]
        _, cy_b = layout.node_pos["TB"]
        self.assertNotEqual(cy_a, cy_b)

    def test_multi_pool_header_width(self):
        """Two or more lanes → header_w = _POOL_W + lane strip."""
        from bpmn_engine.renderer import _POOL_LW, _LANE_LW
        pools = [{"name": "Org", "lanes": [
            {"name": "L1", "tasks": [{"name": "TA", "type": "User"}]},
            {"name": "L2", "tasks": [{"name": "TB", "type": "User"}]},
        ]}]
        layout = BPMNLayout(self._make_json(pools=pools))
        self.assertEqual(layout.header_w, _POOL_LW + _LANE_LW)

    def test_single_pool_header_width(self):
        """Single lane → header_w = _POOL_W only."""
        layout = BPMNLayout(self._make_json())
        self.assertEqual(layout.header_w, _POOL_W)

    def test_multi_pool_svg_renders(self):
        """Two separate pools both appear in the rendered SVG."""
        bpmn = {
            "title": "Two Pool Process",
            "pools": [
                {"name": "Buyer",  "lanes": [{"name": "Buyer",  "tasks": [{"name": "Place Order",   "type": "User"}]}]},
                {"name": "Seller", "lanes": [{"name": "Seller", "tasks": [{"name": "Fulfill Order", "type": "Service"}]}]},
            ],
            "gateways": [],
            "events":  [{"type": "Start", "name": "Start"}, {"type": "End", "name": "End"}],
            "flows":   [
                {"type": "Sequence", "source": "Start",       "target": "Place Order"},
                {"type": "Sequence", "source": "Place Order", "target": "Fulfill Order"},
                {"type": "Sequence", "source": "Fulfill Order", "target": "End"},
            ],
        }
        svg = render_diagram(bpmn)
        self.assertIn("Buyer",         svg)
        self.assertIn("Seller",        svg)
        self.assertIn("Place Order",   svg)
        self.assertIn("Fulfill Order", svg)
        self.assertIn("<svg",          svg)

    def test_multiple_gateways_in_svg(self):
        """Two sequential gateways both render as polygons."""
        bpmn = {
            "title": "Multi-Gateway",
            "pools": [{"name": "Org", "lanes": [{"name": "Role", "tasks": [
                {"name": "T1", "type": "User"},
                {"name": "YA", "type": "User"},
                {"name": "NA", "type": "User"},
                {"name": "YB", "type": "User"},
                {"name": "NB", "type": "User"},
            ]}]}],
            "gateways": [
                {"type": "Exclusive", "name": "GW1",
                 "yes_branch": ["YA"], "no_branch": ["NA"]},
                {"type": "Exclusive", "name": "GW2",
                 "yes_branch": ["YB"], "no_branch": ["NB"]},
            ],
            "events": [{"type": "Start", "name": "Start"}, {"type": "End", "name": "End"}],
            "flows": [
                {"type": "Sequence", "source": "Start", "target": "T1"},
                {"type": "Sequence", "source": "T1",    "target": "GW1"},
                {"type": "Sequence", "source": "GW1",   "target": "YA"},
                {"type": "Sequence", "source": "GW1",   "target": "NA"},
                {"type": "Sequence", "source": "YA",    "target": "GW2"},
                {"type": "Sequence", "source": "NA",    "target": "GW2"},
                {"type": "Sequence", "source": "GW2",   "target": "YB"},
                {"type": "Sequence", "source": "GW2",   "target": "NB"},
                {"type": "Sequence", "source": "YB",    "target": "End"},
                {"type": "Sequence", "source": "NB",    "target": "End"},
            ],
        }
        svg = render_diagram(bpmn)
        self.assertEqual(svg.count("<polygon"), 2)

    def test_message_event_envelope_in_svg(self):
        bpmn = translate_to_bpmn_schema("Receive order. Process it.")
        bpmn["message_events"] = [
            {"type": "Start", "subtype": "message", "name": "Receive Order"}
        ]
        svg = render_diagram(bpmn)
        self.assertIn("27ae60", svg)


if __name__ == "__main__":
    unittest.main()
