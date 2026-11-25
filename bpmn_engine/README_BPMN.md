# bpmn_engine

The modules behind the four pipeline steps. See the main README for how they fit together.

- `translator.py`: turns a text description into a BPMN model (Claude, or a rule-based fallback)
- `api_context.py`: detects the process domain and adds simulated context notes
- `validator.py`: checks the BPMN structure
- `analyzer.py`: rule-based Lean waste analysis on the process graph
- `renderer.py`: lays out and draws the diagram as SVG
- `exporter.py`: exports BPMN 2.0 XML
