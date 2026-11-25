# AI BPMN Generator

Turns a process description written in plain English into a BPMN 2.0 model: pools, lanes, tasks, gateways and flows. The engine validates the model, checks it for Lean waste, draws it as an SVG diagram and exports standard BPMN 2.0 XML.

I built it to explore how a language model can take over the first draft of process modelling, while classic rule-based checks keep the result correct.

## How it works

The engine runs a four-step pipeline:

| Step | Module | What it does |
|------|--------|--------------|
| 1. Translate | `translator.py` | Sends the description to Claude and gets a structured BPMN model back. Without an API key, a simple rule-based parser builds a basic model so the app still runs. |
| 2. Add context | `api_context.py` | Detects the process domain (logistics, finance, HR and others) and adds domain notes to the model. The context values are simulated demo data, not live API calls. |
| 3. Validate and analyse | `validator.py`, `analyzer.py` | Checks the BPMN structure (one start event, at least one end event, valid flows and gateway types) and runs a rule-based Lean waste analysis on the process graph: waiting points, over-processing and manual work, with a health score and recommendations. |
| 4. Render and export | `renderer.py`, `exporter.py` | Lays out the diagram with orthogonal routing, draws it as SVG and exports BPMN 2.0 XML. |

## Run it

```bash
git clone https://github.com/SamsonFHNW/ai_bpmn_generator_project.git
cd ai_bpmn_generator_project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
EXPORT_PATH=bpmn_exports
```

Then start the web app (NiceGUI):

```bash
python app.py
```

Open http://localhost:8000, describe a process and generate the diagram.

You can also call the pipeline from Python:

```python
from app import run_pipeline

result = run_pipeline(
    "A customer places an order. The warehouse checks stock. "
    "If the item is available, ship the order, else notify the customer."
)
print(result["validation_ok"])
print(result["health_report"]["score"])
print(result["xml_file"])
```

## Tests

```bash
python -m pytest tests/ -v
```

54 tests cover translation, validation, Lean analysis, graph algorithms, layout and XML export. 53 run offline. One test needs `ANTHROPIC_API_KEY`, because only the language model can name pools from free text, and it is skipped without a key.

## Project structure

```
ai_bpmn_generator_project/
├── app.py                 Web app and pipeline
├── config.py              Settings from environment variables
├── requirements.txt
├── bpmn_engine/
│   ├── translator.py      Step 1: text to BPMN model (Claude, or rule-based fallback)
│   ├── api_context.py     Step 2: domain detection and simulated context
│   ├── validator.py       Step 3: BPMN structure checks
│   ├── analyzer.py        Step 3: Lean waste analysis on the process graph
│   ├── renderer.py        Step 4: SVG layout and drawing
│   └── exporter.py        Step 4: BPMN 2.0 XML export
├── data/
│   ├── bpmn_schema.json   JSON schema for the BPMN model
│   └── process_examples.json
├── docs/                  Presentation slides
└── tests/
    └── test_flows.py
```

## Tech stack

Python 3.11, NiceGUI, Anthropic Claude API, BPMN 2.0

## Limitations

- The rule-based fallback parser only handles simple, linear descriptions. Complex processes need the language model.
- The domain context in step 2 is simulated and meant to show where real data sources would plug in.

## Author

Samson Hadgu, BSc Business Information Technology student at FHNW
