# AI-Augmented BPMN 2.0 Engine

A powerful AI-driven engine that transforms natural language process descriptions into fully validated, analyzed, and exportable BPMN 2.0 diagrams using a 4-layer intelligent pipeline.

![BPMN](https://img.shields.io/badge/BPMN-2.0-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

The AI BPMN Generator leverages artificial intelligence to democratize business process modeling. Users describe their processes in plain English, and the engine automatically generates professional BPMN 2.0 diagrams with validation, analysis, and export capabilities.

## Features

- **Natural Language to BPMN**: Convert process descriptions into structured BPMN 2.0 diagrams
- **AI-Powered Translation**: Uses Claude AI for intelligent process modeling
- **Schema Validation**: Ensures BPMN 2.0 compliance
- **Process Health Analysis**: ML-based bottleneck and waste detection
- **SVG Rendering**: Visual diagram generation
- **XML Export**: Standard BPMN 2.0 XML output

## Architecture

The engine implements a 4-layer pipeline (ABPMS Manifesto):

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4 — Engine    │ Render SVG + Export BPMN 2.0 XML    │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — ML        │ Graph-theoretic Lean Muda audit     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — APIs      │ Enrich with real-world context       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — LLM       │ Translate natural language → BPMN   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/SamsonFHNW/ai_bpmn_generator_project.git
cd ai_bpmn_generator_project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
EXPORT_PATH=bpmn_exports
```

Get your API key from [Anthropic](https://www.anthropic.com/).

## Usage

### Running the Web Application

```bash
python app.py
```

The application will start on `http://localhost:8000`. Open the URL in your browser to access the interactive BPMN generator.

### Using the Pipeline Programmatically

```python
from app import run_pipeline

# Describe your process in natural language
input_text = "A customer places an order, receives confirmation, 
              then the order is processed and shipped."

# Execute the full pipeline
result = run_pipeline(input_text)

print(f"Validation: {result['validation_ok']}")
print(f"Health Report: {result['health_report']}")
print(f"XML File: {result['xml_file']}")
```

## Project Structure

```
ai_bpmn_generator_project/
├── app.py                  # Main application & pipeline
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── bpmn_engine/
│   ├── __init__.py         # Package exports
│   ├── translator.py       # LLM → BPMN translation
│   ├── validator.py        # BPMN 2.0 validation
│   ├── analyzer.py         # ML process health analysis
│   ├── renderer.py         # SVG diagram rendering
│   ├── exporter.py         # BPMN XML export
│   ├── api_context.py      # API context enrichment
│   ├── README_BPMN.md      # Engine documentation
│   └── data/
│       ├── bpmn_schema.json    # BPMN 2.0 schema
│       └── process_examples.json
├── tests/
│   └── test_flows.py       # Test suite
└── bpmn_exports/           # Generated BPMN files
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `translate_to_bpmn_schema()` | Convert natural language to BPMN JSON |
| `validate_bpmn_structure()` | Validate BPMN 2.0 structure |
| `analyze_process_health()` | ML-based process analysis |
| `generate_bpmn_xml()` | Export to BPMN 2.0 XML |
| `render_diagram()` | Generate SVG visualization |
| `enrich_with_context()` | Add external API context |

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_flows.py -v
```

## Tech Stack

- **Frontend**: NiceGUI (Python UI framework)
- **AI**: Anthropic Claude API
- **BPMN**: BPMN 2.0 standard
- **Python**: 3.11+

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Author

Samson FHNW

---

⭐ Star this repository if you find it useful!
