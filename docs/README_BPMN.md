# BPMN Diagram Studio

## Overview
This project is a professional-grade BPMN 2.0 engine that transforms natural language into valid, optimized, and executable process models using multi-layered AI (LLM + Symbolic Logic).

## Features
- Semantic Parsing (LLM): Pools, Lanes, Task categorization, Gateway logic, Reasoning metadata
- Structural Integrity: BPMN validation, standardized notation, flow distinction
- Visual Optimization: Orthogonal routing, BPMN-standard event borders, message flows
- Process Health Analysis: ML-based waste detection
- BPMN XML export for Camunda/Signavio

## File Structure
- `app.py`: Main application logic and UI
- `bpmn_schema.json`: BPMN 2.0 JSON schema for process modeling
- `requirements.txt`: Python dependencies
- `README_BPMN.md`: Project documentation

## Usage
1. Run the app with `python app.py`
2. Enter a process description in natural language
3. Generate diagram and export BPMN XML

## Extending
- Add new pools, lanes, tasks, gateways, and flows in JSON
- Integrate external BPMN validation APIs or ML modules

## License
MIT
