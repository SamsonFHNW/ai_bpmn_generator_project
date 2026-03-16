"""Centralized configuration — all paths and API keys loaded from .env."""
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# API Keys
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Paths
DATA_PATH: str = str(BASE_DIR / "data")
SCHEMA_FILE: str = str(BASE_DIR / "data" / "bpmn_schema.json")
EXAMPLES_FILE: str = str(BASE_DIR / "data" / "process_examples.json")
EXPORT_PATH: str = str(BASE_DIR / "bpmn_exports")
