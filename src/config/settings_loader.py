"""Loads application settings from config/settings.yaml.

Also calls load_dotenv() so MT5 credentials in .env are available via
os.environ before MT5Provider reads them.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load YAML settings and populate os.environ from .env.

    Raises FileNotFoundError if the YAML file does not exist.
    Does not raise if .env is missing — it is optional.
    """
    load_dotenv()  # silently skips if .env is absent

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{config_path}'. "
            "Ensure config/settings.yaml exists in the project root."
        )

    with path.open("r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh) or {}

    logger.debug("settings_loaded: path=%s", config_path)
    return config
