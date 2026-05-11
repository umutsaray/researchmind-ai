from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional during partial installs
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _secret_value(key: str) -> str:
    try:
        import streamlit as st

        value: Any = st.secrets.get(key, "")
        if value is None:
            return ""
        return str(value).strip()
    except Exception:
        return ""


def _env_file_value(key: str, env_path: str | Path = ".env") -> str:
    path = Path(env_path)
    if not path.exists():
        return ""

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if not clean or clean.startswith("#") or "=" not in clean:
                continue
            env_key, env_value = clean.split("=", 1)
            if env_key.strip() == key:
                return env_value.strip().strip('"').strip("'")
    except OSError:
        return ""

    return ""


def get_config_value(key: str, default: str = "", env_path: str | Path = ".env") -> str:
    return (
        _secret_value(key)
        or os.getenv(key, "").strip()
        or _env_file_value(key, env_path)
        or default
    )


def get_config_bool(key: str, default: bool = False) -> bool:
    value = get_config_value(key)
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def missing_config_keys(keys: list[str]) -> list[str]:
    return [key for key in keys if not get_config_value(key)]
