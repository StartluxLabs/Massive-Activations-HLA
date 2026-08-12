from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal envs
    yaml = None


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in config values."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            return match.group(0)

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env_vars(x) for x in value]
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if yaml is None:
        raise ImportError("PyYAML is required to read YAML configs. Install with `pip install pyyaml`.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return expand_env_vars(data)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    if yaml is None:
        raise ImportError("PyYAML is required to write YAML configs. Install with `pip install pyyaml`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


@dataclass(frozen=True)
class ConfigRef:
    """Reference of the form /path/to/file.yaml:key."""

    path: Path
    key: str | None = None

    @classmethod
    def parse(cls, value: str | Path) -> ConfigRef:
        s = str(value)
        if ":" in s and not s.startswith("http"):
            path, key = s.rsplit(":", 1)
            return cls(Path(path), key or None)
        return cls(Path(s), None)

    def load(self) -> Any:
        data = load_yaml(self.path)
        if self.key is None:
            return data
        cur: Any = data
        for part in self.key.split("."):
            cur = cur[part]
        return cur
