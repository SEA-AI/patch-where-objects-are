"""YAML → nested dataclass loading."""

from __future__ import annotations

import dataclasses
from typing import Any, get_type_hints

import yaml


def parse_override_value(raw: object) -> object:
    """CLI string -> value as YAML reads it (``null``, ``true``, ``1``, ``[16]``).

    Exponent floats without a dot (``1e-4``), which YAML 1.1 reads as strings,
    are parsed as floats.
    """
    if not isinstance(raw, str):
        return raw
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_overrides(argv: list[str]) -> dict:
    """``["--train.seed", "1", ...]`` -> ``{"train.seed": 1, ...}``.

    Every ``--<key>`` needs a value; a flag without one raises.
    """
    overrides = {}
    i = 0
    while i < len(argv):
        if not argv[i].startswith("--"):
            raise ValueError(f"unexpected argument {argv[i]!r}")
        key = argv[i][2:]
        if i + 1 == len(argv) or argv[i + 1].startswith("--"):
            raise ValueError(f"--{key} needs a value, e.g. --{key} true")
        overrides[key] = parse_override_value(argv[i + 1])
        i += 2
    return overrides


def apply_yaml_overrides(raw: dict, overrides: dict) -> None:
    """Apply CLI dot-key overrides in place (e.g. ``train.gpu`` → nested dict)."""
    for dotkey, val in overrides.items():
        keys = dotkey.split(".")
        target = raw
        for k in keys[:-1]:
            target = target.setdefault(k, {})
            if not isinstance(target, dict):
                raise ValueError(f"unknown config key {dotkey!r}")
        target[keys[-1]] = val


def dict_to_dataclass(cls: type, d: dict, prefix: str = "") -> Any:
    """Convert a nested dict to a dataclass (recursive for nested dataclass fields).

    Raises on keys that are not fields of ``cls``.
    """
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [prefix + k for k in d if k not in fields]
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, fld in fields.items():
        if name not in d:
            if fld.default is not dataclasses.MISSING:
                kwargs[name] = fld.default
            elif fld.default_factory is not dataclasses.MISSING:
                kwargs[name] = fld.default_factory()
            continue
        val = d[name]
        field_type = hints.get(name, fld.type)
        if dataclasses.is_dataclass(field_type) and isinstance(val, dict):
            kwargs[name] = dict_to_dataclass(field_type, val, f"{prefix}{name}.")
        else:
            kwargs[name] = val
    return cls(**kwargs)
