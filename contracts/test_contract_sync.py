"""Guards the hand-maintained mirror between contracts/types.ts and contracts/schema.py.

Three agents build against this contract in parallel: the frontend codes to the
TypeScript types, the backend to the Pydantic models, the ML layer to the dataclasses
that wrap them. If the two files drift, nothing fails until final integration — the
most expensive moment to discover it. This test moves that failure to commit time.

Run standalone (`python contracts/test_contract_sync.py`) or under pytest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTRACTS_DIR.parent))

import contracts.schema as schema  # noqa: E402

TYPES_TS = CONTRACTS_DIR / "types.ts"

# TS interfaces with no standalone Pydantic counterpart, and why.
TS_ONLY = {
    # Flattened into HarmonizeRequest.options in TS, a named model in Python.
    "HarmonizeOptions",
}

# Fields that legitimately differ, with justification.
FIELD_EXEMPTIONS: dict[str, set[str]] = {}

# Server -> client models. For these, a Python default paired with a TS-required field
# is correct rather than drift: the server always populates the value, so the frontend
# should not have to null-check it. The reverse (TS optional, Python required) is still
# an error. Request-side models get the strict rule — a Python default there means the
# client may omit the field, so TS must mark it optional.
RESPONSE_MODELS = {
    "Chord",
    "Voice",
    "Violation",
    "HarmonizeResponse",
    "EngineInfo",
    "EnginesResponse",
}


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def parse_ts_interfaces(source: str) -> dict[str, dict[str, bool]]:
    """Map interface name -> {field name: is_optional}, top level only.

    Nested inline object literals (e.g. HarmonizeRequest.options) are skipped by
    tracking brace depth, so only depth-1 fields are compared.
    """
    source = _strip_comments(source)
    interfaces: dict[str, dict[str, bool]] = {}

    for match in re.finditer(r"export\s+interface\s+(\w+)\s*\{", source):
        name = match.group(1)
        depth, i = 1, match.end()
        body_start = i
        while i < len(source) and depth > 0:
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        body = source[body_start : i - 1]

        fields: dict[str, bool] = {}
        depth = 0
        for line in body.splitlines():
            stripped = line.strip()
            if depth == 0:
                field = re.match(r"(\w+)(\?)?\s*:", stripped)
                if field:
                    declared_optional = field.group(2) == "?"
                    nullable = "null" in stripped.split(":", 1)[1]
                    fields[field.group(1)] = declared_optional or nullable
            depth += line.count("{") - line.count("}")

        interfaces[name] = fields
    return interfaces


def pydantic_models() -> dict[str, type]:
    from pydantic import BaseModel

    return {
        name: obj
        for name, obj in vars(schema).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    }


def parse_py_models() -> dict[str, dict[str, bool]]:
    """Map model name -> {field name: is_optional}. Optional == not required."""
    return {
        name: {fname: not f.is_required() for fname, f in model.model_fields.items()}
        for name, model in pydantic_models().items()
    }


def check() -> list[str]:
    ts = parse_ts_interfaces(TYPES_TS.read_text())
    py = parse_py_models()
    errors: list[str] = []

    shared = (set(ts) & set(py)) - TS_ONLY
    if not shared:
        errors.append("No interfaces matched between types.ts and schema.py — parser is broken.")

    for name in sorted(set(py) - set(ts) - TS_ONLY):
        errors.append(f"{name}: defined in schema.py but missing from types.ts")
    for name in sorted(set(ts) - set(py) - TS_ONLY):
        errors.append(f"{name}: defined in types.ts but missing from schema.py")

    for name in sorted(shared):
        ts_fields, py_fields = ts[name], py[name]
        exempt = FIELD_EXEMPTIONS.get(name, set())

        for field in sorted(set(py_fields) - set(ts_fields) - exempt):
            errors.append(f"{name}.{field}: in schema.py but not types.ts")
        for field in sorted(set(ts_fields) - set(py_fields) - exempt):
            errors.append(f"{name}.{field}: in types.ts but not schema.py")

        for field in sorted(set(ts_fields) & set(py_fields) - exempt):
            ts_optional, py_optional = ts_fields[field], py_fields[field]
            if ts_optional == py_optional:
                continue
            # Server-guaranteed field: Python supplies a default, TS treats it as always present.
            if name in RESPONSE_MODELS and py_optional and not ts_optional:
                continue
            ts_state = "optional" if ts_optional else "required"
            py_state = "optional" if py_optional else "required"
            errors.append(
                f"{name}.{field}: optionality mismatch — types.ts says {ts_state}, "
                f"schema.py says {py_state}"
            )

    return errors


def test_contract_files_are_in_sync():
    errors = check()
    assert not errors, "contracts/types.ts and contracts/schema.py have drifted:\n  " + "\n  ".join(errors)


if __name__ == "__main__":
    found = check()
    if found:
        print(f"CONTRACT DRIFT ({len(found)} issue(s)):")
        for error in found:
            print(f"  - {error}")
        raise SystemExit(1)
    models = sorted((set(parse_ts_interfaces(TYPES_TS.read_text())) & set(parse_py_models())) - TS_ONLY)
    print(f"Contract in sync across {len(models)} shared models: {', '.join(models)}")
