"""Guards the hand-maintained mirror between contracts/types.ts and contracts/schema.py.

Three agents build against this contract in parallel: the frontend codes to the
TypeScript types, the backend to the Pydantic models, the ML layer to the dataclasses
that wrap them. If the two files drift, nothing fails until final integration — the
most expensive moment to discover it. This test moves that failure to commit time.

It compares three things per field: presence, optionality, and type. An earlier
version compared only the first two, which meant `KeySignature.tonic` could change
from int to str, and a response field could flip to `Optional[X] = None` — emitting
JSON null while types.ts still promised a value — with the guard staying green. Both
gaps are closed below and covered by self-tests.

Run standalone (`python contracts/test_contract_sync.py`) or under pytest.
"""

from __future__ import annotations

import re
import sys
import types as pytypes
import typing
from dataclasses import dataclass
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTRACTS_DIR.parent))

import contracts.schema as schema  # noqa: E402

TYPES_TS = CONTRACTS_DIR / "types.ts"

# TS interfaces with no standalone Pydantic counterpart, and why.
TS_ONLY: set[str] = set()

# Fields that legitimately differ, with justification.
FIELD_EXEMPTIONS: dict[str, set[str]] = {}

# Server -> client models. For these, a Python field that merely *has a default*
# may be declared required in TS: the server always populates it, so the frontend
# should not have to null-check it. This waiver deliberately does NOT extend to
# nullable fields — `Optional[X] = None` can serialise as JSON null, which breaks
# that guarantee, and catching exactly that case is the point of the distinction.
RESPONSE_MODELS = {
    "Chord",
    "Voice",
    "Violation",
    "HarmonizeResponse",
    "EngineInfo",
    "EnginesResponse",
}


@dataclass(frozen=True)
class Field:
    optional: bool
    nullable: bool
    type: str | None  # normalised; None when the type could not be resolved


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def parse_ts_aliases(source: str) -> dict[str, str]:
    """Resolve `export type X = ...` so Midi/Beats/Mode compare as their real types."""
    return {
        name: body.strip()
        for name, body in re.findall(r"export\s+type\s+(\w+)\s*=\s*([^;]+);", source)
    }


def normalise_ts_type(raw: str, aliases: dict[str, str], depth: int = 0) -> tuple[str | None, bool]:
    """Return (normalised type, nullable). None means 'could not resolve'."""
    text = raw.strip().rstrip(";").strip()
    if not text or depth > 5:
        return None, False

    nullable = False
    parts = [p.strip() for p in text.split("|")]
    if len(parts) > 1:
        kept = [p for p in parts if p not in ("null", "undefined")]
        nullable = len(kept) != len(parts)
        # A union of string literals becomes an ordered member set rather than a bare
        # "string", so gaining or losing a member is visible. Losing one is the
        # dangerous direction: the backend starts 422-ing payloads the frontend still
        # considers valid, while a members-blind guard reports "in sync".
        if kept and all(re.fullmatch(r'"[^"]*"', p) for p in kept):
            members = sorted(p.strip('"') for p in kept)
            return "enum(" + "|".join(members) + ")", nullable
        if len(kept) != 1:
            return None, nullable
        text = kept[0]

    if text.endswith("[]"):
        inner, _ = normalise_ts_type(text[:-2], aliases, depth + 1)
        return (f"{inner}[]" if inner else None), nullable

    if text in aliases:
        resolved, alias_nullable = normalise_ts_type(aliases[text], aliases, depth + 1)
        return resolved, nullable or alias_nullable

    if text in ("number", "string", "boolean"):
        return text, nullable
    if re.fullmatch(r"\w+", text):
        return text, nullable  # a model name
    return None, nullable


def parse_ts_interfaces(source: str) -> dict[str, dict[str, Field]]:
    """Map interface name -> {field: Field}, top level only."""
    source = _strip_comments(source)
    aliases = parse_ts_aliases(source)
    interfaces: dict[str, dict[str, Field]] = {}

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

        fields: dict[str, Field] = {}
        depth = 0
        for line in body.splitlines():
            stripped = line.strip()
            if depth == 0:
                declared = re.match(r"(\w+)(\?)?\s*:\s*(.+)", stripped)
                if declared:
                    field_type, nullable = normalise_ts_type(declared.group(3), aliases)
                    fields[declared.group(1)] = Field(
                        optional=declared.group(2) == "?" or nullable,
                        nullable=nullable,
                        type=field_type,
                    )
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


def normalise_py_type(annotation: object, depth: int = 0) -> tuple[str | None, bool]:
    """Map a Python annotation onto the TypeScript vocabulary. -> (type, nullable)."""
    if depth > 5:
        return None, False

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (typing.Union, pytypes.UnionType):
        non_none = [a for a in args if a is not type(None)]
        nullable = len(non_none) != len(args)
        if len(non_none) != 1:
            return None, nullable
        inner, inner_nullable = normalise_py_type(non_none[0], depth + 1)
        return inner, nullable or inner_nullable

    if origin in (list, set, tuple):
        if not args:
            return None, False
        inner, _ = normalise_py_type(args[0], depth + 1)
        return (f"{inner}[]" if inner else None), False

    if origin is typing.Literal:
        if args and all(isinstance(a, str) for a in args):
            return "enum(" + "|".join(sorted(args)) + ")", False
        return None, False

    if annotation is bool:
        return "boolean", False
    if annotation in (int, float):
        return "number", False
    if annotation is str:
        return "string", False
    if isinstance(annotation, type):
        return annotation.__name__, False
    return None, False


def parse_py_models() -> dict[str, dict[str, Field]]:
    models: dict[str, dict[str, Field]] = {}
    for name, model in pydantic_models().items():
        fields: dict[str, Field] = {}
        for field_name, info in model.model_fields.items():
            field_type, nullable = normalise_py_type(info.annotation)
            fields[field_name] = Field(
                optional=not info.is_required(),
                nullable=nullable,
                type=field_type,
            )
        models[name] = fields
    return models


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
            ts_field, py_field = ts_fields[field], py_fields[field]

            # A response field that can serialise to null while TS promises a value
            # breaks the frontend's no-null-check guarantee. Report it specifically.
            if py_field.nullable and not ts_field.nullable:
                errors.append(
                    f"{name}.{field}: schema.py can serialise null but types.ts does not allow "
                    f"it — declare `| null` in TS, or make the Python field non-nullable"
                )
            elif ts_field.optional != py_field.optional:
                server_guaranteed = (
                    name in RESPONSE_MODELS
                    and py_field.optional
                    and not py_field.nullable
                    and not ts_field.optional
                )
                if not server_guaranteed:
                    ts_state = "optional" if ts_field.optional else "required"
                    py_state = "optional" if py_field.optional else "required"
                    errors.append(
                        f"{name}.{field}: optionality mismatch — types.ts says {ts_state}, "
                        f"schema.py says {py_state}"
                    )

            if ts_field.type and py_field.type and ts_field.type != py_field.type:
                errors.append(
                    f"{name}.{field}: type mismatch — types.ts says {ts_field.type}, "
                    f"schema.py says {py_field.type}"
                )

    return errors


def test_contract_files_are_in_sync():
    errors = check()
    assert not errors, "contracts/types.ts and contracts/schema.py have drifted:\n  " + "\n  ".join(errors)


def test_guard_detects_nullability_flip_on_a_response_model():
    """Catches a response field that would emit JSON null while TS promises a value.

    This is precisely what the RESPONSE_MODELS waiver used to hide: `inversion: int = 0`
    becoming `Optional[int] = None` leaves the field non-required either way, but changes
    what goes on the wire.
    """
    field = schema.Chord.model_fields["inversion"]
    original = field.annotation
    try:
        field.annotation = typing.Optional[int]
        errors = check()
    finally:
        field.annotation = original
    assert any("Chord.inversion" in e and "null" in e for e in errors), errors


def test_guard_detects_a_type_change():
    field = schema.KeySignature.model_fields["tonic"]
    original = field.annotation
    try:
        field.annotation = str
        errors = check()
    finally:
        field.annotation = original
    assert any("KeySignature.tonic" in e and "type mismatch" in e for e in errors), errors


def test_guard_detects_an_added_enum_member():
    """Mode gaining "dorian" must not read as in sync."""
    import typing as t

    field = schema.KeySignature.model_fields["mode"]
    original = field.annotation
    try:
        field.annotation = t.Literal["major", "minor", "dorian"]
        errors = check()
    finally:
        field.annotation = original
    assert any("KeySignature.mode" in e and "type mismatch" in e for e in errors), errors


def test_guard_detects_a_removed_enum_member():
    """The dangerous direction: the backend 422s payloads the frontend still sends."""
    import typing as t

    field = schema.Violation.model_fields["severity"]
    original = field.annotation
    try:
        field.annotation = t.Literal["info", "warning"]  # "error" dropped
        errors = check()
    finally:
        field.annotation = original
    assert any("Violation.severity" in e and "type mismatch" in e for e in errors), errors


def test_guard_detects_a_field_added_only_in_python():
    from pydantic import Field as PydanticField
    from pydantic import create_model

    extra = create_model("HarmonizeOptions", __base__=schema.HarmonizeOptions, styleWeight=(float, PydanticField(default=1.0)))
    original = schema.HarmonizeOptions
    try:
        schema.HarmonizeOptions = extra
        errors = check()
    finally:
        schema.HarmonizeOptions = original
    assert any("styleWeight" in e for e in errors), errors


if __name__ == "__main__":
    found = check()
    if found:
        print(f"CONTRACT DRIFT ({len(found)} issue(s)):")
        for error in found:
            print(f"  - {error}")
        raise SystemExit(1)
    models = sorted((set(parse_ts_interfaces(TYPES_TS.read_text())) & set(parse_py_models())) - TS_ONLY)
    print(f"Contract in sync across {len(models)} shared models: {', '.join(models)}")
