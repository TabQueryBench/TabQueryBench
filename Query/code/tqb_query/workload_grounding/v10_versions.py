"""Model-specific versioning for model-grounded workload lines.

V10 lets the model bind placeholders of a fixed template set; V11 additionally
lets the model choose which templates to ground.  Both lines share one slot
table, so ``v10.1.1_claude-opus-5`` and ``v11.1.1_claude-opus-5`` always name the same model.

A version name reads ``<line>.<vendor>.<model>[-<run>]_<model name>``:

    v11.2.2_glm-5.3          first run of glm-5.3 on V11
    v11.2.2-2_glm-5.3        second run of the same model
    v10.1.4_claude-haiku-4-5 first run of haiku on V10

The vendor digit groups providers (1 Anthropic, 2 Z.AI, 3 OpenAI) and the
model digit picks the model inside it.  The first run carries no run suffix.
The model name is part of the name so a directory, file or score row says
which model produced it without a lookup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


V10_LINE_VERSION = "v10"
V11_LINE_VERSION = "v11"
AGENT_LINE_VERSIONS = (V10_LINE_VERSION, V11_LINE_VERSION)

# Model names are lowercase and never contain "_", which is what separates them
# from the numeric part and from anything a caller appends (run ids, dates).
MODEL_NAME_PATTERN = r"[a-z0-9][a-z0-9.\-]*"
GROUNDING_VERSION_PATTERN = (
    rf"(?:v10|v11)\.[0-9]\.[0-9](?:-[1-9][0-9]*)?_{MODEL_NAME_PATTERN}"
)
GROUNDING_VERSION_RE = re.compile(rf"^{GROUNDING_VERSION_PATTERN}$")
_VERSION_PARTS_RE = re.compile(
    rf"^(?P<family>v10|v11)\.(?P<slot>[0-9]\.[0-9])(?:-(?P<run>[1-9][0-9]*))?"
    rf"(?:_(?P<model>{MODEL_NAME_PATTERN}))?$"
)

# The x.y slot is deliberately assigned here rather than inferred from a model
# name.  That keeps artifact provenance stable when provider aliases change.
MODEL_VERSION_SLOTS: dict[str, str] = {
    # 1.x Anthropic
    "claude-opus-5": "1.1",
    "claude-sonnet-4-6": "1.2",
    "claude-fable-5-1": "1.3",
    "claude-haiku-4-5": "1.4",
    # 2.x Z.AI
    "glm-5.3-flash": "2.1",
    "glm-5.3": "2.2",
    # 3.x OpenAI
    "gpt-5.4": "3.1",
    "gpt-5.5": "3.2",
    "gpt-5.6-sol": "3.3",
    "gpt-5.6-terra": "3.4",
    "gpt-5.6-luna": "3.5",
}

# Models that produced artifacts before the table above was settled but no longer
# own a slot.  They keep resolving so those artifacts stay valid; the model name in
# the version keeps them apart from the slot's current owner.
LEGACY_MODEL_SLOTS: dict[str, str] = {
    "claude-sonnet-5": "1.2",
}

MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-5",
    "opus5": "claude-opus-5",
    "claude/opus": "claude-opus-5",
    "sonnet": "claude-sonnet-4-6",
    "sonnet4.6": "claude-sonnet-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude/sonnet": "claude-sonnet-4-6",
    "sonnet5": "claude-sonnet-5",
    "fable": "claude-fable-5-1",
    "fable-5.1": "claude-fable-5-1",
    "fable 5.1": "claude-fable-5-1",
    "claude/fable": "claude-fable-5-1",
    "haiku": "claude-haiku-4-5",
    "claude/haiku": "claude-haiku-4-5",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "glm": "glm-5.3",
    "glm5.3": "glm-5.3",
    "zai/glm-5.3": "glm-5.3",
    "z-ai/glm-5.3": "glm-5.3",
    "flash": "glm-5.3-flash",
    "glm-flash": "glm-5.3-flash",
    "glm5.3-flash": "glm-5.3-flash",
    "zai/glm-5.3-flash": "glm-5.3-flash",
    "openai/gpt-5.4": "gpt-5.4",
    "gpt5.4": "gpt-5.4",
    "openai/gpt-5.5": "gpt-5.5",
    "gpt5.5": "gpt-5.5",
    "gpt5.6-sol": "gpt-5.6-sol",
    "gpt5.6-terra": "gpt-5.6-terra",
    "gpt5.6-luna": "gpt-5.6-luna",
}


@dataclass(frozen=True)
class V10ModelVersion:
    requested_model: str
    resolved_model: str
    grounding_version: str
    line_version: str = V10_LINE_VERSION
    mapping_source: str = "central_model_grounding_versions"
    run: int = 1

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def canonical_model_name(model: str) -> str:
    requested = str(model or "").strip().lower()
    if not requested:
        raise ValueError("A model is required for model-grounded workload lines")
    return MODEL_ALIASES.get(requested, requested)


def model_slot(model: str) -> str | None:
    resolved = canonical_model_name(model)
    return MODEL_VERSION_SLOTS.get(resolved) or LEGACY_MODEL_SLOTS.get(resolved)


def format_grounding_version(line_family: str, model: str, run: int = 1) -> str:
    resolved = canonical_model_name(model)
    slot = model_slot(resolved)
    if slot is None:
        raise ValueError(f"Model {model!r} has no version slot; add it to MODEL_VERSION_SLOTS")
    if run < 1:
        raise ValueError(f"Run number must be >= 1, got {run}")
    suffix = f"-{run}" if run > 1 else ""
    return f"{line_family}.{slot}{suffix}_{resolved}"


def parse_grounding_version(value: str) -> dict[str, str | int | None]:
    """Split a full or partial version name; the model part may be omitted."""
    match = _VERSION_PARTS_RE.fullmatch(str(value or "").strip().lower())
    if not match:
        raise ValueError(
            f"Invalid grounding version {value!r}; expected e.g. v11.2.2_glm-5.3 or v11.2.2-2_glm-5.3"
        )
    run = int(match.group("run") or 1)
    if match.group("run") == "1":
        raise ValueError(f"{value!r}: the first run carries no run suffix")
    return {
        "family": match.group("family"),
        "slot": match.group("slot"),
        "run": run,
        "model": match.group("model"),
    }


def validate_grounding_version(value: str, line_family: str = V10_LINE_VERSION) -> str:
    """Return a full version name, rejecting one whose model does not own its slot."""
    parts = parse_grounding_version(value)
    if parts["family"] != line_family:
        raise ValueError(f"{value!r} is not a {line_family.upper()} version")
    if not parts["model"]:
        raise ValueError(f"{value!r} names no model; expected {value}_<model>")
    expected = format_grounding_version(line_family, str(parts["model"]), int(parts["run"]))
    if expected != str(value).strip().lower():
        raise ValueError(f"{value!r} does not match the slot table; {parts['model']} is {expected}")
    return expected


def validate_v10_grounding_version(value: str) -> str:
    return validate_grounding_version(value, V10_LINE_VERSION)


def resolve_model_version(
    model: str,
    *,
    line_family: str = V10_LINE_VERSION,
    grounding_version: str = "",
) -> V10ModelVersion:
    """Name the artifact version for ``model``.

    ``grounding_version`` may be a full name, or just ``v11.2.2`` / ``v11.2.2-2``
    to pick the run; either way it must agree with the model's slot.
    """
    if line_family not in AGENT_LINE_VERSIONS:
        raise ValueError(f"Unsupported model-grounded line {line_family!r}; expected one of {AGENT_LINE_VERSIONS}")
    requested = str(model or "").strip()
    resolved = canonical_model_name(requested)
    slot = model_slot(resolved)
    if slot is None:
        raise ValueError(
            f"Model {requested!r} has no {line_family.upper()} version slot; add it to MODEL_VERSION_SLOTS"
        )
    run = 1
    source = "central_model_grounding_versions"
    if grounding_version:
        parts = parse_grounding_version(grounding_version)
        if parts["family"] != line_family:
            raise ValueError(f"{grounding_version!r} is not a {line_family.upper()} version")
        if parts["slot"] != slot:
            raise ValueError(
                f"Model {requested!r} owns slot {line_family}.{slot}, not requested {grounding_version!r}"
            )
        if parts["model"] and parts["model"] != resolved:
            raise ValueError(f"{grounding_version!r} names {parts['model']}, but the planner model is {resolved}")
        run = int(parts["run"])
        source = "repeat_run_version" if run > 1 else "explicit_grounding_version"
    return V10ModelVersion(
        requested_model=requested,
        resolved_model=resolved,
        grounding_version=format_grounding_version(line_family, resolved, run),
        line_version=line_family,
        mapping_source=source,
        run=run,
    )


def resolve_v10_model_version(model: str, *, grounding_version: str = "") -> V10ModelVersion:
    return resolve_model_version(model, line_family=V10_LINE_VERSION, grounding_version=grounding_version)


def _check_slot_table() -> None:
    owners: dict[str, str] = {}
    for model_name, slot in MODEL_VERSION_SLOTS.items():
        if slot in owners:
            raise ValueError(f"Slot {slot} assigned to both {owners[slot]} and {model_name}")
        owners[slot] = model_name
        for family in AGENT_LINE_VERSIONS:
            validate_grounding_version(format_grounding_version(family, model_name), family)


_check_slot_table()
