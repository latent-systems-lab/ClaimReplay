"""Schema and information-boundary validation."""

from __future__ import annotations

from typing import Any, Mapping

from .models import EvidencePolicyView, PolicyView, SupportLevel


_FORBIDDEN_KEY_PARTS = (
    "assessment",
    "gold",
    "label",
    "support",
    "verdict",
    "positive_target",
    "exclusion_reason",
)
_FORBIDDEN_VALUES = {
    *(level.value for level in SupportLevel),
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO",
    "SUPPORT",
    "CONTRADICT",
}


def _scan(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"Policy-visible field {path}.{key} may reveal evaluator-only state")
            _scan(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.strip().upper() in _FORBIDDEN_VALUES:
        raise ValueError(f"Policy-visible value at {path} contains a native/canonical assessment label")


def assert_policy_view_gold_free(
    view: PolicyView | EvidencePolicyView,
    *,
    observable_feature_schema: Mapping[str, str] | None = None,
) -> None:
    """Reject known assessment channels in structured features and metadata.

    Claim/evidence text and opaque IDs are permitted content, not label fields.
    This guard cannot detect labels encoded under innocent feature names;
    adapter authors remain responsible for feature provenance.
    """

    if hasattr(view, "gold_assessments") or hasattr(view, "gold_hash"):
        raise ValueError("Policy view exposes gold assessments")
    _scan(view.metadata.to_dict(), "policy_view.metadata")
    for unit in view.units:
        if hasattr(unit, "observable_features"):
            _scan(unit.observable_features.to_dict(), f"policy_view.units[{unit.unit_id}].observable_features")
        if hasattr(unit, "provenance") or hasattr(unit, "evidence"):
            raise ValueError("Policy-visible units expose evaluator provenance or benchmark evidence")

    schema = dict(observable_feature_schema or {})
    _scan(schema, "policy.observable_feature_schema")
    allowed_kinds = {"number", "string"}
    if any(kind not in allowed_kinds for kind in schema.values()):
        raise ValueError(f"Unsupported observable feature kind; expected {sorted(allowed_kinds)}")
    if not isinstance(view, PolicyView):
        if schema:
            raise ValueError("Evidence-aware views do not expose observable feature dictionaries")
        return
    for unit in view.units:
        features = unit.observable_features.to_dict()
        for name, kind in schema.items():
            if name not in features:
                raise ValueError(f"Unit {unit.unit_id} lacks configured observable feature {name!r}")
            value = features[name]
            if kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise ValueError(f"Unit {unit.unit_id} feature {name!r} must be numeric")
            if kind == "string" and not isinstance(value, str):
                raise ValueError(f"Unit {unit.unit_id} feature {name!r} must be a string")
