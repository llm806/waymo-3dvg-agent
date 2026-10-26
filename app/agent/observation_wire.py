"""Short wire aliases; observations retain the full validated domain contract."""
import copy
import json
from typing import Literal

from pydantic import ConfigDict, Field

from app.schemas.contracts import CandidateVisualObservation, SceneVisualObservation, VisualAttribute, VisualObservation, VehicleColors, VISUAL_ATTRIBUTE_VALUES, VISUAL_COLORS


def compact_alias(name):
    return {'candidates': 'c', 'candidate_id': 'id', 'target_visible': 'seen',
        'visibility': 'vis', 'visibility_reason': 'why', 'attributes': 'a',
        'absent_features': 'absent', 'kind': 'k', 'value': 'v'}.get(name, name)


class CompactAttribute(VisualAttribute):
    model_config = ConfigDict(alias_generator=compact_alias)


class RGBVisualObservation(VisualObservation):
    # Legacy domain records stay readable, but a missing new observation must
    # trigger the existing bounded format correction, not silent frame rejection.
    visibility: Literal['clear', 'partial', 'occluded', 'uncertain']
    visibility_reason: str = Field(min_length=1, max_length=300)


class VehicleVisualObservation(RGBVisualObservation):
    # Require explicit paint roles on new vehicle requests. Legacy observations
    # remain readable without assigning body/roof roles to their old colors.
    vehicle_colors: VehicleColors


class CompactCandidateObservation(CandidateVisualObservation):
    model_config = ConfigDict(alias_generator=compact_alias)
    visibility: Literal['clear', 'partial', 'occluded', 'uncertain']
    visibility_reason: str = Field(min_length=1, max_length=300)
    attributes: list[CompactAttribute] = Field(default_factory=list, max_length=6)


class CompactSceneObservation(SceneVisualObservation):
    model_config = ConfigDict(alias_generator=compact_alias)
    candidates: list[CompactCandidateObservation] = Field(default_factory=list, max_length=32)


def unknown_optional_evidence(value, output_type):
    """Discard unsupported observation values, never repair identity or claims.

    Invalid values provide no evidence. Clear their entire attribute kind so a
    malformed conflicting value cannot turn its valid sibling into sole proof.
    The caller still validates every remaining field and the full ID inventory.
    """
    value = copy.deepcopy(value)
    changes = []
    if not isinstance(value, dict):
        return value, changes
    if issubclass(output_type, SceneVisualObservation):
        compact = issubclass(output_type, CompactSceneObservation)
        key = 'c' if compact else 'candidates'
        records = value.get(key)
        if not isinstance(records, list):
            return value, changes
        records = [([key, index], record) for index, record in enumerate(records)]
    elif issubclass(output_type, VisualObservation):
        compact = False
        records = [([], value)]
    else:
        return value, changes
    attr_key, kind_key, value_key = ('a', 'k', 'v') if compact else ('attributes', 'kind', 'value')
    for path, record in records:
        if not isinstance(record, dict):
            continue
        attributes = record.get(attr_key)
        if isinstance(attributes, list) and len(attributes) <= 6:
            invalid_kinds = {a[kind_key] for a in attributes
                if isinstance(a, dict) and set(a) == {kind_key, value_key}
                and isinstance(a[kind_key], str) and a[kind_key] in VISUAL_ATTRIBUTE_VALUES
                and isinstance(a[value_key], str)
                and a[value_key] not in VISUAL_ATTRIBUTE_VALUES[a[kind_key]]}
            if invalid_kinds:
                # Retain malformed structures for strict validation/retry.
                record[attr_key] = [a for a in attributes if not (
                    isinstance(a, dict) and set(a) == {kind_key, value_key}
                    and isinstance(a[kind_key], str) and a[kind_key] in invalid_kinds
                    and isinstance(a[value_key], str))]
                changes.append({'path': [*path, attr_key], 'unknown_kinds': sorted(invalid_kinds)})
                absence_key = 'absent' if compact else 'absent_features'
                absent = record.get(absence_key)
                if ('feature' in invalid_kinds and isinstance(absent, list) and len(absent) <= 6
                        and all(isinstance(v, str) and v in VISUAL_ATTRIBUTE_VALUES['feature'] for v in absent)):
                    # Feature uncertainty cannot leave behind negative proof.
                    record[absence_key] = []
                    changes.append({'path': [*path, absence_key], 'unknown_features': True})
        colors = record.get('vehicle_colors')
        if isinstance(colors, dict):
            for role in ('body', 'roof'):
                color = colors.get(role)
                if isinstance(color, str) and color not in VISUAL_COLORS:
                    colors[role] = None
                    changes.append({'path': [*path, 'vehicle_colors', role], 'unknown_color': True})
    return value, changes


def wire_response_type(messages, output_type):
    if output_type not in {SceneVisualObservation, VisualObservation}:
        return output_type
    content = messages[-1]['content']
    text = content if isinstance(content, str) else content[0]['text']
    try:
        payload = json.loads(text)
    except ValueError:
        return output_type
    title = payload.get('output_schema', {}).get('title')
    if output_type is VisualObservation and title == 'VehicleVisualObservation':
        return VehicleVisualObservation
    if output_type is VisualObservation and title == 'RGBVisualObservation':
        return RGBVisualObservation
    if output_type is SceneVisualObservation and title == 'CompactSceneObservation':
        return CompactSceneObservation
    return output_type
