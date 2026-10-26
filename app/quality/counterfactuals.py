"""Deterministic text interventions with private, conservative geometry expectations."""

import re
from typing import Literal

from pydantic import model_validator

from app.schemas.contracts import Contract, EvidencePacket, GeneratedExpression, GroundResult
from app.quality.contrast import remaining

VERSION = "explicit-geometric-counterfactual-v1"


class CounterfactualCase(Contract):
    case_id: Literal["drop_cue", "replace_cue"]
    expression: str
    changed_predicate_id: str
    expected_predicate_ids: list[str]
    expected_possible_object_ids: list[str]
    expected_known_true_object_ids: list[str]


class CounterfactualPlan(Contract):
    version: str = VERSION
    status: Literal["ready", "not_applicable"]
    reason: str
    original_expression: str
    cases: list[CounterfactualCase]

    @model_validator(mode="after")
    def check_case_coverage(self):
        if self.status == "ready" and [c.case_id for c in self.cases] != ["drop_cue", "replace_cue"]:
            raise ValueError("Ready plans require exactly one deletion and one replacement")
        if self.status == "not_applicable" and self.cases:
            raise ValueError("Uncovered plans cannot claim executable cases")
        return self


def plan_counterfactuals(packet: EvidencePacket, generated: GeneratedExpression) -> CounterfactualPlan:
    """Recognize one explicit cue. Never silently paraphrase an unsupported sentence."""
    def skip(reason):
        return CounterfactualPlan(status="not_applicable", reason=reason, original_expression=generated.expression, cases=[])

    catalog = {p.predicate_id: p for p in packet.predicates}
    if len(catalog) != len(packet.predicates) or len(set(generated.predicate_ids)) != len(generated.predicate_ids):
        raise ValueError("Duplicate predicate identity")
    if not set(generated.predicate_ids) <= catalog.keys():
        return skip("unknown_predicate_reference")
    categories = [catalog[p] for p in generated.predicate_ids if catalog[p].kind == "category"]
    cues = [catalog[p] for p in generated.predicate_ids if catalog[p].kind != "category"]
    if len(categories) != 1 or len(cues) != 1:
        return skip("requires_one_explicit_category_and_one_supported_cue")
    category, cue = categories[0], cues[0]
    selected = [category, cue]
    if remaining(packet.candidates, selected) != [packet.target_object_id] or any(p.matches[packet.target_object_id] is not True for p in selected):
        return skip("original_claims_not_known_true_and_unique")
    if category.value not in {"vehicle", "pedestrian", "sign", "cyclist"}:
        return skip("unsupported_category")
    noun = re.escape(category.value)
    opposite_id = None
    pattern = None
    if cue.kind in {"nearest", "farthest"}:
        rank = "(?:nearest|closest)" if cue.kind == "nearest" else "(?:farthest|furthest)"
        pattern = rf"the {rank} {noun} (?:to|from) the ego vehicle"
        opposite_id = f"{'farthest' if cue.kind == 'nearest' else 'nearest'}:{category.value}"
    elif cue.kind == "lateral" and cue.value in {"left", "right"}:
        pattern = rf"the {noun} (?:to the|on the) {cue.value} (?:side )?of the ego vehicle"
        opposite_id = f"lateral:{'right' if cue.value == 'left' else 'left'}"
    elif cue.kind == "longitudinal" and cue.value in {"front", "behind"}:
        relation = "in front of" if cue.value == "front" else "behind"
        pattern = rf"the {noun} {relation} the ego vehicle"
        opposite_id = f"longitudinal:{'behind' if cue.value == 'front' else 'front'}"
    if pattern is None or re.fullmatch(pattern + r"[.!]?", generated.expression.strip(), re.IGNORECASE) is None:
        return skip("surface_form_or_compositional_scope_not_supported")
    opposite = catalog.get(opposite_id)
    if opposite is None or opposite.matches[packet.target_object_id] is not False:
        return skip("replacement_does_not_definitely_exclude_original_target")
    dropped = remaining(packet.candidates, [category])
    if len(dropped) <= 1:
        return skip("deleting_cue_does_not_increase_ambiguity")

    def case(case_id, expression, predicates):
        possible = remaining(packet.candidates, predicates)
        known_true = [o.object_id for o in packet.candidates if all(p.matches[o.object_id] is True for p in predicates)]
        return CounterfactualCase(case_id=case_id, expression=expression, changed_predicate_id=cue.predicate_id,
                                  expected_predicate_ids=[p.predicate_id for p in predicates],
                                  expected_possible_object_ids=possible, expected_known_true_object_ids=known_true)

    phrase = opposite.phrase
    replacement = phrase[0].upper() + phrase[1:] if phrase.startswith("the ") else f"The {category.value} {phrase}"
    return CounterfactualPlan(status="ready", reason="explicit_single_cue_with_verified_interventions", original_expression=generated.expression,
                             cases=[case("drop_cue", f"The {category.value}.", [category]),
                                    case("replace_cue", replacement + ".", [category, opposite])])


def verify_counterfactual(case: CounterfactualCase, grounded: GroundResult, mapping: dict[str, str]) -> dict:
    errors = []
    labels = grounded.candidate_ids
    mapped = [mapping.get(label) for label in labels]
    if len(labels) != len(set(labels)) or None in mapped:
        errors.append("invalid_or_duplicate_candidate_reference")
    if len(grounded.parsed_predicate_ids) != len(set(grounded.parsed_predicate_ids)) or set(grounded.parsed_predicate_ids) != set(case.expected_predicate_ids):
        errors.append("counterfactual_explicit_claim_parse_mismatch")
    if set(mapped) != set(case.expected_possible_object_ids):
        errors.append("counterfactual_candidate_set_mismatch")
    if grounded.unsupported_claims:
        errors.append("counterfactual_reader_reports_unsupported_claim")
    actual = set(mapped)
    expected = set(case.expected_possible_object_ids)
    unknown = expected - set(case.expected_known_true_object_ids)
    return {"case_id": case.case_id, "expression": case.expression, "passed": not errors, "errors": errors,
            "claim_parse_passed": set(grounded.parsed_predicate_ids) == set(case.expected_predicate_ids) and not grounded.unsupported_claims,
            "candidate_soundness_passed": None not in actual and actual <= expected,
            "candidate_completeness_passed": expected <= actual,
            "omitted_unknown_possible_count": len(unknown - actual),
            "expected_predicate_ids": case.expected_predicate_ids, "parsed_predicate_ids": grounded.parsed_predicate_ids,
            "expected_possible_object_ids": case.expected_possible_object_ids, "actual_object_ids": mapped,
            "known_true_count": len(case.expected_known_true_object_ids),
            "unknown_possible_count": len(case.expected_possible_object_ids) - len(case.expected_known_true_object_ids),
            "failure_scope": None if not errors else "grounder explicit-claim or conservative-policy consistency; not independent semantic gold"}


def counterfactual_summary(plan: CounterfactualPlan, checks: list[dict], incomplete_reason=None) -> dict:
    complete = [c["case_id"] for c in checks] == [c.case_id for c in plan.cases]
    status = "not_applicable" if plan.status == "not_applicable" else "incomplete" if not complete else "passed" if all(c["passed"] for c in checks) else "failed"
    return {"version": VERSION, "status": status, "reason": incomplete_reason or plan.reason,
            "passed": True if status == "passed" else False if status == "failed" else None,
            "planned_cases": len(plan.cases), "completed_cases": len(checks), "checks": checks,
            "scope": "same-model oracle-candidate semantic interventions; not independent label-quality calibration"}
