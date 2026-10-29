"""Conservative candidate elimination; unknown facts cannot exclude objects."""

import itertools
import math

from app.schemas.contracts import ContrastPlan, ObjectFact, Predicate

POLICY = {"axis_guard_m": 1.0, "rank_range_gap_m": 2.0,
          "ego_direction_policy": "oriented-footprint-axis-guard-v1",
          "description_policy": "evidence-guided-ego-or-explicit-camera-v3",
          "rank_length_gap_m": 0.5, "vertical_rank_gap_m": 0.5, "max_extra_predicates": 3,
          "max_views": 3, "min_crop_area_px": 256,
          "calibration_status": "prototype defaults; not calibrated on independent gold"}

EGO_PROXIMITY_POLICY = 'native-ego-proximity-v1'
EGO_NEAR_ID = 'range:ego_near'
EGO_WORDING_POLICY = 'complete-ego-direction-v1'
PROVEN_EGO_WORDING_POLICY = 'proven-ego-components-v2'


def expand_coordinated_ego_clause(text):
    """Parse shared complements without changing the saved sentence or facts."""
    import re
    longitudinal = r'(?P<longitudinal>ahead(?:\s+of)?|in\s+front\s+of|behind|to\s+the\s+rear\s+of)'
    lateral = r'(?:to\s+the\s+)?(?P<lateral>left|right)\s+of'
    ego = r'\s+(?:the\s+)?ego\s+vehicle\b'
    def expand(match):
        direction = re.sub(r'\s+', ' ', match['longitudinal'].lower())
        first = 'behind' if direction in {'behind', 'to the rear of'} else 'ahead of'
        return first + ' the ego vehicle and to the ' + match['lateral'].lower() + ' of the ego vehicle'
    # Only share an explicit ego complement across adjacent direction phrases.
    # A named object between the directions must remain a separate landmark.
    for first, second in ((longitudinal, lateral), (lateral, longitudinal)):
        text = re.sub(r'\b' + first + r'\s+and\s+' + second + ego, expand, text, flags=re.I)
    return text


def complete_ego_direction_ids(packet, ids):
    """Keep both proved ego components without changing geometric truth tables.

    V2 retains a companion direction only when the native rotated footprint
    clears the existing axis guard. Frozen V1 keeps its 22.5 degree wording
    boundary. This never filters competitors or infers direction from RGB.
    """
    ids = list(dict.fromkeys(ids))
    policy = packet.task.get('ego_wording_policy')
    if policy not in {EGO_WORDING_POLICY, PROVEN_EGO_WORDING_POLICY}:
        return ids
    axes = {'longitudinal:front':0, 'longitudinal:behind':0, 'lateral:left':1, 'lateral:right':1}
    x, y, _ = packet.target.center_m
    for pid in list(ids):
        if pid not in axes:
            continue
        axis = axes[pid]
        along, across = (x,y) if axis == 0 else (y,x)
        companion = ('lateral:left' if y > 0 else 'lateral:right') if axis == 0 else (
            'longitudinal:front' if x > 0 else 'longitudinal:behind')
        supported = ego_direction_match(packet.target, 1-axis, (y if axis == 0 else x) > 0) is True
        if (supported if policy == PROVEN_EGO_WORDING_POLICY else abs(across) > abs(along) * math.tan(math.pi/8)):
            if companion not in ids:
                ids.append(companion)
    return ids


def ego_wording_errors(packet, ids):
    if packet.task.get('ego_wording_policy') not in {EGO_WORDING_POLICY, PROVEN_EGO_WORDING_POLICY}:
        return []
    complete = complete_ego_direction_ids(packet, ids)
    errors = ['incomplete_ego_direction'] if set(complete) - set(ids) else []
    if any(i.startswith(('longitudinal:', 'lateral:')) for i in ids) and (
            not packet.target.lidar_points or unsupported_ego_directions(packet.target, complete)):
        errors.append('ego_direction_not_supported')
    return errors


def ego_near_match(obj):
    """Fixed conservative 3D center radius about the native vehicle origin.

    <=5m supports close; 5..7m is deliberately unknown; >7m excludes close.
    These development thresholds apply to every native candidate, not a target.
    No camera size, ego body dimensions, or bumper clearance is inferred.
    """
    if not obj.geometry_valid or not obj.lidar_points or min(obj.size_m) <= 0:
        return None
    distance = math.dist(obj.center_m, (0,0,0))
    return True if distance <= 5 else False if distance > 7 else None


def ego_near_catalog(packet):
    if packet.task.get('ego_proximity_policy') != EGO_PROXIMITY_POLICY:
        return []
    return [Predicate(predicate_id=EGO_NEAR_ID,kind='range',value='ego_near',
        phrase='close to the ego vehicle', source=EGO_PROXIMITY_POLICY + ': native 3D center radius <=5m; 5..7m unknown; >7m false',
        matches={o.object_id:ego_near_match(o) for o in packet.candidates})]


def states_ego_near(text):
    import re
    return bool(re.search(r'\b(?:very\s+)?(?:close to|near)\s+(?:the\s+)?ego vehicle\b',text,re.I)
        and not re.search(r'\b(?:not|never|no longer)\s+(?:very\s+)?(?:close to|near)\b',text,re.I))


def ego_near_errors(packet, expression, ids):
    stated, selected = states_ego_near(expression), EGO_NEAR_ID in ids
    errors = []
    if stated != selected:
        errors.append('ego_distance_claim_mismatch')
    if (stated or selected) and (packet.task.get('ego_proximity_policy') != EGO_PROXIMITY_POLICY
            or ego_near_match(packet.target) is not True):
        errors.append('ego_distance_not_supported')
    return errors


def ego_direction_match(obj: ObjectFact, axis: int, positive: bool) -> bool | None:
    """A marginally offset center cannot put a straddling object on one side.

    Use the rotated footprint and the existing guard. An extent overlapping the
    axis/guard remains unknown in either direction, including for competitors.
    """
    if not obj.geometry_valid:
        return None
    length, width, _ = obj.size_m
    cosine, sine = abs(math.cos(obj.heading_rad)), abs(math.sin(obj.heading_rad))
    radius = (cosine * length + sine * width) / 2 if axis == 0 else (sine * length + cosine * width) / 2
    signed = obj.center_m[axis] * (1 if positive else -1)
    guard = POLICY['axis_guard_m']
    return True if signed - radius > guard else False if signed + radius < -guard else None


def unsupported_ego_directions(target: ObjectFact, predicate_ids) -> list[str]:
    """Recheck saved claims from native geometry rather than cached predicates."""
    axes = {'longitudinal:front': (0, True), 'longitudinal:behind': (0, False),
            'lateral:left': (1, True), 'lateral:right': (1, False)}
    return sorted(id_ for id_ in set(predicate_ids) if id_ in axes
        and ego_direction_match(target, *axes[id_]) is not True)


def guarded_rank(objects, category, measure, descending, gap):
    """Unknown class can belong to the ranking; invalid geometry cannot exclude it."""
    possible=[o for o in objects if o.category in {category,'unknown'}]
    known=[o for o in possible if o.category==category and o.geometry_valid]
    matches={o.object_id:False for o in objects}
    if not known:
        matches.update({o.object_id:None for o in possible})
        return matches
    sign=-1 if descending else 1
    first=min(known,key=lambda o:(sign*measure(o),o.object_id))
    delta=lambda o:sign*(measure(o)-measure(first))
    stable=len(possible)>1 and all(o.geometry_valid and delta(o)>gap for o in possible if o.object_id!=first.object_id)
    for o in possible:
        matches[o.object_id]=None if not o.geometry_valid else False if delta(o)>gap else True if stable and o.object_id==first.object_id else None
    return matches


def predicate_catalog(objects: list[ObjectFact]) -> list[Predicate]:
    result = []

    def add(kind, value, phrase, matches, source):
        result.append(Predicate(predicate_id=f"{kind}:{value}", kind=kind, value=value,
                                phrase=phrase, source=source, matches=matches))

    for category in sorted({o.category for o in objects}):
        add("category", category, category, {o.object_id: None if o.category=='unknown' and category!='unknown' else o.category == category for o in objects},
            "lidar_box:[LiDARBoxComponent].type")
    for kind, coordinate, positive, value, phrase in [
        ("longitudinal", 0, True, "front", "in front of the ego vehicle"),
        ("longitudinal", 0, False, "behind", "behind the ego vehicle"),
        ("lateral", 1, True, "left", "to the left of the ego vehicle"),
        ("lateral", 1, False, "right", "to the right of the ego vehicle"),
    ]:
        add(kind, value, phrase, {o.object_id: ego_direction_match(o, coordinate, positive) for o in objects},
            f"lidar_box:oriented footprint along {'x' if coordinate == 0 else 'y'}; frame vehicle coordinates; extent + axis guard")
    # Keep the symmetric rank vocabulary for diagnostics of historical records.
    # New plans below cannot use distance ranks or absolute distance bins.
    for category in sorted({o.category for o in objects}-{'unknown'}):
        for kind, phrase, measure, descending, gap in [
            ("nearest", f"the nearest {category} to the ego vehicle", lambda o: o.range_xy_m, False, 2.0),
            ("farthest", f"the farthest {category} from the ego vehicle", lambda o: o.range_xy_m, True, 2.0),
            ("longest", f"the longest {category} in the scene", lambda o: o.size_m[0], True, 0.5),
        ]:
            matches=guarded_rank(objects,category,measure,descending,gap)
            add(kind, category, phrase, matches, f"all native same-class frame boxes; {kind}; guard {gap}m")
    return result


def remaining(objects: list[ObjectFact], predicates: list[Predicate]) -> list[str]:
    return [o.object_id for o in objects if all(p.matches[o.object_id] is not False for p in predicates)]


def contrast_plan(objects: list[ObjectFact], target_id: str, catalog: list[Predicate]) -> ContrastPlan:
    target = next(o for o in objects if o.object_id == target_id)
    category = next(p for p in catalog if p.kind == "category" and p.value == target.category)
    # Retain distance ranks in the diagnostic vocabulary for old records, but
    # never use them to select targets or construct new descriptions/anchors.
    eligible = [p for p in catalog if p.kind not in {"category", "range", "nearest", "farthest", "nearest_to_anchor"}
                and p.matches[target_id] is True]
    chosen = [category]
    status = "ambiguous"
    best_partial = (len(remaining(objects, chosen)), 0, ())
    partial_combo = ()
    # A candidate survives a conjunction iff its bit survives every predicate.
    # This preserves unknown-as-possible semantics while avoiding repeated Python
    # scans of every object for thousands of combinations in each catalog frame.
    masks = {p.predicate_id: sum(1 << i for i, o in enumerate(objects) if p.matches[o.object_id] is not False)
             for p in [category, *eligible]}
    base_mask = masks[category.predicate_id]
    target_mask = 1 << next(i for i, o in enumerate(objects) if o.object_id == target_id)
    for count in range(POLICY["max_extra_predicates"] + 1):
        solutions = []
        for combo in itertools.combinations(eligible, count):
            mask = base_mask
            for p in combo:
                mask &= masks[p.predicate_id]
            if mask == target_mask:
                solutions.append(combo)
            survivors_count = mask.bit_count()
            if (survivors_count, count) <= best_partial[:2]:
                score = (survivors_count, count, tuple(p.predicate_id for p in combo))
                if score < best_partial:
                    best_partial, partial_combo = score, combo
        if solutions:
            # Prefer short, non-ranking spatial descriptions among equal-size plans.
            combo = min(solutions, key=lambda ps: (sum(p.kind in ("nearest", "farthest", "longest") for p in ps),
                                                  sum(len(p.phrase) for p in ps), tuple(p.predicate_id for p in ps)))
            chosen += list(combo)
            status = "unique_in_gt_candidates"
            break
    if status == "ambiguous":
        # Keep useful progress without treating partial elimination as success.
        chosen += list(partial_combo)
    negatives = []
    for o in objects:
        if o.object_id == target_id or o.category != target.category:
            continue
        valid = target.geometry_valid and o.geometry_valid and all(a > 0 and b > 0 for a, b in zip(target.size_m, o.size_m))
        distance = math.dist(target.center_m[:2], o.center_m[:2]) if valid else None
        size_difference = sum(abs(math.log(a / b)) for a, b in zip(target.size_m, o.size_m)) if valid else None
        negatives.append({"object_id": o.object_id, "center_distance_xy_m": distance,
                          "size_log_difference": size_difference, "score": distance / 10 + size_difference if valid else None,
                          "excluded_by": [p.predicate_id for p in chosen if p.matches[o.object_id] is False]})
    negatives.sort(key=lambda n: (n["score"] if n["score"] is not None else float("inf"), n["object_id"]))
    return ContrastPlan(status=status, selected_predicate_ids=[p.predicate_id for p in chosen],
                        remaining_object_ids=remaining(objects, chosen),
                        removal_tests={p.predicate_id: remaining(objects, [q for q in chosen if q != p]) for p in chosen},
                        same_class_count=sum(o.category == target.category for o in objects),
                        hard_negatives=negatives[:5])
