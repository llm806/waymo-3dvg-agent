"""Symmetric anchor relations; every anchor must be independently identifiable."""

import hashlib
import itertools
import math

from app.schemas.contracts import EvidencePacket, Predicate
from app.quality.contrast import POLICY, contrast_plan, remaining, guarded_rank

VERSION = "safe-anchor-relations-v5-source-salience"
VEHICLE_HEADING_POLICY = 'vehicle-heading-landmark-v1'
VEHICLE_FORWARD_REGION_POLICY = 'vehicle-forward-region-v1'
LOCAL_LANDMARK_POLICY = 'local-landmark-selection-v1'
MAX_LANDMARK_DISTANCE_M = 20.0


def local_landmark_pair(packet, obj, anchor):
    """Reference suitability only; never a distance claim or candidate exclusion."""
    if packet.task.get('landmark_locality_policy') != LOCAL_LANDMARK_POLICY:
        return True
    return bool(obj is not None and anchor is not None and obj.object_id != anchor.object_id
        and obj.geometry_valid and anchor.geometry_valid and obj.lidar_points and anchor.lidar_points
        and math.dist(obj.center_m,anchor.center_m) <= MAX_LANDMARK_DISTANCE_M)


def local_landmark_predicate(packet, predicate):
    if (packet.task.get('landmark_locality_policy') != LOCAL_LANDMARK_POLICY
            or not predicate.predicate_id.startswith('image_joint_heading:')):
        return True
    import json
    proof=json.loads(predicate.value)
    native={o.object_id:o for o in packet.candidates}
    matcher=vehicle_forward_region_match if proof['direction'] in {'ahead','rearward'} else vehicle_heading_match
    sign=1 if proof['direction'] in {'front','ahead'} else -1
    witnesses=[native[oid] for oid in proof.get('confirmed_anchor_ids',[]) if oid in native
        and matcher(packet.target,native[oid],sign) is True]
    # Do not silently make "a truck" mean only a nearby truck. Reject the
    # writing plan if its confirmed supporting anchors include a remote one.
    # The original relation table and every unknown competitor remain intact.
    return bool(witnesses) and all(local_landmark_pair(packet,packet.target,a) for a in witnesses)


def vehicle_forward_region_match(obj, anchor, sign=1):
    """Ahead/behind along the landmark heading, without claiming direct alignment.

    Preserve full-box longitudinal separation and same-level guards. This is
    a separate, weaker claim; it never replaces the directly-ahead predicate.
    """
    if obj.object_id == anchor.object_id:
        return False
    if (anchor.category != 'vehicle' or not anchor.geometry_valid or not obj.geometry_valid
            or not anchor.lidar_points or not obj.lidar_points or min(*anchor.size_m,*obj.size_m)<=0):
        return None
    dx,dy=(obj.center_m[i]-anchor.center_m[i] for i in (0,1))
    forward=sign*(dx*math.cos(anchor.heading_rad)+dy*math.sin(anchor.heading_rad))
    angle=obj.heading_rad-anchor.heading_rad
    radius=(abs(math.cos(angle))*obj.size_m[0]+abs(math.sin(angle))*obj.size_m[1])/2
    limit=anchor.size_m[0]/2+radius+POLICY['axis_guard_m']
    height=abs(obj.center_m[2]-anchor.center_m[2])
    if forward < -limit or height > (obj.size_m[2]+anchor.size_m[2])/2+POLICY['axis_guard_m']:
        return False
    return True if forward > limit and height < min(obj.size_m[2],anchor.size_m[2])/2 else None


def vehicle_heading_match(obj, anchor, sign=1):
    """Directly ahead/behind in the vehicle landmark's own native frame.

    Waymo heading rotates +x onto the box front-face normal (label.proto).
    Disjoint longitudinal extents plus the existing metre guard establish front
    or rear; aligned centres must lie in the central half of both lateral
    footprints. Marginal alignment stays unknown, not false. These are frozen
    development guards, not independently calibrated measurements or motion.
    """
    if obj.object_id == anchor.object_id:
        return False
    if (anchor.category != 'vehicle' or not anchor.geometry_valid or not obj.geometry_valid
            or not anchor.lidar_points or min(*anchor.size_m, *obj.size_m) <= 0):
        return None
    cosine, sine = math.cos(anchor.heading_rad), math.sin(anchor.heading_rad)
    dx, dy = (obj.center_m[i] - anchor.center_m[i] for i in (0, 1))
    forward, lateral = sign * (dx*cosine + dy*sine), abs(-dx*sine + dy*cosine)
    angle = obj.heading_rad - anchor.heading_rad
    long_radius = (abs(math.cos(angle))*obj.size_m[0] + abs(math.sin(angle))*obj.size_m[1])/2
    side_radius = (abs(math.sin(angle))*obj.size_m[0] + abs(math.cos(angle))*obj.size_m[1])/2
    front_limit = anchor.size_m[0]/2 + long_radius + POLICY['axis_guard_m']
    side_limit = min(anchor.size_m[1]/2, side_radius)/2
    height_gap = abs(obj.center_m[2]-anchor.center_m[2])
    if (forward < -front_limit or lateral > anchor.size_m[1]/2 + side_radius + POLICY['axis_guard_m']
            or height_gap > (obj.size_m[2]+anchor.size_m[2])/2 + POLICY['axis_guard_m']):
        return False
    return True if forward > front_limit and lateral < side_limit and height_gap < min(obj.size_m[2],anchor.size_m[2])/2 else None


def vehicle_heading_anchor(packet, obj, record, box):
    """A heading needs a recognizable whole vehicle with an actual RGB link."""
    from app.quality.visibility_policy import usable_rgb_observation, allows_partial
    return bool(packet.task.get('vehicle_heading_policy') == VEHICLE_HEADING_POLICY
        and obj is not None and obj.category == 'vehicle' and obj.geometry_valid and obj.lidar_points
        and min(obj.size_m) > 0 and usable_rgb_observation(record,allow_partial=allows_partial(packet))
        and not record.get('visibility_conflict') and box
        and box.get('identity_source') in {'native_box_point_correspondence','official_camera_lidar_association'}
        and not any(a.get('kind') == 'subtype' and a.get('value') in
            {'bus section','excavator body','excavator boom','excavator'} for a in record.get('attributes',[])))


def vertical_rank_catalog(packet: EvidencePacket) -> tuple[list[Predicate], list[dict]]:
    """Height order in named ego regions, including zero-point/unknown competitors.

    Every region and ranking is enumerated symmetrically, never fitted to a target.
    The guard is a development default, not an independently calibrated threshold.
    """
    by_id = {p.predicate_id: p for p in packet.predicates}
    predicates, proofs = [], []
    guard = packet.policy.get("vertical_rank_gap_m", POLICY["vertical_rank_gap_m"])
    for category in sorted({o.category for o in packet.candidates} - {"unknown"}):
        for longitudinal, lateral, distance in itertools.product(("front", "behind"), ("left", "right"), ("all",)):
            qualifiers = [f"longitudinal:{longitudinal}", f"lateral:{lateral}"]
            if not all(q in by_id for q in qualifiers):
                continue
            members = [o for o in packet.candidates if o.category in {category,'unknown'}
                       and all(by_id[q].matches[o.object_id] is not False for q in qualifiers)]
            if len(members) < 2 or any(not o.geometry_valid for o in members):
                continue
            region_phrase = " and ".join(by_id[q].phrase for q in qualifiers)
            for direction, sign in (("lowest", 1), ("highest", -1)):
                ordered = sorted(members, key=lambda o: (sign * o.center_m[2], o.object_id))
                first, second = ordered[:2]
                stable = sign * (second.center_m[2] - first.center_m[2]) > guard
                if not stable or first.category!=category or not all(by_id[q].matches[first.object_id] is True for q in qualifiers):
                    continue
                matches = {o.object_id: False for o in packet.candidates}
                for o in members:
                    delta = sign * (o.center_m[2] - first.center_m[2])
                    matches[o.object_id] = True if o.object_id == first.object_id else False if delta > guard else None
                predicate_id = f"vertical_rank:{direction}:{category}:{longitudinal}:{lateral}:{distance}"
                predicates.append(Predicate(predicate_id=predicate_id, kind="vertical_rank", value=direction,
                    phrase=f"the {direction}-positioned {category} {region_phrase}", matches=matches,
                    source=f"all native same-class frame center.z; region {qualifiers}; possible membership retained; center gap guard {guard}m; not object size"))
                proofs.append({"predicate_id": predicate_id, "qualifier_predicate_ids": qualifiers,
                    "comparison_object_ids": [o.object_id for o in ordered], "center_gap_m": abs(first.center_m[2] - second.center_m[2]),
                    "guard_m": guard, "zero_point_competitors": [o.object_id for o in ordered if not o.lidar_points],
                    "calibration_status": "development default; point-cloud observability and independent quality not certified"})
    return predicates, proofs


def relation_catalog(packet: EvidencePacket, observations=None) -> tuple[list[Predicate], list[dict]]:
    objects = packet.candidates
    if packet.task.get('annotation_search') and observations:
        from app.quality.description_search import camera_observations
        from app.quality.visibility_policy import checks_rgb_visibility
        if checks_rgb_visibility(packet):
            camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
            observations = camera_observations(packet, camera, observations)
    unary = [p for p in packet.predicates if p.kind in ("category", "longitudinal", "lateral", "longest")]
    from app.quality.description_search import scene_reference_policy
    if scene_reference_policy(packet):
        unary = [p for p in unary if p.kind != 'longest']
    anchors = []
    for obj in objects:
        if obj.category == "unknown" or not obj.geometry_valid or not obj.lidar_points or obj.lidar_points <= 0:
            continue
        # Recognition is a necessary condition for an annotation landmark.
        # Reject unavailable anchors before enumerating descriptors against the
        # full scene. Those objects remain competitors in every proof below.
        if packet.task.get('annotation_search') and not (observations or {}).get(obj.object_id, {}).get('target_visible'):
            continue
        category = next(p for p in unary if p.kind == "category" and p.value == obj.category)
        options = [p for p in unary if p.kind != "category" and p.matches[obj.object_id] is True]
        descriptions = []
        for count in range(3 if packet.task.get('annotation_search') else 2):
            for combo in itertools.combinations(options, count):
                if remaining(objects, [category, *combo]) == [obj.object_id]:
                    descriptions.append(combo)
            if descriptions:
                break
        visual = []
        if packet.task.get('annotation_search') and observations:
            from app.quality.description_search import visual_descriptor
            # Appearance and region may identify a landmark jointly even when
            # neither is unique alone. State every qualifier in its descriptor.
            regions = [p for p in options if p.kind in {'longitudinal', 'lateral'}]
            for count in range(3):
                for combo in itertools.combinations(regions, count):
                    ids = remaining(objects, [category, *combo])
                    selected = visual_descriptor(obj.category, ids, obj.object_id, observations)
                    if selected is not None:
                        visual, descriptions = selected, [combo]
                        break
                if descriptions:
                    break
        if not descriptions:
            continue
        if packet.task.get('annotation_search') and observations:
            from app.quality.description_search import landmark_attributes
            visual = landmark_attributes(obj.category, observations.get(obj.object_id, {}), visual)
        if packet.task.get('annotation_search'):
            from app.agent.appearance import filter_visual_candidates
            native_ids = {o.object_id for o in objects}
            unresolved_images = [c['object_id'] for c in packet.provenance.get('scene_image_candidates', [])
                if c['category'] in {obj.category, 'unknown'} and c['object_id'] not in native_ids]
            if filter_visual_candidates(unresolved_images, visual or [], observations or {})[0]:
                continue  # Use a camera-qualified landmark instead of assuming these IDs are the same object.
        cues = min(descriptions, key=lambda ps: (sum(len(p.phrase) for p in ps), tuple(p.predicate_id for p in ps)))
        if not cues:
            phrase = f"the {obj.category}"
        elif cues[0].kind in ("nearest", "farthest", "longest"):
            phrase = cues[0].phrase
        else:
            phrase = f"the {obj.category} " + ' and '.join(p.phrase for p in cues)
        if visual:
            from app.quality.description_search import appearance_phrase
            phrase = appearance_phrase(obj.category, visual)
            if cues:
                phrase += ' ' + ' and '.join(p.phrase for p in cues)
        anchor_id = hashlib.sha256((category.predicate_id + "/" + "/".join(p.predicate_id for p in cues)
            + (str(visual) if visual else '')).encode()).hexdigest()[:10]
        anchors.append({"anchor_id": anchor_id, "object": obj, "phrase": phrase,
                        "proof": [category.predicate_id, *[p.predicate_id for p in cues]],
                        "visual": [a.model_dump() for a in (visual or [])]})
    # Favor large, recognizable landmarks in the actual training photograph.
    # All geometry and visibility/uniqueness proofs remain unchanged.
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    areas = {}
    for box in packet.provenance.get('scene_image_candidates', []):
        if box.get('camera_name') == camera and box.get('xyxy'):
            x1, y1, x2, y2 = box['xyxy']
            areas[box['object_id']] = max(areas.get(box['object_id'], 0), max(0, x2-x1)*max(0, y2-y1))
    # Entirely scene-based selection; no target-derived anchor ordering or hints.
    anchors.sort(key=lambda a: (len(a["proof"]), -areas.get(a["object"].object_id, 0), a["object"].range_xy_m, a["anchor_id"]))
    # At most two anchors per class prevents common vehicles crowding out landmarks.
    balanced, per_class = [], {}
    for anchor in anchors:
        category = anchor["object"].category
        if per_class.get(category, 0) < 2:
            balanced.append(anchor)
            per_class[category] = per_class.get(category, 0) + 1
        if len(balanced) == 6:
            break
    anchors = anchors if packet.task.get('annotation_search') else balanced
    result = []

    def radius(obj, axis):
        length, width, _ = obj.size_m
        angle = obj.heading_rad
        return (abs(math.cos(angle)) * length + abs(math.sin(angle)) * width) / 2 if axis == 0 else (abs(math.sin(angle)) * length + abs(math.cos(angle)) * width) / 2

    for anchor in anchors:
        ref = anchor["object"]
        box = next((b for b in packet.provenance.get('scene_image_candidates', [])
            if b['object_id'] == ref.object_id and b.get('camera_name') == camera), None)
        if vehicle_heading_anchor(packet, ref, (observations or {}).get(ref.object_id, {}), box):
            for name, sign, words in [('front',1,'directly in front of'),('rear',-1,'directly behind')]:
                result.append(Predicate(predicate_id=f'vehicle_heading:{name}:{anchor["anchor_id"]}',
                    kind='vehicle_heading',value=anchor['anchor_id'],phrase=f'{words} {anchor["phrase"]}',
                    matches={o.object_id:vehicle_heading_match(o,ref,sign) for o in objects},
                    source=f'{VEHICLE_HEADING_POLICY}; native vehicle {ref.object_id} front-face heading; '
                        'disjoint longitudinal extents + 1m guard; central-half lateral alignment; same level; unknowns retained'))
        for kind, axis, sign, phrase in [
            ("relative_left", 1, 1, "to the left of"), ("relative_right", 1, -1, "to the right of"),
            ("ahead_of", 0, 1, "ahead of"), ("behind_anchor", 0, -1, "behind"),
        ]:
            matches = {}
            for obj in objects:
                if not obj.geometry_valid:
                    matches[obj.object_id]=None
                    continue
                delta = sign * (obj.center_m[axis] - ref.center_m[axis])
                guard = radius(obj, axis) + radius(ref, axis) + POLICY["axis_guard_m"]
                matches[obj.object_id] = False if obj.object_id == ref.object_id else True if delta > guard else False if delta < -guard else None
            result.append(Predicate(predicate_id=f"{kind}:{anchor['anchor_id']}", kind=kind, value=anchor["anchor_id"],
                                    phrase=f"{phrase} {anchor['phrase']}", matches=matches,
                                    source=f"native frame boxes; anchor {ref.object_id}; descriptor proof {anchor['proof']}; oriented extent + axis guard"))
        for category in sorted({o.category for o in objects} - {"unknown", ref.category}):
            group=[o for o in objects if o.category in {category,'unknown'}]
            if len(group) < 2:
                continue
            gap = POLICY["rank_range_gap_m"]
            matches=guarded_rank(objects,category,lambda o:math.dist(o.center_m[:2],ref.center_m[:2]),False,gap)
            result.append(Predicate(predicate_id=f"nearest_to_anchor:{category}:{anchor['anchor_id']}",
                                    kind="nearest_to_anchor", value=f"{category}:{anchor['anchor_id']}",
                                    phrase=f"the {category} closest to {anchor['phrase']}", matches=matches,
                                    source=f"all same-class native frame centers; anchor {ref.object_id}; descriptor proof {anchor['proof']}; gap {gap}m"))
    proofs = [{"anchor_id": a["anchor_id"], "object_id": a["object"].object_id,
               "phrase": a["phrase"], "descriptor_predicates": a["proof"]} for a in anchors]
    if packet.task.get('annotation_search'):
        for proof, anchor in zip(proofs, anchors):
            proof['descriptor_visual_attributes'] = anchor['visual']
    return result, proofs


def expand_relations(packet: EvidencePacket, observations=None, *, target_observation=None) -> tuple[EvidencePacket, dict]:
    from app.agent.appearance import effective_observations, resolve_target_observation
    from app.quality.visibility_policy import allows_partial
    observations = effective_observations(packet, observations)
    if target_observation is not None:
        # The independent primary RGB check is evidence too, including when a
        # short budget leaves no anonymous scene batch. Do not lose its source
        # camera binding on frame switches. Admission still checks independent
        # identity/visibility conflicts against the unmodified raw records.
        camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
        previous = observations.get(packet.target_object_id, {})
        resolved = resolve_target_observation(target_observation, previous,
            allow_partial=allows_partial(packet)).model_dump()
        record = {**previous, **resolved, 'observed_camera_name':camera}
        cameras = previous.get('camera_observations', {})
        record['camera_observations'] = {**cameras, str(camera):{
            k:v for k,v in record.items() if k != 'camera_observations'}}
        observations[packet.target_object_id] = record
    # Rebuilding evidence must not duplicate predicates after a checkpoint/relook.
    base = [p for p in packet.predicates if p.kind not in {'relative_left', 'relative_right',
        'ahead_of', 'behind_anchor', 'vehicle_heading', 'nearest_to_anchor', 'vertical_rank', 'image_relation'}]
    from app.quality.contrast import ego_near_catalog, EGO_NEAR_ID
    base = [p for p in base if p.predicate_id != EGO_NEAR_ID] + ego_near_catalog(packet)
    packet = packet.model_copy(update={'predicates': base})
    relations, proofs = relation_catalog(packet, observations)
    image_predicates, image_anchors = [], []
    if packet.task.get('annotation_search'):
        from app.quality.description_search import image_relation_catalog
        image_predicates, image_anchors = image_relation_catalog(packet, observations or {})
    vertical, vertical_proofs = vertical_rank_catalog(packet)
    from app.quality.description_search import confirmed_image_camera
    retained = [p for p in packet.predicates if p.kind != 'image_relation'
        or confirmed_image_camera(packet, int(p.predicate_id.split(':')[1]))]
    catalog = [*retained, *relations, *vertical, *image_predicates]
    plan = contrast_plan(packet.candidates, packet.target_object_id, catalog)
    # A compound ranking explicitly states its spatial subset. Preserve those
    # claims in generation/validation metadata rather than hiding them in a rule.
    dependencies = [q for proof in vertical_proofs if proof["predicate_id"] in plan.selected_predicate_ids
                    for q in proof["qualifier_predicate_ids"]]
    if dependencies:
        selected = list(dict.fromkeys([*plan.selected_predicate_ids, *dependencies]))
        chosen = [p for p in catalog if p.predicate_id in selected]
        plan = plan.model_copy(update={"selected_predicate_ids": selected,
            "removal_tests": {p.predicate_id: remaining(packet.candidates, [q for q in chosen if q != p]) for p in chosen},
            "hard_negatives": [{**n, "excluded_by": [p.predicate_id for p in chosen if p.matches[n["object_id"]] is False]} for n in plan.hard_negatives],
            "algorithm": plan.algorithm + "+qualified-vertical-rank-v1"})
    eligible = bool(packet.views) and packet.target.category != "unknown" and bool(packet.target.lidar_points and packet.target.lidar_points > 0) and packet.track.get("category_consistent", True) and (packet.scene is None or packet.scene.target_points > 0)
    if packet.task.get('requires_native_camera_link'):
        from app.datasets.correspondence import has_multimodal_evidence
        eligible = eligible and has_multimodal_evidence(packet)
    updated = packet.model_copy(update={"predicates": catalog, "contrast": plan,
                                       "provenance": {**packet.provenance, 'relation_anchors': [*proofs, *image_anchors]},
                                       "policy": {**packet.policy, "vertical_rank_gap_m": packet.policy.get("vertical_rank_gap_m", POLICY["vertical_rank_gap_m"])},
                                       "ready_for_generation": eligible and (packet.task.get('requires_native_camera_link') or plan.status == "unique_in_gt_candidates")})
    before = len(packet.contrast.remaining_object_ids)
    after = len(plan.remaining_object_ids)
    return updated, {"tool": "expand_relations", "tool_version": VERSION,
                     "anchors": [*proofs, *image_anchors], "vertical_ranks": vertical_proofs, "new_predicates": len(relations) + len(vertical) + len(image_predicates),
                     "remaining_before": before, "remaining_after": after,
                     "eliminated_candidates": before - after,
                     "plan_status": plan.status,
                     "scope": ("native geometry plus observed landmarks and explicit-camera native 2D relations; not independent alignment verification"
                               if packet.task.get('annotation_search') else
                               "conservative native-box geometry only; not pixel/point-cloud verification")}
