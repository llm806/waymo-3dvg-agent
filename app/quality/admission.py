"""Confirm source evidence; search for cues without declaring unknowns unannotatable."""
from itertools import combinations

from app.agent.appearance import filter_visual_candidates, resolve_target_observation, target_visibility_error, effective_observations
from app.quality.contrast import remaining
from app.quality.visibility_policy import allows_partial


def native_identity_conflicts(packet, remaining_ids):
    """Diagnose unresolved coincident native units, never merge or exclude them.

    These conservative proximity/size/image checks only explain a failed full
    cue proof. They cannot invalidate a successful proof or certify duplicate
    labels. The raw 3D boxes and every competing identity remain unchanged.
    """
    from app.quality.geometry import native_box_proximity

    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    boxes = {b['object_id']:b['xyxy'] for b in packet.provenance.get('scene_image_candidates', [])
        if b.get('camera_name') == camera and b.get('xyxy')}
    target = packet.target
    rect = boxes.get(target.object_id)
    if not target.geometry_valid or rect is None:
        return []
    conflicts = []
    for other in packet.candidates:
        if (other.object_id == target.object_id or other.object_id not in remaining_ids
                or other.category != target.category or not other.geometry_valid or other.object_id not in boxes):
            continue
        proximity = native_box_proximity(target, other)
        if proximity is None:
            continue
        ref = boxes[other.object_id]
        intersection = max(0,min(rect[2],ref[2])-max(rect[0],ref[0])) * max(0,min(rect[3],ref[3])-max(rect[1],ref[1]))
        union = (rect[2]-rect[0])*(rect[3]-rect[1])+(ref[2]-ref[0])*(ref[3]-ref[1])-intersection
        if union > 0 and intersection/union >= .75:
            conflicts.append({'object_id':other.object_id,**proximity,
                'source_camera':camera,'image_iou':intersection/union,
                'scope':'unresolved native identities; overlap is diagnostic, not a duplicate-label determination'})
    return conflicts


def distinguishing_cue_bound(packet, observation, observations):
    """Optimistic bound on what the current confirmed vocabulary can express.

    If even ALL supported target facts and confirmed attributes leave rivals,
    no subset or paraphrase can uniquely identify this target. Unknown objects
    stay in the inventory. A unique bound only permits trying a sentence; it
    does not certify that sentence or waive any final checks.
    """
    from app.quality.description_search import expression_candidates, draft_reference_predicate

    effective = effective_observations(packet, observations)
    resolved = resolve_target_observation(observation, effective.get(packet.target_object_id), allow_partial=allows_partial(packet))
    effective = {**effective, packet.target_object_id: resolved.model_dump()}
    predicates = [p for p in packet.predicates
        if p.kind not in {'range', 'nearest', 'farthest', 'nearest_to_anchor'}
        and p.matches.get(packet.target_object_id) is True
        and (not packet.task.get('natural_description') or draft_reference_predicate(packet, p))]
    proof = expression_candidates(packet, predicates, resolved.attributes, effective)
    return {'can_distinguish': proof['native'] == [packet.target_object_id] and not proof['image_only'],
        'predicate_ids': [p.predicate_id for p in predicates],
        'remaining_object_ids': proof['native'], 'remaining_image_object_ids': proof['image_only'],
        'scope': 'optimistic bound over current verified vocabulary; unknown competitors retained'}


def _cue_readability(predicate):
    """Prefer ordinary references before image-edge/height measurements.

    This orders already-proven cues only; it never changes their matches or
    removes unknown competitors. Attribute scopes are semantic requirements,
    even when the standalone attribute minimizer can remove those attributes.
    """
    from app.quality.description_search import image_order_attributes
    parts = predicate.predicate_id.split(':')
    if parts[0] == 'image_attribute':
        tier = 0  # A concrete appearance reference needs no landmark or rank.
    elif parts[0] == 'image_boundary':
        tier = 8  # Last fallback, after real scene relations and horizontal cues.
    elif parts[0] == 'image_visibility':
        tier = 3  # Prefer a landmark relation over camera-only membership.
    elif parts[0] == 'image_order':
        order = parts[-1]
        tier = (4 if order.startswith(('leftmost', 'rightmost', 'second_from_left', 'second_from_right')) else 9)
    elif predicate.kind in {'relative_left', 'relative_right', 'ahead_of', 'behind_anchor', 'vehicle_heading'}:
        tier = 1
    elif parts[0] in {'image_anchor','image_vehicle_heading','image_joint_heading','image_joint_spatial'}:
        tier = 2
    else:
        tier = 4  # Native height/length ranks are also a last resort.
    return tier, len(image_order_attributes(predicate)), len(predicate.phrase.split()), predicate.predicate_id


def scene_cue_preference(packet, predicate):
    """Prefer recognizable, prominent verified landmarks in new scene drafts."""
    from app.quality.description_search import scene_reference_policy
    readability = _cue_readability(predicate)
    if not scene_reference_policy(packet):
        return readability
    anchor = next((a for a in packet.provenance.get('relation_anchors', [])
        if a['anchor_id'] == predicate.value), {})
    attrs = anchor.get('descriptor_visual_attributes', [])
    area = max(((c['xyxy'][2]-c['xyxy'][0])*(c['xyxy'][3]-c['xyxy'][1])
        for c in packet.provenance.get('source_image_candidates', [])
        if c['object_id'] == anchor.get('object_id') and c.get('xyxy')), default=0)
    return (readability[0], -int(bool(attrs)), -area, *readability[1:])


def reference_fallback_tier(predicate):
    if predicate.predicate_id.startswith('image_boundary:'):
        return 2
    if predicate.predicate_id.startswith('image_order:'):
        return 1
    return 0


def observation_plan_sufficient(packet, plan):
    """A pixel fallback must not preempt budgeted scene evidence acquisition."""
    from app.quality.description_search import scene_first_policy
    fallback = any(id_.startswith(('image_boundary:','image_order:')) for id_ in plan.get('predicate_ids',[]))
    return bool(plan.get('unique_plan_found') and not (scene_first_policy(packet) and fallback))


def reference_search_groups(packet, optional):
    from app.quality.description_search import scene_first_policy
    if not scene_first_policy(packet):
        for count in range(3):
            yield combinations(optional,count)
        return
    # Finish ordinary one- AND two-relation plans before offering any pixel cue.
    for tier in range(3):
        choices=[p for p in optional if reference_fallback_tier(p)<=tier]
        for count in range(3):
            yield (extras for extras in combinations(choices,count)
                if max((reference_fallback_tier(p) for p in extras),default=0)==tier)


def admission_plan(packet, observation, observations, anchors=(), *, allow_pixel_fallback=True):
    """Unknown/unobserved competitors survive; absence is never inferred."""
    result = {'passed': False, 'reason': 'precheck_target_not_visible',
        'source_view': packet.source_view, 'source_camera_object_id': packet.source_camera_object_id,
        'predicate_ids': [], 'visual_attributes': [], 'remaining_object_ids': []}
    error = target_visibility_error(packet, observation, observations)
    if error:
        result['reason'] = error
        return result
    observations = effective_observations(packet, observations)
    from app.datasets.correspondence import has_multimodal_evidence
    if not has_multimodal_evidence(packet):
        result['reason'] = 'precheck_no_multimodal_evidence'
        return result
    image_inventory = packet.provenance.get('source_image_candidates')
    if not image_inventory or not any(c['object_id'] == packet.target_object_id
        and c['camera_object_id'] == packet.source_camera_object_id for c in image_inventory):
        result['reason'] = 'precheck_image_inventory_missing'
        return result
    image_inventory = packet.provenance.get('scene_image_candidates')
    if not image_inventory:
        result['reason'] = 'precheck_scene_image_inventory_missing'
        return result
    search = bool(packet.task.get('annotation_search'))
    attrs = resolve_target_observation(observation, observations.get(packet.target_object_id), allow_partial=allows_partial(packet)).attributes
    priority = {'subtype':0, 'feature':1, 'shape':2, 'color':3}
    attrs.sort(key=lambda a: (priority.get(a.kind, 4), a.kind, a.value))
    from app.quality.description_search import descriptive_attributes
    recognition = descriptive_attributes(packet, {**observations,
        packet.target_object_id: resolve_target_observation(observation,
            observations.get(packet.target_object_id), allow_partial=allows_partial(packet)).model_dump()})
    result['reason'] = 'precheck_no_reliable_visual_feature'
    if not attrs and not search:
        return result
    native_ids = {o.object_id for o in packet.candidates}
    image_ids = [c['object_id'] for c in image_inventory
        if c['category'] in {packet.target.category, 'unknown'} and c['object_id'] not in native_ids]
    base = [p for p in packet.predicates if (p.kind == 'category' and p.value == packet.target.category
        or p.kind in {'longitudinal', 'lateral'}) and p.matches.get(packet.target_object_id) is True]
    if not any(p.kind == 'category' for p in base):
        result['reason'] = 'precheck_category_unknown'
        return result
    visible_anchors = {a['anchor_id'] for a in anchors if observations.get(a['object_id'], {}).get('target_visible')}
    optional = [p for p in packet.predicates if p.kind in {'relative_left', 'relative_right', 'ahead_of', 'behind_anchor', 'vehicle_heading'}
        and p.value in visible_anchors
        and p.matches.get(packet.target_object_id) is True]
    if search:
        optional.extend(p for p in packet.predicates if p.kind in {'longest', 'vertical_rank'}
            and p.matches.get(packet.target_object_id) is True)
        optional.extend(p for p in packet.predicates if p.kind == 'image_relation'
            and p.matches.get(packet.target_object_id) is True)
    if search:
        # Catalog insertion order used to favor complete appearance groups and
        # RGB edge ranks. Search the easiest truthful descriptions first.
        from app.quality.description_search import draft_reference_predicate
        if packet.task.get('natural_description'):
            optional = [p for p in optional if draft_reference_predicate(packet, p)]
        if not allow_pixel_fallback:
            optional = [p for p in optional if not reference_fallback_tier(p)]
        optional.sort(key=lambda p: scene_cue_preference(packet,p))
    # At most two verified landmark relations; no distance/rank fallback.
    for group in reference_search_groups(packet,optional):
        for extras in group:
            predicates = [*base, *extras]
            spatial = remaining(packet.candidates, predicates)
            joint, exclusions = filter_visual_candidates(spatial, attrs, observations)
            visual_only, _ = filter_visual_candidates(image_ids, attrs, observations)
            if search:
                from app.quality.description_search import expression_candidates
                effective = {**observations, packet.target_object_id: resolve_target_observation(
                    observation, observations.get(packet.target_object_id), allow_partial=allows_partial(packet)).model_dump()}
                checked = expression_candidates(packet, predicates, attrs, effective)
                joint, visual_only = checked['native'], checked['image_only']
            result.update(predicate_ids=[p.predicate_id for p in predicates],
                visual_attributes=[a.model_dump() for a in attrs], remaining_object_ids=joint,
                remaining_image_object_ids=visual_only, visual_exclusions=exclusions)
            if joint == [packet.target_object_id] and not visual_only:
                # Only keep attributes needed by the actual joint proof. An
                # image relation can distinguish a person without forcing
                # uncertain clothing colors into an otherwise valid sentence.
                essential = list(attrs)
                from app.quality.description_search import image_order_attributes
                scoped_attrs = [a for p in predicates for a in image_order_attributes(p)]
                for attr in reversed(attrs):
                    if attr in recognition or attr in scoped_attrs:
                        continue
                    trial = [a for a in essential if a != attr]
                    if search:
                        checked = expression_candidates(packet, predicates, trial, effective)
                        useful = checked['native'] == [packet.target_object_id] and not checked['image_only']
                    else:
                        useful = bool(trial) and filter_visual_candidates(spatial, trial, observations)[0] == [packet.target_object_id] and not filter_visual_candidates(image_ids, trial, observations)[0]
                    if useful:
                        essential = trial
                from app.quality.description_search import scene_first_policy
                if (search and not scene_first_policy(packet) and not packet.task.get('description_detail_policy')
                        and any(p.kind == 'image_relation' for p in extras)):
                    # A unique camera-qualified reference often needs no extra
                    # ego-location clause. Remove it only after the same joint
                    # proof succeeds without it; region ranks keep their scope.
                    for predicate in base:
                        if predicate.kind not in {'longitudinal', 'lateral'} or any(p.kind == 'vertical_rank' for p in extras):
                            continue
                        trial = [p for p in predicates if p != predicate]
                        checked = expression_candidates(packet, trial, essential, effective)
                        if checked['native'] == [packet.target_object_id] and not checked['image_only']:
                            predicates = trial
                    result['predicate_ids'] = [p.predicate_id for p in predicates]
                result['visual_attributes'] = [a.model_dump() for a in essential]
                if packet.task.get('natural_description'):
                    from app.quality.description_search import without_redundant_camera_visibility
                    predicates = without_redundant_camera_visibility(packet,predicates,essential,effective if search else observations)
                    result['predicate_ids'] = [p.predicate_id for p in predicates]
                spatial = remaining(packet.candidates, predicates)
                result['visual_exclusions'] = filter_visual_candidates(spatial, essential, observations)[1]
                result.update(passed=True, reason='precheck_visible_distinct_target', unique_plan_found=True)
                return result
    if search:
        result.update(passed=True, reason='visible_target_ready_for_description_search', unique_plan_found=False,
            scope='identity and visibility confirmed; uniqueness remains to be checked on the generated sentence')
        return result
    result['reason'] = 'precheck_target_not_distinct'
    return result


def reference_options(packet, observations, required_ids=()):
    """A few verified evidence bundles for the writer; no sentence templates.

    A relation being true does not mean it uniquely identifies the target. Check
    landmark alternatives against the complete joint inventory before suggesting
    them, and retain unknown competitors just as in final tool grounding.
    """
    from app.quality.description_search import expression_candidates, confirmed_attributes, image_order_attributes
    from app.schemas.contracts import VisualAttribute
    catalog = {p.predicate_id:p for p in packet.predicates}
    plan = packet.provenance.get('admission_plan', {})
    confirmed = confirmed_attributes(packet.target.category, observations.get(packet.target_object_id, {}))
    from app.quality.description_search import descriptive_attributes
    recognition = descriptive_attributes(packet, observations)
    mandatory = [p for p in packet.predicates if p.predicate_id in required_ids]
    options, seen = [], set()

    def add(predicates, attributes):
        from app.quality.description_search import draft_reference_predicate, without_redundant_camera_visibility
        if packet.task.get('natural_description') and any(not draft_reference_predicate(packet, p) for p in predicates):
            return
        if packet.task.get('natural_description'):
            predicates = without_redundant_camera_visibility(packet,predicates,attributes,observations)
        ids = tuple(sorted({p.predicate_id for p in predicates}))
        if ids in seen:
            return
        seen.add(ids)
        attrs = list(attributes)
        for predicate in predicates:
            for attribute in image_order_attributes(predicate):
                if attribute not in attrs:
                    attrs.append(attribute)
        if any(a not in confirmed for a in attrs):
            return
        proof = expression_candidates(packet,predicates,attrs,observations)
        if proof['native'] != [packet.target_object_id] or proof['image_only']:
            return
        # Keep the recognizable core type and confirmed body paint, minimizing
        # other details only after rechecking the full proof.
        for attribute in list(reversed(attrs)):
            if attribute in recognition:
                continue
            if attribute.kind == 'subtype' or packet.target.category=='vehicle' and attribute.kind=='color':
                continue
            if any(attribute in image_order_attributes(p) for p in predicates):
                continue
            trial = [a for a in attrs if a != attribute]
            checked = expression_candidates(packet,predicates,trial,observations)
            if checked['native']==[packet.target_object_id] and not checked['image_only']:
                attrs = trial
        options.append({'predicate_ids':list(ids),'visual_attributes':[a.model_dump() for a in attrs],
            'verification':'unique in the complete frozen native/image inventory; not a language certificate'})

    add(mandatory,confirmed)
    from app.quality.description_search import scene_first_policy
    ego=[p for p in packet.predicates if p.kind in {'longitudinal','lateral'}
        and p.matches.get(packet.target_object_id) is True]
    if scene_first_policy(packet):
        for count in range(1,len(ego)+1):
            for directions in combinations(ego,count):
                add([*mandatory,*directions],confirmed)
    if plan.get('unique_plan_found'):
        add([catalog[id_] for id_ in plan['predicate_ids'] if id_ in catalog],confirmed)
    camera = next((v.camera_name for v in packet.views if v.camera_label==packet.source_view),None)
    visible = catalog.get(f'image_visibility:{camera}:{packet.target.category}')
    base = [*mandatory]
    from app.quality.description_search import draft_reference_predicate
    if visible and draft_reference_predicate(packet,visible) and visible.matches.get(packet.target_object_id) is True and visible not in base:
        base.append(visible)
    relations = [p for p in packet.predicates if p.matches.get(packet.target_object_id) is True
        and (p.kind in {'relative_left','relative_right','ahead_of','behind_anchor','vehicle_heading'}
             or scene_first_policy(packet) and p.predicate_id.startswith('image_attribute:')
             or p.predicate_id.startswith(('image_anchor:','image_vehicle_heading:')))]
    relations.sort(key=lambda p: scene_cue_preference(packet,p))
    for predicate in relations[:12]:
        if scene_first_policy(packet):
            add([*mandatory,*ego,predicate],confirmed)
        add([*mandatory,predicate],confirmed)
        add([*base,predicate],confirmed)
        if len(options)>=4:
            break
    def preference(option):
        chosen = [catalog[id_] for id_ in option['predicate_ids']]
        # A short, unique ordinary landmark relation is easier to say than an
        # ordinal comparison, while an already clear appearance needs no landmark.
        orders = any(p.predicate_id.startswith('image_order:') for p in chosen)
        camera_only = any(p.predicate_id.startswith('image_visibility:') for p in chosen)
        ego_preference=int(not any(p.kind in {'longitudinal','lateral'} for p in chosen)) if scene_first_policy(packet) else 0
        return ((max((reference_fallback_tier(p) for p in chosen),default=0) if scene_first_policy(packet) else int(orders)),
            ego_preference, int(camera_only), sum(len(p.phrase.split()) for p in chosen) +
            len(option['visual_attributes']), option['predicate_ids'])
    return sorted(options,key=preference)
