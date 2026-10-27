"""Camera-qualified descriptions checked against the frozen image inventory.

Image coordinates are usable only when the sentence explicitly names the camera.
They supplement, never replace, the target's verified native 3D identity.
"""
import hashlib
from itertools import combinations

from app.agent.appearance import filter_visual_candidates
from app.quality.contrast import remaining
from app.schemas.contracts import Predicate, VisualAttribute, VISUAL_ATTRIBUTE_VALUES
from app.quality.visibility_policy import allows_partial, checks_rgb_visibility, source_camera_inventory, usable_rgb_observation

# Human language is separate from Waymo's fixed enum names in metadata.
CAMERA_VIEWS = {1: 'front view', 2: 'front-left view', 3: 'front-right view',
                4: 'left-side view', 5: 'right-side view'}

SCENE_REFERENCE_POLICY = 'intrinsic-ego-landmark-v1'
MINIMAL_REFERENCE_POLICY = 'minimal-sufficient-grounding-v1'
TRAFFIC_SCENE_POLICY = 'traffic-scene-v1'


def minimal_reference_policy(packet):
    return packet.task.get('reference_policy') == MINIMAL_REFERENCE_POLICY


def scene_first_policy(packet):
    return minimal_reference_policy(packet) and packet.task.get('reference_preference_policy') in {'scene-first-v1', TRAFFIC_SCENE_POLICY}


def traffic_scene_policy(packet):
    return packet.task.get('reference_preference_policy') == TRAFFIC_SCENE_POLICY


def scene_reference_policy(packet):
    return packet.task.get('reference_policy') in {SCENE_REFERENCE_POLICY, MINIMAL_REFERENCE_POLICY}


def draft_reference_predicate(packet, predicate):
    """New writing vocabulary; historical predicates/proofs remain readable."""
    from app.quality.relations import local_landmark_predicate
    if not local_landmark_predicate(packet,predicate):
        return False
    if packet.task.get('sentence_style_policy') == 'simple-sufficient-sentence-v1':
        if predicate.predicate_id.startswith('image_joint_spatial:'):
            return False  # No axis comparisons, elevation comparisons or chained frames.
    from app.quality.contrast import EGO_NEAR_ID, EGO_PROXIMITY_POLICY
    if predicate.predicate_id == EGO_NEAR_ID:
        return packet.task.get('ego_proximity_policy') == EGO_PROXIMITY_POLICY
    from app.quality.spatial_relations import native_only, NATIVE_RELATIONS, PLAIN_SPATIAL_POLICY
    if (packet.task.get('spatial_language_policy') == PLAIN_SPATIAL_POLICY
            and predicate.predicate_id.startswith('image_joint_spatial:')
            and predicate.predicate_id.split(':')[2] in {'native_left','native_right','native_ahead','native_behind'}):
        return False
    if native_only(packet):
        if predicate.predicate_id.startswith(('image_order:','image_boundary:','image_anchor:','image_vehicle_heading:')):
            return False
        if predicate.kind in {'relative_left','relative_right','ahead_of','behind_anchor'}:
            return False  # Use explicitly framed native joint relations instead.
        if predicate.predicate_id.startswith('image_joint_spatial:'):
            return predicate.predicate_id.split(':')[2] in {r.key for r in NATIVE_RELATIONS}
    if traffic_scene_policy(packet) and predicate.predicate_id.startswith('image_boundary:'):
        return False
    if not ordinary_reference_predicate(predicate):
        return False
    if scene_reference_policy(packet) and not minimal_reference_policy(packet):
        return not predicate.predicate_id.startswith(('image_order:', 'image_visibility:'))
    return True


def ordinary_reference_predicate(predicate):
    """Language candidates, not a change to the frozen geometry vocabulary.

    Keep technical extent/center ranks for evidence inspection and old records;
    new natural descriptions must use recognizable objects and ordinary relations.
    """
    if predicate.kind in {'range', 'nearest', 'farthest', 'nearest_to_anchor', 'longest', 'vertical_rank'}:
        return False
    if predicate.predicate_id.startswith('image_order:'):
        # Pixel height/vertical ranks are not road-scene locations. Retain the
        # historical vocabulary for reading evidence, never offer it to writers.
        return predicate.predicate_id.rsplit(':', 1)[-1] in {
            'leftmost','rightmost','leftmost_visible','rightmost_visible',
            'second_from_left','second_from_right'}
    return True


def confirmed_image_camera(packet, camera):
    """Only the source camera can bind the target's RGB identity check."""
    if not checks_rgb_visibility(packet):
        return True
    safe = source_camera_inventory(packet)
    return any(v.camera_name == camera and v.camera_label == packet.source_view
        and (allows_partial(packet) or v.projection_touches_boundary is False and v.target_front_facing is not False)
        and (safe is None or str(camera) in safe) for v in packet.views)


def camera_observations(packet, camera, observations):
    """Image landmarks/order need evidence from the image being described.

    The standalone target precheck is implicitly tied to source_view. Anonymous
    competitor observations carry their camera explicitly; a missing/merged
    camera cannot prove visibility in this image. Keep the full inventory so
    unobserved objects remain competitors.
    """
    if not checks_rgb_visibility(packet):
        return observations
    result = {}
    for oid, record in observations.items():
        camera_record = record.get('camera_observations', {}).get(str(camera))
        if camera_record is not None:
            result[oid] = camera_record
        elif record.get('observed_camera_name') == camera:
            result[oid] = record
        elif (oid == packet.target_object_id and 'observed_camera_name' not in record
                and confirmed_image_camera(packet, camera)):
            result[oid] = record
    from app.agent.appearance import effective_observations
    return effective_observations(packet, result)


def appearance_phrase(category, attributes, *, plural=False):
    values = {a.kind: a.value for a in attributes}
    noun = values.get('subtype', category) if category in {'vehicle', 'sign'} else category
    if plural:
        noun = {'bus': 'buses', 'excavator body': 'excavator bodies'}.get(noun, noun + 's')
    words = [values[k] for k in ('color', 'shape') if k in values]
    phrase = 'the ' + ' '.join([*words, noun])
    if category == 'cyclist' and values.get('subtype') in {'bicycle', 'motorcycle'}:
        phrase += ' on a ' + values['subtype']
    clothing = []
    if 'upper_clothing_color' in values:
        clothing.append('a ' + values['upper_clothing_color'] + ' top')
    if 'lower_clothing_color' in values:
        clothing.append(values['lower_clothing_color'] + ' bottoms')
    if 'helmet_color' in values:
        clothing.append('a ' + values['helmet_color'] + ' helmet')
    if clothing:
        phrase += ' wearing ' + (', '.join(clothing[:-1]) + ', and ' + clothing[-1] if len(clothing)>2 else ' and '.join(clothing))
    features = [a.value for a in attributes if a.kind == 'feature' and a.value not in {'handheld bag', 'handheld camera', 'binoculars', 'visible rear face', 'pole mounted'}]
    if features:
        phrase += ' with ' + ' and '.join(f if f.endswith('stripes') or f.startswith('illuminated ') else ('an ' if f == 'umbrella' else 'a ') + f for f in features)
    held = [noun for value,noun in [('handheld camera','a camera'),('handheld bag','a bag')]
        if any(a.kind=='feature' and a.value==value for a in attributes)]
    if held:
        phrase += (' and' if clothing else '') + ' holding ' + ' and '.join(held)
    if any(a.kind=='feature' and a.value=='binoculars' for a in attributes):
        phrase += ' using binoculars'
    if any(a.kind == 'feature' and a.value == 'pole mounted' for a in attributes):
        phrase += ' mounted on a pole'
    if any(a.kind == 'feature' and a.value == 'visible rear face' for a in attributes):
        phrase += ' with its rear surface visible'
    return phrase


def confirmed_attributes(category, record):
    """Canonical, unambiguous observed attributes shared by all cue searches."""
    if not record.get('target_visible'):
        return []
    grouped = {}
    for a in record.get('attributes', []):
        grouped.setdefault(a['kind'], set()).add(a['value'])
    attributes = []
    for kind in VISUAL_ATTRIBUTE_VALUES:
        values = grouped.get(kind, set())
        if kind != 'feature' and len(values) != 1:
            continue
        for value in sorted(values):
            if kind == 'feature' and value in record.get('absent_features', []):
                continue
            if kind == 'helmet_color' and category not in {'pedestrian', 'cyclist'}:
                continue
            if kind == 'subtype' and category not in {'vehicle', 'sign'} and not (
                    category == 'cyclist' and value in {'bicycle', 'motorcycle'}):
                continue
            attributes.append(VisualAttribute(kind=kind, value=value))
    return attributes


def landmark_attributes(category, record, selected=()):
    """Keep a landmark's recognizable type and one conspicuous confirmed cue."""
    attributes = list(selected)
    confirmed = confirmed_attributes(category, record)
    priority = ({'subtype': 0, 'color': 1, 'shape': 2} if category in {'vehicle', 'sign'} else
                {'helmet_color': 0, 'feature': 1, 'upper_clothing_color': 2, 'subtype': 3})
    core = sorted(confirmed, key=lambda a: (priority.get(a.kind, 9), a.kind, a.value))[:2]
    for attribute in core:
        if attribute not in attributes:
            attributes.append(attribute)
    return attributes


def descriptive_attributes(packet, observations):
    """A bounded recognition brief for new drafts, separate from uniqueness.

    Keep bounded salient confirmed details even if an ordinal alone could
    identify the target. Missing/conflicting observations never become claims.
    Legacy frozen packets keep their original generation/validation policy.
    """
    if not packet.task.get('description_detail_policy') and not scene_first_policy(packet):
        return []
    record = (observations or {}).get(packet.target_object_id, {})
    if checks_rgb_visibility(packet) and not usable_rgb_observation(record, allow_partial=allows_partial(packet)):
        return []
    road_scene = scene_first_policy(packet) or packet.task.get('description_detail_policy') == 'confirmed-road-scene-v2'
    priorities = ({'subtype':0, 'color':1, 'feature':2, 'shape':3} if packet.target.category=='vehicle' else
        {'subtype':0, 'feature':1, 'color':2, 'shape':3} if packet.target.category=='sign' and road_scene else
        {'color':0, 'shape':1, 'subtype':2, 'feature':3} if packet.target.category=='sign' else
        {'helmet_color':0, 'upper_clothing_color':1, 'feature':2, 'lower_clothing_color':3, 'subtype':4})
    from app.quality.joint_reference import feedback_target_features
    requested = {(a['kind'],a['value']) for a in feedback_target_features(packet.task)}
    attrs = sorted(confirmed_attributes(packet.target.category, record),
        key=lambda a:((a.kind,a.value) not in requested,priorities.get(a.kind,9),a.kind,a.value))
    # One value per kind, plus at most one feature: a rack/bag/vest can help,
    # but listing every accessory makes a referring expression harder to read.
    if road_scene:
        selected, seen = [], set()
        for attribute in attrs:
            if attribute.kind not in seen:
                selected.append(attribute)
                seen.add(attribute.kind)
            if len(selected) == 3:
                break
        return selected
    return attrs[:2]


def visual_descriptor(category, ids, object_id, observations, *, visible_camera=None):
    """Find a short, independently observed unique descriptor; unknowns survive."""
    record = observations.get(object_id, {})
    if not record.get('target_visible'):
        return None
    attrs = confirmed_attributes(category, record)
    for count in range(len(attrs) + 1):
        for selected in combinations(attrs, count):
            if filter_visual_candidates(ids, selected, observations, visible_camera=visible_camera)[0] == [object_id]:
                return list(selected)
    return None


def image_order_attributes(predicate):
    """An attribute-qualified order names its scope in both ID and phrase."""
    parts = predicate.predicate_id.split(':')
    if (parts[0] not in {'image_order', 'image_attribute'}
            or len(parts) != (4 if parts[0] == 'image_order' else 3) or '~' not in parts[2]):
        return []
    return [VisualAttribute(kind=part.split('=', 1)[0], value=part.split('=', 1)[1])
        for part in parts[2].split('~')[1:]]


def fully_hidden(record):
    return (record.get('target_visible') is False and record.get('visibility') == 'occluded'
        and bool(record.get('visibility_reason', '').strip()) and not record.get('visibility_conflict'))


def scoped_image_orders(camera, boxes, observed, native_ids, all_ids, source, *, allow_partial=False):
    """Order observed appearance groups; unknown competitors remain possible.

    A white neighbor may overlap a black target in projection without blocking
    'leftmost black vehicle'. Unknown color/type cannot remove any competitor.
    Scopes are enumerated symmetrically from camera-bound observations, never
    selected from the requested target. Keep the existing disjoint-bounds guard.
    """
    scopes = set()
    for box in boxes:
        record = observed.get(box['object_id'], {})
        if box['category'] == 'unknown' or not record.get('target_visible'):
            continue
        attributes = tuple((a.kind, a.value) for a in confirmed_attributes(box['category'], record))
        # Short descriptions first; the complete descriptor also covers proofs
        # that need more than two attributes. Adding confirmed qualifiers only
        # narrows membership, so every intermediate subset need not be enumerated.
        for count in range(1, min(2, len(attributes)) + 1):
            scopes.update((box['category'], scope) for scope in combinations(attributes, count))
        if len(attributes) > 2:
            scopes.add((box['category'], attributes))
    result = []
    for category, scope in sorted(scopes, key=lambda s: (-len(s[1]), s)):
        attrs = [VisualAttribute(kind=kind, value=value) for kind, value in scope]
        group = [b for b in boxes if b['category'] in {category, 'unknown'}
            and not (allow_partial and fully_hidden(observed.get(b['object_id'], {})))]
        survivors, _ = filter_visual_candidates([b['object_id'] for b in group], attrs, observed)
        group = [b for b in group if b['object_id'] in survivors]
        if not group:
            continue
        descriptor = category + ''.join(f'~{kind}={value}' for kind, value in scope)
        if allow_partial:
            # The same RGB extent proof can rank a confirmed appearance group.
            # Keep every unknown member's full bounds and the existing 16px gap.
            for rule in visible_extent_orders(camera, group, observed, native_ids, all_ids, source):
                winners = [oid for oid, match in rule.matches.items() if match is True]
                if len(winners) != 1:
                    continue  # An assembly's outline cannot be assigned to one colored native part.
                winner = winners[0]
                if any(a.model_dump() not in observed.get(winner, {}).get('attributes', []) for a in attrs):
                    continue
                name = rule.predicate_id.rsplit(':', 1)[1]
                phrase = rule.phrase.replace('the visible ' + category,
                    'the visible ' + appearance_phrase(category, attrs)[4:], 1)
                result.append(rule.model_copy(update={'predicate_id': f'image_order:{camera}:{descriptor}:{name}',
                    'phrase': phrase, 'source': rule.source + '; camera-bound attribute scope; unknown members retained'}))
        for name, axis, sign in [('leftmost', 0, 1), ('rightmost', 0, -1)]:
            ordered = sorted(group, key=lambda b: sign * (b['xyxy'][axis] + b['xyxy'][axis+2]))
            winner = ordered[0]
            oid = winner['object_id']
            record = observed.get(oid, {})
            if (oid not in native_ids or winner['category'] != category
                    or not usable_rgb_observation(record, allow_partial=allow_partial)
                    or any(a.model_dump() not in record.get('attributes', []) for a in attrs)):
                continue
            if any((b['xyxy'][axis] - winner['xyxy'][axis+2] if sign > 0
                    else winner['xyxy'][axis] - b['xyxy'][axis+2]) <= 8 for b in ordered[1:]):
                continue
            phrase = appearance_phrase(category, attrs)
            order = name + ('_visible' if allow_partial else '')
            result.append(Predicate(predicate_id=f'image_order:{camera}:{descriptor}:{order}',
                kind='image_relation', value=f'{camera}:{order}', required_capability='rgb',
                phrase=f'the {name}{" visible" if allow_partial else ""} {phrase[4:]} in the {CAMERA_VIEWS[camera]}',
                matches={other: other == oid for other in all_ids},
                source=source + '; camera-bound attribute scope; unknowns retained; recognizable winner; disjoint projected bounds + 8px guard'
                    + ('; visible scope excludes only explicit same-camera fully hidden units' if allow_partial else '')))
    return result


def visible_image_orders(camera, boxes, observed, native_ids, all_ids, source):
    """Order explicitly visible units without letting fully hidden boxes win.

    Unknown and uncertain units still compete. Exclusion requires a negative
    RGB observation from this exact camera, not an absent crop or geometry.
    The selected native unit must be recognizable and separated from every possible
    rival; this does not relax the guards for partial projected extents.
    """
    groups = native_unit_groups(boxes, native_ids)
    present = {b['object_id'] for b in boxes}
    # Whole-assembly RGB labels duplicate the covered units for a unit-order
    # comparison. Keep every member projection, including unknown/hidden units;
    # remove the assembly row only when all of its units have camera bounds.
    boxes = [b for b in boxes if b['object_id'] not in groups
        or not set(groups[b['object_id']]) <= present]
    result = []
    for category in sorted({b['category'] for b in boxes} - {'unknown'}):
        group = []
        for box in boxes:
            if box['category'] not in {category, 'unknown'}:
                continue
            record = observed.get(box['object_id'], {})
            if not fully_hidden(record):
                group.append(box)
        if not group:
            continue
        for name, axis, sign in [('leftmost', 0, 1), ('rightmost', 0, -1),
                                 ('topmost', 1, 1), ('bottommost', 1, -1)]:
            ordered = sorted(group, key=lambda b: sign * (b['xyxy'][axis] + b['xyxy'][axis+2]))
            winner = ordered[0]
            oid = winner['object_id']
            record = observed.get(oid, {})
            if (oid not in native_ids or winner['category'] != category
                    or not usable_rgb_observation(record, allow_partial=True)):
                continue
            if any((other['xyxy'][axis] - winner['xyxy'][axis+2] if sign > 0
                    else winner['xyxy'][axis] - other['xyxy'][axis+2]) <= 8
                    or abs(sum(winner['xyxy'][axis::2]) - sum(other['xyxy'][axis::2])) <= 32
                    for other in ordered[1:]):
                continue
            result.append(Predicate(predicate_id=f'image_order:{camera}:{category}:{name}_visible',
                kind='image_relation', value=f'{camera}:{name}_visible', required_capability='rgb',
                phrase=f'the {name} visible {category} in the {CAMERA_VIEWS[camera]}',
                matches={other: other == oid for other in all_ids},
                source=source + '; camera-bound visible scope; only explicit fully-hidden RGB exclusions; '
                    'unknowns retained; recognizable native winner; disjoint bounds + 8px guard and 16px center tie guard'))
    return result


def visible_extent_orders(camera, boxes, observed, native_ids, all_ids, source):
    """Use visible RGB extents instead of requiring vertically disjoint bodies.

    A projected 3D box cannot supply the winning visible edge. Unknown rivals
    still compete with their entire bounds; only explicit same-camera fully
    hidden observations remove them. No native 2D/3D association is invented.
    """
    result = []
    native_rgb = {'official_camera_lidar_association', 'native_box_point_correspondence', 'native_camera_box',
        'native_camera_box_group_point_correspondence'}
    groups = native_unit_groups(boxes, native_ids)
    covered = {oid for members in groups.values() for oid in members}
    boxes = [b for b in boxes if not (b['object_id'] in covered
        and b.get('identity_source') == 'official_lidar_box_projection')]
    for category in sorted({b['category'] for b in boxes} - {'unknown'}):
        group = [b for b in boxes if b['category'] in {category, 'unknown'}
            and not fully_hidden(observed.get(b['object_id'], {}))]
        if not group:
            continue
        for name, axis, sign, phrase in [('bottom_edge_visible',3,-1,'extending lowest'),
                                       ('top_edge_visible',1,1,'extending highest'),
                                       ('height_visible',None,-1,'with the tallest visible outline')]:
            def metric(box):
                return box['xyxy'][3]-box['xyxy'][1] if axis is None else box['xyxy'][axis]
            ordered = sorted(group, key=lambda b: sign * metric(b))
            winner = ordered[0]
            oid = winner['object_id']
            members = set(groups.get(oid, [oid]))
            if (not members <= native_ids or winner['category'] != category
                    or winner.get('identity_source') not in native_rgb
                    or not usable_rgb_observation(observed.get(oid, {}), allow_partial=True)):
                continue
            if any(sign * (metric(other) - metric(winner)) <= 16 for other in ordered[1:]):
                continue
            result.append(Predicate(predicate_id=f'image_order:{camera}:{category}:{name}',
                kind='image_relation', value=f'{camera}:{name}', required_capability='rgb',
                phrase=f'the visible {category} {phrase} in the {CAMERA_VIEWS[camera]}',
                matches={other: other in members or other == oid for other in all_ids},
                source=source + '; camera-bound visible native RGB extents; 16px gap; '
                    'unknowns retained; projected boxes cannot supply a winning visible extent'))
    return result


def native_unit_groups(inventory, native_ids):
    """Set-valued camera coverage; never convert an assembly into one 3D ID."""
    return {c['object_id']: c['native_unit_object_ids'] for c in inventory
        if c.get('identity_source') == 'native_camera_box_group_point_correspondence'
        and len(set(c.get('native_unit_object_ids', []))) >= 2
        and set(c['native_unit_object_ids']) <= native_ids
        and all(c.get('native_unit_point_support', {}).get(oid, 0) >= 20 for oid in c['native_unit_object_ids'])}


def _image_separation(candidate, reference, axis, sign):
    if candidate['object_id'] == reference['object_id']:
        return False
    rect, ref = candidate['xyxy'], reference['xyxy']
    forward = rect[axis] - ref[axis+2] if sign > 0 else ref[axis] - rect[axis+2]
    opposite = ref[axis] - rect[axis+2] if sign > 0 else rect[axis] - ref[axis+2]
    return True if forward > 8 else False if opposite > 8 else None


def _image_landmarks(packet, camera, boxes, observed):
    """Name recognizable landmarks against the complete same-camera pool."""
    direct = []
    for box in boxes:
        oid, category = box['object_id'], box['category']
        if category == 'unknown' or not observed.get(oid, {}).get('target_visible'):
            continue
        ids = list(dict.fromkeys(c['object_id'] for c in boxes if c['category'] in {category, 'unknown'}))
        attrs = visual_descriptor(category, ids, oid, observed, visible_camera=camera)
        if attrs is None:
            continue
        attrs = landmark_attributes(category, observed.get(oid, {}), attrs)
        aid = hashlib.sha256(f"image:{camera}:{box.get('camera_object_id') or oid}:{attrs}".encode()).hexdigest()[:10]
        direct.append({'anchor_id':aid, 'object_id':oid, 'phrase':appearance_phrase(category, attrs),
            'camera_name':camera, 'descriptor_visual_attributes':[a.model_dump() for a in attrs], 'box':box})
    if packet.task.get('landmark_description_policy') != 'spatially-qualified-v1':
        return direct
    # A recognizable landmark may be unique by appearance AND a verified
    # relation. Start from independently unique seeds, never a circular target
    # reference. Unknown units retain their bounds/unknown native directions.
    seeds=list(direct)
    for box in boxes:
        oid,category=box['object_id'],box['category']
        if (category=='unknown' or any(a['object_id']==oid for a in direct)
                or observed.get(oid,{}).get('visibility_conflict')
                or not usable_rgb_observation(observed.get(oid,{}),allow_partial=allows_partial(packet))):
            continue
        group=[b for b in boxes if b['category'] in {category,'unknown'}]
        ids=list(dict.fromkeys(b['object_id'] for b in group))
        qualifiers=[]
        for predicate in packet.predicates:
            if predicate.kind in {'longitudinal','lateral'} and predicate.matches.get(oid) is True:
                qualifiers.append({'id':predicate.predicate_id,'phrase':predicate.phrase,
                    'matches':{id_:predicate.matches.get(id_) for id_ in ids},'dependencies':[]})
        for seed in seeds:
            if seed['object_id']==oid:
                continue
            for name,axis,sign in [('left',0,-1),('right',0,1)]:
                matches={b['object_id']:_image_separation(b,seed['box'],axis,sign) for b in group}
                if matches.get(oid) is True:
                    qualifiers.append({'id':f'image:{name}:{seed["anchor_id"]}',
                        'phrase':f'to the {name} of {seed["phrase"]}',
                        'matches':matches,'dependencies':[seed['object_id']]})
        found=None
        for count in (1,2):
            for combo in combinations(qualifiers,count):
                possible=[id_ for id_ in ids if all(q['matches'].get(id_) is not False for q in combo)]
                attrs=visual_descriptor(category,possible,oid,observed,visible_camera=camera)
                if attrs is not None:
                    attrs=landmark_attributes(category,observed[oid],attrs)
                    found=(combo,attrs);break
            if found:
                break
        if not found:
            continue
        combo,attrs=found
        signature=[q['id'] for q in combo]
        aid=hashlib.sha256(f'image-qualified:{camera}:{oid}:{attrs}:{signature}'.encode()).hexdigest()[:10]
        direct.append({'anchor_id':aid,'object_id':oid,'camera_name':camera,'box':box,
            'phrase':appearance_phrase(category,attrs)+' '+' and '.join(q['phrase'] for q in combo),
            'descriptor_visual_attributes':[a.model_dump() for a in attrs],
            'descriptor_spatial_predicates':signature,
            'dependency_object_ids':sorted({id_ for q in combo for id_ in q['dependencies']}),
            'descriptor_scope':'complete same-camera category/unknown inventory; attributes plus verified relations'})
    return direct


def _image_appearance_references(packet, camera, boxes, observed, native_ids, all_ids, source):
    """An intrinsic description can identify the RGB unit without any rank.

    Scope is explicit in the sentence and uses the same complete inventory and
    native identity contract as existing camera-qualified image references.
    Unknown/unobserved units inside this image remain possible competitors.
    """
    result = []
    if (not minimal_reference_policy(packet)
            or packet.task.get('image_identity_policy') != 'official-projection-v1'
            or packet.task.get('image_inventory_policy') != 'native-and-projected-v3-native-rgb'):
        return result
    for box in boxes:
        oid, category = box['object_id'], box['category']
        if oid not in native_ids or category == 'unknown':
            continue
        ids = list(dict.fromkeys(c['object_id'] for c in boxes if c['category'] in {category,'unknown'}))
        attrs = visual_descriptor(category,ids,oid,observed,visible_camera=camera)
        if attrs is None:
            continue
        # Offer one short, concrete description, not an empty camera-only cue.
        attrs = landmark_attributes(category,observed.get(oid,{}),attrs)
        if not attrs:
            continue
        descriptor = category + ''.join(f'~{a.kind}={a.value}' for a in sorted(attrs,key=lambda a:(a.kind,a.value)))
        result.append(Predicate(predicate_id=f'image_attribute:{camera}:{descriptor}', kind='image_relation',
            value=f'{camera}:{descriptor}', phrase=appearance_phrase(category,attrs)+f' in the {CAMERA_VIEWS[camera]}',
            matches={id_:id_ == oid for id_ in all_ids}, required_capability='rgb',
            source=source+'; confirmed intrinsic attributes unique in complete same-camera inventory; explicit camera; native identity required'))
    return result


def _image_boundary_references(packet, camera, boxes, observed, all_ids, source):
    """Use confirmed partial visibility at an actual original-image boundary.

    This is not a size/depth guess, image rank or crop-derived corner cue. All
    same-class/unknown units touching that boundary remain competitors unless
    the ordinary joint appearance proof excludes them.
    """
    if (packet.task.get('image_position_policy') != 'verified-boundary-v1'
            or not minimal_reference_policy(packet)
            or packet.task.get('image_identity_policy') != 'official-projection-v1'
            or packet.task.get('image_inventory_policy') != 'native-and-projected-v3-native-rgb'):
        return []
    view = next((v for v in packet.views if v.camera_name == camera),None)
    if view is None:
        return []
    result=[]
    for category in sorted({b['category'] for b in boxes}-{'unknown'}):
        group=[b for b in boxes if b['category'] in {category,'unknown'}]
        for edge,index,extent in (('left',0,0),('right',2,view.width),('top',1,0),('bottom',3,view.height)):
            matches={oid:False for oid in all_ids}
            for box in group:
                oid=box['object_id'];margin=abs(box['xyxy'][index]-extent)
                record=observed.get(oid,{})
                matches[oid]=(False if margin>8 else True if margin<=1
                    and usable_rgb_observation(record,allow_partial=True) and record.get('visibility')=='partial'
                    and not record.get('visibility_conflict') else None)
            if not any(value is True for value in matches.values()):
                continue
            result.append(Predicate(predicate_id=f'image_boundary:{camera}:{category}:{edge}',kind='image_relation',
                value=f'{camera}:{category}:{edge}',phrase=f'the {category} partly cut off by the {edge} edge of the {CAMERA_VIEWS[camera]}',
                matches=matches,required_capability='rgb',source=source+
                '; original image dimensions and boundary-contact bounds; camera-bound RGB confirms recognizable partial unit; unknown boundary units retained'))
    return result


def image_relation_catalog(packet, observations):
    from app.agent.appearance import effective_observations
    observations = effective_observations(packet, observations)
    inventory = packet.provenance.get('scene_image_candidates', [])
    if not inventory or any(not c.get('xyxy') or 'camera_name' not in c for c in inventory):
        return [], []
    native_ids = {o.object_id for o in packet.candidates}
    all_ids = native_ids | {c['object_id'] for c in inventory}
    predicates, anchors = [], []
    for camera in sorted({c['camera_name'] for c in inventory if 'camera_name' in c}):
        if not confirmed_image_camera(packet, camera):
            continue
        observed = camera_observations(packet, camera, observations)
        boxes = [c for c in inventory if c.get('camera_name') == camera and c.get('xyxy')]
        if not boxes:
            continue
        camera_view = CAMERA_VIEWS[camera]
        projected = packet.task.get('image_identity_policy') == 'official-projection-v1'
        inventory_source = 'official same-ID projections plus native camera boxes' if projected else 'complete native camera_box inventory'
        from app.quality.joint_reference import catalog as joint_catalog
        predicates.extend(joint_catalog(packet,camera,boxes,observed,all_ids))
        from app.quality.spatial_relations import catalog as spatial_catalog
        predicates.extend(spatial_catalog(packet,camera,boxes,observed,all_ids))
        predicates.extend(_image_appearance_references(packet,camera,boxes,observed,native_ids,all_ids,inventory_source))
        predicates.extend(_image_boundary_references(packet,camera,boxes,observed,all_ids,inventory_source))
        for category in sorted({c['category'] for c in boxes} - {'unknown'}):
            group = [c for c in boxes if c['category'] in {category, 'unknown'}]
            if allows_partial(packet):
                # This predicate explicitly says VISIBLE in a named camera.
                # Unknown/unrecognized partial crops remain possible; only an
                # explicit, camera-bound fully-hidden observation can exclude
                # an object that projects into this image.
                members = {c['object_id'] for c in group}
                matches = {}
                for oid in all_ids:
                    record = observed.get(oid, {})
                    if oid not in members:
                        matches[oid] = False
                    elif record.get('target_visible'):
                        matches[oid] = True
                    elif (record.get('visibility') == 'occluded'
                            and record.get('visibility_reason', '').strip()):
                        matches[oid] = False
                    else:
                        matches[oid] = None
                predicates.append(Predicate(predicate_id=f'image_visibility:{camera}:{category}',
                    kind='image_relation', value=f'{camera}:visible:{category}',
                    phrase=f'the {category} visible in the {camera_view}', matches=matches,
                    source=inventory_source + '; camera-bound RGB visibility; unknowns retained; explicit visible and camera required',
                    required_capability='rgb'))
            # Extremes are scene-wide, not chosen to fit the requested target.
            for name, axis, sign in [('leftmost', 0, 1), ('rightmost', 0, -1),
                                     ('topmost', 1, 1), ('bottommost', 1, -1)]:
                ordered = sorted(group, key=lambda c: sign * (c['xyxy'][axis] + c['xyxy'][axis+2]) / 2)
                winner = ordered[0]
                if winner['category'] != category:
                    continue
                center = sum(winner['xyxy'][axis::2]) / 2
                if len(ordered) > 1:
                    second = sum(ordered[1]['xyxy'][axis::2]) / 2
                    if abs(center - second) <= 16:
                        continue
                def separated(a, b):
                    return (b['xyxy'][axis]-a['xyxy'][axis+2] if sign > 0
                            else a['xyxy'][axis]-b['xyxy'][axis+2]) > 8
                if projected and any(not separated(winner, other) for other in ordered[1:]):
                    continue  # Projected extents may include occluded parts; center order alone is insufficient.
                oid = winner['object_id']
                if allows_partial(packet) and not usable_rgb_observation(observed.get(oid, {}), allow_partial=True):
                    continue  # Hidden projected extents cannot prove visible image order.
                # A unique image object must have a confirmed native identity
                # before it can uniquely identify a 3D training target.
                matches = {id_: (id_ == oid if oid in native_ids else None) for id_ in native_ids}
                matches.update({c['object_id']: c['object_id'] == oid for c in inventory if c['object_id'] not in native_ids})
                predicates.append(Predicate(predicate_id=f'image_order:{camera}:{category}:{name}',
                    kind='image_relation', value=f'{camera}:{name}',
                    phrase=(f'the {category} at the {"top" if name == "topmost" else "bottom"} of the {camera_view}'
                        if name in {'topmost', 'bottommost'} else f'the {name} {category} in the {camera_view}'),
                    source=inventory_source + ('; disjoint projected bounds + 8px guard' if projected else '; visible 2D center order + 16px tie guard') + '; explicit camera required',
                    required_capability='rgb', matches=matches))
                # Only a clearly separated second object is enumerated. This is
                # visible image order, never an arbitrary box index/distance rank.
                if len(ordered) > 1:
                    second_box = ordered[1]
                    preceding = ordered[:2]
                    adjacent = ordered[2:3]
                    second_center = sum(second_box['xyxy'][axis::2]) / 2
                    if (second_box['category'] == category
                            and (all(observed.get(c['object_id'], {}).get('target_visible')
                                and (not allows_partial(packet) or usable_rgb_observation(observed.get(c['object_id'], {}), allow_partial=True))
                                for c in preceding) if projected
                                else all(min(c['xyxy'][2]-c['xyxy'][0], c['xyxy'][3]-c['xyxy'][1]) >= 24 for c in preceding))
                            and all(abs(second_center - sum(c['xyxy'][axis::2])/2) > 16 for c in adjacent)):
                        if projected and (not separated(winner, second_box)
                                or any(not separated(second_box, other) for other in ordered[2:])):
                            continue
                        second_id = second_box['object_id']
                        direction = {'leftmost':'left','rightmost':'right','topmost':'top','bottommost':'bottom'}[name]
                        second_matches = {id_: (id_ == second_id if second_id in native_ids else None) for id_ in native_ids}
                        second_matches.update({c['object_id']: c['object_id'] == second_id for c in inventory if c['object_id'] not in native_ids})
                        predicates.append(Predicate(predicate_id=f'image_order:{camera}:{category}:second_from_{direction}',
                            kind='image_relation', value=f'{camera}:second_from_{direction}',
                            phrase=f'the second {category} from the {direction} in the {camera_view}',
                            source=inventory_source + '; first two visibly confirmed and disjoint' if projected else inventory_source + '; first two boxes >=24px; adjacent center gaps >16px; explicit camera required',
                            required_capability='rgb', matches=second_matches))
        if projected and checks_rgb_visibility(packet):
            predicates.extend(scoped_image_orders(camera, boxes, observed, native_ids, all_ids, inventory_source,
                allow_partial=allows_partial(packet)))
        if allows_partial(packet):
            predicates.extend(visible_image_orders(camera, boxes, observed, native_ids, all_ids, inventory_source))
            predicates.extend(visible_extent_orders(camera, boxes, observed, native_ids, all_ids, inventory_source))
        for anchor in _image_landmarks(packet,camera,boxes,observed):
            oid, anchor_id, phrase = anchor['object_id'], anchor['anchor_id'], anchor['phrase']
            anchors.append({k:v for k,v in anchor.items() if k != 'box'} | {
                'scope': 'unique visible image landmark in this camera; ' + inventory_source})
            # Camera scope identifies the landmark; front/rear still comes from
            # its native heading, never from image height, size or ego axes.
            from app.quality.relations import vehicle_heading_anchor, vehicle_heading_match, VEHICLE_HEADING_POLICY
            native = {o.object_id:o for o in packet.candidates}
            ref = native.get(oid)
            if vehicle_heading_anchor(packet,ref,observed.get(oid,{}),anchor['box']):
                members = {b['object_id'] for b in boxes}
                complete = (projected and checks_rgb_visibility(packet)
                    and packet.task.get('image_inventory_policy') == 'native-and-projected-v3-native-rgb'
                    and minimal_reference_policy(packet))
                for name, sign, words in [('front',1,'directly in front of'),('rear',-1,'directly behind')]:
                    matches = {}
                    for cid in all_ids:
                        if cid == oid or cid in anchor.get('dependency_object_ids',[]):
                            matches[cid] = False
                        elif cid not in members:
                            matches[cid] = False if complete else None
                        else:
                            # Unlinked 2D objects have no reliable heading-frame
                            # location. They remain unknown even if pixels look
                            # in front/behind. No identity is inferred here.
                            matches[cid] = vehicle_heading_match(native[cid],ref,sign) if cid in native else None
                    predicates.append(Predicate(predicate_id=f'image_vehicle_heading:{camera}:{name}:{anchor_id}',
                        kind='image_relation',value=anchor_id,phrase=f'{words} {phrase} in the {camera_view}',
                        required_capability='rgb',matches=matches,
                        source=f'{VEHICLE_HEADING_POLICY}; unique camera landmark {oid} with verified native link; '
                            'native front-face heading, disjoint extents + 1m guard, central-half alignment, same level; '
                            'complete camera pool and unlinked/unknown rivals retained'))
            for name, axis, sign in [('left_of', 0, -1), ('right_of', 0, 1),
                                     ('above', 1, -1), ('below', 1, 1)]:
                image_matches = {id_: False for id_ in all_ids - native_ids}
                linked_matches = {}
                for candidate in boxes:
                    cid = candidate['object_id']
                    match = _image_separation(candidate,anchor['box'],axis,sign)
                    if cid in anchor.get('dependency_object_ids',[]):
                        match = False  # A reference may not describe itself via its landmark.
                    if cid in native_ids:
                        linked_matches[cid] = match
                    else:
                        image_matches[cid] = match
                # A named-camera landmark relation is evaluated in that complete
                # image inventory, just like the existing named-camera order and
                # visibility predicates. Objects with no entry in this camera
                # cannot satisfy an image relation there. Projected unknown or
                # zero-point units IN the image still retain their actual bounds
                # and unknown relation/appearance. Never apply this to an old or
                # incomplete inventory, or to an unqualified native 3D relation.
                complete_scope = (projected and checks_rgb_visibility(packet)
                    and packet.task.get('image_inventory_policy') == 'native-and-projected-v3-native-rgb'
                    and minimal_reference_policy(packet))
                matches = {id_: linked_matches.get(id_, False if complete_scope else None) for id_ in native_ids}
                matches.update(image_matches)
                words = name.replace('_', ' ')
                if name in {'left_of', 'right_of'}:
                    words = 'to the ' + words
                predicates.append(Predicate(predicate_id=f'image_anchor:{camera}:{name}:{anchor_id}',
                    kind='image_relation', value=anchor_id,
                    phrase=f'{words} {phrase} in the {camera_view}',
                    source=f'{inventory_source}; visible unique landmark {oid}; disjoint image bounds + 8px guard; explicit camera required',
                    required_capability='rgb', matches=matches))
    from app.quality.spatial_relations import native_only
    if native_only(packet):
        predicates=[p for p in predicates if draft_reference_predicate(packet,p)]
        anchors=[]  # Pixel-qualified landmark descriptions cannot leak into 3D claims.
    return predicates, anchors


def expression_candidates(packet, predicates, attributes, observations):
    """Joint proof over native 3D objects and camera-qualified native 2D objects."""
    native_ids = {o.object_id for o in packet.candidates}
    spatial = remaining(packet.candidates, predicates)
    native, exclusions = filter_visual_candidates(spatial, attributes, observations)
    inventory = packet.provenance.get('scene_image_candidates', [])
    image_predicates = [p for p in predicates if p.kind == 'image_relation']
    image_ids = list(dict.fromkeys(c['object_id'] for c in inventory
        if c['category'] in {packet.target.category, 'unknown'}
        and (image_predicates or c['object_id'] not in native_ids)))
    image_ids = [oid for oid in image_ids if all(p.matches.get(oid) is not False for p in image_predicates)]
    cameras = {int(p.predicate_id.split(':')[1]) for p in image_predicates}
    visible_camera = (next(iter(cameras)) if len(cameras) == 1
        and packet.task.get('source_light_policy') == 'visible-region-v1' else None)
    image, _ = filter_visual_candidates(image_ids, attributes, observations, visible_camera=visible_camera)
    # The camera-qualified sentence selects one actual 2D object, whose 3D
    # identity is already verified. No guessed links or unmatched 2D deletions.
    image_unique = bool(image_predicates and image == [packet.target_object_id]
        and all(p.matches.get(packet.target_object_id) is True for p in predicates))
    if image_unique:
        from app.datasets.correspondence import has_multimodal_evidence
        image_unique = has_multimodal_evidence(packet)
    if image_unique:
        native = [packet.target_object_id]
    groups = native_unit_groups(inventory, native_ids)
    # An assembly-level RGB label is covered by several native training units.
    # It adds no new 3D identity, but every member still competes in native above.
    # Never choose one member merely because the RGB assembly is unique.
    return {'native': native, 'image_only': [oid for oid in image if oid not in native_ids and oid not in groups],
            'native_unit_groups': {oid:groups[oid] for oid in image if oid in groups},
            'image_candidates': image, 'camera_identity_proof': image_unique, 'visual_exclusions': exclusions}


def without_redundant_camera_visibility(packet, predicates, attributes, observations):
    """Drop a camera-only qualifier only after the full unscoped proof passes.

    Native/unknown/image-only competitors keep exactly the ordinary proof rules.
    Camera-bound orders and landmark relations still require their camera name;
    a necessary visibility cue is never silently treated as an ego direction.
    """
    visibility = [p for p in predicates if p.predicate_id.startswith('image_visibility:')]
    if not visibility or any(p.kind == 'image_relation' and p not in visibility for p in predicates):
        return list(predicates)
    trial = [p for p in predicates if p not in visibility]
    if any(p.matches.get(packet.target_object_id) is not True for p in trial):
        return list(predicates)
    proof = expression_candidates(packet, trial, attributes, observations)
    if proof['native'] == [packet.target_object_id] and not proof['image_only']:
        return trial
    return list(predicates)
