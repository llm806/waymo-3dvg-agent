"""Conservative wording fallback when fewer than three model calls remain."""
import hashlib
import json

from app.agent.appearance import required_body_color
from app.quality.description_search import CAMERA_VIEWS, appearance_phrase, image_order_attributes
from app.schemas.contracts import GeneratedExpression, VisualAttribute
from app.quality.visibility_policy import checks_rgb_visibility


def _style_seed(packet):
    # Stable across processes, concurrent workers and task resumes. No global RNG.
    identity = [packet.context, packet.frame_timestamp_micros, packet.target_object_id, packet.source_view]
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:4], 'big')


def generation_style(packet):
    from app.quality.description_search import scene_reference_policy
    if scene_reference_policy(packet):
        return ('Prefer a specific object-first noun phrase with confirmed appearance and ego position.',
            'Prefer a short object-first sentence with a verified distinguishing landmark relation.',
            'Prefer a concise object-first phrase; omit any landmark not needed by the verified proof.')[_style_seed(packet) % 3]
    return (
        'Prefer an object-first referring phrase.',
        'Prefer an object-first sentence with a natural predicate or relative clause.',
        'Prefer a short object-first phrase with a supported ego direction.',
        'Prefer a camera-view-first sentence only when a registered image relation names that view.',
    )[_style_seed(packet) % 4]


def _ego_direction(longitudinal, lateral, alternate):
    if longitudinal in {'front', 'behind'} and lateral:
        # Avoid shared-object coordination ("in front of and to the left of"):
        # it is grammatical but more easily misread by the blind claim parser.
        return f'{"ahead" if longitudinal == "front" else "behind"} and to the {lateral} of the ego vehicle'
    if longitudinal == 'front':
        direction = 'in front of' if alternate else 'ahead of'
    elif longitudinal == 'behind':
        direction = 'to the rear of' if alternate else 'behind'
    else:
        direction = ''
    if lateral:
        direction = f'on the {lateral} side of' if alternate else f'to the {lateral} of'
    return direction + ' the ego vehicle' if direction else ''


def render_verified_plan(packet, observations):
    """Return a candidate sentence, never a validation or acceptance result.

    Category, ego directions, one camera visibility/order cue and up to two
    confirmed native landmark relations are handled. Other plans and language
    repairs retain normal model generation.
    The resulting sentence still needs appearance and both blind claim checks.
    """
    plan = packet.provenance.get('admission_plan', {})
    if (packet.task.get('manual_review_feedback') or not packet.task.get('annotation_search')
            or not checks_rgb_visibility(packet)
            or not plan.get('passed') or not plan.get('unique_plan_found')):
        return None
    catalog = {p.predicate_id: p for p in packet.predicates}
    ids = plan.get('predicate_ids', [])
    if f'category:{packet.target.category}' not in ids or any(id_ not in catalog for id_ in ids):
        return None
    selected = [catalog[id_] for id_ in ids]
    from app.quality.description_search import draft_reference_predicate
    if any(not draft_reference_predicate(packet, p) for p in selected):
        return None
    if any(p.matches.get(packet.target_object_id) is not True for p in selected):
        return None
    orders = [p for p in selected if p.kind == 'image_relation']
    if len(orders) > 1:
        return None
    relation_kinds = {'relative_left', 'relative_right', 'ahead_of', 'behind_anchor', 'vehicle_heading'}
    relations = [p for p in selected if p.kind in relation_kinds]
    if len(relations) > 2:
        return None
    if any(p.kind not in {'category', 'longitudinal', 'lateral', 'image_relation', *relation_kinds} for p in selected):
        return None
    anchors = {a['anchor_id']: a for a in packet.provenance.get('relation_anchors', [])}
    for relation in relations:
        anchor = anchors.get(relation.value)
        if (anchor is None or anchor['object_id'] == packet.target_object_id
                or not observations.get(anchor['object_id'], {}).get('target_visible')):
            return None
    parts = orders[0].predicate_id.split(':') if orders else None
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    if parts:
        if parts[0] not in {'image_order', 'image_visibility', 'image_attribute', 'image_boundary'}:
            return None
        if len(parts) != (4 if parts[0] in {'image_order','image_boundary'} else 3) or parts[2].split('~')[0] != packet.target.category:
            return None
        if parts[1] != str(camera) or camera not in CAMERA_VIEWS:
            return None
    attributes = [VisualAttribute.model_validate(a) for a in plan.get('visual_attributes', [])]
    from app.quality.description_search import descriptive_attributes, expression_candidates
    from app.agent.harness import description_predicates
    for attribute in descriptive_attributes(packet, observations):
        if attribute not in attributes:
            attributes.append(attribute)
    if packet.task.get('description_detail_policy'):
        for id_ in description_predicates(packet):
            if id_ not in ids:
                ids = [*ids, id_]
                selected.append(catalog[id_])
        proof = expression_candidates(packet, selected, attributes, observations)
        if len(attributes)>6 or proof['native'] != [packet.target_object_id] or proof['image_only']:
            return None
    # Minimizing distinguishing attributes must not erase a confirmed core noun.
    # The candidate still undergoes appearance verification and both blind checks.
    subtypes = {a['value'] for a in observations.get(packet.target_object_id, {}).get('attributes', [])
        if a['kind'] == 'subtype'} if packet.target.category in {'vehicle', 'sign'} else set()
    if len(subtypes) == 1:
        subtype = VisualAttribute(kind='subtype', value=next(iter(subtypes)))
        if subtype not in attributes:
            if len(attributes) >= 6:
                return None
            attributes.append(subtype)
    for attribute in image_order_attributes(orders[0]) if orders else []:
        if attribute.model_dump() not in observations.get(packet.target_object_id, {}).get('attributes', []):
            return None
        if attribute not in attributes:
            if len(attributes) >= 6:
                return None
            attributes.append(attribute)
    body = required_body_color(packet, observations)
    if body and not any(a.kind == 'color' and a.value == body for a in attributes):
        if len(attributes) >= 6:
            return None
        attributes.append(VisualAttribute(kind='color', value=body))
    seed = _style_seed(packet)
    layout, alternate = seed % 4, bool((seed // 4) % 2)
    # A relative clause after "with a trailer" or clothing could attach to that
    # part instead of the target. Keep canonical image wording for such nouns.
    relative_clause = alternate and not any(a.kind in {
        'feature', 'upper_clothing_color', 'lower_clothing_color', 'helmet_color', 'backpack_color'} for a in attributes)
    subject = appearance_phrase(packet.target.category, attributes)
    # Preserve canonical nouns/attributes, including separately boxed bus or
    # excavator parts. Surface variation never changes the selected facts.
    subject = subject.replace(' suv', ' SUV')
    noun = subject
    view = CAMERA_VIEWS.get(camera)
    order = (parts[3] if parts[0] == 'image_order' else 'boundary:'+parts[3] if parts[0] == 'image_boundary'
        else 'appearance' if parts[0] == 'image_attribute' else 'visible') if parts else 'ego'
    if order == 'ego':
        pass
    elif order == 'appearance':
        subject += f' in the {view}'
    elif order.startswith('boundary:'):
        subject += f' partly cut off by the {order.split(":")[1]} edge of the {view}'
        layout = 0
    elif order == 'visible':
        subject = (subject + f' that is visible in the {view}' if relative_clause
                   else f'the visible {subject[4:]} in the {view}')
    elif order in {'leftmost_visible', 'rightmost_visible'}:
        subject = f'the {order.removesuffix("_visible")} visible {subject[4:]} in the {view}'
    elif order in {'leftmost', 'rightmost'}:
        subject = f'the {order} {subject[4:]} in the {view}'
    elif order in {'second_from_left', 'second_from_right'}:
        subject += f' {"that is " if relative_clause else ""}second from the {order.removeprefix("second_from_")} in the {view}'
    else:
        return None
    longitudinal = next((p.value for p in selected if p.kind == 'longitudinal'), None)
    lateral = next((p.value for p in selected if p.kind == 'lateral'), None)
    direction = _ego_direction(longitudinal, lateral, alternate)
    landmarks = [p.phrase for p in relations]  # Keep registered anchor descriptions verbatim.
    if layout == 3 and parts:
        # Move the camera qualifier with its image cue, never into ego directions.
        local_subject = ('the visible ' + noun[4:] if order == 'visible'
                         else subject.removesuffix(f' in the {view}'))
        predicates = [p for p in [direction, *landmarks] if p]
        expression = f'in the {view}, {local_subject}'
        if predicates:
            expression += ' is ' + ' and '.join(predicates)
    elif layout == 2 and direction:
        expression = subject + ', ' + direction
        if landmarks:
            expression += ', ' + ' and '.join(landmarks)
    elif layout == 1 and (direction or landmarks):
        expression = subject + ' is ' + ' and '.join(p for p in [direction, *landmarks] if p)
    else:
        cues = [p for p in [direction, *landmarks] if p]
        expression = subject + ((', ' if parts else ' ') + ', '.join(cues) if cues else '')
    expression = expression[0].upper() + expression[1:] + '.'
    if len(expression) > 400:
        return None
    from app.agent.harness import repair_redundant_camera_qualifier
    generated = GeneratedExpression(expression=expression, predicate_ids=list(ids), visual_attributes=attributes)
    return repair_redundant_camera_qualifier(packet,generated,observations)[0]
