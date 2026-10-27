"""Anonymous scene observations; missing or conflicting evidence stays unknown."""
import base64
import json
import hashlib
import copy
import threading
from collections import OrderedDict

from app.infrastructure.artifacts import checked_asset
from app.schemas.contracts import SceneVisualObservation, VisualObservation, VisualAttribute
from app.quality.visibility_policy import allows_partial, checks_rgb_visibility, source_camera_inventory, usable_rgb_observation, visibility_evidence

OBSERVATION_SCOPE_GUIDE = (
    'For pedestrians and cyclists, record clearly visible upper/lower garment colors separately. '
    'Use helmet_color for the clearly visible color of a helmet actually worn by this person. '
    'A dark head, hair, cap, hood or an occluder is not evidence of a helmet. '
    'Do not treat helmet color as shirt color, vehicle paint, or absence of another feature. '
    'Use feature=handheld bag for a bag visibly held or carried at the hand/side; '
    'use backpack for a bag visibly worn on the back. Do not confuse these with each other '
    'or with a neighboring person. Report these in attributes, not only visibility_reason. '
    'Use feature=handheld camera for a camera or camcorder visibly held by this person, and '
    'feature=binoculars for confirmed binoculars. Inspect the hands and device separately from '
    'any shoulder bag: a person can hold a camera and also carry a bag. Do not substitute bag '
    'for a camera or binoculars because a device is small. If the device type is unclear, omit it. '
    'Include confirmed held-device features in attributes, not only visibility_reason. '
    'Nighttime or distance alone does not make a target invisible. A recognizable vehicle silhouette '
    'and its own lamp arrangement can establish partial visibility without known paint or subtype. '
    'Record feature=illuminated headlights or illuminated taillights when confirmed on this unit, '
    'not merely in the reason. Distinguish front lamps from red rear lamps, reflections and building lights. '
    'For vehicles, visible_lights checks only lights visibly illuminated in this camera. '
    'Use visible_lights, not absent_features, for negative illuminated-light checks. '
    'Use present for a confirmed lamp cue; absent ONLY after inspecting the entire indicated original-image '
    'region in the full scene and context and confidently finding no visible cue of that kind. '
    'Set complete_region_inspected=true and explain the inspection when using absent. '
    'Unclear exposure/noise, glare, ambiguous ownership or an incompletely inspected region means uncertain. '
    'This does not establish that physical lights are off, absent or broken. Unknown body paint can remain '
    'unknown independently of this camera-specific check. Never infer motion from illuminated lights. '
    'Unknown bag presence or color provides no evidence. Never infer intention to cross a road. '
    'The supplied category is the native dataset coarse label, not a whitelist of everyday subtypes. '
    'For a native vehicle unit, construction machinery such as an excavator is a valid recognizable '
    'vehicle subtype; it need not be a standard road car, truck or bus. '
    'Judge visibility from the selected unit\'s RGB surfaces, never from road eligibility or whether '
    'its specific subtype appears in the attribute schema. Report excavator when confirmed; '
    'for an unlisted subtype omit that attribute while still reporting actual visibility and other confirmed attributes. '
    'A native sign unit can be a portable A-frame construction sign or barricade stand. '
    'Use subtype construction barricade for a selected portable traffic barricade with a striped rail/panel '
    'and its own supporting stand; use a-frame sign for other confirmed folding sign units. '
    'Report salient supported details in attributes, not only visibility_reason. '
    'For confirmed two-color stripes, use feature=orange and white stripes, red and white stripes, '
    'or black and yellow stripes as appropriate; otherwise use feature=stripes when only the pattern is clear. '
    'A two-color striped panel need not have one dominant color: omit kind=color when ambiguous, '
    'but retain its confirmed striped feature. Use feature=warning light for a visible light mounted '
    'on this unit; a still image cannot establish that it is flashing. '
    'Report subtype a-frame sign when the selected unit itself has a recognizable folding A-frame stand, '
    'even if its printed face is partly hidden or viewed from behind. Do not require readable sign text. '
    'A rectangular sign panel mounted on a single pole/post is NOT an A-frame sign. '
    'The a-frame sign subtype requires visible folding legs or a triangular stand belonging to '
    'the selected unit itself. A mounting bracket, support pole, sign back or unreadable face '
    'does not establish that subtype; omit uncertain subtype but retain all other confirmed visible attributes. '
    'For signs, record feature=visible rear face when the back surface is confirmed, and '
    'feature=pole mounted when this panel is visibly attached to its own pole. '
    'Color describes the visible panel surface, including a confirmed rear surface, never an unseen printed face. '
    'Record a clear dominant panel color in attributes; do not substitute a narrow colored edge for the panel color. '
    'Shape, visible rear face and mounting details belong in attributes, not only visibility_reason. '
    'A visible back surface still makes the sign visible; do not mark it occluded merely because '
    'its printed front face is facing away. '
    'Do not confuse an adjacent plastic barrier, pipe, caution tape or vehicle with that selected sign unit. '
    'A real identity/category mismatch, such as a pedestrian occluding the selected vehicle, still requires rejection. '
    'Assess the single annotation unit identified by the marked region, not a larger connected assembly. '
    'Construction machines can have separate native boom and cab/chassis units. '
    'Use subtype excavator boom for a separately boxed boom-and-bucket unit, excavator body for a '
    'separately boxed cab/chassis unit, and excavator for a whole-machine unit or uncertain part scope. '
    'Do not extend a selected boom unit to the whole machine or borrow the boom from an adjacent body unit. '
    'For an articulated bus, use subtype bus section when the selected native unit is one separately '
    'boxed front or rear section. A whole-bus 2D label does not merge those native 3D units. '
    'For an official 3D projection, the unit is one native 3D box; connected vehicle sections or trailers '
    'may have separate boxes. Do not extend this unit to an adjoining section just because they share paint '
    'or are physically connected. A different section leaving the image does not truncate the selected unit. '
    'Judge camera-edge truncation against the original camera image, not the boundary of a derived tight '
    'crop or context crop. Use context to check the selected unit itself. '
    'The marked region and boundary metadata locate the unit but do not prove visibility or identity. '
    'When same-category units overlap, attribute each visible surface to its own unit before reporting paint or type. '
    'If only a roof, window or lights of the selected unit are visible, vehicle_colors.body must be null '
    'unless a painted body panel of THAT unit can be identified separately. Never borrow a foreground '
    'vehicle body inside the rectangle. Glass, including a panoramic roof, is not roof paint. '
    'A roof-only sliver does not establish sedan/SUV/truck subtype; omit unsupported subtype. '
    'Truck is a broad type that includes pickup trucks; report pickup truck when that specific type is confirmed. '
    'Use only the schema values for each attribute kind. Car and vehicle are categories, not subtypes; '
    'omit subtype when a more specific supported type is not confirmed. '
    'Unknown attributes must be omitted; unknown paint must be JSON null, never the string "null". '
)


class SceneObservationCache:
    """Bounded batch-local reuse of actual, anonymous scene observations.

    No target identity checks or uncertain/conflicting records are reused. Every
    hit rechecks the current assets; this caches perception, never file trust.
    Source run references remain durable in the consuming run's checkpoint.
    """
    def __init__(self, capacity=4096):
        if type(capacity) is not int or capacity<1:
            raise ValueError('Observation cache capacity must be positive')
        self.capacity=capacity
        self.entries=OrderedDict()
        self.lock=threading.Lock()

    @staticmethod
    def key(packet,item,scope):
        if item.get('attribute_recheck'):
            return None  # A focused uncertainty check must receive a fresh observation.
        digest=candidate_input_digest(packet,item)
        if digest is None:
            return None
        camera=item.get('camera_name')
        view=next((v for v in packet.views if v.camera_name==camera),None)
        regions={c['object_id']:c for c in packet.provenance.get('scene_image_candidates',[])
            if c.get('camera_name')==camera}
        foreground=[(oid,regions.get(oid,{}).get('category'),regions.get(oid,{}).get('xyxy'))
            for oid in sorted(item.get('potential_foreground_object_ids',[]))]
        identity=[scope,packet.context,packet.frame_timestamp_micros,item['object_id'],item['category'],camera,
            item.get('camera_object_id'),item.get('identity_source'),digest,
            regions.get(item['object_id'],{}).get('xyxy'),
            regions.get(item['object_id'],{}).get('native_unit_object_ids'),
            (view.width,view.height) if view else None,foreground,packet.task.get('visibility_policy'),
            packet.task.get('source_light_policy'),
            packet.provenance.get('components'),packet.provenance.get('tool_sha256')]
        return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def reuse(self,packet,manifest,scope,evidence_root):
        records,receipts={},[]
        for item in manifest:
            oid=item['object_id']
            if oid==packet.target_object_id or oid in records:
                continue
            key=self.key(packet,item,scope)
            with self.lock:
                entry=self.entries.get(key)
                if entry is not None:
                    self.entries.move_to_end(key)
                    entry=copy.deepcopy(entry)
            if entry is None:
                continue
            names=candidate_artifacts(packet,item)
            names.extend(v.artifacts['scene'] for v in packet.views
                if v.camera_name==item.get('camera_name') and 'scene' in v.artifacts)
            for name in set(names):
                checked_asset(evidence_root/packet.packet_id,name,packet.provenance['artifact_sha256'][name])
            records[oid]=entry['record']
            receipts.append({**entry['source'],'object_id':oid,'camera_name':item['camera_name'],
                'packet_id':packet.packet_id,'input_digest':candidate_input_digest(packet,item),'cache_key':key})
        return records,receipts

    def remember(self,packet,manifest,records,scope,run_id,call_folder):
        entries=[]
        for item in manifest:
            oid=item['object_id'];record=records.get(oid,{})
            if (oid==packet.target_object_id or record.get('visibility_conflict')
                    or record.get('observed_camera_name')!=item.get('camera_name')
                    or not usable_rgb_observation(record,allow_partial=allows_partial(packet))):
                continue
            attributes={}
            for attr in record.get('attributes',[]):
                attributes.setdefault(attr['kind'],set()).add(attr['value'])
            if (any(not compatible_attribute_values(kind,values) for kind,values in attributes.items() if kind!='feature')
                    or attributes.get('feature',set()).intersection(record.get('absent_features',[]))):
                continue
            key=self.key(packet,item,scope)
            if key is None or record.get('observed_input_sha256')!=candidate_input_digest(packet,item):
                continue
            entries.append((key,{'record':copy.deepcopy(record),'source':{
                'source_run_id':run_id,'source_packet_id':packet.packet_id,'source_call_folder':call_folder}}))
        with self.lock:
            for key,entry in entries:
                self.entries[key]=entry
                self.entries.move_to_end(key)
            while len(self.entries)>self.capacity:
                self.entries.popitem(last=False)

    def clear(self):
        with self.lock:
            self.entries.clear()


def visibility_instruction(packet):
    if allows_partial(packet):
        return ('Partial occlusion or camera-edge truncation alone does NOT disqualify an object. '
            'Set target_visible=true when the selected native annotation unit itself is separately recognizable, '
            'even when visibility=partial. Explain which visible surfaces establish its identity, separately '
            'from any foreground occluder. If fully hidden or identity is uncertain, target_visible=false. '
            'Describe only clearly visible attributes; never infer hidden parts. Body paint refers to '
            'confirmed visible body surfaces, not guessed paint on hidden portions. '
            'A sign seen from behind may be recognized as a sign, but its unseen face cannot supply attributes. ')
    return ('Reject a selected unit whose own RGB surfaces are occluded, cut off or cannot be identified separately. '
        'A sign seen only from the back is unsuitable. ')


def observation_attributes(record, *, allow_partial=False):
    """Use explicit body paint as color; roof paint stays in raw evidence."""
    attributes = list(record.get('attributes', []))
    if {'kind': 'subtype', 'value': 'pickup truck'} in attributes:
        attributes = [a for a in attributes if a != {'kind': 'subtype', 'value': 'truck'}]
    if {'kind':'subtype','value':'bus section'} in attributes:
        attributes = [a for a in attributes if a != {'kind':'subtype','value':'bus'}]
    if any(a == {'kind':'subtype','value':part} for part in ('excavator boom','excavator body') for a in attributes):
        attributes = [a for a in attributes if a != {'kind':'subtype','value':'excavator'}]
    colors = record.get('vehicle_colors')
    if colors is None:
        return attributes  # Legacy colors retain their existing conflict checks.
    body = colors.get('body')
    raw_colors = {a['value'] for a in attributes if a['kind'] == 'color'}
    attributes = [a for a in attributes if a['kind'] != 'color']
    if not usable_rgb_observation(record, allow_partial=allow_partial) or body not in raw_colors:
        return attributes
    if raw_colors - {body, colors.get('roof')}:
        return attributes  # An unexplained third color remains a conflict.
    return [*attributes, {'kind': 'color', 'value': body}]


def effective_observations(packet, observations):
    """Unclear RGB observations supply no appearance proof; keep raw records."""
    result = {}
    for oid, record in (observations or {}).items():
        partial = allows_partial(packet)
        effective = {**record, 'attributes': observation_attributes(record, allow_partial=partial)}
        # Canonical facts must not later resurrect a discarded color from raw
        # paint metadata. The original observation remains in state and logs.
        effective.pop('vehicle_colors', None)
        if packet.task.get('source_light_policy') != 'visible-region-v1':
            effective.pop('visible_light_region', None)
        if checks_rgb_visibility(packet) and not usable_rgb_observation(record, allow_partial=partial):
            effective.update(target_visible=False, attributes=[], absent_features=[])
        elif partial and record.get('visibility') == 'partial':
            effective['absent_features'] = []  # Hidden parts cannot establish absence.
        result[oid] = effective
    return result


def required_body_color(packet, observations):
    """A clear, confirmed body color must survive description rendering."""
    from app.quality.description_search import minimal_reference_policy
    if minimal_reference_policy(packet):
        return None  # Keep color when claimed/needed, not as compulsory prose.
    if (packet.target.category != 'vehicle' or not checks_rgb_visibility(packet)
            or not packet.provenance.get('admission_plan', {}).get('passed')):
        return None
    record = (observations or {}).get(packet.target_object_id, {})
    if not usable_rgb_observation(record, allow_partial=allows_partial(packet)):
        return None
    colors = {a['value'] for a in observation_attributes(record, allow_partial=allows_partial(packet)) if a['kind'] == 'color'}
    return next(iter(colors)) if len(colors) == 1 else None


def target_visibility_error(packet, observation, observations=None):
    if packet.task.get('view_selection_policy'):
        camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
        if str(camera) not in (source_camera_inventory(packet) or {}):
            return 'precheck_target_view_unsuitable'
    if not observation.target_visible:
        return 'precheck_target_not_visible'
    if not checks_rgb_visibility(packet):
        return None
    partial = allows_partial(packet)
    if not usable_rgb_observation(observation.model_dump(), allow_partial=partial):
        return 'precheck_target_identity_uncertain' if partial else 'precheck_target_occluded_or_uncertain'
    camera = next((v for v in packet.views if v.camera_label == packet.source_view), None)
    if camera is None:
        return 'precheck_target_view_unsuitable'
    if not partial and (camera.projection_touches_boundary is not False
            or packet.target.category == 'sign' and camera.target_front_facing is not True):
        return 'precheck_target_view_unsuitable'
    other = (observations or {}).get(packet.target_object_id, {})
    other = other.get('camera_observations', {}).get(str(camera.camera_name), other)
    # An alternate camera cannot certify or veto visibility in the source camera.
    same_camera = other.get('observed_camera_name') == camera.camera_name
    if same_camera:
        evidence = visibility_evidence(other, allow_partial=partial)
        if other.get('visibility_conflict') or evidence is False:
            return 'precheck_source_visibility_conflict'
        if evidence is None:
            return 'precheck_target_identity_unconfirmed'
    if partial and observation.visibility == 'partial' and observations is not None and not same_camera:
        return 'precheck_target_identity_unconfirmed'
    return None


def candidate_artifacts(packet, item):
    """Exact context/detail inputs for one anonymous candidate."""
    name = item.get('visibility_artifact', item['artifact']) if packet.task.get('visibility_policy') else item['artifact']
    detail = item.get('detail_artifact')
    if not detail:
        primary = next((m for m in packet.provenance.get('appearance_candidates', [])
            if m['candidate_id'] == item['candidate_id'] and m.get('camera_name') == item.get('camera_name')), None)
        if primary:
            detail = primary.get('detail_artifact', primary['artifact'])
    return [name, *([detail] if detail and detail != name else [])]


def candidate_input_digest(packet, item):
    if not item.get('artifact'):
        return None
    names = candidate_artifacts(packet, item)
    if checks_rgb_visibility(packet):
        names.extend(v.artifacts['scene'] for v in packet.views
            if v.camera_name == item.get('camera_name') and 'scene' in v.artifacts)
    hashes = packet.provenance.get('artifact_sha256', {})
    if not all(name in hashes for name in names):
        return None
    inputs = [hashes[name] for name in names]
    if item.get('attribute_recheck'):
        inputs.append({'attribute_recheck':sorted(item['attribute_recheck'])})
    return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()


def redundant_relook(packet, item, record):
    """Only skip a generic reread of identical, already recognizable RGB.

    An uncertain/conflicting observation or a new camera/crop still warrants
    acquisition. Target identity and attribute-conflict rechecks are separate.
    """
    if item.get('attribute_recheck'):
        return False
    record = record.get('camera_observations', {}).get(str(item.get('camera_name')), record)
    if (record.get('observed_camera_name') != item.get('camera_name') or record.get('visibility_conflict')
            or not usable_rgb_observation(record, allow_partial=allows_partial(packet))):
        return False
    digest = candidate_input_digest(packet, item)
    return bool(digest and digest == record.get('observed_input_sha256'))


def observation_messages(packet, evidence_root, manifest=None):
    from app.agent.observation_wire import CompactSceneObservation
    manifest = packet.provenance.get('appearance_candidates', []) if manifest is None else manifest
    schema = CompactSceneObservation.model_json_schema()
    schema['$defs']['CompactCandidateObservation']['properties']['id']['enum'] = [m['candidate_id'] for m in manifest]
    schema['properties']['c'].update(minItems=len(manifest), maxItems=len(manifest))
    content = [{'type': 'text', 'text': json.dumps({'output_schema': schema,
        'candidate_ids': [m['candidate_id'] for m in manifest],
        **({'attribute_checks':[{'candidate_id':m['candidate_id'],'kinds':m['attribute_recheck']}
                for m in manifest if m.get('attribute_recheck')],
            'attribute_check_instruction':'Reinspect these attribute kinds from the actual pixels. No value is suggested. '
                'Return only clearly confirmed values in the ordinary attributes field; omit any still uncertain. '
                'Continue reporting all other independently confirmed attributes and visibility.'}
            if any(m.get('attribute_recheck') for m in manifest) else {})})}]
    views = {v.camera_name: v for v in packet.views}
    regions = {(r['object_id'], r.get('camera_name')): r
        for r in packet.provenance.get('scene_image_candidates', [])}
    anonymous_ids = {m['object_id']: m['candidate_id'] for m in packet.provenance.get('appearance_candidates', [])}
    strict = checks_rgb_visibility(packet)
    if strict:
        cameras = {m.get('camera_name') for m in manifest}
        for view in packet.views:
            if view.camera_name not in cameras or 'scene' not in view.artifacts:
                continue
            cameras.remove(view.camera_name)
            name = view.artifacts['scene']
            image = checked_asset(evidence_root/packet.packet_id, name, packet.provenance['artifact_sha256'][name]).read_bytes()
            # Share the original camera scene once, without marking a preferred
            # target. Derived crop edges cannot establish camera truncation.
            content.extend([{'type': 'text', 'text': f'Original full camera scene: {view.camera_label} '
                f'(camera {view.camera_name}), {view.width}x{view.height} original pixels. '
                'Use this scene to check the camera boundary for the anonymous units below. '
                'Each unit is identified by its own marked context and original-pixel rectangle.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(image).decode()}}])
    for item in manifest:
        names = candidate_artifacts(packet, item)
        name = names[0]
        image = checked_asset(evidence_root/packet.packet_id, name, packet.provenance['artifact_sha256'][name]).read_bytes()
        boundary = ''
        view = views.get(item.get('camera_name'))
        rect = regions.get((item['object_id'], item.get('camera_name')), {}).get('xyxy')
        if strict and view is not None and rect is not None:
            x1, y1, x2, y2 = rect
            touches = x1 <= 0 or y1 <= 0 or x2 >= view.width or y2 >= view.height
            boundary = (f' Camera {view.camera_label} ({view.camera_name}), original image {view.width}x{view.height}; '
                f'selected annotation unit original-pixel rectangle: {rect}; '
                f'marked unit region touches original camera boundary: {touches}. '
                'These bounds locate the unit; they do not certify that its RGB surfaces are visible.')
        foreground = []
        for oid in item.get('potential_foreground_object_ids', []):
            region = regions.get((oid, item.get('camera_name')))
            if region is not None:
                foreground.append({'candidate_id': anonymous_ids.get(oid), 'category': region['category'],
                    'xyxy': region['xyxy']})
        if foreground:
            boundary += (' Other native units whose projections overlap and may be in front: '
                + json.dumps(foreground) + '. These are geometric risks, not proof of occlusion. '
                'Use RGB to identify this selected unit separately from those foreground units; '
                'never report their paint or type as this unit. If this unit cannot be recognized, '
                'return uncertain; use occluded only when RGB confirms it is fully hidden.')
        content.extend([{'type': 'text', 'text': f"Candidate {item['candidate_id']}; category: {item['category']}; identity source: {item.get('identity_source', 'native_camera_box')}. A bounding rectangle locates the unit, not an instance mask: do not describe an occluder or neighbor. The colored rectangle, if present, marks the original region inside a larger context." + boundary},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(image).decode()}}])
        if len(names) > 1:
            detail = names[1]
            from app.agent.harness import observation_image_input
            data = checked_asset(evidence_root/packet.packet_id, detail, packet.provenance['artifact_sha256'][detail]).read_bytes()
            data, mime, note = observation_image_input(data)
            content.extend([{'type': 'text', 'text': f"Tight detail of the SAME candidate {item['candidate_id']}, camera {item['camera_name']}; observe only this unit. The full scene and larger context are not this candidate. Do not borrow another excavator or the rest of an assembly from outside this detail." + note},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,'+base64.b64encode(data).decode()}}])
    uncertain = ('If conflicting, unclear or multiple objects make identity uncertain, target_visible=false and omit uncertain attributes. '
        if allows_partial(packet) else 'If occluded, conflicting, unclear or multiple objects make identity uncertain, target_visible=false and omit uncertain attributes. ')
    return [{'role': 'system', 'content': OBSERVATION_SCOPE_GUIDE + visibility_instruction(packet) + 'Observe every anonymous object crop independently. No target or expression is supplied. '
             'Return JSON with one candidate record per supplied ID. Report only clearly visible color, subtype, shape, '
             'clothing or distinctive features using the schema. ' + uncertain + 'Never infer an attribute from another '
             'candidate. Color means the object surface: body paint for vehicles or the visible panel surface for signs, '
             'not lighting. Color values belong to color or clothing-color kinds, never feature. '
             'For every vehicle, report vehicle_colors with separate body and roof values (null if uncertain). '
             'When body paint is clear, also include that SAME body color in a/attributes as k/kind=color; '
             'both fields are required to confirm it. Never put only the roof color in attributes. '
             'Generic color attributes describe dominant body paint only, never roof, windows or lights. '
             'If roof paint differs from body paint, keep that roof value only under vehicle_colors.roof. '
             'Use vehicle_colors=null or omit it for non-vehicles. '
             'Missing attributes mean unknown, never absent. '
             'Use absent_features ONLY when the entire relevant object part is clearly visible and the feature is '
             'clearly absent (e.g. a fully visible bare roof has no roof rack). Otherwise leave absent_features empty. '
             'For visibility, clear means the actual object is clearly recognizable, unoccluded and not cut off by the image boundary. '
             'Use partial for a cut-off or partly hidden object, occluded for a fully hidden object, uncertain when unsure. '
             'Use the short schema keys: c=candidates, id=candidate_id, seen=target_visible, vis=visibility, '
             'why=visibility_reason, a=attributes, k=kind, v=value, absent=absent_features. '
             'Keep each why to one specific phrase of at most 12 words based on original RGB pixels. '
             'Rectangles and LiDAR dots are not visible object surfaces.'},
            {'role': 'user', 'content': content}]


def checked_observations(packet, observation, manifest=None):
    sources = {m['candidate_id']: m for m in
        (packet.provenance.get('appearance_candidates', []) if manifest is None else manifest)}
    manifest = {cid: m['object_id'] for cid,m in sources.items()}
    records, seen = {}, set()
    for item in observation.candidates:
        if item.candidate_id not in manifest or item.candidate_id in seen:
            raise ValueError('invalid_or_duplicate_visual_candidate_id')
        seen.add(item.candidate_id)
        record = item.model_dump(exclude={'candidate_id'})
        if checks_rgb_visibility(packet):
            # Bind the observation to its actual camera, not an arbitrary view.
            source = sources[item.candidate_id]
            record['observed_camera_name'] = source.get('camera_name')
            record['observed_input_sha256'] = candidate_input_digest(packet, source)
            lights = record.get('visible_lights') or {}
            region = next((c for c in packet.provenance.get('scene_image_candidates', [])
                if c['object_id'] == manifest[item.candidate_id]
                and c.get('camera_name') == source.get('camera_name') and c.get('xyxy')), None)
            view = next((v for v in packet.views if v.camera_name == source.get('camera_name')), None)
            conflict = any({'kind':'feature','value':'illuminated '+name} in record['attributes']
                and lights.get(name) == 'absent' for name in ('headlights','taillights'))
            if (packet.task.get('source_light_policy') == 'visible-region-v1'
                    and source['category'] == 'vehicle' and region and view
                    and 'scene' in view.artifacts and lights.get('complete_region_inspected') is True
                    and lights.get('reason', '').strip() and record['observed_input_sha256'] and not conflict):
                # Derived receipt, never a provider-supplied identity/absence.
                # Consumers may use it only in this explicitly named camera.
                record['visible_light_region'] = {'policy':'visible-region-v1',
                    'camera_name':view.camera_name,'xyxy':region['xyxy'],
                    'input_sha256':record['observed_input_sha256']}
        records[manifest[item.candidate_id]] = record
    return records


def useful_landmark_ids(packet, blocking):
    """Rank a few geometric landmarks for observation, never for admission.

    Visibility below is hypothetical: it only asks which landmark would remove
    competitors *if* RGB later confirms it. No synthetic record is returned or
    stored, and every real visibility/uniqueness check still runs afterwards.
    """
    from app.quality.relations import relation_catalog

    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    hypothetical = {m['object_id']: {'target_visible': True, 'visibility':'clear', 'visibility_reason':'Hypothetical planning only; requires actual RGB observation',
        'attributes': [], 'observed_camera_name':camera}
        for m in packet.provenance.get('appearance_candidates', [])}
    relations, anchors = relation_catalog(packet, hypothetical)
    priorities = {}
    for predicate in relations:
        if predicate.kind not in {'relative_left', 'relative_right', 'ahead_of', 'behind_anchor', 'vehicle_heading'}:
            continue
        if predicate.matches.get(packet.target_object_id) is not True:
            continue
        removed = sum(predicate.matches.get(oid) is False for oid in blocking)
        if removed:
            priorities[predicate.value] = max(removed, priorities.get(predicate.value, 0))
    useful = [a for a in anchors if a['anchor_id'] in priorities
        and a['object_id'] != packet.target_object_id and a['object_id'] not in blocking]
    useful.sort(key=lambda a: (-priorities[a['anchor_id']], len(a['descriptor_predicates']), a['object_id']))
    chosen = [a['object_id'] for a in useful[:4]]
    # A red car cannot be known to be a useful landmark until its appearance is
    # observed. Add a few prominent source-view units even when coarse geometry
    # alone cannot yet name them. This only schedules perception, never acceptance.
    source_boxes = {c['object_id']:c for c in packet.provenance.get('scene_image_candidates', [])
        if c.get('camera_name') == camera and c.get('xyxy')}
    from app.quality.description_search import scene_reference_policy
    if scene_reference_policy(packet):
        chosen = [oid for oid in chosen if oid in source_boxes and source_boxes[oid]['category'] != 'unknown']
    target = packet.target
    native = {o.object_id:o for o in packet.candidates}
    prospective = []
    for oid, box in source_boxes.items():
        if oid == packet.target_object_id or oid in chosen or oid in blocking:
            continue
        if scene_reference_policy(packet) and box['category'] == 'unknown':
            continue  # Still a rival; cannot yet be named as a landmark.
        x1,y1,x2,y2 = box['xyxy']
        if min(x2-x1,y2-y1) < 12:
            continue
        anchor = native.get(oid)
        separates = 0
        if anchor is not None:
            for rival_id in blocking - {packet.target_object_id}:
                rival = native.get(rival_id)
                if rival is not None:
                    separates += any((target.center_m[axis]-anchor.center_m[axis]) *
                        (rival.center_m[axis]-anchor.center_m[axis]) <= 0 for axis in (0,1))
            from app.quality.relations import VEHICLE_HEADING_POLICY, vehicle_heading_match
            if packet.task.get('vehicle_heading_policy') == VEHICLE_HEADING_POLICY:
                separates += sum(vehicle_heading_match(target,anchor,sign) is True
                    and vehicle_heading_match(native[rival],anchor,sign) is False
                    for sign in (1,-1) for rival in blocking if rival in native and rival != target.object_id)
        prospective.append((-separates, -(x2-x1)*(y2-y1), oid))
    per_category = {}
    for oid in chosen:
        category = source_boxes.get(oid, {}).get('category')
        per_category[category] = per_category.get(category, 0) + 1
    for _, _, oid in sorted(prospective):
        if len(chosen) == 4:
            break
        category = source_boxes[oid]['category']
        if scene_reference_policy(packet) and per_category.get(category, 0) >= 2:
            continue
        chosen.append(oid)
        per_category[category] = per_category.get(category, 0) + 1
    return set(chosen)


def source_identity_candidate(packet):
    """Bind the anonymous target ID to marked context in the actual source view."""
    item = next((m for m in packet.provenance.get('appearance_candidates', [])
        if m['object_id'] == packet.target_object_id), None)
    view = next((v for v in packet.views if v.camera_label == packet.source_view), None)
    if item is None or view is None:
        return None
    selected = next((m for m in [item, *item.get('alternatives', [])]
        if m.get('camera_name') == view.camera_name), None)
    # Alternative-camera entries may contain only a tight crop. Always retain
    # the original camera boundary and the marked unit for identity assessment.
    context = view.artifacts.get('appearance_context', view.artifacts['appearance'])
    return {**(selected or {'candidate_id': item['candidate_id'], 'object_id': item['object_id'],
            'category': item['category']}),
        'camera_name': view.camera_name, 'camera_object_id': view.appearance_camera_object_id,
        'identity_source': view.appearance_identity_source,
        'artifact': view.artifacts['appearance'], 'visibility_artifact': context,
        'detail_artifact': view.artifacts['appearance']}


def source_attribute_conflicts(packet, observation, observations):
    """Find contradictory source-camera facts before spending on more crops."""
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    other = (observations or {}).get(packet.target_object_id, {})
    other = other.get('camera_observations', {}).get(str(camera), other)
    if (other.get('observed_camera_name') != camera
            or not usable_rgb_observation(other, allow_partial=allows_partial(packet))):
        return []
    primary = observation_attributes(observation.model_dump(), allow_partial=allows_partial(packet))
    secondary = observation_attributes(other, allow_partial=allows_partial(packet))
    conflicts = []
    for kind in {a['kind'] for a in primary} - {'feature'}:
        values = {a['value'] for a in primary + secondary if a['kind'] == kind}
        if not compatible_attribute_values(kind, values):
            conflicts.append(kind)
    return sorted(conflicts)


def source_attribute_gaps(packet, observation, observations):
    """Plan a fresh source check when anonymous perception found omitted cues.

    These are acquisition hints only. Never promote a reason string, another
    camera's pixels or a scene-only attribute into verified target evidence.
    """
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    other = (observations or {}).get(packet.target_object_id, {})
    other = other.get('camera_observations', {}).get(str(camera), other)
    if (other.get('observed_camera_name') != camera
            or not usable_rgb_observation(other, allow_partial=allows_partial(packet))
            or not usable_rgb_observation(observation.model_dump(), allow_partial=allows_partial(packet))):
        return []
    primary = observation_attributes(observation.model_dump(), allow_partial=allows_partial(packet))
    kinds = {a['kind'] for a in primary}
    return [a for a in observation_attributes(other, allow_partial=allows_partial(packet))
        if a not in primary and (a['kind'] == 'feature' or a['kind'] not in kinds)
        and not (a['kind'] == 'feature' and a['value'] in other.get('absent_features', []))]


def landmark_observation_blockers(packet, observations, target_rivals, *, with_requirements=False):
    """Plan missing landmark comparisons, without certifying any landmark.

    A white vehicle may already be excluded as a gray target, yet its unknown
    lookalikes still need observation before it can name a spatial reference.
    Center-side tests rank acquisition only; final relations keep all guards.
    """
    from app.quality.description_search import camera_observations, confirmed_attributes, scene_first_policy
    if not scene_first_policy(packet):
        return {} if with_requirements else set()
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    records = camera_observations(packet, camera, observations or {})
    native = {o.object_id:o for o in packet.candidates}
    inventory = [b for b in packet.provenance.get('scene_image_candidates', []) if b.get('camera_name') == camera]
    areas = {b['object_id']:(b['xyxy'][2]-b['xyxy'][0])*(b['xyxy'][3]-b['xyxy'][1])
        for b in inventory if b.get('xyxy')}
    regions = {b['object_id']:b['xyxy'] for b in inventory if b.get('xyxy')}
    prospective = []
    for oid, record in records.items():
        anchor = native.get(oid)
        if anchor is None or oid == packet.target_object_id:
            continue
        attrs = confirmed_attributes(anchor.category, record)
        if not attrs or not usable_rgb_observation(record, allow_partial=allows_partial(packet)):
            continue
        gain = sum(any((packet.target.center_m[axis]-anchor.center_m[axis]) *
            (native[rival].center_m[axis]-anchor.center_m[axis]) <= 0 for axis in (0,1))
            for rival in target_rivals if rival in native and rival != packet.target_object_id)
        from app.quality.relations import VEHICLE_HEADING_POLICY, vehicle_heading_match, vehicle_heading_anchor
        heading_gain = 0
        if packet.task.get('vehicle_heading_policy') == VEHICLE_HEADING_POLICY:
            box = next((b for b in inventory if b['object_id'] == oid),None)
            if vehicle_heading_anchor(packet,anchor,record,box):
                heading_gain = sum(vehicle_heading_match(packet.target,anchor,sign) is True
                    and vehicle_heading_match(native[rival],anchor,sign) is False
                    for sign in (1,-1) for rival in target_rivals if rival in native and rival != packet.target_object_id)
            gain += heading_gain
        # Unlinked RGB rivals also motivate a landmark comparison. This is
        # acquisition ranking only; a center crossing never proves a relation.
        target_rect, anchor_rect = regions.get(packet.target_object_id), regions.get(oid)
        if target_rect and anchor_rect:
            gain += sum((target_rect[0]+target_rect[2]-anchor_rect[0]-anchor_rect[2]) *
                (regions[rival][0]+regions[rival][2]-anchor_rect[0]-anchor_rect[2]) <= 0
                for rival in target_rivals if rival not in native and rival in regions)
        if not gain:
            continue
        pool = list(dict.fromkeys(b['object_id'] for b in inventory if b['category'] in {anchor.category,'unknown'}))
        possible = filter_visual_candidates(pool, attrs, records)[0]
        missing = {}
        for rival in possible:
            if rival == oid:
                continue
            record = records.get(rival,{})
            known = {a['kind'] for a in record.get('attributes',[])}
            kinds = {a.kind for a in attrs if a.kind not in known
                or not usable_rgb_observation(record,allow_partial=allows_partial(packet))}
            if kinds:
                missing[rival] = kinds
        if missing:
            # A target rival can still be an independently named landmark:
            # the relation excludes the landmark itself. Favor verified heading
            # utility over generic center-side hypotheses when acquiring proof.
            prospective.append((-int(heading_gain > 0),-gain,-areas.get(oid,0),oid,missing))
    requirements = {}
    for row in sorted(prospective)[:4]:
        # Preserve the useful anchor's priority and resolve almost-described
        # lookalikes before completely unknown tiny crops in its same pool.
        for rival, kinds in sorted(row[-1].items(),key=lambda item:(len(item[1]),-areas.get(item[0],0),item[0])):
            requirements.setdefault(rival,set()).update(kinds)
    return {oid:sorted(kinds) for oid,kinds in requirements.items()} if with_requirements else set(requirements)


def precheck_observation_batches(packet, batch_size=16):
    """Cover every relevant crop, without sending irrelevant classes/directions.

    This is a target-conditioned comparison pool, not an independent blind eval.
    Missing crops remain unknown in admission; they are never silently excluded.
    """
    if packet.task.get('annotation_search'):
        from app.quality.contrast import remaining
        native = {o.object_id: o for o in packet.candidates}
        base = [p for p in packet.predicates if (p.kind == 'category' and p.value == packet.target.category
            or p.kind in {'longitudinal', 'lateral'}) and p.matches.get(packet.target_object_id) is True]
        blocking = set(remaining(packet.candidates, base))
        blocking.update(c['object_id'] for c in packet.provenance.get('scene_image_candidates', [])
            if c['category'] in {packet.target.category, 'unknown'} and c['object_id'] not in native)
        landmarks = useful_landmark_ids(packet, blocking)
        source_camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
        inventory = packet.provenance.get('scene_image_candidates', [])
        from app.quality.description_search import scene_reference_policy, minimal_reference_policy
        if ((not scene_reference_policy(packet) or minimal_reference_policy(packet)) and checks_rgb_visibility(packet)
                and inventory and all(c.get('xyxy') and 'camera_name' in c for c in inventory)):
            # Try a source-camera-qualified sentence first. Its visibility
            # predicate can exclude objects outside this complete camera
            # inventory; observing every unlinked car in other cameras first
            # cannot help this proof. The full competition pool is unchanged.
            in_camera = {c['object_id'] for c in inventory if c['camera_name'] == source_camera
                and c['category'] in {packet.target.category, 'unknown'}}
            blocking &= in_camera
        relevant = blocking | landmarks | {packet.target_object_id}
        manifest = [m for m in packet.provenance.get('appearance_candidates', [])
            if m['object_id'] in relevant]
        source_bound = []
        for item in manifest:
            choice = next((m for m in [item, *item.get('alternatives', [])]
                if m.get('camera_name') == source_camera), None)
            source_bound.append({**choice, 'candidate_id':item['candidate_id'], 'object_id':item['object_id'],
                'category':item['category']} if choice else item)
        manifest = source_bound
        if allows_partial(packet):
            # Anonymous identity checks must use the actual source camera,
            # including after fallback changes that camera.
            selected = source_identity_candidate(packet)
            if selected is not None:
                manifest = [selected if m['object_id'] == packet.target_object_id else m for m in manifest]
        nearby = set()
        if minimal_reference_policy(packet):
            source_boxes = [b for b in inventory if b.get('camera_name') == source_camera and b.get('xyxy')]
            target_box = next((b for b in source_boxes if b['object_id'] == packet.target_object_id), None)
            view = next((v for v in packet.views if v.camera_name == source_camera), None)
            if target_box and view:
                rect = target_box['xyxy']
                edges = [(axis,extent) for axis,extent in ((0,0),(1,0),(2,view.width),(3,view.height))
                    if abs(rect[axis]-extent)<=1]
                for box in source_boxes:
                    other=box['xyxy']
                    if box['category'] in {packet.target.category,'unknown'} and (
                            min(rect[2],other[2])>max(rect[0],other[0]) and min(rect[3],other[3])>max(rect[1],other[1])
                            or any(abs(other[axis]-extent)<=8 for axis,extent in edges)):
                        nearby.add(box['object_id'])
        # These regions can be an occluder, an unresolved duplicate or a real
        # neighbor. Observe them early; overlap/boundary contact proves no ID,
        # visibility or uniqueness and never deletes the remaining schedule.
        manifest = sorted(manifest, key=lambda m: (
            m['object_id'] != packet.target_object_id,
            m['object_id'] not in nearby,
            m['object_id'] not in landmarks if scene_reference_policy(packet) else False,
            m.get('camera_name') != source_camera if packet.task.get('landmark_description_policy')=='spatially-qualified-v1' else False,
            m['object_id'] not in blocking,
            m['object_id'] not in landmarks,
            m.get('camera_name') != source_camera,
            m['category'] != packet.target.category,
            m['object_id'] not in native, m['candidate_id']))
        return [manifest[start:start+batch_size] for start in range(0, len(manifest), batch_size)]
    from app.quality.contrast import remaining
    from app.quality.relations import expand_relations
    predicates = [p for p in packet.predicates if
        (p.kind == 'category' and p.value == packet.target.category or p.kind in {'longitudinal','lateral'})
        and p.matches.get(packet.target_object_id) is True]
    relevant = set(remaining(packet.candidates, predicates))
    relevant.update(c['object_id'] for c in packet.provenance.get('scene_image_candidates', [])
        if c['category'] in {packet.target.category, 'unknown'})
    _, result = expand_relations(packet)
    relevant.update(a['object_id'] for a in result['anchors'])
    manifest = [m for m in packet.provenance.get('appearance_candidates', []) if m['object_id'] in relevant]
    return [manifest[start:start+batch_size] for start in range(0,len(manifest),batch_size)]


def identity_first_batches(packet, batches):
    """Split out the source identity check; retain every other scheduled crop.

    This only changes acquisition order. Unobserved rivals remain unknown and
    admission must still prove uniqueness from actual returned observations.
    """
    if not batches or len(batches[0]) < 2 or batches[0][0]['object_id'] != packet.target_object_id:
        return batches
    return [[batches[0][0]], batches[0][1:], *batches[1:]]


def merge_observations(previous, current, *, allow_partial=False):
    result = _merge_observations(previous, current, allow_partial=allow_partial)
    if not allow_partial:
        return result
    # Global appearance and camera-specific visibility are different evidence.
    # A relook from another camera must not erase the original camera record.
    cameras = dict((previous or {}).get('camera_observations', {}))
    for record in (previous, current):
        if not record or record.get('observed_camera_name') is None:
            continue
        camera = str(record['observed_camera_name'])
        incoming = {k: v for k, v in record.items() if k != 'camera_observations'}
        older = cameras.get(camera)
        old_evidence = visibility_evidence(older or {}, allow_partial=True)
        new_evidence = visibility_evidence(incoming, allow_partial=True)
        conflict = old_evidence is not None and new_evidence is not None and old_evidence != new_evidence
        if conflict or (older or {}).get('visibility_conflict') or incoming.get('visibility_conflict'):
            incoming = {**incoming, 'target_visible': False, 'visibility': 'uncertain',
                'visibility_reason': 'Conflicting observations in this camera.', 'visibility_conflict': True,
                'attributes': [], 'absent_features': []}
        elif older and new_evidence is None:
            incoming = older
        elif older and old_evidence is not None:
            incoming = _merge_observations(older, incoming, allow_partial=True)
        cameras[camera] = incoming
    if cameras:
        confirmed = [r for r in cameras.values() if usable_rgb_observation(r, allow_partial=True)]
        if confirmed:
            result = confirmed[0]
            for record in confirmed[1:]:
                result = _merge_observations(result, record, allow_partial=True)
        else:
            result = {**result, 'target_visible': False, 'attributes': [], 'absent_features': []}
            if any(r.get('visibility_conflict') for r in cameras.values()):
                result.update(visibility='uncertain', visibility_reason='Conflicting observations in this camera.')
    return {**result, 'camera_observations': cameras}


def _merge_observations(previous, current, *, allow_partial=False):
    """Keep conflicting visible values as unknown, rather than overwriting them."""
    if not previous or not previous.get('target_visible'):
        return current
    if not current.get('target_visible'):
        return previous
    strict = 'observed_camera_name' in previous or 'observed_camera_name' in current
    if strict:
        if not usable_rgb_observation(previous, allow_partial=allow_partial):
            return current
        if not usable_rgb_observation(current, allow_partial=allow_partial):
            return previous
    attrs = {(a['kind'], a['value']) for r in (previous, current)
        for a in observation_attributes(r, allow_partial=allow_partial)}
    result = {'target_visible': True, 'attributes': [{'kind': k, 'value': v} for k, v in sorted(attrs)],
            'absent_features': sorted(set(previous.get('absent_features', [])) | set(current.get('absent_features', [])))}
    if strict:
        visibility = 'partial' if allow_partial and 'partial' in {previous.get('visibility'), current.get('visibility')} else 'clear'
        result.update(visibility=visibility, visibility_reason=(previous['visibility_reason']+' '+current['visibility_reason'])[:300],
            observed_camera_name=previous.get('observed_camera_name') if previous.get('observed_camera_name') == current.get('observed_camera_name') else None)
        if visibility == 'partial':
            result['absent_features'] = []
    return result


def compatible_attribute_values(kind, values):
    """Broad nouns do not exclude their parts/subtypes; distinct parts do compete."""
    return len(values) <= 1 or (kind == 'subtype' and any(values <= family for family in (
        {'truck', 'pickup truck'}, {'bus','bus section'}, {'excavator','excavator boom'}, {'excavator','excavator body'},
        {'a-frame sign', 'construction barricade'})))


def resolve_target_observation(observation: VisualObservation, scene_observation: dict | None,
                               *, allow_partial=False) -> VisualObservation:
    """Keep primary attributes unless visible evidence explicitly contradicts them.

    Missing attributes and invisible crops are unknown, not counter-evidence.
    Never modify either raw observation or add scene-only attributes.
    """
    primary = [VisualAttribute.model_validate(a) for a in observation_attributes(observation.model_dump(), allow_partial=allow_partial)]
    grouped = {}
    for attr in primary:
        grouped.setdefault(attr.kind, set()).add(attr.value)
    other = scene_observation or {}
    if other.get('target_visible'):
        for attr in observation_attributes(other, allow_partial=allow_partial):
            if attr['kind'] in grouped and attr['kind'] != 'feature':
                grouped[attr['kind']].add(attr['value'])
    absent = set(other.get('absent_features', [])) if other.get('target_visible') else set()
    attributes = [a for a in primary if
        (a.kind == 'feature' and a.value not in absent)
        or (a.kind != 'feature' and compatible_attribute_values(a.kind, grouped[a.kind]))]
    if (allow_partial and other.get('target_visible')
            and usable_rgb_observation(other, allow_partial=True)):
        # This structural subtype has repeatedly been hallucinated for ordinary
        # post-mounted sign backs. An anonymous view of the same visible unit
        # must corroborate it before it can define a comparison group. Missing
        # support means unknown, never "not an A-frame" or an invisible target.
        corroborated = observation_attributes(other, allow_partial=True)
        attributes = [a for a in attributes if a.kind != 'subtype'
            or a.value not in {'a-frame sign', 'construction barricade'}
            or a.model_dump() in corroborated]
    return observation.model_copy(update={'attributes': attributes, 'vehicle_colors': None})


def filter_visual_candidates(candidate_ids, attributes, observations, *, visible_camera=None):
    # Only mutually exclusive single-valued properties support negative
    # elimination. Omitted features (e.g. roof rack) never mean absent.
    exclusive = {'color', 'subtype', 'shape', 'upper_clothing_color', 'lower_clothing_color', 'helmet_color'}
    grouped = {}
    for attribute in attributes:
        grouped.setdefault(attribute.kind, set()).add(attribute.value)
    survivors, excluded = [], {}
    for oid in candidate_ids:
        record = observations.get(oid, {})
        known = {}
        if record.get('target_visible'):
            for attribute in observation_attributes(record):
                known.setdefault(attribute['kind'], set()).add(attribute['value'])
        contradictions = []
        for kind, claimed in grouped.items():
            observed = known.get(kind, set())
            if kind in exclusive and len(claimed) == len(observed) == 1 and claimed != observed:
                if compatible_attribute_values(kind, claimed | observed):
                    continue
                # Gray/silver are not a safe exclusion under outdoor lighting.
                if kind in {'color', 'helmet_color'} and claimed | observed <= {'gray', 'silver'}:
                    continue
                contradictions.append(kind)
        if record.get('target_visible'):
            explicit_absent = set(record.get('absent_features', []))
            positives = known.get('feature', set())
            contradictions.extend('feature:'+feature for feature in sorted(grouped.get('feature', set()) & (explicit_absent - positives)))
        # Unknown body identity is not negative appearance evidence. A separate
        # complete-region light check can, however, contradict a specifically
        # CAMERA-VISIBLE light clause. Keep every unchecked/uncertain rival and
        # never use this receipt for unqualified native/ego descriptions.
        receipt = record.get('visible_light_region') or {}
        lights = record.get('visible_lights') or {}
        if (visible_camera is not None and receipt.get('policy') == 'visible-region-v1'
                and record.get('observed_camera_name') == receipt.get('camera_name') == visible_camera
                and receipt.get('input_sha256') == record.get('observed_input_sha256')
                and receipt.get('input_sha256') and not record.get('visibility_conflict')
                and lights.get('complete_region_inspected') is True and lights.get('reason', '').strip()):
            for name in ('headlights','taillights'):
                feature = 'illuminated '+name
                if feature in grouped.get('feature',set()) and lights.get(name) == 'absent':
                    contradictions.append('camera_visible_feature:'+feature)
        if contradictions:
            excluded[oid] = contradictions
        else:
            survivors.append(oid)
    return survivors, excluded


FEEDBACK_ACQUISITION_POLICY = 'feedback-landmark-recheck-v1'


def feedback_landmark_manifest(packet):
    """Acquire requested attribute KINDS before abandoning the reviewed view.

    Geometry ranks anonymous candidates; neither desired values nor feedback
    text enter the observer prompt. Unknown image candidates remain competitors.
    """
    if packet.task.get('feedback_acquisition_policy') != FEEDBACK_ACQUISITION_POLICY:
        return []
    import math
    from app.quality.joint_reference import feedback_relations
    from app.quality.relations import vehicle_forward_region_match, vehicle_heading_match, local_landmark_pair
    requests = [r for r in feedback_relations(packet.task) if not r['denied'] and r['attributes']
        and r['relation'] in {'vehicle_behind','vehicle_ahead','vehicle_directly_behind','vehicle_directly_ahead'}]
    if not requests:
        return []
    camera = next((v.camera_name for v in packet.views if v.camera_label == packet.source_view), None)
    native = {o.object_id:o for o in packet.candidates}
    ranked = []
    for item in packet.provenance.get('appearance_candidates', []):
        anchor = native.get(item['object_id'])
        if anchor is None or anchor.object_id == packet.target_object_id:
            continue
        if not local_landmark_pair(packet,packet.target,anchor):
            continue  # Still a scene competitor; not a requested landmark recheck.
        source = next((m for m in [item,*item.get('alternatives',[])] if m.get('camera_name') == camera),None)
        if source is None:
            continue
        kinds = set(); ranks = []
        for request in requests:
            if anchor.category != request['category']:
                continue
            sign = -1 if request['relation'].endswith('behind') else 1
            if vehicle_forward_region_match(packet.target, anchor, sign) is not True:
                continue
            kinds.update(a['kind'] for a in request['attributes'])
            ranks.append(vehicle_heading_match(packet.target, anchor, sign) is not True)
        if kinds:
            distance = math.hypot(*(packet.target.center_m[i]-anchor.center_m[i] for i in (0,1)))
            ranked.append(((min(ranks),distance,anchor.object_id), {**source,'attribute_recheck':sorted(kinds)}))
    # One bounded call; preserve budget for writing and two independent checks.
    return [item for _,item in sorted(ranked,key=lambda row:row[0])[:3]]
