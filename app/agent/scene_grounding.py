"""Blind whole-scene resolution of descriptions too coarse for symbolic cues."""
import base64
import json
import random
import io
import math
import hashlib
from PIL import Image, ImageDraw, ImageFont

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.datasets.evidence import EVIDENCE_ROOT
from app.infrastructure.artifacts import checked_asset

POLICY = 'two-round-blind-scene-v1'
CLAIM_POLICY = 'explicit-claims-only-v1'
BINDING_POLICY = 'anonymous-region-binding-v1'
INPUT_POLICY = 'sentence-scoped-scene-input-v1'
CLAIM_GUIDE = """Evaluate only claims actually made by the sentence, not a required description template.
Appearance alone can uniquely identify an object. No spatial relation, landmark, ego direction, or
relative position is required when the sentence does not assert one. 'Visible in the front view'
specifies the source camera; it does NOT mean 'in front of another object' or require a landmark.
Inspect the entire referring phrase, including visible materials, body parts, attachments, and patterns;
do not reduce it to only color and category or restrict it to a predefined attribute vocabulary.
Verify each such detail directly in the photograph; wording or human preference is not evidence.
For an appearance-only sentence, return relation_pairs=[] and select the uniquely matching candidate
when supported. Missing UNMENTIONED relations are never unsupported_claims or reasons for ambiguity.
Use the geometry graph only for spatial clauses actually present. If a sentence asserts a relation,
that clause still requires the specified native geometry and supporting pairs; never ignore it.
For an indefinite landmark, existence of a supported target-landmark pair suffices for that clause;
other landmarks failing it do not refute it. Compare ALL possible target objects for full-sentence uniqueness.
Distinguish a competing OBJECT from alternative landmarks for the same object. Report actual unresolved
matches, identity uncertainty, or unsupported stated details, not the absence of an optional cue.
Before returning, check that candidate_ids, ambiguous_candidate_ids, expression_supported and your
reason agree: identifying one matching object is not ambiguous merely because it lacks a landmark.
If a stated attribute is unverifiable or a real competitor remains, reject it normally."""
GUIDE = """Resolve the English description in this complete source photograph and anonymous native 3D scene.
You are NOT told the intended object. Treat all native and unmatched image candidates as competitors.
Read the natural sentence itself, including conjunctions and spatial modifiers such as 'directly in front'.
Use the photograph to judge appearance, occlusion, and whether an object actually fits the whole sentence;
use native geometry for ego-relative directions (x forward, y left, z up). Camera left/right is different.
Do not replace 'directly in front' with 'anywhere in the front half-space'. Do not infer motion from lights.
Choose ALL objects that fit, or none if there is no supported unique interpretation. A small/unknown crop
alone neither proves nor disproves a match. If a plausible competitor cannot be ruled out in the full scene,
include it in ambiguous_candidate_ids. Never ignore a zero-point/unknown object solely because it is unknown.
Native projected bounds may include hidden parts and are not visible silhouettes. Image-only objects have
no verified 3D identity: never silently merge them with a native candidate based on appearance or overlap.
Report unsupported/fabricated parts of the sentence. Set expression_supported=false if any essential
claim cannot be verified. Do not choose an object just because its box is large, its ID is first, or it is near.
Return JSON matching the schema. candidate_ids contains native or image IDs from this request only.
Your answer is an automated same-provider check, not an independent human quality certificate."""


NATIVE_SCENE_GUIDE = """Independently resolve the entire referring sentence among ALL anonymous scene candidates.
There is no designated target. Return the supplied JSON schema.
Use TWO distinct sources of evidence in this order:
1. PHOTOGRAPHS establish appearance and visible identity ONLY. Bind the subject and landmark descriptors
   to their OWN labeled regions, comparing all possible objects in the full original photo and every card.
   Crops may overlap or contain occluders: do not transfer one object's attributes to another ID.
   Partial truncation is not rejection by itself. Keep genuinely unresolved appearance/identity competitors.
2. TOOL RELATION TABLES establish ALL spatial assertions, including ordinary in front of/behind another
   vehicle, directly in front of/behind, and ego directions. Do NOT judge any spatial assertion by apparent
   photo position, lane impression, apparent size or mentally recomputing numeric coordinates.
   For each visually matching SUBJECT and LANDMARK, read the exact relation_key in sentence_relation_index.
   In by_subject, find subject_id and test landmark ID membership in true_landmark_ids or unknown_landmark_ids.
   Cross-check that landmark's row. Both tables are identical authoritative native-3D geometry.
   A true pair is spatially supported; a false pair is not. Unknown stays unknown, never false or true.
   A true relationship to one landmark does not rule out a true relationship to a different landmark.
Plain behind means vehicle_behind, not vehicle_directly_behind. Plain in front of means vehicle_ahead,
not vehicle_directly_ahead. Plain relations do not assert same lane, lateral alignment, adjacency or motion.
Do not add these conditions or deny a true table entry because the photo appears to place vehicles side by side.
Vehicle-heading relations follow the LANDMARK's heading; native_left/right/ahead/behind follow EGO axes.
For each subject apply the COMPLETE conjunction of its appearance and all stated native relations.
An indefinite landmark need not be independently unique; the FINAL SUBJECT must be unique.
Return every supporting target-landmark combination in relation_pairs with its exact relation_key.
Report all actual unresolved competing subjects in ambiguous_candidate_ids. Keep all native, image-only,
zero-point and unpictured inventory; an unverified image/native association cannot be guessed or merged.
If a relation is true but an attribute or identity cannot be verified, reject for THAT uncertainty;
do not describe the geometry as absent. Do not silently exclude unknown identity or geometry.
Appearance-only sentences require no relation or landmark. Named camera scopes appearance, not ego direction.
Reject unsupported actually stated motion, distance, identity or appearance. Never infer unspoken details.
Set expression_supported only after BOTH visual claims and authoritative spatial table claims have been checked.
The final candidate IDs, ambiguity IDs, supporting pairs and reason must agree. No expected answer is supplied.
This is an automated same-provider check, not independent research certification."""


class SceneRelationPair(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidate_id: str
    landmark_id: str
    relation_key: str | None = None


class SceneGroundResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expression_supported: StrictBool
    candidate_ids: list[str] = Field(max_length=1000)
    ambiguous_candidate_ids: list[str] = Field(max_length=1000)
    unsupported_claims: list[str] = Field(max_length=100)
    reason: str = Field(min_length=1, max_length=2000)
    relation_pairs: list[SceneRelationPair] = Field(default_factory=list,max_length=1000)


def available(packet):
    from app.datasets.correspondence import has_multimodal_evidence
    return bool(packet.task.get('scene_grounding_policy')==POLICY and has_multimodal_evidence(packet)
        and packet.provenance.get('scene_image_candidates')
        and any(v.camera_label==packet.source_view and v.artifacts.get('scene') for v in packet.views))


def anonymous_image_guides(photo, rows, *, binding=False):
    """Every source box gets the same anonymous label and crop treatment.

    The unmodified full photograph is still supplied separately. These guides
    bind arbitrary IDs to pixels, never mark the intended target or infer a
    foreground identity from a projected bounding box.
    """
    original = Image.open(io.BytesIO(photo)).convert('RGB')
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=16)
    boxes = [(row['candidate_id'],index,box) for row in rows for index,box in enumerate(row['image_boxes_xyxy'])]
    if not boxes:
        return []
    width,height = original.size
    tile_w,tile_h,columns = (320,240,3) if binding else (240,166,5)
    # Bounded image dimensions prevent the provider from shrinking one tall
    # contact sheet until overlapping vehicles and labels become unreadable.
    page_size = 12 if binding else len(boxes)
    sheets = [Image.new('RGB',(columns*tile_w,math.ceil(min(page_size,len(boxes)-start)/columns)*tile_h),'#f4f4f4')
        for start in range(0,len(boxes),page_size)]
    for index,(cid,box_index,box) in enumerate(boxes):
        x1,y1,x2,y2 = box
        x1,y1=max(0,min(width,math.floor(x1))),max(0,min(height,math.floor(y1)))
        x2,y2=max(0,min(width,math.ceil(x2))),max(0,min(height,math.ceil(y2)))
        label = cid + (f' / {box_index+1}' if box_index else '')
        sheet=sheets[index//page_size]
        label_draw=ImageDraw.Draw(sheet)
        position=index%page_size
        left,top=(position%columns)*tile_w,(position//columns)*tile_h
        label_draw.text((left+5,top+3),label,font=font,fill='#111111')
        if x2<=x1 or y2<=y1:
            label_draw.text((left+5,top+35),'Region outside image',font=font,fill='#333333')
            continue
        draw.rectangle((x1,y1,x2,y2),outline='#00ffff',width=2)
        extent=draw.textbbox((0,0),label,font=font)
        label_w,label_h=extent[2]-extent[0]+6,extent[3]-extent[1]+6
        lx,ly=max(0,min(width-label_w,x1)),max(0,y1-label_h)
        draw.rectangle((lx,ly,lx+label_w,ly+label_h),fill='#102632')
        draw.text((lx+3,ly+1),label,font=font,fill='#ffffff')
        # Uniform small padding preserves local context. Projected boxes may
        # contain occluders, so tiles never stand in for verified identity.
        pad=max(2,round(max(x2-x1,y2-y1)*.08))
        cx,cy=max(0,x1-pad),max(0,y1-pad)
        crop=original.crop((cx,cy,min(width,x2+pad),min(height,y2+pad)))
        if binding:
            ImageDraw.Draw(crop).rectangle((x1-cx,y1-cy,x2-cx-1,y2-cy-1),outline='#00ffff',width=3)
        crop.thumbnail((tile_w-10,tile_h-30))
        sheet.paste(crop,(left+(tile_w-crop.width)//2,top+27))
    result=[]
    pages=[(f'all anonymous source-region tiles {i+1}/{len(sheets)}',sheet) for i,sheet in enumerate(sheets)] if binding else [('all anonymous source-region tiles',sheets[0])]
    for kind,img in [('anonymous full-scene regions',overlay),*pages]:
        buffer=io.BytesIO();img.save(buffer,format='JPEG',quality=90)
        data=buffer.getvalue()
        result.append((kind,data))
    return result


def messages(packet, expression, seed):
    """A fresh randomized ID space, raw full image, every native/image unit; no target hints."""
    view = next(v for v in packet.views if v.camera_label==packet.source_view and v.artifacts.get('scene'))
    name = view.artifacts['scene']
    digest = packet.provenance['artifact_sha256'][name]
    photo = checked_asset(EVIDENCE_ROOT/packet.packet_id,name,digest).read_bytes()
    from app.quality.joint_reference import POLICY as JOINT_POLICY
    joint = packet.task.get('joint_reference_policy')==JOINT_POLICY
    binding = joint and packet.task.get('scene_binding_policy')==BINDING_POLICY
    focused = joint and packet.task.get('scene_input_policy')==INPUT_POLICY
    claim_scoped = packet.task.get('scene_claim_policy') == CLAIM_POLICY
    with Image.open(io.BytesIO(photo)) as decoded:
        image_width,image_height = decoded.size
    inventory = [c for c in packet.provenance['scene_image_candidates'] if c['camera_name']==view.camera_name]
    native = {o.object_id:o for o in packet.candidates}
    ids = sorted(set(native) | {c['object_id'] for c in inventory})
    random.Random(seed).shuffle(ids)
    mapping = {f'C{i+1:03d}':oid for i,oid in enumerate(ids)}
    if focused:
        source_ids={c['object_id'] for c in inventory}
        # Both groups retain every candidate. Separate anonymous namespaces
        # prevent a boxless object from being mistaken for a numbered crop.
        visible=[oid for oid in ids if oid in source_ids]
        unpictured=[oid for oid in ids if oid not in source_ids]
        mapping={**{f'V{i+1:02d}':oid for i,oid in enumerate(visible)},
            **{f'N{i+1:03d}':oid for i,oid in enumerate(unpictured)}}
    rows = []
    for cid,oid in mapping.items():
        boxes = [c['xyxy'] for c in inventory if c['object_id']==oid]
        row = {'candidate_id':cid,'image_boxes_xyxy':boxes,'identity':'native_3d' if oid in native else 'unmatched_image'}
        if joint:
            row['has_source_image_region'] = bool(boxes)
            row['original_image_boxes_xyxy'] = boxes
            row['image_boxes_xyxy'] = [[x*image_width/view.width,y*image_height/view.height,
                right*image_width/view.width,bottom*image_height/view.height] for x,y,right,bottom in boxes]
            row['image_identity_sources'] = [c.get('identity_source') for c in inventory if c['object_id']==oid]
        if oid in native:
            obj = native[oid]
            row.update(category=obj.category,center_m=obj.center_m,size_m=obj.size_m,
                heading_rad=obj.heading_rad,geometry_valid=obj.geometry_valid,lidar_points=obj.lidar_points)
        else:
            row['category'] = next(c['category'] for c in inventory if c['object_id']==oid)
        rows.append(row)
    payload = {'expression':expression,'camera':view.camera_label,'image_box_coordinates':{
        'width':image_width if joint else view.width,'height':image_height if joint else view.height,
        'units':'provided image pixels' if joint else 'original source-image pixels'},
        'coordinate_frame':'ego vehicle: x forward, y left, z up', 'candidates':rows,
        'output_schema':SceneGroundResult.model_json_schema()}
    if claim_scoped:
        payload['claim_evaluation'] = {
            'policy': CLAIM_POLICY,
            'scope': 'Every claim in the sentence; no mandatory spatial or landmark clause.',
            'appearance': 'Verify the complete visible description, including distinctive details, in the photograph.',
            'camera': 'A named view limits the source image; it is not an object-to-object spatial claim.',
            'relations': 'Only asserted spatial clauses require graph support and relation_pairs.',
            'uniqueness': 'Exactly one matching object, with no unresolved competing object or identity.'}
    from app.quality.spatial_relations import enabled as spatial_enabled, native_only, scene_graph
    typed = joint and spatial_enabled(packet)
    if typed:
        payload['spatial_relations']=scene_graph(packet,view.camera_name,mapping)
        from app.quality.spatial_relations import READING_POLICY, LEGACY_READING_POLICY, sentence_relation_index
        if packet.task.get('scene_relation_reading_policy') in {READING_POLICY,LEGACY_READING_POLICY}:
            payload['sentence_relation_index']=sentence_relation_index(payload['spatial_relations'],expression,policy=packet.task['scene_relation_reading_policy'])
            if focused:
                keys={d['key'] for d in payload['sentence_relation_index']['definitions']}
                if keys:
                    # Select relation TYPES from the sentence only, never
                    # subjects or landmarks from generated claims/feedback.
                    graph=payload['spatial_relations']
                    graph['definitions']=[d for d in graph['definitions'] if d['key'] in keys]
                    graph['rows']=[r for r in graph['rows'] if r['relation_key'] in keys]
    elif joint:
        from app.quality.relations import vehicle_heading_match, vehicle_forward_region_match, VEHICLE_FORWARD_REGION_POLICY
        reverse = {oid:cid for cid,oid in mapping.items()}
        source_ids = sorted(set(native) & {c['object_id'] for c in inventory})
        graph = []
        for anchor in source_ids:
            if native[anchor].category != 'vehicle':
                continue
            row = {'landmark_candidate_id':reverse[anchor]}
            relations=[('front',1,vehicle_heading_match),('behind',-1,vehicle_heading_match)]
            if packet.task.get('vehicle_forward_region_policy')==VEHICLE_FORWARD_REGION_POLICY:
                relations.extend([('ahead',1,vehicle_forward_region_match),('rearward',-1,vehicle_forward_region_match)])
            for name,sign,matcher in relations:
                values = {oid:matcher(native[oid],native[anchor],sign) for oid in source_ids}
                row[name] = [reverse[oid] for oid,value in values.items() if value is True]
                row[name+'_unknown'] = [reverse[oid] for oid,value in values.items() if value is None]
            graph.append(row)
        # Scene-wide geometry, computed without target, sentence, colors or
        # generated declarations. Unlinked image objects stay unresolved.
        payload['native_vehicle_relations'] = {'checked_native_candidate_ids':[reverse[oid] for oid in source_ids],
            'landmarks':graph,'semantics':'front/behind follows landmark heading with native box separation, lateral alignment and height checks; '
                'ahead/rearward, when present, mean plain in front of/behind with longitudinal separation but NO direct lateral alignment. '
                'within checked native IDs an unlisted pair is false; each *_unknown list is unknown. '
                'Pairs involving unmatched image IDs or IDs outside the checked set are unknown, never false.'}
    mime = 'image/png' if photo.startswith(b'\x89PNG') else 'image/jpeg'
    extra = ('\nResolve attributes AND relation jointly. An indefinite landmark ("a red sedan") need not '
        'be unique by itself; enumerate ALL target-landmark pairs that fit the whole sentence. '
        'For front/behind a vehicle, heading_rad defines its front-face normal: rotate +x by that heading. '
        'Return supporting pairs in relation_pairs using this request candidate IDs. '
        'Use native_vehicle_relations for vehicle front/behind geometry; do NOT override native heading '
        'with apparent image orientation or mentally recompute numeric 3D coordinates. '
        'The graph includes every source native vehicle as landmark and every source native object as target, '
        'without color filtering or a designated target. Judge colors/types and actual image identity in the photograph; '
        'combine those judgments with the graph. Graph support alone does not establish appearance or uniqueness. '
        'A native candidate without a source image region cannot be assigned to a visible colored car '
        'by guessing from its ego coordinates. Match visible objects to the supplied source image regions. '
        'The original photo is followed by anonymous region labels and tiles for ALL source candidates. '
        'Use those labels to bind an ID to actual pixels, not a guessed coordinate location. '
        'All regions receive identical treatment; none is a target hint. A tile may contain an occluder: '
        'judge object identity in the full scene, retaining uncertainty for unlinked/projected regions. '
        'Different landmarks may support the SAME unique target. Any unresolved alternative target remains ambiguous. '
        'image_boxes_xyxy are scaled to the provided image; original_image_boxes_xyxy are archival coordinates only. '
        'Native boxes with official_camera_lidar_association or native_box_point_correspondence have an actual image identity link.' if joint else '')
    if typed:
        extra=extra.replace('Use native_vehicle_relations for vehicle front/behind geometry;',
            'Use spatial_relations for each typed relation;')
        extra+=('\nEvery returned relation pair MUST include relation_key from spatial_relations.definitions. '
            'Read each row as subject relation landmark. For plain in front of use vehicle_ahead ONLY, '
            'not vehicle_directly_ahead; a false directly-ahead row does not disprove plain in front of. '
            'For multiple clauses return support for EVERY clause for EACH selected subject; different '
            'clauses may have different landmarks. Camera above/below is not physical elevation. '
            'Never infer front/behind for a sign or pedestrian from vehicle heading conventions. '
            'No graph row supplies color, clothing or identity: verify them in the complete photo.')
        if native_only(packet):
            extra+=('\nFor this run ALL spatial claims must be supported by native 3D geometry, NEVER pixel layout. '
                'Pixels establish appearance and image identity only. A camera qualifier scopes visible attributes, '
                'not a 2D coordinate frame. native_left/right/ahead/behind follow EGO orientation, '
                'vehicle_ahead/behind follow LANDMARK vehicle heading. native_above/below require native vertical '
                'separation AND overlapping horizontal footprints; a distant object higher in the photo is not above. '
                'Unlinked image objects keep UNKNOWN native relationships. Do not certify them using image layout.')
    if claim_scoped:
        extra = extra.replace('Resolve attributes AND relation jointly.',
            'Only IF the sentence asserts a landmark relation, resolve attributes AND that relation jointly.')
    if 'sentence_relation_index' in payload:
        extra+=('\nUse sentence_relation_index to look up the exact asserted relation for EVERY plausible subject and landmark. '
            'Its explicit statements are derived from the same complete native graph, without a target hint. '
            'SUBJECT and LANDMARK are not interchangeable. A subject can be behind a landmark in EGO coordinates '
            'and directly ahead of that SAME landmark along its VEHICLE heading. Those facts do not contradict. '
            'Before claiming geometry is unsupported, check that exact relation_key and landmark_id row. '
            'Report genuine appearance or identity uncertainty separately; graph support alone never resolves it.')
    if payload.get('sentence_relation_index',{}).get('by_subject'):
        extra+=('\nFor plain behind use vehicle_behind, NOT vehicle_directly_behind. Plain behind does not assert '
            'direct lateral alignment, the same lane, adjacency or following behavior. Different lanes alone cannot '
            'refute this weaker relation. Use by_subject to check ALL candidate landmark IDs for the visible subject; '
            'one different supported landmark is not exclusive and does not disprove another supported pair. '
            'Cross-check the exact landmark row before reporting absent geometry. Still reject real appearance, '
            'identity, unknown-pair or uniqueness failures; do not infer which target is expected.')
    if binding:
        payload['region_identity_check'] = {
            'policy':BINDING_POLICY,
            'rule':'Bind each subject AND landmark to its own visible object before using its geometry row. '
                'Overlapping rectangles and crops can contain several objects. Attributes of a foreground occluder '
                'do not belong to the candidate whose region contains it. Compare all overlapping candidate cards against the full photo. '
                'The cyan outline is the projected region, not a segmentation mask. '
                'If the visible object cannot be bound confidently to an ID, report identity uncertainty; never guess.'}
    if focused:
        columns=['candidate_id','category','identity','center_m','size_m','heading_rad','geometry_valid','lidar_points']
        payload['candidates']=[row for row in rows if row['image_boxes_xyxy']]
        payload['unpictured_native_candidates']={'columns':columns,
            'rows':[[row.get(key) for key in columns] for row in rows if not row['image_boxes_xyxy']],
            'semantics':'Complete remaining native inventory, not excluded competitors. No source image identity is established '
                'for these IDs; do not assign them the appearance of a visible crop by guessing coordinates. '
                'Uncertain identity and relationships remain UNKNOWN, including zero-point objects.'}
        payload['scene_input_policy']=INPUT_POLICY
        extra+=('\nFirst bind the sentence subject and each landmark to the V labels printed on their OWN image cards. '
            'V labels are source-image candidates; N labels retain all other native objects, with no source-region identity. '
            'Read the candidate label again before using its graph row; never swap the subject and landmark IDs. '
            'For geometry use the supplied exact relation rows, not an inverse phrase or a different frame. '
            'The relation table contains ALL source candidates for the relation types named by the sentence, without attribute filtering. '
            'Native numeric data is archival scene context; do not override tool-computed relations by mental arithmetic. '
            'In your reason explicitly name the visible subject ID and landmark ID you actually used.')
    content = [
        {'type':'text','text':json.dumps(payload,ensure_ascii=False,allow_nan=False)},
        {'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(photo).decode()}}]
    guide_receipts=[]
    for kind,data in (anonymous_image_guides(photo,rows,binding=binding) if joint else []):
        content.extend([{'type':'text','text':kind},
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}}])
        guide_receipts.append({'kind':kind,'sha256':hashlib.sha256(data).hexdigest()})
    system_guide = (NATIVE_SCENE_GUIDE if packet.task.get('scene_geometry_authority_policy') == 'native-table-authority-v1' and native_only(packet)
        else GUIDE+('\n'+CLAIM_GUIDE if claim_scoped else '')+extra)
    return [{'role':'system','content':system_guide}, {'role':'user','content':content}], mapping, {
            **({'scene_geometry_authority_policy':'native-table-authority-v1'} if system_guide == NATIVE_SCENE_GUIDE else {}),
            'image_sha256':digest,'camera_name':view.camera_name,'permutation_seed':seed,
            **({'scene_claim_policy':CLAIM_POLICY} if claim_scoped else {}),
            **({'scene_binding_policy':BINDING_POLICY} if binding else {}),
            **({'scene_input_policy':INPUT_POLICY} if focused else {}),
            **({'scene_relation_reading_policy':packet.task['scene_relation_reading_policy']} if 'sentence_relation_index' in payload else {}),
            **({'anonymous_image_guides':guide_receipts} if joint else {}),
            'native_candidate_count':len(native),'source_image_candidate_count':len(inventory)}


def verify(packet, answer, mapping, receipt, *, generated=None):
    ids = answer.candidate_ids
    known = set(mapping)
    errors = []
    if not set(ids+answer.ambiguous_candidate_ids) <= known or len(set(ids))!=len(ids):
        errors.append('scene_grounding_invalid_candidate_ids')
    if not answer.expression_supported or answer.unsupported_claims:
        errors.append('scene_grounding_unsupported_claims')
    if answer.ambiguous_candidate_ids:
        errors.append('scene_grounding_ambiguous')
    grounded = [mapping[cid] for cid in ids if cid in mapping]
    if grounded != [packet.target_object_id]:
        errors.append('scene_grounding_not_unique_target')
    if not available(packet):
        errors.append('scene_grounding_identity_unverified')
    from app.quality.joint_reference import JOINT_PREFIXES
    from app.quality.spatial_relations import enabled as spatial_enabled, relation_key, verified_pair
    joint_rules = [p for p in packet.predicates if p.predicate_id in (generated or {}).get('predicate_ids',[])
        and p.predicate_id.startswith(JOINT_PREFIXES)]
    verified_pairs = []
    pair_diagnostics = []
    binding = packet.task.get('scene_binding_policy')==BINDING_POLICY
    if joint_rules:
        from app.quality.relations import vehicle_heading_match, vehicle_forward_region_match, VEHICLE_FORWARD_REGION_POLICY
        native = {o.object_id:o for o in packet.candidates}
        if not answer.relation_pairs:
            errors.append('joint_reference_pair_missing')
        else:
            proofs=[json.loads(rule.value) for rule in joint_rules]
            covered=[set() for _ in proofs]
            typed=spatial_enabled(packet)
            for pair in answer.relation_pairs:
                target,anchor = mapping.get(pair.candidate_id),mapping.get(pair.landmark_id)
                supported=[]
                for index,proof in enumerate(proofs):
                    key=relation_key(proof)
                    if (pair.candidate_id not in answer.candidate_ids or target not in native
                            or anchor not in proof['confirmed_anchor_ids'] or int(joint_rules[index].predicate_id.split(':')[1])!=receipt['camera_name']
                            or pair.relation_key!=key and (typed or len(proofs)>1 or pair.relation_key is not None)):
                        continue
                    if 'relation' in proof:
                        valid=verified_pair(packet,proof,target,anchor)
                    else:
                        ordinary=proof['direction'] in {'ahead','rearward'}
                        matcher=vehicle_forward_region_match if ordinary else vehicle_heading_match
                        valid=(anchor in native and (not ordinary or packet.task.get('vehicle_forward_region_policy')==VEHICLE_FORWARD_REGION_POLICY)
                            and matcher(native[target],native[anchor],1 if proof['direction'] in {'front','ahead'} else -1) is True)
                    if valid:
                        supported.append(index)
                        covered[index].add(pair.candidate_id)
                if not supported:
                    errors.append('joint_reference_pair_unverified')
                    if binding:
                        relevant=[p for i,p in enumerate(proofs)
                            if int(joint_rules[i].predicate_id.split(':')[1])==receipt['camera_name']
                            and pair.relation_key==relation_key(p)]
                        code='joint_reference_pair_unverified'
                        if target in native and anchor in native and pair.candidate_id in ids and relevant:
                            if all(anchor not in p.get('possible_anchor_ids',p['confirmed_anchor_ids']) for p in relevant):
                                code='joint_reference_landmark_identity_mismatch'
                        elif target in native and anchor in native and any(anchor in p['confirmed_anchor_ids'] for p in proofs):
                            code='joint_reference_relation_key_mismatch'
                        if code not in errors:
                            errors.append(code)
                        pair_diagnostics.append({**pair.model_dump(),'error':code})
                else:
                    verified_pairs.append({'target_object_id':target,'landmark_object_id':anchor,
                        **({'relation_key':pair.relation_key,'predicate_ids':[joint_rules[i].predicate_id for i in supported]} if typed else {})})
            if any(ids!=set(answer.candidate_ids) for ids in covered):
                errors.append('joint_reference_pair_missing')
    return {**receipt,'diagnostic_pass':not errors,'errors':errors,'protocol':POLICY,
        'tool_grounded_candidates':grounded,'ambiguous_candidate_ids':answer.ambiguous_candidate_ids,
        'unsupported_claims':answer.unsupported_claims,'reason':answer.reason,
        **({'verified_relation_pairs':verified_pairs,'joint_reference_policy':'joint-target-landmark-v1'} if joint_rules else {}),
        **({'pair_diagnostics':pair_diagnostics} if binding else {}),
        'scope':'two fresh contexts with full anonymous scene; same-provider visual grounding, not independent certification'}


def binding_retry_allowed(packet, feedback, *, retries_used, remaining_calls, phase):
    """One new blind read for a descriptor/ID mismatch, never for semantic rejection.

    This only requests another independently numbered read. It supplies no
    expected ID or rejection feedback and cannot turn a failed answer into a pass.
    """
    return (packet.task.get('scene_binding_policy')==BINDING_POLICY and retries_used==0
        and remaining_calls >= (2 if phase=='scene_ground' else 1)
        and bool(feedback.get('pair_diagnostics'))
        and all(p['error']=='joint_reference_landmark_identity_mismatch' for p in feedback['pair_diagnostics'])
        and set(feedback['errors'])=={'joint_reference_landmark_identity_mismatch',
            'joint_reference_pair_unverified','joint_reference_pair_missing'})


def two_rounds_passed(checks):
    if not any(c.get('binding_retry_scheduled') for c in checks):
        return len(checks)==2 and all(c['diagnostic_pass'] for c in checks)
    rejected=[c for c in checks if c.get('binding_retry_scheduled')]
    accepted=[c for c in checks if not c.get('binding_retry_scheduled')]
    return (len(rejected)==1 and rejected[0].get('scene_binding_policy')==BINDING_POLICY
        and not rejected[0]['diagnostic_pass']
        and set(rejected[0]['errors'])=={'joint_reference_landmark_identity_mismatch',
            'joint_reference_pair_unverified','joint_reference_pair_missing'}
        and len(accepted)==2 and all(c['diagnostic_pass'] for c in accepted)
        and {c.get('blind_round') for c in accepted}=={'scene_ground','scene_ground_permuted'}
        and len({c['permutation_seed'] for c in checks})==3)
