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


def anonymous_image_guides(photo, rows):
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
    tile_w,tile_h,columns = 240,166,5
    sheet = Image.new('RGB',(columns*tile_w,math.ceil(len(boxes)/columns)*tile_h),'#f4f4f4')
    label_draw = ImageDraw.Draw(sheet)
    for index,(cid,box_index,box) in enumerate(boxes):
        x1,y1,x2,y2 = box
        x1,y1=max(0,min(width,math.floor(x1))),max(0,min(height,math.floor(y1)))
        x2,y2=max(0,min(width,math.ceil(x2))),max(0,min(height,math.ceil(y2)))
        label = cid + (f' / {box_index+1}' if box_index else '')
        left,top=(index%columns)*tile_w,(index//columns)*tile_h
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
        crop=original.crop((max(0,x1-pad),max(0,y1-pad),min(width,x2+pad),min(height,y2+pad)))
        crop.thumbnail((tile_w-10,tile_h-30))
        sheet.paste(crop,(left+(tile_w-crop.width)//2,top+27))
    result=[]
    for kind,img in [('anonymous full-scene regions',overlay),('all anonymous source-region tiles',sheet)]:
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
    with Image.open(io.BytesIO(photo)) as decoded:
        image_width,image_height = decoded.size
    inventory = [c for c in packet.provenance['scene_image_candidates'] if c['camera_name']==view.camera_name]
    native = {o.object_id:o for o in packet.candidates}
    ids = sorted(set(native) | {c['object_id'] for c in inventory})
    random.Random(seed).shuffle(ids)
    mapping = {f'C{i+1:03d}':oid for i,oid in enumerate(ids)}
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
    from app.quality.spatial_relations import enabled as spatial_enabled, native_only, scene_graph
    typed = joint and spatial_enabled(packet)
    if typed:
        payload['spatial_relations']=scene_graph(packet,view.camera_name,mapping)
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
    content = [
        {'type':'text','text':json.dumps(payload,ensure_ascii=False,allow_nan=False)},
        {'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(photo).decode()}}]
    guide_receipts=[]
    for kind,data in (anonymous_image_guides(photo,rows) if joint else []):
        content.extend([{'type':'text','text':kind},
            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}}])
        guide_receipts.append({'kind':kind,'sha256':hashlib.sha256(data).hexdigest()})
    return [{'role':'system','content':GUIDE+extra}, {'role':'user','content':content}], mapping, {
            'image_sha256':digest,'camera_name':view.camera_name,'permutation_seed':seed,
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
                else:
                    verified_pairs.append({'target_object_id':target,'landmark_object_id':anchor,
                        **({'relation_key':pair.relation_key,'predicate_ids':[joint_rules[i].predicate_id for i in supported]} if typed else {})})
            if any(ids!=set(answer.candidate_ids) for ids in covered):
                errors.append('joint_reference_pair_missing')
    return {**receipt,'diagnostic_pass':not errors,'errors':errors,'protocol':POLICY,
        'tool_grounded_candidates':grounded,'ambiguous_candidate_ids':answer.ambiguous_candidate_ids,
        'unsupported_claims':answer.unsupported_claims,'reason':answer.reason,
        **({'verified_relation_pairs':verified_pairs,'joint_reference_policy':'joint-target-landmark-v1'} if joint_rules else {}),
        'scope':'two fresh contexts with full anonymous scene; same-provider visual grounding, not independent certification'}
