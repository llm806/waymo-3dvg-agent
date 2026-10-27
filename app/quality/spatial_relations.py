"""Typed, target-independent relation definitions shared by search and checking."""
from dataclasses import dataclass
import hashlib
import json
import re
import math

LEGACY_POLICY = 'typed-spatial-relations-v1'
POLICY = 'native-3d-spatial-relations-v1'
LEGACY_READING_POLICY = 'explicit-subject-landmark-v1'
READING_POLICY = 'explicit-subject-landmark-v2-bidirectional'
PLAIN_SPATIAL_POLICY = 'plain-spatial-language-v1'


def native_only(packet):
    return packet.task.get('spatial_relation_policy') == POLICY


def enabled(packet):
    return packet.task.get('spatial_relation_policy') in {POLICY,LEGACY_POLICY}


@dataclass(frozen=True)
class Relation:
    key: str
    frame: str
    words: str
    axis: int
    sign: int


# Image above/below is intentionally distinct from physical elevation. No
# orientation is invented for a pedestrian/sign; heading belongs to vehicles.
RELATIONS = (
    Relation('image_left','source_image','to the left of',0,-1),
    Relation('image_right','source_image','to the right of',0,1),
    Relation('image_above','source_image','above',1,-1),
    Relation('image_below','source_image','below',1,1),
    Relation('higher_level','native_vertical','on a higher level than',2,1),
    Relation('lower_level','native_vertical','on a lower level than',2,-1),
)
BY_KEY = {r.key:r for r in RELATIONS}
NATIVE_RELATIONS = (
    Relation('native_left','ego_native','farther left than',1,1),
    Relation('native_right','ego_native','farther right than',1,-1),
    Relation('native_ahead','ego_native','farther ahead than',0,1),
    Relation('native_behind','ego_native','farther behind than',0,-1),
    Relation('native_above','native_vertical_aligned','above',2,1),
    Relation('native_below','native_vertical_aligned','below',2,-1),
)
BY_KEY.update({r.key:r for r in NATIVE_RELATIONS})
HEADING_KEYS = {'front':'vehicle_directly_ahead','rear':'vehicle_directly_behind',
    'ahead':'vehicle_ahead','rearward':'vehicle_behind'}


def relation_key(proof):
    return proof.get('relation') or HEADING_KEYS.get(proof.get('direction'))


def requested_relations(feedback):
    """Planning preference only: feedback cannot establish any spatial fact."""
    patterns={
        'vehicle_directly_ahead':r'正前方|正前面|\b(?:directly|immediately)\s+(?:in front of|ahead of)\b',
        'vehicle_directly_behind':r'正后方|正后面|\b(?:directly|immediately)\s+behind\b',
        'vehicle_ahead':r'前方|前面|\b(?:in front of|ahead of)\b',
        'vehicle_behind':r'后方|后面|\bbehind\b',
        'image_left':r'左边|左侧|左方|\bleft of\b',
        'image_right':r'右边|右侧|右方|\bright of\b',
        'image_above':r'上方|上面|\babove\b',
        'image_below':r'下方|下面|\bbelow\b',
        'higher_level':r'高于|更高|\bhigher (?:level )?than\b',
        'lower_level':r'低于|更低|\blower (?:level )?than\b'}
    keys={key for key,pattern in patterns.items() if re.search(pattern,feedback,re.IGNORECASE)}
    for strict,plain in [('vehicle_directly_ahead','vehicle_ahead'),('vehicle_directly_behind','vehicle_behind')]:
        if strict in keys:
            keys.discard(plain)
    return keys


def sentence_relation_keys(expression):
    """Index recognized clauses only; never infer claims from generated IDs."""
    from app.quality.contrast import expand_coordinated_ego_clause
    expression = expand_coordinated_ego_clause(expression)
    # Consume explicit native clauses first: 'farther behind than' must not
    # also select vehicle_behind, which has a different reference frame.
    keys=set()
    remainder=expression
    for relation in NATIVE_RELATIONS:
        pattern=r'\b'+re.escape(relation.words)+r'\b'
        if re.search(pattern,remainder,re.IGNORECASE):
            keys.add(relation.key)
            remainder=re.sub(pattern,'',remainder,flags=re.IGNORECASE)
    # Ego itself is not a candidate landmark; its unary directions have their
    # own predicates and are not looked up in an object-to-object graph.
    remainder=re.sub(r'\b(?:in front of|ahead of|behind|to the left of|to the right of)\s+(?:the\s+)?ego vehicle\b',
        '',remainder,flags=re.IGNORECASE)
    keys.update(requested_relations(remainder))
    # A conjunction may include both a strict and a plain vehicle relation.
    remainder=re.sub(r'\b(?:directly|immediately)\s+(?:in front of|ahead of|behind)\b','',remainder,flags=re.IGNORECASE)
    keys.update(requested_relations(remainder))
    return keys


def sentence_relation_index(graph, expression, *, policy=READING_POLICY):
    """A sentence-only view of ALL pairs, with no target or appearance hint."""
    keys=sentence_relation_keys(expression)
    definitions={d['key']:d for d in graph['definitions'] if d['key'] in keys}
    rows=[]
    for row in graph['rows']:
        definition=definitions.get(row['relation_key'])
        if not definition:
            continue
        anchor=row['landmark_id']
        frame=definition['reference_frame']
        reference=f'{anchor} vehicle heading' if frame=='landmark_vehicle_heading' else frame
        rows.append({**row,'reference_frame':frame,
            'true_statements':[f'{subject} is {definition["meaning"]} {anchor} (reference: {reference}).'
                for subject in row['true_subject_ids']]})
    by_subject = []
    if policy == READING_POLICY:
        for key in sorted(definitions):
            related = [r for r in rows if r['relation_key'] == key]
            for subject in graph['checked_subject_ids']:
                by_subject.append({'subject_id':subject, 'relation_key':key,
                    'true_landmark_ids':[r['landmark_id'] for r in related if subject in r['true_subject_ids']],
                    'unknown_landmark_ids':[r['landmark_id'] for r in related if subject in r['unknown_subject_ids']]})
    return {'policy':policy,'definitions':list(definitions.values()),'rows':rows,
        **({'by_subject':by_subject, 'lookup_instruction':'After visually identifying a subject and landmark, read the exact subject_id and relation_key row in by_subject. Check membership of the landmark ID in true_landmark_ids or unknown_landmark_ids; cross-check the matching landmark row. Both tables encode identical complete geometry and contain no designated target.'} if policy == READING_POLICY else {}),
        'semantics':'Every listed sentence is SUBJECT relation LANDMARK. Read the landmark row, never invert it. '
            'Ego-native behind and landmark-vehicle ahead can BOTH be true for the same pair. '
            'Use the key asserted by the sentence, not a similarly named key in a different frame. '
            'Unlisted checked subjects are false; unknown_subject_ids and unchecked IDs remain unknown. '
            'This index includes every source landmark for these relation keys; no appearance or target filtering.'}


def pair_match(relation, subject, landmark, boxes, native):
    """Three-valued result; unknown identity/geometry is never an exclusion."""
    if subject == landmark:
        return False
    if relation.frame=='source_image':
        from app.quality.description_search import _image_separation
        if subject not in boxes or landmark not in boxes:
            return None
        return _image_separation(boxes[subject],boxes[landmark],relation.axis,relation.sign)
    first,second=native.get(subject),native.get(landmark)
    if (first is None or second is None or not first.geometry_valid or not second.geometry_valid
            or not first.lidar_points or not second.lidar_points
            or min(*first.size_m,*second.size_m)<=0):
        return None
    from app.quality.contrast import POLICY as GEOMETRY
    if relation.frame=='ego_native':
        def radius(obj):
            angle=obj.heading_rad
            c,s=abs(math.cos(angle)),abs(math.sin(angle))
            return ((c*obj.size_m[0]+s*obj.size_m[1]) if relation.axis==0
                else (s*obj.size_m[0]+c*obj.size_m[1]))/2
        gap=relation.sign*(first.center_m[relation.axis]-second.center_m[relation.axis])
        guard=radius(first)+radius(second)+GEOMETRY['axis_guard_m']
        return True if gap>guard else False if gap < -guard else None
    gap=relation.sign*(first.center_m[2]-second.center_m[2])
    guard=(first.size_m[2]+second.size_m[2])/2+GEOMETRY['axis_guard_m']
    if relation.frame=='native_vertical_aligned':
        # Overhead/underneath needs a common horizontal footprint, not merely
        # a higher box in the photo or a remote object's different elevation.
        # The inner circles are contained in both oriented BEV rectangles.
        aligned=math.dist(first.center_m[:2],second.center_m[:2]) < min(*first.size_m[:2],*second.size_m[:2])/2
        if not aligned:
            return None
    return True if gap>guard else False if gap < -guard else None


def identifiable_anchor(category, oid, box, record, native):
    from app.quality.visibility_policy import usable_rgb_observation
    if (category=='unknown' or box['category']!=category or record.get('visibility_conflict')
            or not usable_rgb_observation(record,allow_partial=True)):
        return False
    # A projected box may include a different foreground object. Image-only
    # native camera boxes can be 2D landmarks, but never acquire guessed 3D IDs.
    return oid not in native or box.get('identity_source') in {
        'native_box_point_correspondence','official_camera_lidar_association'}


def catalog(packet,camera,items,observed,all_ids):
    if not enabled(packet):
        return []
    from app.quality.description_search import confirmed_attributes,landmark_attributes,appearance_phrase,CAMERA_VIEWS
    from app.agent.appearance import filter_visual_candidates
    from app.schemas.contracts import Predicate,VisualAttribute
    native={o.object_id:o for o in packet.candidates}
    boxes={b['object_id']:b for b in items}
    descriptors={}
    for oid,box in boxes.items():
        category=box['category'];record=observed.get(oid,{})
        if not identifiable_anchor(category,oid,box,record,native) or native_only(packet) and oid not in native:
            continue
        attributes=landmark_attributes(category,record)
        # Full salient descriptor, each individually confirmed cue, or category
        # alone when no finer attribute is known. No mandatory color/subtype.
        for group in ([attributes,*[[a] for a in attributes]] if attributes else [[]]):
            key=(category,tuple(sorted((a.kind,a.value) for a in group)))
            descriptors[key]=[VisualAttribute(kind=k,value=v) for k,v in key[1]]
    complete=(packet.task.get('image_inventory_policy')=='native-and-projected-v3-native-rgb'
        and packet.task.get('image_identity_policy')=='official-projection-v1'
        and packet.task.get('reference_policy')=='minimal-sufficient-grounding-v1')
    result=[]
    for (category,key),attrs in sorted(descriptors.items()):
        pool=[oid for oid,b in boxes.items() if b['category'] in {category,'unknown'}]
        possible=filter_visual_candidates(pool,attrs,observed)[0]
        confirmed={oid for oid in possible if identifiable_anchor(category,oid,boxes[oid],observed.get(oid,{}),native)
            and set(key)<={(a.kind,a.value) for a in confirmed_attributes(category,observed[oid])}}
        noun=appearance_phrase(category,attrs).removeprefix('the ')
        descriptor=('an ' if noun[0] in 'aeiou' else 'a ')+noun
        code=hashlib.sha256(json.dumps((category,key)).encode()).hexdigest()[:12]
        for relation in (NATIVE_RELATIONS if native_only(packet) else RELATIONS):
            # Axis order is not distance or a landmark's heading. Do not offer
            # technical frame-qualified comparisons as ordinary prose.
            if packet.task.get('spatial_language_policy') == PLAIN_SPATIAL_POLICY and relation.frame == 'ego_native':
                continue
            supported=confirmed if relation.frame=='source_image' else confirmed & set(native)
            if not supported:
                continue
            matches={}
            for subject in all_ids:
                if subject not in boxes:
                    matches[subject]=False if complete else None
                    continue
                values=[False if subject==anchor else pair_match(relation,subject,anchor,boxes,native)
                    if anchor in supported else None for anchor in possible]
                matches[subject]=True if True in values else None if None in values else False
            proof={'relation':relation.key,'reference_frame':relation.frame,'camera':camera,'category':category,
                'attributes':[a.model_dump() for a in attrs],'possible_anchor_ids':sorted(possible),'confirmed_anchor_ids':sorted(supported)}
            frame_words=" relative to the ego vehicle's orientation" if relation.frame=='ego_native' else ''
            result.append(Predicate(predicate_id=f'image_joint_spatial:{camera}:{relation.key}:{code}',kind='image_relation',
                value=json.dumps(proof,sort_keys=True),phrase=f'{relation.words} {descriptor}{frame_words} in the {CAMERA_VIEWS[camera]}',
                matches=matches,required_capability='rgb',source=POLICY+'; all matching anchors; unknowns and self-exclusion preserved'))
    return result


def verified_pair(packet,proof,subject,landmark):
    if not enabled(packet) or proof.get('relation') not in BY_KEY:
        return False
    if native_only(packet) and proof['relation'] not in {r.key for r in NATIVE_RELATIONS}:
        return False
    boxes={b['object_id']:b for b in packet.provenance.get('scene_image_candidates',[]) if b['camera_name']==proof['camera']}
    native={o.object_id:o for o in packet.candidates}
    return (subject in native and landmark in proof['confirmed_anchor_ids']
        and pair_match(BY_KEY[proof['relation']],subject,landmark,boxes,native) is True)


def scene_graph(packet,camera,mapping):
    """All source pairs, without target, appearance, feedback or proposed claims."""
    from app.quality.relations import vehicle_heading_match, vehicle_forward_region_match, VEHICLE_FORWARD_REGION_POLICY, VEHICLE_HEADING_POLICY
    boxes={b['object_id']:b for b in packet.provenance['scene_image_candidates'] if b['camera_name']==camera}
    native={o.object_id:o for o in packet.candidates}
    reverse={oid:cid for cid,oid in mapping.items()}
    ids=sorted(boxes)
    relations=NATIVE_RELATIONS if native_only(packet) else RELATIONS
    definitions=[{'key':r.key,'reference_frame':r.frame,'meaning':r.words} for r in relations]
    heading=[]
    if packet.task.get('vehicle_heading_policy')==VEHICLE_HEADING_POLICY:
        heading.extend([('vehicle_directly_ahead',1,vehicle_heading_match,'directly in front of'),
            ('vehicle_directly_behind',-1,vehicle_heading_match,'directly behind')])
    if packet.task.get('vehicle_forward_region_policy')==VEHICLE_FORWARD_REGION_POLICY:
        heading.extend([('vehicle_ahead',1,vehicle_forward_region_match,'in front of'),
            ('vehicle_behind',-1,vehicle_forward_region_match,'behind')])
    definitions.extend({'key':key,'reference_frame':'landmark_vehicle_heading','meaning':words,
        'requires_lateral_alignment':key.startswith('vehicle_directly_')} for key,_,_,words in heading)
    rows=[]
    for anchor in ids:
        values={r.key:{oid:pair_match(r,oid,anchor,boxes,native) for oid in ids} for r in relations}
        if anchor in native and native[anchor].category=='vehicle':
            values.update({key:{oid:matcher(native[oid],native[anchor],sign) if oid in native else None for oid in ids}
                for key,sign,matcher,_ in heading})
        rows.extend({'relation_key':key,'landmark_id':reverse[anchor],
            'true_subject_ids':[reverse[oid] for oid,v in pairs.items() if v is True],
            'unknown_subject_ids':[reverse[oid] for oid,v in pairs.items() if v is None]} for key,pairs in values.items())
    return {'definitions':definitions,'checked_subject_ids':[reverse[oid] for oid in ids],'rows':rows,
        'semantics':'Each row asserts SUBJECT relation_key LANDMARK, never the reverse. '
        'Within checked_subject_ids, unlisted subjects in an existing row are false; missing rows or IDs are unknown. '
        'Use each definition reference_frame; never substitute image layout for a native relation. '
        'Vehicle front follows LANDMARK heading; vehicle_ahead/behind require no lateral alignment, '
        'vehicle_directly_ahead/behind require alignment. These are alternative claims: plain in front does NOT require both. '
        'This geometry table alone establishes neither visible identity, appearance nor uniqueness.',
        'spatial_evidence':('Native 3D boxes ONLY. Ego-relative axes are x forward, y left, z up; '
            'above/below additionally require aligned horizontal footprints. Pixels supply NO spatial predicates. '
            'Image-only objects have unknown 3D relationships and remain competitors.' if native_only(packet) else LEGACY_POLICY)}
