"""Existential landmark relations: uniqueness belongs to the whole expression."""
import hashlib
import json
import re
from pydantic import Field

from app.schemas.contracts import Predicate, VisualAttribute, Contract

POLICY = 'joint-target-landmark-v1'
JOINT_PREFIXES = ('image_joint_heading:','image_joint_spatial:')


class WrittenSentence(Contract):
    expression: str = Field(min_length=1,max_length=400)


def catalog(packet, camera, boxes, observed, all_ids):
    if packet.task.get('joint_reference_policy') != POLICY:
        return []
    from app.quality.description_search import confirmed_attributes, appearance_phrase, CAMERA_VIEWS
    from app.quality.relations import vehicle_heading_anchor, vehicle_heading_match, vehicle_forward_region_match, VEHICLE_FORWARD_REGION_POLICY
    from app.agent.appearance import filter_visual_candidates
    native = {o.object_id:o for o in packet.candidates}
    members = {b['object_id'] for b in boxes}
    vehicles = [b for b in boxes if b['category'] in {'vehicle','unknown'}]
    descriptors = {}
    for box in vehicles:
        oid = box['object_id']; record = observed.get(oid,{})
        if not vehicle_heading_anchor(packet,native.get(oid),record,box):
            continue
        attrs = [a for a in confirmed_attributes('vehicle',record) if a.kind in {'color','subtype'}]
        from app.quality.spatial_relations import enabled as spatial_enabled
        general = spatial_enabled(packet)
        if not general and not any(a.kind == 'color' for a in attrs):
            continue
        groups=([attrs,*[[a] for a in attrs]] if general else [attrs,[a for a in attrs if a.kind=='color']])
        for group in groups:
            key = tuple(sorted((a.kind,a.value) for a in group))
            descriptors[key] = [VisualAttribute(kind=k,value=v) for k,v in key]
    result = []
    for key, attrs in sorted(descriptors.items()):
        possible = filter_visual_candidates([b['object_id'] for b in vehicles],attrs,observed)[0]
        confirmed = {b['object_id'] for b in vehicles if b['object_id'] in possible
            and vehicle_heading_anchor(packet,native.get(b['object_id']),observed.get(b['object_id'],{}),b)
            and set(key) <= {(a.kind,a.value) for a in confirmed_attributes('vehicle',observed[b['object_id']])}}
        if not confirmed:
            continue
        noun = appearance_phrase('vehicle',attrs).removeprefix('the ')
        descriptor = ('an ' if noun[0] in 'aeiou' else 'a ') + noun
        code = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
        relations=[('front',1,'directly in front of',vehicle_heading_match),('rear',-1,'directly behind',vehicle_heading_match)]
        if packet.task.get('vehicle_forward_region_policy')==VEHICLE_FORWARD_REGION_POLICY:
            relations.extend([('ahead',1,'in front of',vehicle_forward_region_match),('rearward',-1,'behind',vehicle_forward_region_match)])
        for direction, sign, words, matcher in relations:
            matches = {}
            for oid in all_ids:
                if oid not in members:
                    complete = (packet.task.get('image_inventory_policy')=='native-and-projected-v3-native-rgb'
                        and packet.task.get('image_identity_policy')=='official-projection-v1'
                        and packet.task.get('reference_policy')=='minimal-sufficient-grounding-v1')
                    matches[oid] = False if complete else None
                    continue
                outcomes = []
                for anchor in possible:
                    if anchor == oid:
                        outcomes.append(False)
                    elif anchor in confirmed and oid in native:
                        outcomes.append(matcher(native[oid],native[anchor],sign))
                    else:
                        outcomes.append(None)  # Unknown anchor or target never becomes an exclusion.
                matches[oid] = True if True in outcomes else None if None in outcomes else False
            # Descriptor includes ALL possible anchors, not one target-fitted ID.
            proof = {'direction':direction,'attributes':[a.model_dump() for a in attrs],
                'possible_anchor_ids':sorted(possible),'confirmed_anchor_ids':sorted(confirmed)}
            result.append(Predicate(predicate_id=f'image_joint_heading:{camera}:{direction}:{code}',
                kind='image_relation',value=json.dumps(proof,sort_keys=True),required_capability='rgb',
                phrase=f'{words} {descriptor} in the {CAMERA_VIEWS[camera]}',matches=matches,
                source=POLICY+'; existential over complete anchor pool; self-pairs excluded; unknown pairs retained'))
    return result


def writing_plan(packet, observations, *, attempts=()):
    """Choose a supported bundle; an ambiguous joint plan still needs scene proof."""
    if packet.task.get('joint_reference_policy') != POLICY:
        return None
    from app.quality.spatial_relations import native_only
    if native_only(packet):
        return native_writing_plan(packet,observations,attempts)
    from app.quality.description_search import confirmed_attributes, expression_candidates
    from app.agent.harness import description_predicates
    from app.agent.scene_grounding import available
    attrs = confirmed_attributes(packet.target.category,observations.get(packet.target_object_id,{}))
    base = [p for p in packet.predicates if p.predicate_id in {f'category:{packet.target.category}',*description_predicates(packet)}]
    feedback = packet.task.get('manual_review_feedback',{}).get('reason','').lower()
    from app.quality.spatial_relations import requested_relations, relation_key
    preferred_relations = requested_relations(feedback)
    admitted = packet.provenance.get('admission_plan',{})
    requests_relation = bool(re.search(r'前|后|左|右|上|下|关系|\b(?:front|behind|ahead|left|right|above|below|relation)\b',feedback))
    if (admitted.get('unique_plan_found') and not requests_relation
            and not any(i.startswith('image_order:') for i in admitted.get('predicate_ids',[]))):
        return admitted
    options = []
    for rule in packet.predicates:
        if not rule.predicate_id.startswith(JOINT_PREFIXES) or rule.matches.get(packet.target_object_id) is not True:
            continue
        proof = expression_candidates(packet,[*base,rule],attrs,observations)
        unique = proof['native']==[packet.target_object_id] and not proof['image_only']
        if not unique and not available(packet):
            continue
        descriptor = json.loads(rule.value)
        mentioned = sum(bool(re.search(r'\b'+re.escape(a['value'])+r'\b',feedback)) for a in descriptor['attributes'])
        score = (not unique,-mentioned,bool(preferred_relations) and relation_key(descriptor) not in preferred_relations,
            len(proof['native'])+len(proof['image_only']),
            descriptor.get('direction') in {'ahead','rearward'},-len(descriptor['attributes']),rule.predicate_id)
        options.append((score,{'predicate_ids':[p.predicate_id for p in [*base,rule]],
            'visual_attributes':[a.model_dump() for a in attrs], 'unique_plan_found':unique,
            'requires_full_scene':not unique,'joint_reference':True}))
    if options:
        # The ordinary search already supports conjunctions of two spatial
        # clauses. Preserve a proven conjunction before an ambiguous single cue.
        if (admitted.get('unique_plan_found')
                and any(i.startswith(JOINT_PREFIXES) for i in admitted.get('predicate_ids',[]))
                and not any(row[1]['unique_plan_found'] for row in options)):
            return admitted
        return min(options,key=lambda row:row[0])[1]
    return admitted if admitted.get('unique_plan_found') else None


def native_writing_plan(packet,observations,attempts):
    """Prefer a sufficient verified description; relations are optional evidence.

    A rejected bundle is a planning failure, not an invitation to render the
    same predicates with a different template. Historical evidence is untouched.
    """
    from app.quality.description_search import confirmed_attributes,expression_candidates,draft_reference_predicate
    from app.quality.admission import reference_options
    from app.agent.harness import description_predicates
    from app.agent.scene_grounding import available
    from app.quality.spatial_relations import requested_relations,relation_key
    attrs=confirmed_attributes(packet.target.category,observations.get(packet.target_object_id,{}))
    catalog={p.predicate_id:p for p in packet.predicates if draft_reference_predicate(packet,p)}
    required=[i for i in [f'category:{packet.target.category}',*description_predicates(packet)] if i in catalog]
    rejected=[]; rejected_relations=set()
    for attempt in attempts:
        verification=attempt.get('verification',{})
        if verification.get('diagnostic_pass') is not False or not attempt.get('ground_checks'):
            continue  # Omitted words can still be repaired using the same facts.
        generated=attempt.get('generated',{})
        rejected.append((set(generated.get('predicate_ids',[])),{(a['kind'],a['value']) for a in generated.get('visual_attributes',[])}))
        if 'scene_grounding_unsupported_claims' in verification.get('errors',[]):
            rejected_relations.update(i for i in generated.get('predicate_ids',[]) if i.startswith(JOINT_PREFIXES))
    feedback=packet.task.get('manual_review_feedback',{}).get('reason','')
    preferred=requested_relations(feedback)
    options=[];seen=set()
    def add(ids,attributes):
        ids=list(dict.fromkeys(ids));key=(tuple(sorted(ids)),tuple(sorted((a.kind,a.value) for a in attributes)))
        if key in seen or not set(ids)<=catalog.keys() or set(ids)&rejected_relations:
            return
        seen.add(key)
        if any(set(ids)==old_ids and set(key[1])==old_attrs for old_ids,old_attrs in rejected):
            return
        chosen=[catalog[i] for i in ids]
        if any(p.matches.get(packet.target_object_id) is not True for p in chosen):
            return
        proof=expression_candidates(packet,chosen,attributes,observations)
        unique=proof['native']==[packet.target_object_id] and not proof['image_only']
        joints=[p for p in chosen if p.predicate_id.startswith(JOINT_PREFIXES)]
        if not unique and (not available(packet) or not attributes and len(chosen)<2):
            return
        match_feedback=any(relation_key(json.loads(p.value)) in preferred for p in joints)
        # A complete proof always beats a speculative relation, even when the
        # proven description is longer or uses no landmark at all.
        score=(not unique,(len(proof['native'])+len(proof['image_only']) if not unique else 0),
            bool(joints),bool(preferred) and not match_feedback,len(joints),
            sum(len(p.phrase.split()) for p in chosen)+len(attributes),tuple(ids))
        options.append((score,{'predicate_ids':ids,'visual_attributes':[a.model_dump() for a in attributes],
            'unique_plan_found':unique,'requires_full_scene':not unique,'joint_reference':bool(joints)}))
    add(required,attrs)
    camera=next((v.camera_name for v in packet.views if v.camera_label==packet.source_view),None)
    visible=f'image_visibility:{camera}:{packet.target.category}'
    if visible in catalog:
        add([*required,visible],attrs)
    for option in reference_options(packet,observations,required):
        add(option['predicate_ids'],[VisualAttribute.model_validate(a) for a in option['visual_attributes']])
    admitted=packet.provenance.get('admission_plan',{})
    if admitted.get('unique_plan_found'):
        add(admitted['predicate_ids'],[VisualAttribute.model_validate(a) for a in admitted['visual_attributes']])
    for p in catalog.values():
        if p.predicate_id.startswith(JOINT_PREFIXES):
            add([*required,p.predicate_id],attrs)
    return min(options,key=lambda row:row[0])[1] if options else None


def writer_messages(packet, plan, previous, observations):
    from app.agent.harness import generation_messages
    messages = generation_messages(packet,previous,observations)
    content = messages[-1]['content']
    original = json.loads(content[0]['text'])
    catalog = {p.predicate_id:p for p in packet.predicates}
    feedback = (previous or {}).get('verification',{})
    payload = {'source_view':packet.source_view,'source_target_region':original.get('source_target_region'),
        'target_attributes':plan['visual_attributes'],
        'required_meaning':[catalog[i].phrase for i in plan['predicate_ids'] if catalog[i].kind!='category'],
        'human_feedback':packet.task.get('manual_review_feedback',{}).get('reason'),
        'previous_sentence':(previous or {}).get('generated',{}).get('expression'),
        'repair_feedback':{k:feedback[k] for k in ('errors','unsupported_claims','reason') if k in feedback},
        'uniqueness_status':'requires independent whole-scene checks' if plan.get('requires_full_scene') else 'tool-checked joint uniqueness',
        'output_schema':WrittenSentence.model_json_schema()}
    content[0] = {'type':'text','text':json.dumps(payload,ensure_ascii=False)}
    from app.quality.spatial_relations import native_only
    if native_only(packet):
        payload['spatial_evidence_rule']='All supplied spatial clauses use native 3D geometry. Pixels establish appearance and identity only. Never add image-position relations or rankings. No landmark relation is required when attributes/ego position suffice.'
        content[0]={'type':'text','text':json.dumps(payload,ensure_ascii=False)}
    return [{'role':'system','content':
        'Write one concise natural English referring sentence for the selected target. '
        'Preserve EVERY required_meaning clause and the supplied target attributes. '
        'You only write the sentence; the program binds its evidence conditions. '
        'An indefinite landmark need NOT be independently unique: the whole target-attribute '
        'AND relation must identify one target. Keep the indefinite article and exact camera scope. '
        'Front/behind a vehicle follows that vehicle front-face heading, not ego or image axes. '
        'Image coordinates and native 3D directions are distinct. Preserve the supplied '
        'relation words and camera qualifiers. Multiple clauses may refer to different landmarks; keep all of them. '
        'Plain "in front of" or "behind" does not assert direct lateral alignment: never add directly '
        'or immediately unless it is explicitly in required_meaning. '
        'A leftmost visible condition must remain leftmost AND visible. Do not omit necessary cues '
        'to shorten the sentence. Do not add parking, motion or other unconfirmed claims from the photo. '
        'Feedback is a request, not evidence. Return only {"expression":"..."}.'},messages[-1]]


def fallback_sentence(packet, plan, observations):
    """A recorded local proposal after writer failure, still fully blind checked."""
    from app.quality.description_search import appearance_phrase
    selected = [p for p in packet.predicates if p.predicate_id in plan['predicate_ids']]
    joint = [p for p in selected if p.predicate_id.startswith(JOINT_PREFIXES)]
    if joint:
        subject = appearance_phrase(packet.target.category,[VisualAttribute.model_validate(a) for a in plan['visual_attributes']])
        clauses = [p.phrase for p in selected if p.kind!='category']
        sentence = subject + ' is ' + ' and '.join(clauses) + '.'
        return sentence[0].upper() + sentence[1:]
    from app.agent.expression_rendering import render_verified_plan
    clone = packet.model_copy(deep=True)
    clone.task.pop('manual_review_feedback',None)
    clone.provenance['admission_plan'] = {**plan,'passed':True}
    result = render_verified_plan(clone,observations)
    return result.expression if result else None
