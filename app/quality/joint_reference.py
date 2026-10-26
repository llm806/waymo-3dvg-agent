"""Existential landmark relations: uniqueness belongs to the whole expression."""
import hashlib
import json
import re
from pydantic import Field

from app.schemas.contracts import Predicate, VisualAttribute, Contract

POLICY = 'joint-target-landmark-v1'
FEEDBACK_POLICY = 'verified-feedback-planning-v1'
REVISION_POLICY = 'feedback-evidence-revision-v1'
LEGACY_ACCEPTANCE_POLICY = 'feedback-acceptance-v1'
SUGGESTIONS_POLICY = 'feedback-acceptance-v2-suggestions'
CONCISION_POLICY = 'feedback-acceptance-v3-concision'
ACCEPTANCE_POLICY = 'feedback-acceptance-v4-relation-corrections'
SALIENCE_POLICY = 'aligned-landmark-preference-v1'
ACCEPTANCE_POLICIES = {LEGACY_ACCEPTANCE_POLICY, SUGGESTIONS_POLICY, CONCISION_POLICY, ACCEPTANCE_POLICY}
JOINT_PREFIXES = ('image_joint_heading:','image_joint_spatial:')


def relation_mentions(text):
    """Bounded literal English relation/landmark parsing; no scene facts inferred.

    Chinese feedback can wrap the English phrase, including trailing 是错误的.
    Unrecognized free text stays outside automatic acceptance.
    """
    from app.quality.spatial_relations import requested_relations
    pattern = (r"(?<![a-z])(?P<relation>(?:(?:directly|immediately)\s+)?(?:behind|in front of|ahead of))\s+"
        r"(?:a|an|the)\s+(?:(?P<color>black|white|red|orange|yellow|blue|green|gray|grey|silver|brown)\s+)?"
        r"(?P<type>pickup truck|sedan|suv|van|truck|bus|car|vehicle|pedestrian|person|cyclist|sign)(?![a-z])")
    mentions = []
    for clause in re.split(r'[,，。;；!！\n]|\bbut\b|而是', text, flags=re.I):
        for match in re.finditer(pattern, clause, re.I):
            # Only explicit denial, not the word 'not' inside an unrelated claim.
            prefix = clause[:match.start()]
            suffix = clause[match.end():]
            denied = bool(re.search(r"(?:不是|并非|不要|别用|不应|错误(?:的是)?|\bnot|\bwrong|\bincorrect|do not use|don't use)\s*$", prefix, re.I)
                or re.match(r'\s*(?:是|才是)?\s*(?:错误|错的|不对|不正确)|\s+is\s+(?:wrong|incorrect|false)', suffix, re.I))
            optional = bool(re.search(r'比如|例如|可以|建议|尝试|\b(?:example|could|may|consider)\b', prefix, re.I))
            attrs = []
            if match['color']:
                attrs.append({'kind': 'color', 'value': match['color'].lower().replace('grey','gray')})
            noun = match['type'].lower()
            category = 'pedestrian' if noun in {'person','pedestrian'} else noun if noun in {'sign','cyclist'} else 'vehicle'
            if noun in {'sedan','suv','van','truck','bus','pickup truck'}:
                attrs.append({'kind': 'subtype', 'value': noun})
            mentions.append({'relation': next(iter(requested_relations(match['relation']))),
                'category': category, 'attributes': attrs, 'denied': denied, 'required': denied or not optional,
                'text': match[0]})
    return mentions


def feedback_relations(task):
    if task.get('feedback_acceptance_policy') != ACCEPTANCE_POLICY:
        return []
    return relation_mentions(task.get('manual_review_feedback', {}).get('reason', ''))


def matches_relation_request(request, key, category, attributes):
    return (request['relation'] == key and request['category'] == category
        and {(a['kind'], a['value']) for a in request['attributes']} <=
            {(a['kind'], a['value']) for a in attributes})


def landmark_salience(packet, predicates):
    """Prefer an aligned, nearby verified anchor; never claim a lane or distance.

    Ranking only: full proofs and explicit feedback still have higher priority.
    The relation truth tables and all unknown competitors remain unchanged.
    """
    if packet.task.get('landmark_salience_policy') != SALIENCE_POLICY:
        return ()
    import math
    from app.quality.relations import vehicle_heading_match, vehicle_forward_region_match
    native = {o.object_id: o for o in packet.candidates}
    scores = []
    for p in predicates:
        if not p.predicate_id.startswith('image_joint_heading:'):
            continue
        proof = json.loads(p.value)
        sign = -1 if proof.get('direction') in {'rear', 'rearward'} else 1
        for oid in proof.get('confirmed_anchor_ids', []):
            anchor = native.get(oid)
            if anchor is None or vehicle_forward_region_match(packet.target, anchor, sign) is not True:
                continue
            distance = math.hypot(*(packet.target.center_m[i]-anchor.center_m[i] for i in (0,1)))
            scores.append((vehicle_heading_match(packet.target, anchor, sign) is not True, distance))
    return min(scores) if scores else (True, float('inf'))


def feedback_requirements(task):
    """Recognize motion/spatial requests, separately from evidence and truth.

    This is a bounded check, not a claim to understand arbitrary human feedback.
    A request to OMIT a cue is not a positive request to add that cue.
    """
    text = task.get('manual_review_feedback', {}).get('reason', '')
    clauses = re.split(r'[，,。.!！;；\n]', text)
    positive = ' '.join(c for c in clauses if not re.search(
        r'不要(?!忽略|遗漏)|无需|不需要|不必|没有要求|去掉|删除|删掉|别写|without\b|\b(?:omit|remove|exclude)\b|\b(?:do not|don\x27t|no need to)\b', c, re.I))
    motion = bool(re.search(r'运动|行驶|行进|移动|静止|停放|停车|向[左右前后]开|转弯|左转|右转|速度|驶向|'
        r'\b(?:motion|moving|movement|driving|travell?ing|stationary|parked|parking|speed|velocity|turning|walking|running)\b', positive, re.I))
    ego, _ = ego_requests(positive)
    spatial = bool(re.search(r'相对空间位置|空间位置|空间关系|\b(?:relative position|spatial (?:position|relation))', positive, re.I))
    required_motion, required_ego, required_spatial = motion, ego, spatial
    if task.get('feedback_acceptance_policy') in {SUGGESTIONS_POLICY, CONCISION_POLICY, ACCEPTANCE_POLICY}:
        # Suggestions and examples rank supported plans, not mandatory facts.
        # An explicit MUST starts a new requirement even after an example.
        required_clauses = []
        example = False
        parts = re.split(r'(比如|例如|举例|for example|\be\.g\.|[，,。.!！;；\n])', text, flags=re.I)
        for part in parts:
            if re.fullmatch(r'比如|例如|举例|for example|e\.g\.', part, re.I):
                example = True
                continue
            if re.fullmatch(r'[。.!！;；\n]', part):
                example = False
                continue
            if re.search(r'不要(?!忽略|遗漏)|无需|不需要|不必|没有要求|去掉|删除|删掉|别写|without\b|\b(?:omit|remove|exclude)\b|\b(?:do not|don\x27t|no need to)\b', part, re.I):
                continue
            mandatory = bool(re.search(r'必须|务必|一定要|不可省略|不能省略|不要忽略|不要遗漏|\b(?:must|required|mandatory)\b', part, re.I))
            suggestion = bool(re.search(r'可以|可考虑|建议|尝试|如有|如果|若|尽量|优先|最好|\b(?:can|could|may|consider|try|if|prefer|ideally)\b', part, re.I))
            if mandatory or not (example or suggestion):
                required_clauses.append(part)
        mandatory_task = {'manual_review_feedback': {'reason': ' '.join(required_clauses)}}
        required = feedback_requirements(mandatory_task)  # Legacy parsing, no recursion into suggestions.
        required_motion, required_ego, required_spatial = required['motion'], set(required['ego']), required['spatial']
    concise_pattern = r'简洁|精简|简短|简练|缩短|短一些|短一点|\b(?:concise|concisely|shorter|shorten|brief|succinct)\b'
    concise = task.get('feedback_acceptance_policy') in {CONCISION_POLICY, ACCEPTANCE_POLICY} and bool(re.search(concise_pattern, positive, re.I))
    required_concise = concise and bool(re.search(concise_pattern, ' '.join(required_clauses), re.I))
    return {'concise': concise, 'required_concise': required_concise,
        'motion': motion, 'ego': sorted(ego), 'spatial': spatial,
        'required_motion': required_motion, 'required_ego': sorted(required_ego), 'required_spatial': required_spatial}


def feedback_acceptance(task, generated=None, *, check_style=True):
    """Coverage of recognized feedback; never use feedback as factual evidence.

    Motion labels/track history are audit-only in the single-frame protocol.
    Even a sentence claiming motion cannot pass until an evidence path exists.
    """
    requested = feedback_requirements(task)
    requirements = []
    errors = []; blocking = []
    if requested['motion']:
        requirements.append({'kind': 'motion', 'status': 'unsupported', 'label': '运动状态／行驶方向',
            'required': requested['required_motion'],
            'reason': '当前单帧描述尚无运动证据核验链；速度标签和轨迹仅供审计，不能以位置、车头朝向或单张图像代替运动。'
                + ('这是建议性线索，将继续寻找有依据的外观或空间线索，未采用的运动建议会保留说明。' if not requested['required_motion'] else '')})
        errors.append('manual_feedback_motion_unsupported')
        if requested['required_motion']:
            blocking.append('manual_feedback_motion_unsupported')
    stated = ego_requests((generated or {}).get('expression', ''))[0]
    selected = set((generated or {}).get('predicate_ids', []))
    labels = {'longitudinal:front': '自车前方', 'longitudinal:behind': '自车后方',
              'lateral:left': '自车左侧', 'lateral:right': '自车右侧'}
    for cue in requested['ego']:
        matched = cue in stated and cue in selected
        requirements.append({'kind': cue, 'status': 'pending' if generated is None else 'covered' if matched else 'missing',
            'required': cue in requested['required_ego'],
            'label': labels[cue], 'reason': '' if matched else '须由原生几何支持，并同时写入描述及核验声明。'})
        if generated is not None and not matched:
            errors.append('manual_feedback_spatial_missing')
            if cue in requested['required_ego']:
                blocking.append('manual_feedback_spatial_missing')
    if requested['spatial'] and not requested['ego']:
        from app.quality.spatial_relations import sentence_relation_keys
        matched = bool(stated & selected or sentence_relation_keys((generated or {}).get('expression', ''))
            and any(i.startswith(JOINT_PREFIXES) for i in selected))
        requirements.append({'kind': 'spatial', 'status': 'pending' if generated is None else 'covered' if matched else 'missing',
            'required': requested['required_spatial'],
            'label': '相对空间位置', 'reason': '' if matched else '来源视角不能代替自车方位或物体间空间关系。'})
        if generated is not None and not matched:
            errors.append('manual_feedback_spatial_missing')
            if requested['required_spatial']:
                blocking.append('manual_feedback_spatial_missing')
    actual_relations = relation_mentions((generated or {}).get('expression', ''))
    for request in feedback_relations(task):
        matches = any(matches_relation_request(request, a['relation'], a['category'], a['attributes'])
            for a in actual_relations if not a['denied'])
        covered = not matches if request['denied'] else matches
        error = 'manual_feedback_rejected_relation_repeated' if request['denied'] else 'manual_feedback_landmark_missing'
        requirements.append({'kind': 'landmark_correction', 'status': 'pending' if generated is None else 'covered' if covered else 'missing',
            'required': request['required'], 'label': ('排除：' if request['denied'] else '采用：') + request['text'],
            'reason': '人工纠正不是视觉证据；所指地标的属性、身份和关系仍须核实。不能满足时必须说明，不能悄悄替换地标。'})
        if generated is not None and not covered:
            errors.append(error)
            if request['required']:
                blocking.append(error)
    if requested['concise']:
        limit = concision_budget(task)
        words = len((generated or {}).get('expression', '').split())
        matched = bool(words and words <= limit) if limit is not None else False
        pending = generated is None or not check_style
        requirements.append({'kind': 'concision', 'status': 'pending' if pending else 'covered' if matched else 'missing',
            'required': requested['required_concise'], 'label': '精简文本',
            'reason': f'在保留真实含义和唯一定位线索的前提下，英文描述最多 {limit} 词；当前 {words} 词。' if limit is not None else '缺少返工前原文，无法自动比较是否精简。',
            'max_words': limit, 'actual_words': words if not pending else None})
        if not pending and not matched:
            errors.append('manual_feedback_concision_missing')
            if requested['required_concise']:
                blocking.append('manual_feedback_concision_missing')
    return {'policy': task.get('feedback_acceptance_policy', LEGACY_ACCEPTANCE_POLICY),
        'requirements': requirements, 'errors': sorted(set(errors)), 'blocking_errors': sorted(set(blocking)),
        'passed': False if errors else True if requirements and generated is not None else None,
        'scope': 'recognized motion, spatial and concision requests only; concision is a length check, grounding and appearance must pass separately'}


def concision_budget(task):
    """A measurable rewrite target, not a substitute for semantic verification."""
    previous = task.get('manual_review_feedback', {}).get('previous_expression', '')
    words = len(previous.split())
    if not words:
        return None
    # Meaningful shortening for long prose; do not demand an impossible zero-word sentence.
    return max(1, words - max(1, (words + 4) // 5))


def ego_requests(text):
    """Keep unary ego cues separate from object-landmark relationships."""
    ids=set()
    def coordinated(match):
        ids.update({'longitudinal:front' if match[1].lower()=='ahead' else 'longitudinal:behind','lateral:'+match[2].lower()})
        return ''
    text=re.sub(r'\b(ahead|behind) and to the (left|right) of (?:the )?ego vehicle\b',coordinated,text,flags=re.IGNORECASE)
    patterns={
        'longitudinal:front':r'自车(?:的)?(?:正)?前(?:方|面)|\b(?:in front of|ahead of)\s+(?:the\s+)?ego vehicle\b',
        'longitudinal:behind':r'自车(?:的)?(?:正)?后(?:方|面)|\b(?:behind|to the rear of)\s+(?:the\s+)?ego vehicle\b',
        'lateral:left':r'自车(?:的)?左(?:侧|方|边)|\b(?:(?:to the )?left of|on the left side of)\s+(?:the\s+)?ego vehicle\b',
        'lateral:right':r'自车(?:的)?右(?:侧|方|边)|\b(?:(?:to the )?right of|on the right side of)\s+(?:the\s+)?ego vehicle\b'}
    for key,pattern in patterns.items():
        if re.search(pattern,text,re.IGNORECASE):
            ids.add(key)
            text=re.sub(pattern,'',text,flags=re.IGNORECASE)
    return ids,text


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


def disputed_attribute_kinds(attempt):
    """A narrow repair hint, never evidence or an override of a failed check.

    Only explicit negation of a declared attribute, with no relation assertion,
    can suggest omitting that optional attribute in a NEW fully checked draft.
    """
    from app.quality.spatial_relations import sentence_relation_keys
    claims = [claim for check in attempt.get('ground_checks', []) for claim in check.get('unsupported_claims', [])]
    if not claims:
        return set()
    attrs = attempt.get('generated', {}).get('visual_attributes', [])
    kinds = set()
    for claim in claims:
        if sentence_relation_keys(claim):
            return set()
        denied = {a['kind'] for a in attrs if re.search(
            r"\b(?:not|isn't|cannot confirm)\s+(?:a\s+|an\s+)?"+re.escape(a['value'])+r'\b', claim, re.I)}
        if not denied:
            return set()
        kinds.update(denied)
    return kinds


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
    revision=packet.task.get('feedback_revision_policy')==REVISION_POLICY
    disputed = set().union(*(disputed_attribute_kinds(a) for a in attempts)) if packet.task.get('attribute_repair_policy') == 'disputed-attribute-omission-v1' else set()
    rejected=[]; rejected_relations=set()
    for attempt in attempts:
        verification=attempt.get('verification',{})
        repeated=revision and bool(set(verification.get('errors',[])) & {
            'manual_review_rejected_text_repeated','failed_draft_text_repeated'})
        if verification.get('diagnostic_pass') is not False or not (attempt.get('ground_checks') or repeated):
            continue  # Omitted words can still be repaired using the same facts.
        generated=attempt.get('generated',{})
        rejected.append((set(generated.get('predicate_ids',[])),{(a['kind'],a['value']) for a in generated.get('visual_attributes',[])}))
        if ('scene_grounding_unsupported_claims' in verification.get('errors',[])
                and not (disputed and disputed_attribute_kinds(attempt))):
            rejected_relations.update(i for i in generated.get('predicate_ids',[]) if i.startswith(JOINT_PREFIXES))
    feedback=packet.task.get('manual_review_feedback',{}).get('reason','')
    signed = feedback_relations(packet.task)
    ranking_feedback = ' '.join(r['text'] for r in signed if not r['denied']) if signed else feedback
    preferred=requested_relations(ranking_feedback)
    requested_ego,remainder=ego_requests(ranking_feedback) if revision else (set(),ranking_feedback)
    if revision:
        preferred=requested_relations(remainder)
    general_spatial=revision and bool(re.search(r'相对空间位置|空间位置|空间关系|\b(?:relative position|spatial (?:position|relation))',feedback,re.IGNORECASE))
    if re.search(r'(?:不需要|不要|无需|without|no need for)\s*(?:相对)?(?:空间|spatial|relative)',feedback,re.IGNORECASE):
        general_spatial=False
    feedback_enabled=packet.task.get('manual_feedback_policy')==FEEDBACK_POLICY
    concise = feedback_requirements(packet.task)['concise']
    options=[];seen=set()
    def add(ids,attributes, expand=True):
        attributes = [a for a in attributes if a.kind not in disputed]
        if concise and expand and attributes:
            # Each smaller bundle receives the SAME complete candidate proof.
            variants = [[], *[[a] for a in attributes],
                *[attributes[:i]+attributes[i+1:] for i in range(len(attributes))]]
            for subset in variants:
                add(ids, subset, expand=False)
        ids=list(dict.fromkeys(ids));key=(tuple(sorted(ids)),tuple(sorted((a.kind,a.value) for a in attributes)))
        if key in seen or not set(ids)<=catalog.keys() or set(ids)&rejected_relations:
            return
        seen.add(key)
        if any(set(ids)==old_ids and set(key[1])==old_attrs for old_ids,old_attrs in rejected):
            return
        chosen=[catalog[i] for i in ids]
        if any(p.matches.get(packet.target_object_id) is not True for p in chosen):
            return
        if packet.task.get('feedback_acceptance_policy') in ACCEPTANCE_POLICIES:
            coverage = feedback_acceptance(packet.task, {'predicate_ids': ids,
                'expression': ' and '.join(p.phrase for p in chosen)}, check_style=False)
            if coverage['blocking_errors']:
                return  # Do not silently replace requested supported cues with a shorter plan.
        proof=expression_candidates(packet,chosen,attributes,observations)
        unique=proof['native']==[packet.target_object_id] and not proof['image_only']
        if concise and not expand and not unique:
            return  # Attribute pruning needs a complete proof, not a speculative shorter plan.
        joints=[p for p in chosen if p.predicate_id.startswith(JOINT_PREFIXES)]
        if not unique and (not available(packet) or not attributes and len(chosen)<2):
            return
        match_feedback=any(relation_key(json.loads(p.value)) in preferred for p in joints)
        ego={p.predicate_id for p in chosen if p.kind in {'longitudinal','lateral'}}
        if revision:
            match_feedback=(bool(preferred) and match_feedback or bool(requested_ego) and requested_ego<=ego
                or general_spatial and not preferred and not requested_ego and bool(ego or joints))
        # Feedback ranks existing TRUE evidence only, never creates evidence or
        # removes unknown competitors. Complete proofs still come first.
        mentions=max((sum(bool(re.search(r'\b'+re.escape(a['value'])+r'\b',ranking_feedback,re.IGNORECASE))
            for a in json.loads(p.value).get('attributes',[])) for p in joints
            if relation_key(json.loads(p.value)) in preferred),default=0)
        wants_spatial=bool(preferred or requested_ego or general_spatial)
        preference=(wants_spatial and not match_feedback,-mentions) if feedback_enabled else ()
        if revision and general_spatial and not preferred and not requested_ego:
            preference+= (bool(joints) and not ego,)
        # A complete proof always beats a speculative relation, even when the
        # proven description is longer or uses no landmark at all.
        size = sum(len(p.phrase.split()) for p in chosen) + sum(len(a.value.split()) + 1 for a in attributes)
        score=(not unique,*preference,*((size,) if concise else ()),(len(proof['native'])+len(proof['image_only']) if not unique else 0),
            bool(joints),bool(preferred) and not match_feedback,*landmark_salience(packet,joints),len(joints),
            sum(len(p.phrase.split()) for p in chosen)+len(attributes),tuple(ids))
        options.append((score,{'omitted_disputed_attribute_kinds':sorted(disputed),'predicate_ids':ids,'visual_attributes':[a.model_dump() for a in attributes],
            'unique_plan_found':unique,'requires_full_scene':not unique,'joint_reference':bool(joints),
            **({'feedback_alignment':{'requested_relation_keys':sorted(preferred),
                'selected_relation_keys':sorted({relation_key(json.loads(p.value)) for p in joints}),
                **({'requested_ego_predicate_ids':sorted(requested_ego),'selected_ego_predicate_ids':sorted(ego),
                    'requested_spatial_cue':wants_spatial} if revision else {}),
                'requested_relation_selected':match_feedback,'matched_landmark_attributes':mentions,
                'scope':'planning preference over verified facts; not evidence or approval'}}
                if feedback_enabled and feedback else {})}))
    add(required,attrs)
    camera=next((v.camera_name for v in packet.views if v.camera_label==packet.source_view),None)
    visible=f'image_visibility:{camera}:{packet.target.category}'
    if visible in catalog:
        add([*required,visible],attrs)
    if revision:
        # reference_options returns only symbolically unique bundles. Still
        # propose TRUE ego cues with unresolved competitors for the existing
        # full-scene checks; unknown candidates are never removed here.
        for p in catalog.values():
            if p.kind in {'longitudinal','lateral'}:
                add([*required,p.predicate_id],attrs)
                if visible in catalog:
                    add([*required,visible,p.predicate_id],attrs)
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
    constrained=packet.task.get('manual_feedback_policy')==FEEDBACK_POLICY
    if constrained:
        payload.update(reviewed_sentence=packet.task.get('manual_review_feedback',{}).get('previous_expression'),
            feedback_alignment=plan.get('feedback_alignment'),
            writing_scope='Only required_meaning supplies spatial claims. Human feedback and previous rejection reasons are NOT evidence. '
                'Do not add, reverse, strengthen or replace a relation from those texts or the image. '
                'If the selected plan has no relation, write an appearance-only sentence. '
                'The planner has already considered the feedback; preserve the selected facts even if a previous explanation contradicts them.')
        content[0]={'type':'text','text':json.dumps(payload,ensure_ascii=False)}
    if constrained and packet.task.get('feedback_revision_policy')==REVISION_POLICY:
        payload['writing_scope']+=' Do not repeat or merely reorder a rejected sentence. Only target_attributes supplies appearance claims; do not copy unconfirmed colors from the reviewed text. A source view is not an ego-relative direction.'
        content[0]={'type':'text','text':json.dumps(payload,ensure_ascii=False)}
    if packet.task.get('feedback_acceptance_policy') in ACCEPTANCE_POLICIES:
        payload['feedback_requirements'] = feedback_acceptance(packet.task)['requirements']
        payload['literal_relation_corrections'] = feedback_relations(packet.task)
        if feedback_requirements(packet.task)['concise']:
            payload['concision_instruction'] = (f'Rewrite in at most {concision_budget(packet.task)} English words. '
                'Use compact clothing terms such as brown top and black bottoms; avoid is located and other filler. '
                'Preserve every selected fact, source camera, reference frame and indefinite landmark article. '
                'Do not replace the landmark or relation to shorten. Length compliance is checked before blind verification.')
        payload['feedback_limitations'] = 'Optional suggestions/examples may be unused when unsupported. Never copy example attributes or motion as facts. Use the selected verified plan; unmet suggestions are reported separately.'
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
        'Feedback is a request, not evidence. Return only {"expression":"..."}.'
        + (' '+payload['writing_scope'] if constrained else '')},messages[-1]]


def writer_contract_errors(packet, plan, expression):
    if packet.task.get('manual_feedback_policy')!=FEEDBACK_POLICY or not plan:
        return []
    from app.quality.spatial_relations import sentence_relation_keys
    allowed=set()
    for predicate in packet.predicates:
        if predicate.predicate_id in plan['predicate_ids']:
            allowed.update(sentence_relation_keys(predicate.phrase))
    errors=['unselected_spatial_relation_in_text'] if sentence_relation_keys(expression)-allowed else []
    if packet.task.get('feedback_revision_policy')==REVISION_POLICY:
        stated,_=ego_requests(expression)
        selected={p.predicate_id for p in packet.predicates if p.predicate_id in plan['predicate_ids'] and p.kind in {'longitudinal','lateral'}}
        if selected-stated:
            errors.append('declared_ego_direction_not_in_text')
        if stated-selected:
            errors.append('unselected_ego_direction_in_text')
    return errors


def fallback_sentence(packet, plan, observations):
    """A recorded local proposal after writer failure, still fully blind checked."""
    from app.quality.description_search import appearance_phrase
    selected = [p for p in packet.predicates if p.predicate_id in plan['predicate_ids']]
    joint = [p for p in selected if p.predicate_id.startswith(JOINT_PREFIXES)]
    simple_ego=(packet.task.get('feedback_revision_policy')==REVISION_POLICY
        and any(p.kind in {'longitudinal','lateral'} for p in selected)
        and all(p.kind in {'category','longitudinal','lateral'} or p.predicate_id.startswith('image_visibility:') for p in selected)
        and all(p.matches.get(packet.target_object_id) is True for p in selected))
    if joint or simple_ego:
        subject = appearance_phrase(packet.target.category,[VisualAttribute.model_validate(a) for a in plan['visual_attributes']])
        subject=subject.replace(' suv',' SUV')
        clauses = [p.phrase for p in selected if p.kind!='category']
        if simple_ego:
            clauses=[re.sub(r'^the '+re.escape(packet.target.category)+r'\s+','',clause) for clause in clauses]
        sentence = subject + ' is ' + ' and '.join(clauses) + '.'
        return sentence[0].upper() + sentence[1:]
    from app.agent.expression_rendering import render_verified_plan
    clone = packet.model_copy(deep=True)
    clone.task.pop('manual_review_feedback',None)
    clone.provenance['admission_plan'] = {**plan,'passed':True}
    result = render_verified_plan(clone,observations)
    return result.expression if result else None
