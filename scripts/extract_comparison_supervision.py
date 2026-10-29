"""Read a frozen ZIP and exact run snapshots; emit optional auxiliary labels only.

Run with python -B -m scripts.extract_comparison_supervision --zip PATH.
--output labels.jsonl requires --audit-output evidence.jsonl. Neither file is an
inference input. No server, model calls, export code, or database writes are used.
Unknown labels must be masked by downstream training, never treated as negatives.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
import zipfile

from app.agent.appearance import compatible_attribute_values, observation_attributes, resolve_target_observation
from app.schemas.contracts import VISUAL_ATTRIBUTE_VALUES, VisualObservation

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {1: 'front view', 2: 'front-left view', 3: 'front-right view',
         4: 'left-side view', 5: 'right-side view'}
RANK = re.compile(r'leftmost|rightmost|topmost|bottommost|second\b|longest|'
                  r'outline.{0,25}(?:lowest|highest|tallest)|extending (?:lowest|highest)|'
                  r'extends (?:lower|higher) than the others|most vertical space|at the (?:top|bottom) of', re.I)
RGB = {'official_camera_lidar_association', 'native_box_point_correspondence', 'native_camera_box'}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def attrs_in(text):
    """Literal scope words only; do not copy all target appearance attributes."""
    found = set()
    text = text.lower()
    for kind, values in VISUAL_ATTRIBUTE_VALUES.items():
        if kind in {'color', 'upper_clothing_color', 'lower_clothing_color', 'helmet_color', 'backpack_color'}:
            continue
        for value in values:
            words = {'handheld bag': r'(?:holding|carrying) (?:a )?bag'}.get(value, re.escape(value))
            if re.search(r'\b' + words + r'(?:s|es)?\b', text):
                found.add((kind, value))
    # Prefer an explicitly named subtype/part to its broad prefix.
    for broad, narrow in [('truck', 'pickup truck'), ('bus', 'bus section'),
                          ('excavator', 'excavator boom'), ('excavator', 'excavator body')]:
        if ('subtype', narrow) in found:
            found.discard(('subtype', broad))
    for color in VISUAL_ATTRIBUTE_VALUES['color']:
        upper = re.search(r'\b' + color + r' (?:top|jacket|shirt)\b', text)
        lower = re.search(r'\b' + color + r' (?:bottoms|trousers|pants)\b', text)
        if upper:
            found.add(('upper_clothing_color', color))
        if lower:
            found.add(('lower_clothing_color', color))
        helmet = re.search(r'\b' + color + r' helmet\b', text)
        if helmet:
            found.add(('helmet_color', color))
        backpack=re.search(r'\b'+color+r' backpack\b',text)
        if backpack:found.add(('backpack_color',color))
        if not upper and not lower and not helmet and not backpack and re.search(r'\b' + color + r'\b', text):
            found.add(('color', color))
    return found


def semantics(sample, pid, generated):
    text = sample['expression'].lower()
    if pid.startswith('longest:'):
        # A scene-wide longest predicate cannot certify a camera-only comparison.
        if 'longest' not in text:
            raise ValueError('rank_not_literal')
        if re.search(r'longest\b[^,.]*\bview\b', text):
            raise ValueError('camera_scope_vs_scene_length_claim')
        return dict(camera_id=None, category=pid.split(':')[1], qualifiers=[],
                    visible_only=False, operator='longest', reference='native_box_length_m',
                    scope_text='longest ' + pid.split(':')[1])
    parts = pid.split(':')
    if len(parts) != 4 or parts[0] != 'image_order':
        raise ValueError('unsupported_rank_operator')
    camera, descriptor, op = int(parts[1]), parts[2], parts[3]
    if camera != sample['camera_id'] or VIEWS[camera] not in text:
        raise ValueError('comparison_camera_not_literal_source_camera')
    category = descriptor.split('~')[0]
    claimed = {tuple(a.split('=', 1)) for a in descriptor.split('~')[1:]}
    if op.startswith(('leftmost', 'rightmost', 'topmost', 'bottommost')):
        word = op.split('_')[0]
        position = {'topmost': 'top', 'bottommost': 'bottom'}.get(word)
        if not re.search(r'\b' + word + r'\b', text) and not (
                position and re.search(r'at the ' + position + r' of', text)):
            raise ValueError('rank_not_literal')
    elif op.startswith('second_from_'):
        if not re.search(r'\bsecond(?:\s+\w+)?\s+from the ' + op.split('_')[-1], text):
            raise ValueError('rank_not_literal')
    if op in {'bottom_edge_visible', 'top_edge_visible', 'height_visible'}:
        cue = {'bottom_edge_visible': r'(?:(?:reaches|extending) lowest|extends lower than the others)',
               'top_edge_visible': r'(?:(?:reaches|extending) highest|extends higher than the others)',
               'height_visible': r'(?:outline is tallest|tallest visible outline|takes up the most vertical space)'}[op]
        if not re.search(cue, text):
            raise ValueError('edge_vs_center_or_rank_not_literal')
    comparison = re.search(r'\bamong\s+(.+)', text)
    if comparison:
        group = re.split(r'\bin the (?:front|left|right)', comparison[1])[0]
        literal = attrs_in(group)
        if 'same garment colors' in group:
            # An explicit anaphor binds just the stated garment colors.
            literal |= {(a['kind'], a['value']) for a in generated.get('visual_attributes', [])
                        if a['kind'] in {'upper_clothing_color', 'lower_clothing_color'}}
    elif op.startswith(('leftmost', 'rightmost', 'topmost', 'bottommost')):
        word = op.split('_')[0]
        match = re.search(r'\b' + word + r'\b([^,.]*)', text)
        if match:
            group = re.split(r'\bin (?:the |that )|\bis\b|\bahead\b|\bto the\b', match[1])[0]
            literal = attrs_in(group)
            if re.search(r'\bone\b', group):
                literal |= attrs_in(text[:match.start()])
        elif op.startswith(('topmost', 'bottommost')) and re.search(r'at the (?:top|bottom) of', text):
            group, literal = 'camera-position clause; verified coarse class', set()
        else:
            raise ValueError('rank_not_literal')
    elif op.startswith('second_from_'):
        if not re.search(r'\bsecond(?:\s+\w+)?\s+from the ' + op.split('_')[-1], text):
            raise ValueError('rank_not_literal')
        group, literal = 'second-from clause; verified coarse class', set()
    elif op in {'bottom_edge_visible', 'top_edge_visible', 'height_visible'}:
        cue = {'bottom_edge_visible': r'(?:(?:reaches|extending) lowest|extends lower than the others)',
               'top_edge_visible': r'(?:(?:reaches|extending) highest|extends higher than the others)',
               'height_visible': r'(?:outline is tallest|tallest visible outline|takes up the most vertical space)'}[op]
        if not re.search(cue, text):
            raise ValueError('edge_vs_center_or_rank_not_literal')
        # Bare target attributes before an extent clause do not define its group.
        group, literal = 'extent clause; no explicit attribute comparison group', set()
    else:
        raise ValueError('unsupported_rank_operator')
    if literal != claimed:
        raise ValueError('literal_group_vs_verified_claim_disagree')
    visible = op.endswith('_visible')
    if visible and 'visible' not in text:
        raise ValueError('visible_scope_not_literal')
    if op in {'bottom_edge_visible', 'top_edge_visible', 'height_visible'}:
        reference = {'bottom_edge_visible': 'visible_outline_bottom_y',
                     'top_edge_visible': 'visible_outline_top_y',
                     'height_visible': 'visible_outline_height'}[op]
    else:
        reference = 'image_center_x' if ('left' in op or 'right' in op) else 'image_center_y'
    return dict(camera_id=camera, category=category,
                qualifiers=[dict(kind=k, value=v) for k, v in sorted(literal)],
                visible_only=visible, operator=op, reference=reference, scope_text=group.strip())


def usable(record):
    return (record.get('target_visible') is True and record.get('visibility') in {'clear', 'partial'}
            and bool(record.get('visibility_reason', '').strip()) and not record.get('visibility_conflict'))


def observations(state, camera):
    result = {}
    for oid, raw in state.get('candidate_appearance', {}).items():
        record = raw.get('camera_observations', {}).get(str(camera))
        if record is None and raw.get('observed_camera_name') == camera:
            record = raw
        if record is None and oid == state['current_packet']['target_object_id'] and 'observed_camera_name' not in raw:
            record = raw  # Persisted standalone target precheck is source-camera bound.
        if record is not None:
            result[oid] = record
    # Reuse the saved source precheck; only explicit contradictory observations
    # can remove its attributes. Missing candidate attributes are not a veto.
    oid = state['current_packet']['target_object_id']
    primary = state.get('resolved_target_observation')
    if primary:
        other = result.get(oid)
        if other and (other.get('visibility_conflict') or not usable(other)):
            return result
        if other:
            if other.get('visibility') == 'partial':
                other = {**other, 'absent_features': []}
            primary = resolve_target_observation(VisualObservation.model_validate(primary), other,
                                                 allow_partial=True).model_dump(mode='json')
        result[oid] = primary
    return result


def membership(category, rows, record, semantic):
    no, unknown = [], []
    if category not in {semantic['category'], 'unknown'}:
        return 'out', ['different_native_category']
    if category == 'unknown':
        unknown.append('unknown_category')
    if semantic['camera_id'] is not None and not rows:
        unknown.append('no_comparison_camera_inventory')
    if semantic['visible_only']:
        if (record.get('target_visible') is False and record.get('visibility') == 'occluded'
                and record.get('visibility_reason') and not record.get('visibility_conflict')):
            no.append('explicit_same_camera_fully_hidden')
        elif not usable(record):
            unknown.append('missing_or_uncertain_same_camera_visibility')
    known = defaultdict(set)
    if usable(record):
        for attr in observation_attributes(record, allow_partial=True):
            known[attr['kind']].add(attr['value'])
    for attr in semantic['qualifiers']:
        kind, value = attr['kind'], attr['value']
        values = known[kind]
        if value in values and (kind == 'feature' or len(values) == 1):
            if kind == 'feature' and value in record.get('absent_features', []):
                unknown.append('conflicting_feature:' + value)
            continue
        if (kind == 'feature' and value in record.get('absent_features', [])
                and usable(record) and record.get('visibility') == 'clear' and value not in values):
            no.append('explicit_feature_absent:' + value)
        elif (kind != 'feature' and len(values) == 1
              and not compatible_attribute_values(kind, values | {value})
              and not (kind == 'color' and values | {value} <= {'gray', 'silver'})):
            no.append('confirmed_different_attribute:' + kind)
        else:
            unknown.append('missing_or_conflicting_attribute:' + kind)
    return ('out', no) if no else ('unknown', unknown) if unknown else ('in', ['confirmed_group_conditions'])


def metric(obj, rows, record, semantic):
    op = semantic['operator']
    if op == 'longest':
        if obj.get('geometry_valid') and all(math.isfinite(x) for x in obj['size_m']):
            return {'value': obj['size_m'][0], 'bounds': None, 'basis': 'native_length', 'guard': .5}
        return None
    if not usable(record):
        return None
    # Multiple boxes/assembly links cannot silently pick a more convenient extent.
    if len(rows) != 1:
        return None
    row = rows[0]
    x1, y1, x2, y2 = row['xyxy']
    if not all(math.isfinite(x) for x in row['xyxy']) or x2 <= x1 or y2 <= y1:
        return None
    if semantic['reference'].startswith('visible_outline'):
        # These are bounds supporting the SAVED extreme proof, not new contour
        # observations. pairs_for permits only a verified native-RGB winner
        # against a confirmed member, never arbitrary rival-to-rival edge order.
        if row.get('identity_source') not in RGB | {'official_lidar_box_projection'}:
            return None
        value = {'bottom_edge_visible': y2, 'top_edge_visible': y1,
                 'height_visible': y2-y1}[op]
        return dict(value=value, bounds=row['xyxy'], basis='saved_extremum_camera_bound',
                    identity_source=row['identity_source'], guard=16)
    axis = 0 if semantic['reference'] == 'image_center_x' else 1
    return dict(value=(row['xyxy'][axis] + row['xyxy'][axis+2])/2,
                bounds=row['xyxy'], basis='recognized_unit_disjoint_camera_bounds', guard=16)


def pairs_for(members, metrics, semantic, target_id):
    pairs, proofs = [], []
    descending = (semantic['operator'] == 'longest' or semantic['operator'].startswith(('rightmost', 'bottommost'))
                  or semantic['operator'] in {'second_from_right', 'second_from_bottom', 'bottom_edge_visible', 'height_visible'})
    for left, right in combinations(sorted(metrics), 2):
        if members[left] != 'in' or members[right] != 'in':
            continue
        a, b = metrics[left], metrics[right]
        extent = semantic['reference'].startswith('visible_outline')
        if extent and (target_id not in {left, right}
                       or metrics[target_id].get('identity_source') not in RGB):
            continue
        if abs(a['value'] - b['value']) <= max(a['guard'], b['guard']):
            continue
        if semantic['reference'] in {'image_center_x', 'image_center_y'}:
            axis = 0 if semantic['reference'].endswith('x') else 1
            lo, hi = (a, b) if a['value'] < b['value'] else (b, a)
            if hi['bounds'][axis] - lo['bounds'][axis+2] <= 8:
                continue
        before, after = (left, right) if (a['value'] > b['value']) == descending else (right, left)
        if extent and before != target_id:
            continue
        pairs.append(dict(before=before, after=after))
        proofs.append(dict(before=before, after=after, measurements={left: a, right: b},
                           support='verified_target_extremum_plus_bound_gap' if extent else 'direct_pair_geometry'))
    return pairs, proofs


def exact_match(sample, state, archive):
    packet = state['current_packet']
    errors = []
    for key, field in [('run_id', 'sample_id'), ('final_expression', 'expression')]:
        if state.get(key) != sample[field]:
            errors.append(key)
    for key, field in [('context', 'context'), ('frame_timestamp_micros', 'frame_timestamp_micros'),
                       ('target_object_id', 'track_id'), ('source_view', 'source_view'),
                       ('source_camera_object_id', 'source_camera_object_id')]:
        if packet.get(key) != sample[field]:
            errors.append(key)
    target = packet['target']
    if target['object_id'] != sample['track_id'] or target['category'] != sample['category']:
        errors.append('target_identity_category')
    if {k: target[k] for k in ('center_m', 'size_m', 'heading_rad')} != sample['target_box']:
        errors.append('target_box')
    if sample['scene'] != f"scenes/{packet['scene']['scene_id']}/scene.json":
        errors.append('scene_id')
    scene = json.loads(archive.read(sample['scene']))
    if (scene['context'], scene['frame_timestamp_micros'], scene['official_split']) != (
            sample['context'], sample['frame_timestamp_micros'], sample['official_split']):
        errors.append('scene_identity_split')
    camera = [c for c in scene['cameras'] if c['camera_name'] == sample['camera_id']]
    views = [v for v in packet['views'] if v['camera_label'] == sample['source_view']]
    if len(views) != 1 or views[0]['camera_name'] != sample['camera_id']:
        errors.append('source_camera')
    source_rows = [r for r in packet['provenance'].get('scene_image_candidates', [])
                   if r['object_id'] == sample['track_id'] and r['camera_name'] == sample['camera_id']]
    if len(source_rows) != 1 or source_rows[0].get('camera_object_id') != sample['source_camera_object_id']:
        errors.append('source_camera_native_identity_link')
    if len(camera) != 1 or str(PurePosixPath(sample['scene']).parent / camera[0]['image']['path']) != sample['image_path']:
        errors.append('source_image_path')
    archive.getinfo(sample['image_path'])
    native = json.loads(archive.read(str(PurePosixPath(sample['scene']).parent / 'objects.json')))
    objects = {o['object_id']: o for o in native}
    packet_objects = {o['object_id']: o for o in packet['candidates']}
    if len(objects) != len(native) or objects != packet_objects:
        errors.append('frozen_native_candidate_inventory')
    attempt = (state.get('attempts') or [{}])[-1]
    if (attempt.get('generated', {}).get('expression') != sample['expression']
            or len(attempt.get('ground_checks', [])) < 2
            or not all(c.get('diagnostic_pass') is True and not c.get('unsupported_claims')
                       for c in attempt.get('ground_checks', [])[-2:])):
        errors.append('final_text_checks')
    return errors, objects, attempt


def extract(sample, state, archive):
    errors, objects, attempt = exact_match(sample, state, archive)
    label = {k: sample[k] for k in ('sample_id', 'context', 'frame_timestamp_micros', 'track_id', 'category')}
    label.update(expression=sample['expression'], expression_sha256=digest(sample['expression']), comparisons=[])
    audit = dict(sample_id=sample['sample_id'], expression_sha256=label['expression_sha256'],
                 match_errors=errors, comparisons=[])
    if errors:
        return label, audit
    packet = state['current_packet']
    generated = attempt['generated']
    predicates = {p['predicate_id']: p for p in packet['predicates']}
    rank_ids = [p for p in generated['predicate_ids'] if p.startswith(('image_order:', 'longest:', 'vertical_rank:'))]
    audit.update(rank_statement_count=len(rank_ids), lexical_rank_detected=bool(RANK.search(sample['expression'])),
                 packet_id=packet['packet_id'], final_generated=generated,
                 ground_checks=attempt['ground_checks'][-2:],
                 appearance_check=attempt.get('appearance_check'),
                 resolved_target_observation=state.get('resolved_target_observation'),
                 raw_candidate_observations=state.get('candidate_appearance', {}),
                 original_image=str(ROOT / 'runtime' / sample['image_path']),
                 source_camera_id=sample['camera_id'])
    for index, pid in enumerate(rank_ids):
        proof = dict(predicate_id=pid)
        audit['comparisons'].append(proof)
        try:
            if not all(pid in c.get('parsed_predicate_ids', []) for c in attempt['ground_checks'][-2:]):
                raise ValueError('rank_not_verified_twice')
            if pid not in predicates or predicates[pid]['matches'].get(sample['track_id']) is not True:
                raise ValueError('verified_rank_missing_target_support')
            semantic = semantics(sample, pid, generated)
        except ValueError as exc:
            proof['withheld'] = str(exc)
            continue
        comparison_id = str(index)
        camera = semantic['camera_id']
        observed = observations(state, camera) if camera else {}
        inventory = [r for r in packet['provenance'].get('scene_image_candidates', []) if r['camera_name'] == camera]
        by_id = defaultdict(list)
        for row in inventory:
            by_id[row['object_id']].append(row)
        members, metrics, reasons = {}, {}, {}
        for oid, obj in objects.items():
            members[oid], reasons[oid] = membership(obj['category'], by_id[oid], observed.get(oid, {}), semantic)
            if members[oid] == 'in':
                measurement = metric(obj, by_id[oid], observed.get(oid, {}), semantic)
                if measurement:
                    metrics[oid] = measurement
        unresolved = []
        for oid, rows in by_id.items():
            if oid not in objects:
                status, why = membership(rows[0]['category'], rows, observed.get(oid, {}), semantic)
                unresolved.append(dict(image_object_id=oid, native_object_id=None, membership=status))
                reasons[oid] = ['not_uniquely_linked_to_native_object', *why]
        pairs, pair_proofs = pairs_for(members, metrics, semantic, sample['track_id'])
        label['comparisons'].append(dict(comparison_id=comparison_id,
            **{k: v for k, v in semantic.items() if k != 'scope_text'},
            candidates=[dict(native_object_id=oid, membership=status) for oid, status in sorted(members.items())],
            unresolved_image_objects=unresolved, pairs=pairs))
        proof.update(comparison_id=comparison_id, semantics=semantic,
                     verified_predicate={k: v for k, v in predicates[pid].items() if k != 'matches'},
                     candidate_reasons=reasons, observations=observed, camera_inventory=inventory,
                     measurements=metrics, pair_proofs=pair_proofs,
                     insufficient_pair_reasons=([] if pairs else ['fewer_than_two_confirmed_members' if
                         sum(v == 'in' for v in members.values()) < 2 else 'no_guarded_pair_with_suitable_geometry_or_visible_extent']))
    return label, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--database', type=Path, default=ROOT / 'runtime/annotation/runs.sqlite3')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    if bool(args.output) != bool(args.audit_output):
        parser.error('--output and --audit-output must be supplied together')
    paths = [p.resolve() for p in (args.zip, args.database, args.output, args.audit_output) if p]
    if len(paths) != len(set(paths)):
        parser.error('input and output paths must be distinct')
    summary = Counter()
    coverage = {k: defaultdict(Counter) for k in ('operator', 'category', 'context')}
    failures, withheld, unknown, unresolved, insufficient = [], Counter(), Counter(), Counter(), Counter()
    examples = []
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(args.zip, 'r'))
        db = sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True)
        stack.callback(db.close)
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')  # One consistent read snapshot; never initializes RunStore.
        out = stack.enter_context(args.output.open('x', encoding='utf-8')) if args.output else None
        audit_out = stack.enter_context(args.audit_output.open('x', encoding='utf-8')) if args.audit_output else None
        manifest = json.loads(archive.read('manifest.json'))
        samples = [json.loads(line) for line in archive.read(manifest['samples']).splitlines() if line.strip()]
        if len({s['sample_id'] for s in samples}) != len(samples):
            raise ValueError('duplicate sample_id in frozen ZIP')
        if manifest['records'] != len(samples):
            raise ValueError('ZIP manifest sample count mismatch')
        summary['zip_samples'] = len(samples)
        summary['zip_scenes'] = manifest['scenes']
        for sample in samples:
            summary['processed'] += 1
            row = db.execute('SELECT snapshot FROM runs WHERE run_id=?', (sample['sample_id'],)).fetchone()
            try:
                if row is None:
                    raise ValueError('sample_id_not_found_in_database')
                label, audit = extract(sample, json.loads(row[0]), archive)
                audit.update(snapshot_sha256=digest(row[0]), database=str(args.database.resolve()),
                             zip_path=str(args.zip.resolve()),
                             extractor_sha256=digest(Path(__file__).read_text(encoding='utf-8')))
                if audit['match_errors']:
                    raise ValueError(','.join(audit['match_errors']))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                failure = dict(sample_id=sample['sample_id'], reason=str(exc))
                failures.append(failure)
                if audit_out:
                    audit_out.write(json.dumps(failure, ensure_ascii=False) + '\n')
                continue
            summary['matched'] += 1
            ranked = bool(audit['rank_statement_count'])
            multi = any(sum(c['membership'] == 'in' for c in comp['candidates']) >= 2 for comp in label['comparisons'])
            paired = any(comp['pairs'] for comp in label['comparisons'])
            summary['ranked_samples'] += int(ranked)
            summary['rank_statements'] += audit['rank_statement_count']
            summary['samples_with_resolved_comparison_semantics'] += int(bool(label['comparisons']))
            summary['samples_with_2_confirmed_members'] += int(multi)
            summary['samples_with_pairs'] += int(paired)
            summary['pairs'] += sum(len(comp['pairs']) for comp in label['comparisons'])
            summary['lexical_rank_without_verified_rank'] += bool(audit['lexical_rank_detected'] and not ranked)
            for key in ('category', 'context'):
                cov = coverage[key][sample[key]]
                cov.update(samples=1, ranked=int(ranked), two_members=int(multi), paired=int(paired))
            for proof in audit['comparisons']:
                if proof.get('withheld'):
                    withheld[proof['withheld']] += 1
                    op = 'longest' if proof['predicate_id'].startswith('longest:') else proof['predicate_id'].rsplit(':', 1)[-1]
                    coverage['operator'][op].update(statements=1, withheld=1)
                    continue
                comp = next(c for c in label['comparisons'] if c['comparison_id'] == proof['comparison_id'])
                coverage['operator'][comp['operator']].update(statements=1,
                    two_members=int(sum(c['membership'] == 'in' for c in comp['candidates']) >= 2),
                    paired=int(bool(comp['pairs'])), pairs=len(comp['pairs']))
                for c in comp['candidates']:
                    if c['membership'] == 'unknown':
                        unknown.update(proof['candidate_reasons'][c['native_object_id']])
                        summary['unknown_native_membership_labels'] += 1
                        if 'no_comparison_camera_inventory' not in proof['candidate_reasons'][c['native_object_id']]:
                            summary['unknown_native_labels_in_comparison_image'] += 1
                for c in comp['unresolved_image_objects']:
                    unresolved[c['membership']] += 1
                for p in proof['pair_proofs']:
                    summary['pairs_' + p['support']] += 1
                insufficient.update(proof['insufficient_pair_reasons'])
            if paired and len(examples) < 12:
                examples.append(dict(sample_id=sample['sample_id'], expression=sample['expression'],
                    image=audit['original_image'], context=sample['context'], frame=sample['frame_timestamp_micros'],
                    comparisons=[dict(operator=c['operator'], qualifiers=c['qualifiers'],
                        members=[m['native_object_id'] for m in c['candidates'] if m['membership'] == 'in'],
                        pairs=c['pairs'][:4]) for c in label['comparisons'] if c['pairs']]))
            if out:
                out.write(json.dumps(label, ensure_ascii=False, allow_nan=False) + '\n')
                audit_out.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + '\n')
    report = dict(counts=summary, match_failures=failures, withheld_semantics=withheld,
        unknown_native_reasons=unknown, unresolved_image_membership=unresolved,
        insufficient_pair_reasons=insufficient,
        coverage={key: dict(sorted(value.items())) for key, value in coverage.items()}, examples=examples,
        interpretation='offline internal evidence extraction; partial pair supervision; not independent human certification; '
            'visible extent pairs only reuse a verified target extremum with native RGB bounds and a >16px guard, '
            'never infer rival-to-rival contour order from boxes or from the target extremum; '
            'second-from operators orient pair order but do not assert a complete ordinal list')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
