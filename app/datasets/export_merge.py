"""Merge independent Waymo-3DRef ZIPs/directories: python merge.py --output trainset parts/*.zip"""
import argparse
from contextlib import contextmanager
import glob
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

import numpy as np

if __package__:
    from .export_loader import DATASET_NAME, asset, camera_parameters, iter_samples, load_sample, read_manifest, validate_sample
else:
    from loader import DATASET_NAME, asset, camera_parameters, iter_samples, load_sample, read_manifest, validate_sample


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


@contextmanager
def dataset_lock(parent):
    # One lock for both splits, so simultaneous train/val imports cannot bypass isolation.
    with (parent / '.waymo-3dref-merge.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def unpack(source, destination):
    """Extract to an empty workspace, rejecting traversal, duplicate entries and links."""
    destination.mkdir()
    seen = set()
    with zipfile.ZipFile(source) as archive:
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename)
            if (not parts.parts or parts.is_absolute() or '..' in parts.parts
                    or '\\' in entry.filename or ':' in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe ZIP entry: ' + entry.filename)
            relative = parts.as_posix().rstrip('/')
            if relative in seen:
                raise ValueError('Duplicate ZIP entry: ' + relative)
            seen.add(relative)
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('ZIP entry escapes extraction root')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)


def same_file(first, second):
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open('rb') as left, second.open('rb') as right:
        while data := left.read(1024 * 1024):
            if data != right.read(len(data)):
                return False
    return True


def copy_asset(source, destination, *, points=False, link=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        equal = same_file(source, destination)
        if not equal and points:
            with np.load(source, allow_pickle=False) as left, np.load(destination, allow_pickle=False) as right:
                equal = (set(left.files) == set(right.files) and all(
                    left[k].dtype == right[k].dtype and np.array_equal(left[k], right[k]) for k in left.files))
        if not equal:
            raise ValueError('Conflicting scene asset: ' + str(destination.relative_to(destination.parents[2])))
        return
    # Only immutable old-output assets or our private extraction workspace may be linked.
    # External extracted shards must remain independent of the published dataset.
    if link:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copyfile(source, destination)


def merge_keyed(first, second, key, label):
    merged = {}
    for value in [*first, *second]:
        identity = value[key]
        if identity in merged and merged[identity] != value:
            raise ValueError('Conflicting ' + label + ': ' + str(identity))
        merged[identity] = value
    return [merged[k] for k in sorted(merged)]


def merge_scene(source_root, record, stage, *, link_assets=False):
    source_path = asset(source_root, record['scene'])
    scene = read_json(source_path)
    scene_id = scene['scene_id']
    # Canonical paths keep merged records portable and prevent namespace collisions.
    import re
    if not re.fullmatch('[a-f0-9]{24}', scene_id) or record['scene'] != f'scenes/{scene_id}/scene.json':
        raise ValueError('Noncanonical scene identity/path')
    incoming = source_path.parent
    target = stage / 'scenes' / scene_id
    previous = read_json(target / 'scene.json') if (target / 'scene.json').exists() else None
    if previous:
        if {k: v for k, v in previous.items() if k != 'cameras'} != {k: v for k, v in scene.items() if k != 'cameras'}:
            raise ValueError('Conflicting scene metadata: ' + scene_id)
        cameras = merge_keyed(previous['cameras'], scene['cameras'], 'camera_name', 'camera metadata')
    else:
        cameras = scene['cameras']
    copy_asset(asset(incoming, scene['points']['path']), target / scene['points']['path'], points=True, link=link_assets)
    for camera in scene['cameras']:
        copy_asset(asset(incoming, camera['image']['path']), target / camera['image']['path'], link=link_assets)
    objects = read_json(asset(incoming, scene['objects']['path']))
    objects_path = target / scene['objects']['path']
    if previous:
        old_objects = read_json(objects_path)
        if sorted(old_objects, key=lambda o: o['object_id']) != sorted(objects, key=lambda o: o['object_id']):
            raise ValueError('Conflicting native scene labels: ' + scene_id)
    else:
        write_json(objects_path, objects)
    calibration = read_json(asset(incoming, scene['calibration']['path']))
    calibration_path = target / scene['calibration']['path']
    camera_keys = {'camera_calibration', 'camera_frame_metadata', 'camera_models'}
    if previous:
        old = read_json(calibration_path)
        if {k: v for k, v in old.items() if k not in camera_keys} != {k: v for k, v in calibration.items() if k not in camera_keys}:
            raise ValueError('Conflicting scene calibration: ' + scene_id)
        for key in ('camera_calibration', 'camera_frame_metadata'):
            calibration[key] = merge_keyed(old[key], calibration[key], 'key.camera_name', key)
    calibration['camera_models'] = {str(c['camera_name']): camera_parameters(calibration, c, scene['context'], scene['frame_timestamp_micros']) for c in cameras}
    write_json(calibration_path, calibration)
    write_json(target / 'scene.json', {**scene, 'cameras': cameras})


def merge_record(records, tracks, record):
    validate_sample(record)
    identity = record['sample_id']
    track = (record['context'], record['track_id'])
    if track in tracks and tracks[track] != identity:
        raise ValueError('Different annotation versions for the same track; select one before merging: ' + str(track))
    if identity in records:
        previous = records[identity]
        if previous != record:
            raise ValueError('Conflicting annotation content: ' + identity)
        return False
    records[identity] = dict(record)
    tracks[track] = identity
    return True


def merge_dataset(sources, output, split=None):
    output = Path(output).absolute()
    inferred = {'trainset': 'train', 'valset': 'validation'}.get(output.name)
    split = split or inferred
    if inferred is None or split != inferred:
        raise ValueError('Output must be named trainset or valset, with a matching split')
    if output.is_symlink() or output.exists() and getattr(output.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
        raise ValueError('Output cannot be a symbolic link')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    expanded = []
    for pattern in sources:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise ValueError('Input not found: ' + str(pattern))
        expanded.extend(Path(p).resolve() for p in matches)
    if not expanded:
        raise ValueError('At least one source package is required')
    if any(p == output or p.is_relative_to(output) or output.is_relative_to(p) for p in expanded):
        raise ValueError('Keep input packages separate from the output directory')
    with dataset_lock(output.parent):
        if output.exists():
            if not output.is_dir() or set(p.name for p in output.iterdir()) - {'samples.jsonl', 'manifest.json', 'loader.py', 'merge.py', 'scenes', '__pycache__'}:
                raise ValueError('Output contains unmanaged files; use a dataset-only directory')
            if not (output / 'manifest.json').is_file():
                if any(output.iterdir()):
                    raise ValueError('Output is not a recognized dataset')
            else:
                expanded.insert(0, output)
        work = Path(tempfile.mkdtemp(prefix='.' + output.name + '-merge-', dir=output.parent)).resolve()
        assert work.parent == output.parent and work != output
        stage, backup = work / 'dataset', work / 'previous'
        stage.mkdir()
        try:
            records, tracks, seen_scenes = {}, {}, set()
            duplicates = 0
            for index, source in enumerate(expanded):
                if source.is_dir():
                    root = source
                elif zipfile.is_zipfile(source):
                    root = work / f'input-{index}'
                    unpack(source, root)
                else:
                    raise ValueError('Input must be a dataset ZIP or extracted directory: ' + str(source))
                manifest = read_manifest(root)
                declared = manifest.get('dataset_split') or manifest.get('intended_split')
                if declared and declared != split:
                    raise ValueError('Package split differs from output split: ' + str(source))
                actual = 0
                for record in iter_samples(root):
                    load_sample(root, record)
                    actual += 1
                    if not merge_record(records, tracks, record):
                        duplicates += 1
                    marker = (root, record['scene'])
                    if marker not in seen_scenes:
                        merge_scene(root, record, stage, link_assets=root == output or root.is_relative_to(work))
                        seen_scenes.add(marker)
                if actual != manifest.get('records'):
                    raise ValueError('Manifest sample count differs from samples.jsonl: ' + str(source))
            if not records:
                raise ValueError('No grounding samples to merge')
            counterpart = output.parent / ('valset' if split == 'train' else 'trainset')
            if counterpart.exists() and counterpart != output and any(counterpart.iterdir()):
                other_split = read_json(asset(counterpart, 'manifest.json')).get('dataset_split')
                if other_split and other_split == split:
                    raise ValueError('Counterpart directory has an inconsistent dataset split')
                other_contexts = {r['context'] for r in iter_samples(counterpart)}
                overlap = {r['context'] for r in records.values()} & other_contexts
                if overlap:
                    raise ValueError('Train/validation context overlap: ' + ', '.join(sorted(overlap)))
            rows = sorted(records.values(), key=lambda r: (r['context'], r['frame_timestamp_micros'], r['track_id']))
            (stage / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
            manifest = {'dataset_name': DATASET_NAME, 'dataset_split': split,
                'samples': 'samples.jsonl', 'records': len(rows), 'scenes': len({r['scene'] for r in rows}),
                'duplicates_removed_this_merge': duplicates,
                'heading_convention': 'counterclockwise about +z; local length axis is +x',
                'box_order': ['cx', 'cy', 'cz', 'length', 'width', 'height', 'heading_rad'],
                'loader': 'loader.py', 'merge_tool': 'merge.py', 'dependencies': ['numpy', 'pillow']}
            write_json(stage / 'manifest.json', manifest)
            module_folder = Path(__file__).resolve().parent
            loader = module_folder / ('export_loader.py' if __package__ else 'loader.py')
            shutil.copyfile(loader, stage / 'loader.py')
            shutil.copyfile(Path(__file__), stage / 'merge.py')
            for row in rows:
                load_sample(stage, row)
            if output.exists():
                output.replace(backup)
            try:
                stage.replace(output)
            except BaseException:
                if backup.exists():
                    try:
                        backup.replace(output)
                    except OSError as error:
                        raise OSError('Previous dataset preserved for recovery at ' + str(backup)) from error
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return manifest
        finally:
            # A failed rollback must keep the previous dataset recoverable.
            if not backup.exists():
                shutil.rmtree(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', help='ZIPs, extracted directories, or wildcard patterns')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation'])
    args = parser.parse_args()
    try:
        report = merge_dataset(args.sources, args.output, args.split)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Merge failed: {error}\n')
    print(f"{DATASET_NAME}: {report['records']} samples, {report['scenes']} scenes -> {args.output}")
