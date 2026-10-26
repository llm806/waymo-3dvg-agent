"""Portable ZIP loader. Install numpy and pillow; run: python loader.py --root ."""
import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

DATASET_NAME = 'Waymo-3DRef'
SAMPLE_FIELDS = ('sample_id', 'context', 'frame_timestamp_micros', 'track_id', 'category',
                 'official_split', 'scene', 'expression', 'source_view', 'camera_id',
                 'image_path', 'source_camera_object_id', 'target_box')
CAMERAS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
MANIFEST_FIELDS = {'dataset_name', 'intended_split', 'dataset_split', 'samples', 'records', 'scenes',
                   'loader', 'merge_tool', 'dependencies', 'box_order', 'heading_convention', 'duplicates_removed_this_merge'}
SCENE_FIELDS = {'scene_id', 'context', 'frame_timestamp_micros', 'official_split', 'coordinate_frame',
                'points', 'point_features', 'point_count', 'sensor_returns', 'calibration', 'cameras', 'objects'}


def validate_sample(record):
    """Accept only the public training contract."""
    if set(record) != set(SAMPLE_FIELDS):
        raise ValueError('Unexpected fields in public grounding sample: ' + record['sample_id'])
    return record


def source_camera(source_view):
    for camera_id, name in CAMERAS.items():
        if source_view == name:
            return camera_id
    raise ValueError('Missing or invalid annotation source_view')


def rigid_transform(values):
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError('Invalid camera/vehicle pose')
    return matrix


def camera_parameters(data, camera, context, timestamp):
    """Normalize one original Waymo camera; retain raw timing for rolling shutter."""
    camera_id = camera['camera_name']
    def selected(name):
        rows = [row for row in data[name] if row['key.camera_name'] == camera_id]
        if len(rows) != 1 or rows[0]['key.segment_context_name'] != context:
            raise ValueError('Missing, duplicate or mismatched camera calibration: ' + str(camera_id))
        return rows[0]
    calibration, image = selected('camera_calibration'), selected('camera_frame_metadata')
    frame = data['frame_pose']
    if (image['key.frame_timestamp_micros'] != timestamp
            or (frame['key.segment_context_name'], frame['key.frame_timestamp_micros']) != (context, timestamp)):
        raise ValueError('Camera/frame pose identity mismatch')
    prefix = '[CameraCalibrationComponent]'
    width, height = calibration[f'{prefix}.width'], calibration[f'{prefix}.height']
    intrinsic = np.asarray([calibration[f'{prefix}.intrinsic.{key}']
        for key in ('f_u', 'f_v', 'c_u', 'c_v', 'k1', 'k2', 'p1', 'p2', 'k3')], dtype=np.float64)
    if ((width, height) != (camera['width'], camera['height']) or min(width, height) <= 0
            or not np.isfinite(intrinsic).all() or min(intrinsic[:2]) <= 0):
        raise ValueError('Invalid camera intrinsics/dimensions')
    for key, field in (('pose_timestamp_s', 'pose_timestamp'), ('trigger_time_s', 'rolling_shutter_params.camera_trigger_time'),
                       ('readout_done_time_s', 'rolling_shutter_params.camera_readout_done_time')):
        value = image[f'[CameraImageComponent].{field}']
        if value is None or not np.isfinite(value) or camera[key] != value:
            raise ValueError('Camera timing differs from scene')
    vehicle_from_camera = rigid_transform(calibration[f'{prefix}.extrinsic.transform'])
    world_from_image_vehicle = rigid_transform(image['[CameraImageComponent].pose.transform'])
    world_from_frame_vehicle = rigid_transform(frame['[VehiclePoseComponent].world_from_vehicle.transform'])
    # Column vectors: Waymo camera (+x forward,+y left,+z up) -> OpenCV (+x right,+y down,+z forward).
    opencv_from_waymo = np.asarray([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    ego2cam = opencv_from_waymo @ np.linalg.inv(vehicle_from_camera) @ np.linalg.inv(world_from_image_vehicle) @ world_from_frame_vehicle
    fu, fv, cu, cv = intrinsic[:4]
    return {'camera_id': camera_id, 'source_view': CAMERAS[camera_id], 'image_size': [width, height],
            'K': [[float(fu), 0, float(cu)], [0, float(fv), float(cv)], [0, 0, 1]],
            'distortion': intrinsic[4:].tolist(), 'distortion_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
            'ego2cam': ego2cam.tolist(), 'camera_coordinate_frame': 'OpenCV: x right, y down, z forward',
            'transform_convention': 'column vectors; camera_xyz = ego2cam @ frame_vehicle_xyz_homogeneous',
            'projection_reference': 'camera pose_timestamp; apply distortion; rolling shutter is not compensated by ego2cam',
            'pose_timestamp_s': camera['pose_timestamp_s'],
            'raw_camera_calibration': calibration, 'raw_camera_frame_metadata': image, 'raw_frame_pose': frame}


def asset(root, relative):
    root = Path(root).resolve()
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or '\\' in relative or ':' in relative:
        raise ValueError('Invalid dataset path: ' + relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing dataset file: ' + relative)
    return path


def read_manifest(root):
    manifest = json.loads(asset(root, 'manifest.json').read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or set(manifest) - MANIFEST_FIELDS
            or manifest.get('dataset_name') != DATASET_NAME or manifest.get('samples') != 'samples.jsonl'
            or any(type(manifest.get(key)) is not int or manifest[key] <= 0 for key in ('records', 'scenes'))):
        raise ValueError('Invalid dataset manifest')
    return manifest


def iter_samples(root):
    root = Path(root)
    manifest = read_manifest(root)
    with asset(root, manifest['samples']).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                validate_sample(record)
                yield record


def load_sample(root, record):
    """Return model inputs, supervision, metadata without the original project/SDK."""
    validate_sample(record)
    if not record['expression'].strip():
        raise ValueError('Invalid grounding sample')
    folder = asset(root, record['scene']).parent
    scene = json.loads((folder / 'scene.json').read_text(encoding='utf-8'))
    if (set(scene) != SCENE_FIELDS or (scene['context'], scene['frame_timestamp_micros'], scene['official_split'])
            != (record['context'], record['frame_timestamp_micros'], record['official_split'])):
        raise ValueError('Sample/scene identity mismatch')
    camera_id = source_camera(record['source_view'])
    cameras = [c for c in scene['cameras'] if c['camera_name'] == camera_id]
    if record['camera_id'] != camera_id or len(cameras) != 1:
        raise ValueError('Sample/source camera mismatch')
    camera = cameras[0]
    image_path = asset(folder, camera['image']['path'])
    if asset(root, record['image_path']) != image_path:
        raise ValueError('Sample/source image mismatch')
    with Image.open(image_path) as image:
        if image.format != 'JPEG' or image.size != (camera['width'], camera['height']):
            raise ValueError('Invalid camera image')
        image.verify()
    data = json.loads(asset(folder, scene['calibration']['path']).read_text(encoding='utf-8'))
    calibration = camera_parameters(data, camera, scene['context'], scene['frame_timestamp_micros'])
    stored = dict(data['camera_models'][str(camera_id)])
    expected = dict(calibration)
    for key in ('K', 'distortion', 'ego2cam'):
        actual, native = np.asarray(stored.pop(key)), np.asarray(expected.pop(key))
        if actual.shape != native.shape or not np.allclose(actual, native, rtol=1e-8, atol=1e-8):
            raise ValueError('Normalized camera calibration differs from native metadata')
    if stored != expected:
        raise ValueError('Normalized camera calibration differs from native metadata')
    with np.load(asset(folder, scene['points']['path']), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    count = scene['point_count']
    shapes = {'points': (count, 6), 'laser_id': (count,), 'return_id': (count,),
              'range_image_pixel': (count, 2), 'camera_projection': (count, 6), 'nlz': (count,)}
    if set(arrays) != set(shapes) or any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k, shape in shapes.items()):
        raise ValueError('Invalid point cloud arrays')
    objects = json.loads(asset(folder, scene['objects']['path']).read_text(encoding='utf-8'))
    target = next((o for o in objects if o['object_id'] == record['track_id']), None)
    box = record['target_box']
    if target is None or any(target[k] != box[k] for k in ('center_m', 'size_m', 'heading_rad')) or target['category'] != record['category']:
        raise ValueError('Target differs from native scene labels')
    inputs = {**arrays, 'point_features': scene['point_features'], 'expression': record['expression'],
              'image': image_path, 'camera_id': camera_id, 'calibration': calibration,
              'source_view': record['source_view'], 'coordinate_frame': scene['coordinate_frame']}
    supervision = {'target_box': np.asarray([*box['center_m'], *box['size_m'], box['heading_rad']], dtype=np.float32),
                   'category': record['category'], 'scene_objects': objects}
    metadata = {k: record[k] for k in ('sample_id', 'context', 'frame_timestamp_micros', 'track_id', 'official_split')}
    return inputs, supervision, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    count = 0
    for row in iter_samples(args.root):
        load_sample(args.root, row)
        count += 1
    print(f'Loaded {count} grounding samples successfully.')
