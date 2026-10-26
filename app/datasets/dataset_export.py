"""Snapshot successful task annotations into a portable, automatically validated dataset."""
import json
import tempfile
import zipfile
from pathlib import Path

from app.agent.selection import passed_description
from app.agent.model_calls import image_model_error
from app.datasets.correspondence import verified_native_objects
from app.datasets.export_loader import DATASET_NAME, camera_parameters, source_camera
from app.datasets.training import read_scene
from app.datasets.waymo import WaymoReader
from app.infrastructure.artifacts import asset_path
from app.quality.geometry import inside_oriented_box
from app.schemas.training_contracts import NativeBox
from app.quality.sample_review import review_status, manually_approved, expression as sample_expression


def rejection(state):
    if review_status(state) in {'rejected', 'discarded'}:
        return 'manual_' + review_status(state)
    if state.get('provider_scope') != 'real-model':
        return 'not_real_model'
    if image_model_error(state.get('model', ''), state.get('base_url', '')):
        return 'model_not_vision_capable'
    if state.get('inflight'):
        return 'not_successful'
    text = sample_expression(state) if manually_approved(state) else state.get('final_expression')
    if not isinstance(text, str) or not text.strip():
        return 'no_final_text'
    if manually_approved(state):
        return None  # Native identity, full scene, source assets and split still checked below.
    if not passed_description(state) or state.get('inflight'):
        return 'not_successful'
    attempt = (state.get('attempts') or [{}])[-1]
    checks = attempt.get('ground_checks', [])
    if (attempt.get('generated', {}).get('expression') != state['final_expression']
            or len(checks) < 2 or not all(c.get('diagnostic_pass') is True for c in checks[-2:])):
        return 'validation_incomplete'
    if attempt.get('generated', {}).get('visual_attributes') and attempt.get('appearance_check', {}).get('passed') is not True:
        return 'appearance_not_verified'
    if state.get('current_packet', {}).get('task', {}).get('admission_policy') and not state.get('admission', {}).get('passed'):
        return 'admission_not_passed'
    return None


def export_selection(service, job_ids=None, intended_split=None):
    """Validate task scope before creating a worker or loading scene assets."""
    requested = None if job_ids is None else set(job_ids)
    if requested is not None and not 1 <= len(requested) <= 1000:
        raise ValueError('请选择 1–1000 个任务进行导出。')
    jobs = service.store.summaries()
    if requested is not None and requested - {job['job_id'] for job in jobs}:
        raise ValueError('部分所选任务已不存在，请刷新任务列表后重选。')
    selected = []
    for job in jobs:
        if requested is not None and job['job_id'] not in requested:
            continue
        if job.get('status') != 'completed' or job.get('executor_active') is not False:
            if requested is not None:
                raise ValueError('所选任务尚未完成或仍在执行，请等待处理结束后导出：' + job['job_id'])
            continue
        selected.append(job)
    if not selected:
        raise ValueError('暂无已完成且执行器已退出的任务可导出。')
    splits = {job.get('spec', {}).get('dataset_split', 'train') for job in selected}
    if len(splits) != 1:
        raise ValueError('训练集和验证集任务不能一起导出，请分别选择并分开导出。')
    split = splits.pop()
    if split not in {'train', 'validation'} or intended_split not in {None, split}:
        raise ValueError('导出类型与所选任务不一致，请选择对应的训练集或验证集导出。')
    return split, [job['job_id'] for job in selected]


def build_dataset(service, root, job_ids=None, intended_split=None, *, progress=None, checkpoint=None):
    """Return an owned temporary ZIP stream; the caller must close it after download."""
    from app.routers.samples import sample
    if intended_split not in {None, 'train', 'validation'}:
        raise ValueError('Invalid intended dataset split')
    root = Path(root)
    runs = service.runner.store
    intended_split, selected_jobs = export_selection(service, job_ids, intended_split)
    requested = set(selected_jobs)
    official_split = 'training' if intended_split == 'train' else 'validation'
    output = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode='w+b')
    def report(**values):
        if checkpoint is not None:
            checkpoint()
        if progress is not None:
            progress(**values)

    # zipfile copies large assets in chunks. Check cancellation on every write,
    # including inside a single point-cloud file, without changing ZIP contents.
    class CheckedOutput:
        def __getattr__(self, name):
            return getattr(output, name)

        def write(self, data):
            report()
            written = output.write(data)
            report(bytes_written=output.tell())
            return written

    try:
        report(stage='selecting', message='正在读取任务列表', completed=0, total=0)
        # Deletion cannot remove source assets halfway through packaging.
        with runs.maintenance(checkpoint=checkpoint):
            with runs.connect() as db:
                jobs = [r['job_id'] for r in db.execute('SELECT job_id FROM batches ORDER BY created_at DESC, job_id')]
            if requested is not None:
                if requested - set(jobs):
                    raise ValueError('部分所选任务已不存在，请刷新任务列表后重选。')
                jobs = [job for job in jobs if job in requested]
            candidates = {}
            blocked_tracks = {}
            completed_jobs = []
            for job_index, job_id in enumerate(jobs):
                report(stage='selecting', message=f'正在检查任务 {job_index + 1}/{len(jobs)}', completed=job_index, total=len(jobs))
                job = service.store.detail(job_id)
                if job.get('status') != 'completed' or job.get('executor_active') is not False:
                    if requested is not None:
                        raise ValueError('所选任务尚未完成或仍在执行，请等待处理结束后导出：' + job_id)
                    continue
                completed_jobs.append(job_id)
                if job.get('spec', {}).get('dataset_split', 'train') != intended_split:
                    raise ValueError('任务数据集发生变化，请刷新后分开导出。')
                for item_index, item in enumerate(job['items']):
                    report(message=f'正在检查任务 {job_index + 1}/{len(jobs)} · 标注 {item_index + 1}/{len(job["items"])}')
                    if not item.get('run_id'):
                        continue
                    state = runs.snapshot(item['run_id'])
                    if review_status(state) in {'rejected', 'discarded'}:
                        key = runs.target_key(state)
                        blocked_tracks[key] = max(blocked_tracks.get(key, float('-inf')), state.get('created_at', 0))
                    reason = rejection(state)
                    if reason:
                        continue
                    # Reuse the scene viewer's task/run/frame-reselection binding checks.
                    try:
                        _, _, state, packet, _ = sample(service.store, job_id, item['ordinal'], job)
                    except ValueError as exc:
                        raise ValueError(f'任务 {job_id}、物体 #{item["ordinal"]} 导出核对失败：{exc}') from exc
                    reason = rejection(state)
                    if reason:
                        continue
                    candidates.setdefault(state['run_id'], (state, packet))
            jobs = completed_jobs
            if not jobs:
                raise ValueError('暂无已完成且执行器已退出的任务可导出。')
            unique = {}
            report(stage='selecting', message='正在按既有规则去重并核对审核状态', completed=len(jobs), total=len(jobs))
            for state, packet in sorted(candidates.values(), key=lambda c: (c[0].get('created_at', 0), c[0]['run_id']), reverse=True):
                report()
                if state.get('created_at', 0) <= blocked_tracks.get((packet.context, packet.target_object_id), float('-inf')):
                    continue
                key = (packet.context, packet.target_object_id)
                if key not in unique:
                    unique[key] = (state, packet)
            if not unique:
                raise ValueError('暂无自动核验或人工审批通过的有效标注，无法导出数据集。')
            scene_cameras = {}
            for state, packet in unique.values():
                if packet.scene is None:
                    raise ValueError('成功样本缺少完整场景：' + state['run_id'])
                scene_cameras.setdefault(packet.scene.scene_id, set()).add(source_camera(packet.source_view))
            reader = WaymoReader(root / 'waymo_trainset')
            scenes = {}
            objects_written = set()
            records = []
            report(stage='packaging', message='正在准备场景资源', completed=0, total=len(unique))
            with zipfile.ZipFile(CheckedOutput(), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for state, packet in sorted(unique.values(), key=lambda c: c[1].scene.scene_id if c[1].scene else ''):
                    report(message=f'正在校验样本 {len(records) + 1}/{len(unique)} 的完整点云和原图')
                    path = asset_path(root, packet.scene.manifest_path)
                    scene_id = packet.scene.scene_id
                    if scene_id not in scenes:
                        selected_cameras = scene_cameras[scene_id]
                        scene, arrays = read_scene(path.parent, verify_hashes=False, camera_ids=selected_cameras)
                        if scene.conversion.get('implementation') != 'official-waymo-sdk':
                            raise ValueError('场景缺少官方 SDK 转换记录：' + scene_id)
                        # Samples are grouped by scene; retain only this scene's arrays.
                        scenes[scene_id] = (scene, path)
                        portable = scene.model_dump(mode='json', exclude={'schema_version', 'conversion', 'diagnostics', 'source_components'})
                        portable['cameras'] = [c for c in portable['cameras'] if c['camera_name'] in selected_cameras]
                        for ref in [portable['points'], portable['calibration'], *(c['image'] for c in portable['cameras'])]:
                            ref.pop('sha256', None)
                        portable['objects'] = {'path': 'objects.json'}
                        calibration = json.loads(asset_path(path.parent, scene.calibration.path).read_text(encoding='utf-8'))
                        for key in ('camera_calibration', 'camera_frame_metadata'):
                            calibration[key] = [row for row in calibration[key] if row['key.camera_name'] in selected_cameras]
                        calibration['camera_models'] = {str(c['camera_name']): camera_parameters(calibration, c, scene.context,
                            scene.frame_timestamp_micros) for c in portable['cameras']}
                        archive.writestr(f'scenes/{scene_id}/{scene.calibration.path}', json.dumps(calibration, ensure_ascii=False, allow_nan=False))
                        for ref in [scene.points, *(c.image for c in scene.cameras if c.camera_name in selected_cameras)]:
                            report(message=f'正在写入场景资源：{scene_id}/{ref.path}')
                            archive.write(asset_path(path.parent, ref.path), f'scenes/{scene_id}/{ref.path}', compress_type=zipfile.ZIP_STORED)
                        archive.writestr(f'scenes/{scene_id}/scene.json', json.dumps(portable, ensure_ascii=False, allow_nan=False))
                    scene, original_path = scenes[scene_id]
                    if scene.official_split != official_split or reader.official_split(packet.context) != official_split:
                        raise ValueError('任务与原始数据官方划分不一致，禁止混合导出：' + state['run_id'])
                    if (original_path != path or scene.scene_id != scene_id
                            or (scene.context, scene.frame_timestamp_micros) != (packet.context, packet.frame_timestamp_micros)
                            or scene.point_count != packet.scene.total_points):
                        raise ValueError('标注与场景身份不一致：' + state['run_id'])
                    report(message=f'正在核对样本 {len(records) + 1}/{len(unique)} 的原生框和点云支持')
                    objects = verified_native_objects(reader, packet)
                    count = int(inside_oriented_box(arrays['points'][:, :3], packet.target.center_m, packet.target.size_m, packet.target.heading_rad).sum())
                    if count <= 0 or count != packet.scene.target_points:
                        raise ValueError('目标点云支持与校验记录不一致：' + state['run_id'])
                    objects_path = f'scenes/{scene_id}/objects.json'
                    if objects_path not in objects_written:
                        archive.writestr(objects_path, json.dumps([o.model_dump(mode='json') for o in objects.values()], ensure_ascii=False, allow_nan=False))
                        objects_written.add(objects_path)
                    camera_id = source_camera(packet.source_view)
                    camera = next(c for c in scene.cameras if c.camera_name == camera_id)
                    records.append({'sample_id': state['run_id'],
                        'context': packet.context, 'frame_timestamp_micros': packet.frame_timestamp_micros,
                        'track_id': packet.target_object_id, 'category': packet.target.category,
                        'official_split': scene.official_split, 'scene': f'scenes/{scene_id}/scene.json',
                        'expression': sample_expression(state), 'source_view': packet.source_view,
                        'camera_id': camera_id, 'image_path': f'scenes/{scene_id}/{camera.image.path}',
                        'source_camera_object_id': packet.source_camera_object_id,
                        'target_box': NativeBox(center_m=packet.target.center_m, size_m=packet.target.size_m, heading_rad=packet.target.heading_rad).model_dump(mode='json')})
                    report(completed=len(records), message=f'已打包 {len(records)}/{len(unique)} 条样本')
                manifest = {'dataset_name': DATASET_NAME, 'intended_split': intended_split,
                    'samples': 'samples.jsonl', 'records': len(records),
                    'scenes': len(scenes),
                    'merge_tool': 'merge.py',
                    'loader': 'loader.py', 'dependencies': ['numpy', 'pillow'],
                    'heading_convention': 'counterclockwise about +z; local length axis is +x',
                    'box_order': ['cx', 'cy', 'cz', 'length', 'width', 'height', 'heading_rad']}
                report(stage='finalizing', message='正在写入清单和加载工具并完成 ZIP', completed=0, total=0)
                archive.writestr('samples.jsonl', ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
                archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
                archive.write(Path(__file__).with_name('export_loader.py'), 'loader.py')
                archive.write(Path(__file__).with_name('export_merge.py'), 'merge.py')
        report(bytes_written=output.seek(0, 2))
        output.seek(0)
        return output
    except BaseException:
        output.close()
        raise


def download_chunks(stream):
    try:
        while chunk := stream.read(1024 * 1024):
            yield chunk
    finally:
        stream.close()
