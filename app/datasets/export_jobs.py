"""Single-process export lifecycle; temporary ZIPs have one explicit owner."""
import threading
import time
import uuid

from app.datasets.dataset_export import build_dataset, export_selection


class ExportCancelled(Exception):
    pass


class ExportJobs:
    retention_seconds = 3600
    active_states = {'running', 'ready', 'downloading', 'cancelling'}

    def __init__(self, service, root):
        self.service, self.root = service, root
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False

    def _snapshot(self, job):
        return {k: v for k, v in job.items() if k not in {'cancel', 'stream', 'thread'}}

    def current(self):
        with self.lock:
            return self._snapshot(next(reversed(self.jobs.values()))) if self.jobs else None

    def status(self, export_id):
        with self.lock:
            return self._snapshot(self.jobs[export_id])

    def create(self, job_ids, split, filename):
        split, job_ids = export_selection(self.service, job_ids, split)
        with self.lock:
            self.expire()
            if self.closed:
                raise ValueError('服务正在关闭，请稍后重试。')
            if any(j['status'] in self.active_states for j in self.jobs.values()):
                raise ValueError('已有导出正在处理，请先下载或终止当前导出。')
            now = time.time()
            job = dict(export_id=uuid.uuid4().hex[:24], status='running', stage='selecting',
                       message='正在准备导出', completed=0, total=0, bytes_written=0, bytes_sent=0,
                       created_at=now, updated_at=now, filename=filename, split=split,
                       cancel=threading.Event(), stream=None)
            worker = threading.Thread(target=self._build, args=(job, job_ids), daemon=True,
                                      name='dataset-export-' + job['export_id'])
            job['thread'] = worker
            self.jobs[job['export_id']] = job
            try:
                worker.start()
            except RuntimeError:
                del self.jobs[job['export_id']]
                raise
            return self._snapshot(job)

    def _check(self, job):
        if job['cancel'].is_set():
            raise ExportCancelled()

    def _build(self, job, job_ids):
        stream = None
        def progress(**values):
            self._check(job)
            if values:
                with self.lock:
                    self._check(job)
                    if 'bytes_written' in values:
                        values['bytes_written'] = max(job['bytes_written'], values['bytes_written'])
                    job.update(values, updated_at=time.time())
        try:
            stream = build_dataset(self.service, self.root, job_ids, intended_split=job['split'],
                                   progress=progress, checkpoint=lambda: self._check(job))
            with self.lock:
                self._check(job)
                job.update(stream=stream, status='ready', stage='ready', message='打包完成，等待浏览器下载', updated_at=time.time())
                stream = None  # Transfer ownership to the ready job.
        except Exception as exc:
            # build_dataset closes partial output before propagating errors.
            if stream is not None:
                stream.close()
            with self.lock:
                cancelled = job['cancel'].is_set()
                job.update(status='cancelled' if cancelled else 'failed', stage='finished',
                           message='已终止导出，本次临时文件已清理。' if cancelled else '导出失败，临时文件已清理：' + str(exc),
                           updated_at=time.time())

    def cancel(self, export_id):
        with self.lock:
            job = self.jobs[export_id]
            if job['status'] in self.active_states:
                job['cancel'].set()
                if job['status'] == 'ready':
                    job['stream'].close()
                    job.update(stream=None, status='cancelled', stage='finished', message='已终止导出，本次临时文件已清理。')
                else:
                    job.update(status='cancelling', message='正在终止导出并清理临时文件…')
                job['updated_at'] = time.time()
            return self._snapshot(job)

    def open_download(self, export_id):
        with self.lock:
            job = self.jobs[export_id]
            if job['status'] != 'ready':
                raise ValueError('当前导出尚未就绪、已下载或已终止，请查看页面状态。')
            job.update(status='downloading', stage='downloading', completed=0, total=job['bytes_written'],
                       message='正在将 ZIP 传给浏览器', updated_at=time.time())
            return job['stream'], self._snapshot(job)

    def download_chunks(self, export_id, stream):
        job = self.jobs[export_id]
        try:
            while True:
                self._check(job)
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
                with self.lock:
                    job.update(bytes_sent=job['bytes_sent'] + len(chunk),
                               completed=job['bytes_sent'] + len(chunk), updated_at=time.time())
        except ExportCancelled:
            pass
        finally:
            self.finish_download(export_id)

    def finish_download(self, export_id):
        with self.lock:
            job = self.jobs[export_id]
            if job['stream'] is None:
                return
            job['stream'].close()
            cancelled = job['cancel'].is_set()
            complete = job['bytes_sent'] == job['bytes_written']
            job.update(stream=None, status='cancelled' if cancelled else 'downloaded' if complete else 'failed',
                       stage='finished', updated_at=time.time(),
                       message='已终止导出，本次临时文件已清理。' if cancelled else
                       'ZIP 已传给浏览器，本次临时文件已清理。请在浏览器下载列表查看保存结果。' if complete else
                       '下载连接已中断，本次临时文件已清理，请重新导出。')

    def expire(self):
        with self.lock:
            for export_id, job in list(self.jobs.items()):
                if time.time() - job['updated_at'] <= self.retention_seconds:
                    continue
                if job['status'] == 'ready':
                    self.cancel(export_id)
                    job.update(status='expired', message='导出包等待下载已超过一小时，临时文件已清理，请重新导出。')
                elif job['status'] not in self.active_states:
                    del self.jobs[export_id]

    def close(self):
        with self.lock:
            self.closed = True
            for export_id in self.jobs:
                self.cancel(export_id)
            workers = [j['thread'] for j in self.jobs.values()]
        for worker in workers:
            worker.join()
        with self.lock:
            # Lifespan shutdown follows connection draining. Also close a ZIP
            # reserved for a response whose body iterator never started.
            for export_id, job in self.jobs.items():
                if job['stream'] is not None:
                    self.finish_download(export_id)
