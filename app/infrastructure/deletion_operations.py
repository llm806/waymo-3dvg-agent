"""Background deletion checks with bounded, pollable UI progress."""
import threading
import time
from uuid import uuid4


class DeletionOperations:
    def __init__(self, service):
        self.service = service
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False

    @staticmethod
    def snapshot(job):
        return {key:value for key,value in job.items() if key not in {'thread','request','last_report'}}

    def status(self, operation_id):
        with self.lock:
            return self.snapshot(self.jobs[operation_id])

    def create(self, action, job_ids, token=None, return_to_pool=None):
        request = (action,tuple(sorted(set(job_ids))),token,return_to_pool)
        with self.lock:
            if self.closed:
                raise ValueError('服务正在关闭，请稍后重试。')
            for job in self.jobs.values():
                if job['status'] == 'running':
                    if job['request'] == request:
                        return self.snapshot(job)
                    raise ValueError('已有删除核对或清理正在执行，请等待当前操作结束。')
            # Keep only small completed public results, never filesystem plans.
            now = time.time()
            self.jobs = {key:job for key,job in self.jobs.items() if now-job['updated_at'] < 3600}
            while len(self.jobs) >= 32:
                del self.jobs[next(iter(self.jobs))]
            job = dict(operation_id=uuid4().hex[:24],action=action,status='running',stage='waiting',
                       message='正在准备核对' if action == 'preview' else '正在准备删除',
                       completed=0,total=0,files_checked=0,created_at=now,updated_at=now,
                       result=None,request=request,last_report=0)
            thread = threading.Thread(target=self._run,args=(job,),daemon=True,name='task-deletion-'+job['operation_id'])
            job['thread'] = thread
            self.jobs[job['operation_id']] = job
            try:
                thread.start()
            except RuntimeError:
                del self.jobs[job['operation_id']]
                raise
            return self.snapshot(job)

    def _run(self, job):
        def report(**values):
            now = time.monotonic()
            with self.lock:
                if (values.get('stage',job['stage']) == job['stage'] and now-job['last_report'] < .1
                        and 'database_write_seconds' not in values and values.get('completed') != values.get('total')):
                    return
                job.update(values,updated_at=time.time(),last_report=now)
        try:
            action,job_ids,token,return_to_pool = job['request']
            if action == 'preview':
                result = self.service.deletion_preview(list(job_ids),progress=report)
            elif action == 'delete':
                result = self.service.delete(list(job_ids),token,return_to_pool,progress=report)
            else:
                raise ValueError('无效的删除操作。')
            with self.lock:
                job.update(status='completed',stage='finished',completed=1,total=1,result=result,
                           message='核对完成' if action == 'preview' else '删除处理完成',updated_at=time.time())
        except Exception as exc:
            with self.lock:
                job.update(status='failed',stage='finished',message=str(exc),updated_at=time.time())

    def close(self):
        with self.lock:
            self.closed = True
            threads = [job['thread'] for job in self.jobs.values()]
        # Let the durable transaction finish/restore before closing its store.
        for thread in threads:
            thread.join()
