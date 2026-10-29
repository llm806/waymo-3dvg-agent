"""Local readiness only: no provider request, converter launch or secret disclosure."""
import hashlib
import os
import shutil
from urllib.parse import urlparse

from fastapi import APIRouter, Request

from app.datasets.catalog import load_catalog
from app.datasets.evidence import ROOT

router=APIRouter(prefix='/api/system',tags=['readiness'])


def workspace_id():
    return hashlib.sha256(str(ROOT.resolve()).casefold().encode()).hexdigest()[:24]


@router.get('/status')
def status(request: Request):
    checks=[]
    catalog=None
    try:
        catalog=load_catalog(compact=True)
        checks.append({'name':'标注物体','ready':True,'detail':f"{catalog['summary']['selected']} 个可标注物体"})
    except (FileNotFoundError,ValueError,KeyError):
        checks.append({'name':'目标目录','ready':False,'detail':'先运行 python -m scripts.build_catalog，扫描不调用模型。'})
    model=os.getenv('DASHSCOPE_MODEL','qwen3.8-flash')
    base_url=os.getenv('DASHSCOPE_BASE_URL','https://dashscope.aliyuncs.com/compatible-mode/v1')
    host=urlparse(base_url).hostname
    configured=bool(os.getenv('DASHSCOPE_API_KEY','').strip())
    from app.agent.model_calls import image_model_error
    model_error=image_model_error(model,base_url)
    checks.append({'name':'模型配置','ready':configured and model_error is None,
        'detail':model_error or (f'{model} · {host}' if configured else '在本地 .env 配置 DASHSCOPE_API_KEY，再重启服务。')})
    converter=(ROOT/'preprocessing/.venv/pyvenv.cfg').is_file() and (os.name!='nt' or shutil.which('wsl.exe') is not None)
    checks.append({'name':'官方转换环境','ready':converter,'detail':'已找到本地官方 SDK 环境；转换时校验实际执行结果。' if converter else '缺少 preprocessing/.venv 或 WSL；新场景无法转换。'})
    worker=request.app.state.console
    runs=worker.store.selection_runs()
    from app.agent.agent_runtime import VERSION
    from app.agent.selection import unavailable_target_keys
    retries = worker.store.candidate_retries()
    preparation_attempts = worker.store.preparation_attempted_keys()
    attempted, processed = unavailable_target_keys(runs, VERSION, retries=retries, preparation_attempts=preparation_attempts)
    replacement_unavailable, _ = unavailable_target_keys(runs, VERSION, include_prior_attempts=True,
        retries=retries, preparation_attempts=preparation_attempts)
    excluded = worker.store.excluded_target_keys()
    attempted |= excluded
    processed |= excluded
    replacement_unavailable |= excluded
    from app.datasets.evidence import TASK
    return {'workspace_id':workspace_id(),'annotation_ready':all(c['ready'] for c in checks),
        'supports_candidate_replacement': True,
        'supports_unlimited_batches': True,
        'supports_vision_model_checks': True,
        'annotation_method_version': VERSION, 'description_policy': TASK['description_policy'],
        'checks':checks,'provider':{'model':model,'host':host,'configured':configured},
        'processed_target_ids':[t['target_id'] for t in (catalog or {}).get('targets',[]) if (t['context'],t['object_id']) in processed],
        'attempted_target_ids':[t['target_id'] for t in (catalog or {}).get('targets',[]) if (t['context'],t['object_id']) in attempted],
        'replacement_unavailable_target_ids':[t['target_id'] for t in (catalog or {}).get('targets',[])
            if (t['context'],t['object_id']) in replacement_unavailable],
        'scope':'local configuration checks; provider access and quality are not claimed verified'}
