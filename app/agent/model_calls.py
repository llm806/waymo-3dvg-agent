"""Bounded compatible SDK calls with durable request/response records."""

import hashlib
import copy
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from importlib.metadata import version
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, ValidationError

from app.datasets.evidence import atomic_json
from app.schemas.contracts import SceneVisualObservation
from app.agent.observation_wire import unknown_optional_evidence, wire_response_type


class ModelCallFailure(RuntimeError):
    pass


TRANSIENT_RETRY_DELAYS = (2.0, 5.0, 15.0)
CALL_WALL_SECONDS = 300
STREAM_LOG_FLUSH_SECONDS = 0.25
STREAM_LOG_FLUSH_CHARS = 65536


def transient_retry_delay(failure, attempt):
    """Retry explicit temporary HTTP failures, never a partly received stream."""
    if (failure.get('status') != 'api_failed'
            or failure.get('http_status') not in {408, 409, 429, 500, 502, 503, 504}
            or failure.get('response_id') or failure.get('stream_chunk_count')
            or failure.get('answer') or attempt >= len(TRANSIENT_RETRY_DELAYS)):
        return None
    return max(TRANSIENT_RETRY_DELAYS[attempt], min(300, max(0, failure.get('retry_after_seconds', 0))))


def release_auth_rejection(state, failure):
    """Release only a proven pre-inference rejection; never retry it here.

    Keep the request/response and a durable release marker. Interrupted streams,
    malformed answers and uncertain network outcomes still consume their slots.
    """
    pending = state.get('inflight')
    if (not pending or pending.get('reservation_released')
            or failure.get('status') != 'api_failed'
            or failure.get('http_status') != 401
            or failure.get('provider_error_code') != 'invalid_api_key'
            or failure.get('stream_chunk_count') != 0
            or any(failure.get(k) for k in ('response_id', 'answer', 'reasoning_content', 'usage'))
            or state.get('model_calls_used', 0) < 1):
        return False
    state['model_calls_used'] -= 1
    state['auth_rejections'] = state.get('auth_rejections', 0) + 1
    pending['reservation_released'] = True
    return True


def image_model_error(model, base_url):
    """Reject documented text-only DashScope models, not entire model families.

    Provider-specific: DeepSeek's own newer endpoints have different image
    support. New/unknown models are not declared text-only from their names.
    https://help.aliyun.com/zh/model-studio/deepseek-v4-flash
    """
    host = (urlparse(base_url).hostname or '').lower()
    dashscope = (host in {'dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com', 'dashscope-us.aliyuncs.com'}
        or host.endswith('.dashscope.aliyuncs.com')
        or host.endswith('.maas.aliyuncs.com'))
    text_only = re.fullmatch(r'deepseek-v4-(?:flash(?:-0731)?|pro(?:-0813)?)(?:-us)?', model)
    if dashscope and text_only:
        return (f'{model} 在当前百炼接入仅支持文本，不能用于图像标注。'
            '请使用支持图像输入的模型，例如 qwen3.7-flash-2026-07-15。')
    return None


def validated_response(answer, output_type, expected_candidate_ids=None):
    """Validate IDs/structure strictly, with conservative optional observations.

    Never repair JSON, fill identity/visibility fields or drop text claims.
    Unsupported optional observations become unknown, with an explicit audit
    trail. The caller retains the exact original answer.
    """
    normalization = None
    try:
        parsed = output_type.model_validate_json(answer)
    except ValidationError as validation_error:
        try:
            value = json.loads(answer)
        except (ValueError, TypeError):
            raise validation_error
        if (isinstance(value, dict) and set(value) == {'output'}
                and isinstance(value['output'], dict)
                and 'output' not in output_type.model_fields):
            value = value['output']
            normalization = 'single-output-wrapper-v1'
            try:
                output_type.model_validate(value)
            except ValidationError as inner_error:
                validation_error = inner_error
        value, changes = unknown_optional_evidence(value, output_type)
        if not normalization and not changes:
            raise validation_error
        try:
            parsed = output_type.model_validate(value)
        except ValidationError:
            # Diagnostics must refer to the original answer's field indexes.
            raise validation_error from None
        if changes:
            normalization = {'policy': 'unknown-optional-observation-v1',
                'transport': normalization, 'discarded_evidence': changes}
    if expected_candidate_ids is not None and isinstance(parsed, SceneVisualObservation):
        ids = [c.candidate_id for c in parsed.candidates]
        if len(ids) != len(set(ids)) or set(ids) != set(expected_candidate_ids):
            missing = sorted(set(expected_candidate_ids) - set(ids))
            extra = sorted(set(ids) - set(expected_candidate_ids))
            error = ValueError(f'Return exactly one record per supplied ID. Missing: {missing}; unexpected: {extra}; duplicate IDs: {len(ids) != len(set(ids))}.')
            raise ValidationError.from_exception_data('SceneVisualObservation', [
                {'type': 'value_error', 'loc': ('candidates',), 'input': None, 'ctx': {'error': error}}])
    return parsed, normalization


def validation_diagnostics(exc, answer):
    """Add response field values and candidate IDs to bounded retry feedback."""
    try:
        root = json.loads(answer)
    except (ValueError, TypeError):
        root = None
    if isinstance(root, dict) and set(root) == {'output'}:
        root = root['output']
    errors = []
    for error in exc.errors(include_input=False, include_url=False):
        detail = {'location': list(error['loc']), 'type': error['type'], 'message': error['msg']}
        value = root
        for key in error['loc']:
            if isinstance(value, dict) and isinstance(value.get('id', value.get('candidate_id')), str):
                detail['candidate_id'] = value.get('id', value.get('candidate_id'))[:120]
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
                value = value[key]
            else:
                value = None
                break
        if isinstance(value, str):
            detail['received_value'] = value[:120]
        errors.append(detail)
    return errors


def connection_diagnostic(exc: Exception, response_received: bool = False) -> dict:
    """Record types/codes only: exception messages may contain keys or URLs."""
    chain, seen, current = [], set(), exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = [type(e).__name__ for e in chain]
    codes = [{"errno": getattr(e, "errno", None), "winerror": getattr(e, "winerror", None)}
             for e in chain if isinstance(e, OSError)]
    not_sent = (not response_received and any(n in {"ConnectError", "ConnectTimeout", "PoolTimeout"} for n in names)
        and not any(n in {'ReadError', 'ReadTimeout', 'WriteError', 'WriteTimeout', 'RemoteProtocolError'} for n in names))
    denied = any(isinstance(e, PermissionError) or getattr(e, "errno", None) in {1, 13}
                 or getattr(e, "winerror", None) == 10013 for e in chain)
    # Classify a small allowlist locally; never persist arbitrary exception text.
    text = ' '.join(str(e).lower() for e in chain)
    cause = next((label for label, tokens in (
        ('tls_certificate_rejected', ('certificate_verify_failed', 'certificate verify failed')),
        ('dns_resolution_failed', ('getaddrinfo failed', 'name or service not known', 'name resolution')),
        ('connection_refused', ('connection refused', 'actively refused')),
        ('connection_reset', ('connection reset', 'forcibly closed')),
        ('tls_handshake_failed', ('ssl', 'tls', 'unexpected eof')),
    ) if any(token in text for token in tokens)),
        'connect_timeout' if 'ConnectTimeout' in names else 'unspecified_transport_error')
    return {"network_error_types": names, "os_error_codes": codes,
            "remote_outcome": "not_sent" if not_sent else "unknown",
            "retryable": not_sent and not denied and cause != 'tls_certificate_rejected',
            "network_failure_kind": cause,
            "network_diagnosis": "local_network_access_denied" if denied else "connection_not_established" if not_sent else "remote_outcome_unknown"}


def format_retry_messages(messages, output_type, errors):
    """Keep the same task/allowed actions, adding only a bounded format correction."""
    updated = copy.deepcopy(messages)
    user = updated[-1]
    text = user["content"] if isinstance(user["content"], str) else user["content"][0]["text"]
    payload = json.loads(text)
    # Phase inputs may restrict candidate IDs, counts and tool choices. Preserve
    # their exact schema instead of replacing it with an unconstrained default.
    if "output_schema" not in payload:
        payload["output_schema"] = wire_response_type(messages, output_type).model_json_schema()
    keys = list(payload['output_schema'].get('properties', {}))
    payload["format_feedback"] = {"errors": errors, 'allowed_top_level_keys': keys,
        "instruction": 'Return the result JSON object itself at the root. '
        'Do not wrap it in output, result or data; do not echo output_schema, candidate_ids, '
        'format_feedback or type unless that key is explicitly allowed. '
        'Keep the original task, observations, IDs and allowed actions. Every supplied candidate '
        'needs its own record and required ID/visibility fields. Do not merge records or invent values. '
        'Shorten explanation strings to the schema limits; visibility_reason/why is at most 300 characters. '
        'Use only the allowed values for each attribute KIND, not values belonging to another kind. '
        'For observations, omit unsupported or uncertain attributes; car/vehicle are categories, '
        'not supported subtypes. Use JSON null for unknown paint, never the string "null". '
        'For generated text or grounding, do not hide unsupported claims by omitting their parsed attributes; '
        'correct the expression or report the unsupported claim.'}
    encoded = json.dumps(payload)
    if isinstance(user["content"], str):
        user["content"] = encoded
    else:
        user["content"][0]["text"] = encoded
    return updated


def has_json_instruction(messages):
    texts = []
    for message in messages:
        if message.get("role") not in {"system", "user"}:
            continue
        content = message.get("content", "")
        texts.extend([content] if isinstance(content, str) else
                     [part.get("text", "") for part in content if part.get("type") == "text"])
    return any(re.search(r"\bjson\b", text, re.IGNORECASE) for text in texts)


def json_mode_messages(messages):
    """Keep wire requests valid without mutating durable phase input messages."""
    updated = copy.deepcopy(messages)
    if not has_json_instruction(updated):
        instruction = "Return only a valid JSON object matching the supplied output schema."
        if updated and updated[0].get("role") == "system" and isinstance(updated[0].get("content"), str):
            updated[0]["content"] += "\n" + instruction
        else:
            updated.insert(0, {"role": "system", "content": instruction})
    return updated


def provider_diagnostic(error):
    """Known safe causes only; arbitrary provider messages can echo private input."""
    message = str(error.get("message", error.get('Message', ''))).lower()
    code = re.sub('[^a-z]', '', str(error.get('code', error.get('Code', ''))).lower())
    if any(x in code for x in ('arrearage', 'insufficientbalance', 'insufficientquota', 'quotaexhausted', 'accessdenied', 'invalidapikey')):
        return {'provider_diagnosis':'provider_account_rejected'}
    if any(x in code for x in ('modelnotfound', 'modelnotexist')):
        return {'provider_diagnosis':'provider_model_rejected'}
    if "messages" in message and "json" in message and "must contain" in message:
        return {"provider_diagnosis": "json_mode_instruction_missing", "provider_parameter": "messages"}
    if "image" in message and any(x in message for x in ("dimension", "size", "width", "height", "aspect ratio")):
        return {"provider_diagnosis": "image_input_invalid"}
    if "enable_thinking" in message and any(x in message for x in ("restricted", "not support", "only support")):
        return {"provider_diagnosis": "thinking_mode_unsupported", "provider_parameter": "enable_thinking"}
    if 'response_format' in message and any(x in message for x in ('not support', 'unsupported', 'not allowed')):
        return {'provider_diagnosis':'response_format_unsupported', 'provider_parameter':'response_format'}
    if any(x in message for x in ('insufficient balance', 'quota exhausted', 'insufficient quota', 'arrearage', 'access denied')):
        return {'provider_diagnosis':'provider_account_rejected'}
    if 'model' in message and any(x in message for x in ('does not exist', 'not found', 'not supported', 'not authorized')):
        return {'provider_diagnosis':'provider_model_rejected'}
    if ('image' in message and any(x in message for x in ('too many', 'number of images', 'image count'))):
        return {'provider_diagnosis':'image_count_rejected'}
    if any(x in message for x in ('context length', 'maximum context', 'input length', 'too many tokens')):
        return {'provider_diagnosis':'input_length_rejected'}
    return {}


def request_repair_option(failure, options):
    """Only explicitly rejected optional transport fields are safe to omit.

    Each omission can happen once per run, in a NEW accounted call. Never
    change images, schema, evidence or retry a partly received/unknown request.
    """
    if (failure.get('status') != 'api_failed' or failure.get('http_status') != 400
            or failure.get('stream_chunk_count') != 0
            or any(failure.get(k) for k in ('response_id','answer','reasoning_content','usage'))):
        return None
    option = {'thinking_mode_unsupported':'enable_thinking',
        'response_format_unsupported':'response_format'}.get(failure.get('provider_diagnosis'))
    return option if option and options.get(option) is not False else None


def isolated_request_failure(failure):
    """One rejected input is local; known auth/config errors remain shared."""
    return (failure.get('http_status') in {400, 413, 422}
        and failure.get('provider_diagnosis') not in {'json_mode_instruction_missing',
            'thinking_mode_unsupported', 'response_format_unsupported',
            'provider_account_rejected', 'provider_model_rejected'}
        and failure.get('remote_outcome') != 'unknown'
        and not failure.get('provider_response_received')
        and not any(failure.get(k) for k in ('response_id', 'stream_chunk_count', 'answer')))


def saved_call_diagnostic(folder):
    """Read historical evidence; derived diagnoses never overwrite the original."""
    result_file = folder / "result.json"
    if not result_file.exists():
        return {}
    result = json.loads(result_file.read_text(encoding="utf-8"))
    detail = {k: result[k] for k in ("http_status", "provider_error_code", "provider_diagnosis", "provider_parameter",
        "finish_reason", "failure_reason", "error_type", "network_diagnosis", "network_failure_kind",
        "remote_outcome", "retryable", "network_error_types", "os_error_codes", "validation_errors",
        "retry_after_seconds", 'provider_message_sha256', 'provider_message_chars') if k in result}
    if result.get('status') == 'running':
        detail.update(remote_outcome='unknown', retryable=False)
    if (result.get('status') == 'api_failed' and result.get('http_status') in {400,413,422}
            and not detail.get('provider_diagnosis')):
        detail['provider_diagnosis'] = 'request_rejected'
    detail['temporary_provider_failure'] = transient_retry_delay(result, 0) is not None
    detail['provider_response_received'] = bool(any(result.get(k)
        for k in ('response_id','stream_chunk_count','answer','reasoning_content')))
    request_file = folder / "request.json"
    if result.get("status") == "api_failed" and result.get("http_status") == 400 and detail.get("provider_diagnosis") == 'request_rejected' and request_file.exists():
        request = json.loads(request_file.read_text(encoding="utf-8"))
        if request.get("response_format", {}).get("type") == "json_object" and not has_json_instruction(request.get("messages", [])):
            detail.update(provider_diagnosis="json_mode_instruction_missing", provider_parameter="messages",
                          diagnosis_source="saved_request_contract_check")
    return detail


class QwenCalls:
    def __init__(self):
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise ValueError("DASHSCOPE_API_KEY missing; use uv run --env-file .env")
        self.model = os.getenv("DASHSCOPE_MODEL", "qwen3.8-flash")
        self.base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        error = image_model_error(self.model, self.base_url)
        if error:
            raise ValueError(error)
        self.client = OpenAI(api_key=key, base_url=self.base_url, timeout=120, max_retries=0)

    def close(self):
        self.client.close()

    def call(self, messages: list[dict], output_type: type[BaseModel], folder: Path, heartbeat=None,
             request_options=None) -> BaseModel:
        error = image_model_error(self.model, self.base_url)
        if error:
            raise ValueError(error)
        folder.mkdir(parents=True, exist_ok=True)
        wire_type = wire_response_type(messages, output_type)
        expected_ids = None
        if output_type is SceneVisualObservation:
            content = messages[-1]['content']
            try:
                expected_ids = json.loads(content if isinstance(content, str) else content[0]['text']).get('candidate_ids')
            except (ValueError, TypeError):
                pass
        def parse(answer):
            value, normalization = validated_response(answer, wire_type, expected_ids)
            return (value if wire_type is output_type else output_type.model_validate(value.model_dump())), normalization
        wire_messages = json_mode_messages(messages)
        wire_messages.insert(0, {'role': 'system', 'content':
            'Return only the result JSON object at the root, without an output/result/data wrapper '
            'or echoed input/schema. Allowed root keys: ' + ', '.join(wire_type.model_fields[f].alias or f for f in wire_type.model_fields)
            + '. Follow the supplied schema and candidate IDs/count exactly. Keep visibility_reason/why under 300 characters.'})
        request = {"model": self.model, "messages": wire_messages, "stream": True,
                   "stream_options": {"include_usage": True}, "extra_body": {"enable_thinking": False},
                   "response_format": {"type": "json_object"},
                   "max_tokens": 16384 if output_type is SceneVisualObservation else 2048}
        if (request_options or {}).get('enable_thinking') is False:
            request.pop('extra_body')
        if (request_options or {}).get('response_format') is False:
            request.pop('response_format')
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        result_path = folder / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous["request_sha256"] != request_hash:
                raise RuntimeError("Checkpoint request mismatch")
            if previous["status"] == "completed":
                return parse(previous["answer"])[0]
            if previous["status"] == "running":
                # Remote outcome is unknown. Never silently replay and double-bill.
                raise ModelCallFailure('interrupted_response')
            raise ModelCallFailure(previous["status"])
        atomic_json(folder / "request.json", request)
        record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                  "request_sha256": request_hash, "model_requested": self.model, "base_url": self.base_url,
                  "python": platform.python_version(), "sdk_version": version("openai")}
        parts = [part for message in wire_messages if isinstance(message.get('content'), list)
            for part in message['content']]
        record['request_shape'] = {'image_count':sum(p.get('type') == 'image_url' for p in parts),
            'output_type':output_type.__name__, 'max_tokens':request['max_tokens']}
        atomic_json(result_path, record)
        started = time.perf_counter()
        answer, reasoning = [], []
        chunk_count, log_seconds, heartbeat_seconds = 0, 0.0, 0.0
        parsed = None
        try:
            with self.client.chat.completions.create(**request) as stream:
                with (folder / "chunks.jsonl").open("w", encoding="utf-8") as log:
                    last_flush, pending_chars = time.perf_counter(), 0
                    for chunk in stream:
                        now = time.perf_counter()
                        if now - started > CALL_WALL_SECONDS:
                            raise ModelCallFailure('response_deadline_exceeded')
                        record.setdefault('first_chunk_seconds', round(now - started, 3))
                        chunk_count += 1
                        if heartbeat is not None:
                            before_heartbeat = time.perf_counter()
                            heartbeat()
                            heartbeat_seconds += time.perf_counter() - before_heartbeat
                        before_log = time.perf_counter()
                        line = chunk.model_dump_json() + "\n"
                        log.write(line)
                        pending_chars += len(line)
                        # Keep every raw chunk, but coalesce costly Windows
                        # flushes. Context exit also flushes the final tail on
                        # completion or an exception. A killed process still
                        # leaves a running request that must never be replayed.
                        if (now - last_flush >= STREAM_LOG_FLUSH_SECONDS
                                or pending_chars >= STREAM_LOG_FLUSH_CHARS
                                or chunk.usage or any(c.finish_reason for c in chunk.choices)):
                            log.flush()
                            last_flush, pending_chars = time.perf_counter(), 0
                        log_seconds += time.perf_counter() - before_log
                        record.update(response_id=chunk.id, model_returned=chunk.model)
                        if chunk.usage:
                            record["usage"] = chunk.usage.model_dump()
                        for choice in chunk.choices:
                            if choice.finish_reason:
                                record["finish_reason"] = choice.finish_reason
                            if choice.delta.content:
                                record.setdefault('first_content_seconds', round(time.perf_counter() - started, 3))
                                answer.append(choice.delta.content)
                            thought = getattr(choice.delta, "reasoning_content", None)
                            if thought:
                                reasoning.append(thought)
            record["answer"] = "".join(answer)
            record["reasoning_content"] = "".join(reasoning)
            if record.get("finish_reason") != "stop":
                raise ModelCallFailure("incomplete_response")
            parsed, normalization = parse(record["answer"])
            if normalization:
                record['response_normalization'] = normalization
            record["status"] = "completed"
        except ValidationError as exc:
            record.update(status="invalid_output", error_type=type(exc).__name__,
                          validation_errors=validation_diagnostics(exc, record.get('answer')))
        except ModelCallFailure as exc:
            record.update(status="invalid_output", error_type=type(exc).__name__, failure_reason=str(exc))
            if str(exc) == 'response_deadline_exceeded':
                record.update(status='connection_failed', remote_outcome='unknown', retryable=False,
                    network_diagnosis='response_deadline_exceeded')
        except APIStatusError as exc:
            record.update(status="api_failed", error_type=type(exc).__name__,
                          http_status=exc.status_code, request_id=exc.request_id)
            retry_after = exc.response.headers.get('retry-after')
            if retry_after:
                try:
                    seconds = float(retry_after)
                except ValueError:
                    try:
                        seconds = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        seconds = 0
                if 0 <= seconds < float('inf'):
                    record['retry_after_seconds'] = min(300, seconds)
            # Preserve safe structured diagnostics, never exception text or
            # arbitrary provider messages that can echo credentials/requests.
            body = exc.body
            error = body.get('error', body) if isinstance(body, dict) else {'message':str(body or '')}
            if isinstance(error, str):
                error = {'message':error}
            if isinstance(error, dict):
                code = error.get('code', error.get('Code'))
                if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", code):
                    record["provider_error_code"] = code
                record.update(provider_diagnostic(error))
                if isinstance(error.get('param'), str) and error['param'] in {'messages','images','image_url','max_tokens','model','response_format','enable_thinking'}:
                    record['provider_parameter'] = error['param']
                message = str(error.get('message', error.get('Message', '')))
                record.update(provider_message_sha256=hashlib.sha256(message.encode('utf-8', errors='replace')).hexdigest(),
                    provider_message_chars=len(message))
            if exc.status_code in {400,413,422}:
                record.setdefault('provider_diagnosis', 'request_rejected')
        except APIConnectionError as exc:
            record.update(status="connection_failed", error_type=type(exc).__name__,
                          **connection_diagnostic(exc, bool(record.get("response_id"))))
        except Exception as exc:
            # Stream read/write failures can be raised directly by the underlying
            # HTTPX transport. They have an unknown outcome and must not be replayed.
            if not any(c.__module__.startswith("httpx") for c in type(exc).__mro__):
                raise
            record.update(status="connection_failed", error_type=type(exc).__name__,
                          **connection_diagnostic(exc, bool(record.get("response_id"))))
        finally:
            record.setdefault("answer", "".join(answer))
            record.setdefault("reasoning_content", "".join(reasoning))
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            record.update(stream_chunk_count=chunk_count, stream_log_seconds=round(log_seconds, 3),
                          heartbeat_seconds=round(heartbeat_seconds, 3))
            atomic_json(result_path, record)
        if parsed is None:
            raise ModelCallFailure(record["status"])
        return parsed
