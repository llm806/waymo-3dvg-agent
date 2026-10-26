"""One local worker over the existing harness; SQLite owns state and controls."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.agent.agent_runtime import TERMINAL, initial_run, run_agent
from app.schemas.contracts import EvidencePacket
from app.datasets.evidence import EVIDENCE_ROOT, atomic_json
from app.agent.harness import RUN_ROOT
from app.agent.model_calls import QwenCalls, saved_call_diagnostic
from app.quality.relations import expand_relations
from app.infrastructure.storage import RunBusy, RunStore
from app.infrastructure.model_slots import ModelSlots


@dataclass(frozen=True)
class StartReceipt:
    run_id: str
    reused_result: bool


class ConsoleRunner:
    def __init__(self, root: Path = RUN_ROOT, calls_factory=QwenCalls, *, model_concurrency=None):
        self.root, self.calls_factory = root, calls_factory
        self.model_slots = ModelSlots(int(os.getenv('ANNOTATION_MODEL_CONCURRENCY', '4'))
            if model_concurrency is None else model_concurrency)
        self.store = RunStore(root / "runs.sqlite3")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annotation")
        self.lock = threading.Lock()
        self.active: str | None = None
        self.active_batch: str | None = None
        self.closed = False

    def start(self, packet: EvidencePacket, max_model_calls: int, max_repairs: int,
              resume_id: str | None = None) -> StartReceipt:
        if packet.provenance.get('tool_sha256'):
            from app.quality.contrast import predicate_catalog
            recorded={p.predicate_id:p.matches for p in packet.predicates}
            if any(recorded.get(p.predicate_id)!=p.matches for p in predicate_catalog(packet.candidates)):
                raise ValueError('Prepared geometric rules are obsolete; prepare the target again before starting')
        with self.store.maintenance(), self.lock:
            if self.closed or self.active or self.active_batch:
                raise RunBusy("Another annotation is executing; pause/stop it first")
            if any(self.store.executor_active(s["run_id"]) for s in self.store.all_runs()):
                raise RunBusy("Another local annotation executor holds an active lease")
            owner = uuid4().hex
            self.store.acquire_executor(owner)
            calls = None
            try:
                calls = self.calls_factory()
                scope = getattr(calls, "scope", "real-model")
                if resume_id:
                    prior = self.store.snapshot(resume_id)
                    if prior["status"] in TERMINAL or prior.get("kind") == "counterfactual_audit":
                        raise ValueError("Completed/blocked run cannot be resumed")
                    if prior["packet_id"] != packet.packet_id:
                        raise ValueError("Packet mismatch")
                    if self.store.executor_active(resume_id):
                        raise RunBusy("Executor lease remains active")
                    max_model_calls, max_repairs = prior["model_call_budget"], prior["max_repairs"]
                    cumulative_limit = prior.get("cumulative_call_limit")
                else:
                    cumulative_limit = 12
                initial = initial_run(packet, calls, max_repairs, max_model_calls, cumulative_limit)
                run_id = resume_id or initial["run_id"]
                if resume_id and run_id != initial["run_id"]:
                    raise ValueError("Code/provider/protocol changed; unsafe to resume this checkpoint")
                existing = self.store.snapshot(run_id, missing_ok=True)
                if existing and existing["status"] in TERMINAL:
                    calls.close()
                    self.store.release_executor(owner)
                    return StartReceipt(run_id, reused_result=True)
                if existing and not resume_id:
                    raise ValueError("Run exists; use its explicit resume control")
                used = self.store.target_usage(packet, scope)
                if not resume_id and cumulative_limit and used >= cumulative_limit:
                    raise ValueError("Target cumulative call budget exhausted (12)")
                if not resume_id:
                    minimum = 4 if packet.ready_for_generation else 5 if expand_relations(packet)[0].ready_for_generation else 0
                    if minimum and min(max_model_calls, 12 - used) < minimum:
                        raise ValueError(f"累计剩余 {12 - used} 次调用；此目标完成生成和双轮验证至少需 {minimum} 次（含必要的补关系决策）。")
                if resume_id:
                    self.store.request_control(run_id, "resume")
                else:
                    initial["console_managed"] = True
                    self.store.create(initial)
                self.active = run_id
                self.pool.submit(self._execute, packet, calls, max_repairs, max_model_calls, cumulative_limit, run_id, owner)
                return StartReceipt(run_id, reused_result=False)
            except Exception:
                if calls is not None:
                    calls.close()
                self.store.release_executor(owner)
                raise

    def _execute(self, packet, calls, max_repairs, max_calls, limit, run_id, owner):
        try:
            run_agent(packet, calls, max_repairs, self.root, max_calls, limit,
                resume_run_id=run_id, executor_owner=owner, model_slots=self.model_slots)
        except Exception as exc:
            # No secret-bearing exception strings in the public console.
            atomic_json(self.root / run_id / "worker-error.json", {"error_type": type(exc).__name__,
                        "message": "Worker exited; inspect durable call state before resuming"})
        finally:
            calls.close()
            self.store.release_executor(owner)
            with self.lock:
                self.active = None

    def close(self):
        with self.lock:
            self.closed = True
            if self.active:
                try:
                    self.store.request_control(self.active, "pause")
                except ValueError:
                    pass
        # Current provider call is allowed to settle; no new call starts afterwards.
        self.pool.shutdown(wait=True)


def load_packet(packet_id: str) -> EvidencePacket:
    path = EVIDENCE_ROOT / packet_id / "packet.json"
    if not path.is_file():
        raise ValueError("Evidence packet not found")
    return EvidencePacket.model_validate_json(path.read_text(encoding="utf-8"))


def call_details(root: Path, run: dict, *, include_requests=True) -> list[dict]:
    folder = root / run["run_id"]
    if run.get("kind") == "counterfactual_audit":
        folder = root / run["parent_run_id"] / "audits" / run["run_id"]
    details = []
    pattern = "*/input.json" if run.get("kind") == "counterfactual_audit" else "model/*/input.json"
    for path in sorted(folder.glob(pattern)):
        call_folder = path.parent
        request = json.loads(path.read_text(encoding="utf-8")) if include_requests else None
        # Keep the complete source request on disk; show a readable image reference.
        for message in request or []:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        part["image_url"] = {"url": "[image evidence; see current sample]"}
        result_file = call_folder / "result.json"
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
        item = {"folder": call_folder.name,
                "status": result.get("status", "reserved"),
                **{k: result[k] for k in ("answer", "reasoning_content", "usage", "elapsed_seconds", "started_at", "model_requested",
                                         "model_returned", "finish_reason", "error_type", "validation_errors", "network_error_types",
                                         "os_error_codes", "remote_outcome", "retryable", "network_diagnosis") if k in result}}
        item.update(saved_call_diagnostic(call_folder))
        if include_requests:
            item['request'] = request
        if include_requests and result.get("status") == "running":
            chunks = call_folder / "chunks.jsonl"
            if chunks.exists():
                # Read at most the last 256 KB; no growing full-stream polling cost.
                with chunks.open("rb") as f:
                    f.seek(max(0, chunks.stat().st_size - 262144))
                    lines = f.read().decode("utf-8", errors="ignore").splitlines()
                answers, reasoning = [], []
                for line in lines:
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        if text := choice.get("delta", {}).get("content"):
                            answers.append(text)
                        if text := choice.get("delta", {}).get("reasoning_content"):
                            reasoning.append(text)
                item["answer_tail"] = "".join(answers)[-12000:]
                item["reasoning_tail"] = "".join(reasoning)[-12000:]
        details.append(item)
    return details
