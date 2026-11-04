from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
import uuid

from .config import Settings, public_base_url
from .results import checkpoint_progress, empty_summary, normalize_samples, parse_result
from .store import TaskStore


MODEL_ID = "qwen2.5-vl-7b-instruct"
DATASET_ID = "mathvista"
DATASET_CANONICAL_ID = "MathVista_MINI"
DATASET_COUNT = 1000


class ConflictError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class MathVistaService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = TaskStore(self.settings.database_path)
        self.store.mark_interrupted(utc_now())
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mathvista")
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = RLock()
        self.upstream_commit = self._git_commit()

    def _git_commit(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.settings.repo_root, text=True
            ).strip()
        except Exception:
            return "unknown"

    def _validate_dataset(self) -> None:
        from vlmeval.dataset.image_vqa import MathVista

        if DATASET_CANONICAL_ID not in MathVista.DATASET_URL:
            raise RuntimeError(
                f"Pinned VLMEvalKit does not provide required dataset {DATASET_CANONICAL_ID}"
            )

    def capabilities(self) -> dict:
        judges = []
        for judge in self.settings.judges.values():
            judges.append({
                "id": judge.id,
                "name": judge.name,
                "role": judge.role,
                "available": judge.available,
                "base_url": public_base_url(judge.base_url),
                "unavailable_reason": None if judge.available else "judge endpoint or API key is not configured",
            })
        return {
            "suite": {
                "id": "vlmevalkit-mathvista",
                "name": "VLMEvalKit MathVista",
                "version": "1.0.0",
                "upstream_commit": self.upstream_commit,
            },
            "target_model": {
                "id": MODEL_ID,
                "name": "Qwen2.5-VL-7B-Instruct",
                "transport": "openai-compatible-vllm",
                "base_url": public_base_url(self.settings.target_base_url),
            },
            "dataset": {
                "id": DATASET_ID,
                "name": "MathVista",
                "canonical_id": DATASET_CANONICAL_ID,
                "split": "testmini",
                "sample_count": DATASET_COUNT,
            },
            "judges": judges,
            "protocol": {"temperature": 0, "top_p": 1, "max_tokens": 2048},
        }

    @staticmethod
    def _public_task(task: dict) -> dict:
        return {key: value for key, value in task.items() if key not in {"work_dir", "pid", "idempotency_key"}}

    def _active_task(self) -> dict | None:
        active = self.store.active()
        return active[0] if active else None

    def create(self, payload: dict, idempotency_key: str | None) -> tuple[dict, bool]:
        reused = self.store.by_idempotency_key(idempotency_key)
        if reused is not None:
            return self._public_task(reused), True
        if self._active_task() is not None:
            raise ConflictError("another formal MathVista task is already active")
        if payload["model_id"] != MODEL_ID or payload["dataset_id"] != DATASET_ID:
            raise ValueError("only Qwen2.5-VL-7B-Instruct on MathVista is supported")
        judge = self.settings.judges.get(payload["judge_id"])
        if judge is None:
            raise ValueError(f"unsupported judge: {payload['judge_id']}")
        if not judge.available:
            raise ValueError(f"judge is not configured: {judge.id}")
        parent_task_id = payload.get("parent_task_id")
        if parent_task_id:
            parent = self.store.get(parent_task_id)
            if parent is None:
                raise ValueError("parent task does not exist")
            if judge.role != "backup" or parent["config"]["judge_role"] != "primary":
                raise ValueError("parent_task_id is only valid for a primary-to-backup full rerun")

        self._validate_dataset()
        task_id = f"mathvista-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        work_dir = self.settings.output_dir / task_id
        work_dir.mkdir(parents=True, exist_ok=False)
        config = {
            "model_id": MODEL_ID,
            "model_name": "Qwen2.5-VL-7B-Instruct",
            "dataset_id": DATASET_ID,
            "dataset_name": "MathVista",
            "dataset_canonical_id": DATASET_CANONICAL_ID,
            "dataset_md5": "f199b98e178e5a2a20e7048f5dcb0464",
            "sample_count": DATASET_COUNT,
            "judge_id": judge.id,
            "judge_name": judge.name,
            "judge_role": judge.role,
            "judge_base_url": public_base_url(judge.base_url),
            "target_base_url": public_base_url(self.settings.target_base_url),
            "upstream_commit": self.upstream_commit,
            "generation": {"temperature": 0, "top_p": 1, "max_tokens": 2048},
        }
        execution = checkpoint_progress(work_dir, DATASET_COUNT)
        execution["in_progress"] = 0
        task = {
            "task_id": task_id,
            "status": "queued",
            "quality_status": "partial",
            "created_at": now,
            "updated_at": now,
            "attempt": 1,
            "parent_task_id": parent_task_id,
            "idempotency_key": idempotency_key,
            "config": config,
            "execution_status": execution,
            "result": {"summary": empty_summary(), "quality_status": "partial"},
            "error": None,
            "work_dir": str(work_dir),
            "pid": None,
        }
        self.store.create(task)
        (work_dir / "task_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._executor.submit(self._run, task_id, False)
        return self._public_task(task), False

    def _command(self, task: dict, reuse: bool) -> tuple[list[str], dict[str, str]]:
        judge = self.settings.judges[task["config"]["judge_id"]]
        command = [
            sys.executable,
            str(self.settings.repo_root / "run.py"),
            "--data", DATASET_CANONICAL_ID,
            "--model", self.settings.target_model,
            "--base-url", self.settings.target_base_url,
            "--model-class", "LMDeployAPI",
            "--work-dir", task["work_dir"],
            "--mode", "all",
            "--api-nproc", str(self.settings.api_nproc),
            "--max-tokens", "2048",
            "--temperature", "0",
            "--top-p", "1",
            "--timeout", str(self.settings.request_timeout),
            "--judge", judge.name,
            "--judge-base-url", judge.base_url,
            "--judge-api-nproc", str(self.settings.judge_api_nproc),
            "--judge-timeout", str(self.settings.judge_timeout),
        ]
        if reuse:
            command.extend(["--reuse", "--reuse-aux", "all"])
        env = os.environ.copy()
        env["LMDEPLOY_API_KEY"] = self.settings.target_api_key
        env["OPENAI_API_KEY"] = judge.api_key
        env["VLMEVALKIT_TASK_ID"] = task["task_id"]
        return command, env

    def _run(self, task_id: str, reuse: bool) -> None:
        task = self.store.get(task_id)
        if task is None:
            return
        now = utc_now()
        self.store.update(
            task_id,
            status="running",
            updated_at=now,
            execution_status={**task["execution_status"], "in_progress": 1},
            error=None,
        )
        command, env = self._command(task, reuse)
        log_path = Path(task["work_dir"]) / "engine.log"
        try:
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.settings.repo_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                with self._lock:
                    self._processes[task_id] = process
                self.store.update(task_id, pid=process.pid)
                exit_code = process.wait()
            current = self.store.get(task_id)
            if current is None or current["status"] in {"paused", "interrupted"}:
                return
            if exit_code != 0:
                execution = checkpoint_progress(Path(task["work_dir"]), DATASET_COUNT)
                self.store.update(
                    task_id,
                    status="failed",
                    quality_status="partial" if execution["processed"] else "invalid",
                    updated_at=utc_now(),
                    execution_status={**execution, "in_progress": 0},
                    error=f"VLMEvalKit exited with code {exit_code}; see engine.log",
                    pid=None,
                )
                return
            summary, _, quality = parse_result(
                Path(task["work_dir"]), task["config"]["judge_id"], DATASET_COUNT
            )
            execution = {
                "total": DATASET_COUNT,
                "processed": summary["processed"],
                "succeeded": summary["processed"] - summary["inference_errors"] - summary["judge_errors"],
                "in_progress": 0,
                "pending": max(0, DATASET_COUNT - summary["processed"]),
                "inference_errors": summary["inference_errors"],
                "judge_errors": summary["judge_errors"],
                "progress": 100 if summary["processed"] == DATASET_COUNT else round(summary["processed"] / 10, 2),
            }
            status = "completed" if quality == "qualified" else "failed"
            self.store.update(
                task_id,
                status=status,
                quality_status=quality,
                updated_at=utc_now(),
                execution_status=execution,
                result={"summary": summary, "quality_status": quality},
                error=None if quality == "qualified" else "result did not pass the formal quality gate",
                pid=None,
            )
        except Exception as exc:
            self.store.update(
                task_id,
                status="failed",
                quality_status="invalid",
                updated_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
                pid=None,
            )
        finally:
            with self._lock:
                self._processes.pop(task_id, None)

    def get(self, task_id: str) -> dict | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        if task["status"] in {"queued", "running", "pausing"}:
            execution = checkpoint_progress(Path(task["work_dir"]), DATASET_COUNT)
            if task["status"] == "queued":
                execution["in_progress"] = 0
            self.store.update(task_id, execution_status=execution, updated_at=utc_now())
            task = self.store.get(task_id)
        return self._public_task(task)

    def list(self, limit: int = 50) -> list[dict]:
        return [self._public_task(task) for task in self.store.list(limit)]

    def pause(self, task_id: str) -> dict | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        if task["status"] not in {"queued", "running", "pausing"}:
            return self._public_task(task)
        self.store.update(task_id, status="pausing", updated_at=utc_now())
        with self._lock:
            process = self._processes.get(task_id)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        execution = checkpoint_progress(Path(task["work_dir"]), DATASET_COUNT)
        execution["in_progress"] = 0
        self.store.update(
            task_id,
            status="paused",
            quality_status="partial" if execution["processed"] else "invalid",
            updated_at=utc_now(),
            execution_status=execution,
            error=None,
            pid=None,
        )
        return self.get(task_id)

    def retry_failures(self, task_id: str) -> dict | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        if self._active_task() is not None:
            raise ConflictError("another formal MathVista task is already active")
        if task["status"] not in {"paused", "failed", "interrupted"}:
            raise ConflictError("only paused, failed, or interrupted tasks can be retried")
        self.store.update(
            task_id,
            status="queued",
            quality_status="partial",
            updated_at=utc_now(),
            attempt=task["attempt"] + 1,
            error=None,
        )
        self._executor.submit(self._run, task_id, True)
        return self.get(task_id)

    def result(self, task_id: str) -> dict | None:
        return self.get(task_id)

    def samples(self, task_id: str, offset: int, limit: int) -> dict | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        samples = normalize_samples(
            self._detail_file(Path(task["work_dir"])), task["config"]["judge_id"]
        )
        return {"items": samples[offset:offset + limit], "total": len(samples)}

    @staticmethod
    def _detail_file(work_dir: Path) -> Path | None:
        from .results import find_detail_file

        return find_detail_file(work_dir)

    def artifact(self, task_id: str) -> bytes | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        artifact = Path(task["work_dir"]) / "results.jsonl"
        if not artifact.exists():
            samples = normalize_samples(
                self._detail_file(Path(task["work_dir"])), task["config"]["judge_id"]
            )
            artifact.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples),
                encoding="utf-8",
            )
        return artifact.read_bytes()
