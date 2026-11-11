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

from .chartqa_data import DATASETS, PreparedDataset, prepare_dataset
from .config import Settings, public_base_url
from .results import checkpoint_progress, empty_summary, find_detail_file, normalize_samples, parse_result
from .store import TaskStore


MODEL_ID = "qwen2.5-vl-7b-instruct"


class ConflictError(RuntimeError):
    pass


class DatasetUnavailableError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class ChartQAService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = TaskStore(self.settings.database_path)
        self.store.mark_interrupted(utc_now())
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chartqa")
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

    def _source_path(self, dataset_id: str) -> Path:
        if dataset_id == "chartqa-test":
            return self.settings.chartqa_test_path
        if dataset_id == "chartqapro":
            return self.settings.chartqapro_path
        raise ValueError(f"unsupported dataset: {dataset_id}")

    def _prepare(self, dataset_id: str) -> PreparedDataset:
        try:
            prepared = prepare_dataset(
                dataset_id,
                self._source_path(dataset_id),
                self.settings.data_dir,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise DatasetUnavailableError(str(exc)) from exc
        self._validate_runtime_id(prepared)
        return prepared

    @staticmethod
    def _validate_runtime_id(prepared: PreparedDataset) -> None:
        if prepared.dataset_id == "chartqa-test":
            from vlmeval.dataset.image_vqa import ImageVQADataset

            supported = ImageVQADataset.DATASET_URL
        else:
            from vlmeval.dataset.chartqapro import ChartQAPro

            supported = ChartQAPro.DATASET_URL
        if prepared.runtime_id not in supported:
            raise RuntimeError(
                f"pinned VLMEvalKit does not register local dataset {prepared.runtime_id}"
            )

    def _cached_count(self, dataset_id: str) -> int | None:
        runtime_id = DATASETS[dataset_id]["runtime_id"]
        path = self.settings.data_dir / f"{runtime_id}.local.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("source_path") == str(self._source_path(dataset_id).resolve()):
                return int(payload["sample_count"])
        except (OSError, ValueError, KeyError):
            pass
        return None

    def capabilities(self) -> dict:
        datasets = []
        for dataset_id, spec in DATASETS.items():
            source_path = self._source_path(dataset_id)
            datasets.append({
                "id": dataset_id,
                "name": spec["name"],
                "canonical_id": spec["canonical_id"],
                "runtime_id": spec["runtime_id"],
                "sample_count": self._cached_count(dataset_id),
                "source": "local-parquet",
                "available": source_path.is_file(),
                "path": str(source_path),
                "unavailable_reason": None if source_path.is_file() else "local Parquet file is missing",
            })
        return {
            "suite": {
                "id": "vlmevalkit-chartqa",
                "name": "VLMEvalKit ChartQA",
                "version": "1.0.0",
                "upstream_commit": self.upstream_commit,
            },
            "target_model": {
                "id": MODEL_ID,
                "name": "Qwen2.5-VL-7B-Instruct",
                "transport": "openai-compatible-vllm",
                "base_url": public_base_url(self.settings.target_base_url),
            },
            "datasets": datasets,
            "evaluator": {
                "id": "chartqa-official-local",
                "name": "Official deterministic evaluator",
                "uses_judge": False,
            },
            "protocol": {
                "prompt_mode": "direct",
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 2048,
            },
        }

    @staticmethod
    def _public_task(task: dict) -> dict:
        return {
            key: value for key, value in task.items()
            if key not in {"work_dir", "pid", "idempotency_key"}
        }

    def _active_task(self) -> dict | None:
        active = self.store.active()
        return active[0] if active else None

    def create(self, payload: dict, idempotency_key: str | None) -> tuple[dict, bool]:
        reused = self.store.by_idempotency_key(idempotency_key)
        if reused is not None:
            return self._public_task(reused), True
        if self._active_task() is not None:
            raise ConflictError("another formal ChartQA task is already active")
        if payload.get("model_id") != MODEL_ID:
            raise ValueError("only Qwen2.5-VL-7B-Instruct is supported")
        dataset_id = payload.get("dataset_id")
        if dataset_id not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset_id}")
        if payload.get("parent_task_id"):
            raise ValueError("parent_task_id is not used by deterministic ChartQA evaluation")

        prepared = self._prepare(dataset_id)
        task_id = f"chartqa-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        work_dir = self.settings.output_dir / task_id
        work_dir.mkdir(parents=True, exist_ok=False)
        config = {
            "model_id": MODEL_ID,
            "model_name": "Qwen2.5-VL-7B-Instruct",
            "dataset_id": dataset_id,
            "dataset_name": prepared.name,
            "dataset_canonical_id": prepared.canonical_id,
            "dataset_runtime_id": prepared.runtime_id,
            "dataset_source": "local-parquet",
            "dataset_source_path": str(prepared.source_path),
            "dataset_sha256": prepared.source_sha256,
            "dataset_cache_path": str(prepared.tsv_path),
            "sample_count": prepared.sample_count,
            "evaluator_id": "chartqa-official-local",
            "uses_judge": False,
            "target_base_url": public_base_url(self.settings.target_base_url),
            "upstream_commit": self.upstream_commit,
            "generation": {"temperature": 0, "top_p": 1, "max_tokens": 2048},
        }
        execution = checkpoint_progress(work_dir, prepared.runtime_id, prepared.sample_count)
        execution["in_progress"] = 0
        task = {
            "task_id": task_id,
            "status": "queued",
            "quality_status": "partial",
            "created_at": now,
            "updated_at": now,
            "attempt": 1,
            "parent_task_id": None,
            "idempotency_key": idempotency_key,
            "config": config,
            "execution_status": execution,
            "result": {"summary": empty_summary(prepared.sample_count), "quality_status": "partial"},
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
        config = task["config"]
        command = [
            sys.executable,
            str(self.settings.repo_root / "run.py"),
            "--data", config["dataset_runtime_id"],
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
        ]
        if reuse:
            command.extend(["--reuse", "--reuse-aux", "all"])
        env = os.environ.copy()
        env["LMDEPLOY_API_KEY"] = self.settings.target_api_key
        env["LMUData"] = str(self.settings.data_dir)
        env["VLMEVAL_LOCAL_DATA_ONLY"] = "1"
        env["VLMEVALKIT_TASK_ID"] = task["task_id"]
        return command, env

    def _progress(self, task: dict) -> dict:
        return checkpoint_progress(
            Path(task["work_dir"]),
            task["config"]["dataset_runtime_id"],
            int(task["config"]["sample_count"]),
        )

    def _run(self, task_id: str, reuse: bool) -> None:
        task = self.store.get(task_id)
        if task is None:
            return
        self.store.update(
            task_id,
            status="running",
            updated_at=utc_now(),
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
                execution = self._progress(task)
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
            config = task["config"]
            summary, _, quality = parse_result(
                Path(task["work_dir"]),
                config["dataset_id"],
                config["dataset_runtime_id"],
                int(config["sample_count"]),
            )
            total = int(config["sample_count"])
            execution = {
                "total": total,
                "processed": summary["processed"],
                "succeeded": summary["processed"] - summary["inference_errors"],
                "in_progress": 0,
                "pending": max(0, total - summary["processed"]),
                "inference_errors": summary["inference_errors"],
                "progress": round(100 * summary["processed"] / total, 2) if total else 0,
            }
            self.store.update(
                task_id,
                status="completed" if quality == "qualified" else "failed",
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
            execution = self._progress(task)
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
        execution = self._progress(task)
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
            raise ConflictError("another formal ChartQA task is already active")
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
        detail = find_detail_file(
            Path(task["work_dir"]), task["config"]["dataset_runtime_id"]
        )
        samples = normalize_samples(detail, task["config"]["dataset_id"])
        return {"items": samples[offset:offset + limit], "total": len(samples)}

    def artifact(self, task_id: str) -> bytes | None:
        task = self.store.get(task_id)
        if task is None:
            return None
        artifact = Path(task["work_dir"]) / "results.jsonl"
        if not artifact.exists():
            detail = find_detail_file(
                Path(task["work_dir"]), task["config"]["dataset_runtime_id"]
            )
            samples = normalize_samples(detail, task["config"]["dataset_id"])
            artifact.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples),
                encoding="utf-8",
            )
        return artifact.read_bytes()
