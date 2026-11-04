# MathVista platform service

This branch adds a persistent, single-worker task service around the pinned
VLMEvalKit commit `986a74ea4f6d8118e0c24d6cd63d2ff09d48afc6`. It does not replace the
upstream MathVista evaluator. The service invokes `run.py` with the existing
OpenAI-compatible `--base-url` path and parses VLMEvalKit's native artifacts.

## Fixed formal protocol

- Dataset: `MathVista_MINI` (public 1,000-item testmini split, pinned MD5).
- Target: `Qwen2.5-VL-7B-Instruct` served by vLLM.
- Generation: temperature 0, top-p 1, max tokens 2048.
- Primary judge: `qwen3.8-flash`.
- Backup judge: `gpt-5.5`.
- A task locks one judge for its entire lifetime. A backup run is a new full
  task linked through `parent_task_id`; partial results are never merged.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements/service.txt
cp .env.mathvista.example .env
set -a && source .env && set +a
python -m uvicorn vlmeval_service.app:app --host 127.0.0.1 --port 8201
```

Run service tests with:

```bash
python -m unittest tests.test_mathvista_service -v
```

## API

The service exposes `/health` plus the following prefix:

```text
/api/v1/eval/capabilities
/api/v1/eval/tasks
/api/v1/eval/tasks/{task_id}
/api/v1/eval/tasks/{task_id}/pause
/api/v1/eval/tasks/{task_id}/retry-failures
/api/v1/eval/tasks/{task_id}/result
/api/v1/eval/tasks/{task_id}/samples
/api/v1/eval/tasks/{task_id}/artifacts/results.jsonl
```

Only one formal task may be active. SQLite persists configuration snapshots,
task state, attempts, parent linkage, and quality status. Native XLSX/CSV/PKL
artifacts and normalized JSONL are stored below `VLMEVAL_OUTPUT_DIR`.

## Quality gate

`qualified` requires all 1,000 samples, no inference errors, no judge errors,
an official MathVista score artifact, and one locked judge. Smoke runs and
partial/interrupted tasks remain auditable but cannot become formal results.

API keys are passed to subprocesses through environment variables and are not
included in command arguments, task snapshots, logs, or downloaded JSONL.
