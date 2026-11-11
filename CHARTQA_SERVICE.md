# ChartQA local task service

This branch serves `ChartQA_TEST` and `ChartQAPro` with the fixed
`Qwen2.5-VL-7B-Instruct` target model. Both datasets use their deterministic
VLMEvalKit evaluators; no LLM judge is called.

## Local data

The service never downloads either dataset. Place the two source files at the
paths configured by `VLMEVAL_CHARTQA_TEST_PATH` and
`VLMEVAL_CHARTQAPRO_PATH`. The default repository-relative layout is:

```text
data/ChartQA_TEST/test-00000-of-00001.parquet
data/ChartQAPro/test-00000-of-00001.parquet
```

The local snapshot currently supplied in this workspace is pinned as:

| Dataset | Rows | SHA256 |
|---|---:|---|
| ChartQA_TEST | 2,500 | `84889bae30aaf6e24ff899e1604221606c0b65e658b6c70a7d02c79f50fc87df` |
| ChartQAPro | 1,948 | `feb03a5579e49114b45350b75aed7f674e285f795a425c83e22f400095c91489` |

At task creation the source is validated, SHA256 hashed and converted to a
local-only TSV cache in `LMUData`. The source hash and actual row count are
locked in `task_config.json`. Data, generated caches, model weights, keys,
SQLite files and task artifacts must remain outside Git.

## Start

```bash
python -m pip install -r requirements/service.txt
cp .env.chartqa.example /etc/safetyllm/vlmevalkit-chartqa.env
python -m uvicorn vlmeval_service.app:app --host 127.0.0.1 --port 8203
```

Start vLLM separately on port `8002` and expose the served model name configured
by `VLMEVAL_TARGET_MODEL`. Check the service with:

```bash
curl http://127.0.0.1:8203/health
curl http://127.0.0.1:8203/api/v1/eval/capabilities
```

Create a task using `dataset_id=chartqa-test` or `dataset_id=chartqapro`:

```json
{
  "model_id": "qwen2.5-vl-7b-instruct",
  "dataset_id": "chartqapro",
  "parent_task_id": null
}
```

The task API exposes create/list/get, pause, retry-failures, result, paginated
samples and `results.jsonl`. A result is `qualified` only when every source row
has a prediction, inference errors are zero and the official score artifact is
present. Run a 3-10 sample smoke outside the formal result directory before a
full evaluation.
