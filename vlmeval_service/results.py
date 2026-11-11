from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


FAIL_MSG = "Failed to obtain answer via API."


def empty_summary(total: int = 0) -> dict:
    return {
        "total": total,
        "processed": 0,
        "correct": 0,
        "incorrect": 0,
        "overall_accuracy": None,
        "inference_errors": 0,
        "metrics": [],
    }


def _newest(paths) -> Path | None:
    candidates = [path for path in paths if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def find_score_file(work_dir: Path, runtime_id: str) -> Path | None:
    candidates = []
    for path in work_dir.rglob(f"*{runtime_id}*"):
        lower = path.name.lower()
        if path.is_file() and ("_acc" in lower or "score" in lower) and path.suffix.lower() in {".csv", ".xlsx"}:
            candidates.append(path)
    return _newest(candidates)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".pkl":
        value = pd.read_pickle(path)
        if isinstance(value, pd.DataFrame):
            return value
    raise ValueError(f"unsupported result table: {path}")


def find_detail_file(work_dir: Path, runtime_id: str) -> Path | None:
    candidates = []
    for path in work_dir.rglob(f"*{runtime_id}*"):
        if not path.is_file() or path.suffix.lower() not in {".xlsx", ".csv", ".pkl"}:
            continue
        lower = path.name.lower()
        if any(tag in lower for tag in ("_acc", "score", "checkpoint", "_prev")):
            continue
        try:
            columns = set(_read_table(path).columns)
        except Exception:
            continue
        if {"index", "question", "prediction", "answer"}.issubset(columns):
            candidates.append(path)
    preferred = [path for path in candidates if "_results" in path.name.lower()]
    return _newest(preferred or candidates)


def _safe(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _list(value: Any) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return [value]
    return parsed if isinstance(parsed, list) else [parsed]


def _chartqa_test_score(row: pd.Series, prediction: str) -> float:
    if row.get("eval_score") is not None and not pd.isna(row.get("eval_score")):
        return float(row.get("eval_score"))
    from vlmeval.dataset.utils.vqa_eval import relaxed_correctness

    answers = _list(row.get("answer"))
    return float(any(relaxed_correctness(str(answer), prediction) for answer in answers))


def _chartqapro_score(row: pd.Series, prediction: str) -> float:
    from vlmeval.dataset.utils.chartqapro import relaxed_correctness_chartqapro

    answers = _list(row.get("answer"))
    years = _list(row.get("year"))
    if _safe(row.get("question_type")) == "Conversational":
        years = years[-1:]
    target = str(answers[-1]).strip(".").strip()
    return float(relaxed_correctness_chartqapro(
        target,
        prediction.strip(".").strip(),
        year_flags=years,
    ))


def normalize_samples(detail_file: Path | None, dataset_id: str) -> list[dict]:
    if detail_file is None:
        return []
    data = _read_table(detail_file)
    items = []
    for _, row in data.iterrows():
        prediction = _safe(row.get("prediction"))
        if FAIL_MSG in prediction or not prediction.strip():
            status, error, score = "inference_error", prediction or "empty prediction", None
        else:
            status, error = "completed", None
            score = (
                _chartqa_test_score(row, prediction)
                if dataset_id == "chartqa-test"
                else _chartqapro_score(row, prediction)
            )
        item = {
            "case_id": _safe(row.get("index")),
            "question": _safe(row.get("question")),
            "prediction": prediction,
            "answer": _safe(row.get("answer")),
            "score": score,
            "correct": None if score is None else bool(math.isclose(score, 1.0)),
            "status": status,
            "error": error,
        }
        if dataset_id == "chartqapro":
            item["question_type"] = _safe(row.get("question_type"))
        elif row.get("split") is not None and not pd.isna(row.get("split")):
            item["split"] = _safe(row.get("split"))
        items.append(item)
    return items


def parse_result(
    work_dir: Path,
    dataset_id: str,
    runtime_id: str,
    total: int,
) -> tuple[dict, list[dict], str]:
    score_file = find_score_file(work_dir, runtime_id)
    detail_file = find_detail_file(work_dir, runtime_id)
    samples = normalize_samples(detail_file, dataset_id)
    summary = empty_summary(total)
    summary["processed"] = len(samples)
    summary["inference_errors"] = sum(item["status"] == "inference_error" for item in samples)
    completed = [item for item in samples if item["score"] is not None]
    summary["correct"] = sum(item["correct"] is True for item in completed)
    summary["incorrect"] = sum(item["correct"] is False for item in completed)

    groups: dict[str, list[float]] = defaultdict(list)
    for item in completed:
        groups["Overall"].append(float(item["score"]))
        if dataset_id == "chartqapro" and item.get("question_type"):
            groups[item["question_type"]].append(float(item["score"]))
        elif item.get("split"):
            groups[item["split"]].append(float(item["score"]))
    for name, scores in groups.items():
        metric = {
            "dimension": "Question Type" if dataset_id == "chartqapro" else "Split",
            "value": name,
            "total": len(scores),
            "score_sum": round(sum(scores), 6),
            "accuracy": round(100 * sum(scores) / len(scores), 4),
        }
        summary["metrics"].append(metric)
        if name == "Overall":
            summary["overall_accuracy"] = metric["accuracy"]

    artifact = work_dir / "results.jsonl"
    with artifact.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    qualified = (
        summary["processed"] == total
        and summary["inference_errors"] == 0
        and summary["overall_accuracy"] is not None
        and score_file is not None
    )
    quality = "qualified" if qualified else ("partial" if samples else "invalid")
    return summary, samples, quality


def checkpoint_progress(work_dir: Path, runtime_id: str, total: int) -> dict:
    processed = 0
    for path in work_dir.rglob("*.pkl"):
        try:
            value = pd.read_pickle(path)
            count = len(value) if hasattr(value, "__len__") else 0
        except Exception:
            continue
        if "checkpoint" in path.name.lower() or runtime_id.lower() in path.name.lower():
            processed = max(processed, count)
    return {
        "total": total,
        "processed": processed,
        "succeeded": processed,
        "in_progress": 1 if processed < total else 0,
        "pending": max(0, total - processed),
        "inference_errors": 0,
        "progress": round(100 * min(processed, total) / total, 2) if total else 0,
    }
