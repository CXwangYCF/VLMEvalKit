from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


FAIL_MSG = "Failed to obtain answer via API."


def empty_summary(total: int = 1000) -> dict:
    return {
        "total": total,
        "processed": 0,
        "correct": 0,
        "incorrect": 0,
        "overall_accuracy": None,
        "inference_errors": 0,
        "judge_errors": 0,
        "metrics": [],
    }


def _newest(paths) -> Path | None:
    candidates = [p for p in paths if p.is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def find_score_file(work_dir: Path) -> Path | None:
    return _newest(work_dir.rglob("*MathVista_MINI*score.csv"))


def find_detail_file(work_dir: Path) -> Path | None:
    candidates = []
    for path in work_dir.rglob("*MathVista_MINI*.xlsx"):
        if "score" in path.name.lower():
            continue
        try:
            columns = set(pd.read_excel(path, nrows=1).columns)
        except Exception:
            continue
        if {"index", "question", "prediction", "answer", "res", "log"}.issubset(columns):
            candidates.append(path)
    return _newest(candidates)


def _safe(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _is_correct(row: pd.Series) -> bool:
    try:
        from vlmeval.dataset.utils.mathvista import post_check

        return bool(post_check(row, prefetch=False))
    except Exception:
        return _safe(row.get("res")).strip() == _safe(row.get("answer")).strip()


def normalize_samples(detail_file: Path | None, judge_id: str) -> list[dict]:
    if detail_file is None:
        return []
    data = pd.read_excel(detail_file)
    items = []
    for _, row in data.iterrows():
        prediction = _safe(row.get("prediction"))
        log = _safe(row.get("log"))
        extracted = _safe(row.get("res"))
        if FAIL_MSG in prediction:
            status, error, correct = "inference_error", prediction, None
        elif "All 5 retries failed" in log or not extracted:
            status, error, correct = "judge_error", log or "empty extracted answer", None
        else:
            status, error, correct = "completed", None, _is_correct(row)
        items.append({
            "case_id": _safe(row.get("index")),
            "question": _safe(row.get("question")),
            "prediction": prediction,
            "extracted_answer": extracted,
            "answer": _safe(row.get("answer")),
            "correct": correct,
            "status": status,
            "judge_id": judge_id,
            "error": error,
        })
    return items


def parse_result(work_dir: Path, judge_id: str, total: int = 1000) -> tuple[dict, list[dict], str]:
    score_file = find_score_file(work_dir)
    detail_file = find_detail_file(work_dir)
    samples = normalize_samples(detail_file, judge_id)
    summary = empty_summary(total)
    summary["processed"] = len(samples)
    summary["inference_errors"] = sum(item["status"] == "inference_error" for item in samples)
    summary["judge_errors"] = sum(item["status"] == "judge_error" for item in samples)
    summary["correct"] = sum(item["correct"] is True for item in samples)
    summary["incorrect"] = sum(item["correct"] is False for item in samples)

    if score_file is not None:
        score = pd.read_csv(score_file)
        for _, row in score.iterrows():
            metric = {
                "dimension": "Task&Skill",
                "value": _safe(row.get("Task&Skill")),
                "correct": int(row.get("hit", 0)),
                "total": int(row.get("tot", 0)),
                "accuracy": float(row.get("acc", 0)),
            }
            summary["metrics"].append(metric)
            if metric["value"] == "Overall":
                summary["overall_accuracy"] = metric["accuracy"]

    artifact = work_dir / "results.jsonl"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    qualified = (
        summary["processed"] == total
        and summary["inference_errors"] == 0
        and summary["judge_errors"] == 0
        and summary["overall_accuracy"] is not None
    )
    quality = "qualified" if qualified else ("partial" if samples else "invalid")
    return summary, samples, quality


def checkpoint_progress(work_dir: Path, total: int = 1000) -> dict:
    inference_count = 0
    judge_count = 0
    for path in work_dir.rglob("*.pkl"):
        try:
            value = pd.read_pickle(path)
            count = len(value) if hasattr(value, "__len__") else 0
        except Exception:
            continue
        if "checkpoint" in path.name:
            inference_count = max(inference_count, count)
        elif "MathVista_MINI" in path.name:
            judge_count = max(judge_count, count)
    if judge_count:
        progress = 70 + 30 * min(judge_count, total) / total
    else:
        progress = 70 * min(inference_count, total) / total
    processed = max(inference_count, judge_count)
    return {
        "total": total,
        "processed": processed,
        "succeeded": max(0, processed),
        "in_progress": 1 if processed < total else 0,
        "pending": max(0, total - processed),
        "inference_errors": 0,
        "judge_errors": 0,
        "progress": round(progress, 2),
    }
