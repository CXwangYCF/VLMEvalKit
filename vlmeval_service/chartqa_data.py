from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    name: str
    canonical_id: str
    runtime_id: str
    source_path: Path
    source_sha256: str
    sample_count: int
    tsv_path: Path


DATASETS = {
    "chartqa-test": {
        "name": "ChartQA_TEST",
        "canonical_id": "ChartQA_TEST",
        "runtime_id": "ChartQA_TEST_LOCAL",
    },
    "chartqapro": {
        "name": "ChartQAPro",
        "canonical_id": "ChartQAPro",
        "runtime_id": "ChartQAPro_LOCAL",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return repr(value) if isinstance(value, list) else value


def _image_bytes(value: Any, index: int) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"missing binary image at row {index}")
    return value


def _normalize(dataset_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("dataset is empty")

    if dataset_id == "chartqa-test":
        required = {"query", "label", "image"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"ChartQA_TEST missing columns: {sorted(missing)}")
        output = pd.DataFrame({
            "index": [str(i) for i in range(len(frame))],
            "question": frame["query"],
            "answer": frame["label"].map(_as_scalar),
            "image": [
                base64.b64encode(_image_bytes(value, i)).decode("ascii")
                for i, value in enumerate(frame["image"])
            ],
        })
        if "type" in frame:
            output["split"] = frame["type"].map(_as_scalar)
        return output

    required = {"Question", "Answer", "Question Type", "Year", "image"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ChartQAPro missing columns: {sorted(missing)}")
    allowed_types = {
        "Factoid", "Multi Choice", "Conversational", "Fact Checking", "Hypothetical"
    }
    question_types = set(frame["Question Type"].dropna().astype(str))
    if not question_types.issubset(allowed_types):
        raise ValueError(f"unknown ChartQAPro question types: {sorted(question_types - allowed_types)}")
    return pd.DataFrame({
        "index": [str(i) for i in range(len(frame))],
        "question": frame["Question"].map(_as_scalar),
        "answer": frame["Answer"].map(_as_scalar),
        "question_type": frame["Question Type"],
        "year": frame["Year"].map(_as_scalar),
        "paragraph": frame["Paragraph"] if "Paragraph" in frame else "",
        "image": [
            base64.b64encode(_image_bytes(value, i)).decode("ascii")
            for i, value in enumerate(frame["image"])
        ],
    })


def prepare_dataset(dataset_id: str, source_path: Path, lmu_data: Path) -> PreparedDataset:
    spec = DATASETS.get(dataset_id)
    if spec is None:
        raise ValueError(f"unsupported dataset: {dataset_id}")
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"local Parquet dataset is missing: {source_path}")

    source_sha256 = file_sha256(source_path)
    lmu_data.mkdir(parents=True, exist_ok=True)
    tsv_path = lmu_data / f"{spec['runtime_id']}.tsv"
    metadata_path = lmu_data / f"{spec['runtime_id']}.local.json"

    if tsv_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
        if metadata.get("source_sha256") == source_sha256:
            count = int(metadata.get("sample_count", 0))
            if count > 0:
                return PreparedDataset(
                    dataset_id, spec["name"], spec["canonical_id"], spec["runtime_id"],
                    source_path, source_sha256, count, tsv_path,
                )

    try:
        frame = pd.read_parquet(source_path, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read local ChartQA Parquet files; "
            "install requirements/service.txt"
        ) from exc
    normalized = _normalize(dataset_id, frame)
    for field in ("question", "answer"):
        if normalized[field].map(lambda value: not str(value).strip()).any():
            raise ValueError(f"empty {field} found in {source_path}")

    temporary = tsv_path.with_suffix(".tsv.tmp")
    normalized.to_csv(temporary, sep="\t", index=False)
    temporary.replace(tsv_path)
    metadata_path.write_text(json.dumps({
        "dataset_id": dataset_id,
        "canonical_id": spec["canonical_id"],
        "runtime_id": spec["runtime_id"],
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "sample_count": len(normalized),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return PreparedDataset(
        dataset_id, spec["name"], spec["canonical_id"], spec["runtime_id"],
        source_path, source_sha256, len(normalized), tsv_path,
    )
