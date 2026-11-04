from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from .service import ConflictError, DatasetUnavailableError, MathVistaService


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(pattern=r"^qwen2\.5-vl-7b-instruct$")
    dataset_id: str = Field(pattern=r"^mathvista$")
    judge_id: str = Field(pattern=r"^(qwen3\.8-flash|gpt-5\.5)$")
    parent_task_id: str | None = None


service = MathVistaService()
app = FastAPI(title="VLMEvalKit MathVista task service", version="1.0.0")


def task_or_404(task: dict | None) -> dict:
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/health")
def health() -> dict:
    caps = service.capabilities()
    return {
        "status": "ok",
        "suite": caps["suite"],
        "target_model": caps["target_model"],
        "dataset": caps["dataset"],
        "judges": caps["judges"],
    }


@app.get("/api/v1/eval/capabilities")
def capabilities() -> dict:
    return service.capabilities()


@app.post("/api/v1/eval/tasks", status_code=202)
def create_task(
    payload: CreateTaskRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    try:
        task, reused = service.create(payload.model_dump(), idempotency_key)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if reused:
        response.headers["X-Task-Reused"] = "true"
    return task


@app.get("/api/v1/eval/tasks")
def list_tasks(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    return {"items": service.list(limit)}


@app.get("/api/v1/eval/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    return task_or_404(service.get(task_id))


@app.post("/api/v1/eval/tasks/{task_id}/pause")
def pause_task(task_id: str) -> dict:
    return task_or_404(service.pause(task_id))


@app.post("/api/v1/eval/tasks/{task_id}/retry-failures", status_code=202)
def retry_failures(task_id: str) -> dict:
    try:
        return task_or_404(service.retry_failures(task_id))
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/eval/tasks/{task_id}/result")
def result(task_id: str) -> dict:
    return task_or_404(service.result(task_id))


@app.get("/api/v1/eval/tasks/{task_id}/samples")
def samples(
    task_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    return task_or_404(service.samples(task_id, offset, limit))


@app.get("/api/v1/eval/tasks/{task_id}/artifacts/results.jsonl")
def artifact(task_id: str) -> Response:
    content = service.artifact(task_id)
    if content is None:
        raise HTTPException(status_code=404, detail="task not found")
    return Response(
        content=content,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{task_id}-results.jsonl"'},
    )
