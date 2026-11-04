from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]


def _database_path(url: str) -> Path:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError("VLMEVAL_DATABASE_URL currently supports sqlite:/// URLs only")
    path = Path(url[len(prefix):])
    return path if path.is_absolute() else REPO_ROOT / path


def public_base_url(value: str) -> str:
    """Return only scheme/host/port; never expose credentials or paths."""
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "configured"


@dataclass(frozen=True)
class JudgeConfig:
    id: str
    name: str
    role: str
    base_url: str
    api_key: str

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.name)


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT
    data_dir: Path = Path(os.getenv("LMUData", str(REPO_ROOT / "LMUData")))
    output_dir: Path = Path(os.getenv("VLMEVAL_OUTPUT_DIR", str(REPO_ROOT / "service_data" / "tasks")))
    database_url: str = os.getenv(
        "VLMEVAL_DATABASE_URL", "sqlite:///service_data/vlmeval_tasks.db"
    )
    target_base_url: str = os.getenv("VLMEVAL_TARGET_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    target_api_key: str = os.getenv("VLMEVAL_TARGET_API_KEY", "local-no-key")
    target_model: str = os.getenv("VLMEVAL_TARGET_MODEL", "qwen2.5-vl-7b-instruct")
    primary_judge_base_url: str = os.getenv(
        "VLMEVAL_PRIMARY_JUDGE_BASE_URL", "http://127.0.0.1:8080/v1"
    ).rstrip("/")
    primary_judge_api_key: str = os.getenv("VLMEVAL_PRIMARY_JUDGE_API_KEY", "local-no-key")
    primary_judge_model: str = os.getenv("VLMEVAL_PRIMARY_JUDGE_MODEL", "qwen3.8-flash")
    backup_judge_base_url: str = os.getenv("VLMEVAL_BACKUP_JUDGE_BASE_URL", "").rstrip("/")
    backup_judge_api_key: str = os.getenv("VLMEVAL_BACKUP_JUDGE_API_KEY", "")
    backup_judge_model: str = os.getenv("VLMEVAL_BACKUP_JUDGE_MODEL", "gpt-5.5")
    api_nproc: int = int(os.getenv("VLMEVAL_API_NPROC", "8"))
    judge_api_nproc: int = int(os.getenv("VLMEVAL_JUDGE_API_NPROC", "8"))
    request_timeout: int = int(os.getenv("VLMEVAL_REQUEST_TIMEOUT", "1800"))
    judge_timeout: int = int(os.getenv("VLMEVAL_JUDGE_TIMEOUT", "600"))

    @property
    def database_path(self) -> Path:
        return _database_path(self.database_url)

    @property
    def judges(self) -> dict[str, JudgeConfig]:
        return {
            "qwen3.8-flash": JudgeConfig(
                id="qwen3.8-flash",
                name=self.primary_judge_model,
                role="primary",
                base_url=self.primary_judge_base_url,
                api_key=self.primary_judge_api_key,
            ),
            "gpt-5.5": JudgeConfig(
                id="gpt-5.5",
                name=self.backup_judge_model,
                role="backup",
                base_url=self.backup_judge_base_url,
                api_key=self.backup_judge_api_key,
            ),
        }
