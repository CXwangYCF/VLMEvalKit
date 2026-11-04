from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from vlmeval_service.config import Settings
from vlmeval_service.results import parse_result
from vlmeval_service.service import DATASET_MD5, DatasetUnavailableError, MathVistaService


class MathVistaServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = Settings(
            data_dir=root / "LMUData",
            output_dir=root / "tasks",
            database_url=f"sqlite:///{(root / 'tasks.db').as_posix()}",
            target_base_url="http://target.test/v1",
            target_api_key="target-secret",
            target_model="qwen2.5-vl-7b-instruct",
            primary_judge_base_url="http://primary.test/v1",
            primary_judge_api_key="primary-secret",
            primary_judge_model="qwen3.8-flash",
            backup_judge_base_url="http://backup.test/v1",
            backup_judge_api_key="backup-secret",
            backup_judge_model="gpt-5.5",
        )
        self.service = MathVistaService(self.settings)
        self.service._validate_dataset = Mock()
        self.service._executor.submit = Mock()

    def tearDown(self):
        self.service._executor.shutdown(wait=False, cancel_futures=True)
        self.temp.cleanup()

    def test_capabilities_expose_both_judges_without_keys(self):
        payload = self.service.capabilities()
        self.assertEqual(payload["dataset"]["canonical_id"], "MathVista_MINI")
        self.assertEqual(payload["dataset"]["sample_count"], 1000)
        self.assertEqual(payload["dataset"]["source"], "local-only")
        self.assertEqual([j["role"] for j in payload["judges"]], ["primary", "backup"])
        self.assertNotIn("primary-secret", json.dumps(payload))

    def test_idempotency_reuses_same_task(self):
        request = {
            "model_id": "qwen2.5-vl-7b-instruct",
            "dataset_id": "mathvista",
            "judge_id": "qwen3.8-flash",
            "parent_task_id": None,
        }
        first, reused = self.service.create(request, "same-key")
        second, reused_second = self.service.create(request, "same-key")
        self.assertFalse(reused)
        self.assertTrue(reused_second)
        self.assertEqual(first["task_id"], second["task_id"])

    def test_command_locks_one_judge_and_keeps_keys_out_of_argv(self):
        request = {
            "model_id": "qwen2.5-vl-7b-instruct",
            "dataset_id": "mathvista",
            "judge_id": "qwen3.8-flash",
            "parent_task_id": None,
        }
        task, _ = self.service.create(request, "command-key")
        stored = self.service.store.get(task["task_id"])
        command, env = self.service._command(stored, False)
        joined = " ".join(command)
        self.assertIn("--judge qwen3.8-flash", joined)
        self.assertNotIn("gpt-5.5", joined)
        self.assertNotIn("secret", joined)
        self.assertEqual(env["OPENAI_API_KEY"], "primary-secret")
        self.assertEqual(env["LMDEPLOY_API_KEY"], "target-secret")
        self.assertEqual(env["LMUData"], str(self.settings.data_dir))
        self.assertEqual(env["VLMEVAL_LOCAL_DATA_ONLY"], "1")

    def test_backup_run_requires_primary_parent(self):
        primary, _ = self.service.create({
            "model_id": "qwen2.5-vl-7b-instruct",
            "dataset_id": "mathvista",
            "judge_id": "qwen3.8-flash",
            "parent_task_id": None,
        }, "primary-key")
        self.service.store.update(primary["task_id"], status="failed")
        backup, _ = self.service.create({
            "model_id": "qwen2.5-vl-7b-instruct",
            "dataset_id": "mathvista",
            "judge_id": "gpt-5.5",
            "parent_task_id": primary["task_id"],
        }, "backup-key")
        self.assertEqual(backup["config"]["judge_role"], "backup")
        self.assertEqual(backup["parent_task_id"], primary["task_id"])

    def test_local_dataset_preflight_rejects_missing_file(self):
        image_vqa = types.ModuleType("vlmeval.dataset.image_vqa")
        image_vqa.MathVista = type("MathVista", (), {"DATASET_URL": {"MathVista_MINI": "unused"}})
        with patch.dict(sys.modules, {"vlmeval.dataset.image_vqa": image_vqa}):
            with self.assertRaises(DatasetUnavailableError):
                MathVistaService._validate_dataset(self.service)

    def test_local_dataset_preflight_accepts_pinned_md5(self):
        self.settings.data_dir.mkdir(parents=True)
        data_path = self.settings.data_dir / "MathVista_MINI.tsv"
        data_path.write_bytes(b"test fixture")
        image_vqa = types.ModuleType("vlmeval.dataset.image_vqa")
        image_vqa.MathVista = type("MathVista", (), {"DATASET_URL": {"MathVista_MINI": "unused"}})
        with patch.dict(sys.modules, {"vlmeval.dataset.image_vqa": image_vqa}), \
                patch("vlmeval_service.service.file_md5", return_value=DATASET_MD5):
            self.assertEqual(MathVistaService._validate_dataset(self.service), data_path)


class ResultParserTests(unittest.TestCase):
    @patch("vlmeval_service.results._is_correct", side_effect=[True, False])
    def test_parser_builds_official_metrics_and_quality_gate(self, _correct: Mock):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detail = pd.DataFrame([
                {"index": "1", "question": "q1", "prediction": "1", "answer": "1", "res": "1", "log": "Succeed"},
                {"index": "2", "question": "q2", "prediction": "3", "answer": "4", "res": "3", "log": "Succeed"},
            ])
            detail.to_excel(root / "model_MathVista_MINI_qwen3.8-flash.xlsx", index=False)
            pd.DataFrame([
                {"Task&Skill": "Overall", "tot": 2, "hit": 1, "acc": 50.0},
                {"Task&Skill": "geometry", "tot": 2, "hit": 1, "acc": 50.0},
            ]).to_csv(root / "model_MathVista_MINI_qwen3.8-flash_score.csv", index=False)
            summary, samples, quality = parse_result(root, "qwen3.8-flash", total=2)
            self.assertEqual(summary["overall_accuracy"], 50.0)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(len(samples), 2)
            self.assertEqual(quality, "qualified")
            self.assertEqual(len((root / "results.jsonl").read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
