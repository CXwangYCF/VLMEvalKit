from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from vlmeval_service.chartqa_data import PreparedDataset, _normalize
from vlmeval_service.config import Settings
from vlmeval_service.results import parse_result
from vlmeval_service.service import ChartQAService


class ChartQAServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = Settings(
            data_dir=root / "LMUData",
            chartqa_test_path=root / "chartqa-test.parquet",
            chartqapro_path=root / "chartqapro.parquet",
            output_dir=root / "tasks",
            database_url=f"sqlite:///{(root / 'tasks.db').as_posix()}",
            target_base_url="http://target.test/v1",
            target_api_key="target-secret",
            target_model="qwen2.5-vl-7b-instruct",
        )
        self.service = ChartQAService(self.settings)
        self.service._executor.submit = Mock()
        self.prepared = PreparedDataset(
            dataset_id="chartqapro",
            name="ChartQAPro",
            canonical_id="ChartQAPro",
            runtime_id="ChartQAPro_LOCAL",
            source_path=root / "chartqapro.parquet",
            source_sha256="a" * 64,
            sample_count=2,
            tsv_path=root / "LMUData" / "ChartQAPro_LOCAL.tsv",
        )
        self.service._prepare = Mock(return_value=self.prepared)

    def tearDown(self):
        self.service._executor.shutdown(wait=False, cancel_futures=True)
        self.temp.cleanup()

    def test_capabilities_expose_both_datasets_and_no_judge(self):
        payload = self.service.capabilities()
        self.assertEqual([item["id"] for item in payload["datasets"]], ["chartqa-test", "chartqapro"])
        self.assertFalse(payload["evaluator"]["uses_judge"])
        self.assertNotIn("judges", payload)

    def test_create_locks_dataset_hash_and_is_idempotent(self):
        request = {
            "model_id": "qwen2.5-vl-7b-instruct",
            "dataset_id": "chartqapro",
            "parent_task_id": None,
        }
        first, reused = self.service.create(request, "same-key")
        second, reused_second = self.service.create(request, "same-key")
        self.assertFalse(reused)
        self.assertTrue(reused_second)
        self.assertEqual(first["task_id"], second["task_id"])
        self.assertEqual(first["config"]["dataset_sha256"], "a" * 64)
        self.assertFalse(first["config"]["uses_judge"])

    def test_command_has_no_judge_and_forces_local_data(self):
        task, _ = self.service.create({
            "model_id": "qwen2.5-vl-7b-instruct",
            "dataset_id": "chartqapro",
            "parent_task_id": None,
        }, "command-key")
        stored = self.service.store.get(task["task_id"])
        command, env = self.service._command(stored, False)
        joined = " ".join(command)
        self.assertIn("--data ChartQAPro_LOCAL", joined)
        self.assertNotIn("--judge", joined)
        self.assertEqual(env["LMDEPLOY_API_KEY"], "target-secret")
        self.assertEqual(env["VLMEVAL_LOCAL_DATA_ONLY"], "1")

    def test_parent_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.create({
                "model_id": "qwen2.5-vl-7b-instruct",
                "dataset_id": "chartqapro",
                "parent_task_id": "old-task",
            }, "parent-key")


class DatasetNormalizationTests(unittest.TestCase):
    def test_chartqa_test_parquet_fields_are_normalized(self):
        frame = pd.DataFrame([{
            "query": "How many?", "label": ["10"], "image": b"png", "type": "human"
        }])
        result = _normalize("chartqa-test", frame)
        self.assertEqual(result.loc[0, "question"], "How many?")
        self.assertEqual(result.loc[0, "answer"], "['10']")
        self.assertEqual(result.loc[0, "split"], "human")

    def test_chartqapro_rejects_unknown_question_type(self):
        frame = pd.DataFrame([{
            "Question": ["Q"], "Answer": ["A"], "Question Type": "Unknown",
            "Year": ["NO"], "image": b"png",
        }])
        with self.assertRaises(ValueError):
            _normalize("chartqapro", frame)


class ResultParserTests(unittest.TestCase):
    @patch("vlmeval_service.results._chartqapro_score", side_effect=[1.0, 0.5])
    def test_chartqapro_parser_builds_category_metrics(self, _score: Mock):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame([
                {"index": "1", "question": "q1", "prediction": "1", "answer": "['1']", "year": "['NO']", "question_type": "Factoid"},
                {"index": "2", "question": "q2", "prediction": "3", "answer": "['4']", "year": "['NO']", "question_type": "Hypothetical"},
            ]).to_excel(root / "model_ChartQAPro_LOCAL.xlsx", index=False)
            pd.DataFrame([["Factoid", 1.0], ["Overall", 0.75]]).to_csv(
                root / "model_ChartQAPro_LOCAL_acc.csv", index=False
            )
            summary, samples, quality = parse_result(
                root, "chartqapro", "ChartQAPro_LOCAL", total=2
            )
            self.assertEqual(summary["overall_accuracy"], 75.0)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(len(samples), 2)
            self.assertEqual(quality, "qualified")
            self.assertTrue((root / "results.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
