import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "image_gating_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("image_gating_eval_mod", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load image_gating_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestImageGatingEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module()

    def test_parse_bool_label_variants(self) -> None:
        self.assertTrue(self.mod.parse_bool_label("true"))
        self.assertTrue(self.mod.parse_bool_label("1"))
        self.assertFalse(self.mod.parse_bool_label("false"))
        self.assertFalse(self.mod.parse_bool_label("0"))

    def test_parse_example_row_supports_alias_keys(self) -> None:
        ex = self.mod.parse_example_row(
            {"foreign": "성격", "english": "personality", "should_generate": "false"}
        )
        self.assertEqual(ex.front, "성격")
        self.assertEqual(ex.back, "personality")
        self.assertFalse(ex.label)

    def test_load_examples_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "labels.jsonl"
            rows = [
                {"front": "웃다", "back": "to smile", "label": True},
                {"front": "그리고", "back": "and", "label": False},
            ]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            examples = self.mod.load_examples(path)
            self.assertEqual(len(examples), 2)
            self.assertTrue(examples[0].label)
            self.assertFalse(examples[1].label)

    def test_metrics(self) -> None:
        summary = self.mod.metrics(
            [True, True, False, False],
            [True, False, True, False],
        )
        self.assertEqual(summary["tp"], 1)
        self.assertEqual(summary["fp"], 1)
        self.assertEqual(summary["tn"], 1)
        self.assertEqual(summary["fn"], 1)
        self.assertEqual(summary["accuracy"], 0.5)

    def test_parse_variants_json_or_lines(self) -> None:
        variants = self.mod.parse_variants('{"variants":["a","b"]}')
        self.assertEqual(variants, ["a", "b"])
        variants = self.mod.parse_variants("- x\n- y")
        self.assertEqual(variants, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
