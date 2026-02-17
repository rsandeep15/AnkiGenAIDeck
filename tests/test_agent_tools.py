import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_TOOLS_PATH = REPO_ROOT / "scripts" / "agent_tools.py"


def _load_agent_tools_module():
    spec = importlib.util.spec_from_file_location("agent_tools_mod", AGENT_TOOLS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load scripts/agent_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAgentToolsHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_tools = _load_agent_tools_module()

    def test_load_card_rows_from_list_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pairs.json"
            payload = [{"Front": "안녕하세요.", "Back": "Hello."}]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rows = self.agent_tools.load_card_rows(path)
            self.assertEqual(rows, payload)

    def test_load_card_rows_from_cards_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pairs.json"
            payload = {"cards": [{"front": "학교에 가세요.", "back": "Please go to school."}]}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rows = self.agent_tools.load_card_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["front"], "학교에 가세요.")

    def test_normalize_cards_supports_fallback_keys(self) -> None:
        rows = [
            {"foreign": "성격", "english": "personality"},
            {"korean": "친절해요.", "meaning": "is kind"},
        ]
        cards = self.agent_tools.normalize_cards(rows, "Front", "Back")
        self.assertEqual(cards[0], ("성격", "personality"))
        self.assertEqual(cards[1], ("친절해요.", "is kind"))

    def test_cards_import_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pairs.json"
            payload = [
                {"Front": "학교에 가세요.", "Back": "Please go to school."},
                {"Front": "집에 오세요.", "Back": "Please come home."},
            ]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = subprocess.run(
                [
                    str(REPO_ROOT / ".venv" / "bin" / "python"),
                    str(AGENT_TOOLS_PATH),
                    "cards-import",
                    "--deck",
                    "TestDeck",
                    "--input",
                    str(path),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summary = json.loads(result.stdout.strip())
            self.assertTrue(summary["ok"])
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["to_add"], 2)


if __name__ == "__main__":
    unittest.main()
