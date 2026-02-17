import re
import subprocess
import unittest
from pathlib import Path


# Matches current OpenAI API key formats like "sk-proj-..." and other "sk-..." variants.
OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")

# Keep this scoped to text-based source/docs that should never contain real keys.
SCANNED_SUFFIXES = {
    ".py",
    ".js",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
}


class TestNoHardcodedOpenAIKeys(unittest.TestCase):
    def test_repo_has_no_hardcoded_openai_keys(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        listed = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        files = [line.strip() for line in listed.stdout.splitlines() if line.strip()]

        hits: list[str] = []
        for rel_path in files:
            path = repo_root / rel_path
            if not path.is_file():
                continue
            if path.suffix.lower() not in SCANNED_SUFFIXES:
                continue
            # Intentional local env files can contain secrets and should never be tracked anyway.
            if path.name.startswith(".env"):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if OPENAI_KEY_PATTERN.search(content):
                hits.append(rel_path)

        self.assertEqual(
            hits,
            [],
            msg=f"Found potential hardcoded OpenAI API key material in: {', '.join(hits)}",
        )


if __name__ == "__main__":
    unittest.main()
