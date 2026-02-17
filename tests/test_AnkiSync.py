import unittest

import AnkiSync as sync


class TestAnkiSyncHelpers(unittest.TestCase):
    def test_normalize_json_payload_strips_code_block(self) -> None:
        payload = "```json\n[{\"english\": \"hi\"}]\n```"
        self.assertEqual(sync.normalize_json_payload(payload), '[{"english": "hi"}]')

    def test_parse_word_pairs_returns_list(self) -> None:
        content = '[{"english": "hi", "foreign": "안녕"}]'
        result = sync.parse_word_pairs(content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["english"], "hi")

    def test_build_note_structure(self) -> None:
        note = sync.build_note("Deck", "Front", "Back", "Basic")
        self.assertEqual(note["deckName"], "Deck")
        self.assertEqual(note["modelName"], "Basic")
        self.assertEqual(note["fields"]["Front"], "Front")
        self.assertEqual(note["fields"]["Back"], "Back")

    def test_parse_word_pairs_rejects_non_list(self) -> None:
        content = '{"english": "hi", "foreign": "안녕"}'
        with self.assertRaises(RuntimeError):
            sync.parse_word_pairs(content)

    def test_parse_word_pairs_rejects_invalid_json(self) -> None:
        with self.assertRaises(RuntimeError):
            sync.parse_word_pairs('{"english": "hi"')

    def test_build_prompt_includes_romanized_flag(self) -> None:
        prompt = sync.build_prompt(include_romanized=True)
        self.assertIn("romanized", prompt)
        self.assertIn("Include a \"romanized\" key", prompt)

        prompt = sync.build_prompt(include_romanized=False)
        self.assertIn("Do not include romanization keys", prompt)

    def test_strip_english_duplicate_suffix(self) -> None:
        self.assertEqual(
            sync.strip_english_duplicate("성격 personality", "personality"),
            "성격",
        )

    def test_strip_english_duplicate_parenthetical(self) -> None:
        self.assertEqual(
            sync.strip_english_duplicate("성격 (personality)", "personality"),
            "성격",
        )


if __name__ == "__main__":
    unittest.main()
