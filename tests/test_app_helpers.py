import os
import unittest
from unittest.mock import patch

import app


class TestAppHelpers(unittest.TestCase):
    def test_extract_image_filename_handles_single_quotes(self) -> None:
        html = "<div><img src='12345.png' /></div>"
        self.assertEqual(app.extract_image_filename(html), "12345.png")

    def test_extract_image_filename_handles_double_quotes(self) -> None:
        html = '<div><img src="nested/path/67890.png"></div>'
        self.assertEqual(app.extract_image_filename(html), "67890.png")


class TestDeckImagesRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.app.test_client()
        app.app.testing = True

    @patch("app.invoke")
    def test_deck_images_returns_entries_with_fallback_names(self, mock_invoke) -> None:
        hashed_filename = "12345-abcdef.png"
        base_name = "12345.png"
        image_path = app.IMAGE_DIR / base_name
        image_path.write_bytes(b"fake")

        mock_invoke.side_effect = [
            [42],
            [
                {
                    "fields": {
                        "Front": {"value": f'<img src="{hashed_filename}">'},
                        "Back": {"value": "Hello"},
                    }
                }
            ],
        ]

        response = self.client.get("/api/deck-images?deck=Test")
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(len(data["images"]), 1)
        self.assertTrue(data["images"][0]["image_url"].endswith(base_name))
        self.assertEqual(data["images"][0]["front_text"], "")
        self.assertEqual(data["images"][0]["sound_filename"], "")

        image_path.unlink()

    @patch("app.invoke")
    def test_deck_images_paginates_results(self, mock_invoke) -> None:
        image_one = app.IMAGE_DIR / "1.png"
        image_two = app.IMAGE_DIR / "2.png"
        image_one.write_bytes(b"fake")
        image_two.write_bytes(b"fake")

        mock_invoke.side_effect = [
            [1, 2],
            [
                {
                    "fields": {
                        "Front": {"value": '<img src="1.png">'},
                        "Back": {"value": "Hello"},
                    }
                },
                {
                    "fields": {
                        "Front": {"value": '<img src="2.png">'},
                        "Back": {"value": "World"},
                    }
                },
            ],
        ]

        response = self.client.get("/api/deck-images?deck=Test&page=1&page_size=1")
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["images"]), 1)

        image_one.unlink()
        image_two.unlink()


class TestAppUtilities(unittest.TestCase):
    def test_clean_field_text_strips_sound_tags_and_html(self) -> None:
        raw = "<div>안녕[sound:123.mp3]</div>&nbsp;&nbsp;하세요"
        self.assertEqual(app.clean_field_text(raw), "안녕 하세요")

    def test_estimate_sync_duration_bounds(self) -> None:
        seconds, label = app.estimate_sync_duration(0)
        self.assertGreaterEqual(seconds, 30)
        self.assertIn("About", label)

        seconds, label = app.estimate_sync_duration(1000)
        self.assertLessEqual(seconds, 180)
        self.assertIn("Roughly", label)

    def test_estimate_media_duration_bounds(self) -> None:
        seconds, label = app.estimate_media_duration(0, per_card_seconds=6.0)
        self.assertGreaterEqual(seconds, 45)
        self.assertIn("About", label)

        seconds, label = app.estimate_media_duration(1000, per_card_seconds=6.0)
        self.assertLessEqual(seconds, 1800)
        self.assertIn("Roughly", label)

    def test_allowed_file_accepts_pdf_only(self) -> None:
        self.assertTrue(app.allowed_file("lesson.pdf"))
        self.assertFalse(app.allowed_file("lesson.txt"))

    def test_parse_progress_line(self) -> None:
        self.assertEqual(app.parse_progress_line("PROGRESS 3/10"), (3, 10))
        self.assertEqual(app.parse_progress_line("PROGRESS  7 /  20"), (7, 20))
        self.assertIsNone(app.parse_progress_line("PROGRESS seven/ten"))

    def test_parse_summary_line(self) -> None:
        summary = app.parse_summary_line('SUMMARY: {"ok": true, "added": 2}')
        self.assertEqual(summary, {"ok": True, "added": 2})
        self.assertIsNone(app.parse_summary_line("SUMMARY: not-json"))

    def test_tokenize_text(self) -> None:
        tokens = app.tokenize_text("Hello, world! 안녕하세요?")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("안녕하세요", tokens)

    def test_get_response_text_prefers_output_text(self) -> None:
        class FakeResponse:
            output_text = "hello"

        self.assertEqual(app.get_response_text(FakeResponse()), "hello")

    def test_normalize_search_term_strips_quotes_and_whitespace(self) -> None:
        raw = "  hello\n\t\"world\"  "
        self.assertEqual(app.normalize_search_term(raw), "hello world")

    def test_build_field_query_handles_single_and_multi_word_terms(self) -> None:
        self.assertEqual(app.build_field_query("Front", "hello"), "Front:*hello*")
        self.assertEqual(app.build_field_query("Back", "hello world"), 'Back:"hello world"')


class TestDeckChatContext(unittest.TestCase):
    @patch("app.invoke")
    def test_build_deck_context_full_returns_all(self, mock_invoke) -> None:
        mock_invoke.side_effect = [
            [1, 2],
            [
                {"fields": {"Front": {"value": "apple"}, "Back": {"value": "사과"}}},
                {"fields": {"Front": {"value": "banana"}, "Back": {"value": "바나나"}}},
            ],
        ]
        snippets, total = app.build_deck_context_full("Test")
        self.assertEqual(len(snippets), 2)
        self.assertIn("apple", snippets[0])
        self.assertEqual(total, 2)

    def test_deck_cards_requires_deck_param(self) -> None:
        client = app.app.test_client()
        response = client.get("/api/deck-cards")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])

    def test_deck_images_requires_deck_param(self) -> None:
        client = app.app.test_client()
        response = client.get("/api/deck-images")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])

    def test_deck_image_stats_requires_deck_param(self) -> None:
        client = app.app.test_client()
        response = client.get("/api/deck-image-stats")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])

    def test_media_endpoint_requires_filename(self) -> None:
        client = app.app.test_client()
        response = client.get("/api/media")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])

    def test_export_chat_logs_requires_valid_limit(self) -> None:
        client = app.app.test_client()
        response = client.get("/api/chat-logs/export?limit=bad")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])

    @patch("app.invoke")
    def test_deck_search_requires_term_param(self, mock_invoke) -> None:
        client = app.app.test_client()
        response = client.get("/api/deck-search")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        mock_invoke.assert_not_called()

    @patch("app.invoke")
    def test_deck_search_returns_matches(self, mock_invoke) -> None:
        mock_invoke.side_effect = [
            [10],
            [{"note": 42, "deckName": "AdvancedBeginner1::Grammar"}],
            [
                {
                    "fields": {
                        "Front": {"value": "성격"},
                        "Back": {"value": "personality"},
                    }
                }
            ],
        ]
        client = app.app.test_client()
        response = client.get("/api/deck-search?term=personality")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["deck"], "AdvancedBeginner1::Grammar")
        self.assertEqual(data["results"][0]["match"], "back")


if __name__ == "__main__":
    unittest.main()
