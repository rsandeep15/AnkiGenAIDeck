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


if __name__ == "__main__":
    unittest.main()
