import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import AnkiDeckToImages as images


class TestAnkiDeckToImages(unittest.TestCase):
    def test_strip_image_tags_removes_all_img_elements(self) -> None:
        html = '<div>front<img src="a.png"/></div><p><IMG SRC="b.png"></p>'
        result = images.strip_image_tags(html)
        self.assertNotIn("<img", result.lower())
        self.assertEqual(result.strip(), "<div>front</div><p></p>")

    def test_sanitize_text_strips_html_and_collapses_space(self) -> None:
        raw = "<div>Hello&nbsp;&nbsp;world</div><br>!"
        self.assertEqual(images.sanitize_text(raw), "Hello world !")

    def test_build_image_prompt_inserts_text(self) -> None:
        template = "Draw: {text}"
        self.assertEqual(images.build_image_prompt(template, "cat"), "Draw: cat")

    @patch("AnkiDeckToImages.invoke")
    @patch("AnkiDeckToImages.OpenAI")
    def test_process_card_removes_existing_image_when_gating_false(
        self, mock_openai: MagicMock, mock_invoke: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output_text="false")
        mock_openai.return_value = mock_client

        front = "<div>Hello</div><img src='old.png'/>"
        back = "<div>World</div><img src='old_back.png'/>"

        status, text, reason = images.process_card(
            card=(1, front, back),
            api_key="test",
            image_model="gpt-image-1",
            prompt_template="{text}",
            skip_gating=False,
            dry_run=False,
        )

        self.assertEqual(status, "skip")
        self.assertEqual(reason, "Gating model returned false; existing image removed.")
        mock_invoke.assert_called_once()
        (args, kwargs) = mock_invoke.call_args
        self.assertEqual(args[0], "updateNoteFields")
        updated_fields = kwargs["note"]["fields"]
        self.assertNotIn("<img", updated_fields["Front"].lower())

    @patch("AnkiDeckToImages.generate_image", return_value=Path("fake.png"))
    @patch("AnkiDeckToImages.invoke")
    @patch("AnkiDeckToImages.OpenAI")
    def test_process_card_generates_and_attaches_image_when_gating_true(
        self,
        mock_openai: MagicMock,
        mock_invoke: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output_text="true")
        mock_openai.return_value = mock_client

        status, text, reason = images.process_card(
            card=(99, "안녕", "hello"),
            api_key="test",
            image_model="gpt-image-1",
            prompt_template="{text}",
            skip_gating=False,
            dry_run=False,
        )

        self.assertEqual(status, "added")
        mock_generate.assert_called_once()
        mock_invoke.assert_called_once()
        (args, kwargs) = mock_invoke.call_args
        self.assertEqual(args[0], "updateNoteFields")
        self.assertIn("picture", kwargs["note"])

    @patch("AnkiDeckToImages.generate_image")
    @patch("AnkiDeckToImages.invoke")
    @patch("AnkiDeckToImages.OpenAI")
    def test_process_card_dry_run_skips_side_effects(
        self,
        mock_openai: MagicMock,
        mock_invoke: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        status, text, reason = images.process_card(
            card=(42, "안녕", "hello"),
            api_key="test",
            image_model="gpt-image-1",
            prompt_template="{text}",
            skip_gating=False,
            dry_run=True,
        )
        self.assertEqual(status, "dry_run")
        self.assertIsNone(reason)
        mock_generate.assert_not_called()
        mock_invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
