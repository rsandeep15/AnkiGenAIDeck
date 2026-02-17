import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import AnkiDeckToImages as images


class TestAnkiDeckToImages(unittest.TestCase):
    def test_select_candidates_limit_only(self) -> None:
        candidates = [(1, "a", "A"), (2, "b", "B"), (3, "c", "C")]
        selected = images.select_candidates(candidates, limit=2, shuffle=False, seed=42)
        self.assertEqual(selected, [(1, "a", "A"), (2, "b", "B")])

    def test_select_candidates_shuffle_is_reproducible(self) -> None:
        candidates = [(1, "a", "A"), (2, "b", "B"), (3, "c", "C"), (4, "d", "D")]
        first = images.select_candidates(candidates, limit=0, shuffle=True, seed=7)
        second = images.select_candidates(candidates, limit=0, shuffle=True, seed=7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, candidates)

    def test_parse_gating_decision_accepts_common_true_variants(self) -> None:
        self.assertTrue(images.parse_gating_decision("true"))
        self.assertTrue(images.parse_gating_decision("TRUE."))
        self.assertTrue(images.parse_gating_decision('{"allow": true}'))

    def test_parse_gating_decision_accepts_common_false_variants(self) -> None:
        self.assertFalse(images.parse_gating_decision("false"))
        self.assertFalse(images.parse_gating_decision("False, skip image"))
        self.assertFalse(images.parse_gating_decision('{"allow": false}'))

    def test_emotion_state_prior_prefers_emotion_phrases(self) -> None:
        self.assertTrue(images.emotion_state_prior("성격", "to be excited"))
        self.assertTrue(images.emotion_state_prior("행복하다", "happy"))
        self.assertFalse(images.emotion_state_prior("연필", "pencil"))

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

    @patch("AnkiDeckToImages.generate_image", return_value=Path("fake.png"))
    @patch("AnkiDeckToImages.invoke")
    @patch("AnkiDeckToImages.OpenAI")
    def test_process_card_emotion_prior_overrides_gating_false(
        self,
        mock_openai: MagicMock,
        mock_invoke: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output_text="false")
        mock_openai.return_value = mock_client

        status, _, _ = images.process_card(
            card=(123, "긴장하다", "to be nervous"),
            api_key="test",
            image_model="gpt-image-1",
            prompt_template="{text}",
            skip_gating=False,
            dry_run=False,
        )

        self.assertEqual(status, "added")
        mock_generate.assert_called_once()
        mock_invoke.assert_called_once()

    @patch("AnkiDeckToImages.generate_image")
    @patch("AnkiDeckToImages.invoke")
    @patch("AnkiDeckToImages.OpenAI")
    def test_process_card_dry_run_skips_side_effects(
        self,
        mock_openai: MagicMock,
        mock_invoke: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.responses.create.return_value = SimpleNamespace(output_text="true")
        mock_openai.return_value = mock_client

        status, text, reason = images.process_card(
            card=(42, "안녕", "hello"),
            api_key="test",
            image_model="gpt-image-1",
            prompt_template="{text}",
            skip_gating=False,
            dry_run=True,
        )
        self.assertEqual(status, "would_add")
        self.assertIsNone(reason)
        mock_generate.assert_not_called()
        mock_invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
