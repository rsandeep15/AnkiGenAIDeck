import unittest
from unittest.mock import MagicMock, patch

import AnkiDeckToSpeech as speech


class TestAnkiDeckToSpeech(unittest.TestCase):
    def test_prepare_text_for_tts_strips_html_and_whitespace(self) -> None:
        raw = "<div>Hello&nbsp;&nbsp;world</div><br>!"
        self.assertEqual(speech.prepare_text_for_tts(raw), "Hello world !")

    def test_prepare_text_for_tts_strips_sound_tags(self) -> None:
        raw = "안녕[sound:123.mp3]"
        self.assertEqual(speech.prepare_text_for_tts(raw), "안녕")

    @patch("AnkiDeckToSpeech.invoke")
    @patch("AnkiDeckToSpeech.create_audio_file")
    @patch("AnkiDeckToSpeech.OpenAI")
    def test_process_card_skips_when_no_speakable_text(
        self,
        mock_openai: MagicMock,
        mock_create_audio: MagicMock,
        mock_invoke: MagicMock,
    ) -> None:
        status, _, reason = speech.process_card(
            card=(1, "<br>", "back"),
            api_key="fake",
            model="gpt-4o-mini-tts",
            voice="onyx",
            instructions="speak",
        )
        self.assertEqual(status, "skip")
        self.assertIn("No speakable text", reason)
        mock_create_audio.assert_not_called()
        mock_invoke.assert_not_called()

    @patch("AnkiDeckToSpeech.invoke")
    @patch("AnkiDeckToSpeech.create_audio_file")
    @patch("AnkiDeckToSpeech.OpenAI")
    def test_process_card_generates_audio_and_updates_note(
        self,
        mock_openai: MagicMock,
        mock_create_audio: MagicMock,
        mock_invoke: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        card = (5, "안녕", "hello")
        status, _, reason = speech.process_card(
            card=card,
            api_key="fake",
            model="gpt",
            voice="onyx",
            instructions="speak",
        )

        self.assertEqual(status, "added")
        mock_create_audio.assert_called_once()
        mock_invoke.assert_called_once()
        args, kwargs = mock_invoke.call_args
        self.assertEqual(args[0], "updateNoteFields")
        self.assertIn("audio", kwargs["note"])
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
