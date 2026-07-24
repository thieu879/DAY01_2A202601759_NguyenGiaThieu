"""
Checkpoint 4 - Part 4: mini assistant.
Run: python -m pytest tests/test_part4.py -v
All API calls are mocked, so no real API key is needed.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


REQUIRED_KEYS = {"num_turns", "total_tokens", "total_cost", "history"}


def _make_stream(text: str):
    """Create a mock Gemini stream using chunk.text values."""

    chunks = []
    for piece in (text[: len(text) // 2], text[len(text) // 2 :]):
        chunk = MagicMock()
        chunk.text = piece
        chunks.append(chunk)
    final = MagicMock()
    final.text = None
    chunks.append(final)
    return chunks


class TestRunAssistantBasic(unittest.TestCase):
    def test_function_exists_and_is_callable(self):
        self.assertTrue(callable(MOD.run_assistant))

    @patch("google.genai.Client")
    def test_quit_immediately_returns_stats_dict(self, MockClient):
        MockClient.return_value = MagicMock()
        get_input = MagicMock(side_effect=["quit"])

        result = MOD.run_assistant("Ban la tro ly.", get_input=get_input)

        self.assertIsInstance(result, dict)
        for key in REQUIRED_KEYS:
            self.assertIn(key, result, f"Missing key: {key}")
        self.assertEqual(result["num_turns"], 0)

    @patch("google.genai.Client")
    def test_exit_is_case_insensitive(self, MockClient):
        MockClient.return_value = MagicMock()
        get_input = MagicMock(side_effect=["EXIT"])

        result = MOD.run_assistant("Ban la tro ly.", get_input=get_input)

        self.assertEqual(result["num_turns"], 0)

    @patch("google.genai.Client")
    def test_max_turns_zero_returns_without_reading_input(self, MockClient):
        MockClient.return_value = MagicMock()
        get_input = MagicMock(side_effect=[])

        result = MOD.run_assistant("Ban la tro ly.", get_input=get_input, max_turns=0)

        self.assertEqual(result["num_turns"], 0)


class TestRunAssistantScenario(unittest.TestCase):
    PERSONA = "Ban la tro giang than thien cua khoa AI."

    def _run_conversation(self, MockClient, user_messages, replies):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = [
            _make_stream(reply) for reply in replies
        ]
        get_input = MagicMock(side_effect=list(user_messages) + ["quit"])
        result = MOD.run_assistant(self.PERSONA, get_input=get_input)
        return result, mock_client

    @patch("google.genai.Client")
    def test_two_turns_counted_and_stats_positive(self, MockClient):
        result, _ = self._run_conversation(
            MockClient,
            ["Xin chao", "Ke mot su that thu vi"],
            ["Chao ban, minh giup gi duoc?", "Vietnam co hon 3000 km bo bien."],
        )
        self.assertEqual(result["num_turns"], 2)
        self.assertGreater(result["total_tokens"], 0)
        self.assertGreater(result["total_cost"], 0.0)

    @patch("google.genai.Client")
    def test_api_called_with_stream_and_persona(self, MockClient):
        _, mock_client = self._run_conversation(MockClient, ["Xin chao"], ["Chao ban!"])
        self.assertTrue(mock_client.models.generate_content_stream.called)
        _, kwargs = mock_client.models.generate_content_stream.call_args
        self.assertEqual(kwargs.get("model"), MOD.OPENAI_MODEL)
        config = kwargs.get("config")
        self.assertTrue(
            self.PERSONA in str(getattr(config, "system_instruction", "")),
            "Persona must be sent as system instruction",
        )

    @patch("google.genai.Client")
    def test_history_contains_last_turn(self, MockClient):
        result, _ = self._run_conversation(
            MockClient,
            ["Cau hoi thu nhat", "Cau hoi thu hai"],
            ["Tra loi thu nhat.", "Tra loi thu hai."],
        )
        history_text = " ".join(m["content"] for m in result["history"])
        self.assertIn("Cau hoi thu hai", history_text)
        self.assertIn("Tra loi thu hai", history_text)

    @patch("google.genai.Client")
    def test_history_trimmed_to_three_turns(self, MockClient):
        user_messages = [f"Cau hoi so {i}" for i in range(1, 6)]
        replies = [f"Tra loi so {i}." for i in range(1, 6)]
        result, _ = self._run_conversation(MockClient, user_messages, replies)

        self.assertEqual(result["num_turns"], 5)
        self.assertLessEqual(
            len(result["history"]),
            6,
            "History must be trimmed to at most 3 turns (6 messages)",
        )

    @patch("google.genai.Client")
    def test_max_turns_limits_conversation(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = [
            _make_stream(f"Tra loi {i}") for i in range(10)
        ]
        get_input = MagicMock(side_effect=[f"Cau {i}" for i in range(10)])

        result = MOD.run_assistant(self.PERSONA, get_input=get_input, max_turns=2)

        self.assertEqual(result["num_turns"], 2)


if __name__ == "__main__":
    unittest.main()
