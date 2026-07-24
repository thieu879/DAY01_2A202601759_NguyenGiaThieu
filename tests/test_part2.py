"""
Checkpoint 2 - Part 2: system prompt and token counting.
Run: python -m pytest tests/test_part2.py -v
All API calls are mocked, so no real API key is needed.
"""

import unittest
from unittest.mock import MagicMock, patch

from tests._loader import MOD


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    return resp


class TestChatWithSystemPrompt(unittest.TestCase):
    @patch("google.genai.Client")
    def test_returns_tuple_str_float(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response("OK")

        result = MOD.chat_with_system_prompt("You are a teacher.", "Hello")

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], float)
        self.assertGreater(result[1], 0.0)

    @patch("google.genai.Client")
    def test_contents_and_system_instruction_are_sent(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response()

        system_prompt = "You are a history expert."
        user_prompt = "How many dynasties did Vietnam have?"
        MOD.chat_with_system_prompt(system_prompt, user_prompt)

        _, kwargs = mock_client.models.generate_content.call_args
        self.assertEqual(kwargs.get("contents"), user_prompt)
        config = kwargs.get("config")
        self.assertIsNotNone(config, "config must be passed to generate_content()")
        self.assertEqual(getattr(config, "system_instruction", None), system_prompt)

    @patch("google.genai.Client")
    def test_system_prompt_content_is_sent(self, MockClient):
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response()

        system_prompt = "PERSONA_DAC_BIET_XYZ"
        MOD.chat_with_system_prompt(system_prompt, "Question")

        _, kwargs = mock_client.models.generate_content.call_args
        config = kwargs.get("config")
        self.assertTrue(
            system_prompt in str(config),
            "system prompt must be included in the config",
        )


class TestCountTokens(unittest.TestCase):
    def test_returns_positive_int_for_non_empty_text(self):
        result = MOD.count_tokens("Xin chao Viet Nam")
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_longer_text_has_more_tokens(self):
        short = MOD.count_tokens("word " * 10)
        long = MOD.count_tokens("word " * 200)
        self.assertGreater(long, short)

    def test_unknown_model_falls_back_gracefully(self):
        result = MOD.count_tokens("some text here", model="unknown-model-123")
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)


class TestEstimateCost(unittest.TestCase):
    PROMPT = "Tell me a fun fact about Vietnam. " * 5
    RESPONSE = "Vietnam is a major coffee exporter. " * 5

    def test_returns_dict_with_required_keys(self):
        result = MOD.estimate_cost(self.PROMPT, self.RESPONSE)
        required_keys = {
            "input_tokens",
            "output_tokens",
            "input_cost",
            "output_cost",
            "total_cost",
        }
        self.assertIsInstance(result, dict)
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_token_counts_are_positive_ints(self):
        result = MOD.estimate_cost(self.PROMPT, self.RESPONSE)
        self.assertIsInstance(result["input_tokens"], int)
        self.assertIsInstance(result["output_tokens"], int)
        self.assertGreater(result["input_tokens"], 0)
        self.assertGreater(result["output_tokens"], 0)

    def test_total_equals_input_plus_output(self):
        result = MOD.estimate_cost(self.PROMPT, self.RESPONSE)
        self.assertAlmostEqual(
            result["total_cost"],
            result["input_cost"] + result["output_cost"],
            places=10,
        )

    def test_mini_is_cheaper_than_gpt4o(self):
        cost_4o = MOD.estimate_cost(self.PROMPT, self.RESPONSE, model="gpt-4o")
        cost_mini = MOD.estimate_cost(self.PROMPT, self.RESPONSE, model="gpt-4o-mini")
        self.assertLess(cost_mini["total_cost"], cost_4o["total_cost"])


if __name__ == "__main__":
    unittest.main()
