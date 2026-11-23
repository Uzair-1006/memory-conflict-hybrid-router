import importlib.util
import os
import unittest
from unittest.mock import patch

from memcore.llm import OpenAIChat


@unittest.skipUnless(importlib.util.find_spec("openai"), "openai package not installed")
class OpenAIChatConfigurationTests(unittest.TestCase):
    def setUp(self):
        env_loader = patch("memcore.llm._load_env_file")
        env_loader.start()
        self.addCleanup(env_loader.stop)

    def test_accepts_string_chat_completion_response(self):
        class Completions:
            def create(self, **kwargs):
                return '{"ok": true}'

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        client = OpenAIChat.__new__(OpenAIChat)
        client._client = Client()
        client.model = "openai/gpt-4o-mini"
        client.temperature = 0.0

        self.assertEqual(client.complete_json("test"), '{"ok": true}')

    def test_rejects_placeholder_key_in_env(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "replace_with_your_rotated_key",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "No valid API key"):
                OpenAIChat()

    def test_openrouter_key_uses_openrouter_endpoint(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "sk-or-v1-test",
            "PIPELINE_MODEL": "openai/gpt-4o-mini",
        }, clear=True):
            client = OpenAIChat()
            self.assertEqual(client.model, "openai/gpt-4o-mini")
            self.assertEqual(
                str(client._client.base_url),
                "https://openrouter.ai/api/v1/",
            )

    def test_github_models_token_uses_github_endpoint(self):
        with patch.dict(os.environ, {
            "GITHUB_TOKEN": "github_pat_test",
            "GITHUB_MODELS_MODEL": "openai/gpt-4o-mini",
        }, clear=True):
            client = OpenAIChat()
            self.assertEqual(
                str(client._client.base_url),
                "https://models.github.ai/inference/",
            )


if __name__ == "__main__":
    unittest.main()
