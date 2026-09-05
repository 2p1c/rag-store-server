import unittest
from unittest.mock import MagicMock, patch

from rag_store.translate import TranslateError, needs_translation, translate_query


class NeedsTranslationTest(unittest.TestCase):
    def test_chinese(self):
        self.assertTrue(needs_translation("苹果"))
        self.assertTrue(needs_translation("RAG 架构"))

    def test_english(self):
        self.assertFalse(needs_translation("apple fruit"))


class TranslateQueryTest(unittest.TestCase):
    def test_english_does_not_call_api(self):
        post = MagicMock()
        self.assertEqual(translate_query("apple", post=post), "apple")
        post.assert_not_called()

    def test_chinese_posts_to_deepseek_and_strips_quotes(self):
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '"apple fruit"\n'}}]
        }
        post = MagicMock(return_value=response)
        with patch(
            "rag_store.translate.deepseek_api_key", return_value="sk-test"
        ):
            out = translate_query("苹果", post=post)
        self.assertEqual(out, "apple fruit")
        url = post.call_args.args[0]
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][1]["content"], "苹果")

    def test_missing_key_raises(self):
        with patch(
            "rag_store.translate.deepseek_api_key", return_value=""
        ):
            with self.assertRaises(TranslateError):
                translate_query("苹果")


if __name__ == "__main__":
    unittest.main()
