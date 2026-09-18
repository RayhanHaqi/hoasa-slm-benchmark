"""Truncation/prompt tests with a fake character-level tokenizer."""

import unittest

from hoasa_benchmark.prompting import (
    build_generation_input,
    build_training_text,
    truncated_user_text,
)


class FakeTokenizer:
    """One character == one token id (ord); deterministic chat template."""

    eos_token = "<eos>"
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, **kwargs):
        ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            ids = ids[: max(0, int(max_length))]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class TruncationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_truncate_review_alone(self):
        review = "x" * 100
        text, tokens, truncated = truncated_user_text(self.tokenizer, review, 10)
        self.assertEqual(tokens, 10)
        self.assertTrue(truncated)
        self.assertEqual(text, "x" * 10)

    def test_generation_reserves_new_tokens(self):
        review = "y" * 200
        built = build_generation_input(
            self.tokenizer,
            review,
            "SYS",
            max_user_tokens=100,
            max_seq_length=60,
            max_new_tokens=5,
        )
        self.assertLess(built["user_tokens"], 100)
        self.assertTrue(built["truncated"])
        self.assertTrue(built["reduced_below_limit"])
        self.assertLessEqual(built["prompt_tokens"] + 5, 60)

    def test_generation_within_limit_keeps_full_review(self):
        review = "z" * 20
        built = build_generation_input(
            self.tokenizer,
            review,
            "SYS",
            max_user_tokens=100,
            max_seq_length=1000,
            max_new_tokens=5,
        )
        self.assertEqual(built["user_tokens"], 20)
        self.assertFalse(built["truncated"])
        self.assertFalse(built["reduced_below_limit"])
        self.assertIn(review, built["prompt"])

    def test_training_text_preserves_target_and_eos(self):
        review = "a" * 300
        target = '{"ac": "neg"}'
        built = build_training_text(
            self.tokenizer,
            review,
            target,
            "SYS",
            max_user_tokens=1500,
            max_seq_length=2048,
            max_new_tokens=256,
        )
        self.assertIn(target, built["text"])
        self.assertTrue(built["text"].endswith("<eos>"))
        self.assertTrue(built["eos_present"])
        self.assertLessEqual(built["full_tokens"], 2048)
        self.assertEqual(built["target_tokens"], len(target))

    def test_training_shrinks_user_before_target(self):
        review = "b" * 300
        target = "T" * 50
        built = build_training_text(
            self.tokenizer,
            review,
            target,
            "S",
            max_user_tokens=200,
            max_seq_length=130,
            max_new_tokens=10,
        )
        self.assertIn(target, built["text"])
        self.assertLess(built["user_tokens"], 200)
        self.assertLessEqual(built["full_tokens"], 130)
        self.assertEqual(built["target_tokens"], 50)

    def test_training_refuses_when_target_alone_does_not_fit(self):
        with self.assertRaises(ValueError):
            build_training_text(
                self.tokenizer,
                "c" * 50,
                "T" * 500,
                "S",
                max_user_tokens=100,
                max_seq_length=100,
                max_new_tokens=10,
            )

    def test_identical_user_truncation_training_and_inference(self):
        review = "d" * 500
        generation = build_generation_input(
            self.tokenizer,
            review,
            "SYS",
            max_user_tokens=100,
            max_seq_length=400,
            max_new_tokens=50,
        )
        training = build_training_text(
            self.tokenizer,
            review,
            '{"ac": "neg"}',
            "SYS",
            max_user_tokens=100,
            max_seq_length=400,
            max_new_tokens=50,
        )
        self.assertEqual(generation["user_text"], training["user_text"])
        self.assertEqual(generation["user_tokens"], training["user_tokens"])


if __name__ == "__main__":
    unittest.main()
