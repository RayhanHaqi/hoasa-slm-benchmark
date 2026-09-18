"""Parser and metric tests: strict JSON handling and fixed-class metrics."""

import json
import unittest

from hoasa_benchmark.parsing import compute_metrics, parse_output
from hoasa_benchmark.spec import ASPECTS, LABELS, canonical_json


def gold(target=None):
    target = target or {aspect: "neut" for aspect in ASPECTS}
    return {"id": "test-000000", "target": target}


class ParseOutputTests(unittest.TestCase):
    def test_perfect_canonical_json(self):
        payload = canonical_json({aspect: "neut" for aspect in ASPECTS})
        parsed = parse_output(payload)
        self.assertTrue(parsed.syntax_valid)
        self.assertTrue(parsed.output_valid)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.parsed_payload(), {aspect: "neut" for aspect in ASPECTS})

    def test_key_order_and_whitespace_accepted(self):
        shuffled = ["wifi", "ac"] + [a for a in ASPECTS if a not in ("wifi", "ac")]
        lines = ["{"]
        for index, aspect in enumerate(shuffled):
            comma = "," if index + 1 < len(shuffled) else ""
            lines.append(f'  "{aspect}": "{"neg" if aspect == "wifi" else "pos" if aspect == "ac" else "neut"}"{comma}')
        lines.append("}")
        parsed = parse_output("\n".join(lines))
        self.assertTrue(parsed.schema_valid)
        self.assertEqual(parsed.values["wifi"], "neg")
        self.assertEqual(parsed.values["ac"], "pos")

    def test_one_wrong_value_only_that_aspect_invalid(self):
        target = {aspect: "neut" for aspect in ASPECTS}
        target["tv"] = "positive"
        payload = json.dumps(target, ensure_ascii=False)
        parsed = parse_output(payload)
        self.assertTrue(parsed.output_valid)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.invalid_values, ["tv"])
        self.assertIsNone(parsed.values["tv"])
        self.assertEqual(parsed.values["ac"], "neut")

    def test_missing_key_only_that_aspect_invalid(self):
        target = {aspect: "pos" for aspect in ASPECTS}
        del target["linen"]
        parsed = parse_output(json.dumps(target))
        self.assertEqual(parsed.missing_keys, ["linen"])
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.values["linen"], None)
        self.assertEqual(parsed.values["service"], "pos")

    def test_extra_key_schema_false_but_values_score(self):
        target = {aspect: "neut" for aspect in ASPECTS}
        target["extra"] = "pos"
        parsed = parse_output(json.dumps(target))
        self.assertTrue(parsed.output_valid)
        self.assertFalse(parsed.schema_valid)
        self.assertEqual(parsed.extra_keys, ["extra"])
        self.assertEqual(parsed.parsed_payload(), {aspect: "neut" for aspect in ASPECTS})

    def test_malformed_prose_and_fences_rejected(self):
        for raw in (
            "```json\n{}\n```",
            '{"ac": "neut",}',
            '{"ac": "neut"} extra',
            "",
        ):
            parsed = parse_output(raw)
            self.assertFalse(parsed.syntax_valid, raw)
            self.assertFalse(parsed.output_valid, raw)
            self.assertEqual(parsed.reason, "malformed_json")
            self.assertTrue(all(value is None for value in parsed.values.values()))

    def test_non_object(self):
        for raw in ("[1, 2]", "42", '"neut"', "null"):
            parsed = parse_output(raw)
            self.assertTrue(parsed.syntax_valid, raw)
            self.assertFalse(parsed.output_valid, raw)
            self.assertEqual(parsed.reason, "non_object")
            self.assertFalse(parsed.schema_valid)

    def test_duplicate_keys_syntax_valid_but_output_invalid(self):
        raw = '{"ac": "neg", "ac": "pos", "air_panas": "neut"}'
        parsed = parse_output(raw)
        self.assertTrue(parsed.syntax_valid)
        self.assertTrue(parsed.duplicate_keys)
        self.assertFalse(parsed.output_valid)
        self.assertFalse(parsed.schema_valid)
        self.assertIsNone(parsed.parsed_payload())
        self.assertEqual(parsed.reason, "duplicate_keys")
        self.assertTrue(all(value is None for value in parsed.values.values()))

    def test_validity_payload_keys(self):
        parsed = parse_output(json.dumps({aspect: "neut" for aspect in ASPECTS}))
        self.assertEqual(
            set(parsed.validity()),
            {
                "syntax_valid",
                "output_valid",
                "schema_valid",
                "duplicate_keys",
                "missing_keys",
                "invalid_values",
                "extra_keys",
                "reason",
            },
        )


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        records = [
            {"id": f"test-{i:06d}", "target": {a: LABELS[i % 4] for a in ASPECTS}}
            for i in range(4)
        ]
        parses = [parse_output(canonical_json(record["target"])) for record in records]
        metrics = compute_metrics(records, parses)
        self.assertEqual(metrics["mean_aspect_macro_f1"], 1.0)
        self.assertEqual(metrics["overall_aspect_accuracy"], 1.0)
        self.assertEqual(metrics["whole_review_exact_accuracy"], 1.0)
        self.assertEqual(metrics["syntax_valid_rate"], 1.0)
        self.assertEqual(metrics["schema_valid_rate"], 1.0)

    def test_one_wrong_aspect(self):
        target = {aspect: "neut" for aspect in ASPECTS}
        records = [{"id": "test-000000", "target": target}]
        wrong = dict(target, tv="neg")
        metrics = compute_metrics(records, [parse_output(json.dumps(wrong))])
        self.assertAlmostEqual(metrics["overall_aspect_accuracy"], 9 / 10)
        self.assertEqual(metrics["whole_review_exact"], 0)
        self.assertEqual(metrics["per_aspect"]["tv"]["accuracy"], 0.0)
        self.assertEqual(metrics["per_aspect"]["ac"]["accuracy"], 1.0)

    def test_invalid_counts_in_denominator_and_macro(self):
        target = {aspect: "neg" for aspect in ASPECTS}
        records = [{"id": "test-000000", "target": target}]
        metrics = compute_metrics(records, [parse_output("not json")])
        self.assertEqual(metrics["overall_aspect_accuracy"], 0.0)
        self.assertEqual(metrics["mean_aspect_macro_f1"], 0.0)
        self.assertEqual(metrics["per_aspect"]["ac"]["confusion"]["neg"]["INVALID"], 1)
        self.assertEqual(metrics["output_valid_rate"], 0.0)

    def test_all_invalid_with_mixed_golds(self):
        records = [
            {"id": "test-000000", "target": {a: "neg" for a in ASPECTS}},
            {"id": "test-000001", "target": {a: "pos" for a in ASPECTS}},
        ]
        metrics = compute_metrics(records, [parse_output(""), parse_output("[1]")])
        self.assertEqual(metrics["overall_aspect_accuracy"], 0.0)
        self.assertEqual(metrics["whole_review_exact"], 0)
        self.assertEqual(metrics["invalid_predictions"], 2)

    def test_zero_support_class(self):
        # neg_pos never occurs as gold: zero_division=0 must not raise or score it.
        target = {aspect: "neut" for aspect in ASPECTS}
        records = [
            {"id": "test-000000", "target": target},
            {"id": "test-000001", "target": dict(target, ac="neg")},
        ]
        parses = [parse_output(json.dumps(target)), parse_output(json.dumps(target))]
        metrics = compute_metrics(records, parses)
        self.assertEqual(metrics["per_aspect"]["ac"]["per_class"]["neg_pos"]["support"], 0)
        self.assertEqual(metrics["per_aspect"]["ac"]["per_class"]["neg_pos"]["f1"], 0.0)
        self.assertGreater(metrics["mean_aspect_macro_f1"], 0.0)

    def test_empty_records(self):
        metrics = compute_metrics([], [])
        self.assertEqual(metrics["examples"], 0)
        self.assertEqual(metrics["overall_aspect_accuracy"], 0.0)
        self.assertEqual(metrics["mean_aspect_macro_f1"], 0.0)
        self.assertEqual(metrics["syntax_valid_rate"], 0.0)

    def test_missing_aspect_invalid_only_that_aspect(self):
        target = {aspect: "neut" for aspect in ASPECTS}
        partial = {aspect: "neut" for aspect in ASPECTS if aspect != "wifi"}
        records = [{"id": "test-000000", "target": target}]
        metrics = compute_metrics(records, [parse_output(json.dumps(partial))])
        self.assertAlmostEqual(metrics["overall_aspect_accuracy"], 9 / 10)
        self.assertEqual(metrics["per_aspect"]["wifi"]["accuracy"], 0.0)
        self.assertEqual(metrics["per_aspect"]["ac"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
