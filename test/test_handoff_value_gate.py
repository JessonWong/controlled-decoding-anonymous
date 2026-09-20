import unittest

import numpy as np

from training.train_handoff_value_gate import (
    build_value_samples,
    fit_grouped_model,
    layer_normalize_rows,
    select_handoff_threshold,
    sha256_payload,
)


def source_row(index, *, split="train", record_name=None, role="partial"):
    record_name = record_name or f"record-{index // 2}.pt"
    partial_action = 100 + index
    reference_action = 200 + index
    action = partial_action if role == "partial" else reference_action
    pair_id = f"pair-{index}"
    prefix = f"prefix {index}"
    return {
        "arm_id": f"{pair_id}:{role}",
        "branch_serialization_sha256": f"state-{index}",
        "pair_id": pair_id,
        "logibreak_group_id": pair_id,
        "arm_role": role,
        "action_token_id": action,
        "prompt": f"prompt {index}",
        "forced_prefix_text": prefix,
        "completion": prefix + " continuation",
        "remaining_tokens": 20,
        "local_terminal": False,
        "prefix_tokenization_audit": {"forced_prefix_roundtrip": True},
        "plan_row": {
            "candidate_id": f"candidate-{index}",
            "split": split,
            "record_name": record_name,
            "dataset_idx": index,
            "position": index,
            "position_bin": "early",
            "budget": 4,
            "partial_action": partial_action,
            "reference_action": reference_action,
            "base_token_id": partial_action,
            "feature_names": ["position_fraction"],
            "features": [index / 80.0],
        },
    }


def judged_rows(sources, labels):
    rows = []
    for index, (source, label) in enumerate(zip(sources, labels)):
        rows.append(
            {
                "source_record_index": index,
                "source_record_sha256": sha256_payload(source),
                "que": source["prompt"],
                "ans": source["completion"],
                "duo_score": 5 if label else 4,
                "judge_input_fallback": {"used": False, "reason": None},
            }
        )
    return rows


class ValueSampleTests(unittest.TestCase):
    def test_source_record_index_is_authoritative(self):
        sources = [
            source_row(0, split="train", record_name="train.pt"),
            source_row(1, split="legacy_validation", record_name="validation.pt"),
        ]
        scores = judged_rows(sources, [1, 0])
        scores.reverse()

        samples = build_value_samples(sources, scores)

        self.assertEqual([row["jailbreak_outcome"] for row in samples], [1, 0])
        self.assertTrue(samples[0]["action_equals_base"])
        self.assertNotIn("completion", samples[0])

    def test_hash_mismatch_fails_closed(self):
        sources = [
            source_row(0, split="train", record_name="train.pt"),
            source_row(1, split="legacy_validation", record_name="validation.pt"),
        ]
        scores = judged_rows(sources, [1, 0])
        scores[0]["source_record_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_value_samples(sources, scores)

    def test_prompt_group_cannot_cross_frozen_splits(self):
        sources = [
            source_row(0, split="train", record_name="same.pt"),
            source_row(1, split="legacy_validation", record_name="same.pt"),
        ]
        with self.assertRaisesRegex(ValueError, "cross frozen splits"):
            build_value_samples(sources, judged_rows(sources, [1, 0]))


class GroupedModelTests(unittest.TestCase):
    def test_layer_normalization_is_per_sample(self):
        values = np.asarray([[1.0, 2.0, 3.0], [10.0, 10.0, 12.0]])
        normalized = layer_normalize_rows(values)
        np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
        self.assertTrue(np.isfinite(normalized).all())

    def test_grouped_hidden_fit_scores_every_row(self):
        features = []
        labels = []
        groups = []
        for group in range(6):
            for label in (0, 1):
                features.append([float(label), float(group) / 10.0, 1.0])
                labels.append(label)
                groups.append(f"group-{group}")
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        def factory(c_value, seed):
            from training.train_handoff_value_gate import _make_hidden_estimator

            return _make_hidden_estimator(c_value, seed)

        model, oof, heldout, report = fit_grouped_model(
            features,
            labels,
            np.asarray(groups, dtype=object),
            features[:2],
            estimator_factory=factory,
            c_grid=(0.1, 1.0),
            cv_folds=3,
            seed=7,
        )
        self.assertEqual(oof.shape, labels.shape)
        self.assertEqual(heldout.shape, (2,))
        self.assertTrue(np.isfinite(oof).all())
        self.assertIn(report["selected_C"], (0.1, 1.0))
        self.assertEqual(model.classes_.tolist(), [0, 1])

    def test_threshold_is_selected_only_from_given_scores(self):
        labels = np.asarray([1, 1, 1, 0, 0, 0])
        scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
        selected = select_handoff_threshold(
            labels, scores, minimum_precision=0.9, minimum_handoffs=2
        )
        self.assertEqual(selected["status"], "precision_constraint_satisfied_on_train_oof")
        self.assertGreaterEqual(selected["precision"], 0.9)
        self.assertEqual(selected["handoff_count"], 3)


if __name__ == "__main__":
    unittest.main()
