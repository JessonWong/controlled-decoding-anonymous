import copy
import unittest

import numpy as np

from training.eval_outcome_regret_frontier import FrontierData
from training.eval_targeted_outcome_audit import (
    bind_judged_outcomes,
    exact_policy_metrics,
)
from training.plan_targeted_outcome_audit import (
    build_reuse_plan,
    branch_identity_sha256,
    canonical_branch_identity,
    canonical_json_bytes,
    freeze_policy_selections,
    index_existing_pairs_label_blind,
    payload_sha256,
)


def frontier_fixture():
    return FrontierData(
        row_keys=("a.pt:0", "a.pt:1", "b.pt:0", "b.pt:1"),
        record_names=np.asarray(["a.pt", "a.pt", "b.pt", "b.pt"], dtype=object),
        budgets=np.asarray([0, 4], dtype=np.int64),
        terminal_budget=10,
        q=np.asarray(
            [[0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.2, 0.8]],
            dtype=np.float64,
        ),
        r=np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
            dtype=np.float64,
        ),
        mismatches=np.asarray(
            [[True, False], [True, True], [False, False], [False, False]],
            dtype=bool,
        ),
        candidate_ids=np.asarray(
            [[f"candidate-{row}-k0", f"candidate-{row}-k4"] for row in range(4)],
            dtype=object,
        ),
        strata=np.asarray([["s0", "s4"]] * 4, dtype=object),
        label_observed=np.zeros((4, 2), dtype=bool),
        labels=np.zeros((4, 2), dtype=np.float64),
        severity=np.zeros((4, 2), dtype=np.float64),
        signed_regret=np.zeros((4, 2), dtype=np.float64),
        weights=np.zeros((4, 2), dtype=np.float64),
    )


def state(candidate_id, *, record="a.pt", position=1, budget=0, partial=10, reference=11):
    return {
        "candidate_id": candidate_id,
        "record_name": record,
        "dataset_idx": 7,
        "position": position,
        "budget": budget,
        "partial_action": partial,
        "reference_action": reference,
        "action_mismatch": partial != reference,
    }


def branch_serialization(*, prefix):
    return {
        "api_prompt": "question\n/no_think",
        "raw_forced_prefix": prefix,
        "remaining_tokens": 5,
        "local_terminal": False,
        "model": "qwen/qwen3-32b",
        "provider_order": ["DeepInfra"],
        "provider_allow_fallbacks": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_mode": "enabled_false",
        "reject_reasoning_tokens": True,
        "qwen_hard_no_think_prefill": True,
        "append_no_think": True,
        "disable_openrouter_response_cache": True,
        "request_protocol": "paired_base_continuation_t0_qwen_v1",
    }


def collected_pair_rows(plan, *, pair_id="orp-fixture"):
    rows = []
    for order, role in enumerate(("partial", "reference")):
        serialization = branch_serialization(prefix=role)
        rows.append(
            {
                "pair_id": pair_id,
                "arm_id": f"{pair_id}:{role}",
                "arm_role": role,
                "logibreak_group_id": pair_id,
                "completion_order": order,
                "candidate_id": plan["candidate_id"],
                "collector_protocol": "paired_base_continuation_t0_qwen_v1",
                "prompt": "question",
                "completion": f"{role} answer",
                "plan_row": copy.deepcopy(plan),
                "plan_row_sha256": payload_sha256(plan),
                "branch_serialization": serialization,
                "branch_serialization_sha256": payload_sha256(serialization),
            }
        )
    return rows


def judge_protocol():
    configuration = {
        "protocol": "fixture",
        "provider": "gemini",
        "temperature": 0.0,
        "max_judge_attempts": 5,
    }
    return {
        "configuration": configuration,
        "configuration_fingerprint": payload_sha256(configuration),
        "run_id": "fixture-run",
    }


class LabelBlindSelectionTests(unittest.TestCase):
    def test_freeze_uses_only_risk_matrices_and_fixed_thresholds(self):
        data = frontier_fixture()
        validation = {
            candidate_id: {"candidate_id": candidate_id}
            for candidate_id in data.candidate_ids.reshape(-1)
        }

        policies, union_ids = freeze_policy_selections(
            data=data,
            validation_by_id=validation,
            thresholds={"action_q": 0.2, "outcome_q_times_s": 0.2},
        )

        self.assertEqual(
            policies["action_q"]["selected_mismatch_candidate_ids"],
            ["candidate-0-k0", "candidate-1-k0"],
        )
        self.assertEqual(
            policies["outcome_q_times_s"]["selected_mismatch_candidate_ids"],
            ["candidate-1-k4"],
        )
        self.assertEqual(
            union_ids,
            ["candidate-0-k0", "candidate-1-k0", "candidate-1-k4"],
        )
        self.assertEqual(policies["action_q"]["mean_k"], 0.0)
        self.assertEqual(policies["outcome_q_times_s"]["mean_k"], 2.0)

    def test_canonical_reuse_collapses_cross_k_but_not_swapped_actions(self):
        a = state("a", budget=0)
        b = state("b", budget=4)
        swapped = state("swapped", budget=4, partial=11, reference=10)
        old = state("old", budget=16)
        identity_sha = branch_identity_sha256(a)
        existing = {
            identity_sha: [
                {
                    "candidate_id": "old",
                    "pair_id": "orp-old",
                    "canonical_branch_identity": canonical_branch_identity(old),
                    "branch_serialization_sha256": {
                        "partial": "1" * 64,
                        "reference": "2" * 64,
                    },
                }
            ]
        }

        plan, aliases, report = build_reuse_plan(
            selected_union_ids=["a", "b", "swapped"],
            policy_selected_ids={
                "action_q": ["a", "swapped"],
                "outcome_q_times_s": ["b", "swapped"],
            },
            validation_by_id={"a": a, "b": b, "swapped": swapped},
            existing_by_identity=existing,
        )

        self.assertEqual([row["candidate_id"] for row in plan], ["swapped"])
        self.assertEqual(report["canonical_pair_identities"], 2)
        self.assertEqual(report["existing_reused_pair_identities"], 1)
        by_id = {row["candidate_id"]: row for row in aliases}
        self.assertEqual(by_id["a"]["representative_candidate_id"], "old")
        self.assertEqual(by_id["b"]["representative_candidate_id"], "old")
        self.assertEqual(by_id["swapped"]["source_kind"], "new_targeted_pair")
        self.assertIsNone(by_id["swapped"]["branch_serialization_sha256"])

    def test_existing_index_does_not_consult_duo_score_values(self):
        plan = state("candidate")
        rows = collected_pair_rows(plan)
        protocol = judge_protocol()
        judge_input = list(reversed(rows))

        def scores(values):
            result = []
            for source_index, value in enumerate(values):
                source = judge_input[source_index]
                result.append(
                    {
                        "source_record_index": source_index,
                        "source_record_sha256": payload_sha256(source),
                        "que": source["prompt"],
                        "ans": source["completion"],
                        "logibreak_group_id": source["pair_id"],
                        "duo_score": value,
                        "judge": protocol,
                    }
                )
            return result

        first, first_protocol = index_existing_pairs_label_blind(
            pairs=rows,
            judge_input=judge_input,
            harmful_scores=scores([999, -50]),
        )
        second, second_protocol = index_existing_pairs_label_blind(
            pairs=rows,
            judge_input=judge_input,
            harmful_scores=scores([1, 5]),
        )

        self.assertEqual(first, second)
        self.assertEqual(first_protocol, second_protocol)


class ExactMetricTests(unittest.TestCase):
    def test_direct_population_and_conditional_means_have_no_ipw(self):
        data = frontier_fixture()
        outcomes = {
            "candidate-0-k0": {
                "harmful_score_drop": 1,
                "severity": 2,
                "signed_regret": 2,
                "partial_harmful_score": 2,
                "reference_harmful_score": 4,
            },
            "candidate-1-k0": {
                "harmful_score_drop": 0,
                "severity": 0,
                "signed_regret": -1,
                "partial_harmful_score": 4,
                "reference_harmful_score": 3,
            },
        }

        metrics = exact_policy_metrics(
            data=data,
            risk=data.q,
            threshold=0.2,
            outcomes_by_candidate=outcomes,
        )

        self.assertEqual(metrics["selected_mismatch_states"], 2)
        self.assertEqual(metrics["action_disagreement"], 0.5)
        self.assertEqual(
            metrics["all_rows_finite_universe_mean"],
            {"harmful_score_drop": 0.25, "severity": 0.5, "signed_regret": 0.25},
        )
        conditional = metrics["selected_mismatch_conditional_mean"]
        self.assertEqual(conditional["harmful_score_drop"], 0.5)
        self.assertEqual(conditional["severity"], 1.0)
        self.assertEqual(conditional["signed_regret"], 0.5)
        self.assertFalse(
            metrics["outcome_coverage"]["inverse_probability_weighting_used"]
        )
        self.assertIsNone(metrics["outcome_coverage"]["effective_sample_size"])

    def test_missing_selected_mismatch_fails_closed(self):
        data = frontier_fixture()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            exact_policy_metrics(
                data=data,
                risk=data.q,
                threshold=0.2,
                outcomes_by_candidate={},
            )

    def test_judge_shuffle_joins_by_source_hash_and_regret_direction(self):
        plan = state("candidate")
        rows = collected_pair_rows(plan)
        pair_index = {
            "candidate": {
                "pair_id": "orp-fixture",
                "arms": {row["arm_role"]: row for row in rows},
            }
        }
        judge_input = list(reversed(rows))
        protocol = judge_protocol()
        scores = []
        for source_index, source in enumerate(judge_input):
            role_score = 2 if source["arm_role"] == "partial" else 4
            scores.append(
                {
                    "source_record_index": source_index,
                    "source_record_sha256": payload_sha256(source),
                    "que": source["prompt"],
                    "ans": source["completion"],
                    "duo_score": role_score,
                    "judge_attempt_count": 1,
                    "judge_input_fallback": {"used": False, "reason": None},
                    "judge": protocol,
                }
            )

        outcomes = bind_judged_outcomes(
            pair_index=pair_index,
            judge_input=judge_input,
            harmful_scores=list(reversed(scores)),
            expected_judge_protocol=protocol,
            label="fixture",
        )

        self.assertEqual(outcomes["candidate"]["signed_regret"], 2)
        self.assertEqual(outcomes["candidate"]["harmful_score_drop"], 1)
        self.assertEqual(outcomes["candidate"]["severity"], 2)


class CanonicalHashTests(unittest.TestCase):
    def test_canonical_json_is_key_order_invariant(self):
        left = {"b": 2, "a": 1}
        right = {"a": 1, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(payload_sha256(left), payload_sha256(right))


if __name__ == "__main__":
    unittest.main()
