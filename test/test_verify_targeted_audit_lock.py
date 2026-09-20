import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.verify_targeted_audit_lock import (
    EXPECTED_JUDGE_CONFIGURATION,
    EXPECTED_JUDGE_CONFIGURATION_FINGERPRINT,
    EXPECTED_JUDGE_RUNNER_CONTRACT,
    EXPECTED_AUTHENTICATED_PAYLOADS,
    EXPECTED_IMPLEMENTATION_FILE_KEYS,
    EXPECTED_INPUT_FILE_KEYS,
    EXPECTED_RESOURCE_CONTRACT,
    canonical_json_bytes,
    validate_targeted_audit_lock,
)


EVAL_SOURCE = ROOT / "test" / "eval_harmful_score.py"
JUDGE_CLIENT_SOURCE = ROOT / "test" / "judge_client.py"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    plan = tmp_path / "new_pair_plan.jsonl"
    plan_rows = [
        {
            "candidate_id": f"candidate-{index}",
            "record_name": f"record-{index}.pt",
            "dataset_idx": index,
            "position": 10 + index,
            "partial_action": 100 + index,
            "reference_action": 200 + index,
        }
        for index in range(5)
    ]
    plan.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plan_rows),
        encoding="utf-8",
    )
    action_ids = [f"candidate-{index}" for index in range(8)]
    outcome_ids = [f"candidate-{index}" for index in range(3, 13)]
    union_ids = [f"candidate-{index}" for index in range(13)]
    identities = [{"canonical_group": index} for index in range(10)]
    identity_hashes = [
        hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        for identity in identities
    ]
    alias_rows = []
    for index, candidate_id in enumerate(union_ids):
        group = index % 10
        is_new = group < 5
        representative_id = (
            f"candidate-{group}" if is_new else f"existing-candidate-{group}"
        )
        policy_names = []
        if candidate_id in action_ids:
            policy_names.append("action_q")
        if candidate_id in outcome_ids:
            policy_names.append("outcome_q_times_s")
        alias_rows.append(
            {
                "candidate_id": candidate_id,
                "canonical_branch_identity": identities[group],
                "canonical_branch_identity_sha256": identity_hashes[group],
                "policy_names": policy_names,
                "source_kind": (
                    "new_targeted_pair" if is_new else "existing_judged_pair"
                ),
                "representative_candidate_id": representative_id,
                "representative_pair_id": None if is_new else f"pair-{group}",
                "branch_serialization_sha256": None
                if is_new
                else {
                    arm: hashlib.sha256(f"branch-{group}-{arm}".encode()).hexdigest()
                    for arm in ("partial", "reference")
                },
            }
        )
    aliases = tmp_path / "reuse_aliases.jsonl"
    aliases.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in alias_rows),
        encoding="utf-8",
    )
    input_files = {}
    authenticated_payloads = {}
    authenticated_by_input = {
        input_name: (lock_field, embedded_field)
        for lock_field, (input_name, embedded_field) in (
            EXPECTED_AUTHENTICATED_PAYLOADS.items()
        )
    }
    for name in EXPECTED_INPUT_FILE_KEYS:
        path = tmp_path / f"input-{name}.json"
        payload = {"name": name}
        if name in authenticated_by_input:
            lock_field, embedded_field = authenticated_by_input[name]
            payload_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            payload[embedded_field] = payload_sha
            authenticated_payloads[lock_field] = payload_sha
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        input_files[name] = {"path": str(path), "sha256": _file_sha(path)}
    implementation_paths = {
        "collect_outcome_regret_pairs.py": ROOT
        / "training"
        / "collect_outcome_regret_pairs.py",
        "eval_harmful_score.py": EVAL_SOURCE,
        "eval_outcome_regret_frontier.py": ROOT
        / "training"
        / "eval_outcome_regret_frontier.py",
        "eval_targeted_outcome_audit.py": ROOT
        / "training"
        / "eval_targeted_outcome_audit.py",
        "judge_client.py": JUDGE_CLIENT_SOURCE,
        "plan_targeted_outcome_audit.py": ROOT
        / "training"
        / "plan_targeted_outcome_audit.py",
    }
    assert set(implementation_paths) == EXPECTED_IMPLEMENTATION_FILE_KEYS
    implementation_files = {
        name: {"path": str(path), "sha256": _file_sha(path)}
        for name, path in implementation_paths.items()
    }
    lock_path = tmp_path / "targeted_audit_lock.json"
    lock = {
        "lock_schema_version": 1,
        "schema_version": 1,
        "fresh_test_used": False,
        "outcome_labels_used_for_policy_or_threshold_selection": False,
        "deployment_ready": False,
        "guarantees_harmful_score_non_degradation": False,
        "input_files": input_files,
        "implementation_files": implementation_files,
        "authenticated_payloads": authenticated_payloads,
        "selection": {
            "policy_selected_mismatch_candidate_ids": {
                "action_q": action_ids,
                "outcome_q_times_s": outcome_ids,
            },
            "selected_union_candidate_ids": union_ids,
            "canonical_pair_identity_sha256s": identity_hashes,
            "new_representative_candidate_ids": [
                row["candidate_id"] for row in plan_rows
            ],
        },
        "output_files": {
            "new_pair_plan": {"file": plan.name, "sha256": _file_sha(plan)},
            "reuse_aliases": {"file": aliases.name, "sha256": _file_sha(aliases)},
        },
        "resource_contract": dict(EXPECTED_RESOURCE_CONTRACT),
        "judge_protocol": {
            "configuration_fingerprint": EXPECTED_JUDGE_CONFIGURATION_FINGERPRINT,
            "configuration": EXPECTED_JUDGE_CONFIGURATION,
            "runner_contract": EXPECTED_JUDGE_RUNNER_CONTRACT,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(lock)
    ).hexdigest()
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock_path, plan, aliases, lock


def _validate(lock: Path, plan: Path, aliases: Path):
    return validate_targeted_audit_lock(
        lock_path=lock,
        plan_path=plan,
        reuse_aliases_path=aliases,
        eval_source_path=EVAL_SOURCE,
        judge_client_source_path=JUDGE_CLIENT_SOURCE,
    )


def _rewrite_authenticated_lock(path: Path, lock: dict) -> None:
    lock = dict(lock)
    lock.pop("lock_payload_sha256", None)
    lock["lock_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(lock)
    ).hexdigest()
    path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class TargetedAuditLockTests(unittest.TestCase):
    def test_valid_targeted_lock_binds_files_budget_and_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, plan, aliases, _ = _write_fixture(Path(directory))
            report = _validate(lock, plan, aliases)
            self.assertIs(report["verified"], True)
            self.assertEqual(report["new_pair_plan_sha256"], _file_sha(plan))
            self.assertEqual(report["reuse_aliases_sha256"], _file_sha(aliases))
            self.assertEqual(report["new_pairs"], 5)
            self.assertEqual(report["target_api_calls"], 10)
            self.assertEqual(report["requested_output_tokens"], 526)
            self.assertEqual(report["judge_records"], 10)
            self.assertEqual(
                report["judge_configuration_fingerprint"],
                EXPECTED_JUDGE_CONFIGURATION_FINGERPRINT,
            )

    def test_targeted_lock_rejects_plan_changed_after_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, plan, aliases, _ = _write_fixture(Path(directory))
            plan.write_text(plan.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new_pair_plan SHA256 mismatch"):
                _validate(lock, plan, aliases)

    def test_targeted_lock_rejects_unauthenticated_lock_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path, plan, aliases, lock = _write_fixture(Path(directory))
            lock["resource_contract"]["target_api_calls"] = 11
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lock payload hash mismatch"):
                _validate(lock_path, plan, aliases)

    def test_targeted_lock_rejects_authenticated_budget_change(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path, plan, aliases, lock = _write_fixture(Path(directory))
            lock["resource_contract"]["target_api_calls"] = 11
            _rewrite_authenticated_lock(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "resource_contract"):
                _validate(lock_path, plan, aliases)

    def test_targeted_lock_rejects_candidate_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path, plan, aliases, lock = _write_fixture(Path(directory))
            selection = lock["selection"]
            selection["new_representative_candidate_ids"] = list(
                reversed(selection["new_representative_candidate_ids"])
            )
            _rewrite_authenticated_lock(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "new_representative_candidate_ids"):
                _validate(lock_path, plan, aliases)

    def test_targeted_lock_rejects_incomplete_alias_union(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path, plan, aliases, _ = _write_fixture(Path(directory))
            lines = aliases.read_text(encoding="utf-8").splitlines()
            aliases.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["output_files"]["reuse_aliases"]["sha256"] = _file_sha(aliases)
            _rewrite_authenticated_lock(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "exactly 13 rows"):
                _validate(lock_path, plan, aliases)

    def test_targeted_lock_rejects_alias_identity_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path, plan, aliases, _ = _write_fixture(Path(directory))
            rows = [json.loads(line) for line in aliases.read_text().splitlines()]
            rows[0]["canonical_branch_identity"] = {"canonical_group": 999}
            aliases.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["output_files"]["reuse_aliases"]["sha256"] = _file_sha(aliases)
            _rewrite_authenticated_lock(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "canonical identity hash"):
                _validate(lock_path, plan, aliases)

    def test_targeted_lock_rejects_inconsistent_alias_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path, plan, aliases, _ = _write_fixture(Path(directory))
            rows = [json.loads(line) for line in aliases.read_text().splitlines()]
            self.assertEqual(
                rows[10]["canonical_branch_identity_sha256"],
                rows[0]["canonical_branch_identity_sha256"],
            )
            rows[10]["representative_candidate_id"] = "candidate-1"
            aliases.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["output_files"]["reuse_aliases"]["sha256"] = _file_sha(aliases)
            _rewrite_authenticated_lock(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "sharing one canonical identity"):
                _validate(lock_path, plan, aliases)

    @unittest.skip("Machine-specific Slurm wrappers are excluded from the anonymous source release.")
    def test_targeted_sbatch_contract_is_isolated_and_fail_closed(self):
        collect = (
            ROOT
            / "scripts"
            / "submit_qwen3_32b_outcome_regret_targeted_collect.sbatch"
        ).read_text(encoding="utf-8")
        judge = (
            ROOT / "scripts" / "submit_qwen3_32b_outcome_regret_targeted_judge.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('AUDIT_ROOT="${RUN_ROOT}/targeted_exact_audit_v1"', collect)
        self.assertIn('AUDIT_ROOT="${RUN_ROOT}/targeted_exact_audit_v1"', judge)
        self.assertIn("readonly EXPECTED_CALLS=10", collect)
        self.assertIn("readonly EXPECTED_REQUESTED_TOKENS=526", collect)
        self.assertIn('--max_api_request_attempts "${EXPECTED_CALLS}"', collect)
        self.assertIn("readonly EXPECTED_RECORDS=10", judge)
        self.assertIn("--judge_model gemini-3.5-flash", judge)
        self.assertIn("--judge_temperature 0", judge)
        self.assertIn("--judge_seed 1234", judge)
        self.assertIn("--gemini_safety_threshold BLOCK_NONE", judge)
        self.assertIn("--max_judge_attempts 5", judge)
        self.assertIn("Refusing existing targeted collection", collect)
        self.assertIn("Refusing existing targeted judge output", judge)
        self.assertLess(
            collect.index("training/verify_targeted_audit_lock.py"),
            collect.index('[[ -f "${ROOT}/api_key.py" ]]'),
        )
        self.assertLess(
            judge.index("training/verify_targeted_audit_lock.py"),
            judge.index("import api_key"),
        )


if __name__ == "__main__":
    unittest.main()
