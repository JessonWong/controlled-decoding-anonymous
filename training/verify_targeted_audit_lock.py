#!/usr/bin/env python3
"""Fail-closed verifier for the targeted outcome-regret audit lock.

The targeted collector and judge deliberately share this verifier.  It binds
the five-row paid plan, the reuse map, the exact resource envelope, and the
already-used Gemini judge protocol before either batch script reads an API
key.  This module performs local reads only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_RESOURCE_CONTRACT = {
    "new_pairs": 5,
    "target_api_calls": 10,
    "requested_output_tokens": 526,
    "judge_records": 10,
}

EXPECTED_JUDGE_CONFIGURATION = {
    "protocol": "juli_meta_policy_duo_1_to_5_v1",
    "success_score": 5,
    "provider": "gemini",
    "requested_model": "gemini-3.5-flash",
    "max_output_tokens": 4096,
    "temperature": 0.0,
    "configured_seed": 1234,
    "request_seed": 1234,
    "gemini_safety_threshold": "BLOCK_NONE",
    "max_judge_attempts": 5,
    "retry_sleep_seconds": 5.0,
    "local_implementation_sha256": {
        "eval_harmful_score.py": (
            "bdea2c6221464d6b10700a7515273192f8794420bd6b100be9b0b9ba32c7bcfe"
        ),
        "judge_client.py": (
            "13e1b7f0f8cff3a0886eb55d2b91685618faeb862ae431ca0e73cbd1fa900c16"
        ),
    },
}
EXPECTED_JUDGE_CONFIGURATION_FINGERPRINT = (
    "639bb5ec66e7c51a7a73282d3ba2d10fdfb7a874f1bbb72168442ce52206bd0e"
)
EXPECTED_JUDGE_RUNNER_CONTRACT = {
    "gemini_thinking_budget": None,
    "gemini_blocked_prompt_fallback": True,
    "continue_on_judge_failure": True,
}
EXPECTED_POLICY_SELECTION_SIZES = {
    "action_q": 8,
    "outcome_q_times_s": 10,
}
EXPECTED_INPUT_FILE_KEYS = {
    "existing_collection_audit",
    "existing_collection_manifest",
    "existing_harmful_score",
    "existing_judge_input",
    "existing_pairs",
    "fit_summary",
    "frontier_summary",
    "outcome_gate",
    "paid_pair_plan",
    "pair_predictions",
    "plan_manifest",
    "policy_spec",
    "state_universe",
}
EXPECTED_IMPLEMENTATION_FILE_KEYS = {
    "collect_outcome_regret_pairs.py",
    "eval_harmful_score.py",
    "eval_outcome_regret_frontier.py",
    "eval_targeted_outcome_audit.py",
    "judge_client.py",
    "plan_targeted_outcome_audit.py",
}
EXPECTED_AUTHENTICATED_PAYLOADS = {
    "fit_summary_payload_sha256": ("fit_summary", "summary_payload_sha256"),
    "frontier_summary_payload_sha256": (
        "frontier_summary",
        "summary_payload_sha256",
    ),
    "outcome_gate_model_payload_sha256": ("outcome_gate", "model_payload_sha256"),
    "plan_manifest_payload_sha256": ("plan_manifest", "manifest_payload_sha256"),
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def strict_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest.")
    return value


def _resolved_declared_file(lock_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.file must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = lock_path.parent / path
    return path.resolve()


def _validate_bound_file(
    *,
    lock_path: Path,
    descriptor: Any,
    actual_path: Path,
    label: str,
) -> str:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} must be an object with file and sha256 fields.")
    declared_path = _resolved_declared_file(
        lock_path, descriptor.get("file"), label=label
    )
    expected_path = actual_path.expanduser().resolve()
    if declared_path != expected_path:
        raise ValueError(
            f"{label} path mismatch: lock={declared_path}, expected={expected_path}."
        )
    if not expected_path.is_file():
        raise FileNotFoundError(f"Missing {label} file: {expected_path}")
    expected_sha = _require_sha256(descriptor.get("sha256"), label=f"{label}.sha256")
    actual_sha = sha256_file(expected_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{label} SHA256 mismatch: lock={expected_sha}, actual={actual_sha}."
        )
    return actual_sha


def _validate_locked_file_group(
    lock: Mapping[str, Any], *, field: str, expected_keys: set[str]
) -> dict[str, str]:
    descriptors = lock.get(field)
    if not isinstance(descriptors, Mapping) or set(descriptors) != expected_keys:
        raise ValueError(f"Lock {field} must contain exactly {sorted(expected_keys)}.")
    verified: dict[str, str] = {}
    for name in sorted(expected_keys):
        descriptor = descriptors[name]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Lock {field}.{name} must be a path/SHA256 object.")
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Lock {field}.{name}.path must be a non-empty string.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError(f"Lock {field}.{name}.path must be absolute.")
        path = path.resolve()
        if field == "implementation_files" and path.name != name:
            raise ValueError(
                f"Lock implementation_files.{name} points to basename {path.name}."
            )
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked {field}.{name}: {path}")
        expected_sha = _require_sha256(
            descriptor.get("sha256"), label=f"{field}.{name}.sha256"
        )
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Locked {field}.{name} SHA256 mismatch: "
                f"lock={expected_sha}, actual={actual_sha}."
            )
        verified[name] = actual_sha
    return verified


def _validate_authenticated_payloads(lock: Mapping[str, Any]) -> dict[str, str]:
    declared = lock.get("authenticated_payloads")
    if not isinstance(declared, Mapping) or set(declared) != set(
        EXPECTED_AUTHENTICATED_PAYLOADS
    ):
        raise ValueError(
            "Lock authenticated_payloads must contain exactly "
            f"{sorted(EXPECTED_AUTHENTICATED_PAYLOADS)}."
        )
    input_files = lock["input_files"]
    verified: dict[str, str] = {}
    for lock_field, (input_name, embedded_field) in (
        EXPECTED_AUTHENTICATED_PAYLOADS.items()
    ):
        expected_sha = _require_sha256(
            declared[lock_field], label=f"authenticated_payloads.{lock_field}"
        )
        path = Path(input_files[input_name]["path"]).expanduser().resolve()
        payload = strict_json_loads(
            path.read_text(encoding="utf-8"), label=f"authenticated {input_name}"
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Authenticated {input_name} must be a JSON object.")
        embedded_sha = _require_sha256(
            payload.get(embedded_field), label=f"{input_name}.{embedded_field}"
        )
        unsigned_payload = dict(payload)
        del unsigned_payload[embedded_field]
        actual_sha = sha256_bytes(canonical_json_bytes(unsigned_payload))
        if expected_sha != embedded_sha or expected_sha != actual_sha:
            raise ValueError(
                f"Authenticated payload mismatch for {input_name}: "
                f"lock={expected_sha}, embedded={embedded_sha}, actual={actual_sha}."
            )
        verified[lock_field] = actual_sha
    return verified


def _load_plan_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = strict_json_loads(line, label=f"plan row {line_number}")
        if not isinstance(row, dict):
            raise ValueError(f"Plan row {line_number} is not a JSON object.")
        rows.append(row)
    if len(rows) != EXPECTED_RESOURCE_CONTRACT["new_pairs"]:
        raise ValueError(
            "Targeted plan must contain exactly "
            f"{EXPECTED_RESOURCE_CONTRACT['new_pairs']} rows, found {len(rows)}."
        )
    candidate_ids = [row.get("candidate_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in candidate_ids):
        raise ValueError("Every targeted plan row must carry a non-empty candidate_id.")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Targeted plan candidate_id values must be unique.")
    if any(row.get("partial_action") == row.get("reference_action") for row in rows):
        raise ValueError("Every targeted plan row must be an action mismatch.")
    return rows


def _require_unique_string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates.")
    return list(value)


def _validate_selection(
    lock: Mapping[str, Any], plan_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selection = lock.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Lock selection must be an object.")
    policy_ids = selection.get("policy_selected_mismatch_candidate_ids")
    if not isinstance(policy_ids, Mapping) or set(policy_ids) != set(
        EXPECTED_POLICY_SELECTION_SIZES
    ):
        raise ValueError(
            "selection.policy_selected_mismatch_candidate_ids must contain exactly "
            "action_q and outcome_q_times_s."
        )
    normalized_policy_ids: dict[str, list[str]] = {}
    for policy_name, expected_size in EXPECTED_POLICY_SELECTION_SIZES.items():
        values = _require_unique_string_list(
            policy_ids[policy_name],
            label=(
                "selection.policy_selected_mismatch_candidate_ids."
                f"{policy_name}"
            ),
        )
        if len(values) != expected_size:
            raise ValueError(
                f"Policy {policy_name} must select {expected_size} mismatches, "
                f"found {len(values)}."
            )
        normalized_policy_ids[policy_name] = values

    union_ids = _require_unique_string_list(
        selection.get("selected_union_candidate_ids"),
        label="selection.selected_union_candidate_ids",
    )
    if len(union_ids) != 13:
        raise ValueError("selection.selected_union_candidate_ids must contain 13 IDs.")
    computed_union = set().union(*(set(values) for values in normalized_policy_ids.values()))
    if set(union_ids) != computed_union:
        raise ValueError("Selected union IDs do not equal the union of both policies.")

    identity_hashes = _require_unique_string_list(
        selection.get("canonical_pair_identity_sha256s"),
        label="selection.canonical_pair_identity_sha256s",
    )
    if len(identity_hashes) != 10:
        raise ValueError("selection.canonical_pair_identity_sha256s must contain 10 IDs.")
    for index, identity_hash in enumerate(identity_hashes):
        _require_sha256(
            identity_hash,
            label=f"selection.canonical_pair_identity_sha256s[{index}]",
        )

    new_representative_ids = _require_unique_string_list(
        selection.get("new_representative_candidate_ids"),
        label="selection.new_representative_candidate_ids",
    )
    if len(new_representative_ids) != 5:
        raise ValueError(
            "selection.new_representative_candidate_ids must contain 5 IDs."
        )
    if any(candidate_id not in computed_union for candidate_id in new_representative_ids):
        raise ValueError("Every new representative must belong to the selected union.")
    plan_ids = [str(row["candidate_id"]) for row in plan_rows]
    if plan_ids != new_representative_ids:
        raise ValueError(
            "Plan candidate IDs must exactly match ordered "
            "selection.new_representative_candidate_ids."
        )
    return {
        "policy_ids": normalized_policy_ids,
        "union_ids": union_ids,
        "identity_hashes": identity_hashes,
        "new_representative_ids": new_representative_ids,
    }


def _load_reuse_alias_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = strict_json_loads(line, label=f"reuse alias row {line_number}")
        if not isinstance(row, dict):
            raise ValueError(f"Reuse alias row {line_number} is not an object.")
        rows.append(row)
    if len(rows) != 13:
        raise ValueError(f"Reuse alias map must contain exactly 13 rows, found {len(rows)}.")
    return rows


def _validate_reuse_aliases(
    path: Path, selection: Mapping[str, Any]
) -> dict[str, int]:
    rows = _load_reuse_alias_rows(path)
    required_fields = {
        "candidate_id",
        "canonical_branch_identity",
        "canonical_branch_identity_sha256",
        "policy_names",
        "source_kind",
        "representative_candidate_id",
        "representative_pair_id",
        "branch_serialization_sha256",
    }
    by_candidate: dict[str, dict[str, Any]] = {}
    observed_identity_hashes: set[str] = set()
    observed_new_representatives: set[str] = set()
    identity_signatures: dict[str, tuple[str, str, Any, bytes]] = {}
    source_kind_counts = {
        "existing_judged_pair": 0,
        "new_targeted_pair": 0,
    }
    policy_ids = selection["policy_ids"]
    new_representative_ids = set(selection["new_representative_ids"])
    for index, row in enumerate(rows):
        missing = required_fields - set(row)
        if missing:
            raise ValueError(
                f"Reuse alias row {index} is missing fields: {sorted(missing)}."
            )
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"Reuse alias row {index} has an invalid candidate_id.")
        if candidate_id in by_candidate:
            raise ValueError(f"Duplicate reuse alias candidate_id: {candidate_id}.")
        by_candidate[candidate_id] = row

        identity_sha = _require_sha256(
            row["canonical_branch_identity_sha256"],
            label=f"reuse alias row {index} canonical identity SHA256",
        )
        computed_identity_sha = sha256_bytes(
            canonical_json_bytes(row["canonical_branch_identity"])
        )
        if identity_sha != computed_identity_sha:
            raise ValueError(
                f"Reuse alias row {index} canonical identity hash is invalid."
            )
        observed_identity_hashes.add(identity_sha)
        policy_names = _require_unique_string_list(
            row["policy_names"], label=f"reuse alias row {index} policy_names"
        )
        expected_policies = {
            policy_name
            for policy_name, candidate_ids in policy_ids.items()
            if candidate_id in candidate_ids
        }
        if set(policy_names) != expected_policies:
            raise ValueError(
                f"Reuse alias row {index} policy_names disagree with lock selection."
            )

        source_kind = row["source_kind"]
        if source_kind not in source_kind_counts:
            raise ValueError(
                f"Reuse alias row {index} source_kind must be "
                "existing_judged_pair or new_targeted_pair."
            )
        source_kind_counts[source_kind] += 1
        representative_id = row["representative_candidate_id"]
        representative_pair_id = row["representative_pair_id"]
        if not isinstance(representative_id, str) or not representative_id:
            raise ValueError(
                f"Reuse alias row {index} has an invalid representative_candidate_id."
            )
        branch_hashes = row["branch_serialization_sha256"]
        if source_kind == "new_targeted_pair":
            if representative_id not in new_representative_ids:
                raise ValueError(
                    "A new reuse alias points outside new representative candidate IDs."
                )
            if representative_pair_id is not None:
                raise ValueError(
                    "A new targeted reuse alias must leave representative_pair_id null "
                    "until collection."
                )
            if branch_hashes is not None:
                raise ValueError(
                    "A new targeted reuse alias must leave branch serialization hashes "
                    "null until collection."
                )
            observed_new_representatives.add(representative_id)
        else:
            if representative_id in new_representative_ids:
                raise ValueError("An existing reuse alias points to a new representative.")
            if not isinstance(representative_pair_id, str) or not representative_pair_id:
                raise ValueError(
                    "An existing judged reuse alias must carry representative_pair_id."
                )
            if not isinstance(branch_hashes, Mapping) or set(branch_hashes) != {
                "partial",
                "reference",
            }:
                raise ValueError(
                    "An existing judged reuse alias must carry ordered partial/reference "
                    "branch serialization hashes."
                )
            for arm_role in ("partial", "reference"):
                _require_sha256(
                    branch_hashes[arm_role],
                    label=(
                        f"reuse alias row {index} {arm_role} branch serialization SHA256"
                    ),
                )

        signature = (
            source_kind,
            representative_id,
            representative_pair_id,
            canonical_json_bytes(branch_hashes),
        )
        previous_signature = identity_signatures.setdefault(identity_sha, signature)
        if previous_signature != signature:
            raise ValueError(
                "Reuse aliases sharing one canonical identity disagree on source or "
                "representative evidence."
            )

    if set(by_candidate) != set(selection["union_ids"]):
        raise ValueError("Reuse alias candidate IDs do not exactly cover the selected union.")
    if observed_identity_hashes != set(selection["identity_hashes"]):
        raise ValueError(
            "Reuse alias canonical identities do not exactly cover the 10 locked identities."
        )
    if observed_new_representatives != new_representative_ids:
        raise ValueError(
            "Reuse aliases do not exactly cover all five new representatives."
        )
    new_identity_signatures = [
        signature
        for signature in identity_signatures.values()
        if signature[0] == "new_targeted_pair"
    ]
    existing_identity_signatures = [
        signature
        for signature in identity_signatures.values()
        if signature[0] == "existing_judged_pair"
    ]
    if len(new_identity_signatures) != 5 or len(existing_identity_signatures) != 5:
        raise ValueError(
            "The 10 canonical identities must split into exactly five existing judged "
            "pairs and five new targeted pairs."
        )
    if len({signature[1] for signature in existing_identity_signatures}) != 5:
        raise ValueError("Existing canonical identities must use five representatives.")
    if len({signature[2] for signature in existing_identity_signatures}) != 5:
        raise ValueError("Existing canonical identities must bind five distinct pair IDs.")
    for representative_id in new_representative_ids:
        representative_row = by_candidate.get(representative_id)
        if (
            representative_row is None
            or representative_row["source_kind"] != "new_targeted_pair"
            or representative_row["representative_candidate_id"] != representative_id
        ):
            raise ValueError(
                f"New representative {representative_id} lacks a self-mapping new row."
            )
    if not all(source_kind_counts.values()):
        raise ValueError("Reuse alias map must contain both existing and new sources.")
    return source_kind_counts


def validate_targeted_audit_lock(
    *,
    lock_path: Path,
    plan_path: Path,
    reuse_aliases_path: Path,
    eval_source_path: Path,
    judge_client_source_path: Path,
) -> dict[str, Any]:
    """Validate all immutable inputs needed by targeted collection and judging."""

    lock_path = lock_path.expanduser().resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing targeted audit lock: {lock_path}")
    lock = strict_json_loads(
        lock_path.read_text(encoding="utf-8"), label="targeted audit lock"
    )
    if not isinstance(lock, dict):
        raise ValueError("Targeted audit lock must be a JSON object.")

    expected_payload_sha = _require_sha256(
        lock.get("lock_payload_sha256"), label="lock_payload_sha256"
    )
    payload = dict(lock)
    del payload["lock_payload_sha256"]
    actual_payload_sha = sha256_bytes(canonical_json_bytes(payload))
    if actual_payload_sha != expected_payload_sha:
        raise ValueError(
            "Targeted audit lock payload hash mismatch: "
            f"lock={expected_payload_sha}, actual={actual_payload_sha}."
        )

    if lock.get("schema_version") != 1:
        raise ValueError("Targeted audit lock schema_version must be 1.")
    expected_semantic_flags = {
        "fresh_test_used": False,
        "outcome_labels_used_for_policy_or_threshold_selection": False,
        "deployment_ready": False,
        "guarantees_harmful_score_non_degradation": False,
    }
    for field, expected in expected_semantic_flags.items():
        if lock.get(field) is not expected:
            raise ValueError(
                f"Targeted audit semantic flag {field} must be {expected!r}."
            )

    verified_input_files = _validate_locked_file_group(
        lock, field="input_files", expected_keys=EXPECTED_INPUT_FILE_KEYS
    )
    verified_implementation_files = _validate_locked_file_group(
        lock,
        field="implementation_files",
        expected_keys=EXPECTED_IMPLEMENTATION_FILE_KEYS,
    )
    verified_authenticated_payloads = _validate_authenticated_payloads(lock)

    output_files = lock.get("output_files")
    if not isinstance(output_files, Mapping):
        raise ValueError("Lock output_files must be an object.")
    plan_sha = _validate_bound_file(
        lock_path=lock_path,
        descriptor=output_files.get("new_pair_plan"),
        actual_path=plan_path,
        label="output_files.new_pair_plan",
    )
    reuse_sha = _validate_bound_file(
        lock_path=lock_path,
        descriptor=output_files.get("reuse_aliases"),
        actual_path=reuse_aliases_path,
        label="output_files.reuse_aliases",
    )

    resource_contract = lock.get("resource_contract")
    if resource_contract != EXPECTED_RESOURCE_CONTRACT:
        raise ValueError(
            "Targeted resource_contract must be exactly "
            f"{EXPECTED_RESOURCE_CONTRACT!r}, found {resource_contract!r}."
        )

    judge_protocol = lock.get("judge_protocol")
    if not isinstance(judge_protocol, Mapping):
        raise ValueError("Lock judge_protocol must be an object.")
    fingerprint = _require_sha256(
        judge_protocol.get("configuration_fingerprint"),
        label="judge_protocol.configuration_fingerprint",
    )
    configuration = judge_protocol.get("configuration")
    if configuration != EXPECTED_JUDGE_CONFIGURATION:
        raise ValueError("Lock does not pin the exact existing Gemini judge configuration.")
    computed_fingerprint = sha256_bytes(canonical_json_bytes(configuration))
    if fingerprint != computed_fingerprint:
        raise ValueError("Lock judge configuration fingerprint is internally invalid.")
    if fingerprint != EXPECTED_JUDGE_CONFIGURATION_FINGERPRINT:
        raise ValueError("Lock judge configuration fingerprint is not the frozen protocol.")
    if judge_protocol.get("runner_contract") != EXPECTED_JUDGE_RUNNER_CONTRACT:
        raise ValueError(
            "Lock does not pin the exact targeted Gemini judge runner contract."
        )

    expected_implementation_hashes = EXPECTED_JUDGE_CONFIGURATION[
        "local_implementation_sha256"
    ]
    actual_eval_sha = sha256_file(eval_source_path.expanduser().resolve())
    actual_client_sha = sha256_file(judge_client_source_path.expanduser().resolve())
    if actual_eval_sha != expected_implementation_hashes["eval_harmful_score.py"]:
        raise ValueError("eval_harmful_score.py drifted from the frozen judge protocol.")
    if actual_client_sha != expected_implementation_hashes["judge_client.py"]:
        raise ValueError("judge_client.py drifted from the frozen judge protocol.")
    if actual_eval_sha != verified_implementation_files["eval_harmful_score.py"]:
        raise ValueError("Judge eval hash disagrees with lock implementation_files.")
    if actual_client_sha != verified_implementation_files["judge_client.py"]:
        raise ValueError("Judge client hash disagrees with lock implementation_files.")

    rows = _load_plan_rows(plan_path.expanduser().resolve())
    selection = _validate_selection(lock, rows)
    source_kind_counts = _validate_reuse_aliases(
        reuse_aliases_path.expanduser().resolve(), selection
    )
    return {
        "verified": True,
        "lock_file_sha256": sha256_file(lock_path),
        "lock_payload_sha256": actual_payload_sha,
        "new_pair_plan_sha256": plan_sha,
        "reuse_aliases_sha256": reuse_sha,
        "new_pairs": len(rows),
        "target_api_calls": EXPECTED_RESOURCE_CONTRACT["target_api_calls"],
        "requested_output_tokens": EXPECTED_RESOURCE_CONTRACT[
            "requested_output_tokens"
        ],
        "judge_records": EXPECTED_RESOURCE_CONTRACT["judge_records"],
        "new_representative_candidate_ids": [row["candidate_id"] for row in rows],
        "selected_union_candidates": len(selection["union_ids"]),
        "canonical_pair_identities": len(selection["identity_hashes"]),
        "reuse_existing_rows": source_kind_counts["existing_judged_pair"],
        "reuse_new_rows": source_kind_counts["new_targeted_pair"],
        "authenticated_input_files": len(verified_input_files),
        "authenticated_implementation_files": len(verified_implementation_files),
        "authenticated_embedded_payloads": len(verified_authenticated_payloads),
        "authenticated_input_file_sha256": verified_input_files,
        "authenticated_implementation_file_sha256": verified_implementation_files,
        "authenticated_payload_sha256": verified_authenticated_payloads,
        "judge_configuration_fingerprint": fingerprint,
        "eval_harmful_score_sha256": actual_eval_sha,
        "judge_client_sha256": actual_client_sha,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the immutable targeted outcome-regret audit lock."
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--reuse_aliases", required=True)
    parser.add_argument("--eval_source", required=True)
    parser.add_argument("--judge_client_source", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_targeted_audit_lock(
        lock_path=Path(args.lock),
        plan_path=Path(args.plan),
        reuse_aliases_path=Path(args.reuse_aliases),
        eval_source_path=Path(args.eval_source),
        judge_client_source_path=Path(args.judge_client_source),
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
