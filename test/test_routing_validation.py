import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_validation import RoutingValidationError, validate_routing_attribution


def routing_metadata(**overrides):
    metadata = {
        "model": "z-ai/glm-5",
        "provider_order": ["DeepInfra"],
        "provider_allow_fallbacks": False,
        "router_metadata_requested": True,
        "api_calls": 1000,
        "actual_provider_call_counts": {"DeepInfra": 1000},
        "actual_response_model_call_counts": {"z-ai/glm-5": 1000},
        "missing_router_metadata_calls": 0,
        "max_routing_attempt": 1,
        "disable_openrouter_response_cache": True,
        "response_cache_status_call_counts": {"<missing>": 1000},
        "actual_provider_counts": {"DeepInfra": 2000},
        "actual_response_model_counts": {"z-ai/glm-5": 2000},
        "missing_router_metadata_choices": 0,
    }
    metadata.update(overrides)
    return metadata


class RoutingAttributionValidationTest(unittest.TestCase):
    def validate(self, metadata, **kwargs):
        return validate_routing_attribution(
            metadata,
            expected_provider="DeepInfra",
            expected_model="z-ai/glm-5",
            **kwargs,
        )

    def test_complete_attribution_returns_detached_audit_without_warnings(self):
        metadata = routing_metadata()
        original = copy.deepcopy(metadata)

        audit = self.validate(metadata, returned_choice_count=2000)

        self.assertEqual(metadata, original)
        self.assertIsNot(audit["provider_call_counts"], metadata["actual_provider_call_counts"])
        self.assertEqual(audit["call_attribution_fraction"], 1.0)
        self.assertEqual(audit["choice_attribution_fraction"], 1.0)
        self.assertEqual(audit["warnings"], [])

    def test_accepts_bounded_unattributed_calls_and_choices(self):
        metadata = routing_metadata(
            api_calls=3893,
            actual_provider_call_counts={"DeepInfra": 3888},
            actual_response_model_call_counts={"z-ai/glm-5": 3888},
            missing_router_metadata_calls=5,
            actual_provider_counts={"DeepInfra": 3812},
            actual_response_model_counts={"z-ai/glm-5": 3812},
            missing_router_metadata_choices=5,
        )

        audit = self.validate(metadata, returned_choice_count=3817)

        self.assertEqual(audit["unattributed_calls"], 5)
        self.assertAlmostEqual(audit["missing_call_fraction"], 5 / 3893)
        self.assertEqual(audit["unattributed_choices"], 5)
        self.assertAlmostEqual(audit["missing_choice_fraction"], 5 / 3817)
        self.assertEqual(len(audit["warnings"]), 4)

    def test_rejects_unattributed_fraction_above_limit(self):
        metadata = routing_metadata(
            api_calls=4000,
            actual_provider_call_counts={"DeepInfra": 3991},
            actual_response_model_call_counts={"z-ai/glm-5": 3991},
            missing_router_metadata_calls=9,
        )

        with self.assertRaisesRegex(
            RoutingValidationError,
            "Unattributed call fraction",
        ):
            self.validate(metadata)

    def test_rejects_missing_response_model_fraction_above_limit(self):
        with self.subTest(level="calls"):
            metadata = routing_metadata(
                actual_response_model_call_counts={"z-ai/glm-5": 997},
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "Missing response-model call fraction",
            ):
                self.validate(metadata)

        with self.subTest(level="choices"):
            metadata = routing_metadata(
                actual_response_model_counts={"z-ai/glm-5": 1995},
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "Missing response-model choice fraction",
            ):
                self.validate(metadata, returned_choice_count=2000)

    def test_accepts_exact_missing_fraction_boundary(self):
        metadata = routing_metadata(
            api_calls=500,
            actual_provider_call_counts={"DeepInfra": 499},
            actual_response_model_call_counts={"z-ai/glm-5": 499},
            missing_router_metadata_calls=1,
        )

        audit = self.validate(metadata)

        self.assertEqual(audit["missing_call_fraction"], 0.002)

    def test_rejects_unexpected_provider_in_calls_or_choices(self):
        with self.subTest(level="calls"):
            metadata = routing_metadata(
                actual_provider_call_counts={"DeepInfra": 999, "Together": 1}
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "unexpected providers",
            ):
                self.validate(metadata)

        with self.subTest(level="choices"):
            metadata = routing_metadata(
                actual_provider_counts={"DeepInfra": 1999, "Together": 1}
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "unexpected providers",
            ):
                self.validate(metadata, returned_choice_count=2000)

    def test_rejects_requested_or_observed_model_drift(self):
        with self.subTest(level="requested"):
            metadata = routing_metadata(model="z-ai/glm-4")
            with self.assertRaisesRegex(RoutingValidationError, "Requested model"):
                self.validate(metadata)

        with self.subTest(level="calls"):
            metadata = routing_metadata(
                actual_response_model_call_counts={"z-ai/glm-4": 1000}
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "unexpected response models",
            ):
                self.validate(metadata)

        with self.subTest(level="choices"):
            metadata = routing_metadata(
                actual_response_model_counts={"z-ai/glm-4": 2000}
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "unexpected response models",
            ):
                self.validate(metadata, returned_choice_count=2000)

    def test_rejects_fallback_or_unpinned_routing_configuration(self):
        cases = {
            "fallbacks": {"provider_allow_fallbacks": True},
            "order": {"provider_order": ["DeepInfra", "Together"]},
            "metadata": {"router_metadata_requested": False},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(RoutingValidationError):
                    self.validate(routing_metadata(**overrides))

    def test_rejects_call_and_choice_reconciliation_errors(self):
        with self.subTest(level="calls"):
            metadata = routing_metadata(
                actual_provider_call_counts={"DeepInfra": 999}
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "Call attribution does not reconcile",
            ):
                self.validate(metadata)

        with self.subTest(level="choices"):
            metadata = routing_metadata(
                actual_provider_counts={"DeepInfra": 1999}
            )
            with self.assertRaisesRegex(
                RoutingValidationError,
                "Choice attribution does not reconcile",
            ):
                self.validate(metadata, returned_choice_count=2000)

    def test_rejects_second_routing_attempt(self):
        with self.assertRaisesRegex(RoutingValidationError, "expected at most 1"):
            self.validate(routing_metadata(max_routing_attempt=2))

    def test_accepts_existing_inference_field_aliases(self):
        metadata = routing_metadata()
        metadata["requested_model"] = metadata.pop("model")
        metadata["requested_provider_order"] = metadata.pop("provider_order")
        metadata["max_routing_attempt_cumulative"] = metadata.pop(
            "max_routing_attempt"
        )

        audit = self.validate(metadata)

        self.assertEqual(audit["api_calls"], 1000)
        self.assertEqual(audit["max_routing_attempt"], 1)

    def test_rejects_conflicting_aliases_and_invalid_counts(self):
        with self.subTest(case="aliases"):
            metadata = routing_metadata(requested_model="z-ai/glm-4")
            with self.assertRaisesRegex(RoutingValidationError, "Conflicting"):
                self.validate(metadata)

        with self.subTest(case="negative"):
            metadata = routing_metadata(missing_router_metadata_calls=-1)
            with self.assertRaisesRegex(RoutingValidationError, "non-negative"):
                self.validate(metadata)

        with self.subTest(case="boolean"):
            metadata = routing_metadata(api_calls=True)
            with self.assertRaisesRegex(RoutingValidationError, "integer"):
                self.validate(metadata)

    def test_validates_configurable_missing_fraction(self):
        metadata = routing_metadata(
            api_calls=100,
            actual_provider_call_counts={"DeepInfra": 99},
            actual_response_model_call_counts={"z-ai/glm-5": 99},
            missing_router_metadata_calls=1,
        )

        audit = self.validate(metadata, max_missing_fraction=0.01)
        self.assertEqual(audit["max_missing_fraction"], 0.01)

        with self.assertRaisesRegex(RoutingValidationError, "finite number"):
            self.validate(metadata, max_missing_fraction=float("nan"))

    def test_requires_disabled_response_cache_and_reconciled_non_hit_statuses(self):
        audit = self.validate(
            routing_metadata(),
            require_response_cache_disabled=True,
        )
        self.assertTrue(audit["response_cache_disabled"])
        self.assertEqual(audit["response_cache_hits"], 0)

        with self.subTest(case="not-disabled"):
            metadata = routing_metadata(disable_openrouter_response_cache=False)
            with self.assertRaisesRegex(
                RoutingValidationError,
                "must be explicitly disabled",
            ):
                self.validate(
                    metadata,
                    require_response_cache_disabled=True,
                )

        with self.subTest(case="cache-hit"):
            metadata = routing_metadata(
                response_cache_status_call_counts={"MISS": 999, "HIT": 1}
            )
            with self.assertRaisesRegex(RoutingValidationError, "cache HIT"):
                self.validate(
                    metadata,
                    require_response_cache_disabled=True,
                )

        with self.subTest(case="count-mismatch"):
            metadata = routing_metadata(
                response_cache_status_call_counts={"<missing>": 999}
            )
            with self.assertRaisesRegex(RoutingValidationError, "do not reconcile"):
                self.validate(
                    metadata,
                    require_response_cache_disabled=True,
                )

    def test_accepts_inference_alias_for_response_cache_disable(self):
        metadata = routing_metadata()
        metadata["openrouter_response_cache_disabled"] = metadata.pop(
            "disable_openrouter_response_cache"
        )
        audit = self.validate(
            metadata,
            require_response_cache_disabled=True,
        )
        self.assertTrue(audit["response_cache_disabled"])


if __name__ == "__main__":
    unittest.main()
