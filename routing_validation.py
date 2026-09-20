"""Pure validation helpers for OpenRouter routing-attribution summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


class RoutingValidationError(ValueError):
    """Raised when routing evidence violates the requested routing policy."""


_MISSING = object()


def _aliased_value(
    metadata: Mapping[str, Any],
    canonical_name: str,
    *aliases: str,
) -> Any:
    names = (canonical_name, *aliases)
    present = [(name, metadata[name]) for name in names if name in metadata]
    if not present:
        joined = " or ".join(repr(name) for name in names)
        raise RoutingValidationError(f"Missing required routing field {joined}.")

    first_name, first_value = present[0]
    for name, value in present[1:]:
        if value != first_value:
            raise RoutingValidationError(
                f"Conflicting routing fields {first_name!r}={first_value!r} "
                f"and {name!r}={value!r}."
            )
    return first_value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoutingValidationError(
            f"{field_name} must be a non-negative integer, got {value!r}."
        )
    return value


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _nonnegative_int(value, field_name)
    if parsed == 0:
        raise RoutingValidationError(f"{field_name} must be positive.")
    return parsed


def _count_mapping(
    value: Any,
    field_name: str,
    *,
    require_positive_total: bool = True,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise RoutingValidationError(f"{field_name} must be a mapping.")

    counts: dict[str, int] = {}
    for name, count in value.items():
        if not isinstance(name, str) or not name:
            raise RoutingValidationError(
                f"{field_name} contains an invalid name {name!r}."
            )
        counts[name] = _nonnegative_int(count, f"{field_name}[{name!r}]")

    if require_positive_total and sum(counts.values()) == 0:
        raise RoutingValidationError(f"{field_name} must record at least one item.")
    return counts


def _unexpected_names(
    counts: Mapping[str, int],
    expected_name: str,
    *,
    casefold: bool,
) -> list[str]:
    expected = expected_name.casefold() if casefold else expected_name
    return sorted(
        name
        for name in counts
        if (name.casefold() if casefold else name) != expected
    )


def validate_routing_attribution(
    metadata: Mapping[str, Any],
    *,
    expected_provider: str,
    expected_model: str,
    max_missing_fraction: float = 0.002,
    returned_choice_count: int | None = None,
    require_response_cache_disabled: bool = False,
) -> dict[str, Any]:
    """Validate pinned-provider routing evidence and return a detached audit.

    Missing provider attribution is telemetry loss, not direct evidence of
    provider drift. It is accepted only up to ``max_missing_fraction``. Any
    positively attributed provider or response model must match the expected
    values.

    ``metadata`` is never mutated. Existing cache and inference aliases are
    accepted for the requested model, provider order, and routing-attempt
    fields. Callers with split summaries should assemble a copied canonical
    mapping before validation.
    """

    if not isinstance(metadata, Mapping):
        raise RoutingValidationError("metadata must be a mapping.")
    if not isinstance(expected_provider, str) or not expected_provider.strip():
        raise RoutingValidationError("expected_provider must be a non-empty string.")
    if not isinstance(expected_model, str) or not expected_model.strip():
        raise RoutingValidationError("expected_model must be a non-empty string.")
    if (
        isinstance(max_missing_fraction, bool)
        or not isinstance(max_missing_fraction, Real)
        or not math.isfinite(float(max_missing_fraction))
        or not 0.0 <= float(max_missing_fraction) <= 1.0
    ):
        raise RoutingValidationError(
            "max_missing_fraction must be a finite number between 0 and 1."
        )
    max_missing_fraction = float(max_missing_fraction)

    requested_model = _aliased_value(metadata, "model", "requested_model")
    if requested_model != expected_model:
        raise RoutingValidationError(
            f"Requested model {requested_model!r} does not match "
            f"{expected_model!r}."
        )

    provider_order = _aliased_value(
        metadata,
        "provider_order",
        "requested_provider_order",
    )
    if (
        not isinstance(provider_order, (list, tuple))
        or len(provider_order) != 1
        or not isinstance(provider_order[0], str)
        or provider_order[0].casefold() != expected_provider.casefold()
    ):
        raise RoutingValidationError(
            "Provider routing must be pinned to exactly "
            f"[{expected_provider!r}], got {provider_order!r}."
        )
    if metadata.get("provider_allow_fallbacks", _MISSING) is not False:
        raise RoutingValidationError("Provider fallbacks must be explicitly disabled.")
    if metadata.get("router_metadata_requested", _MISSING) is not True:
        raise RoutingValidationError(
            "OpenRouter routing metadata must be explicitly requested."
        )

    api_calls = _positive_int(metadata.get("api_calls"), "api_calls")
    provider_call_counts = _count_mapping(
        metadata.get("actual_provider_call_counts"),
        "actual_provider_call_counts",
    )
    unexpected_providers = _unexpected_names(
        provider_call_counts,
        expected_provider,
        casefold=True,
    )
    if unexpected_providers:
        raise RoutingValidationError(
            f"Calls were served by unexpected providers {unexpected_providers!r}."
        )

    unattributed_calls = _nonnegative_int(
        metadata.get("missing_router_metadata_calls"),
        "missing_router_metadata_calls",
    )
    attributed_calls = sum(provider_call_counts.values())
    if attributed_calls + unattributed_calls != api_calls:
        raise RoutingValidationError(
            "Call attribution does not reconcile: "
            f"{attributed_calls} attributed + {unattributed_calls} unattributed "
            f"!= {api_calls} API calls."
        )
    missing_call_fraction = unattributed_calls / api_calls
    if missing_call_fraction > max_missing_fraction:
        raise RoutingValidationError(
            f"Unattributed call fraction {missing_call_fraction:.6%} exceeds "
            f"the allowed {max_missing_fraction:.6%}."
        )

    response_model_call_counts = _count_mapping(
        metadata.get("actual_response_model_call_counts"),
        "actual_response_model_call_counts",
    )
    unexpected_models = _unexpected_names(
        response_model_call_counts,
        expected_model,
        casefold=False,
    )
    if unexpected_models:
        raise RoutingValidationError(
            f"Calls returned unexpected response models {unexpected_models!r}."
        )
    response_model_calls = sum(response_model_call_counts.values())
    if response_model_calls > api_calls:
        raise RoutingValidationError(
            "Response-model call counts exceed the number of API calls."
        )
    missing_response_model_calls = api_calls - response_model_calls
    missing_response_model_call_fraction = missing_response_model_calls / api_calls
    if missing_response_model_call_fraction > max_missing_fraction:
        raise RoutingValidationError(
            "Missing response-model call fraction "
            f"{missing_response_model_call_fraction:.6%} exceeds the allowed "
            f"{max_missing_fraction:.6%}."
        )

    max_routing_attempt = _nonnegative_int(
        _aliased_value(
            metadata,
            "max_routing_attempt",
            "max_routing_attempt_cumulative",
        ),
        "max_routing_attempt",
    )
    if max_routing_attempt > 1:
        raise RoutingValidationError(
            f"Observed routing attempt {max_routing_attempt}; expected at most 1."
        )

    warnings: list[str] = []
    if unattributed_calls:
        warnings.append(
            f"{unattributed_calls}/{api_calls} successful API calls "
            "could not be attributed from returned routing telemetry."
        )
    if missing_response_model_calls:
        warnings.append(
            f"{missing_response_model_calls}/{api_calls} successful API calls "
            "did not record a response model."
        )

    audit: dict[str, Any] = {
        "expected_provider": expected_provider,
        "expected_model": expected_model,
        "max_missing_fraction": max_missing_fraction,
        "api_calls": api_calls,
        "provider_call_counts": dict(provider_call_counts),
        "attributed_calls": attributed_calls,
        "unattributed_calls": unattributed_calls,
        "call_attribution_fraction": attributed_calls / api_calls,
        "missing_call_fraction": missing_call_fraction,
        "response_model_call_counts": dict(response_model_call_counts),
        "response_model_calls": response_model_calls,
        "missing_response_model_calls": missing_response_model_calls,
        "missing_response_model_call_fraction": (
            missing_response_model_call_fraction
        ),
        "max_routing_attempt": max_routing_attempt,
        "warnings": warnings,
    }

    if require_response_cache_disabled:
        cache_disabled = _aliased_value(
            metadata,
            "disable_openrouter_response_cache",
            "openrouter_response_cache_disabled",
        )
        if cache_disabled is not True:
            raise RoutingValidationError(
                "OpenRouter response caching must be explicitly disabled."
            )
        cache_status_counts = _count_mapping(
            metadata.get("response_cache_status_call_counts"),
            "response_cache_status_call_counts",
        )
        if sum(cache_status_counts.values()) != api_calls:
            raise RoutingValidationError(
                "Response-cache status counts do not reconcile with API calls."
            )
        cache_hits = sum(
            count
            for status, count in cache_status_counts.items()
            if status.casefold() == "hit"
        )
        if cache_hits:
            raise RoutingValidationError(
                f"Observed {cache_hits} OpenRouter response-cache HIT calls."
            )
        audit.update(
            {
                "response_cache_disabled": True,
                "response_cache_status_call_counts": dict(cache_status_counts),
                "response_cache_hits": 0,
            }
        )

    if returned_choice_count is not None:
        returned_choice_count = _positive_int(
            returned_choice_count,
            "returned_choice_count",
        )
        provider_choice_counts = _count_mapping(
            metadata.get("actual_provider_counts"),
            "actual_provider_counts",
        )
        unexpected_choice_providers = _unexpected_names(
            provider_choice_counts,
            expected_provider,
            casefold=True,
        )
        if unexpected_choice_providers:
            raise RoutingValidationError(
                "Choices were served by unexpected providers "
                f"{unexpected_choice_providers!r}."
            )

        unattributed_choices = _nonnegative_int(
            metadata.get("missing_router_metadata_choices"),
            "missing_router_metadata_choices",
        )
        attributed_choices = sum(provider_choice_counts.values())
        if attributed_choices + unattributed_choices != returned_choice_count:
            raise RoutingValidationError(
                "Choice attribution does not reconcile: "
                f"{attributed_choices} attributed + "
                f"{unattributed_choices} unattributed != "
                f"{returned_choice_count} returned choices."
            )
        missing_choice_fraction = unattributed_choices / returned_choice_count
        if missing_choice_fraction > max_missing_fraction:
            raise RoutingValidationError(
                f"Unattributed choice fraction {missing_choice_fraction:.6%} "
                f"exceeds the allowed {max_missing_fraction:.6%}."
            )

        response_model_choice_counts = _count_mapping(
            metadata.get("actual_response_model_counts"),
            "actual_response_model_counts",
        )
        unexpected_choice_models = _unexpected_names(
            response_model_choice_counts,
            expected_model,
            casefold=False,
        )
        if unexpected_choice_models:
            raise RoutingValidationError(
                "Choices returned unexpected response models "
                f"{unexpected_choice_models!r}."
            )
        response_model_choices = sum(response_model_choice_counts.values())
        if response_model_choices > returned_choice_count:
            raise RoutingValidationError(
                "Response-model choice counts exceed returned choices."
            )
        missing_response_model_choices = (
            returned_choice_count - response_model_choices
        )
        missing_response_model_choice_fraction = (
            missing_response_model_choices / returned_choice_count
        )
        if missing_response_model_choice_fraction > max_missing_fraction:
            raise RoutingValidationError(
                "Missing response-model choice fraction "
                f"{missing_response_model_choice_fraction:.6%} exceeds the "
                f"allowed {max_missing_fraction:.6%}."
            )

        if unattributed_choices:
            warnings.append(
                f"{unattributed_choices}/{returned_choice_count} returned choices "
                "could not be attributed from returned routing telemetry."
            )
        if missing_response_model_choices:
            warnings.append(
                f"{missing_response_model_choices}/"
                f"{returned_choice_count} returned choices did not record a "
                "response model."
            )
        audit.update(
            {
                "returned_choice_count": returned_choice_count,
                "provider_choice_counts": dict(provider_choice_counts),
                "attributed_choices": attributed_choices,
                "unattributed_choices": unattributed_choices,
                "choice_attribution_fraction": (
                    attributed_choices / returned_choice_count
                ),
                "missing_choice_fraction": missing_choice_fraction,
                "response_model_choice_counts": dict(
                    response_model_choice_counts
                ),
                "response_model_choices": response_model_choices,
                "missing_response_model_choices": (
                    missing_response_model_choices
                ),
                "missing_response_model_choice_fraction": (
                    missing_response_model_choice_fraction
                ),
            }
        )

    return audit
