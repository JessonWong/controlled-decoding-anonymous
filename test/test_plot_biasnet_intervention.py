import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest
import torch

from scripts.plot_biasnet_intervention import (
    display_token,
    distribution_metrics,
    gate_scales,
    plot_figure,
    select_files,
    validate_cache_contract,
)


def test_display_token_makes_whitespace_and_unicode_portable():
    assert display_token(" a\n个") == "·a↵U+4E2A"


def test_distribution_metrics_uses_explicit_kl_direction_and_set_difference():
    base = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64))
    biased = torch.log(torch.tensor([[0.1, 0.3, 0.6]], dtype=torch.float64))

    result = distribution_metrics(base, biased, top_k=2)

    expected_biased_to_base = sum(
        q * math.log(q / p) for p, q in zip((0.6, 0.3, 0.1), (0.1, 0.3, 0.6))
    )
    expected_base_to_biased = sum(
        p * math.log(p / q) for p, q in zip((0.6, 0.3, 0.1), (0.1, 0.3, 0.6))
    )
    assert result["kl_biased_to_base"].item() == pytest.approx(
        expected_biased_to_base
    )
    assert result["kl_base_to_biased"].item() == pytest.approx(
        expected_base_to_biased
    )
    assert result["topk_replacements"].item() == 1
    assert result["topk_symmetric_difference"].item() == 2


def test_distribution_metrics_is_zero_for_no_intervention():
    scores = torch.tensor([[1.0, -0.2, 0.3, 2.1]])

    result = distribution_metrics(scores, scores.clone(), top_k=3)

    assert result["kl_biased_to_base"].item() == pytest.approx(0.0, abs=1e-7)
    assert result["kl_base_to_biased"].item() == pytest.approx(0.0, abs=1e-7)
    assert result["topk_replacements"].item() == 0


def test_distribution_metrics_flags_an_underdetermined_topk_boundary():
    scores = torch.tensor([[3.0, 2.0, 1.0, 1.0]])

    result = distribution_metrics(scores, scores.clone(), top_k=3)

    assert result["base_topk_boundary_ties"].item() == 2
    assert result["base_topk_ambiguous"].item() is True


def test_gate_scales_replays_hard_gate_with_forced_warmup():
    payload = {"risk_gate_scores": torch.tensor([[0.9, 0.9, 0.1, 0.8]])}
    config = {
        "risk_gate_training": "hard",
        "risk_gate_threshold": 0.5,
        "risk_gate_warmup_tokens": 2,
    }

    scales = gate_scales(payload, config, length=4, enabled=True)

    assert scales.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_gate_scales_replays_soft_gate_and_minimum_cutoff():
    payload = {"risk_gate_scores": torch.tensor([[0.9, 0.5, 0.1]])}
    config = {
        "risk_gate_training": "runtime_soft",
        "risk_gate_threshold": 0.5,
        "risk_gate_soft_temperature": 0.1,
        "risk_gate_min_scale": 0.02,
        "risk_gate_warmup_tokens": 1,
    }

    scales = gate_scales(payload, config, length=3, enabled=True)

    assert scales[0].item() == 1.0
    assert scales[1].item() == pytest.approx(0.5)
    assert scales[2].item() > 0.98


def test_select_files_uses_checkpoint_held_out_names():
    files = [Path("a.pt"), Path("b.pt"), Path("c.pt")]
    config = {"held_out_files": ["b.pt"]}

    assert select_files(files, config, "checkpoint_held_out", 0) == [Path("b.pt")]
    assert select_files(files, config, "checkpoint_train", 0) == [
        Path("a.pt"),
        Path("c.pt"),
    ]


def test_cache_contract_rejects_wrong_static_prior():
    payload = {
        "log_probs": torch.zeros(1, 2, 4),
        "labels": torch.zeros(1, 2, dtype=torch.long),
        "metadata": {"mc_static_prior_mode": "uniform_dirichlet_v1"},
    }
    config = {
        "vocab_size": 4,
        "mc_input_representation": "floor_logprob",
        "mc_base_score_representation": "floor_logprob",
        "mc_static_prior_mode": "global_unigram_dirichlet_v1",
    }

    with pytest.raises(ValueError, match="static prior mismatch"):
        validate_cache_contract(payload, config, Path("record.pt"))


def test_bc_layout_has_requested_axes_and_only_two_panels(tmp_path, monkeypatch):
    rows = [
        {
            "label_token_text": token,
            "residual_scale": 1.0,
            "kl_biased_to_base": value,
            "kl_base_to_biased": value / 2,
        }
        for token, value in (("one", 0.2), ("two", 0.7))
    ]
    args = SimpleNamespace(
        kl_direction="biased_to_base",
        example_max_tokens=2,
        panels="bc",
        histogram_max_quantile=1.0,
        histogram_bins=5,
        top_k=10,
        title=None,
        dpi=50,
    )
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    plot_figure(
        tmp_path / "figure.png",
        tmp_path / "figure.pdf",
        rows,
        {"record_file": "example.pt", "positions": rows},
        tokenizer=None,
        args=args,
        experiment_label="MC50 + global-uniform, κ=2",
    )

    figure = plt.gcf()
    assert len(figure.axes) == 2
    histogram_axis, token_axis = figure.axes
    assert histogram_axis.get_xlabel() == "KL Divergence"
    assert histogram_axis.get_ylabel() == "Frequency"
    assert token_axis.get_xlabel() == ""
    assert token_axis.get_ylabel() == "KL Divergence"
    assert (tmp_path / "figure.png").is_file()
    assert (tmp_path / "figure.pdf").is_file()
