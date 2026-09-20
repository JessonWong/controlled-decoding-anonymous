import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from context_encoder import (
    CONTEXT_MANIFEST, CONTEXT_PROTOCOL, FrozenContextEncoder, cache_states,
    file_sha256, json_sha256, load_context_manifest, load_context_sidecar, render_context,
)
from modeling_biasnet import BiasConfig, BiasNet
from training.cache_context_features import atomic_json, materialize_record, main as cache_main
from training.recover_context_states import recover_record
from training.train_biasnet import CachedLogitsDataset
from inference_openrouter import BiasMCInputs, apply_bias_model, generate_one
from test.test_inference_openrouter_gate import (
    CharacterTokenizer, FakeRiskGate, BatchedSequenceRiskGate, generation_args,
)


def contract(dim=8):
    return {"schema_version": 1, "protocol": CONTEXT_PROTOCOL,
            "model_name_or_path": "toy", "tokenizer_name_or_path": "toy",
            "model_identity": {"revision": "toy"}, "tokenizer_identity": {"sha256": "toy"},
            "hidden_size": dim, "dtype": "float32", "quantization": "none",
            "pooling": "last_non_padding_final_layer", "overflow": "error", "max_length": 128,
            "position_ids": "non_padding_cumsum_v1"}


def make_model(context=False):
    model = BiasNet(BiasConfig(hidden_size=8, vocab_size=8, input_projection_mode="count_sketch",
                               context_conditioning=context, context_dim=8 if context else None,
                               context_bottleneck=3, context_encoder_contract=contract() if context else None))
    model.set_up_proj()
    return model.eval()


class RecordingEncoder:
    contract = contract()

    def __init__(self):
        self.calls = self.input_tokens = 0
        self.latency_seconds = 0.0
        self.seen = []

    def encode(self, prompts, prefixes):
        self.calls += 1
        self.input_tokens += len(prompts)
        self.latency_seconds += 0.1
        self.seen.extend(zip(prompts, prefixes))
        return torch.tensor([[len(p)] * 8 for p in prefixes], dtype=torch.float16)


class ContextModelTest(unittest.TestCase):
    def test_zero_initialization_preserves_output_and_rng(self):
        torch.manual_seed(42)
        model = make_model()
        x = torch.randn(3, 8)
        baseline = model(x)
        state = torch.get_rng_state().clone()
        model.enable_context_conditioning(8, 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(model(x, context_features=torch.randn(3, 8)), baseline, rtol=0, atol=0)

    def test_gradient_flow_and_frozen_features(self):
        torch.manual_seed(1)
        model = make_model(True)
        features = torch.randn(4, 8, requires_grad=True)
        x = torch.randn(4, 8)
        optimizer = torch.optim.SGD([model.context_up.weight, model.context_down.weight], lr=0.1)
        loss = model(x, context_features=features).square().mean()
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertGreater(model.context_up.weight.grad.abs().sum().item(), 0)
        self.assertEqual(model.context_down.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        model.zero_grad()
        model(x, context_features=features).square().mean().backward()
        self.assertGreater(model.context_down.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_roundtrip_and_legacy_loading(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                model = make_model(enabled)
                if enabled:
                    torch.nn.init.normal_(model.context_up.weight)
                x = torch.randn(2, 8)
                kwargs = {"context_features": torch.randn(2, 8)} if enabled else {}
                expected = model(x, **kwargs)
                model.save_pretrained(tmp)
                if not enabled:
                    path = Path(tmp) / "config.json"
                    config = json.loads(path.read_text())
                    path.write_text(json.dumps({k: v for k, v in config.items() if not k.startswith("context_")}))
                restored = BiasNet.from_pretrained(tmp)
                restored.set_up_proj()
                restored.eval()
                self.assertEqual(restored.context_conditioning, enabled)
                torch.testing.assert_close(restored(x, **kwargs), expected, rtol=0, atol=0)

    def test_missing_invalid_and_disabled_context_fail(self):
        x = torch.randn(2, 8)
        model = make_model(True)
        for features in (None, torch.randn(1, 8), torch.ones(2, 8, dtype=torch.long), torch.full((2, 8), float("nan"))):
            with self.subTest(features=features), self.assertRaises(ValueError):
                model(x, context_features=features)
        with self.assertRaises(ValueError):
            make_model()(x, context_features=torch.randn(2, 8))

    def test_mc_views_share_context_and_average_probabilities(self):
        model = make_model(True)
        torch.nn.init.normal_(model.context_up.weight, std=0.1)
        x, base, ctx = torch.randn(3, 8), torch.randn(3, 8), torch.randn(1, 8)
        expected = torch.softmax(base + model(x, context_features=ctx.expand(3, -1)), -1).mean(0)
        actual = apply_bias_model(model, BiasMCInputs(x, base, 3), context_features=ctx)
        torch.testing.assert_close(actual.exp()[0], expected)


class ContextCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.cache = self.root / "mc", self.root / "context"
        self.data.mkdir()
        self.cache.mkdir()
        self.source = self.data / "toy.pt"
        self.payload = {"log_probs": torch.randn(1, 4, 8), "labels": torch.tensor([[1, 2, 3, 4]]),
                        "prompt_text": "hello", "sampled_prefix_texts": ["", "A", "AB", "ABC"],
                        "risk_gate_mask": torch.tensor([[True, False, True, False]])}
        torch.save(self.payload, self.source)
        self.encoder = RecordingEncoder()
        self.entry = materialize_record(self.source, self.cache / self.source.name, self.encoder, 2)
        self.manifest = {"schema_version": 1, "contract": self.encoder.contract,
                         "contract_sha256": json_sha256(self.encoder.contract),
                         "files": {self.source.name: self.entry}}
        atomic_json(self.cache / CONTEXT_MANIFEST, self.manifest)

    def test_pre_action_text_and_dataset_masks_remain_aligned(self):
        self.assertEqual(self.encoder.seen, [("hello", p) for p in ["", "A", "AB", "ABC"]])
        dataset = CachedLogitsDataset(str(self.data), dtype=torch.float32, max_tokens_per_sample=3,
                                      risk_gate_training="hard", drop_zero_weight_tokens=True,
                                      context_cache_dir=str(self.cache))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset[0]), 10)
        self.assertEqual(dataset.labels.tolist(), [1, 3])
        self.assertEqual(dataset.context_features[:, 0].tolist(), [0, 2])
        legacy = CachedLogitsDataset(str(self.data), dtype=torch.float32)
        self.assertEqual(len(legacy[0]), 9)

    def test_changed_source_or_reordered_prefixes_fail(self):
        wrong = copy.deepcopy(self.payload)
        wrong["sampled_prefix_texts"][1:3] = list(reversed(wrong["sampled_prefix_texts"][1:3]))
        with self.assertRaisesRegex(ValueError, "alignment"):
            load_context_sidecar(self.source, wrong, self.cache, self.manifest)
        wrong = copy.deepcopy(self.payload)
        wrong["log_probs"][0, 0, 0] += 1
        torch.save(wrong, self.source)
        with self.assertRaisesRegex(ValueError, "changed"):
            load_context_sidecar(self.source, wrong, self.cache, self.manifest)

    def test_sidecar_corruption_or_wrong_contract_fail(self):
        path = self.cache / self.source.name
        sidecar = torch.load(path, weights_only=True)
        sidecar["context_features"][0, 0] += 1
        torch.save(sidecar, path)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_context_sidecar(self.source, self.payload, self.cache, self.manifest)
        self.manifest["files"][path.name]["sidecar_sha256"] = file_sha256(path)
        self.manifest["contract_sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "contract"):
            load_context_sidecar(self.source, self.payload, self.cache, self.manifest)

    def test_legacy_recovery_requires_source_binding(self):
        legacy = {k: v for k, v in self.payload.items() if k not in ("prompt_text", "sampled_prefix_texts")}
        torch.save(legacy, self.source)
        with self.assertRaises(ValueError):
            cache_states(legacy)
        states = {"schema_version": 1, "protocol": CONTEXT_PROTOCOL, "files": {
            self.source.name: {"source_cache_sha256": file_sha256(self.source),
                               "prompt_text": "hello", "sampled_prefix_texts": ["", "A", "AB", "ABC"]}}}
        self.manifest["state_manifest_sha256"] = json_sha256(states)
        self.manifest["files"][self.source.name] = materialize_record(
            self.source, self.cache / self.source.name, self.encoder, 2, states, json_sha256(states),
        )
        features = load_context_sidecar(self.source, legacy, self.cache, self.manifest)
        self.assertEqual(features[:, 0].tolist(), [0, 1, 2, 3])
        states["files"][self.source.name]["source_cache_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checksum"):
            materialize_record(self.source, self.cache / "bad.pt", self.encoder, 2, states, json_sha256(states))

    def test_sparse_legacy_recovery_excludes_current_label(self):
        question, answer = "toy question", "ABC"
        name = hashlib.md5((question + answer).encode()).hexdigest() + ".pt"
        raw_path, fused_path = self.root / name, self.data / name
        raw = {"log_probs": torch.zeros(1, 2, 8), "labels": torch.tensor([[1, 3]]),
               "valid_sample_counts": torch.tensor([[50, 50]]),
               "risk_gate_active_positions": torch.tensor([[0, 2]]),
               "metadata": {"dataset_idx": 0, "samples_per_token": 50,
                            "data_sha256": hashlib.sha256((question + answer).encode()).hexdigest()}}
        torch.save(raw, raw_path)
        torch.save(raw, fused_path)
        def decode(ids, **kwargs):
            self.assertFalse(kwargs["skip_special_tokens"])
            return "".join("ABC"[i - 1] for i in ids)
        tokenizer = SimpleNamespace(encode=lambda text, **kw: [1, 2, 3], decode=decode)
        ledger = {"output_sha256": file_sha256(fused_path), "source_sha256": file_sha256(raw_path),
                  "dataset_idx": 0, "data_sha256": raw["metadata"]["data_sha256"]}
        states = recover_record(fused_path, raw, raw_path, raw, [{"prompt": question, "rejected": answer}],
                                tokenizer, {"samples_per_token": 50, "append_no_think": True}, ledger)
        self.assertEqual(states["sampled_prefix_texts"], ["", "AB"])
        self.assertEqual(states["prompt_text"], question + "\n/no_think")


class ContextGenerationTest(unittest.TestCase):
    def test_gated_steps_encode_pre_action_only_when_enabled(self):
        for score, expected_calls in ((0.0, 2), (0.9, 0)):
            encoder = RecordingEncoder()
            scores = torch.full((1, 8), -20.0)
            scores[0, 1] = 20
            with self.subTest(score=score), \
                    mock.patch("inference_openrouter.query_next_token_id", return_value=(2, {})), \
                    mock.patch("inference_openrouter.estimate_log_probs", return_value=scores):
                generate_one(object(), CharacterTokenizer(), "hello",
                             generation_args(max_new_tokens=2, risk_gate_min_scale=0.5),
                             torch.device("cpu"), make_model(True), FakeRiskGate(score), context_encoder=encoder)
            self.assertEqual(encoder.calls, expected_calls)
            if expected_calls:
                self.assertEqual(encoder.seen, [("hello", ""), ("hello", "A")])

    def test_speculative_rollback_encodes_committed_prefix(self):
        encoder = RecordingEncoder()
        gate = BatchedSequenceRiskGate([[0.9], [0.9], [0.9, 0.9, 0.0, 0.9], [0.9]])
        args = generation_args(max_new_tokens=6, risk_gate_min_scale=0.5,
                               risk_gate_speculative_draft=True, risk_gate_speculative_min_base_streak=2,
                               risk_gate_speculative_draft_tokens=80)
        scores = torch.full((1, 8), -20.0)
        scores[0, 3] = 20
        with mock.patch("inference_openrouter.query_next_token_id", return_value=(1, {})), \
                mock.patch("inference_openrouter.generate_speculative_draft", return_value=(
                    [2, 2, 2, 2], {"requested_tokens": 4, "finish_reason": "length", "native_finish_reason": "length"})), \
                mock.patch("inference_openrouter.estimate_log_probs", return_value=scores):
            result = generate_one(object(), CharacterTokenizer(), "hello", args, torch.device("cpu"),
                                  make_model(True), gate, context_encoder=encoder)
        self.assertEqual(result, "AABBCA")
        self.assertEqual(encoder.seen, [("hello", "AABB")])

    def test_context_once_per_step_before_action(self):
        encoder = RecordingEncoder()
        audit = {}
        scores = torch.full((1, 8), -20.0)
        scores[0, 1] = 20
        with mock.patch("inference_openrouter.estimate_log_probs", return_value=scores) as estimate:
            output = generate_one(object(), CharacterTokenizer(), "hello", generation_args(max_new_tokens=2),
                                  torch.device("cpu"), make_model(True), None,
                                  context_encoder=encoder, generation_audit=audit)
        self.assertEqual(output, "AA")
        self.assertEqual(encoder.seen, [("hello", ""), ("hello", "A")])
        self.assertEqual(estimate.call_count, 2)
        self.assertEqual(audit["summary"]["context_encoder_calls"], 2)

    def test_missing_context_and_anytime_fail_before_queries(self):
        model = make_model(True)
        for encoder, anytime in ((None, None), (RecordingEncoder(), object())):
            with self.subTest(anytime=anytime), mock.patch("inference_openrouter.estimate_log_probs") as estimate:
                with self.assertRaises(ValueError):
                    generate_one(object(), CharacterTokenizer(), "hello", generation_args(),
                                 torch.device("cpu"), model, None, context_encoder=encoder, anytime_policy=anytime)
                estimate.assert_not_called()


class LocalEncoderIntegrationTest(unittest.TestCase):
    def test_encoder_cache_resume_training_and_reload_on_cpu(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast
        from training.train_biasnet import main as train_main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder_path, data_path, context_path, output_path = [root / p for p in ("encoder", "data", "context", "out")]
            data_path.mkdir()
            backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "user": 3,
                                          "assistant": 4, "hello": 5, "one": 6, "two": 7}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
            tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            tokenizer.save_pretrained(encoder_path)
            GPT2Model(GPT2Config(vocab_size=8, n_embd=8, n_layer=1, n_head=2, n_positions=128)).save_pretrained(encoder_path)
            encoder = FrozenContextEncoder.load(str(encoder_path), dtype="float32", max_length=128,
                                                device="cpu", local_files_only=True)
            batched = encoder.encode(["hello", "hello"], ["", "one two"])
            single = torch.cat([encoder.encode(["hello"], [p]) for p in ("", "one two")])
            torch.testing.assert_close(batched, single, atol=1e-3, rtol=1e-3)
            self.assertFalse(batched.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in encoder.backbone.parameters()))
            self.assertEqual(render_context(encoder.tokenizer, "hello", "one"), "user hello assistant one")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                encoder.encode(["hello"], ["one " * 130])
            for i in range(2):
                torch.save({"log_probs": torch.randn(1, 3, 8), "labels": torch.tensor([[1, 2, 3]]),
                            "prompt_text": "hello", "sampled_prefix_texts": ["", "one", "one two"]}, data_path / f"{i}.pt")
            cache_args = ["--input_dir", str(data_path), "--output_dir", str(context_path),
                          "--encoder_model", str(encoder_path), "--encoder_dtype", "float32",
                          "--device", "cpu", "--local_files_only", "--batch_size", "2", "--max_length", "128"]
            cache_main(cache_args)
            cache_main([*cache_args, "--resume"])
            manifest = load_context_manifest(context_path)
            self.assertEqual(manifest["status"], "complete")
            restored_encoder = FrozenContextEncoder.from_contract(manifest["contract"], device="cpu", local_files_only=True)
            online = restored_encoder.encode(["hello"] * 3, ["", "one", "one two"])
            source = data_path / "0.pt"
            cached = load_context_sidecar(source, torch.load(source, weights_only=True), context_path, manifest)
            torch.testing.assert_close(online, cached, atol=1e-3, rtol=1e-3)
            initial = make_model()
            initial_path = root / "initial"
            initial.save_pretrained(initial_path)
            with mock.patch("sys.argv", ["train_biasnet", "--data_dir", str(data_path),
                             "--context_cache_dir", str(context_path), "--output_dir", str(output_path),
                             "--init_checkpoint", str(initial_path),
                             "--hidden_size", "8", "--vocab_size", "8", "--input_projection_mode", "count_sketch",
                             "--context_bottleneck", "3", "--epochs", "1", "--batch_size", "2",
                             "--learning_rate", "1e-5", "--context_learning_rate", "1e-4",
                             "--held_out_file_count", "1", "--eval_before_training"]), \
                    mock.patch("torch.cuda.is_available", return_value=False), \
                    mock.patch("torch.optim.AdamW", wraps=torch.optim.AdamW) as optimizer:
                train_main()
            groups = optimizer.call_args.args[0]
            self.assertEqual([g["lr"] for g in groups], [1e-5, 1e-4])
            self.assertEqual(len(groups[1]["params"]), 2)
            trained = BiasNet.from_pretrained(output_path)
            trained.set_up_proj()
            trained.eval()
            self.assertEqual(trained.config.context_encoder_contract, manifest["contract"])
            self.assertGreater(trained.context_up.weight.abs().sum().item(), 0)
            torch.testing.assert_close(trained.lm_head.weight, initial.lm_head.weight, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(trained(torch.randn(3, 8), context_features=online)).all())


if __name__ == "__main__":
    unittest.main()
