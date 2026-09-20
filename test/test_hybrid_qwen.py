from collections import Counter

from empirical_vocab.hybrid_qwen import (
    build_hybrid_qwen_vocab,
    counter_to_hybrid_counts,
)


class FakeTokenizer:
    eos_token_id = 9
    all_special_ids = [0, 9]

    def __len__(self):
        return 10

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {
            "": [],
            "a": [1],
            "b": [2],
            "ab": [1, 2],
            "ba": [2, 1],
        }[text]

    def decode(self, ids, skip_special_tokens=False):
        assert not skip_special_tokens
        return {1: "a", 2: "b", 9: ""}[ids[0]]


def test_hybrid_vocab_preserves_qwen_ids_and_caps_extensions():
    tokenizer = FakeTokenizer()
    counters = [{"a": 2, "ab": 4, "ba": 1, "": 3}]
    vocab, frequencies = build_hybrid_qwen_vocab(
        counters, tokenizer, extension_capacity=1
    )

    assert frequencies == Counter({"ab": 4, "ba": 1})
    assert vocab.base_vocab_size == 10
    assert vocab.extension_start_id == 10
    assert vocab.extension_strings == ("ab",)
    assert vocab.oov_id == 11
    assert vocab.size == 12

    mapped = counter_to_hybrid_counts(counters[0], vocab, tokenizer)
    assert mapped == Counter({10: 4, 9: 3, 1: 2, 11: 1})
    assert sum(mapped.values()) == 10


def test_decode_distinguishes_occupied_unused_and_oov_slots():
    tokenizer = FakeTokenizer()
    vocab, _ = build_hybrid_qwen_vocab(
        [{"ab": 1}], tokenizer, extension_capacity=3
    )

    assert vocab.decode_action(1, tokenizer) == "a"
    assert vocab.decode_action(10, tokenizer) == "ab"
    assert vocab.decode_action(11, tokenizer) is None
    assert vocab.decode_action(vocab.oov_id, tokenizer) is None


def test_roundtrip_and_full_vocab_mask(tmp_path):
    tokenizer = FakeTokenizer()
    vocab, frequencies = build_hybrid_qwen_vocab(
        [{"ab": 3, "ba": 1}], tokenizer, extension_capacity=3
    )
    path = tmp_path / "hybrid.json"
    vocab.save(
        path,
        tokenizer_name_or_path="fake-qwen",
        extension_frequencies=dict(frequencies),
    )

    loaded = type(vocab).load(path)
    assert loaded == vocab
    mask = loaded.valid_action_mask(tokenizer)
    assert mask.tolist() == [
        False, True, True, True, True, True, True, True, True, True,
        True, True, False, False,
    ]
