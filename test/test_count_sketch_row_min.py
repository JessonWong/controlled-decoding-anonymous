import unittest

import torch

from modeling_biasnet import (
    BiasConfig,
    BiasNet,
    COUNT_SKETCH_INPUT_CENTERING_ROW_MIN,
)


class CountSketchRowMinTest(unittest.TestCase):
    def make_model(self, centering="none") -> BiasNet:
        model = BiasNet(
            BiasConfig(
                hidden_size=8,
                vocab_size=32,
                input_projection_mode="count_sketch",
                input_hidden_normalization="none",
                count_sketch_hashes=4,
                count_sketch_seed=17,
                count_sketch_input_centering=centering,
            )
        )
        model.set_up_proj()
        return model

    def test_row_min_centering_removes_dense_floor(self):
        centered = self.make_model(COUNT_SKETCH_INPUT_CENTERING_ROW_MIN)
        legacy = self.make_model()
        dense_floor = torch.full((2, 32), -9.0)
        dense_floor[0, [2, 7, 19]] = torch.tensor([-1.0, -3.0, -2.0])
        dense_floor[1, [1, 5]] = torch.tensor([-4.0, -2.0])
        explicit_centered = dense_floor - dense_floor.amin(dim=-1, keepdim=True)

        actual = centered.inverse_mapping(dense_floor)
        expected = legacy.inverse_mapping(explicit_centered)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_row_min_centering_is_shift_invariant(self):
        model = self.make_model(COUNT_SKETCH_INPUT_CENTERING_ROW_MIN)
        features = torch.full((2, 32), -8.0)
        features[0, [2, 7, 19]] = torch.tensor([-1.0, -3.0, -2.0])
        features[1] = features[0] + 123.0

        hidden = model.inverse_mapping(features)

        torch.testing.assert_close(hidden[0], hidden[1], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
