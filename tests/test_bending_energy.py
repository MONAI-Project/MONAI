import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.losses.deform import BendingEnergyLoss

TEST_CASES = [
    [
        {},
        {"input": torch.ones((1, 3, 5, 5, 5))},
        0.0,
    ],
    [
        {},
        {"input": torch.arange(0, 5)[None, None, None, None, :].expand(1, 3, 5, 5, 5)},
        0.0,
    ],
    [
        {},
        {"input": torch.arange(0, 5)[None, None, None, None, :].expand(1, 3, 5, 5, 5) ** 2},
        4.0,
    ],
]


class TestBendingEnergy(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = BendingEnergyLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = BendingEnergyLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 5, 5, 5)))
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 3, 4, 5, 5)))
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 3, 5, 4, 5)))
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 3, 5, 5, 4)))

    def test_ill_opts(self):
        input = torch.rand(1, 3, 5, 5, 5)
        with self.assertRaisesRegex(ValueError, ""):
            BendingEnergyLoss(reduction="unknown")(input)
        with self.assertRaisesRegex(ValueError, ""):
            BendingEnergyLoss(reduction=None)(input)


if __name__ == "__main__":
    unittest.main()
