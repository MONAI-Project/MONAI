# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest
from unittest.case import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.losses import HausdorffDTLoss, LogHausdorffDTLoss
from monai.utils import optional_import

_, has_scipy = optional_import("scipy")

TEST_CASES = []
for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
    TEST_CASES.append(
        [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], device=device),
            },
            0.509329,
        ]
    )
    TEST_CASES.append(
        [  # shape: (1, 1, 1, 2, 2), (1, 1, 1, 2, 2)
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[[1.0, -1.0], [-1.0, 1.0]]]]], device=device),
                "target": torch.tensor([[[[[1.0, 0.0], [1.0, 1.0]]]]], device=device),
            },
            0.509329,
        ]
    )
    TEST_CASES.append(
        [  # shape: (1, 1, 2, 2, 2), (1, 1, 2, 2, 2)
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[[1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]]], device=device),
                "target": torch.tensor([[[[[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]]], device=device),
            },
            0.375718,
        ]
    )
    TEST_CASES.append(
        [  # shape: (1, 2, 2, 2), (1, 2, 2, 2)
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]], device=device),
            },
            0.326994,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]], device=device),
            },
            0.455082,
        ]
    )
    TEST_CASES.append(
        [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
            {"include_background": False, "sigmoid": True},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]], device=device),
            },
            0.144659,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 2, 3, 1), (2, 1, 3, 1)
            {"include_background": True, "to_onehot_y": True, "sigmoid": True, "reduction": "none"},
            {
                "input": torch.tensor(
                    [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                    device=device,
                ),
                "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]], device=device),
            },
            [[[[0.407765]], [[0.407765]]], [[[0.5000]], [[0.5000]]]],
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 2, 3, 1), (2, 1, 3, 1)
            {"include_background": True, "to_onehot_y": True, "softmax": True},
            {
                "input": torch.tensor(
                    [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                    device=device,
                ),
                "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]], device=device),
            },
            0.357016,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 2, 3, 1), (2, 1, 3, 1)
            {"include_background": True, "to_onehot_y": True, "softmax": True, "reduction": "sum"},
            {
                "input": torch.tensor(
                    [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                    device=device,
                ),
                "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]], device=device),
            },
            1.428062,
        ]
    )
    TEST_CASES.append(
        [  # shape: (1, 1, 2, 2), (1, 1, 2, 2)
            {"include_background": True, "sigmoid": True},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], device=device),
            },
            0.509329,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
            {"include_background": True, "other_act": torch.tanh},
            {
                "input": torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]], device=device),
            },
            1.870039,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 2, 3), (2, 1, 3)
            {"include_background": True, "to_onehot_y": True, "other_act": lambda x: torch.log_softmax(x, dim=1)},
            {
                "input": torch.tensor(
                    [[[[-1.0], [0.0], [1.0]], [[1.0], [0.0], [-1.0]]], [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]],
                    device=device,
                ),
                "target": torch.tensor([[[[1.0], [0.0], [0.0]]], [[[1.0], [1.0], [0.0]]]], device=device),
            },
            4.366613,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
            {"include_background": True, "other_act": torch.tanh, "batch": True},
            {
                "input": torch.tensor([[[[1.0, -0.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]], device=device),
            },
            1.607137,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
            {"include_background": True, "other_act": torch.tanh, "batch": True},
            {
                "input": torch.tensor([[[[1.0, -0.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]], device=device),
            },
            1.607137,
        ]
    )
    TEST_CASES.append(
        [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
            {"include_background": True, "other_act": torch.tanh, "batch": False},
            {
                "input": torch.tensor([[[[1.0, -0.0], [-1.0, 1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]], device=device),
                "target": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 0.0]]]], device=device),
            },
            1.607137,
        ]
    )

TEST_CASES_LOG = [[*inputs, np.log(np.array(output) + 1)] for *inputs, output in TEST_CASES]


def _describe_test_case(test_func, test_number, params):
    input_param, input_data, _ = params.args
    return f"params:{input_param}, shape:{input_data['input'].shape}, device:{input_data['input'].device}"


@skipUnless(has_scipy, "Scipy required")
class TestHausdorffDTLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES, doc_func=_describe_test_case)
    def test_shape(self, input_param, input_data, expected_val):
        result = HausdorffDTLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = HausdorffDTLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 1, 2, 3)), torch.ones((1, 4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            HausdorffDTLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            HausdorffDTLoss(reduction="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            HausdorffDTLoss(reduction=None)(chn_input, chn_target)

    @parameterized.expand([(False, False, False), (False, True, False), (False, False, True)])
    def test_input_warnings(self, include_background, softmax, to_onehot_y):
        chn_input = torch.ones((1, 1, 1, 3))
        chn_target = torch.ones((1, 1, 1, 3))
        with self.assertWarns(Warning):
            loss = HausdorffDTLoss(include_background=include_background, softmax=softmax, to_onehot_y=to_onehot_y)
            loss.forward(chn_input, chn_target)


@skipUnless(has_scipy, "Scipy required")
class TesLogtHausdorffDTLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES_LOG, doc_func=_describe_test_case)
    def test_shape(self, input_param, input_data, expected_val):
        result = LogHausdorffDTLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-5)

    def test_ill_shape(self):
        loss = LogHausdorffDTLoss()
        with self.assertRaisesRegex(AssertionError, ""):
            loss.forward(torch.ones((1, 1, 2, 3)), torch.ones((1, 4, 5, 6)))

    def test_ill_opts(self):
        with self.assertRaisesRegex(ValueError, ""):
            LogHausdorffDTLoss(sigmoid=True, softmax=True)
        chn_input = torch.ones((1, 1, 3))
        chn_target = torch.ones((1, 1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            LogHausdorffDTLoss(reduction="unknown")(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            LogHausdorffDTLoss(reduction=None)(chn_input, chn_target)

    @parameterized.expand([(False, False, False), (False, True, False), (False, False, True)])
    def test_input_warnings(self, include_background, softmax, to_onehot_y):
        chn_input = torch.ones((1, 1, 1, 3))
        chn_target = torch.ones((1, 1, 1, 3))
        with self.assertWarns(Warning):
            loss = LogHausdorffDTLoss(include_background=include_background, softmax=softmax, to_onehot_y=to_onehot_y)
            loss.forward(chn_input, chn_target)


if __name__ == "__main__":
    unittest.main()
