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

import numpy as np
import torch
from parameterized import parameterized

from monai.networks.utils import copy_model_state
from monai.utils import set_determinism


class _TestModelOne(torch.nn.Module):

    def __init__(self, n_n, n_m, n_class):
        super().__init__()
        self.layer = torch.nn.Linear(n_n, n_m)
        self.class_layer = torch.nn.Linear(n_m, n_class)

    def forward(self, x):
        x = self.layer(x)
        x = self.class_layer(x)
        return x


class _TestModelTwo(torch.nn.Module):

    def __init__(self, n_n, n_m, n_d, n_class):
        super().__init__()
        self.layer = torch.nn.Linear(n_n, n_m)
        self.layer_1 = torch.nn.Linear(n_m, n_d)
        self.class_layer = torch.nn.Linear(n_d, n_class)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_1(x)
        x = self.class_layer(x)
        return x


TEST_CASES = []
__devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
for _x in __devices:
    for _y in __devices:
        TEST_CASES.append((_x, _y))


class TestModuleState(unittest.TestCase):

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand(TEST_CASES)
    def test_set_state(self, device_0, device_1):
        set_determinism(0)
        model_one = _TestModelOne(10, 20, 3)
        model_two = _TestModelTwo(10, 20, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        model_dict, ch, unch = copy_model_state(model_one, model_two)
        x = np.random.randn(4, 10)
        x = torch.tensor(x, device=device_0, dtype=torch.float32)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [-0.36076584, -0.03177825, -0.7702266],
                [-0.0526831, -0.15855855, -0.01149344],
                [-0.3760508, -0.22485238, -0.0634037],
                [0.5977675, -0.67991066, 0.1919502],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 2)
        self.assertEqual(len(unch), 2)

    @parameterized.expand(TEST_CASES)
    def test_set_full_state(self, device_0, device_1):
        set_determinism(0)
        model_one = _TestModelOne(10, 20, 3)
        model_two = _TestModelOne(10, 20, 3)
        model_one.to(device_0)
        model_two.to(device_1)
        # test module input
        model_dict, ch, unch = copy_model_state(model_one, model_two)
        # test dict input
        model_dict, ch, unch = copy_model_state(model_dict, model_two)
        x = np.random.randn(4, 10)
        x = torch.tensor(x, device=device_0, dtype=torch.float32)
        output = model_one(x).detach().cpu().numpy()
        model_two.to(device_0)
        output_1 = model_two(x).detach().cpu().numpy()
        np.testing.assert_allclose(output, output_1, atol=1e-3)
        self.assertEqual(len(ch), 4)
        self.assertEqual(len(unch), 0)

    @parameterized.expand(TEST_CASES)
    def test_set_exclude_vars(self, device_0, device_1):
        set_determinism(0)
        model_one = _TestModelOne(10, 20, 3)
        model_two = _TestModelTwo(10, 20, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        # test skip layer.bias
        model_dict, ch, unch = copy_model_state(model_one, model_two, exclude_vars="layer.bias")
        x = np.random.randn(4, 10)
        x = torch.tensor(x, device=device_0, dtype=torch.float32)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [-0.34172416, 0.0375042, -0.98340976],
                [-0.03364138, -0.08927619, -0.2246768],
                [-0.35700908, -0.15556987, -0.27658707],
                [0.61680925, -0.6106281, -0.02123314],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 1)
        self.assertEqual(len(unch), 3)

    @parameterized.expand(TEST_CASES)
    def test_set_map_across(self, device_0, device_1):
        set_determinism(0)
        model_one = _TestModelOne(10, 10, 3)
        model_two = _TestModelTwo(10, 10, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        # test weight map
        model_dict, ch, unch = copy_model_state(
            model_one, model_two, mapping={"layer_1.weight": "layer.weight", "layer_1.bias": "layer_1.weight"}
        )
        model_one.load_state_dict(model_dict)
        x = np.random.randn(4, 10)
        x = torch.tensor(x, device=device_0, dtype=torch.float32)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [0.8244487, -0.19650555, 0.65723234],
                [0.71239626, 0.25617486, 0.5247122],
                [0.24168758, 1.0301148, 0.39089814],
                [0.25791705, 0.8653245, 0.14833644],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 2)
        self.assertEqual(len(unch), 2)

    @parameterized.expand(TEST_CASES)
    def test_set_prefix(self, device_0, device_1):
        set_determinism(0)
        model_one = torch.nn.Sequential(_TestModelOne(10, 20, 3))
        model_two = _TestModelTwo(10, 20, 10, 4)
        model_one.to(device_0)
        model_two.to(device_1)
        # test skip layer.bias
        model_dict, ch, unch = copy_model_state(
            model_one, model_two, dst_prefix="0.", exclude_vars="layer.bias", inplace=False
        )
        model_one.load_state_dict(model_dict)
        x = np.random.randn(4, 10)
        x = torch.tensor(x, device=device_0, dtype=torch.float32)
        output = model_one(x).detach().cpu().numpy()
        expected = np.array(
            [
                [-0.360766, -0.031778, -0.770227],
                [-0.052683, -0.158559, -0.011493],
                [-0.376051, -0.224852, -0.063404],
                [0.597767, -0.679911, 0.19195],
            ]
        )
        np.testing.assert_allclose(output, expected, atol=1e-3)
        self.assertEqual(len(ch), 2)
        self.assertEqual(len(unch), 2)


if __name__ == "__main__":
    unittest.main()
