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

from monai.networks import eval_mode
from monai.transforms import CopyItemsd
from monai.utils import ensure_tuple
from tests.util import assert_allclose

TEST_CASE_1 = ["img", 1, "img_1"]

TEST_CASE_2 = [["img", "seg"], 1, ["img_1", "seg_1"]]

TEST_CASE_3 = ["img", 2, ["img_1", "img_2"]]

TEST_CASE_4 = [["img", "seg"], 2, ["img_1", "seg_1", "img_2", "seg_2"]]


class TestCopyItemsd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_numpy_values(self, keys, times, names):
        input_data = {"img": np.array([[0, 1], [1, 2]]), "seg": np.array([[3, 4], [4, 5]])}
        result = CopyItemsd(keys=keys, times=times, names=names)(input_data)
        for name in ensure_tuple(names):
            self.assertTrue(name in result)
        result["img_1"] += 1
        np.testing.assert_allclose(result["img_1"], np.array([[1, 2], [2, 3]]))
        np.testing.assert_allclose(result["img"], np.array([[0, 1], [1, 2]]))

    def test_default_names(self):
        input_data = {"img": np.array([[0, 1], [1, 2]]), "seg": np.array([[3, 4], [4, 5]])}
        result = CopyItemsd(keys=["img", "seg"], times=2, names=None)(input_data)
        for name in ["img_0", "seg_0", "img_1", "seg_1"]:
            self.assertTrue(name in result)

    def test_tensor_values(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
        input_data = {
            "img": torch.tensor([[0, 1], [1, 2]], device=device),
            "seg": torch.tensor([[0, 1], [1, 2]], device=device),
        }
        # test default `times=1`
        result = CopyItemsd(keys="img", names="img_1")(input_data)
        self.assertTrue("img_1" in result)
        result["img_1"] += 1
        assert_allclose(result["img"], torch.tensor([[0, 1], [1, 2]], device=device))
        assert_allclose(result["img_1"], torch.tensor([[1, 2], [2, 3]], device=device))

    def test_array_values(self):
        input_data = {"img": [[0, 1], [1, 2]], "seg": [[0, 1], [1, 2]]}
        result = CopyItemsd(keys="img", times=1, names="img_1")(input_data)
        self.assertTrue("img_1" in result)
        result["img_1"][0][0] += 1
        np.testing.assert_allclose(result["img"], [[0, 1], [1, 2]])
        np.testing.assert_allclose(result["img_1"], [[1, 1], [1, 2]])

    def test_graph_tensor_values(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
        net = torch.nn.PReLU().to(device)
        with eval_mode(net):
            pred = net(torch.tensor([[0.0, 1.0], [1.0, 2.0]], device=device))
        input_data = {"pred": pred, "seg": torch.tensor([[0.0, 1.0], [1.0, 2.0]], device=device)}
        result = CopyItemsd(keys="pred", times=1, names="pred_1")(input_data)
        self.assertTrue("pred_1" in result)
        result["pred_1"] += 1.0
        assert_allclose(result["pred"], torch.tensor([[0.0, 1.0], [1.0, 2.0]], device=device))
        assert_allclose(result["pred_1"], torch.tensor([[1.0, 2.0], [2.0, 3.0]], device=device))


if __name__ == "__main__":
    unittest.main()
