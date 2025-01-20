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

from monai.data import MetaTensor
from monai.transforms import ConcatItemsd
from tests.util import assert_allclose


class TestConcatItemsd(unittest.TestCase):
    def test_tensor_values(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
        input_data = {
            "img1": torch.tensor([[0, 1], [1, 2]], device=device),
            "img2": torch.tensor([[0, 1], [1, 2]], device=device),
        }
        result = ConcatItemsd(keys=["img1", "img2"], name="cat_img")(input_data)
        self.assertIn("cat_img", result)
        result["cat_img"] += 1
        assert_allclose(result["img1"], torch.tensor([[0, 1], [1, 2]], device=device))
        assert_allclose(result["cat_img"], torch.tensor([[1, 2], [2, 3], [1, 2], [2, 3]], device=device))

    def test_metatensor_values(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu:0")
        input_data = {
            "img1": MetaTensor([[0, 1], [1, 2]], device=device),
            "img2": MetaTensor([[0, 1], [1, 2]], device=device),
        }
        result = ConcatItemsd(keys=["img1", "img2"], name="cat_img")(input_data)
        self.assertIn("cat_img", result)
        self.assertIsInstance(result["cat_img"], MetaTensor)
        self.assertEqual(result["img1"].meta, result["cat_img"].meta)
        result["cat_img"] += 1
        assert_allclose(result["img1"], torch.tensor([[0, 1], [1, 2]], device=device))
        assert_allclose(result["cat_img"], torch.tensor([[1, 2], [2, 3], [1, 2], [2, 3]], device=device))

    def test_numpy_values(self):
        input_data = {"img1": np.array([[0, 1], [1, 2]]), "img2": np.array([[0, 1], [1, 2]])}
        result = ConcatItemsd(keys=["img1", "img2"], name="cat_img")(input_data)
        self.assertIn("cat_img", result)
        result["cat_img"] += 1
        np.testing.assert_allclose(result["img1"], np.array([[0, 1], [1, 2]]))
        np.testing.assert_allclose(result["cat_img"], np.array([[1, 2], [2, 3], [1, 2], [2, 3]]))

    def test_single_numpy(self):
        input_data = {"img": np.array([[0, 1], [1, 2]])}
        result = ConcatItemsd(keys="img", name="cat_img")(input_data)
        result["cat_img"] += 1
        np.testing.assert_allclose(result["img"], np.array([[0, 1], [1, 2]]))
        np.testing.assert_allclose(result["cat_img"], np.array([[1, 2], [2, 3]]))

    def test_single_tensor(self):
        input_data = {"img": torch.tensor([[0, 1], [1, 2]])}
        result = ConcatItemsd(keys="img", name="cat_img")(input_data)
        result["cat_img"] += 1
        assert_allclose(result["img"], torch.tensor([[0, 1], [1, 2]]))
        assert_allclose(result["cat_img"], torch.tensor([[1, 2], [2, 3]]))

    def test_single_metatensor(self):
        input_data = {"img": MetaTensor([[0, 1], [1, 2]])}
        result = ConcatItemsd(keys="img", name="cat_img")(input_data)
        result["cat_img"] += 1
        assert_allclose(result["img"], torch.tensor([[0, 1], [1, 2]]))
        assert_allclose(result["cat_img"], torch.tensor([[1, 2], [2, 3]]))


if __name__ == "__main__":
    unittest.main()
