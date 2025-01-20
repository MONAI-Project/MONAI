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

from numpy import ndarray
from torch import Tensor

from monai.data.meta_tensor import MetaTensor
from monai.transforms.utility.dictionary import Lambdad
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor
from tests.util import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestLambdad(NumpyImageTestCase2D):
    def test_lambdad_identity(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)
            data = {"img": img, "prop": 1.0, "label": 1.0}

            def noise_func(x):
                return x + 1.0

            expected = {"img": noise_func(data["img"]), "prop": 1.0, "new_label": 2.0}
            ret = Lambdad(keys=["img", "prop", "label"], func=noise_func, overwrite=[True, False, "new_label"])(data)
            assert_allclose(expected["img"], ret["img"], type_test=False)
            assert_allclose(expected["prop"], ret["prop"], type_test=False)
            assert_allclose(expected["new_label"], ret["new_label"], type_test=False)

    def test_lambdad_slicing(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)
            data = {"img": img}

            def slice_func(x):
                return x[:, :, :6, ::2]

            lambd = Lambdad(keys=data.keys(), func=slice_func)
            expected = {}
            expected = slice_func(data["img"])
            out = lambd(data)
            out_img = out["img"]
            assert_allclose(expected, out_img, type_test=False)
            self.assertIsInstance(out_img, MetaTensor)
            self.assertEqual(len(out_img.applied_operations), 1)
            inv_img = lambd.inverse(out)["img"]
            self.assertIsInstance(inv_img, MetaTensor)
            self.assertEqual(len(inv_img.applied_operations), 0)

    def test_lambdad_track_meta_false(self):
        for p in TEST_NDARRAYS:
            img = p(self.imt)
            data = {"img": img}

            def to_numpy(x):
                return convert_to_numpy(x)

            lambd = Lambdad(keys=data.keys(), func=to_numpy, track_meta=False)
            out = lambd(data)
            out_img = out["img"]
            self.assertIsInstance(out_img, ndarray)

            def to_tensor(x):
                return convert_to_tensor(x)

            lambd = Lambdad(keys=data.keys(), func=to_tensor, track_meta=False)
            out = lambd(data)
            out_img = out["img"]
            self.assertIsInstance(out_img, Tensor)


if __name__ == "__main__":
    unittest.main()
