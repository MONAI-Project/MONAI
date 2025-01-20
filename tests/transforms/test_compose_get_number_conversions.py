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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import Compose
from monai.transforms.compose import OneOf
from monai.transforms.transform import Transform
from monai.transforms.utils import get_number_image_type_conversions
from monai.utils import convert_to_numpy, convert_to_tensor

NP_ARR = np.ones((10, 10, 10))
PT_ARR = torch.as_tensor(NP_ARR)
KEY = "IMAGE"


def _apply(x, fn):
    if isinstance(x, dict):
        d = deepcopy(x)
        d[KEY] = fn(d[KEY])
        return d
    return fn(x)


class Load(Transform):

    def __init__(self, as_tensor):
        self.fn = lambda _: PT_ARR if as_tensor else NP_ARR

    def __call__(self, x):
        return _apply(x, self.fn)


class N(Transform):

    def __call__(self, x):
        return _apply(x, convert_to_numpy)


class T(Transform):

    def __call__(self, x):
        return _apply(x, convert_to_tensor)


class NT(Transform):

    def __call__(self, x):
        return _apply(x, lambda x: x)


class TCPU(Transform):

    def __call__(self, x):
        return _apply(x, lambda x: convert_to_tensor(x).cpu())


class TGPU(Transform):

    def __call__(self, x):
        return _apply(x, lambda x: convert_to_tensor(x).cuda())


TESTS: list[tuple] = []
for is_dict in (False, True):
    # same type depends on input
    TESTS.append(((N(), N()), is_dict, NP_ARR, 0))
    TESTS.append(((N(), N()), is_dict, PT_ARR, 1))
    TESTS.append(((T(), T()), is_dict, NP_ARR, 1))
    TESTS.append(((T(), T()), is_dict, PT_ARR, 0))

    # loading depends on loader's output type and following transform
    TESTS.append(((Load(as_tensor=False), N()), is_dict, "fname.nii", 0))
    TESTS.append(((Load(as_tensor=True), N()), is_dict, "fname.nii", 1))
    TESTS.append(((Load(as_tensor=False), T()), is_dict, "fname.nii", 1))
    TESTS.append(((Load(as_tensor=True), T()), is_dict, "fname.nii", 0))
    TESTS.append(((Load(as_tensor=True), NT()), is_dict, "fname.nii", 0))
    TESTS.append(((Load(as_tensor=True), NT()), is_dict, "fname.nii", 0))

    # no changes for ambivalent transforms
    TESTS.append(((NT(), NT()), is_dict, NP_ARR, 0))
    TESTS.append(((NT(), NT()), is_dict, PT_ARR, 0))

    # multiple conversions
    TESTS.append(((N(), T(), N()), is_dict, PT_ARR, 3))
    TESTS.append(((N(), NT(), T(), T(), NT(), NT(), N()), is_dict, PT_ARR, 3))

    # shouldn't matter that there are nested composes
    TESTS.append(((N(), NT(), T(), Compose([T(), NT(), NT(), N()])), is_dict, PT_ARR, 3))

    # changing device also counts
    if torch.cuda.is_available():
        TESTS.append(((TCPU(), TGPU(), TCPU()), is_dict, PT_ARR, 2))


class TestComposeNumConversions(unittest.TestCase):

    @parameterized.expand(TESTS)
    def test_get_number_of_conversions(self, transforms, is_dict, input, expected):
        input = input if not is_dict else {KEY: input, "Other": NP_ARR}
        tr = Compose(transforms)
        n = get_number_image_type_conversions(tr, input, key=KEY if is_dict else None)
        self.assertEqual(n, expected)

    def test_raises(self):
        tr = Compose([N(), OneOf([T(), T()])])
        with self.assertRaises(RuntimeError):
            get_number_image_type_conversions(tr, NP_ARR)


if __name__ == "__main__":
    unittest.main()
