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
from typing import TYPE_CHECKING
from unittest import skipUnless

import numpy as np
from parameterized import parameterized

from monai.transforms import ToPILd
from monai.utils import optional_import
from tests.util import TEST_NDARRAYS, assert_allclose

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageImage
    from PIL.Image import fromarray as pil_image_fromarray

    has_pil = True
else:
    pil_image_fromarray, has_pil = optional_import("PIL.Image", name="fromarray")
    PILImageImage, _ = optional_import("PIL.Image", name="Image")

im = [[1.0, 2.0], [3.0, 4.0]]
TESTS = [[{"keys": "image"}, {"image": p(im)}] for p in TEST_NDARRAYS]
if has_pil:
    TESTS.append([{"keys": "image"}, {"image": pil_image_fromarray(np.array(im))}])


class TestToPIL(unittest.TestCase):
    @parameterized.expand(TESTS)
    @skipUnless(has_pil, "Requires `pillow` package.")
    def test_values(self, input_param, test_data):
        result = ToPILd(**input_param)(test_data)[input_param["keys"]]
        self.assertTrue(isinstance(result, PILImageImage))
        assert_allclose(np.array(result), test_data[input_param["keys"]], type_test=False)


if __name__ == "__main__":
    unittest.main()
