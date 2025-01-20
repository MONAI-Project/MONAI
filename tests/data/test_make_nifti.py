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

import os
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d
from monai.utils import optional_import
from tests.util import make_nifti_image

_, has_nib = optional_import("nibabel")

TESTS = []
for affine in (None, np.eye(4), torch.eye(4)):
    for dir in (None, tempfile.mkdtemp()):
        for fname in (None, "fname"):
            TESTS.append([{"affine": affine, "dir": dir, "fname": fname}])


@unittest.skipUnless(has_nib, "Requires nibabel")
class TestMakeNifti(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_make_nifti(self, params):
        im, _ = create_test_image_2d(100, 88)
        created_file = make_nifti_image(im, verbose=True, **params)
        self.assertTrue(os.path.isfile(created_file))


if __name__ == "__main__":
    unittest.main()
