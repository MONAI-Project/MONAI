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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandAxisFlip
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.util import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion


class TestRandAxisFlip(NumpyImageTestCase2D):
    def test_correct_results(self):
        for p in TEST_NDARRAYS_ALL:
            flip = RandAxisFlip(prob=1.0)
            flip.set_random_state(seed=321)
            im = p(self.imt[0])
            call_param = {"img": im}
            result = flip(**call_param)

            # test lazy
            test_resampler_lazy(flip, result, call_param=call_param, seed=321)
            flip.lazy = False

            expected = [np.flip(channel, flip._axis) for channel in self.imt[0]]
            assert_allclose(result, p(np.stack(expected)), type_test="tensor")
            test_local_inversion(flip, result, im)

            set_track_meta(False)
            result = flip(im)
            self.assertNotIsInstance(result, MetaTensor)
            self.assertIsInstance(result, torch.Tensor)
            set_track_meta(True)


if __name__ == "__main__":
    unittest.main()
