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

from monai.networks.blocks.text_embedding import TextEncoder
from tests.util import skip_if_downloading_fails


class TestTextEncoder(unittest.TestCase):
    def test_test_encoding_shape(self):
        with skip_if_downloading_fails():
            # test 2D encoder
            text_encoder = TextEncoder(
                spatial_dims=2, out_channels=32, encoding="clip_encoding_universal_model_32", pretrained=True
            )
            text_encoding = text_encoder()
            self.assertEqual(text_encoding.shape, (32, 256, 1, 1))

            # test 3D encoder
            text_encoder = TextEncoder(
                spatial_dims=3, out_channels=32, encoding="clip_encoding_universal_model_32", pretrained=True
            )
            text_encoding = text_encoder()
            self.assertEqual(text_encoding.shape, (32, 256, 1, 1, 1))

        # test random enbedding 3D
        text_encoder = TextEncoder(spatial_dims=3, out_channels=32, encoding="rand_embedding", pretrained=True)
        text_encoding = text_encoder()
        self.assertEqual(text_encoding.shape, (32, 256, 1, 1, 1))

        # test random enbedding 2D
        text_encoder = TextEncoder(spatial_dims=2, out_channels=32, encoding="rand_embedding", pretrained=True)
        text_encoding = text_encoder()
        self.assertEqual(text_encoding.shape, (32, 256, 1, 1))


if __name__ == "__main__":
    unittest.main()
