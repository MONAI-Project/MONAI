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
import shutil
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai._extensions import load_module
from monai.networks.layers import GaussianMixtureModel
from tests.util import skip_if_darwin, skip_if_no_cuda, skip_if_quick, skip_if_windows

TEST_CASES = [
    [
        # Case Description
        "2 batches, 1 dimensions, 1 channels, 2 classes, 2 mixtures",
        # Class Count
        2,
        # Mixture Count
        1,
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 0, 0, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0.2, 1, 0.8, 0.5]
            ],
        ],
        # Labels
        [
            # Batch 0
            [
                # Channel 0
                [1, -1, 0, -1, 1]
            ],
            # Batch 1
            [
                # Channel 0
                [1, 1, 0, 0, -1]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [0, 0, 1, 1, 0],
                # Channel 1
                [1, 1, 0, 0, 1],
            ],
            # Batch 1
            [
                # Channel 0
                [0, 0, 1, 1, 0.5],
                # Channel 1
                [1, 1, 0, 0, 0.5],
            ],
        ],
    ],
    [
        # Case Description
        "1 batches, 1 dimensions, 5 channels, 2 classes, 1 mixtures",
        # Class Count
        2,
        # Mixture Count
        1,
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [1.0, 0.9, 0.0, 0.0, 0.0],
                # Channel 1
                [0.0, 0.0, 0.3, 0.3, 0.4],
                # Channel 2
                [0.9, 0.8, 0.0, 0.0, 0.0],
                # Channel 3
                [0.7, 0.9, 0.0, 0.0, 0.0],
                # Channel 4
                [0.2, 0.1, 0.2, 0.2, 0.1],
            ]
        ],
        # Labels
        [
            # Batch 0
            [
                # Channel 0
                [0, 0, -1, 1, 1]
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 0, 0, 0],
                # Channel 1
                [0, 0, 1, 1, 1],
            ]
        ],
    ],
    [
        # Case Description
        "1 batches, 2 dimensions, 2 channels, 4 classes, 4 mixtures",
        # Class Count
        4,
        # Mixture Count
        1,
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [
                    [0.8, 0.8, 0.0, 0.0, 0.0],
                    [1.0, 0.9, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8, 0.9],
                    [0.0, 0.0, 0.0, 0.9, 1.0],
                ],
                # Channel 1
                [
                    [0.8, 0.8, 0.0, 0.0, 0.0],
                    [0.7, 0.7, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.5, 0.0, 0.0, 0.0],
                    [0.7, 0.6, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        # Labels
        [
            # Batch 0
            [
                # Channel 0
                [[-1, 1, -1, 0, -1], [1, -1, -1, -1, -1], [-1, -1, 0, -1, -1], [2, 2, -1, 3, -1], [-1, -1, -1, -1, 3]]
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                # Channel 1
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                # Channel 2
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                ],
                # Channel 3
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ],
            ]
        ],
    ],
    [
        # Case Description
        "1 batches, 3 dimensions, 1 channels, 2 classes, 1 mixtures",
        # Class Count
        2,
        # Mixture Count
        1,
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Slice 0
                    [[0.7, 0.6, 0.0], [0.5, 0.4, 0.0], [0.0, 0.0, 0.0]],
                    # Slice 1
                    [[0.5, 0.6, 0.0], [0.4, 0.3, 0.0], [0.0, 0.0, 0.0]],
                    # Slice 2
                    [[0.3, 0.3, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]],
                ]
            ]
        ],
        # Labels
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Slice 0
                    [[0, -1, -1], [0, -1, -1], [-1, -1, 1]],
                    # Slice 1
                    [[0, 0, -1], [-1, -1, 1], [-1, 1, 1]],
                    # Slice 2
                    [[0, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                ]
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Slice 0
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    # Slice 1
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    # Slice 2
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                # Channel 1
                [
                    # Slice 0
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                    # Slice 1
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                    # Slice 2
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                ],
            ]
        ],
    ],
]


@skip_if_quick
class GMMTestCase(unittest.TestCase):
    def setUp(self):
        self._var = os.environ.get("TORCH_EXTENSIONS_DIR")
        self.tempdir = tempfile.mkdtemp()
        os.environ["TORCH_EXTENSIONS_DIR"] = self.tempdir

    def tearDown(self) -> None:
        if self._var is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = f"{self._var}"
        shutil.rmtree(self.tempdir)

    @parameterized.expand(TEST_CASES)
    @skip_if_no_cuda
    def test_cuda(self, test_case_description, mixture_count, class_count, features, labels, expected):
        # Device to run on
        device = torch.device("cuda")

        # Create tensors
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.int32, device=device)

        # Create GMM
        gmm = GaussianMixtureModel(features_tensor.size(1), mixture_count, class_count, verbose_build=True)
        # reload GMM to confirm the build
        _ = GaussianMixtureModel(features_tensor.size(1), mixture_count, class_count, verbose_build=False)
        # reload quietly
        _ = GaussianMixtureModel(features_tensor.size(1), mixture_count, class_count, verbose_build=True)

        # Apply GMM
        gmm.learn(features_tensor, labels_tensor)
        results_tensor = gmm.apply(features_tensor)

        # Read back results
        results = results_tensor.cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(results, expected, atol=1e-3)

    @skip_if_darwin
    @skip_if_windows
    def test_load(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ImportError, ".*symbol.*"):  # expecting import error if no cuda
                load_module("gmm", {"CHANNEL_COUNT": 2, "MIXTURE_COUNT": 2, "MIXTURE_SIZE": 3}, verbose_build=True)
        else:
            load_module("gmm", {"CHANNEL_COUNT": 2, "MIXTURE_COUNT": 2, "MIXTURE_SIZE": 3}, verbose_build=True)


if __name__ == "__main__":
    unittest.main()
