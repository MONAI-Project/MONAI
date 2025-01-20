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
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized

from monai import __version__
from monai.apps import download_mmar, load_from_mmar
from monai.apps.mmars import MODEL_DESC
from monai.apps.mmars.mmars import _get_val
from monai.utils import version_leq
from tests.util import skip_if_downloading_fails, skip_if_quick

TEST_CASES = [["clara_pt_prostate_mri_segmentation"], ["clara_pt_covid19_ct_lesion_segmentation"]]
TEST_EXTRACT_CASES = [
    (
        {"item": "clara_pt_prostate_mri_segmentation", "map_location": "cuda" if torch.cuda.is_available() else "cpu"},
        "UNet",
        np.array(
            [
                [[-0.0838, 0.0116, -0.0861], [-0.0792, 0.2216, -0.0301], [-0.0379, 0.0006, -0.0399]],
                [[-0.0347, 0.0979, 0.0754], [0.1689, 0.3759, 0.2584], [-0.0698, 0.2740, 0.1414]],
                [[-0.0772, 0.1046, -0.0103], [0.0917, 0.1942, 0.0284], [-0.0165, -0.0181, 0.0247]],
            ]
        ),
    ),
    (
        {
            "item": "clara_pt_covid19_ct_lesion_segmentation",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "SegResNet",
        np.array(
            [
                [
                    [0.01671106, 0.08502351, -0.1766469],
                    [-0.13039736, -0.06137804, 0.03924942],
                    [0.02268324, 0.159056, -0.03485069],
                ],
                [
                    [0.04788467, -0.09365353, -0.05802464],
                    [-0.19500689, -0.13514304, -0.08191573],
                    [0.0238207, 0.08029253, 0.10818923],
                ],
                [
                    [-0.11541673, -0.10622888, 0.039689],
                    [0.18462701, -0.0499289, 0.14309818],
                    [0.00528282, 0.02152331, 0.1698219],
                ],
            ]
        ),
    ),
    (
        {
            "item": "clara_pt_fed_learning_brain_tumor_mri_segmentation",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
            "model_file": os.path.join("models", "server", "best_FL_global_model.pt"),
        },
        "SegResNet",
        np.array(
            [
                [
                    [0.01874463, 0.12237817, 0.09269974],
                    [0.07691482, 0.00621202, -0.06682577],
                    [-0.07718472, 0.08637864, -0.03222707],
                ],
                [
                    [0.05117761, 0.07428649, -0.03053505],
                    [0.11045473, 0.07083791, 0.06547518],
                    [0.09555705, -0.03950734, -0.00819483],
                ],
                [
                    [0.03704128, 0.062543, 0.0380853],
                    [-0.02814676, -0.03078287, -0.01383446],
                    [-0.08137762, 0.01385882, 0.01229484],
                ],
            ]
        ),
    ),
    (
        {
            "item": "clara_pt_pathology_metastasis_detection",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "TorchVisionFCModel",
        np.array(
            [
                [-0.00540746, -0.00274996, -0.00837622, 0.05415914, 0.03555066, -0.00071636, -0.02325751],
                [0.00564625, 0.00674562, -0.1098334, -0.2936509, -0.28384757, -0.13580588, -0.00737865],
                [-0.02159783, 0.04615543, 0.29717407, 0.6001161, 0.53496915, 0.2528417, 0.04530451],
                [0.0225903, -0.07556137, -0.3070122, -0.43984795, -0.26286602, -0.00172576, 0.05003437],
                [-0.0320133, 0.00855468, 0.06824744, -0.04786247, -0.30358723, -0.3960023, -0.24895012],
                [0.02412516, 0.03411723, 0.06513759, 0.24332047, 0.41664436, 0.38999054, 0.15957521],
                [-0.01303542, -0.00166874, -0.01965466, -0.06620175, -0.15635538, -0.10023144, -0.01698002],
            ]
        ),
    ),
]


@unittest.skip("deprecating mmar tests")
class TestMMMARDownload(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skip_if_quick
    def test_download(self, idx):
        with skip_if_downloading_fails():
            with self.assertLogs(level="INFO", logger="monai.apps"):
                download_mmar(idx)
            download_mmar(idx, progress=False)  # repeated to check caching
            with tempfile.TemporaryDirectory() as tmp_dir:
                download_mmar(idx, mmar_dir=tmp_dir, progress=False)
                download_mmar(idx, mmar_dir=Path(tmp_dir), progress=False, version=1)  # repeated to check caching
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, idx)))

    @parameterized.expand(TEST_EXTRACT_CASES)
    @skip_if_quick
    @unittest.skipIf(version_leq(__version__, "0.6"), "requires newer monai")
    def test_load_ckpt(self, input_args, expected_name, expected_val):
        with skip_if_downloading_fails():
            output = load_from_mmar(**input_args)
        self.assertEqual(output.__class__.__name__, expected_name)
        x = next(output.parameters())  # verify the first element
        np.testing.assert_allclose(x[0][0].detach().cpu().numpy(), expected_val, rtol=1e-3, atol=1e-3)

    def test_unique(self):
        # model ids are unique
        keys = sorted(m["id"] for m in MODEL_DESC)
        self.assertEqual(keys, sorted(set(keys)))

    def test_search(self):
        self.assertEqual(_get_val({"a": 1, "b": 2}, key="b"), 2)
        self.assertEqual(_get_val({"a": {"c": {"c": 4}}, "b": {"c": 2}}, key="b"), {"c": 2})
        self.assertEqual(_get_val({"a": {"c": 4}, "b": {"c": 2}}, key="c"), 4)
        self.assertEqual(_get_val({"a": {"c": None}, "b": {"c": 2}}, key="c"), 2)


if __name__ == "__main__":
    unittest.main()
