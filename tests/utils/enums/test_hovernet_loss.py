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

import random
import unittest

import numpy as np
import torch
from parameterized import parameterized
from torch.nn import functional as F

from monai.apps.pathology.losses import HoVerNetLoss
from monai.transforms import GaussianSmooth, Rotate
from monai.transforms.intensity.array import ComputeHoVerMaps
from monai.utils.enums import HoVerNetBranch

device = "cuda" if torch.cuda.is_available() else "cpu"

s = 10e-8
t = 1.0 - s
H = 40
W = 40
N = 5
B = 2


class PrepareTestInputs:

    def __init__(self, inputs):
        self.inputs = {HoVerNetBranch.NP: inputs[1], HoVerNetBranch.HV: inputs[3]}
        self.targets = {HoVerNetBranch.NP: inputs[0], HoVerNetBranch.HV: inputs[2]}

        if len(inputs) > 4:
            self.targets[HoVerNetBranch.NC] = inputs[4]
            self.inputs[HoVerNetBranch.NC] = inputs[5]


def test_shape_generator(num_classes=1, num_objects=3, batch_size=1, height=5, width=5, rotation=0.0, smoothing=False):
    t_g = torch.zeros((batch_size, height, width), dtype=torch.int64)
    t_p = None
    hv_g = torch.zeros((batch_size, 2, height, width))
    hv_p = torch.zeros((batch_size, 2, height, width))

    rad_min = 2
    rad_max = min(max(height // 3, width // 3, rad_min), 5)

    for b in range(batch_size):
        random.seed(10 + b)
        inst_map = torch.zeros((height, width), dtype=torch.int64)
        for inst_id in range(1, num_objects + 1):
            x = random.randint(rad_max, width - rad_max)
            y = random.randint(rad_max, height - rad_max)
            rad = random.randint(rad_min, rad_max)
            spy, spx = np.ogrid[-x : height - x, -y : width - y]
            circle = torch.tensor((spx * spx + spy * spy) <= rad * rad)

            if num_classes > 1:
                t_g[b, circle] = np.ceil(random.random() * num_classes)
            else:
                t_g[b, circle] = 1

            inst_map[circle] = inst_id

        hv_g[b] = ComputeHoVerMaps()(inst_map[None])
        hv_g[b] = hv_g[b].squeeze(0)
        if rotation > 0.0:
            hv_p[b] = Rotate(angle=rotation, keep_size=True, mode="bilinear")(hv_g[b])

    n_g = t_g > 0
    if rotation == 0.0:
        hv_p = hv_g * 0.99

    # rotation of prediction needs to happen before one-hot encoding
    if rotation > 0.0:
        n_p = Rotate(angle=rotation, keep_size=True, mode="nearest")(n_g)
        n_p = F.one_hot(n_p.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)
        if num_classes > 1:
            t_p = Rotate(angle=rotation, keep_size=True, mode="nearest")(t_g)
            t_p = F.one_hot(t_p.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)
            t_g = F.one_hot(t_g.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)
        else:
            t_g = None
    else:
        n_p = F.one_hot(n_g.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)
        if num_classes > 1:
            t_p = F.one_hot(t_g.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)
            t_g = F.one_hot(t_g.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)
        else:
            t_g = None

    n_g = F.one_hot(n_g.to(torch.int64)).to(torch.float32).permute(0, 3, 1, 2)

    if smoothing:
        n_p = GaussianSmooth()(n_p)
        if num_classes > 1:
            t_p = GaussianSmooth()(t_p)
        hv_p = hv_p * 0.1
    else:
        n_p = torch.clamp(n_p, s, t)
        if num_classes > 1:
            t_p = torch.clamp(t_p, s, t)

    # Apply log to emulate logits
    if t_p is not None:
        return n_g, n_p.log(), hv_g, hv_p, t_g, t_p.log()
    else:
        return n_g, n_p.log(), hv_g, hv_p


inputs_test = [
    PrepareTestInputs(test_shape_generator(height=H, width=W)),
    PrepareTestInputs(test_shape_generator(num_classes=N, height=H, width=W)),
    PrepareTestInputs(test_shape_generator(num_classes=N, batch_size=B, height=H, width=W)),
    PrepareTestInputs(test_shape_generator(num_classes=N, batch_size=B, height=H, width=W, rotation=0.15)),
    PrepareTestInputs(test_shape_generator(num_classes=N, batch_size=B, height=H, width=W, rotation=0.2)),
    PrepareTestInputs(test_shape_generator(num_classes=N, batch_size=B, height=H, width=W, rotation=0.25)),
]

TEST_CASE_0 = [  # batch size of 1, no type prediction
    {"prediction": inputs_test[0].inputs, "target": inputs_test[0].targets},
    0.003,
]

TEST_CASE_1 = [  # batch size of 1, 2 classes with type prediction
    {"prediction": inputs_test[1].inputs, "target": inputs_test[1].targets},
    0.2762,
]

TEST_CASE_2 = [  # batch size of 2, 2 classes with type prediction
    {"prediction": inputs_test[2].inputs, "target": inputs_test[2].targets},
    0.4852,
]

TEST_CASE_3 = [  # batch size of 2, 3 classes with minor rotation of nuclear prediction
    {"prediction": inputs_test[3].inputs, "target": inputs_test[3].targets},
    3.6348,
]

TEST_CASE_4 = [  # batch size of 2, 3 classes with medium rotation of nuclear prediction
    {"prediction": inputs_test[4].inputs, "target": inputs_test[4].targets},
    4.5312,
]

TEST_CASE_5 = [  # batch size of 2, 3 classes with medium rotation of nuclear prediction
    {"prediction": inputs_test[5].inputs, "target": inputs_test[5].targets},
    5.4929,
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5]

ILL_CASES = [
    [
        {
            "prediction": {"np": inputs_test[0].inputs[HoVerNetBranch.NP]},
            "target": {
                "np": inputs_test[0].targets[HoVerNetBranch.NP],
                HoVerNetBranch.HV: inputs_test[0].targets[HoVerNetBranch.HV],
            },
        }
    ]
]


class TestHoverNetLoss(unittest.TestCase):

    @parameterized.expand(CASES)
    def test_shape(self, input_param, expected_loss):
        loss = HoVerNetLoss()
        result = loss(**input_param).to(device)
        self.assertAlmostEqual(float(result), expected_loss, places=2)

    @parameterized.expand(ILL_CASES)
    def test_ill_input_hyper_params(self, input_param):
        with self.assertRaises(ValueError):
            loss = HoVerNetLoss()
            _ = loss(**input_param).to(device)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
