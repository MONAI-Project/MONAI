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

import torch
from parameterized import parameterized

from monai.losses import PatchAdversarialLoss

shapes_tensors = {"2d": [4, 1, 64, 64], "3d": [4, 1, 64, 64, 64]}
reductions = ["sum", "mean"]
criterion = ["bce", "least_squares", "hinge"]

TEST_CASE_CREATION_FAIL = [{"reduction": "sum", "criterion": "invalid"}]

TEST_CASES_LOSS_LOGIC_2D = []
TEST_CASES_LOSS_LOGIC_3D = []

for c in criterion:
    for r in reductions:
        TEST_CASES_LOSS_LOGIC_2D.append([{"reduction": r, "criterion": c}, shapes_tensors["2d"]])
        TEST_CASES_LOSS_LOGIC_3D.append([{"reduction": r, "criterion": c}, shapes_tensors["3d"]])

TEST_CASES_LOSS_LOGIC_LIST = []
for c in criterion:
    TEST_CASES_LOSS_LOGIC_LIST.append([{"reduction": "none", "criterion": c}, shapes_tensors["2d"]])
    TEST_CASES_LOSS_LOGIC_LIST.append([{"reduction": "none", "criterion": c}, shapes_tensors["3d"]])


class TestPatchAdversarialLoss(unittest.TestCase):

    def get_input(self, shape, is_positive):
        """
        Get tensor for the tests. The tensor is around (-1) or (+1), depending on
        is_positive.
        """
        if is_positive:
            offset = 1
        else:
            offset = -1
        return torch.ones(shape) * (offset) + 0.01 * torch.randn(shape)

    def test_criterion(self):
        """
        Make sure that unknown criterion fail.
        """
        with self.assertRaises(ValueError):
            PatchAdversarialLoss(**TEST_CASE_CREATION_FAIL[0])

    @parameterized.expand(TEST_CASES_LOSS_LOGIC_2D + TEST_CASES_LOSS_LOGIC_3D)
    def test_loss_logic(self, input_param: dict, shape_input: list):
        """
        We want to make sure that the adversarial losses do what they should.
        If the discriminator takes in a tensor that looks positive, yet the label is fake,
        the loss should be bigger than that obtained with a tensor that looks negative.
        Same for the real label, and for the generator.
        """
        loss = PatchAdversarialLoss(**input_param)
        fakes = self.get_input(shape_input, is_positive=False)
        reals = self.get_input(shape_input, is_positive=True)
        # Discriminator: fake label
        loss_disc_f_f = loss(fakes, target_is_real=False, for_discriminator=True)
        loss_disc_f_r = loss(reals, target_is_real=False, for_discriminator=True)
        assert loss_disc_f_f < loss_disc_f_r
        # Discriminator: real label
        loss_disc_r_f = loss(fakes, target_is_real=True, for_discriminator=True)
        loss_disc_r_r = loss(reals, target_is_real=True, for_discriminator=True)
        assert loss_disc_r_f > loss_disc_r_r
        # Generator:
        loss_gen_f = loss(fakes, target_is_real=True, for_discriminator=False)  # target_is_real is overridden
        loss_gen_r = loss(reals, target_is_real=True, for_discriminator=False)  # target_is_real is overridden
        assert loss_gen_f > loss_gen_r

    @parameterized.expand(TEST_CASES_LOSS_LOGIC_LIST)
    def test_multiple_discs(self, input_param: dict, shape_input):
        shapes = [shape_input] + [shape_input[0:2] + [int(i / j) for i in shape_input[2:]] for j in range(1, 3)]
        inputs = [self.get_input(shapes[i], is_positive=True) for i in range(len(shapes))]
        loss = PatchAdversarialLoss(**input_param)
        assert len(loss(inputs, for_discriminator=True, target_is_real=True)) == 3


if __name__ == "__main__":
    unittest.main()
