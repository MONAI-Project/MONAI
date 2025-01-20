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
from parameterized import parameterized

from monai.networks.layers import grid_pull
from monai.networks.utils import meshgrid_ij
from monai.utils import optional_import
from tests.testing_data.cpp_resample_answers import Expected_1D_GP_bwd, Expected_1D_GP_fwd
from tests.util import skip_if_no_cpp_extension

BType, has_b_type = optional_import("monai._C", name="BoundType")
PType, has_p_type = optional_import("monai._C", name="InterpolationType")


def make_grid(shape, dtype=None, device=None, requires_grad=True):
    ranges = [torch.arange(float(s), dtype=dtype, device=device, requires_grad=requires_grad) for s in shape]
    grid = torch.stack(meshgrid_ij(*ranges), dim=-1)
    return grid[None]


# 1D combinations of bounds/interpolations
bounds = set(BType.__members__.values()) if has_b_type else []
interps = set(PType.__members__.values()) if has_p_type else []
device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_1D_GP = []
for bound in bounds:
    for interp in interps:
        if not Expected_1D_GP_fwd or not Expected_1D_GP_bwd:
            break  # skip if the testing data are unavailable
        expected_val = Expected_1D_GP_fwd.pop(0)

        for input_g in (True, False):
            for grid_g in (True, False):
                expected_grad = Expected_1D_GP_bwd.pop(0)
                test_case = [
                    {
                        "input": torch.arange(10, dtype=torch.float, requires_grad=input_g, device=device).reshape(
                            (1, 1, 10)
                        ),
                        "grid": make_grid((20,), dtype=torch.float, device=device, requires_grad=grid_g) + 0.5,
                        "interpolation": interp,
                        "bound": bound,
                    },
                    {"val": torch.tensor([[expected_val]]), "device": device, "grad": torch.tensor(expected_grad)},
                ]
                TEST_1D_GP.append(test_case)


@skip_if_no_cpp_extension
class TestGridPull(unittest.TestCase):
    @parameterized.expand(TEST_1D_GP, skip_on_empty=True)
    def test_grid_pull(self, input_param, expected):
        result = grid_pull(**input_param)
        if input_param["input"].requires_grad:
            input_param["input"].retain_grad()
        if input_param["grid"].requires_grad:
            input_param["grid"].retain_grad()
        if input_param["input"].requires_grad or input_param["grid"].requires_grad:
            result.sum().backward()

        grads = []
        if input_param["input"].requires_grad:
            grads.append(input_param["input"].grad.view(-1))
        if input_param["grid"].requires_grad:
            grads.append(input_param["grid"].grad.view(-1))
        if not grads:
            grads = torch.tensor(0.0, device=result.device)
        elif len(grads) == 1:
            grads = grads[0]
        else:
            grads = torch.cat(grads, dim=0)
        self.assertTrue(f"{result.device}".startswith(expected["device"]))
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected["val"].cpu().numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(grads.detach().cpu().numpy(), expected["grad"].cpu().numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
