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
from torch.autograd import Variable

from monai.optimizers import Novograd


def build_test_cases(data):
    [weight, bias, input] = data
    weight = Variable(weight, requires_grad=True)
    bias = Variable(bias, requires_grad=True)
    input = Variable(input)

    default_params = {"lr": 1e-3, "amsgrad": False, "grad_averaging": False, "weight_decay": 0}

    test_case_same_param = [{"params": [weight, bias]}]
    test_case_diff_param = [
        {"params": [weight]},
        {"params": [bias], "lr": 1e-2, "amsgrad": True, "grad_averaging": True, "weight_decay": 0.1},
    ]

    test_cases = [
        [test_case_same_param, default_params, weight, bias, input],
        [test_case_diff_param, default_params, weight, bias, input],
    ]
    return test_cases


TEST_CASES_ALL = build_test_cases([torch.randn(10, 5), torch.randn(10), torch.randn(5)])  # normal parameters

TEST_CASES_ALL += build_test_cases(  # non-contiguous parameters
    [torch.randn(10, 5, 2)[..., 0], torch.randn(10, 2)[..., 0], torch.randn(5)]
)

if torch.cuda.is_available():
    TEST_CASES_ALL += build_test_cases(  # gpu parameters
        [torch.randn(10, 5).cuda(), torch.randn(10).cuda(), torch.randn(5).cuda()]
    )
if torch.cuda.device_count() > 1:
    TEST_CASES_ALL += build_test_cases(  # multi-gpu parameters
        [torch.randn(10, 5).cuda(0), torch.randn(10).cuda(1), torch.randn(5).cuda(0)]
    )


class TestNovograd(unittest.TestCase):
    """
    This class takes `Pytorch's test_optim function:
    https://github.com/pytorch/pytorch/blob/v1.9.0/test/test_optim.py for reference.

    """

    @parameterized.expand(TEST_CASES_ALL)
    def test_step(self, specify_param, default_param, weight, bias, input):
        optimizer = Novograd(specify_param, **default_param)

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().item()
        for _ in range(100):
            optimizer.step(fn)
        self.assertLess(fn().item(), initial_value)

    def test_ill_arg(self):
        param = {"params": [Variable(torch.randn(10), requires_grad=True)]}
        with self.assertRaisesRegex(ValueError, "Invalid learning rate: -1"):
            Novograd(param, lr=-1)
        with self.assertRaisesRegex(ValueError, "Invalid epsilon value: -1"):
            Novograd(param, eps=-1)
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
            Novograd(param, betas=(1.0, 0.98))
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 1: -1"):
            Novograd(param, betas=(0.9, -1))
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            Novograd(param, weight_decay=-1)


if __name__ == "__main__":
    unittest.main()
