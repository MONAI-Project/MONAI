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

from monai.networks.nets import DenseNet121
from monai.networks.utils import replace_modules, replace_modules_temp
from tests.util import TEST_DEVICES

TESTS = []
for device in TEST_DEVICES:
    for match_device in (True, False):
        # replace 1
        TESTS.append(("features.denseblock1.denselayer1.layers.relu1", True, match_device, *device))
        # replace 1 (but not strict)
        TESTS.append(("features.denseblock1.denselayer1.layers.relu1", False, match_device, *device))
        # replace multiple
        TESTS.append(("relu", False, match_device, *device))


class TestReplaceModule(unittest.TestCase):
    def setUp(self):
        self.net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        self.num_relus = self.get_num_modules(torch.nn.ReLU)
        self.total = self.get_num_modules()
        self.assertGreater(self.num_relus, 0)

    def get_num_modules(self, mod: type[torch.nn.Module] | None = None) -> int:
        m = [m for _, m in self.net.named_modules()]
        if mod is not None:
            m = [_m for _m in m if isinstance(_m, mod)]
        return len(m)

    def check_replaced_modules(self, name, match_device):
        # total num modules should remain the same
        self.assertEqual(self.total, self.get_num_modules())
        num_relus_mod = self.get_num_modules(torch.nn.ReLU)
        num_softmax = self.get_num_modules(torch.nn.Softmax)
        # list of returned modules should be as long as number of softmax
        self.assertEqual(self.num_relus, num_relus_mod + num_softmax)
        if name == "relu":
            # at least 2 softmaxes
            self.assertGreaterEqual(num_softmax, 2)
        else:
            # one softmax
            self.assertEqual(num_softmax, 1)
        if match_device:
            self.assertEqual(len(list({i.device for i in self.net.parameters()})), 1)

    @parameterized.expand(TESTS)
    def test_replace(self, name, strict_match, match_device, device):
        self.net.to(device)
        # replace module(s)
        replaced = replace_modules(self.net, name, torch.nn.Softmax(), strict_match, match_device)
        self.check_replaced_modules(name, match_device)
        # number of returned modules should equal number of softmax modules
        self.assertEqual(len(replaced), self.get_num_modules(torch.nn.Softmax))
        # all replaced modules should be ReLU
        for r in replaced:
            self.assertIsInstance(r[1], torch.nn.ReLU)
        # if a specfic module was named, check that the name matches exactly
        if name == "features.denseblock1.denselayer1.layers.relu1":
            self.assertEqual(replaced[0][0], name)

    @parameterized.expand(TESTS)
    def test_replace_context_manager(self, name, strict_match, match_device, device):
        self.net.to(device)
        with replace_modules_temp(self.net, name, torch.nn.Softmax(), strict_match, match_device):
            self.check_replaced_modules(name, match_device)
        # Check that model was correctly reverted
        self.assertEqual(self.get_num_modules(), self.total)
        self.assertEqual(self.get_num_modules(torch.nn.ReLU), self.num_relus)
        self.assertEqual(self.get_num_modules(torch.nn.Softmax), 0)

    def test_raises(self):
        # name doesn't exist in module
        with self.assertRaises(AttributeError):
            replace_modules(self.net, "non_existent_module", torch.nn.Softmax(), strict_match=True)
        with self.assertRaises(AttributeError):
            with replace_modules_temp(self.net, "non_existent_module", torch.nn.Softmax(), strict_match=True):
                pass


if __name__ == "__main__":
    unittest.main()
