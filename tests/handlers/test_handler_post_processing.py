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

from monai.engines import SupervisedEvaluator
from monai.handlers import PostProcessing
from monai.transforms import Activationsd, AsDiscreted, Compose, CopyItemsd
from tests.util import assert_allclose

# test lambda function as `transform`
TEST_CASE_1 = [{"transform": lambda x: dict(pred=x["pred"] + 1.0)}, False, torch.tensor([[[[1.9975], [1.9997]]]])]
# test composed postprocessing transforms as `transform`
TEST_CASE_2 = [
    {
        "transform": Compose(
            [
                CopyItemsd(keys="filename", times=1, names="filename_bak"),
                AsDiscreted(keys="pred", threshold=0.5, to_onehot=2),
            ]
        ),
        "event": "iteration_completed",
    },
    True,
    torch.tensor([[[[1.0], [1.0]], [[0.0], [0.0]]]]),
]


class TestHandlerPostProcessing(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_compute(self, input_params, decollate, expected):
        data = [
            {"image": torch.tensor([[[[2.0], [3.0]]]]), "filename": ["test1"]},
            {"image": torch.tensor([[[[6.0], [8.0]]]]), "filename": ["test2"]},
        ]
        # set up engine, PostProcessing handler works together with postprocessing transforms of engine
        engine = SupervisedEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=data,
            epoch_length=2,
            network=torch.nn.PReLU(),
            postprocessing=Compose([Activationsd(keys="pred", sigmoid=True)]),
            val_handlers=[PostProcessing(**input_params)],
            decollate=decollate,
        )
        engine.run()

        if isinstance(engine.state.output, list):
            # test decollated list items
            for o, e in zip(engine.state.output, expected):
                assert_allclose(o["pred"], e, atol=1e-4, rtol=1e-4, type_test=False)
                filename = o.get("filename_bak")
                if filename is not None:
                    self.assertEqual(filename, "test2")
        else:
            # test batch data
            assert_allclose(engine.state.output["pred"], expected, atol=1e-4, rtol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
