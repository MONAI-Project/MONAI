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
import unittest
from os.path import join as pathjoin
from pathlib import Path

import torch.distributed as dist

from monai.bundle import ConfigParser, ConfigWorkflow
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from monai.networks import get_state_dict
from tests.util import DistCall, DistTestCase, SkipIfBeforePyTorchVersion, SkipIfNoModule, skip_if_no_cuda

TESTS_PATH = TESTS_PATH = Path(__file__).parents[3].as_posix()
_root_dir = os.path.abspath(pathjoin(TESTS_PATH))
_data_dir = pathjoin(_root_dir, "testing_data")
_logging_file = pathjoin(_data_dir, "logging.conf")


@SkipIfNoModule("ignite")
@SkipIfBeforePyTorchVersion((1, 11, 1))
class TestFLMonaiAlgo(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    @skip_if_no_cuda
    def test_train(self):
        train_configs = [pathjoin(_data_dir, "config_fl_train.json"), pathjoin(_data_dir, "multi_gpu_train.json")]
        eval_configs = [
            pathjoin(_data_dir, "config_fl_train.json"),
            pathjoin(_data_dir, "config_fl_evaluate.json"),
            pathjoin(_data_dir, "multi_gpu_evaluate.json"),
        ]
        train_workflow = ConfigWorkflow(config_file=train_configs, workflow_type="train", logging_file=_logging_file)
        # simulate the case that this application has specific requirements for a bundle workflow
        train_workflow.add_property(name="loader", required=True, config_id="train#training_transforms#0", desc="NA")

        # initialize algo
        algo = MonaiAlgo(
            bundle_root=_data_dir,
            train_workflow=ConfigWorkflow(config_file=train_configs, workflow_type="train", logging_file=_logging_file),
            eval_workflow=ConfigWorkflow(config_file=eval_configs, workflow_type="train", logging_file=_logging_file),
            config_filters_filename=pathjoin(_root_dir, "testing_data", "config_fl_filters.json"),
        )
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config(train_configs)
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=get_state_dict(network))
        # test train
        for i in range(2):
            print(f"Testing round {i + 1} of {2}...")
            # test evaluate
            metric_eo = algo.evaluate(data=data, extra={})
            self.assertIsInstance(metric_eo, ExchangeObject)
            metric = metric_eo.metrics
            self.assertIsInstance(metric["accuracy"], float)

            # test train
            algo.train(data=data, extra={})
            weights_eo = algo.get_weights()
            self.assertIsInstance(weights_eo, ExchangeObject)
            self.assertTrue(weights_eo.is_valid_weights())
            self.assertIsInstance(weights_eo.weights, dict)
            self.assertTrue(len(weights_eo.weights) > 0)

    @DistCall(nnodes=1, nproc_per_node=2, init_method="no_init")
    @skip_if_no_cuda
    def test_evaluate(self):
        config_file = [
            pathjoin(_data_dir, "config_fl_train.json"),
            pathjoin(_data_dir, "config_fl_evaluate.json"),
            pathjoin(_data_dir, "multi_gpu_evaluate.json"),
        ]
        # initialize algo
        algo = MonaiAlgo(
            bundle_root=_data_dir,
            config_train_filename=None,
            eval_workflow=ConfigWorkflow(config_file=config_file, workflow_type="train", logging_file=_logging_file),
            config_filters_filename=pathjoin(_data_dir, "config_fl_filters.json"),
        )
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        self.assertTrue(dist.get_rank() in (0, 1))

        # initialize model
        parser = ConfigParser()
        parser.read_config(
            [pathjoin(_data_dir, "config_fl_train.json"), pathjoin(_data_dir, "config_fl_evaluate.json")]
        )
        parser.parse()
        network = parser.get_parsed_content("network")
        data = ExchangeObject(weights=get_state_dict(network))
        # test evaluate
        metric_eo = algo.evaluate(data=data, extra={})
        self.assertIsInstance(metric_eo, ExchangeObject)
        metric = metric_eo.metrics
        self.assertIsInstance(metric["accuracy"], float)


if __name__ == "__main__":
    unittest.main()
