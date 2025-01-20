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

import glob
import os
import shutil
import unittest
from copy import deepcopy
from os.path import join as pathjoin
from pathlib import Path

from parameterized import parameterized

from monai.bundle import ConfigParser, ConfigWorkflow
from monai.bundle.utils import DEFAULT_HANDLERS_ID
from monai.fl.client.monai_algo import MonaiAlgo
from monai.fl.utils.constants import ExtraItems
from monai.fl.utils.exchange_object import ExchangeObject
from monai.utils import path_to_uri
from tests.util import SkipIfNoModule

_root_dir = Path(__file__).resolve().parents[3]
_data_dir = os.path.join(_root_dir, "testing_data")
_logging_file = pathjoin(_data_dir, "logging.conf")

TEST_TRAIN_1 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_TRAIN_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "config_filters_filename": None,
    }
]
TEST_TRAIN_3 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]

TEST_TRAIN_4 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
            tracking={
                "handlers_id": DEFAULT_HANDLERS_ID,
                "configs": {
                    "save_execute_config": f"{_data_dir}/config_executed.json",
                    "trainer": {
                        "_target_": "MLFlowHandler",
                        "tracking_uri": path_to_uri(_data_dir) + "/mlflow_override",
                        "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
                        "close_on_complete": True,
                    },
                },
            },
        ),
        "config_evaluate_filename": None,
        "config_filters_filename": None,
    }
]

TEST_EVALUATE_1 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "eval_workflow": ConfigWorkflow(
            config_file=[
                os.path.join(_data_dir, "config_fl_train.json"),
                os.path.join(_data_dir, "config_fl_evaluate.json"),
            ],
            workflow_type="train",
            logging_file=_logging_file,
            tracking="mlflow",
            tracking_uri=path_to_uri(_data_dir) + "/mlflow_1",
            experiment_name="monai_eval1",
        ),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_EVALUATE_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "config_evaluate_filename": [
            os.path.join(_data_dir, "config_fl_train.json"),
            os.path.join(_data_dir, "config_fl_evaluate.json"),
        ],
        "eval_kwargs": {
            "tracking": "mlflow",
            "tracking_uri": path_to_uri(_data_dir) + "/mlflow_2",
            "experiment_name": "monai_eval2",
        },
        "eval_workflow_name": "training",
        "config_filters_filename": None,
    }
]
TEST_EVALUATE_3 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": None,
        "eval_workflow": ConfigWorkflow(
            config_file=[
                os.path.join(_data_dir, "config_fl_train.json"),
                os.path.join(_data_dir, "config_fl_evaluate.json"),
            ],
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]

TEST_GET_WEIGHTS_1 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "send_weight_diff": False,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_train.json"),
        "config_evaluate_filename": None,
        "send_weight_diff": True,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_WEIGHTS_3 = [
    {
        "bundle_root": _data_dir,
        "train_workflow": ConfigWorkflow(
            config_file=os.path.join(_data_dir, "config_fl_train.json"),
            workflow_type="train",
            logging_file=_logging_file,
        ),
        "config_evaluate_filename": None,
        "send_weight_diff": True,
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]


@SkipIfNoModule("ignite")
@SkipIfNoModule("mlflow")
class TestFLMonaiAlgo(unittest.TestCase):
    @parameterized.expand([TEST_TRAIN_1, TEST_TRAIN_2, TEST_TRAIN_3, TEST_TRAIN_4])
    def test_train(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})
        algo.abort()

        # initialize model
        parser = ConfigParser(config=deepcopy(algo.train_workflow.parser.get()))
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test train
        algo.train(data=data, extra={})
        algo.finalize()

        # test experiment management
        if "save_execute_config" in algo.train_workflow.parser:
            self.assertTrue(os.path.exists(f"{_data_dir}/mlflow_override"))
            shutil.rmtree(f"{_data_dir}/mlflow_override")
            self.assertTrue(os.path.exists(f"{_data_dir}/config_executed.json"))
            os.remove(f"{_data_dir}/config_executed.json")

    @parameterized.expand([TEST_EVALUATE_1, TEST_EVALUATE_2, TEST_EVALUATE_3])
    def test_evaluate(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # initialize model
        parser = ConfigParser(config=deepcopy(algo.eval_workflow.parser.get()))
        parser.parse()
        network = parser.get_parsed_content("network")

        data = ExchangeObject(weights=network.state_dict())

        # test evaluate
        algo.evaluate(data=data, extra={})

        # test experiment management
        if "save_execute_config" in algo.eval_workflow.parser:
            self.assertGreater(len(list(glob.glob(f"{_data_dir}/mlflow_*"))), 0)
            for f in list(glob.glob(f"{_data_dir}/mlflow_*")):
                shutil.rmtree(f)
            self.assertGreater(len(list(glob.glob(f"{_data_dir}/eval/config_*"))), 0)
            for f in list(glob.glob(f"{_data_dir}/eval/config_*")):
                os.remove(f)

    @parameterized.expand([TEST_GET_WEIGHTS_1, TEST_GET_WEIGHTS_2, TEST_GET_WEIGHTS_3])
    def test_get_weights(self, input_params):
        # initialize algo
        algo = MonaiAlgo(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl"})

        # test train
        if input_params["send_weight_diff"]:  # should not work as test doesn't receive a global model
            with self.assertRaises(ValueError):
                weights = algo.get_weights(extra={})
        else:
            weights = algo.get_weights(extra={})
            self.assertIsInstance(weights, ExchangeObject)


if __name__ == "__main__":
    unittest.main()
