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
from pathlib import Path

from parameterized import parameterized

from monai.bundle import ConfigWorkflow
from monai.fl.client import MonaiAlgoStats
from monai.fl.utils.constants import ExtraItems, FlStatistics
from monai.fl.utils.exchange_object import ExchangeObject
from tests.util import SkipIfNoModule

_root_dir = Path(__file__).resolve().parents[1]
_data_dir = os.path.join(_root_dir, "testing_data")
_logging_file = os.path.join(_data_dir, "logging.conf")

TEST_GET_DATA_STATS_1 = [
    {
        "bundle_root": _data_dir,
        "workflow": ConfigWorkflow(
            workflow_type="train",
            config_file=os.path.join(_data_dir, "config_fl_stats_1.json"),
            logging_file=_logging_file,
            meta_file=None,
        ),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_DATA_STATS_2 = [
    {
        "bundle_root": _data_dir,
        "config_train_filename": os.path.join(_data_dir, "config_fl_stats_2.json"),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]
TEST_GET_DATA_STATS_3 = [
    {
        "bundle_root": _data_dir,
        "workflow": ConfigWorkflow(
            workflow_type="train",
            config_file=[
                os.path.join(_data_dir, "config_fl_stats_1.json"),
                os.path.join(_data_dir, "config_fl_stats_2.json"),
            ],
            logging_file=_logging_file,
            meta_file=None,
        ),
        "config_filters_filename": os.path.join(_data_dir, "config_fl_filters.json"),
    }
]


@SkipIfNoModule("ignite")
class TestFLMonaiAlgo(unittest.TestCase):
    @parameterized.expand([TEST_GET_DATA_STATS_1, TEST_GET_DATA_STATS_2, TEST_GET_DATA_STATS_3])
    def test_get_data_stats(self, input_params):
        # initialize algo
        algo = MonaiAlgoStats(**input_params)
        algo.initialize(extra={ExtraItems.CLIENT_NAME: "test_fl", ExtraItems.APP_ROOT: _data_dir})

        requested_stats = {FlStatistics.HIST_BINS: 100, FlStatistics.HIST_RANGE: [-500, 500]}
        # test train
        stats = algo.get_data_stats(extra=requested_stats)
        self.assertIsInstance(stats, ExchangeObject)


if __name__ == "__main__":
    unittest.main()
