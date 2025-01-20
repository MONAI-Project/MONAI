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

import logging
import os
import tempfile
import unittest

import torch.distributed as dist

from monai.utils import RankFilter
from tests.util import DistCall, DistTestCase


class DistributedRankFilterTest(DistTestCase):
    def setUp(self):
        self.log_dir = tempfile.TemporaryDirectory()

    @DistCall(nnodes=1, nproc_per_node=2)
    def test_rankfilter(self):
        logger = logging.getLogger(__name__)
        log_filename = os.path.join(self.log_dir.name, "records.log")
        h1 = logging.FileHandler(filename=log_filename)
        h1.setLevel(logging.WARNING)

        logger.addHandler(h1)

        logger.addFilter(RankFilter())
        logger.warning("test_warnings")

        dist.barrier()
        if dist.get_rank() == 0:
            with open(log_filename) as file:
                lines = [line.rstrip() for line in file]
            log_message = " ".join(lines)
            self.assertEqual(log_message.count("test_warnings"), 1)

    def tearDown(self) -> None:
        self.log_dir.cleanup()


class SingleRankFilterTest(unittest.TestCase):
    def tearDown(self) -> None:
        self.log_dir.cleanup()

    def setUp(self):
        self.log_dir = tempfile.TemporaryDirectory()

    def test_rankfilter_single_proc(self):
        logger = logging.getLogger(__name__)
        log_filename = os.path.join(self.log_dir.name, "records_sp.log")
        h1 = logging.FileHandler(filename=log_filename)
        h1.setLevel(logging.WARNING)
        logger.addHandler(h1)
        logger.addFilter(RankFilter())
        logger.warning("test_warnings")

        with open(log_filename) as file:
            lines = [line.rstrip() for line in file]
        logger.removeHandler(h1)
        h1.close()
        log_message = " ".join(lines)
        self.assertEqual(log_message.count("test_warnings"), 1)


if __name__ == "__main__":
    unittest.main()
