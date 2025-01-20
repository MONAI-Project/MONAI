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
import tempfile
import unittest

import torch

from monai.utils import optional_import
from tests.util import SkipIfNoModule

try:
    _, has_ignite = optional_import("ignite")
    from ignite.engine import Engine

    from monai.handlers import LogfileHandler
except ImportError:
    has_ignite = False


class TestHandlerLogfile(unittest.TestCase):
    def setUp(self):
        if has_ignite:
            # set up engine
            def _train_func(engine, batch):
                return torch.tensor(0.0)

            self.engine = Engine(_train_func)

            logger = self.engine.logger

            # remove all other handlers to prevent output
            while logger is not None:
                del logger.handlers[:]
                logger = logger.parent

    @SkipIfNoModule("ignite")
    def test_logfile(self):
        with tempfile.TemporaryDirectory() as tempdir:
            handler = LogfileHandler(output_dir=tempdir)
            handler.attach(self.engine)

            self.engine.run(range(3))

            self.assertTrue(os.path.isfile(os.path.join(tempdir, "log.txt")))

    @SkipIfNoModule("ignite")
    def test_filename(self):
        filename = "something_else.txt"

        with tempfile.TemporaryDirectory() as tempdir:
            handler = LogfileHandler(output_dir=tempdir, filename=filename)
            handler.attach(self.engine)

            self.engine.run(range(3))

            self.assertTrue(os.path.isfile(os.path.join(tempdir, filename)))

    @SkipIfNoModule("ignite")
    def test_createdir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output_dir = os.path.join(tempdir, "new_dir")

            handler = LogfileHandler(output_dir=output_dir)
            handler.attach(self.engine)

            self.engine.run(range(3))

            self.assertTrue(os.path.isfile(os.path.join(output_dir, "log.txt")))


if __name__ == "__main__":
    unittest.main()
