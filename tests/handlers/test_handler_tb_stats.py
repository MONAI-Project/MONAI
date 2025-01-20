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
import tempfile
import unittest
from unittest.mock import MagicMock

from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.handlers import TensorBoardStatsHandler
from monai.utils import optional_import

SummaryWriter, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")


def get_event_filter(e):

    def event_filter(_, event):
        if event in e:
            return True
        return False

    return event_filter


@unittest.skipUnless(has_tb, "Requires SummaryWriter installation")
class TestHandlerTBStats(unittest.TestCase):

    def test_metrics_print(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1

            # set up testing handler
            stats_handler = TensorBoardStatsHandler(log_dir=tempdir, iteration_log=False, epoch_log=True)
            stats_handler.attach(engine)
            engine.run(range(3), max_epochs=2)
            stats_handler.close()
            # check logging output
            self.assertTrue(len(glob.glob(tempdir)) > 0)

    @parameterized.expand([[True], [get_event_filter([1, 2])]])
    def test_metrics_print_mock(self, epoch_log):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1

            # set up testing handler
            stats_handler = TensorBoardStatsHandler(log_dir=tempdir, iteration_log=False, epoch_log=epoch_log)
            stats_handler._default_epoch_writer = MagicMock()
            stats_handler.attach(engine)

            max_epochs = 4
            engine.run(range(3), max_epochs=max_epochs)
            stats_handler.close()
            # check logging output
            if epoch_log is True:
                self.assertEqual(stats_handler._default_epoch_writer.call_count, max_epochs)
            else:
                self.assertEqual(stats_handler._default_epoch_writer.call_count, 2)  # 2 = len([1, 2]) from event_filter

    def test_metrics_writer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1
                engine.state.test = current_metric

            # set up testing handler
            writer = SummaryWriter(log_dir=tempdir)
            stats_handler = TensorBoardStatsHandler(
                summary_writer=writer,
                iteration_log=True,
                epoch_log=False,
                output_transform=lambda x: {"loss": x[0] * 2.0},
                global_epoch_transform=lambda x: x * 3.0,
                state_attributes=["test"],
            )
            stats_handler.attach(engine)
            engine.run(range(3), max_epochs=2)
            writer.close()
            # check logging output
            self.assertTrue(len(glob.glob(tempdir)) > 0)

    @parameterized.expand([[True], [get_event_filter([1, 3])]])
    def test_metrics_writer_mock(self, iteration_log):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return [batch + 1.0]

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1
                engine.state.test = current_metric

            # set up testing handler
            writer = SummaryWriter(log_dir=tempdir)
            stats_handler = TensorBoardStatsHandler(
                summary_writer=writer,
                iteration_log=iteration_log,
                epoch_log=False,
                output_transform=lambda x: {"loss": x[0] * 2.0},
                global_epoch_transform=lambda x: x * 3.0,
                state_attributes=["test"],
            )
            stats_handler._default_iteration_writer = MagicMock()
            stats_handler.attach(engine)

            num_iters = 3
            max_epochs = 2
            engine.run(range(num_iters), max_epochs=max_epochs)
            writer.close()

            if iteration_log is True:
                self.assertEqual(stats_handler._default_iteration_writer.call_count, num_iters * max_epochs)
            else:
                self.assertEqual(
                    stats_handler._default_iteration_writer.call_count, 2
                )  # 2 = len([1, 3]) from event_filter


if __name__ == "__main__":
    unittest.main()
