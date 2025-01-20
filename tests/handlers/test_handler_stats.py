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
import re
import tempfile
import unittest
from io import StringIO

import torch
from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.handlers import StatsHandler


def get_event_filter(e):

    def event_filter(_, event):
        if event in e:
            return True
        return False

    return event_filter


class TestHandlerStats(unittest.TestCase):

    @parameterized.expand([[True], [get_event_filter([1, 2])]])
    def test_metrics_print(self, epoch_log):
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "testing_metric"

        # set up engine
        def _train_func(engine, batch):
            return [torch.tensor(0.0)]

        engine = Engine(_train_func)

        # set up dummy metric
        @engine.on(Events.EPOCH_COMPLETED)
        def _update_metric(engine):
            current_metric = engine.state.metrics.get(key_to_print, 0.1)
            engine.state.metrics[key_to_print] = current_metric + 0.1

        # set up testing handler
        logger = logging.getLogger(key_to_handler)
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        stats_handler = StatsHandler(iteration_log=False, epoch_log=epoch_log, name=key_to_handler)
        stats_handler.attach(engine)

        max_epochs = 4
        engine.run(range(3), max_epochs=max_epochs)

        # check logging output
        output_str = log_stream.getvalue()
        log_handler.close()
        has_key_word = re.compile(f".*{key_to_print}.*")
        content_count = 0
        for line in output_str.split("\n"):
            if has_key_word.match(line):
                content_count += 1
        if epoch_log is True:
            self.assertEqual(content_count, max_epochs)
        else:
            self.assertEqual(content_count, 2)  # 2 = len([1, 2]) from event_filter

    @parameterized.expand([[True], [get_event_filter([1, 3])]])
    def test_loss_print(self, iteration_log):
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "myLoss"

        # set up engine
        def _train_func(engine, batch):
            return [torch.tensor(0.0)]

        engine = Engine(_train_func)

        # set up testing handler
        logger = logging.getLogger(key_to_handler)
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        stats_handler = StatsHandler(
            iteration_log=iteration_log, epoch_log=False, name=key_to_handler, tag_name=key_to_print
        )
        stats_handler.attach(engine)

        num_iters = 3
        max_epochs = 2
        engine.run(range(num_iters), max_epochs=max_epochs)

        # check logging output
        output_str = log_stream.getvalue()
        log_handler.close()
        has_key_word = re.compile(f".*{key_to_print}.*")
        content_count = 0
        for line in output_str.split("\n"):
            if has_key_word.match(line):
                content_count += 1
        if iteration_log is True:
            self.assertEqual(content_count, num_iters * max_epochs)
        else:
            self.assertEqual(content_count, 2)  # 2 = len([1, 3]) from event_filter

    def test_loss_dict(self):
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)
        key_to_handler = "test_logging"
        key_to_print = "myLoss1"

        # set up engine
        def _train_func(engine, batch):
            return [torch.tensor(0.0)]

        engine = Engine(_train_func)

        # set up testing handler
        logger = logging.getLogger(key_to_handler)
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        stats_handler = StatsHandler(name=key_to_handler, output_transform=lambda x: {key_to_print: x[0]})
        stats_handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        # check logging output
        output_str = log_stream.getvalue()
        log_handler.close()
        has_key_word = re.compile(f".*{key_to_print}.*")
        content_count = 0
        for line in output_str.split("\n"):
            if has_key_word.match(line):
                content_count += 1
        self.assertGreater(content_count, 0)

    def test_loss_file(self):
        key_to_handler = "test_logging"
        key_to_print = "myLoss"

        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_loss_stats.log")
            handler = logging.FileHandler(filename, mode="w")
            handler.setLevel(logging.INFO)

            # set up engine
            def _train_func(engine, batch):
                return [torch.tensor(0.0)]

            engine = Engine(_train_func)

            # set up testing handler
            logger = logging.getLogger(key_to_handler)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            stats_handler = StatsHandler(name=key_to_handler, tag_name=key_to_print)
            stats_handler.attach(engine)

            engine.run(range(3), max_epochs=2)
            handler.close()
            stats_handler.logger.removeHandler(handler)
            with open(filename) as f:
                output_str = f.read()
                has_key_word = re.compile(f".*{key_to_print}.*")
                content_count = 0
                for line in output_str.split("\n"):
                    if has_key_word.match(line):
                        content_count += 1
                self.assertGreater(content_count, 0)

    def test_exception(self):
        # set up engine
        def _train_func(engine, batch):
            raise RuntimeError("test exception.")

        engine = Engine(_train_func)

        # set up testing handler
        stats_handler = StatsHandler()
        stats_handler.attach(engine)

        with self.assertRaises(RuntimeError):
            engine.run(range(3), max_epochs=2)

    def test_attributes_print(self):
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)
        key_to_handler = "test_logging"

        # set up engine
        def _train_func(engine, batch):
            return [torch.tensor(0.0)]

        engine = Engine(_train_func)

        # set up dummy metric
        @engine.on(Events.EPOCH_COMPLETED)
        def _update_metric(engine):
            if not hasattr(engine.state, "test1"):
                engine.state.test1 = 0.1
                engine.state.test2 = 0.2
            else:
                engine.state.test1 += 0.1
                engine.state.test2 += 0.2

        # set up testing handler
        logger = logging.getLogger(key_to_handler)
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        stats_handler = StatsHandler(name=key_to_handler, state_attributes=["test1", "test2", "test3"])
        stats_handler.attach(engine)

        engine.run(range(3), max_epochs=2)

        # check logging output
        output_str = log_stream.getvalue()
        log_handler.close()
        has_key_word = re.compile(".*State values.*")
        content_count = 0
        for line in output_str.split("\n"):
            if has_key_word.match(line):
                content_count += 1
        self.assertGreater(content_count, 0)

    def test_default_logger(self):
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        log_handler.setLevel(logging.INFO)
        key_to_print = "myLoss"

        # set up engine
        def _train_func(engine, batch):
            return [torch.tensor(0.0)]

        engine = Engine(_train_func)
        engine.logger.addHandler(log_handler)

        # set up testing handler
        stats_handler = StatsHandler(name=None, tag_name=key_to_print)
        engine.logger.setLevel(logging.WARNING)
        with self.assertWarns(Warning):  # engine logging level warn
            stats_handler.attach(engine)
        # leverage `engine.logger` to print info
        engine.logger.setLevel(logging.INFO)
        level = logging.root.getEffectiveLevel()
        logging.basicConfig(level=logging.INFO)
        engine.run(range(3), max_epochs=2)
        logging.basicConfig(level=level)

        # check logging output
        output_str = log_stream.getvalue()
        log_handler.close()
        has_key_word = re.compile(f".*{key_to_print}.*")
        content_count = 0
        for line in output_str.split("\n"):
            if has_key_word.match(line):
                content_count += 1
        self.assertGreater(content_count, 0)


if __name__ == "__main__":
    unittest.main()
