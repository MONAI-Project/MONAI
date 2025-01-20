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

from ignite.engine import Engine, Events

from monai.handlers import EarlyStopHandler


class TestHandlerEarlyStop(unittest.TestCase):

    def test_early_stop_train_loss(self):

        def _train_func(engine, batch):
            return {"loss": 1.5}

        trainer = Engine(_train_func)
        EarlyStopHandler(
            patience=5, score_function=lambda x: x.state.output["loss"], trainer=trainer, epoch_level=False
        ).attach(trainer)

        trainer.run(range(4), max_epochs=2)
        self.assertEqual(trainer.state.iteration, 6)
        self.assertEqual(trainer.state.epoch, 2)

    def test_early_stop_val_metric(self):

        def _train_func(engine, batch):
            pass

        trainer = Engine(_train_func)
        validator = Engine(_train_func)
        validator.state.metrics["val_acc"] = 0.90

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            validator.state.metrics["val_acc"] += 0.01
            validator.run(range(3))

        handler = EarlyStopHandler(
            patience=3,
            score_function=lambda x: x.state.metrics["val_acc"],
            trainer=None,
            min_delta=0.1,
            cumulative_delta=True,
            epoch_level=True,
        )
        handler.attach(validator)
        handler.set_trainer(trainer=trainer)

        trainer.run(range(3), max_epochs=5)
        self.assertEqual(trainer.state.iteration, 12)
        self.assertEqual(trainer.state.epoch, 4)


if __name__ == "__main__":
    unittest.main()
