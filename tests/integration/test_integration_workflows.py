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
import shutil
import tempfile
import unittest
import warnings
from glob import glob

import nibabel as nib
import numpy as np
import torch
from ignite.engine import Events
from ignite.metrics import Accuracy

import monai
from monai.data import create_test_image_3d
from monai.engines import IterationEvents, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    SaveImage,
    SaveImaged,
    ScaleIntensityd,
)
from monai.utils import optional_import, set_determinism
from tests.testing_data.integration_answers import test_integration_value
from tests.util import DistTestCase, TimedCall, assert_allclose, pytorch_after, skip_if_quick

SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")

TASK = "integration_workflows"


def run_training_test(root_dir, device="cuda:0", amp=False, num_workers=4):
    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"image": img, "label": seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{"image": img, "label": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys=["image", "label"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers)
    # create a validation data loader
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    # create UNet, DiceLoss and Adam optimizer
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    summary_writer = SummaryWriter(log_dir=root_dir)

    val_postprocessing = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )

    class _TestEvalIterEvents:
        def attach(self, engine):
            engine.add_event_handler(IterationEvents.FORWARD_COMPLETED, self._forward_completed)

        def _forward_completed(self, engine):
            pass

    val_handlers = [
        StatsHandler(iteration_log=False),
        TensorBoardStatsHandler(summary_writer=summary_writer, iteration_log=False),
        TensorBoardImageHandler(
            log_dir=root_dir, batch_transform=from_engine(["image", "label"]), output_transform=from_engine("pred")
        ),
        CheckpointSaver(save_dir=root_dir, save_dict={"net": net}, save_key_metric=True),
        _TestEvalIterEvents(),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=val_postprocessing,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        metric_cmp_fn=lambda cur, prev: cur >= prev,  # if greater or equal, treat as new best metric
        val_handlers=val_handlers,
        amp=bool(amp),
        to_kwargs={"memory_format": torch.preserve_format},
        amp_kwargs={"dtype": torch.float16 if bool(amp) else torch.float32} if pytorch_after(1, 10, 0) else {},
    )

    train_postprocessing = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )

    class _TestTrainIterEvents:
        def attach(self, engine):
            engine.add_event_handler(IterationEvents.FORWARD_COMPLETED, self._forward_completed)
            engine.add_event_handler(IterationEvents.LOSS_COMPLETED, self._loss_completed)
            engine.add_event_handler(IterationEvents.BACKWARD_COMPLETED, self._backward_completed)
            engine.add_event_handler(IterationEvents.MODEL_COMPLETED, self._model_completed)

        def _forward_completed(self, engine):
            pass

        def _loss_completed(self, engine):
            pass

        def _backward_completed(self, engine):
            pass

        def _model_completed(self, engine):
            pass

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine("loss", first=True)),
        TensorBoardStatsHandler(
            summary_writer=summary_writer, tag_name="train_loss", output_transform=from_engine("loss", first=True)
        ),
        CheckpointSaver(save_dir=root_dir, save_dict={"net": net, "opt": opt}, save_interval=2, epoch_level=True),
        _TestTrainIterEvents(),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=train_postprocessing,
        key_train_metric={"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        train_handlers=train_handlers,
        amp=bool(amp),
        optim_set_to_none=True,
        to_kwargs={"memory_format": torch.preserve_format},
        amp_kwargs={"dtype": torch.float16 if bool(amp) else torch.float32} if pytorch_after(1, 10, 0) else {},
    )
    trainer.run()

    # test train and validation stats
    train_stats = trainer.get_stats("output")
    assert_allclose(train_stats["output"][0]["loss"], trainer.state.output[0]["loss"])
    val_stats = evaluator.get_stats("metrics")

    return val_stats["best_validation_metric"]


def run_inference_test(root_dir, model_file, device="cuda:0", amp=False, num_workers=4):
    images = sorted(glob(os.path.join(root_dir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    val_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys=["image", "label"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    # create UNet, DiceLoss and Adam optimizer
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    val_postprocessing = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
            # test the case that `pred` in `engine.state.output`, while `image_meta_dict` in `engine.state.batch`
            SaveImaged(keys="pred", output_dir=root_dir, output_postfix="seg_transform"),
        ]
    )
    val_handlers = [
        StatsHandler(iteration_log=False),
        CheckpointLoader(load_path=f"{model_file}", load_dict={"net": net}),
    ]

    saver = SaveImage(output_dir=root_dir, output_postfix="seg_handler")

    def save_func(engine):
        for o in from_engine("pred")(engine.state.output):
            saver(o)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=val_postprocessing,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
        val_handlers=val_handlers,
        amp=bool(amp),
    )
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, save_func)
    evaluator.run()

    return evaluator.state.best_metric


@skip_if_quick
class IntegrationWorkflows(DistTestCase):
    def setUp(self):
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"seg{i:d}.nii.gz"))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        monai.config.print_config()

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    def train_and_infer(self, idx=0):
        results = []
        set_determinism(seed=0)
        best_metric = run_training_test(self.data_dir, device=self.device, amp=(idx == 2))
        model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pt")))[-1]
        infer_metric = run_inference_test(self.data_dir, model_file, device=self.device, amp=(idx == 2))

        print("best metric", best_metric)
        print("infer metric", infer_metric)
        if idx == 2:
            self.assertTrue(test_integration_value(TASK, key="best_metric_2", data=best_metric, rtol=1e-2))
        else:
            self.assertTrue(test_integration_value(TASK, key="best_metric", data=best_metric, rtol=1e-2))
        # check inference properties
        if idx == 2:
            self.assertTrue(test_integration_value(TASK, key="infer_metric_2", data=infer_metric, rtol=1e-2))
        else:
            self.assertTrue(test_integration_value(TASK, key="infer_metric", data=infer_metric, rtol=1e-2))
        results.append(best_metric)
        results.append(infer_metric)

        def _test_saved_files(postfix):
            output_files = sorted(glob(os.path.join(self.data_dir, "img*", f"*{postfix}.nii.gz")))
            values = []
            for output in output_files:
                ave = np.mean(nib.load(output).get_fdata())
                values.append(ave)
            if idx == 2:
                self.assertTrue(test_integration_value(TASK, key="output_sums_2", data=values, rtol=1e-2))
            else:
                self.assertTrue(test_integration_value(TASK, key="output_sums", data=values, rtol=1e-2))

        _test_saved_files(postfix="seg_handler")
        _test_saved_files(postfix="seg_transform")
        try:
            os.remove(model_file)
        except Exception as e:
            warnings.warn(f"Fail to remove {model_file}: {e}.")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return results

    def test_training(self):
        repeated = []
        test_rounds = 3
        for i in range(test_rounds):
            results = self.train_and_infer(idx=i)
            repeated.append(results)
        np.testing.assert_allclose(repeated[0], repeated[1])

    @TimedCall(seconds=300, skip_timing=not torch.cuda.is_available(), daemon=False)
    def test_timing(self):
        self.train_and_infer(idx=2)


if __name__ == "__main__":
    unittest.main()
