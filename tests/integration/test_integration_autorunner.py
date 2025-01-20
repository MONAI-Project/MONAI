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

import nibabel as nib
import numpy as np
import torch

from monai.apps.auto3dseg import AutoRunner
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils import optional_import
from tests.util import (
    SkipIfBeforePyTorchVersion,
    get_testing_algo_template_path,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
)

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")
_, has_nni = optional_import("nni")

num_images_perfold = max(torch.cuda.device_count(), 4)
num_images_per_batch = 2

sim_datalist: dict[str, list[dict]] = {
    "testing": [{"image": f"ts_image__{idx:03d}.nii.gz"} for idx in range(num_images_perfold)],
    "training": [
        {
            "fold": f,
            "image": f"tr_image_{(f * num_images_perfold + idx):03d}.nii.gz",
            "label": f"tr_label_{(f * num_images_perfold + idx):03d}.nii.gz",
        }
        for f in range(num_images_per_batch + 1)
        for idx in range(num_images_perfold)
    ],
}

train_param = (
    {
        "num_images_per_batch": num_images_per_batch,
        "num_epochs": 2,
        "num_epochs_per_validation": 1,
        "num_warmup_epochs": 1,
        "use_pretrain": False,
        "pretrained_path": "",
        "num_steps_per_image": 1,
    }
    if torch.cuda.is_available()
    else {}
)

pred_param = {"files_slices": slice(0, 1), "mode": "mean", "sigmoid": True}


@skip_if_quick
@SkipIfBeforePyTorchVersion((1, 11, 1))  # for mem_get_info
@unittest.skipIf(not has_tb, "no tensorboard summary writer")
class TestAutoRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        test_path = self.test_dir.name

        sim_dataroot = os.path.join(test_path, "dataroot")
        if not os.path.isdir(sim_dataroot):
            os.makedirs(sim_dataroot)

        # Generate a fake dataset
        for d in sim_datalist["testing"] + sim_datalist["training"]:
            im, seg = create_test_image_3d(24, 24, 24, rad_max=10, num_seg_classes=1)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = os.path.join(sim_dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = os.path.join(sim_dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        sim_json_datalist = os.path.join(sim_dataroot, "sim_input.json")
        ConfigParser.export_config_file(sim_datalist, sim_json_datalist)

        data_src_cfg = os.path.join(test_path, "data_src_cfg.yaml")
        data_src = {
            "name": "sim_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": sim_json_datalist,
            "dataroot": sim_dataroot,
            "multigpu": False,
            "class_names": ["label_class"],
        }

        ConfigParser.export_config_file(data_src, data_src_cfg)
        self.data_src_cfg = data_src_cfg
        self.test_path = test_path

    @skip_if_no_cuda
    def test_autorunner(self) -> None:
        work_dir = os.path.join(self.test_path, "work_dir")
        runner = AutoRunner(
            work_dir=work_dir,
            input=self.data_src_cfg,
            templates_path_or_url=get_testing_algo_template_path(),
            allow_skip=False,
        )
        runner.set_training_params(train_param)  # 2 epochs
        runner.set_num_fold(1)
        with skip_if_downloading_fails():
            runner.run()

    @skip_if_no_cuda
    def test_autorunner_ensemble(self) -> None:
        work_dir = os.path.join(self.test_path, "work_dir")
        runner = AutoRunner(
            work_dir=work_dir,
            input=self.data_src_cfg,
            templates_path_or_url=get_testing_algo_template_path(),
            allow_skip=False,
        )
        runner.set_training_params(train_param)  # 2 epochs
        runner.set_ensemble_method("AlgoEnsembleBestByFold")
        runner.set_num_fold(1)
        with skip_if_downloading_fails():
            runner.run()

    @skip_if_no_cuda
    def test_autorunner_gpu_customization(self) -> None:
        work_dir = os.path.join(self.test_path, "work_dir")
        runner = AutoRunner(
            work_dir=work_dir,
            input=self.data_src_cfg,
            templates_path_or_url=get_testing_algo_template_path(),
            allow_skip=False,
        )
        gpu_customization_specs = {
            "universal": {"num_trials": 1, "range_num_images_per_batch": [1, 2], "range_num_sw_batch_size": [1, 2]}
        }
        runner.set_gpu_customization(gpu_customization=True, gpu_customization_specs=gpu_customization_specs)
        runner.set_training_params(train_param)  # 2 epochs
        runner.set_num_fold(1)
        with skip_if_downloading_fails():
            runner.run()

    @skip_if_no_cuda
    @unittest.skipIf(not has_nni, "nni required")
    def test_autorunner_hpo(self) -> None:
        work_dir = os.path.join(self.test_path, "work_dir")
        runner = AutoRunner(
            work_dir=work_dir,
            input=self.data_src_cfg,
            hpo=True,
            ensemble=False,
            templates_path_or_url=get_testing_algo_template_path(),
            allow_skip=False,
        )
        hpo_param = {
            "num_epochs_per_validation": train_param["num_epochs_per_validation"],
            "num_images_per_batch": train_param["num_images_per_batch"],
            "num_epochs": train_param["num_epochs"],
            "num_warmup_epochs": train_param["num_warmup_epochs"],
            "use_pretrain": train_param["use_pretrain"],
            "pretrained_path": train_param["pretrained_path"],
            # below are to shorten the time for dints
            "training#num_epochs_per_validation": train_param["num_epochs_per_validation"],
            "training#num_images_per_batch": train_param["num_images_per_batch"],
            "training#num_epochs": train_param["num_epochs"],
            "training#num_warmup_epochs": train_param["num_warmup_epochs"],
            "searching#num_epochs_per_validation": train_param["num_epochs_per_validation"],
            "searching#num_images_per_batch": train_param["num_images_per_batch"],
            "searching#num_epochs": train_param["num_epochs"],
            "searching#num_warmup_epochs": train_param["num_warmup_epochs"],
            "nni_dry_run": True,
        }
        search_space = {"learning_rate": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}}
        runner.set_num_fold(1)
        runner.set_nni_search_space(search_space)
        runner.set_hpo_params(params=hpo_param)
        with skip_if_downloading_fails():
            runner.run()

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
