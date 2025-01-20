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
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from monai.apps.auto3dseg import BundleGen, DataAnalyzer
from monai.apps.auto3dseg.utils import export_bundle_algo_history, import_bundle_algo_history
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils import set_determinism
from tests.util import (
    SkipIfBeforePyTorchVersion,
    get_testing_algo_template_path,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
)

num_images_perfold = max(torch.cuda.device_count(), 4)
num_images_per_batch = 2

sim_datalist: dict[str, list[dict]] = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
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


def create_sim_data(dataroot, sim_datalist, sim_dim, **kwargs):
    """
    Create simulated data using create_test_image_3d.

    Args:
        dataroot: data directory path that hosts the "nii.gz" image files.
        sim_datalist: a list of data to create.
        sim_dim: the image sizes, e.g. a tuple of (64, 64, 64).
    """
    if not os.path.isdir(dataroot):
        os.makedirs(dataroot)

    # Generate a fake dataset
    for d in sim_datalist["testing"] + sim_datalist["training"]:
        im, seg = create_test_image_3d(sim_dim[0], sim_dim[1], sim_dim[2], **kwargs)
        nib_image = nib.Nifti1Image(im, affine=np.eye(4))
        image_fpath = os.path.join(dataroot, d["image"])
        nib.save(nib_image, image_fpath)

        if "label" in d:
            nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
            label_fpath = os.path.join(dataroot, d["label"])
            nib.save(nib_image, label_fpath)


def run_auto3dseg_before_bundlegen(test_path, work_dir):
    """
    Run the Auto3DSeg modules before the BundleGen step.
    Args:
        test_path: a path to contain `sim_dataroot` which is for the simulated dataset file.
        work_dir: working directory

    Returns:
        Paths of the outputs from the previous steps
    """

    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    # write to a json file
    dataroot_dir = os.path.join(test_path, "sim_dataroot")
    datalist_file = os.path.join(work_dir, "sim_datalist.json")
    ConfigParser.export_config_file(sim_datalist, datalist_file)
    create_sim_data(dataroot_dir, sim_datalist, (24, 24, 24), rad_max=10, rad_min=1, num_seg_classes=1)

    datastats_file = os.path.join(work_dir, "datastats.yaml")
    analyser = DataAnalyzer(datalist_file, dataroot_dir, output_path=os.path.join(work_dir, "datastats.yaml"))
    analyser.get_all_case_stats()

    return dataroot_dir, datalist_file, datastats_file


@skip_if_no_cuda
@SkipIfBeforePyTorchVersion((1, 11, 1))
@skip_if_quick
class TestBundleGen(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(0)
        self.test_dir = tempfile.TemporaryDirectory()

    def test_move_bundle_gen_folder(self) -> None:
        test_path = self.test_dir.name
        work_dir = os.path.join(test_path, "workdir")
        dataroot_dir, datalist_file, datastats_file = run_auto3dseg_before_bundlegen(test_path, work_dir)
        data_src = {
            "name": "fake_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": datalist_file,
            "dataroot": dataroot_dir,
            "multigpu": False,
            "class_names": ["label_class"],
        }
        data_src_cfg = os.path.join(work_dir, "data_src_cfg.yaml")
        ConfigParser.export_config_file(data_src, data_src_cfg)

        sys_path = sys.path.copy()
        with skip_if_downloading_fails():
            bundle_generator = BundleGen(
                algo_path=work_dir,
                data_stats_filename=datastats_file,
                data_src_cfg_name=data_src_cfg,
                templates_path_or_url=get_testing_algo_template_path(),
            )

        bundle_generator.generate(work_dir, num_fold=1)
        history_before = bundle_generator.get_history()
        export_bundle_algo_history(history_before)

        sys.path = sys_path  # prevent the import_bundle_algo_history from using the path "work_dir/algorithm_templates"
        tempfile.TemporaryDirectory()
        work_dir_new = os.path.join(test_path, "workdir_2")
        shutil.move(work_dir, work_dir_new)
        history_after = import_bundle_algo_history(work_dir_new, only_trained=False)
        self.assertEqual(len(history_before), len(history_after))

    def tearDown(self) -> None:
        set_determinism(None)
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
