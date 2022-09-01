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

import os
import tempfile
import unittest
import nibabel as nib
import numpy as np

from monai.apps.auto3dseg import DataAnalyzer, BundleGen, ScriptEnsembleBuilder, EnsembleBestN
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d



fake_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_011.fake.nii.gz", "label": "tr_label_011.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_012.fake.nii.gz", "label": "tr_label_012.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_013.fake.nii.gz", "label": "tr_label_013.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_014.fake.nii.gz", "label": "tr_label_014.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_015.fake.nii.gz", "label": "tr_label_015.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_016.fake.nii.gz", "label": "tr_label_016.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_017.fake.nii.gz", "label": "tr_label_017.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_018.fake.nii.gz", "label": "tr_label_018.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_019.fake.nii.gz", "label": "tr_label_019.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_020.fake.nii.gz", "label": "tr_label_020.fake.nii.gz"},
    ],
}


class TestEnsembleBuilder(unittest.TestCase):
    def setUp(self) -> None:

        self.test_dir = tempfile.TemporaryDirectory()
        test_path = self.test_dir.name

        self.dataroot = os.path.join(test_path, 'dataroot')
        self.work_dir = os.path.join(test_path, "workdir")
        self.da_output_yaml = os.path.join(self.work_dir, "datastats.yaml")
        self.data_src_cfg = os.path.join(self.work_dir, "data_src_cfg.yaml")

        if not os.path.isdir(self.dataroot):
            os.makedirs(self.dataroot)

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)

        # Generate a fake dataset
        for d in fake_datalist["testing"] + fake_datalist["training"]:
            im, seg = create_test_image_3d(39, 47, 46, rad_max=10, num_seg_classes = 1)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = os.path.join(self.dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = os.path.join(self.dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        # write to a json file
        self.fake_json_datalist = os.path.join(self.dataroot, "fake_input.json")
        ConfigParser.export_config_file(fake_datalist, self.fake_json_datalist)

        progress_tracker = ["analysis", "configure"]

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)


        if "analysis" in progress_tracker:
            da = DataAnalyzer(
                self.fake_json_datalist,
                self.dataroot,
                output_path=self.da_output_yaml)

            datastat = da.get_all_case_stats()

        data_src = {
            "name": "fake_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": self.fake_json_datalist,
            "dataroot": self.dataroot,
            "multigpu": False,
            "class_names": ["label_class"]
        }


        ConfigParser.export_config_file(data_src, self.data_src_cfg)

        bundle_generator = BundleGen(
            data_stats_filename=self.da_output_yaml,
            data_lists_filename=self.data_src_cfg)

        bundle_generator.generate(self.work_dir, [0,1,2,3,4])

        # todo: training is not encapsulated yet
        history = bundle_generator.get_history()

        num_epoch = 2
        n_iter = int(num_epoch * len(fake_datalist["training"]) * 4 / 5)
        n_iter_val = int(n_iter / 2)

        self.algo_paths = []
        self.best_metrics = []
        for i, record in enumerate(history):
            self.assertEqual(len(record.keys()), 1, "each record should have one model")
            for name, algo in record.items():
                algo.train(
                    num_iterations=n_iter,
                    num_iterations_per_validation=n_iter_val,
                    single_gpu=not data_src["multigpu"]
                )
                self.algo_paths.append(algo.output_path)
                self.best_metrics.append(algo.get_score())

    def test_data(self) -> None:

        builder = ScriptEnsembleBuilder(self.algo_paths, self.best_metrics, self.data_src_cfg)
        builder.set_ensemble_method(EnsembleBestN(n_best=3))
        ensemble = builder.get_ensemble()

        result = ensemble.predict()

    def tearDown(self) -> None:
        pass

if __name__ == "__main__":
    unittest.main()
