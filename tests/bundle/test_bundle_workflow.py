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
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.bundle import ConfigWorkflow, create_workflow
from monai.data import Dataset
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, LoadImaged, SaveImaged
from tests.nonconfig_workflow import NonConfigWorkflow, PythonicWorkflowImpl

MODULE_PATH = Path(__file__).resolve().parents[1]

TEST_CASE_1 = [os.path.join(MODULE_PATH, "testing_data", "inference.json")]

TEST_CASE_2 = [os.path.join(MODULE_PATH, "testing_data", "inference.yaml")]

TEST_CASE_3 = [os.path.join(MODULE_PATH, "testing_data", "config_fl_train.json")]

TEST_CASE_4 = [os.path.join(MODULE_PATH, "testing_data", "responsive_inference.json")]

TEST_CASE_NON_CONFIG_WRONG_LOG = [None, "logging.conf", "Cannot find the logging config file: logging.conf."]


class TestBundleWorkflow(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.expected_shape = (128, 128, 128)
        test_image = np.random.rand(*self.expected_shape)
        self.filename = os.path.join(self.data_dir, "image.nii")
        self.filename1 = os.path.join(self.data_dir, "image1.nii")
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), self.filename)
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), self.filename1)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def _test_inferer(self, inferer):
        # should initialize before parsing any bundle content
        inferer.initialize()
        # test required and optional properties
        self.assertListEqual(inferer.check_properties(), [])
        # test read / write the properties, note that we don't assume it as JSON or YAML config here
        self.assertEqual(inferer.bundle_root, "will override")
        self.assertEqual(inferer.device, torch.device("cpu"))
        net = inferer.network_def
        self.assertTrue(isinstance(net, UNet))
        sliding_window = inferer.inferer
        self.assertTrue(isinstance(sliding_window, SlidingWindowInferer))
        preprocessing = inferer.preprocessing
        self.assertTrue(isinstance(preprocessing, Compose))
        postprocessing = inferer.postprocessing
        self.assertTrue(isinstance(postprocessing, Compose))
        # test optional properties get
        self.assertTrue(inferer.key_metric is None)
        inferer.bundle_root = "/workspace/data/spleen_ct_segmentation"
        inferer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inferer.network_def = deepcopy(net)
        inferer.inferer = deepcopy(sliding_window)
        inferer.preprocessing = deepcopy(preprocessing)
        inferer.postprocessing = deepcopy(postprocessing)
        # test optional properties set
        inferer.key_metric = "set optional properties"

        # should initialize and parse again as changed the bundle content
        inferer.initialize()
        inferer.run()
        inferer.finalize()
        # verify inference output
        loader = LoadImage(image_only=True)
        pred_file = os.path.join(self.data_dir, "image", "image_seg.nii.gz")
        self.assertTupleEqual(loader(pred_file).shape, self.expected_shape)
        os.remove(pred_file)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_inference_config(self, config_file):
        override = {
            "network": "$@network_def.to(@device)",
            "dataset#_target_": "Dataset",
            "dataset#data": [{"image": self.filename}],
            "postprocessing#transforms#2#output_postfix": "seg",
            "output_dir": self.data_dir,
        }
        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow_type="infer",
            config_file=config_file,
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)

        # test property path
        inferer = ConfigWorkflow(
            config_file=config_file,
            workflow_type="infer",
            properties_path=os.path.join(MODULE_PATH, "testing_data", "fl_infer_properties.json"),
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)
        self.assertEqual(inferer.workflow_type, "infer")

    @parameterized.expand([TEST_CASE_4])
    def test_responsive_inference_config(self, config_file):
        input_loader = LoadImaged(keys="image")
        output_saver = SaveImaged(keys="pred", output_dir=self.data_dir, output_postfix="seg")

        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow_type="infer",
            config_file=config_file,
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
        )
        # FIXME: temp add the property for test, we should add it to some formal realtime infer properties
        inferer.add_property(name="dataflow", required=True, config_id="dataflow")

        inferer.initialize()
        inferer.dataflow.update(input_loader({"image": self.filename}))
        inferer.run()
        output_saver(inferer.dataflow)
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "image", "image_seg.nii.gz")))

        # bundle is instantiated and idle, just change the input for next inference
        inferer.dataflow.clear()
        inferer.dataflow.update(input_loader({"image": self.filename1}))
        inferer.run()
        output_saver(inferer.dataflow)
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "image1", "image1_seg.nii.gz")))

        inferer.finalize()

    @parameterized.expand([TEST_CASE_3])
    def test_train_config(self, config_file):
        # test standard MONAI model-zoo config workflow
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=config_file,
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
            init_id="initialize",
            run_id="run",
            final_id="finalize",
        )
        # should initialize before parsing any bundle content
        trainer.initialize()
        # test required and optional properties
        self.assertListEqual(trainer.check_properties(), [])
        # test override optional properties
        trainer.parser.update(
            pairs={"validate#evaluator#postprocessing": "$@validate#postprocessing if @val_interval > 0 else None"}
        )
        trainer.initialize()
        self.assertListEqual(trainer.check_properties(), [])
        # test read / write the properties
        dataset = trainer.train_dataset
        self.assertIsInstance(dataset, Dataset)
        inferer = trainer.train_inferer
        self.assertIsInstance(inferer, SimpleInferer)
        # test optional properties get
        self.assertIsNone(trainer.train_key_metric)
        trainer.train_dataset = deepcopy(dataset)
        trainer.train_inferer = deepcopy(inferer)
        # test optional properties set
        trainer.train_key_metric = "set optional properties"

        # should initialize and parse again as changed the bundle content
        trainer.initialize()
        trainer.run()
        trainer.finalize()

    def test_non_config(self):
        # test user defined python style workflow
        inferer = NonConfigWorkflow(self.filename, self.data_dir)
        self.assertEqual(inferer.meta_file, None)
        self._test_inferer(inferer)

    @parameterized.expand([TEST_CASE_NON_CONFIG_WRONG_LOG])
    def test_non_config_wrong_log_cases(self, meta_file, logging_file, expected_error):
        with self.assertRaisesRegex(FileNotFoundError, expected_error):
            NonConfigWorkflow(self.filename, self.data_dir, meta_file, logging_file)

    def test_pythonic_workflow(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = {"roi_size": (64, 64, 32)}
        meta_file = os.path.join(MODULE_PATH, "testing_data", "metadata.json")
        property_path = os.path.join(MODULE_PATH, "testing_data", "python_workflow_properties.json")
        workflow = PythonicWorkflowImpl(
            workflow_type="infer", config_file=config_file, meta_file=meta_file, properties_path=property_path
        )
        workflow.initialize()
        # Load input data
        input_loader = LoadImaged(keys="image")
        workflow.dataflow.update(input_loader({"image": self.filename}))
        self.assertEqual(workflow.bundle_root, ".")
        self.assertEqual(workflow.device, device)
        self.assertEqual(workflow.version, "0.1.0")
        # check config override correctly
        self.assertEqual(workflow.inferer.roi_size, (64, 64, 32))
        workflow.run()
        # update input data and run again
        workflow.dataflow.update(input_loader({"image": self.filename1}))
        workflow.run()
        pred = workflow.dataflow["pred"]
        self.assertEqual(pred.shape[2:], self.expected_shape)
        self.assertEqual(pred.meta["filename_or_obj"], self.filename1)
        workflow.finalize()

    def test_create_pythonic_workflow(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = {"roi_size": (64, 64, 32)}
        meta_file = os.path.join(MODULE_PATH, "testing_data", "metadata.json")
        property_path = os.path.join(MODULE_PATH, "testing_data", "python_workflow_properties.json")
        sys.path.append(MODULE_PATH)
        workflow = create_workflow(
            "tests.nonconfig_workflow.PythonicWorkflowImpl",
            workflow_type="infer",
            config_file=config_file,
            meta_file=meta_file,
            properties_path=property_path,
        )
        # Load input data
        input_loader = LoadImaged(keys="image")
        workflow.dataflow.update(input_loader({"image": self.filename}))
        self.assertEqual(workflow.bundle_root, ".")
        self.assertEqual(workflow.device, device)
        self.assertEqual(workflow.version, "0.1.0")
        # check config override correctly
        self.assertEqual(workflow.inferer.roi_size, (64, 64, 32))

        # check set property override correctly
        workflow.inferer = SlidingWindowInferer(roi_size=config_file["roi_size"], sw_batch_size=1, overlap=0.5)
        workflow.initialize()
        self.assertEqual(workflow.inferer.overlap, 0.5)

        workflow.run()
        # update input data and run again
        workflow.dataflow.update(input_loader({"image": self.filename1}))
        workflow.run()
        pred = workflow.dataflow["pred"]
        self.assertEqual(pred.shape[2:], self.expected_shape)
        self.assertEqual(pred.meta["filename_or_obj"], self.filename1)

        # test add properties
        workflow.add_property(name="net", required=True, desc="network for the training.")
        self.assertIn("net", workflow.properties)
        workflow.finalize()


if __name__ == "__main__":
    unittest.main()
