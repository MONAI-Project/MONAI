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
from copy import deepcopy
from numbers import Number

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.apps.auto3dseg import DataAnalyzer
from monai.auto3dseg import (
    Analyzer,
    FgImageStats,
    FgImageStatsSumm,
    FilenameStats,
    ImageStats,
    ImageStatsSumm,
    LabelStats,
    LabelStatsSumm,
    Operations,
    SampleOperations,
    SegSummarizer,
    SummaryOperations,
    datafold_read,
    verify_report_format,
)
from monai.bundle import ConfigParser
from monai.data import DataLoader, Dataset, create_test_image_2d, create_test_image_3d
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    SqueezeDimd,
    ToDeviced,
)
from monai.utils.enums import DataStatsKeys
from tests.util import skip_if_no_cuda

device = "cpu"
n_workers = 2

sim_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
    ],
}

SIM_CPU_TEST_CASES = [
    [{"sim_dim": (32, 32, 32), "label_key": "label"}],
    [{"sim_dim": (32, 32, 32, 2), "label_key": "label"}],
    [{"sim_dim": (32, 32, 32), "label_key": None}],
    [{"sim_dim": (32, 32, 32), "label_key": "None"}],
]

SIM_GPU_TEST_CASES = [[{"sim_dim": (32, 32, 32), "label_key": "label"}], [{"sim_dim": (32, 32, 32), "label_key": None}]]


def create_sim_data(dataroot: str, sim_datalist: dict, sim_dim: tuple, image_only: bool = False, **kwargs) -> None:
    """
    Create simulated data using create_test_image_3d.

    Args:
        dataroot: data directory path that hosts the "nii.gz" image files.
        sim_datalist: a list of data to create.
        sim_dim: the image sizes, for examples: a tuple of (64, 64, 64) for 3d, or (128, 128) for 2d
    """
    if not os.path.isdir(dataroot):
        os.makedirs(dataroot)

    # Generate a fake dataset
    for d in sim_datalist["testing"] + sim_datalist["training"]:
        if len(sim_dim) == 2:  # 2D image
            im, seg = create_test_image_2d(sim_dim[0], sim_dim[1], **kwargs)
        elif len(sim_dim) == 3:  # 3D image
            im, seg = create_test_image_3d(sim_dim[0], sim_dim[1], sim_dim[2], **kwargs)
        elif len(sim_dim) == 4:  # multi-modality 3D image
            im_list = []
            seg_list = []
            for _ in range(sim_dim[3]):
                im_3d, seg_3d = create_test_image_3d(sim_dim[0], sim_dim[1], sim_dim[2], **kwargs)
                im_list.append(im_3d[..., np.newaxis])
                seg_list.append(seg_3d[..., np.newaxis])
            im = np.concatenate(im_list, axis=3)
            seg = np.concatenate(seg_list, axis=3)
        else:
            raise ValueError(f"Invalid argument input. sim_dim has f{len(sim_dim)} values. 2-4 values are expected.")
        nib_image = nib.Nifti1Image(im, affine=np.eye(4))
        image_fpath = os.path.join(dataroot, d["image"])
        nib.save(nib_image, image_fpath)

        if not image_only and "label" in d:
            nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
            label_fpath = os.path.join(dataroot, d["label"])
            nib.save(nib_image, label_fpath)


class TestOperations(Operations):
    """
    Test example for user operation
    """

    __test__ = False  # indicate to pytest that this class is not intended for collection

    def __init__(self) -> None:
        self.data = {"max": np.max, "mean": np.mean, "min": np.min}


class TestAnalyzer(Analyzer):
    """
    Test example for a simple Analyzer
    """

    __test__ = False  # indicate to pytest that this class is not intended for collection

    def __init__(self, key, report_format, stats_name="test"):
        self.key = key
        super().__init__(stats_name, report_format)

    def __call__(self, data):
        d = dict(data)
        report = deepcopy(self.get_report_format())
        report["stats"] = self.ops["stats"].evaluate(d[self.key])
        d[self.stats_name] = report
        return d


class TestImageAnalyzer(Analyzer):
    """
    Test example for a simple Analyzer
    """

    __test__ = False  # indicate to pytest that this class is not intended for collection

    def __init__(self, image_key="image", stats_name="test_image"):
        self.image_key = image_key
        report_format = {"test_stats": None}

        super().__init__(stats_name, report_format)
        self.update_ops("test_stats", TestOperations())

    def __call__(self, data):
        d = dict(data)
        report = deepcopy(self.get_report_format())
        report["test_stats"] = self.ops["test_stats"].evaluate(d[self.image_key])
        d[self.stats_name] = report
        return d


class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        work_dir = self.test_dir.name
        self.dataroot_dir = os.path.join(work_dir, "sim_dataroot")
        self.datalist_file = os.path.join(work_dir, "sim_datalist.json")
        self.datastat_file = os.path.join(work_dir, "datastats.yml")
        ConfigParser.export_config_file(sim_datalist, self.datalist_file)

    @parameterized.expand(SIM_CPU_TEST_CASES)
    def test_data_analyzer_cpu(self, input_params):
        sim_dim = input_params["sim_dim"]
        label_key = input_params["label_key"]
        image_only = not bool(label_key)
        rmax = max(int(sim_dim[0] / 4), 1)
        create_sim_data(
            self.dataroot_dir, sim_datalist, sim_dim, image_only=image_only, rad_max=rmax, rad_min=1, num_seg_classes=1
        )

        analyser = DataAnalyzer(
            self.datalist_file, self.dataroot_dir, output_path=self.datastat_file, label_key=label_key, device=device
        )
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(sim_datalist["training"])

    def test_data_analyzer_histogram(self):
        create_sim_data(
            self.dataroot_dir, sim_datalist, [32] * 3, image_only=True, rad_max=8, rad_min=1, num_seg_classes=1
        )
        analyser = DataAnalyzer(
            self.datalist_file,
            self.dataroot_dir,
            output_path=self.datastat_file,
            label_key=None,
            device=device,
            histogram_only=True,
        )
        datastat = analyser.get_all_case_stats()
        assert len(datastat["stats_by_cases"]) == len(sim_datalist["training"])

    @parameterized.expand(SIM_GPU_TEST_CASES)
    @skip_if_no_cuda
    def test_data_analyzer_gpu(self, input_params):
        sim_dim = input_params["sim_dim"]
        label_key = input_params["label_key"]
        image_only = not bool(label_key)
        rmax = max(int(sim_dim[0] / 4), 1)
        create_sim_data(
            self.dataroot_dir, sim_datalist, sim_dim, image_only=image_only, rad_max=rmax, rad_min=1, num_seg_classes=1
        )
        analyser = DataAnalyzer(
            self.datalist_file, self.dataroot_dir, output_path=self.datastat_file, label_key=label_key, device="cuda"
        )
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(sim_datalist["training"])

    def test_basic_operation_class(self):
        op = TestOperations()
        test_data = np.random.rand(10, 10).astype(np.float64)
        test_ret_1 = op.evaluate(test_data)
        test_ret_2 = op.evaluate(test_data, axis=0)
        assert isinstance(test_ret_1, dict) and isinstance(test_ret_2, dict)
        assert ("max" in test_ret_1) and ("max" in test_ret_2)
        assert ("mean" in test_ret_1) and ("mean" in test_ret_2)
        assert ("min" in test_ret_1) and ("min" in test_ret_2)
        assert isinstance(test_ret_1["max"], np.float64)
        assert isinstance(test_ret_2["max"], np.ndarray)
        assert test_ret_1["max"].ndim == 0
        assert test_ret_2["max"].ndim == 1

    def test_sample_operations(self):
        op = SampleOperations()
        test_data_np = np.random.rand(10, 10).astype(np.float64)
        test_data_mt = MetaTensor(test_data_np, device=device)
        test_ret_np = op.evaluate(test_data_np)
        test_ret_mt = op.evaluate(test_data_mt)
        assert isinstance(test_ret_np["max"], Number)
        assert isinstance(test_ret_np["percentile"], list)
        assert isinstance(test_ret_mt["max"], Number)
        assert isinstance(test_ret_mt["percentile"], list)

        op.update({"sum": np.sum})
        test_ret_np = op.evaluate(test_data_np)
        assert "sum" in test_ret_np

    def test_summary_operations(self):
        op = SummaryOperations()
        test_dict = {"min": [0, 1, 2, 3], "max": [2, 3, 4, 5], "mean": [1, 2, 3, 4], "sum": [2, 4, 6, 8]}
        test_ret = op.evaluate(test_dict)
        assert isinstance(test_ret["max"], Number)
        assert isinstance(test_ret["min"], Number)

        op.update({"sum": np.sum})
        test_ret = op.evaluate(test_dict)
        assert "sum" in test_ret
        assert isinstance(test_ret["sum"], Number)

    def test_basic_analyzer_class(self):
        test_data = {}
        test_data["image_test"] = np.random.rand(10, 10)
        report_format = {"stats": None}
        user_analyzer = TestAnalyzer("image_test", report_format)
        user_analyzer.update_ops("stats", TestOperations())
        result = user_analyzer(test_data)
        assert result["test"]["stats"]["max"] == np.max(test_data["image_test"])
        assert result["test"]["stats"]["min"] == np.min(test_data["image_test"])
        assert result["test"]["stats"]["mean"] == np.mean(test_data["image_test"])

    def test_transform_analyzer_class(self):
        transform = Compose([LoadImaged(keys=["image"]), TestImageAnalyzer(image_key="image")])
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            assert "test_image" in d
            assert "test_stats" in d["test_image"]
            assert "max" in d["test_image"]["test_stats"]
            assert "min" in d["test_image"]["test_stats"]
            assert "mean" in d["test_image"]["test_stats"]

    def test_image_stats_case_analyzer(self):
        analyzer = ImageStats(image_key="image")
        transform = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),  # this creates label to be (1,H,W,D)
                ToDeviced(keys=["image"], device=device, non_blocking=True),
                Orientationd(keys=["image"], axcodes="RAS"),
                EnsureTyped(keys=["image"], data_type="tensor"),
                analyzer,
            ]
        )
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            report_format = analyzer.get_report_format()
            assert verify_report_format(d["image_stats"], report_format)

    def test_foreground_image_stats_cases_analyzer(self):
        analyzer = FgImageStats(image_key="image", label_key="label")
        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys=["label"], func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            analyzer,
        ]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            report_format = analyzer.get_report_format()
            assert verify_report_format(d["image_foreground_stats"], report_format)

    def test_label_stats_case_analyzer(self):
        analyzer = LabelStats(image_key="image", label_key="label")
        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
                ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                EnsureTyped(keys=["image", "label"], data_type="tensor"),
                Lambdad(keys=["label"], func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
                SqueezeDimd(keys=["label"], dim=0),
                analyzer,
            ]
        )
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            report_format = analyzer.get_report_format()
            assert verify_report_format(d["label_stats"], report_format)

    def test_filename_case_analyzer(self):
        analyzer_image = FilenameStats("image", DataStatsKeys.BY_CASE_IMAGE_PATH)
        analyzer_label = FilenameStats("label", DataStatsKeys.BY_CASE_IMAGE_PATH)
        transform_list = [LoadImaged(keys=["image", "label"]), analyzer_image, analyzer_label]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            assert DataStatsKeys.BY_CASE_IMAGE_PATH in d

    def test_filename_case_analyzer_image_only(self):
        analyzer_image = FilenameStats("image", DataStatsKeys.BY_CASE_IMAGE_PATH)
        analyzer_label = FilenameStats(None, DataStatsKeys.BY_CASE_IMAGE_PATH)
        transform_list = [LoadImaged(keys=["image"]), analyzer_image, analyzer_label]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            assert DataStatsKeys.BY_CASE_IMAGE_PATH in d
            assert d[DataStatsKeys.BY_CASE_IMAGE_PATH] == "None"

    def test_image_stats_summary_analyzer(self):
        summary_analyzer = ImageStatsSumm("image_stats")

        transform_list = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image"], device=device, non_blocking=True),
            Orientationd(keys=["image"], axcodes="RAS"),
            EnsureTyped(keys=["image"], data_type="tensor"),
            ImageStats(image_key="image"),
        ]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            stats.append(transform(batch_data[0]))
        summary_report = summary_analyzer(stats)
        report_format = summary_analyzer.get_report_format()
        assert verify_report_format(summary_report, report_format)

    def test_fg_image_stats_summary_analyzer(self):
        summary_analyzer = FgImageStatsSumm("image_foreground_stats")

        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            FgImageStats(image_key="image", label_key="label"),
        ]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            stats.append(transform(batch_data[0]))
        summary_report = summary_analyzer(stats)
        report_format = summary_analyzer.get_report_format()
        assert verify_report_format(summary_report, report_format)

    def test_label_stats_summary_analyzer(self):
        summary_analyzer = LabelStatsSumm("label_stats")

        transform_list = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=["image", "label"], device=device, non_blocking=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            EnsureTyped(keys=["image", "label"], data_type="tensor"),
            Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            LabelStats(image_key="image", label_key="label"),
        ]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            stats.append(transform(batch_data[0]))
        summary_report = summary_analyzer(stats)
        report_format = summary_analyzer.get_report_format()
        assert verify_report_format(summary_report, report_format)

    def test_seg_summarizer(self):
        summarizer = SegSummarizer("image", "label")
        keys = ["image", "label"]
        transform_list = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),  # this creates label to be (1,H,W,D)
            ToDeviced(keys=keys, device=device, non_blocking=True),
            Orientationd(keys=keys, axcodes="RAS"),
            EnsureTyped(keys=keys, data_type="tensor"),
            Lambdad(keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x),
            SqueezeDimd(keys=["label"], dim=0),
            summarizer,
        ]
        transform = Compose(transform_list)
        create_sim_data(self.dataroot_dir, sim_datalist, (32, 32, 32), rad_max=8, rad_min=1, num_seg_classes=1)
        files, _ = datafold_read(sim_datalist, self.dataroot_dir, fold=-1)
        ds = Dataset(data=files)
        self.dataset = DataLoader(ds, batch_size=1, shuffle=False, num_workers=n_workers, collate_fn=no_collation)
        stats = []
        for batch_data in self.dataset:
            d = transform(batch_data[0])
            stats.append(d)
        report = summarizer.summarize(stats)
        assert str(DataStatsKeys.IMAGE_STATS) in report
        assert str(DataStatsKeys.FG_IMAGE_STATS) in report
        assert str(DataStatsKeys.LABEL_STATS) in report

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
