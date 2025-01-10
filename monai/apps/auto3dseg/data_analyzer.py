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

import warnings
from os import path
from typing import Any, cast

import numpy as np
import torch
from torch.multiprocessing import get_context

from monai.apps.auto3dseg.transforms import EnsureSameShaped
from monai.apps.utils import get_logger
from monai.auto3dseg import SegSummarizer
from monai.auto3dseg.utils import datafold_read
from monai.bundle import config_parser
from monai.bundle.config_parser import ConfigParser
from monai.data import DataLoader, Dataset, partition_dataset
from monai.data.utils import no_collation
from monai.transforms import Compose, EnsureTyped, LoadImaged, Orientationd
from monai.utils import ImageMetaKey, StrEnum, min_version, optional_import
from monai.utils.enums import DataStatsKeys, ImageStatsKeys


def strenum_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)


if optional_import("yaml")[1]:
    config_parser.yaml.SafeDumper.add_multi_representer(StrEnum, strenum_representer)

tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")
logger = get_logger(module_name=__name__)

__all__ = ["DataAnalyzer"]


class DataAnalyzer:
    """
    The DataAnalyzer automatically analyzes given medical image dataset and reports the statistics.
    The module expects file paths to the image data and utilizes the LoadImaged transform to read the
    files, which supports nii, nii.gz, png, jpg, bmp, npz, npy, and dcm formats. Currently, only
    segmentation task is supported, so the user needs to provide paths to the image and label files
    (if have). Also, label data format is preferred to be (1,H,W,D), with the label index in the
    first dimension. If it is in onehot format, it will be converted to the preferred format.

    Args:
        datalist: a Python dictionary storing group, fold, and other information of the medical
            image dataset, or a string to the JSON file storing the dictionary.
        dataroot: user's local directory containing the datasets.
        output_path: path to save the analysis result.
        average: whether to average the statistical value across different image modalities.
        do_ccp: apply the connected component algorithm to process the labels/images
        device: a string specifying hardware (CUDA/CPU) utilized for the operations.
        worker: number of workers to use for loading datasets in each GPU/CPU sub-process.
        image_key: a string that user specify for the image. The DataAnalyzer will look it up in the
            datalist to locate the image files of the dataset.
        label_key: a string that user specify for the label. The DataAnalyzer will look it up in the
            datalist to locate the label files of the dataset. If label_key is NoneType or "None",
            the DataAnalyzer will skip looking for labels and all label-related operations.
        hist_bins: bins to compute histogram for each image channel.
        hist_range: ranges to compute histogram for each image channel.
        fmt: format used to save the analysis results. Currently support ``"json"`` and ``"yaml"``, defaults to "yaml".
        histogram_only: whether to only compute histograms. Defaults to False.
        extra_params: other optional arguments. Currently supported arguments are :
            'allowed_shape_difference' (default 5) can be used to change the default tolerance of
            the allowed shape differences between the image and label items. In case of shape mismatch below
            the tolerance, the label image will be resized to match the image using nearest interpolation.


    Examples:
        .. code-block:: python

            from monai.apps.auto3dseg.data_analyzer import DataAnalyzer

            datalist = {
                "testing": [{"image": "image_003.nii.gz"}],
                "training": [
                    {"fold": 0, "image": "image_001.nii.gz", "label": "label_001.nii.gz"},
                    {"fold": 0, "image": "image_002.nii.gz", "label": "label_002.nii.gz"},
                    {"fold": 1, "image": "image_001.nii.gz", "label": "label_001.nii.gz"},
                    {"fold": 1, "image": "image_004.nii.gz", "label": "label_004.nii.gz"},
                ],
            }

            dataroot = '/datasets' # the directory where you have the image files (nii.gz)
            DataAnalyzer(datalist, dataroot)

    Notes:
        The module can also be called from the command line interface (CLI).

    For example:

    .. code-block:: bash

        python -m monai.apps.auto3dseg \\
            DataAnalyzer \\
            get_all_case_stats \\
            --datalist="my_datalist.json" \\
            --dataroot="my_dataroot_dir"

    """

    def __init__(
        self,
        datalist: str | dict,
        dataroot: str = "",
        output_path: str = "./datastats.yaml",
        average: bool = True,
        do_ccp: bool = False,
        device: str | torch.device = "cuda",
        worker: int = 4,
        image_key: str = "image",
        label_key: str | None = "label",
        hist_bins: list | int | None = 0,
        hist_range: list | None = None,
        fmt: str = "yaml",
        histogram_only: bool = False,
        **extra_params: Any,
    ):
        if path.isfile(output_path):
            warnings.warn(f"File {output_path} already exists and will be overwritten.")
            logger.debug(f"{output_path} will be overwritten by a new datastat.")

        self.datalist = datalist
        self.dataroot = dataroot
        self.output_path = output_path
        self.average = average
        self.do_ccp = do_ccp
        self.device = torch.device(device)
        self.worker = worker
        self.image_key = image_key
        self.label_key = None if label_key == "None" else label_key
        self.hist_bins = hist_bins
        self.hist_range: list = [-500, 500] if hist_range is None else hist_range
        self.fmt = fmt
        self.histogram_only = histogram_only
        self.extra_params = extra_params

    @staticmethod
    def _check_data_uniformity(keys: list[str], result: dict) -> bool:
        """
        Check data uniformity since DataAnalyzer provides no support to multi-modal images with different
        affine matrices/spacings due to monai transforms.

        Args:
            keys: a list of string-type keys under image_stats dictionary.

        Returns:
            False if one of the selected key values is not constant across the dataset images.

        """

        if DataStatsKeys.SUMMARY not in result or DataStatsKeys.IMAGE_STATS not in result[DataStatsKeys.SUMMARY]:
            return True
        constant_props = [result[DataStatsKeys.SUMMARY][DataStatsKeys.IMAGE_STATS][key] for key in keys]
        for prop in constant_props:
            if "stdev" in prop and np.any(prop["stdev"]):
                logger.debug(f"summary image_stats {prop} has non-zero stdev {prop['stdev']}.")
                return False

        return True

    def get_all_case_stats(self, key="training", transform_list=None):
        """
        Get all case stats. Caller of the DataAnalyser class. The function initiates multiple GPU or CPU processes of the internal
        _get_all_case_stats functions, which iterates datalist and call SegSummarizer to generate stats for each case.
        After all case stats are generated, SegSummarizer is called to combine results.

        Args:
            key: dataset key
            transform_list: option list of transforms before SegSummarizer

        Returns:
            A data statistics dictionary containing
                "stats_summary" (summary statistics of the entire datasets). Within stats_summary
                there are "image_stats"  (summarizing info of shape, channel, spacing, and etc
                using operations_summary), "image_foreground_stats" (info of the intensity for the
                non-zero labeled voxels), and "label_stats" (info of the labels, pixel percentage,
                image_intensity, and each individual label in a list)
                "stats_by_cases" (List type value. Each element of the list is statistics of
                an image-label info. Within each element, there are: "image" (value is the
                path to an image), "label" (value is the path to the corresponding label), "image_stats"
                (summarizing info of shape, channel, spacing, and etc using operations),
                "image_foreground_stats" (similar to the previous one but one foreground image), and
                "label_stats" (stats of the individual labels )

        Notes:
            Since the backend of the statistics computation are torch/numpy, nan/inf value
            may be generated and carried over in the computation. In such cases, the output
            dictionary will include .nan/.inf in the statistics.

        """
        result: dict[DataStatsKeys, Any] = {DataStatsKeys.SUMMARY: {}, DataStatsKeys.BY_CASE: []}
        result_bycase: dict[DataStatsKeys, Any] = {DataStatsKeys.SUMMARY: {}, DataStatsKeys.BY_CASE: []}
        if self.device.type == "cpu":
            nprocs = 1
            logger.info("Using CPU for data analyzing!")
        else:
            nprocs = torch.cuda.device_count()
            logger.info(f"Found {nprocs} GPUs for data analyzing!")
        if nprocs > 1:
            tmp_ctx: Any = get_context("forkserver")
            with tmp_ctx.Manager() as manager:
                manager_list = manager.list()
                processes = []
                for rank in range(nprocs):
                    p = tmp_ctx.Process(
                        target=self._get_all_case_stats, args=(rank, nprocs, manager_list, key, transform_list)
                    )
                    processes.append(p)
                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
                # merge DataStatsKeys.BY_CASE
                for _ in manager_list:
                    result_bycase[DataStatsKeys.BY_CASE].extend(_[DataStatsKeys.BY_CASE])
        else:
            result_bycase = self._get_all_case_stats(0, 1, None, key, transform_list)

        summarizer = SegSummarizer(
            self.image_key,
            self.label_key,
            average=self.average,
            do_ccp=self.do_ccp,
            hist_bins=self.hist_bins,
            hist_range=self.hist_range,
            histogram_only=self.histogram_only,
        )
        n_cases = len(result_bycase[DataStatsKeys.BY_CASE])
        result[DataStatsKeys.SUMMARY] = summarizer.summarize(cast(list, result_bycase[DataStatsKeys.BY_CASE]))
        result[DataStatsKeys.SUMMARY]["n_cases"] = n_cases
        result_bycase[DataStatsKeys.SUMMARY] = result[DataStatsKeys.SUMMARY]
        if not self._check_data_uniformity([ImageStatsKeys.SPACING], result):
            logger.info("Data spacing is not completely uniform. MONAI transforms may provide unexpected result")
        if self.output_path:
            logger.info(f"Writing data stats to {self.output_path}.")
            ConfigParser.export_config_file(
                result, self.output_path, fmt=self.fmt, default_flow_style=None, sort_keys=False
            )
            by_case_path = self.output_path.replace(f".{self.fmt}", f"_by_case.{self.fmt}")
            if by_case_path == self.output_path:  # self.output_path not ended with self.fmt?
                by_case_path += f".by_case.{self.fmt}"
            logger.info(f"Writing by-case data stats to {by_case_path}, this may take a while.")
            ConfigParser.export_config_file(
                result_bycase, by_case_path, fmt=self.fmt, default_flow_style=None, sort_keys=False
            )
        # release memory
        if self.device.type == "cuda":
            # release unreferenced tensors to mitigate OOM
            # limitation: https://github.com/pytorch/pytorch/issues/12873#issuecomment-482916237
            torch.cuda.empty_cache()
        result[DataStatsKeys.BY_CASE] = result_bycase[DataStatsKeys.BY_CASE]
        return result

    def _get_all_case_stats(
        self,
        rank: int = 0,
        world_size: int = 1,
        manager_list: list | None = None,
        key: str = "training",
        transform_list: list | None = None,
    ) -> Any:
        """
        Get all case stats from a partitioned datalist. The function can only be called internally by get_all_case_stats.
        Args:
            rank: GPU process rank, 0 for CPU process
            world_size: total number of GPUs, 1 for CPU process
            manager_list: multiprocessing manager list object, if using multi-GPU.
            key: dataset key
            transform_list: option list of transforms before SegSummarizer
        """
        summarizer = SegSummarizer(
            self.image_key,
            self.label_key,
            average=self.average,
            do_ccp=self.do_ccp,
            hist_bins=self.hist_bins,
            hist_range=self.hist_range,
            histogram_only=self.histogram_only,
        )
        keys = list(filter(None, [self.image_key, self.label_key]))
        if transform_list is None:
            transform_list = [
                LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
                EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float),
                Orientationd(keys=keys, axcodes="RAS"),
            ]
            if self.label_key is not None:
                allowed_shape_difference = self.extra_params.pop("allowed_shape_difference", 5)
                transform_list.append(
                    EnsureSameShaped(
                        keys=self.label_key,
                        source_key=self.image_key,
                        allowed_shape_difference=allowed_shape_difference,
                    )
                )

        transform = Compose(transform_list)
        files, _ = datafold_read(datalist=self.datalist, basedir=self.dataroot, fold=-1, key=key)
        if world_size <= len(files):
            files = partition_dataset(data=files, num_partitions=world_size)[rank]
        else:
            files = partition_dataset(data=files, num_partitions=len(files))[rank] if rank < len(files) else []
        dataset = Dataset(data=files, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.worker,
            collate_fn=no_collation,
            pin_memory=self.device.type == "cuda",
        )
        result_bycase: dict[DataStatsKeys, Any] = {DataStatsKeys.SUMMARY: {}, DataStatsKeys.BY_CASE: []}
        device = self.device if self.device.type == "cpu" else torch.device("cuda", rank)
        if device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            logger.info(f"device={device} but CUDA device is not available, using CPU instead.")
            device = torch.device("cpu")
        if not has_tqdm:
            warnings.warn("tqdm is not installed. not displaying the caching progress.")

        try:
            for batch_data in tqdm(dataloader) if (has_tqdm and rank == 0) else dataloader:
                batch_data = batch_data[0]
                try:
                    batch_data[self.image_key] = batch_data[self.image_key].to(device)
                    _label_argmax = False
                    if self.label_key is not None:
                        label = batch_data[self.label_key]
                        label = torch.argmax(label, dim=0) if label.shape[0] > 1 else label[0]
                        _label_argmax = True  # track if label is argmaxed
                        batch_data[self.label_key] = label.to(device)
                    d = summarizer(batch_data)
                except BaseException as err:
                    filename = self._get_filename(batch_data)
                    logger.info(f"Unable to process data {filename} on {device}. {err}")
                    if self.device.type == "cuda":
                        logger.info(
                            "DataAnalyzer `device` set to GPU execution hit an exception. Falling back to `cpu`."
                        )
                        try:
                            batch_data[self.image_key] = batch_data[self.image_key].to("cpu")
                            if self.label_key is not None:
                                label = batch_data[self.label_key]
                                if not _label_argmax:
                                    label = torch.argmax(label, dim=0) if label.shape[0] > 1 else label[0]
                                batch_data[self.label_key] = label.to("cpu")
                            d = summarizer(batch_data)
                        except BaseException as err:
                            logger.info(f"Unable to process data {filename} on {device}. {err}")
                            continue
                    else:
                        continue

                stats_by_cases = {
                    DataStatsKeys.BY_CASE_IMAGE_PATH: d[DataStatsKeys.BY_CASE_IMAGE_PATH],
                    DataStatsKeys.BY_CASE_LABEL_PATH: d[DataStatsKeys.BY_CASE_LABEL_PATH],
                }
                if not self.histogram_only:
                    stats_by_cases[DataStatsKeys.IMAGE_STATS] = d[DataStatsKeys.IMAGE_STATS]
                if self.hist_bins != 0:
                    stats_by_cases[DataStatsKeys.IMAGE_HISTOGRAM] = d[DataStatsKeys.IMAGE_HISTOGRAM]

                if self.label_key is not None:
                    stats_by_cases.update(
                        {
                            DataStatsKeys.FG_IMAGE_STATS: d[DataStatsKeys.FG_IMAGE_STATS],
                            DataStatsKeys.LABEL_STATS: d[DataStatsKeys.LABEL_STATS],
                        }
                    )
                result_bycase[DataStatsKeys.BY_CASE].append(stats_by_cases)
        except RuntimeError as e:
            filename = self._get_filename(batch_data)
            raise RuntimeError(f"Interrupted while processing data {filename} on {device}. {e}") from e

        if manager_list is None:
            return result_bycase
        else:
            manager_list.append(result_bycase)

    def _get_filename(self, batch_data):
        if "image_meta_dict" in batch_data.keys():
            return batch_data["image_meta_dict"][ImageMetaKey.FILENAME_OR_OBJ]
        else:
            return batch_data[self.image_key].meta[ImageMetaKey.FILENAME_OR_OBJ]
