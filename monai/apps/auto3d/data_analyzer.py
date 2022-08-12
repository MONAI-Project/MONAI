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

"""
Step 1 of the AutoML pipeline. The dataset is analysized with this script.
"""

import copy
import time
import warnings
from functools import partial
from typing import Any, Dict, List, Union

import numpy as np
import torch

from monai import data, transforms
from monai.apps.auto3d.data_utils import datafold_read, recursive_getkey, recursive_getvalue, recursive_setvalue
from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import no_collation
from monai.transforms.utils_pytorch_numpy_unification import max, mean, median, min, percentile, std
from monai.utils import min_version, optional_import
from monai.utils.misc import label_union

yaml, _ = optional_import("yaml")
tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")
measure, _ = optional_import("scipy.ndimage.measurements")

logger = get_logger(module_name=__name__)

__all__ = ["DataAnalyzer"]


class DataAnalyzer:
    """
    The DataAnalyzer automatically analyzes given medical image dataset and reports the statistics.
    The module expects file paths to the image data and utilizes the LoadImaged transform to read the files.
    which supports nii, nii.gz, png, jpg, bmp, npz, npy, and dcm formats. Currently, only segmentation
    problem is supported, so the user needs to provide paths to the image and label files. Also, label
    data format is preferred to be (1,H,W,D), with the label index in the first dimension. If it is in
    onehot format, it will be converted to the preferred format.

    Args:
        datalist: a Python dictionary storing group, fold, and other information of the medical
            image dataset, or a string to the JSON file storing the dictionary.
        dataroot: user's local directory containing the datasets.
        output_path: path to save the analysis result.
        average: whether to average the statistical value across different image modalities.
        do_ccp: apply the connected component algorithm to process the labels/images
        device: a string specifying hardware (CUDA/CPU) utilized for the operations.
        worker: number of workers to use for parallel processing.

    For example:

    .. code-block:: python

        from monai.apps.auto3d.data_analyzer import DataAnalyzer

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

    Note:
        The module can also be called from the command line interface (CLI) if MONAI is
        installed with the source code.

    For example:

    .. code-block:: bash

        python monai/apps/auto3d \
            DataAnalyzer \
            get_all_case_stats \
            --datalist="my_datalist.json" \
            --dataroot="my_dataroot_dir"

    """

    def __init__(
        self,
        datalist: Union[str, Dict],
        dataroot: str,
        output_path: str = "./data_stats.yaml",
        average: bool = False,
        do_ccp: bool = True,
        device: Union[str, torch.device] = "cuda",
        worker: int = 2,
    ):
        """
        The initializer will load the data and register the functions for data statistics gathering.
        """
        self.output_path = output_path
        files, _ = datafold_read(datalist=datalist, basedir=dataroot, fold=-1)
        ds = data.Dataset(
            data=files,
            transform=transforms.Compose(
                [
                    transforms.LoadImaged(keys=["image", "label"]),
                    transforms.EnsureChannelFirstd(keys=["image", "label"]),  # this creates label to be (1,H,W,D)
                    transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                    transforms.EnsureTyped(keys=["image", "label"], data_type="tensor"),
                    transforms.Lambdad(
                        keys="label", func=lambda x: torch.argmax(x, dim=0, keepdim=True) if x.shape[0] > 1 else x
                    ),
                    transforms.SqueezeDimd(keys=["label"], dim=0),  # make label (H,W,D)
                ]
            ),
        )
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=worker, collate_fn=no_collation)
        # Whether to average all the modalities in the summary
        # If SUMMARY_AVERAGE is set to false,
        # the stats for each modality are calculated separately
        self.SUMMARY_AVERAGE = average
        self.DO_CONNECTED_COMP = do_ccp
        self.device = device
        self.data: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        # gather all summary function for combining case stats
        self.gather_summary: Dict[str, Dict] = {}
        self._register_functions()
        self._register_operations()

    def _register_functions(self):
        """
        Register all the statistic functions for calculating stats for individual cases and overall. If one installs
        monai in the editable source code manner, then a new statistics calculation can be customized in the DataAnalyzer
        class.

        For example:

        .. code-block:: python

            class DataAnalyzer:
                # other functions defined in the class
                # below are for new functions

                def my_fun(x_list: List[torch.tensor]):
                    # notice the data may be a list of data from different modalities.
                    # So my_fun(x) should return a list of stats.
                    y_list = []
                    for x in x_list:
                        y_list.append(x.sum())
                    return y_list

                def _get_my_stats(self):  # in the DataAnalyzer class. case_stats will be written to the yaml file
                    processed_data = self.data['image']
                    case_stats = {'my_stats': my_fun(processed_data)}.
                    return case_stats

                def _get_my_stats_sumumary(self):
                    case_stats_summary = {'stats1': summary_function}. The summary_function will process

                    case_stats_summary = {
                        "my_stats_summary": {"sum": self._get_my_stats}
                    }
                    self.gather_summary.update(case_stats_summary)

                def _register_functions(self):
                    # add
                    self.functions = ...
                    self.function_summary = ...
                    self.function_summary.append(self._get_my_stats_sumumary)

        """
        self.functions = [self._get_case_image_stats, self._get_case_foreground_image_stats, self._get_label_stats]
        self.functions_summary = [
            self._get_case_image_stats_summary,
            self._get_case_foreground_image_stats_summary,
            self._get_label_stats_summary,
        ]

    def _register_operations(self):
        """
        Register data operations (max/mean/median/...) for the stats gathering processes.
        """
        # define basic operations for stats
        self.operations = {
            "max": max,
            "mean": mean,
            "median": median,
            "min": min,
            "percentile": partial(percentile, q=np.array([0.5, 10, 90, 99.5])),
            "stdev": std,
        }
        # allow mapping the output of self.operations to new keys (save computation time)
        # the output from torch.quantile is mapped to four keys.
        self.operations_mappingkey = {
            "percentile": ["percentile_00_5", "percentile_10_0", "percentile_90_0", "percentile_99_5"]
        }
        # define summary functions for output from self.operations. For example,
        # how to combine max intensity from each case (image)
        # operation summary definition must accept dim input like torch.mean
        self.operations_summary = {
            "max": max,
            "mean": mean,
            "median": mean,
            "min": min,
            "percentile_00_5": mean,
            "percentile_99_5": mean,
            "percentile_10_0": mean,
            "percentile_90_0": mean,
            "stdev": mean,
        }

    def _get_case_image_stats(self) -> Dict:
        """
        Generate image statistics for cases in datalist ({'image','label'})
        Statistics values are stored under the key "image_stats"

        Returns:
            a dictionary of the images stats
            - image_stats
                - shape
                - channel
                - cropped_shape
                - spacing
                - intensity
        """
        # retrieve transformed data from self.data
        start = time.time()
        ndas = self.data["image"]
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        if "nda_croppeds" not in self.data:
            self.data["nda_croppeds"] = [self._get_foreground_image(_) for _ in ndas]
        nda_croppeds = self.data["nda_croppeds"]
        # perform calculation
        case_stats = {
            "image_stats": {
                "shape": [list(_.shape) for _ in ndas],
                "channels": len(ndas),
                "cropped_shape": [list(_.shape) for _ in nda_croppeds],
                "spacing": np.tile(np.diag(self.data["image_meta_dict"]["affine"])[:3], [len(ndas), 1]).tolist(),
                "intensity": [self._stats_opt(_) for _ in nda_croppeds],
            }
        }
        logger.debug(f"Get image stats spent {time.time()-start}")
        return case_stats

    def _get_case_image_stats_summary(self):
        """
        Update self.gather_summary by case-by-case.
        """
        # this dictionary describes how to gather values in the summary
        case_stats_summary = {
            "image_stats": {
                "shape": partial(self._stats_opt_summary, average=self.SUMMARY_AVERAGE),
                "channels": self._stats_opt_summary,
                "cropped_shape": partial(self._stats_opt_summary, average=self.SUMMARY_AVERAGE),
                "spacing": partial(self._stats_opt_summary, average=self.SUMMARY_AVERAGE),
                "intensity": partial(self._intensity_summary, average=self.SUMMARY_AVERAGE),
            }
        }
        self.gather_summary.update(case_stats_summary)

    def _get_case_foreground_image_stats(self) -> Dict:
        """
        Generate intensity statistics based on foreground images for cases in the datalist
        ({'image','label'}). The foreground is defined by points where labels are positive numbers.
        The statistics will be values with the key name "intensity" under parent the key
        "image_foreground_stats".

        Returns
            a dictionary with following structure
            - image_foreground_stats
                - intensity
                    - max
                    - mean
                    - median
                    - ...
        """
        # retrieve transformed data from self.data
        start = time.time()
        ndas = self.data["image"]
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_l = self.data["label"]
        if "nda_foreground" not in self.data:
            self.data["nda_foreground"] = [self._get_foreground_label(_, ndas_l) for _ in ndas]
        nda_foreground = self.data["nda_foreground"]

        case_stats = {"image_foreground_stats": {"intensity": [self._stats_opt(_) for _ in nda_foreground]}}
        logger.debug(f"Get foreground image data stats spent {time.time() - start}")
        return case_stats

    def _get_case_foreground_image_stats_summary(self):
        """
        Update gather_summary from foreground cases one by one.
        """
        case_stats_summary = {
            "image_foreground_stats": {"intensity": partial(self._intensity_summary, average=self.SUMMARY_AVERAGE)}
        }
        self.gather_summary.update(case_stats_summary)

    def _get_label_stats(self) -> Dict:
        """
        Generate label statisics for all the cases in the datalist based on ({"images", "labels"}).
        Each label has its own statistics (the connected components info, shape, and
        corresponding image region intensity). The statistics are saved in the values with key name
        "label_stats" in the return variable.

        Returns
            a dictionary with following structures:
            - label_stats
                - labels: class_IDs of the label + background class
                - pixel_percentanges
                - image_intensity
                - label_N (N=0,1,...)
                    - image_intensity
                    - shape
                    - ncomponents
        """
        # retrieve transformed data from self.data
        start = time.time()
        ndas = self.data["image"]
        ndas = [ndas[i] for i in range(ndas.shape[0])]
        ndas_l = self.data["label"]
        unique_label = torch.unique(ndas_l).data.cpu().numpy().astype(np.int8).tolist()
        case_stats = {
            "label_stats": {
                "labels": unique_label,
                "pixel_percentage": None,
                "image_intensity": [self._stats_opt(_[ndas_l > 0]) for _ in ndas],
            }
        }
        start = time.time()
        pixel_percentage = {}
        for index in unique_label:
            label_dict: Dict[str, Any] = {}
            mask_index = ndas_l == index
            s = time.time()
            label_dict["image_intensity"] = [self._stats_opt(_[mask_index]) for _ in ndas]
            logger.debug(f" label {index} stats takes {time.time() - s}")
            pixel_percentage[index] = torch.sum(mask_index).data.cpu().numpy()
            if self.DO_CONNECTED_COMP:
                label_dict["shape"] = []
                label_dict["ncomponents"] = None
                # find all connected components and their bounding shape
                structure = np.ones(np.ones(len(ndas_l.shape), dtype=np.int32) * 3, dtype=np.int32)
                labeled, ncomponents = measure.label(mask_index.data.cpu().numpy(), structure)
                label_dict.update({"ncomponents": ncomponents})
                for ncomp in range(1, ncomponents + 1):
                    comp_idx = np.argwhere(labeled == ncomp)
                    comp_idx_min = np.min(comp_idx, axis=0).tolist()
                    comp_idx_max = np.max(comp_idx, axis=0).tolist()
                    bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
                    label_dict["shape"].append(bbox_shape)
            case_stats["label_stats"].update({f"label_{index}": label_dict})
        # update pixel_percentage
        total_percent = np.sum(list(pixel_percentage.values()))
        for key, value in pixel_percentage.items():
            pixel_percentage[key] = float(value / total_percent)
        case_stats["label_stats"].update({"pixel_percentage": pixel_percentage})
        logger.debug(f"Get label stats spent {time.time()-start}")
        return case_stats

    def _get_label_stats_summary(self):
        """
        Get the label statistics for each unique label and update them into gather_summary.
        """
        case_stats_summary = {
            "label_stats": {
                "labels": label_union,
                "pixel_percentage": self._pixelpercent_summary,
                "image_intensity": partial(self._intensity_summary, average=self.SUMMARY_AVERAGE),
            }
        }
        key_chain = ["label_stats", "labels"]
        opt = label_union
        unique_label = opt(
            list(filter(None, [recursive_getvalue(case, key_chain) for case in self.results["stats_by_cases"]]))
        )
        for index in unique_label:
            label_dict_summary = {"image_intensity": partial(self._intensity_summary, average=self.SUMMARY_AVERAGE)}
            if self.DO_CONNECTED_COMP:
                label_dict_summary["shape"] = partial(self._stats_opt_summary, is_label=True)
                label_dict_summary["ncomponents"] = partial(self._stats_opt_summary, is_label=True)
            case_stats_summary["label_stats"].update({f"label_{index}": label_dict_summary})
        self.gather_summary.update(case_stats_summary)

    @staticmethod
    def _pixelpercent_summary(x):
        """
        Define the summary function for the pixel percentage over the whole dataset.

        Args
            x: list of dictionaries dict = {'label1': percent, 'label2': percent}. The dict may miss some labels.

        Returns
            a dictionary showing the percentage of labels, with numeric keys (0, 1, ...)
        """
        percent_summary = {}
        for _ in x:
            for key, value in _.items():
                percent_summary[key] = percent_summary.get(key, 0) + value
        total_percent = np.sum(list(percent_summary.values()))
        for key, value in percent_summary.items():
            percent_summary[key] = float(value / total_percent)
        return percent_summary

    def _intensity_summary(self, x: List, average: bool = False) -> Dict:
        """
        Define the summary function for stats over the whole dataset
        Combine overall intensity statistics for all cases in datalist. The intensity features are
        min, max, mean, std, percentile defined in self._stats_opt().
        Values may be averaged over all the cases if the `average` is set to be True

        Args:
            x: list of the list of intensity stats [[{max:, min:, },{max:, min:, }]]
            average: if average is true, operation will be applied along axis 0 and average out the values

        Returns
            a dictionary of the intensity stats. Keys include 'max', 'mean', and others defined in self.operations

        """
        result = {}
        for key in x[0][0].keys():  # .keys() not required, len(x) = N data
            value = []
            for case in x:
                value.append([_[key] for _ in case])
            dim = (0,) if not average else None
            value = self.operations_summary[key](value, dim=dim)
            result[key] = np.array(value).tolist()
        return result

    def _stats_opt_summary(self, datastat_list, average=False, is_label=False):
        """
        Combine other stats calculation methods (like shape/min/max/std). Does not guarantee
        correct output for custimized stats structure. Check the following input structures.

        Args:
            data: [case_stats, case_stats, ...].
                For images,
                    case_stats are list [stats_modality1, stats_modality2, ...],
                    stats_modality1 can be single value, or it can be a 1d list.
                For labels,
                    case_stats are list [stat1, stat2, ...]. stat1 can be 1d list, 2d list, and single value.
            average: the operation is performed after mixing all modalities.
            is_label: If the data is from label stats.

        Returns
            a dictonary with following property of data in keys like "max", "mean" and others defined in operations_summary
        """
        axis = None
        if type(datastat_list[0]) is list or type(datastat_list[0]) is np.array:
            if not is_label:
                # size = [num of cases, number of modalities, stats]
                datastat_list = np.concatenate([[np.array(_) for _ in datastat_list]])
            else:
                # size = [num of cases, stats]
                datastat_list = np.concatenate([np.array(_) for _ in datastat_list])
            axis = (0,)
            if average and len(datastat_list.shape) > 2:
                axis = (0, 1)
        # Calculate statistics from the data using numpy. Only used for summary
        result = {}
        for name, ops in self.operations.items():
            # get results
            _result = ops(np.array(datastat_list), dim=axis).tolist()  # post process with key mapping
            mappingkeys = self.operations_mappingkey.get(name)
            if mappingkeys is not None:
                result.update({mappingkeys[i]: _result[i] for i in range(len(_result))})
            else:
                result[name] = _result
        return result

    def _stats_opt(self, raw_data):
        """
        Calculate statistics calculation operations (ops) on the images/labels

        Args:
            raw_data: ndarray image or label

        Returns:
            a dictionary to list out the statistics based on give operations (ops). For example, keys can include 'max', 'min',
            'median', 'percentile_00_5', percentile_90_0', 'stdev'.

        """
        result = {}
        for name, ops in self.operations.items():
            if len(raw_data) == 0:
                raw_data = torch.tensor([0.0], device=self.device)
            if not torch.is_tensor(raw_data):
                raw_data = torch.from_numpy(raw_data).to(self.device)
            #  compute the results
            _result = ops(raw_data).data.cpu().numpy().tolist()
            # post process the data
            mappingkeys = self.operations_mappingkey.get(name)
            if mappingkeys is not None:
                result.update({mappingkeys[i]: _result[i] for i in range(len(_result))})
            else:
                result[name] = _result
        return result

    @staticmethod
    def _get_foreground_image(image: MetaTensor) -> MetaTensor:
        """
        Get a foreground image by removing all-zero rectangles on the edges of the image
        Note for the developer: update select_fn if the foreground is defined differently.

        Args:
            image: ndarray image to segment.
        Returns:
            ndarray of foreground image by removing all-zero edges. Note: the size of the ouput is smaller than the input.
        """
        crop_foreground = transforms.CropForeground(select_fn=lambda x: x > 0)
        image_foreground = MetaTensor(crop_foreground(image))
        return image_foreground

    @staticmethod
    def _get_foreground_label(image: MetaTensor, label: MetaTensor) -> MetaTensor:
        """
        Get foreground image pixel values and mask out the non-labeled area.

        Args
            image: ndarray image to segment.
            label: ndarray the image input and annotated with class IDs.

        Return
            1D array of foreground image with label > 0
        """
        label_foreground = MetaTensor(image[label > 0])
        return label_foreground

    def get_case_stats(self, batch_data):
        """
        Get stats for each case {'image', 'label'} in the datalist. The data case is stored in self.data
        Args:
            batch_data: monai dataloader batch data
                images: image with shape [modality, image_shape]
                label: label with shape [image_shape]

        Returns:
            a dictionary to summarize all the statistics for each case in following structure
            - image_stats
                - shape, channels,cropped_shape, spacing, intensity
            - image_foreground_stats
                - intensity
            - label_stats
                - labels, pxiel_percentage, image_intensity, label_0, label_1

        """
        self.data["image"] = batch_data["image"].to(self.device)
        self.data["label"] = batch_data["label"].to(self.device)
        self.data["image_meta_dict"] = batch_data["image_meta_dict"]
        self.data["label_meta_dict"] = batch_data["label_meta_dict"]
        case_stats = {}
        for func in self.functions:
            case_stats.update(func())
        return case_stats

    def _get_case_summary(self):
        """
        Function to combine case stats. The stats for each case is stored in self.results['stats_by_cases'].
        Each case stats is a dictionary. The function first get all the leaf-keys of self.gather_summary.
        self.gather_summary is a dictionary of the same structure with the final summary yaml
        output (self.results['stats_summary']), because it is updated by case_stats_summary.
        The operations is retrived by recursive_getvalue and the combined value is calculated.

        summarize the results from each case using functions _intensity_summary, _stats_opt_summary.
        """
        # initialize gather_summary
        [func() for func in self.functions_summary]
        # read
        key_chains = recursive_getkey(self.gather_summary)
        self.results["stats_summary"] = copy.deepcopy(self.gather_summary)
        for key_chain in key_chains:
            opt = recursive_getvalue(self.gather_summary, key_chain)
            value = opt(
                list(filter(None, [recursive_getvalue(case, key_chain) for case in self.results["stats_by_cases"]]))
            )
            recursive_setvalue(key_chain, value, self.results["stats_summary"])

    def get_all_case_stats(self) -> Dict:
        """
        Get all case stats. Caller of the DataAnalyser class. The function iterates datalist and
        call get_case_stats to generate stats. Then get_case_summary is called to combine results.

        Returns
            - the data statistics dictionary
        """
        start = time.time()
        self.results["stats_summary"] = {}
        self.results["stats_by_cases"] = []
        s = start
        if not has_tqdm:
            warnings.warn("tqdm is not installed. not displaying the caching progress.")

        for batch_data in tqdm(self.dataset) if has_tqdm else self.dataset:
            images_file = batch_data[0]["image_meta_dict"]["filename_or_obj"]
            label_file = batch_data[0]["label_meta_dict"]["filename_or_obj"]
            logger.debug(f"Load data spent {time.time() - s}")
            case_stat = {"image": images_file, "label": label_file}
            case_stat.update(self.get_case_stats(batch_data[0]))
            self.results["stats_by_cases"].append(case_stat)
            logger.debug(f"Process data spent {time.time() - s}")
            s = time.time()
        self._get_case_summary()
        ConfigParser.export_config_file(self.results, self.output_path, fmt="yaml", default_flow_style=None)
        logger.debug(f"total time {time.time() - start}")
        return self.results
