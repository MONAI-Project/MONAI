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

from typing import Hashable, Mapping, Optional, Sequence, Union

import torch

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.nets import HoVerNet
from monai.transforms.transform import MapTransform
from monai.utils import optional_import

from .array import PostProcessHoVerNetOutput

find_contours, _ = optional_import("skimage.measure", name="find_contours")
moments, _ = optional_import("skimage.measure", name="moments")

__all__ = ["PostProcessHoVerNetOutputDict", "PostProcessHoVerNetOutputD", "PostProcessHoVerNetOutputd"]


class PostProcessHoVerNetOutputd(MapTransform):
    """
    Dictionary-based transform that post processing image tiles. It assumes network has three branches, with a segmentation branch that 
    returns `np_pred`, a hover map branch that returns `hv_pred` and an optional classification branch that returns `nc_pred`. After this 
    tranform, it will return pixel-wise nuclear instance segmentation prediction and a instance-level information dictionary.

    Args:
        hv_pred_key: hover map branch output key. Defaults to `HoVerNet.Branch.HV.value`.
        nv_pred_key: classification branch output key. Defaults to `HoVerNet.Branch.NC.value`.
        inst_info_dict_key: a dict contaning a instance-level information dictionary will be added, which including bounding_box,
            centroid and contour. If output_classes is not None, the dictionary will also contain pixel-wise nuclear type prediction.
            Defaults to "inst_info".
        output_classes: number of types considered at output of NC branch.
        return_centroids: whether to generate coords for each nucleus instance.
            Defaults to True.
        threshold_pred: threshold the float values of prediction to int 0 or 1 with specified theashold. Defaults to 0.5.
        threshold_overall: threshold the float values of overall gradient map to int 0 or 1 with specified theashold.
            Defaults to 0.4.
        min_size: objects smaller than this size are removed. Defaults to 10.
        sigma: std. could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        kernel_size: the size of the Sobel kernel. Defaults to 17.
        radius: the radius of the disk-shaped footprint. Defaults to 2.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = PostProcessHoVerNetOutput.backend

    def __init__(
        self,
        keys: KeysCollection = HoVerNet.Branch.NP.value,
        hv_pred_key: str = HoVerNet.Branch.HV.value,
        nv_pred_key: str = HoVerNet.Branch.NC.value,
        inst_info_dict_key: str = "inst_info",
        output_classes: Optional[int] = None,
        return_centroids: bool = True,
        threshold_pred: float = 0.5,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 17,
        radius: int = 2,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.NP_pred_key = keys
        self.hv_pred_key = hv_pred_key
        self.nv_pred_key = nv_pred_key
        self.inst_info_dict_key = inst_info_dict_key
        self.output_classes = output_classes
        self.return_centroids = return_centroids

        self.converter = PostProcessHoVerNetOutput(
            output_classes=output_classes,
            return_centroids=return_centroids,
            threshold_pred=threshold_pred,
            threshold_overall=threshold_overall,
            min_size=min_size,
            sigma=sigma,
            kernel_size=kernel_size,
            radius=radius,
        )

    def __call__(self, pred: Mapping[Hashable, NdarrayOrTensor]):
        """
        Args:
            pred: a dict combined output of classification(NC, optional), segmentation(NP) and hover map(HV) branches.
        
        Returns:
            pixel-wise nuclear instance segmentation prediction and a instance-level information dictionary stored in
            `inst_info_dict_key`.
        """
        d = dict(pred)
        for key in self.key_iterator(d):
            np_pred = d[key]
            hv_pred = d[self.hv_pred_key]
            if self.output_classes is not None:
                NC_pred = d[self.nv_pred_key]
            else:
                NC_pred = None

            d[key], inst_info_dict = self.converter(np_pred, hv_pred, NC_pred)
            d[self.inst_info_dict_key] = inst_info_dict

        return d


PostProcessHoVerNetOutputDict = PostProcessHoVerNetOutputD = PostProcessHoVerNetOutputd
