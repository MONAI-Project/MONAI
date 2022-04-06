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

import warnings
from typing import Optional, Union

import numpy as np
import torch
from monai._extensions.loader import load_module
from .metric import CumulativeIterationMetric

__all__ = ["MorphologicalHausdorffDistanceMetric"]


class MorphologicalHausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Work is based onthe principle of application of  mathemathical morphology more precisely dilatation
    , more details is presented in works [1] and [2]. Calculated  is roughly related to the largest
thickness of the difference between the true and estimatedmasks, and constitutes Approximation of true Hausdorff distance.
 The correlation between true and morphological Hausdorf distance is around 0.9, less for robust version .
Hovewer this implementation is also around 100 times faster than exact MONAI Hausdorff distance calculation.
 One can measure 3 diffrent metrics based on Hausdorff ditance.

    1)simple Hausdorff distance - it tell what is the distance between two most distant points one from y and other from y_pred

    2)robust Hausdorff distance - modification of 1) where we stop analyzing points 
    when we already analyzed given percent of total number of points - is less sensitive to outliers. It is also slightly faster

    3)mean Hausdorff distance - can be understood as approximately 
    mean distance between pair of points where one point is from y and other from y_pred.

    4)additionally one can request as a result one dimensional tensor
     that will contain data for each point what is HD value for it 
     (like we would do Hausdorff distance calculation only for this point and all points from other mask)

    Compute Hausdorff Distance between two tensors. In addition, specify the `percent` (for robust Hausdorff distance )
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.

    Args:
        percent: an optional float number between 0 and 1. If specified, the corresponding
            percent of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to 1.0 .

    1) Érick Oliveira Rodrigues,An efficient and locality-oriented Hausdorff distance algorithm: 
        Proposal and analysis of paradigms and implementations, Pattern Recognition,Volume 117,2021
    2) Karimi, Davood & Salcudean, Septimiu. (2019). 
    Reducing the Hausdorff Distance in Medical Image Segmentation With Convolutional Neural Networks. 
    IEEE Transactions on Medical Imaging. PP. 1-1. 10.1109/TMI.2019.2930068.

    """

    def __init__(
        self,
        percent: float = 1.0,
    ) -> None:
        super().__init__()
        self.percent = percent
        self.compiled_extension = load_module(
            "hausdorff_cpp"
        )


    def compute_morphological_hausdorff_distance(
        self,
        y_pred:  torch.Tensor,
        y: torch.Tensor,
        compare_values: torch.Tensor):
        """
        Compute the Hausdorff distance.

        Args:
            y_pred: input data to compute, It must be 3 dimensional
            y: ground truth to compute mean the distance. It must be 3 dimensional, Dimensionality needs to be identical as in y_pred

            percent: an optional float number between 0 and 1. If specified, the corresponding
                percent of the Hausdorff Distance rather than the maximum result will be achieved.
                Defaults to 1.0 .
            compare_values: 0 dimensional tensor marking what value are we intrested in the supplied y and y_pred,
             defined becouse  in case of multiorgan segmentations frequently we can have information
              about multiple organs in the same mask just marked by diffrent numbers
                                for example in case of boolean array we can suupply it like torch.ones(1, dtype =bool)

        """

        if y.shape != y_pred.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")
        sizz = y.shape

        return self.compiled_extension.getHausdorffDistance(y_pred,y,sizz[0],sizz[1],sizz[2], self.percent,  compare_values )
