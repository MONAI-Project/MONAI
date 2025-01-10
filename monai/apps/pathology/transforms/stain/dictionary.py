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
A collection of dictionary-based wrappers around the pathology transforms
defined in :py:class:`monai.apps.pathology.transforms.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

import numbers
from collections.abc import Hashable, Mapping

import numpy as np

from monai.config import KeysCollection
from monai.transforms.transform import MapTransform

from .array import ExtractHEStains, HEDJitter, NormalizeHEStains


class HEDJitterd(MapTransform):

    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.HEDJitter`.
    Class to randomly perturbe HED color space values of image using stain deconvolution.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        theta: jitter range factor

        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(self, keys: KeysCollection, theta: float = 0.0, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.hedjitter = HEDJitter(theta=theta)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.hedjitter(d[key])
        return d


class ExtractHEStainsd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.ExtractHEStains`.
    Class to extract a target stain from an image, using stain deconvolution.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile)
            and pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to (1.9705, 1.0308).
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        max_cref: tuple | np.ndarray = (1.9705, 1.0308),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.extractor = ExtractHEStains(tli=tli, alpha=alpha, beta=beta, max_cref=max_cref)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.extractor(d[key])
        return d


class NormalizeHEStainsd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.NormalizeHEStains`.

    Class to normalize patches/images to a reference or target image stain.

    Performs stain deconvolution of the source image using the ExtractHEStains
    class, to obtain the stain matrix and calculate the stain concentration matrix
    for the image. Then, performs the inverse Beer-Lambert transform to recreate the
    patch using the target H&E stain matrix provided. If no target stain provided, a default
    reference stain is used. Similarly, if no maximum stain concentrations are provided, a
    reference maximum stain concentrations matrix is used.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile) and
            pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15.
        target_he: target stain matrix. Defaults to None.
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to None.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        target_he: tuple | np.ndarray = ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)),
        max_cref: tuple | np.ndarray = (1.9705, 1.0308),
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeHEStains(tli=tli, alpha=alpha, beta=beta, target_he=target_he, max_cref=max_cref)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d


ExtractHEStainsDict = ExtractHEStainsD = ExtractHEStainsd
NormalizeHEStainsDict = NormalizeHEStainsD = NormalizeHEStainsd
HEDJitterDict = HEDJitterD = HEDJitterd
