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
from collections.abc import Hashable, Mapping

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils.misc import ImageMetaKey


class EnsureSameShaped(MapTransform):
    """
    Checks if segmentation label images (in keys) have the same spatial shape as the main image (in source_key),
    and raise an error if the shapes are significantly different.
    If the shapes are only slightly different (within an allowed_shape_difference in each dim), then resize the label using
    nearest interpolation. This transform is designed to correct datasets with slight label shape mismatches.
    Generally image and segmentation label must have the same spatial shape, however some public datasets are having slight
    shape mismatches, which will cause potential crashes when calculating loss or metric functions.
    """

    def __init__(
        self,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
        source_key: str = "image",
        allowed_shape_difference: int = 5,
        warn: bool = True,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be compared to the source_key item shape.
            allow_missing_keys: do not raise exception if key is missing.
            source_key: key of the item with the reference shape.
            allowed_shape_difference: raises error if shapes are different more than this value in any dimension,
                otherwise corrects for the shape mismatch using nearest interpolation.
            warn: if `True` prints a warning if the label image is resized


        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.source_key = source_key
        self.allowed_shape_difference = allowed_shape_difference
        self.warn = warn

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        image_shape = d[self.source_key].shape[1:]
        for key in self.key_iterator(d):
            label_shape = d[key].shape[1:]
            if label_shape != image_shape:
                filename = ""
                if hasattr(d[key], "meta") and isinstance(d[key].meta, Mapping):  # type: ignore[attr-defined]
                    filename = d[key].meta.get(ImageMetaKey.FILENAME_OR_OBJ)  # type: ignore[attr-defined]

                if np.allclose(list(label_shape), list(image_shape), atol=self.allowed_shape_difference):
                    if self.warn:
                        warnings.warn(
                            f"The {key} with shape {label_shape} was resized to match the source shape {image_shape}"
                            f", the metadata was not updated {filename}."
                        )
                    d[key] = torch.nn.functional.interpolate(
                        input=d[key].unsqueeze(0), size=image_shape, mode="nearest-exact"
                    ).squeeze(0)
                else:
                    raise ValueError(
                        f"The {key} shape {label_shape} is different from the source shape {image_shape} {filename}."
                    )
        return d
