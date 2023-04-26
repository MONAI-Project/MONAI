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

from monai.config import KeysCollection
from monai.data.meta_tensor import MetaTensor
from monai.transforms import MapTransform
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.lazy.functional import apply_pending

__all__ = ["ApplyPendingd"]


class ApplyPendingd(InvertibleTransform, MapTransform):
    """
    ApplyPendingd can be inserted into a pipeline that is being executed lazily in order
    to ensure resampling happens before the next transform. It doesn't do anything itself,
    but its presence causes the pipeline to be executed as it doesn't implement ``LazyTrait``

    See ``Compose`` for a detailed explanation of the lazy resampling feature.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        if not isinstance(data, dict):
            raise ValueError("'data' must be of type dict but is '{type(data)}'")

        return data

    def inverse(self, data):
        if not isinstance(data, dict):
            raise ValueError("'data' must be of type dict but is '{type(data)}'")

        return data


ApplyPendingD = ApplyPendingDict = ApplyPendingd
