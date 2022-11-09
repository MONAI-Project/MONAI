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

from monai.config import KeysCollection
from monai.utils import ensure_tuple



def from_engine_hovernet(keys: KeysCollection, nested_keys: str, first: bool = False):
    """
    Utility function to simplify the `batch_transform` or `output_transform` args of ignite components
    when handling dictionary or list of dictionaries(for example: `engine.state.batch` or `engine.state.output`).
    Users only need to set the expected keys, then it will return a callable function to extract data from
    dictionary and construct a tuple respectively.

    If data is a list of dictionaries after decollating, extract expected keys and construct lists respectively,
    for example, if data is `[{"A": {"C": 1, "D": 2}, "B": {"C": 2, "D": 2}}, {"A":  {"C": 3, "D": 2}}, "B":  {"C": 4, "D": 2}}}]`, from_engine(["A", "B"], "C"): `([1, 3], [2, 4])`.

    It can help avoid a complicated `lambda` function and make the arg of metrics more straight-forward.
    For example, set the first key as the prediction and the second key as label to get the expected data
    from `engine.state.output` for a metric::

        from monai.handlers import MeanDice, from_engine

        metric = MeanDice(
            include_background=False,
            output_transform=from_engine(keys=["pred", "label"], nested_keys=HoVerNetBranch.NP.value)
        )

    Args:
        keys: specified keys to extract data from dictionary or decollated list of dictionaries.
        nested_keys: specified keys to extract nested data from dictionary or decollated list of dictionaries.
        first: whether only extract specified keys from the first item if input data is a list of dictionaries,
            it's used to extract the scalar data which doesn't have batch dim and was replicated into every
            dictionary when decollating, like `loss`, etc.


    """
    keys = ensure_tuple(keys)

    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(data[k][nested_keys] for k in keys)
        if isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [data[0][k][nested_keys] if first else [i[k][nested_keys] for i in data] for k in keys]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper