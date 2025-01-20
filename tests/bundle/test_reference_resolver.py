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

import unittest

import torch
from parameterized import parameterized

import monai
from monai.bundle.config_item import ComponentLocator, ConfigComponent, ConfigExpression, ConfigItem
from monai.bundle.reference_resolver import ReferenceResolver
from monai.data import DataLoader
from monai.transforms import LoadImaged, RandTorchVisiond
from monai.utils import min_version, optional_import

_, has_tv = optional_import("torchvision", "0.8.0", min_version)

# test instance with no dependencies
TEST_CASE_1 = [
    {
        # all the recursively parsed config items
        "transform#1": {"_target_": "LoadImaged", "keys": ["image"]},
        "transform#1#_target_": "LoadImaged",
        "transform#1#keys": ["image"],
        "transform#1#keys#0": "image",
    },
    "transform#1",
    LoadImaged,
]
# test depends on other component and executable code
TEST_CASE_2 = [
    {
        # some the recursively parsed config items
        "dataloader": {"_target_": "DataLoader", "dataset": "@dataset", "collate_fn": "$monai.data.list_data_collate"},
        "dataset": {"_target_": "Dataset", "data": [1, 2]},
        "dataloader#_target_": "DataLoader",
        "dataloader#dataset": "@dataset",
        "dataloader#collate_fn": "$monai.data.list_data_collate",
        "dataset#_target_": "Dataset",
        "dataset#data": [1, 2],
        "dataset#data#0": 1,
        "dataset#data#1": 2,
    },
    "dataloader",
    DataLoader,
]
# test config has key `name`
TEST_CASE_3 = [
    {
        # all the recursively parsed config items
        "transform::1": {"_target_": "RandTorchVisiond", "keys": "image", "name": "ColorJitter", "brightness": 0.25},
        "transform#1::_target_": "RandTorchVisiond",
        "transform::1::keys": "image",
        "transform::1#name": "ColorJitter",
        "transform::1::brightness": 0.25,
    },
    "transform#1",
    RandTorchVisiond,
]


class TestReferenceResolver(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2] + ([TEST_CASE_3] if has_tv else []))
    def test_resolve(self, configs, expected_id, output_type):
        locator = ComponentLocator()
        resolver = ReferenceResolver()
        # add items to resolver
        for k, v in configs.items():
            k = k.replace("#", "::")
            if ConfigComponent.is_instantiable(v):
                resolver.add_item(ConfigComponent(config=v, id=k, locator=locator))
            elif ConfigExpression.is_expression(v):
                resolver.add_item(ConfigExpression(config=v, id=k, globals={"monai": monai, "torch": torch}))
            else:
                resolver.add_item(ConfigItem(config=v, id=k))

        result = resolver.get_resolved_content(expected_id)  # the root id is `expected_id` here
        self.assertTrue(isinstance(result, output_type))

        # test lazy instantiation
        item = resolver.get_item(expected_id, resolve=True)
        config = item.get_config()
        config["_disabled_"] = False
        item.update_config(config=config)
        if isinstance(item, ConfigComponent):
            result = item.instantiate()
        else:
            result = item.get_config()
        self.assertTrue(isinstance(result, output_type))

    def test_circular_references(self):
        locator = ComponentLocator()
        resolver = ReferenceResolver()
        configs = {"A": "@B", "B": "@C", "C": "@A"}
        for k, v in configs.items():
            resolver.add_item(ConfigComponent(config=v, id=k, locator=locator))
        for k in ["A", "B", "C"]:
            with self.assertRaises(ValueError):
                resolver.get_resolved_content(k)


if __name__ == "__main__":
    unittest.main()
