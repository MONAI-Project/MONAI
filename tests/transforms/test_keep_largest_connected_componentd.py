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

from monai.transforms import KeepLargestConnectedComponentd
from tests.util import TEST_NDARRAYS, assert_allclose

grid_1 = [[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]
grid_2 = [[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 2], [1, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]
grid_3 = [
    [
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ],
    [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0],
    ],
]
grid_4 = [
    [
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
]
grid_5 = [[[0, 0, 1, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]]]

VALID_CASES = []
for p in TEST_NDARRAYS:
    VALID_CASES.append(
        [
            "value_1",
            {"keys": ["img"], "independent": False, "applied_labels": 1, "is_onehot": False},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    VALID_CASES.append(
        [
            "value_2",
            {"keys": ["img"], "independent": False, "applied_labels": [2], "is_onehot": False},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    VALID_CASES.append(
        [
            "independent_value_1_2",
            {"keys": ["img"], "independent": True, "applied_labels": [1, 2], "is_onehot": False},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    VALID_CASES.append(
        [
            "dependent_value_1_2",
            {"keys": ["img"], "independent": False, "applied_labels": [1, 2], "is_onehot": False},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    VALID_CASES.append(
        [
            "value_1",
            {"keys": ["img"], "independent": True, "applied_labels": [1], "is_onehot": False},
            {"img": p(grid_2)},
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
        ]
    )

    VALID_CASES.append(
        [
            "independent_value_1_2",
            {"keys": ["img"], "independent": True, "applied_labels": [1, 2], "is_onehot": False},
            {"img": p(grid_2)},
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
        ]
    )

    VALID_CASES.append(
        [
            "dependent_value_1_2",
            {"keys": ["img"], "independent": False, "applied_labels": [1, 2], "is_onehot": False},
            {"img": p(grid_2)},
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]),
        ]
    )

    VALID_CASES.append(
        [
            "value_1_connect_1",
            {"keys": ["img"], "independent": False, "applied_labels": [1], "connectivity": 1, "is_onehot": False},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    VALID_CASES.append(
        [
            "independent_value_1_2_connect_1",
            {"keys": ["img"], "independent": True, "applied_labels": [1, 2], "connectivity": 1, "is_onehot": False},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    VALID_CASES.append(
        [
            "onehot_none_dependent_value_1_2_connect_1",
            {"keys": ["img"], "independent": False, "applied_labels": [1, 2], "connectivity": 1},
            {"img": p(grid_1)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    VALID_CASES.append(
        [
            "onehot_independent_batch_2_apply_label_1_connect_1",
            {"keys": ["img"], "independent": True, "applied_labels": [1], "connectivity": 1, "is_onehot": True},
            {"img": p(grid_3)},
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                ]
            ),
        ]
    )

    VALID_CASES.append(
        [
            "onehot_independent_batch_2_apply_label_1_connect_2",
            {"keys": ["img"], "independent": True, "applied_labels": [1], "connectivity": 2, "is_onehot": True},
            {"img": p(grid_3)},
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 1.0],
                    ],
                ]
            ),
        ]
    )

    VALID_CASES.append(
        [
            "onehot_independent_batch_2_apply_label_1_2_connect_2",
            {"keys": ["img"], "independent": True, "applied_labels": [1, 2], "connectivity": 2, "is_onehot": True},
            {"img": p(grid_3)},
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ]
    )

    VALID_CASES.append(
        [
            "onehot_dependent_batch_2_apply_label_1_2_connect_2",
            {"keys": ["img"], "independent": False, "applied_labels": [1, 2], "connectivity": 2, "is_onehot": True},
            {"img": p(grid_4)},
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ]
    )

    VALID_CASES.append(
        [
            "onehot_none_dependent_batch_2_apply_label_1_2_connect_1",
            {"keys": ["img"], "independent": False, "applied_labels": [1, 2], "connectivity": 1},
            {"img": p(grid_4)},
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        ]
    )

    VALID_CASES.append(
        [
            "single_channel_onehot",
            {"keys": ["img"], "independent": False, "applied_labels": 0, "connectivity": 1, "is_onehot": True},
            {"img": p(grid_5)},
            torch.tensor([[[0, 0, 1, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]]),
        ]
    )


class TestKeepLargestConnectedComponentd(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, args, input_dict, expected):
        converter = KeepLargestConnectedComponentd(**args)
        result = converter(input_dict)
        assert_allclose(result["img"], expected, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
