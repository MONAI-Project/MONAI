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
from copy import deepcopy

import torch
import torch.nn.functional as F
from parameterized import parameterized

from monai.transforms import KeepLargestConnectedComponent
from monai.transforms.utils_pytorch_numpy_unification import moveaxis
from monai.utils.type_conversion import convert_to_dst_type
from tests.util import TEST_NDARRAYS, assert_allclose


def to_onehot(x):
    out = moveaxis(F.one_hot(torch.as_tensor(x).long())[0], -1, 0)
    out, *_ = convert_to_dst_type(out, x)
    return out


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

grid_6 = [[[0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 1]]]

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            "value_1",
            {"independent": False, "applied_labels": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    TESTS.append(
        [
            "value_2",
            {"independent": False, "applied_labels": [2], "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "independent_value_1_2",
            {"independent": True, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "dependent_value_1_2",
            {"independent": False, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 1, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    TESTS.append(
        [
            "value_1",
            {"independent": True, "applied_labels": [1], "is_onehot": False},
            p(grid_2),
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "independent_value_1_2",
            {"independent": True, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_2),
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "dependent_value_1_2",
            {"independent": False, "applied_labels": [1, 2], "is_onehot": False},
            p(grid_2),
            torch.tensor([[[0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 2], [0, 0, 1, 2, 2], [0, 0, 0, 0, 1]]]),
        ]
    )

    TESTS.append(
        [
            "value_1_connect_1",
            {"independent": False, "applied_labels": [1], "connectivity": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 2]]]),
        ]
    )

    TESTS.append(
        [
            "independent_value_1_2_connect_1",
            {"independent": True, "applied_labels": [1, 2], "connectivity": 1, "is_onehot": False},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "onehot_none_dependent_value_1_2_connect_1",
            {"independent": False, "applied_labels": [1, 2], "connectivity": 1},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [1, 2, 1, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            "onehot_independent_batch_2_apply_label_1_connect_1",
            {"independent": True, "applied_labels": [1], "connectivity": 1, "is_onehot": True},
            p(grid_3),
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

    TESTS.append(
        [
            "onehot_independent_batch_2_apply_label_1_connect_2",
            {"independent": True, "applied_labels": [1], "connectivity": 2, "is_onehot": True},
            p(grid_3),
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

    TESTS.append(
        [
            "onehot_independent_batch_2_apply_label_1_2_connect_2",
            {"independent": True, "applied_labels": [1, 2], "connectivity": 2, "is_onehot": True},
            p(grid_3),
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

    TESTS.append(
        [
            "onehot_dependent_batch_2_apply_label_1_2_connect_2",
            {"independent": False, "applied_labels": [1, 2], "connectivity": 2, "is_onehot": True},
            p(grid_4),
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

    TESTS.append(
        [
            "onehot_none_dependent_batch_2_apply_label_1_2_connect_1",
            {"independent": False, "applied_labels": [1, 2], "connectivity": 1},
            p(grid_4),
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

    TESTS.append(
        [
            "all_non_zero_labels",
            {"independent": True},
            p(grid_1),
            torch.tensor([[[0, 0, 1, 0, 0], [0, 2, 1, 1, 1], [0, 2, 1, 0, 0], [0, 2, 0, 1, 0], [2, 2, 0, 0, 0]]]),
        ]
    )
    # no connected regions
    TESTS.append(["0 regions", {"num_components": 0}, p(grid_6), p(torch.zeros(1, 4, 7))])
    # 1 connected region
    TESTS.append(
        [
            "1 region",
            {"num_components": 1},
            p(grid_6),
            p(
                torch.tensor(
                    [[[0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]]]
                )
            ),
        ]
    )
    # 2 connected regions
    TESTS.append(
        [
            "2 regions",
            {"num_components": 2},
            p(grid_6),
            p(
                torch.tensor(
                    [[[0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 1]]]
                )
            ),
        ]
    )
    # 3+ connected regions unchanged (as input has 3)
    for num_connected in (3, 4):
        TESTS.append([f"{num_connected} regions", {"num_components": num_connected}, p(grid_6), p(grid_6)])


class TestKeepLargestConnectedComponent(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, args, input_image, expected):
        converter = KeepLargestConnectedComponent(**args)
        result = converter(input_image)
        assert_allclose(result, expected, type_test="tensor")

    @parameterized.expand(TESTS)
    def test_correct_results_before_after_onehot(self, _, args, input_image, expected):
        """
        From torch==1.7, torch.argmax changes its mechanism that if there are multiple maximal values then the
        indices of the first maximal value are returned (before this version, the indices of the last maximal value
        are returned).
        Therefore, we can may use of this changes to convert the onehotted labels into un-onehot format directly
        and then check if the result stays the same.

        """
        converter = KeepLargestConnectedComponent(**args)
        result = converter(deepcopy(input_image))

        if "is_onehot" in args:
            args["is_onehot"] = not args["is_onehot"]
        # if not onehotted, onehot it and make sure result stays the same
        if input_image.shape[0] == 1:
            img = to_onehot(input_image)
            result2 = KeepLargestConnectedComponent(**args)(img)
            result2 = result2.argmax(0)[None]
            assert_allclose(result, result2, type_test="tensor")
        # if onehotted, un-onehot and check result stays the same
        else:
            img = input_image.argmax(0)[None]
            result2 = KeepLargestConnectedComponent(**args)(img)
            assert_allclose(result.argmax(0)[None], result2, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
