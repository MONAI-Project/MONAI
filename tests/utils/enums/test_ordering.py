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

import numpy as np
from parameterized import parameterized

from monai.utils.enums import OrderingTransformations, OrderingType
from monai.utils.ordering import Ordering

TEST_2D_NON_RANDOM = [
    [
        {
            "ordering_type": OrderingType.RASTER_SCAN,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": (),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 1, 2, 3],
    ],
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": (),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 1, 3, 2],
    ],
    [
        {
            "ordering_type": OrderingType.RASTER_SCAN,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": (),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [2, 3, 0, 1],
    ],
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": (),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [2, 3, 1, 0],
    ],
    [
        {
            "ordering_type": OrderingType.RASTER_SCAN,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": ((1, 0),),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 2, 1, 3],
    ],
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": ((1, 0),),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 2, 3, 1],
    ],
    [
        {
            "ordering_type": OrderingType.RASTER_SCAN,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": (),
            "rot90_axes": ((0, 1),),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [1, 3, 0, 2],
    ],
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": (),
            "rot90_axes": ((0, 1),),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [1, 3, 2, 0],
    ],
    [
        {
            "ordering_type": OrderingType.RASTER_SCAN,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": ((1, 0),),
            "rot90_axes": ((0, 1),),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 1, 2, 3],
    ],
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": ((1, 0),),
            "rot90_axes": ((0, 1),),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 1, 3, 2],
    ],
]


TEST_3D = [
    [
        {
            "ordering_type": OrderingType.RASTER_SCAN,
            "spatial_dims": 3,
            "dimensions": (1, 2, 2, 2),
            "reflected_spatial_dims": (),
            "transpositions_axes": (),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        },
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]
]

TEST_ORDERING_TYPE_FAILURE = [
    [
        {
            "ordering_type": "hilbert",
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": ((1, 0),),
            "rot90_axes": ((0, 1),),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        }
    ]
]

TEST_ORDERING_TRANSFORMATION_FAILURE = [
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": ((1, 0),),
            "rot90_axes": ((0, 1),),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                "flip",
            ),
        }
    ]
]

TEST_REVERT = [
    [
        {
            "ordering_type": OrderingType.S_CURVE,
            "spatial_dims": 2,
            "dimensions": (1, 2, 2),
            "reflected_spatial_dims": (True, False),
            "transpositions_axes": (),
            "rot90_axes": (),
            "transformation_order": (
                OrderingTransformations.TRANSPOSE.value,
                OrderingTransformations.ROTATE_90.value,
                OrderingTransformations.REFLECT.value,
            ),
        }
    ]
]


class TestOrdering(unittest.TestCase):
    @parameterized.expand(TEST_2D_NON_RANDOM + TEST_3D)
    def test_ordering(self, input_param, expected_sequence_ordering):
        ordering = Ordering(**input_param)
        self.assertTrue(np.array_equal(ordering.get_sequence_ordering(), expected_sequence_ordering, equal_nan=True))

    @parameterized.expand(TEST_ORDERING_TYPE_FAILURE)
    def test_ordering_type_failure(self, input_param):
        with self.assertRaises(ValueError):
            Ordering(**input_param)

    @parameterized.expand(TEST_ORDERING_TRANSFORMATION_FAILURE)
    def test_ordering_transformation_failure(self, input_param):
        with self.assertRaises(ValueError):
            Ordering(**input_param)

    @parameterized.expand(TEST_REVERT)
    def test_revert(self, input_param):
        sequence = np.random.randint(0, 100, size=input_param["dimensions"]).flatten()

        ordering = Ordering(**input_param)

        reverted_sequence = sequence[ordering.get_sequence_ordering()]
        reverted_sequence = reverted_sequence[ordering.get_revert_sequence_ordering()]

        self.assertTrue(np.array_equal(sequence, reverted_sequence, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
