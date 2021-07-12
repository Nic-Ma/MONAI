# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms import RandCropByLabelClasses, ClassesToIndices

TEST_CASE_0 = [
    # One-Hot label
    {
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "num_classes": None,
        "spatial_size": [2, 2, -1],
        "ratios": [1, 1, 1],
        "num_samples": 2,
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "image_threshold": 0,
    },
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    list,
    (3, 2, 2, 3),
]

TEST_CASE_1 = [
    # Argmax label
    {
        "label": np.random.randint(0, 2, size=[1, 3, 3, 3]),
        "num_classes": 2,
        "spatial_size": [2, 2, 2],
        "ratios": [1, 1],
        "num_samples": 2,
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "image_threshold": 0,
    },
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    list,
    (3, 2, 2, 2),
]

TEST_CASE_2 = [
    # provide label at runtime
    {
        "label": None,
        "num_classes": 2,
        "spatial_size": [2, 2, 2],
        "ratios": [1, 1],
        "num_samples": 2,
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "image_threshold": 0,
    },
    {
        "img": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label": np.random.randint(0, 2, size=[1, 3, 3, 3]),
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
    },
    list,
    (3, 2, 2, 2),
]


class TestRandCropByLabelClasses(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = RandCropByLabelClasses(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)

    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    def test_indices(self, input_param, input_data, expected_type, expected_shape):
        input_param["indices"] = ClassesToIndices(num_classes=input_param["num_classes"])(input_param["label"])
        result = RandCropByLabelClasses(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)
        # test set indices at runtime
        input_data["indices"] = input_param["indices"]
        result = RandCropByLabelClasses(**input_param)(**input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
