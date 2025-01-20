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

import os
import unittest
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized
from PIL import Image

from monai.transforms import UltrasoundConfidenceMapTransform
from monai.utils import optional_import
from tests.util import assert_allclose

_, has_scipy = optional_import("scipy")

TESTS_PATH = Path(__file__).parents[1]

TEST_INPUT = np.array(
    [
        [1, 2, 3, 23, 13, 22, 5, 1, 2, 3],
        [1, 2, 3, 12, 4, 6, 9, 1, 2, 3],
        [1, 2, 3, 8, 7, 10, 11, 1, 2, 3],
        [1, 2, 3, 14, 15, 16, 17, 1, 2, 3],
        [1, 2, 3, 18, 19, 20, 21, 1, 2, 3],
        [1, 2, 3, 24, 25, 26, 27, 1, 2, 3],
        [1, 2, 3, 28, 29, 30, 31, 1, 2, 3],
        [1, 2, 3, 32, 33, 34, 35, 1, 2, 3],
        [1, 2, 3, 36, 37, 38, 39, 1, 2, 3],
        [1, 2, 3, 40, 41, 42, 43, 1, 2, 3],
    ],
    dtype=np.float32,
)

TEST_MASK = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

SINK_ALL_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [
            0.8884930952884654,
            0.8626656901726876,
            0.8301161870669913,
            0.9757179300830185,
            0.9989819637626414,
            0.9994717624885747,
            0.9954377526794013,
            0.8898638133944221,
            0.862604343021387,
            0.8277862494812598,
        ],
        [
            0.7765718877433174,
            0.7363731552518268,
            0.6871875923653379,
            0.9753673327387775,
            0.9893175316399789,
            0.9944181334242039,
            0.9936979128319371,
            0.7778001700035326,
            0.7362622619974832,
            0.6848377775329241,
        ],
        [
            0.6648416226360719,
            0.6178079903692397,
            0.5630152545966568,
            0.8278402502498404,
            0.82790391019578,
            0.8289702087149963,
            0.8286730258710652,
            0.6658773633169731,
            0.6176836507071695,
            0.5609165245633834,
        ],
        [
            0.5534420483956817,
            0.5055401989946189,
            0.451865872383879,
            0.7541423053657541,
            0.7544115886347456,
            0.7536884376055174,
            0.7524927915364896,
            0.5542943466824017,
            0.505422678400297,
            0.4502051549732117,
        ],
        [
            0.4423657561928356,
            0.398221575954319,
            0.35030055029978124,
            0.4793202144786371,
            0.48057175662074125,
            0.4812057229564038,
            0.48111949176149327,
            0.44304092606050766,
            0.39812149713417405,
            0.34902458531143377,
        ],
        [
            0.3315561576450342,
            0.29476346732036784,
            0.2558303772864961,
            0.35090405668257535,
            0.3515225984307705,
            0.35176548159366317,
            0.3516979775419521,
            0.33205839061494885,
            0.2946859567272435,
            0.2549042599220772,
        ],
        [
            0.22094175240967673,
            0.19431840633358133,
            0.16672448058324435,
            0.22716195845848167,
            0.22761996456848282,
            0.22782525614780919,
            0.22781876632199002,
            0.22127471252104777,
            0.19426593309729956,
            0.16612306610996525,
        ],
        [
            0.11044782531624744,
            0.09623229814933323,
            0.08174664901235043,
            0.11081911718888311,
            0.11102310514207447,
            0.1111041051969924,
            0.11108329076967229,
            0.11061376973431204,
            0.09620592927336903,
            0.08145227209865454,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

SINK_MID_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [
            0.9999957448889315,
            0.9999781044114231,
            0.9999142422442185,
            0.999853253199584,
            0.9999918403054282,
            0.9999874855193227,
            0.9999513619364747,
            0.9999589247003497,
            0.9999861765528631,
            0.9999939213967494,
        ],
        [
            0.9999918011366045,
            0.9999588498417253,
            0.9998388659316617,
            0.9998496524281603,
            0.9999154673258592,
            0.9997827845182361,
            0.9998160234579786,
            0.9999163964511287,
            0.9999743435786168,
            0.9999894752861168,
        ],
        [
            0.9999883847481621,
            0.9999427334014465,
            0.9997703972600652,
            0.9853967608835997,
            0.9852517829915376,
            0.9853308520519438,
            0.9854102394414211,
            0.9998728503298413,
            0.9999642585978225,
            0.999986204909933,
        ],
        [
            0.999985544721449,
            0.9999296195017368,
            0.9997066149628903,
            0.9753803016111353,
            0.9750688049429371,
            0.9749211929217173,
            0.9750052047129354,
            0.9998284130289159,
            0.9999558481338295,
            0.9999837966320273,
        ],
        [
            0.9999832723447848,
            0.9999192263814408,
            0.9996472692076177,
            0.90541293509353,
            0.9049945536526819,
            0.9051142437853055,
            0.9057005861296792,
            0.9997839348839027,
            0.9999490318922627,
            0.9999820419085812,
        ],
        [
            0.9999815409510937,
            0.9999113168889934,
            0.9995930143319085,
            0.8370025145062345,
            0.8358345435164332,
            0.8358231468627223,
            0.8369430449157075,
            0.9997408260265034,
            0.9999437526409107,
            0.9999808010740554,
        ],
        [
            0.9999803198262347,
            0.9999057164296593,
            0.9995461103528891,
            0.7047260555380003,
            0.7023346743490383,
            0.7022946969603594,
            0.7045662738042475,
            0.9997017258131392,
            0.9999399744001316,
            0.9999799785302944,
        ],
        [
            0.9999795785255197,
            0.9999022923125928,
            0.999510772973329,
            0.46283993237260707,
            0.4577365087549323,
            0.4571888733219068,
            0.4614967878524538,
            0.9996710272733927,
            0.9999376682163403,
            0.9999795067125865,
        ],
        [
            0.9999792877553907,
            0.9999009179811408,
            0.9994950057121632,
            0.05049460567213739,
            0.030946131978013824,
            0.0,
            0.019224121648385283,
            0.9996568912408903,
            0.9999367861122628,
            0.9999793358521326,
        ],
    ],
    dtype=np.float32,
)

SINK_MIN_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [
            0.9999961997987318,
            0.9999801752476248,
            0.9999185667341594,
            0.9993115972922259,
            0.9999536433504382,
            0.9997590064584757,
            0.9963282396026231,
            0.9020645423682648,
            0.965641014946897,
            0.9847003633599846,
        ],
        [
            0.9999926824858815,
            0.9999628275604145,
            0.9998472915971415,
            0.9992953054409239,
            0.9995550237000549,
            0.9972853256638443,
            0.9958871482234863,
            0.8006505271617617,
            0.9360757301263053,
            0.9734843475613124,
        ],
        [
            0.9999896427490426,
            0.9999484707116104,
            0.9997841142091455,
            0.9321779021295554,
            0.9308591506422442,
            0.9299937642438358,
            0.9286536283468563,
            0.6964658886602826,
            0.9106656689679997,
            0.9652109119709528,
        ],
        [
            0.9999871227708508,
            0.9999369646510842,
            0.9997276125796202,
            0.9006206490361908,
            0.8987968702587018,
            0.8965696900664386,
            0.8941507574801211,
            0.5892568658180841,
            0.8892240419729905,
            0.9590996257620853,
        ],
        [
            0.9999851119906539,
            0.9999280075234918,
            0.9996788394671484,
            0.778755271203017,
            0.7763917808258874,
            0.7737517385551721,
            0.7707980517990098,
            0.4788014936236403,
            0.8715671104783401,
            0.954632732759503,
        ],
        [
            0.9999835837292402,
            0.999921323618806,
            0.9996389455307461,
            0.7222961578407286,
            0.7186158832946955,
            0.7146983167265393,
            0.7105768254632475,
            0.3648911004360315,
            0.8575943501305144,
            0.9514642802768379,
        ],
        [
            0.9999825081019064,
            0.999916683268467,
            0.9996093996776352,
            0.6713490686473397,
            0.6664914636518112,
            0.6613110504728309,
            0.6558325489984669,
            0.247299682539502,
            0.8473037957967624,
            0.9493580587294981,
        ],
        [
            0.999981856118739,
            0.9999138938063622,
            0.9995907248497593,
            0.6331535096751639,
            0.6271637176135582,
            0.6206687804556549,
            0.6136262027168252,
            0.12576864809108962,
            0.8407892431959736,
            0.9481472656653798,
        ],
        [
            0.9999816006081851,
            0.9999127861527936,
            0.9995832399159849,
            0.6133274396648696,
            0.6086364734302403,
            0.6034602717119345,
            0.5978473214165134,
            0.0,
            0.8382338778894218,
            0.9477082231321966,
        ],
    ],
    dtype=np.float32,
)

SINK_MASK_OUTPUT = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.9047934405899283, 0.9936046284605553, 0.9448690902377527, 0.0, 0.0, 0.0, 0.8363773255131761],
        [0.0, 0.0, 0.0, 0.90375200446097, 0.9434594475474036, 0.4716831449516178, 0.0, 0.0, 0.0, 0.7364197333910302],
        [
            0.0,
            0.0,
            0.0,
            0.09080438801405301,
            0.06774182873204163,
            0.038207095016625024,
            0.0,
            0.0,
            0.0,
            0.6745641479264269,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.01731082802870267,
            0.013540929458217351,
            0.007321202161532623,
            0.0,
            0.0,
            0.0,
            0.6341231654271253,
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0006444251665178544,
            0.0005397129128756325,
            0.0003048384803626333,
            0.0,
            0.0,
            0.0,
            0.6070178708536365,
        ],
        [
            0.0,
            0.0,
            0.0,
            5.406078586212675e-05,
            4.416783924970537e-05,
            2.4597362039020103e-05,
            0.0,
            0.0,
            0.0,
            0.5889413683184284,
        ],
        [
            0.0,
            0.0,
            0.0,
            4.39259327223233e-06,
            3.6050656774754658e-06,
            2.0127120155893425e-06,
            0.0,
            0.0,
            0.0,
            0.5774279920364456,
        ],
        [
            0.0,
            0.0,
            0.0,
            4.0740501726718113e-07,
            3.374875487404489e-07,
            1.9113630985667455e-07,
            0.0,
            0.0,
            0.0,
            0.5709897726747111,
        ],
        [
            3.2266922388030425e-17,
            1.801110982679718e-14,
            9.325899448306927e-12,
            3.913608442133728e-07,
            3.9581822403393465e-07,
            4.02383505118481e-07,
            4.14820241328287e-07,
            4.281640797396309e-06,
            0.0023900192231620593,
            0.5686882523793125,
        ],
    ],
    dtype=np.float32,
)


@unittest.skipUnless(has_scipy, "Requires scipy")
class TestUltrasoundConfidenceMapTransform(unittest.TestCase):
    def setUp(self):
        self.input_img_np = np.expand_dims(TEST_INPUT, axis=0)  # mock image (numpy array)
        self.input_mask_np = np.expand_dims(TEST_MASK, axis=0)  # mock mask (numpy array)

        self.input_img_torch = torch.from_numpy(TEST_INPUT).unsqueeze(0)  # mock image (torch tensor)
        self.input_mask_torch = torch.from_numpy(TEST_MASK).unsqueeze(0)  # mock mask (torch tensor)

        self.real_input_img_paths = [
            os.path.join(TESTS_PATH, "testing_data", "ultrasound_confidence_map", "neck_input.png"),
            os.path.join(TESTS_PATH, "testing_data", "ultrasound_confidence_map", "femur_input.png"),
        ]

        self.real_result_npy_paths = [
            os.path.join(TESTS_PATH, "testing_data", "ultrasound_confidence_map", "neck_result.npy"),
            os.path.join(TESTS_PATH, "testing_data", "ultrasound_confidence_map", "femur_result.npy"),
        ]

        self.real_input_paramaters = [
            {"alpha": 2.0, "beta": 90, "gamma": 0.03},
            {"alpha": 2.0, "beta": 90, "gamma": 0.06},
        ]

    def test_parameters(self):
        # Unknown mode
        with self.assertRaises(ValueError):
            UltrasoundConfidenceMapTransform(mode="unknown")

        # Unknown sink_mode
        with self.assertRaises(ValueError):
            UltrasoundConfidenceMapTransform(sink_mode="unknown")

    @parameterized.expand(
        [("all", SINK_ALL_OUTPUT), ("mid", SINK_MID_OUTPUT), ("min", SINK_MIN_OUTPUT), ("mask", SINK_MASK_OUTPUT, True)]
    )
    def test_ultrasound_confidence_map_transform(self, sink_mode, expected_output, use_mask=False):
        # RGB image
        input_img_rgb = np.expand_dims(np.repeat(self.input_img_np, 3, axis=0), axis=0)
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode=sink_mode)

        if use_mask:
            result_torch = transform(input_img_rgb_torch, self.input_mask_torch)
            result_np = transform(input_img_rgb, self.input_mask_np)
        else:
            result_torch = transform(input_img_rgb_torch)
            result_np = transform(input_img_rgb)

        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(expected_output), rtol=1e-4, atol=1e-4)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, expected_output, rtol=1e-4, atol=1e-4)

    @parameterized.expand(
        [
            ("all", SINK_ALL_OUTPUT),
            ("mid", SINK_MID_OUTPUT),
            ("min", SINK_MIN_OUTPUT),
            ("mask", SINK_MASK_OUTPUT, True),  # Adding a flag for mask cases
        ]
    )
    def test_multi_channel_2d(self, sink_mode, expected_output, use_mask=False):
        input_img_rgb = np.expand_dims(np.repeat(self.input_img_np, 17, axis=0), axis=0)
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode=sink_mode)

        if use_mask:
            result_torch = transform(input_img_rgb_torch, self.input_mask_torch)
            result_np = transform(input_img_rgb, self.input_mask_np)
        else:
            result_torch = transform(input_img_rgb_torch)
            result_np = transform(input_img_rgb)

        self.assertIsInstance(result_torch, torch.Tensor)
        assert_allclose(result_torch, torch.tensor(expected_output), rtol=1e-4, atol=1e-4)
        self.assertIsInstance(result_np, np.ndarray)
        assert_allclose(result_np, expected_output, rtol=1e-4, atol=1e-4)

    @parameterized.expand([("all",), ("mid",), ("min",), ("mask",)])
    def test_non_one_first_dim(self, sink_mode):
        transform = UltrasoundConfidenceMapTransform(sink_mode=sink_mode)
        input_img_rgb = np.repeat(self.input_img_np, 3, axis=0)
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        if sink_mode == "mask":
            with self.assertRaises(ValueError):
                transform(input_img_rgb_torch, self.input_mask_torch)
            with self.assertRaises(ValueError):
                transform(input_img_rgb, self.input_mask_np)
        else:
            with self.assertRaises(ValueError):
                transform(input_img_rgb_torch)
            with self.assertRaises(ValueError):
                transform(input_img_rgb)

    @parameterized.expand([("all",), ("mid",), ("min",), ("mask",)])
    def test_no_first_dim(self, sink_mode):
        input_img_rgb = self.input_img_np[0]
        input_img_rgb_torch = torch.from_numpy(input_img_rgb)

        transform = UltrasoundConfidenceMapTransform(sink_mode=sink_mode)

        with self.assertRaises(ValueError):
            transform(input_img_rgb_torch)
        with self.assertRaises(ValueError):
            transform(input_img_rgb)

        if sink_mode == "mask":
            with self.assertRaises(ValueError):
                transform(input_img_rgb_torch, self.input_mask_torch)
            with self.assertRaises(ValueError):
                transform(input_img_rgb, self.input_mask_np)

    @parameterized.expand([("all",), ("mid",), ("min",)])
    def test_sink_mode(self, mode):
        transform = UltrasoundConfidenceMapTransform(sink_mode=mode)

        # This should not raise an exception for torch tensor
        result_torch = transform(self.input_img_torch)
        self.assertIsInstance(result_torch, torch.Tensor)

        # This should not raise an exception for numpy array
        result_np = transform(self.input_img_np)
        self.assertIsInstance(result_np, np.ndarray)

    def test_sink_mask(self):
        transform = UltrasoundConfidenceMapTransform(sink_mode="mask")

        # This should not raise an exception for torch tensor with mask
        result_torch = transform(self.input_img_torch, self.input_mask_torch)
        self.assertIsInstance(result_torch, torch.Tensor)

        # This should not raise an exception for numpy array with mask
        result_np = transform(self.input_img_np, self.input_mask_np)
        self.assertIsInstance(result_np, np.ndarray)

        # This should raise an exception for torch tensor without mask
        with self.assertRaises(ValueError):
            transform(self.input_img_torch)

        # This should raise an exception for numpy array without mask
        with self.assertRaises(ValueError):
            transform(self.input_img_np)

    def test_func(self):
        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="all")
        output = transform(self.input_img_np)
        assert_allclose(output, SINK_ALL_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mid")
        output = transform(self.input_img_np)
        assert_allclose(output, SINK_MID_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="min")
        output = transform(self.input_img_np)
        assert_allclose(output, SINK_MIN_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mask")
        output = transform(self.input_img_np, self.input_mask_np)
        assert_allclose(output, SINK_MASK_OUTPUT, rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="all")
        output = transform(self.input_img_torch)
        assert_allclose(output, torch.tensor(SINK_ALL_OUTPUT), rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mid")
        output = transform(self.input_img_torch)
        assert_allclose(output, torch.tensor(SINK_MID_OUTPUT), rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="min")
        output = transform(self.input_img_torch)
        assert_allclose(output, torch.tensor(SINK_MIN_OUTPUT), rtol=1e-4, atol=1e-4)

        transform = UltrasoundConfidenceMapTransform(alpha=2.0, beta=90.0, gamma=0.05, mode="B", sink_mode="mask")
        output = transform(self.input_img_torch, self.input_mask_torch)
        assert_allclose(output, torch.tensor(SINK_MASK_OUTPUT), rtol=1e-4, atol=1e-4)

    def test_against_official_code(self):
        # This test is to compare the output of the transform with the official code
        # The official code is available at:
        # https://campar.in.tum.de/Main/AthanasiosKaramalisCode

        for input_img_path, result_npy_path, params in zip(
            self.real_input_img_paths, self.real_result_npy_paths, self.real_input_paramaters
        ):
            input_img = np.array(Image.open(input_img_path))
            input_img = np.expand_dims(input_img, axis=0)

            result_img = np.load(result_npy_path)

            transform = UltrasoundConfidenceMapTransform(sink_mode="all", **params)
            output = transform(input_img)

            assert_allclose(output, result_img, rtol=1e-4, atol=1e-4)

    def test_against_official_code_using_cg(self):
        # This test is to compare the output of the transform with the official code
        # The official code is available at:
        # https://campar.in.tum.de/Main/AthanasiosKaramalisCode

        for input_img_path, result_npy_path, params in zip(
            self.real_input_img_paths, self.real_result_npy_paths, self.real_input_paramaters
        ):
            input_img = np.array(Image.open(input_img_path))
            input_img = np.expand_dims(input_img, axis=0)

            result_img = np.load(result_npy_path)

            transform = UltrasoundConfidenceMapTransform(
                sink_mode="all", use_cg=True, cg_tol=1.0e-6, cg_maxiter=300, **params
            )
            output = transform(input_img)

            assert_allclose(output, result_img, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
