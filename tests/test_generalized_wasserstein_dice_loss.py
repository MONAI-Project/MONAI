# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @inproceedings{fidon2017generalised,
#  title={Generalised {W}asserstein dice score for imbalanced multi-class segmentation using holistic convolutional networks},
#  author={Fidon, Lucas and Li, Wenqi and Garcia-Peraza-Herrera, Luis C and Ekanayake, Jinendra and Kitchen, Neil and Ourselin, S{\'e}bastien and #Vercauteren, Tom},
#  booktitle={International MICCAI Brainlesion Workshop},
#  pages={64--76},
#  year={2017},
#  organization={Springer}
#}

import unittest
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from monai.losses import GeneralizedWassersteinDiceLoss


class TestGeneralizedWassersteinDiceLoss(unittest.TestCase):
    def test_bin_seg_2d(self):
        M = np.array(
            [[0.,1.],
             [1.,0.]]
        )
        
        target = torch.tensor(
            [[0,0,0,0],
             [0,1,1,0],
             [0,1,1,0],
             [0,0,0,0]]
        )

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(1 - target, num_classes=2).permute(0, 3, 1, 2).float()

        # initialize the loss
        loss = GeneralizedWassersteinDiceLoss(dist_matrix=M)

        # the loss for pred_very_good should be close to 0
        loss_good = float(loss.forward(pred_very_good, target))
        self.assertAlmostEqual(loss_good, 0., places=3)

        # same test, but with target with a class dimension
        target_4dim = target.unsqueeze(1)  # shape (1, 1, H, W)
        loss_good = float(loss.forward(pred_very_good, target_4dim))
        self.assertAlmostEqual(loss_good, 0., places=3)

        # the loss for pred_very_poor should be close to 1
        loss_poor = float(loss.forward(pred_very_poor, target))
        self.assertAlmostEqual(loss_poor, 1., places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        M = np.array(
            [[0.,1.],
             [1.,0.]]
        )
        
        target = torch.tensor(
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(1 - target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the loss
        loss = GeneralizedWassersteinDiceLoss(dist_matrix=M)

        # loss for pred_very_good should be close to 0
        loss_good = float(loss.forward(pred_very_good, target))
        self.assertAlmostEqual(loss_good, 0., places=3)

        # loss for pred_very_poor should be close to 1
        loss_poor = float(loss.forward(pred_very_poor, target))
        self.assertAlmostEqual(loss_poor, 1., places=3)


if __name__ == '__main__':
    unittest.main()
