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

import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks import one_hot
from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction


class AUCMLoss(_Loss):
    def __init__(
        self,
        margin: float = 1.0,
        imratio: List[float] = None,
        num_classes: int = 2,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ):
        super().__init__(reduction=LossReduction(reduction).value)
        
        if margin < 0 or margin > 1:
            raise ValueError("imratio must be between 0 and 1")
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        
        self.margin = margin
        
        self.p = imratio
        self.num_classes = num_classes
        if self.p:
            assert len(self.p) == self.num_classes , "imratio must be a list of length num_classes"
        else:
            self.p = [0.0] * self.num_classes
        
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.a = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device)
        self.alpha = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, auto=True) -> torch.Tensor:
        """
        Args:
            y_pred
        """
        
        
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        n_pred_ch = y_pred.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                y_pred = torch.softmax(y_pred, dim=1)
        
        if self.other_act is not None:
            y_pred = self.other_act(y_pred)
        
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes = n_pred_ch)
            
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape. y_pred.shape: {}, y_true.shape: {}".format(y_pred.shape, y_true.shape))
        
        total_loss = 0
        for idx in range(self.num_classes):
            if len(y_pred[:, idx].shape) == 1:
               y_pred = y_pred[:, idx].reshape(-1, 1)
            if len(y_true[:, idx].shape) == 1:
               y_true = y_true[:, idx].reshape(-1, 1)
            if auto or not self.p:
               self.p[idx] = (y_true==1).sum()/y_true.shape[0]

            loss =  + (1-self.p[idx]) * torch.mean((y_pred - self.a[idx])**2 * (1==y_true).float()) \
                    + self.p[idx] * torch.mean((y_pred - self.b[idx])**2 *(0==y_true).float()) \
                    + 2 * self.alpha[idx] * (self.p[idx] * (1 - self.p[idx]) \
                    + torch.mean((self.p[idx] * y_pred * (0==y_true).float() \
                    - (1 - self.p[idx]) * y_pred * (1==y_true).float()))) \
                    - self.p[idx] * (1  -self.p[idx]) * self.alpha[idx]**2

            total_loss += loss

        return total_loss