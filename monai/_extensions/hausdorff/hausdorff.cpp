// Copyright (c) MONAI Consortium
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>

#include <vector>

// CUDA  declarations

int getHausdorffDistance_CUDA(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput,
    const int xDim,
    const int yDim,
    const int zDim,
    const float robustnessPercent);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int getHausdorffDistance(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput,
    const int xDim,
    const int yDim,
    const int zDim,
    const float robustnessPercent = 1.0) {
  CHECK_INPUT(goldStandard);
  CHECK_INPUT(algoOutput);

  return getHausdorffDistance_CUDA(goldStandard, algoOutput, xDim, yDim, zDim, robustnessPercent);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("getHausdorffDistance", &getHausdorffDistance, "Basic version of Hausdorff distance");
}

