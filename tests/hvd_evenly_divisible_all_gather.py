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

import torch

from monai.utils import evenly_divisible_all_gather
from monai.utils.module import optional_import
from tests.util import assert_allclose

hvd, has_hvd = optional_import("horovod", name="torch")


class HvdEvenlyDivisibleAllGather:
    def test_data(self):
        # initialize Horovod
        hvd.init()
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
        self._run()

    def _run(self):
        # if hvd.rank() == 0:
        data1 = torch.tensor([[1, 2], [3, 4]])
        data2 = torch.tensor([[1.0, 2.0]])
        data3 = torch.tensor(7)

        if hvd.rank() == 1:
            data1 = torch.tensor([[5, 6]])
            data2 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
            data3 = torch.tensor(8)

        result1 = evenly_divisible_all_gather(data=data1, concat=True)
        assert_allclose(result1, torch.tensor([[1, 2], [3, 4], [5, 6]]))
        result2 = evenly_divisible_all_gather(data=data2, concat=False)
        for r, e in zip(result2, [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0], [5.0, 6.0]])]):
            assert_allclose(r, e)
        result3 = evenly_divisible_all_gather(data=data3, concat=False)
        for r in result3:
            assert_allclose(r.ndimension(), 0)


if __name__ == "__main__":
    """
    1. Install Horovod:
    `HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_NCCL_LINK=SHARED pip install --no-cache-dir horovod`

    2. Execute on 2 GPUs in a single machine:
    `horovodrun -np 2 python test_evenly_divisible_all_gather_hvd.py`

    """
    HvdEvenlyDivisibleAllGather().test_data()
