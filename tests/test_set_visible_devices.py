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

from tests.util import SkipIfAtLeastPyTorchVersion, skip_if_no_cuda


class TestVisibleDevices(unittest.TestCase):
    @staticmethod
    def run_process_and_get_exit_code(code_to_execute):
        value = os.system(code_to_execute)
        return int(bin(value).replace("0b", "").rjust(16, "0")[:8], 2)

    @skip_if_no_cuda
    @SkipIfAtLeastPyTorchVersion((2, 2, 1))
    def test_visible_devices(self):
        num_gpus_before = self.run_process_and_get_exit_code(
            'python -c "import os; import torch; '
            + "os.environ['CUDA_VISIBLE_DEVICES'] = ''; exit(torch.cuda.device_count())\""
        )
        num_gpus_after = self.run_process_and_get_exit_code(
            'python -c "import os; import monai; import torch; '
            + "os.environ['CUDA_VISIBLE_DEVICES'] = ''; exit(torch.cuda.device_count())\""
        )
        self.assertEqual(num_gpus_before, num_gpus_after)

        # test import monai won't affect setting CUDA_VISIBLE_DEVICES
        num_gpus_after_monai = self.run_process_and_get_exit_code(
            'python -c "import os; import torch; import monai; '
            + "os.environ['CUDA_VISIBLE_DEVICES'] = '0'; exit(torch.cuda.device_count())\""
        )
        self.assertEqual(num_gpus_after_monai, 1)


if __name__ == "__main__":
    unittest.main()
