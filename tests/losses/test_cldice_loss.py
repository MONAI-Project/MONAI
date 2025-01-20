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
import torch
from parameterized import parameterized

from monai.losses import SoftclDiceLoss, SoftDiceclDiceLoss

TEST_CASES = [
    [{"y_pred": torch.ones((7, 3, 11, 10)), "y_true": torch.ones((7, 3, 11, 10))}, 0.0],
    [{"y_pred": torch.ones((2, 3, 13, 14, 5)), "y_true": torch.ones((2, 3, 13, 14, 5))}, 0.0],
]


class TestclDiceLoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_result(self, y_pred_data, expected_val):
        loss = SoftclDiceLoss()
        loss_dice = SoftDiceclDiceLoss()
        result = loss(**y_pred_data)
        result_dice = loss_dice(**y_pred_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result_dice.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_with_cuda(self):
        loss = SoftclDiceLoss()
        loss_dice = SoftDiceclDiceLoss()
        i = torch.ones((100, 3, 256, 256))
        j = torch.ones((100, 3, 256, 256))
        if torch.cuda.is_available():
            i = i.cuda()
            j = j.cuda()
        output = loss(i, j)
        output_dice = loss_dice(i, j)
        np.testing.assert_allclose(output.detach().cpu().numpy(), 0.0, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(output_dice.detach().cpu().numpy(), 0.0, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
