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

import tempfile
import unittest
from os import environ

from monai.handlers import ClearMLStatsHandler
from monai.utils import optional_import

Task, has_clearml = optional_import("clearml", name="Task")
get_active_config_file, has_get_active_config_file = optional_import(
    "clearml.backend_config.defs", name="get_active_config_file"
)
_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")


@unittest.skipUnless(has_clearml, "Requires 'clearml' installation")
@unittest.skipUnless(has_tb, "Requires SummaryWriter installation")
@unittest.skipIf(not has_get_active_config_file, "ClearML 'get_active_config_file' not found")
class TestHandlerClearMLStatsHandler(unittest.TestCase):

    def test_task_init(self):
        handle, path = tempfile.mkstemp()
        with open(handle, "w") as new_config:
            if get_active_config_file():
                with open(get_active_config_file()) as old_config:
                    new_config.write(old_config.read())
            new_config.write(
                "\nsdk.development.vcs_repo_detect_async: false\nsdk.development.report_use_subprocess: false\n"
            )
        environ["CLEARML_CONFIG_FILE"] = path
        try:
            Task.force_store_standalone_script(True)
            Task.set_offline(offline_mode=True)
            ClearMLStatsHandler(
                project_name="MONAI",
                task_name="monai_experiment",
                output_uri=True,
                tags=None,
                reuse_last_task_id=True,
                continue_last_task=False,
                auto_connect_frameworks=True,
                auto_connect_arg_parser=False,
            )
        except Exception as exc:
            self.fail(exc)
        self.assertEqual(Task.current_task().name, "monai_experiment")
        self.assertEqual(Task.current_task()._project_name[1], "MONAI")
        # Close ClearML Task
        Task.current_task().close()


if __name__ == "__main__":
    unittest.main()
