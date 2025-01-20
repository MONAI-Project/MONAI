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

import argparse
import copy
import datetime
import functools
import importlib
import json
import operator
import os
import queue
import ssl
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import warnings
from contextlib import contextmanager
from functools import partial, reduce
from subprocess import PIPE, Popen
from typing import Callable
from urllib.error import ContentTooShortError, HTTPError

import numpy as np
import torch
import torch.distributed as dist

from monai.apps.utils import download_url
from monai.config import NdarrayTensor
from monai.config.deviceconfig import USE_COMPILED
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import create_test_image_2d, create_test_image_3d
from monai.data.meta_tensor import MetaTensor, get_track_meta
from monai.networks import convert_to_onnx, convert_to_torchscript
from monai.utils import optional_import
from monai.utils.misc import MONAIEnvVars
from monai.utils.module import compute_capabilities_after, pytorch_after
from monai.utils.tf32 import detect_default_tf32
from monai.utils.type_conversion import convert_data_type

nib, _ = optional_import("nibabel")
http_error, has_req = optional_import("requests", name="HTTPError")

quick_test_var = "QUICKTEST"
_tf32_enabled = None
_test_data_config: dict = {}


def testing_data_config(*keys):
    """get _test_data_config[keys0][keys1]...[keysN]"""
    if not _test_data_config:
        with open(os.path.join(os.path.dirname(__file__), "testing_data", "data_config.json")) as c:
            _config = json.load(c)
            for k, v in _config.items():
                _test_data_config[k] = v
    return reduce(operator.getitem, keys, _test_data_config)


def get_testing_algo_template_path():
    """
    a local folder to the testing algorithm template or a url to the compressed template file.
    Default to None, which effectively uses bundle_gen's ``default_algo_zip`` path.

    https://github.com/Project-MONAI/MONAI/blob/1.1.0/monai/apps/auto3dseg/bundle_gen.py#L380-L381
    """
    return MONAIEnvVars.testing_algo_template()


def clone(data: NdarrayTensor) -> NdarrayTensor:
    """
    Clone data independent of type.

    Args:
        data (NdarrayTensor): This can be a Pytorch Tensor or numpy array.

    Returns:
        Any: Cloned data object
    """
    return copy.deepcopy(data)


def assert_allclose(
    actual: NdarrayOrTensor,
    desired: NdarrayOrTensor,
    type_test: bool | str = True,
    device_test: bool = False,
    *args,
    **kwargs,
):
    """
    Assert that types and all values of two data objects are close.

    Args:
        actual: Pytorch Tensor or numpy array for comparison.
        desired: Pytorch Tensor or numpy array to compare against.
        type_test: whether to test that `actual` and `desired` are both numpy arrays or torch tensors.
            if type_test == "tensor", it checks whether the `actual` is a torch.tensor or metatensor according to
            `get_track_meta`.
        device_test: whether to test the device property.
        args: extra arguments to pass on to `np.testing.assert_allclose`.
        kwargs: extra arguments to pass on to `np.testing.assert_allclose`.


    """
    if isinstance(type_test, str) and type_test == "tensor":
        if get_track_meta():
            np.testing.assert_equal(isinstance(actual, MetaTensor), True, "must be a MetaTensor")
        else:
            np.testing.assert_equal(
                isinstance(actual, torch.Tensor) and not isinstance(actual, MetaTensor), True, "must be a torch.Tensor"
            )
    elif type_test:
        # check both actual and desired are of the same type
        np.testing.assert_equal(isinstance(actual, np.ndarray), isinstance(desired, np.ndarray), "numpy type")
        np.testing.assert_equal(isinstance(actual, torch.Tensor), isinstance(desired, torch.Tensor), "torch type")

    if isinstance(desired, torch.Tensor) or isinstance(actual, torch.Tensor):
        if device_test:
            np.testing.assert_equal(str(actual.device), str(desired.device), "torch device check")  # type: ignore
        actual = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else actual
        desired = desired.detach().cpu().numpy() if isinstance(desired, torch.Tensor) else desired
    np.testing.assert_allclose(actual, desired, *args, **kwargs)


@contextmanager
def skip_if_downloading_fails():
    try:
        yield
    except (ContentTooShortError, HTTPError, ConnectionError) + (http_error,) if has_req else () as e:  # noqa: B030
        raise unittest.SkipTest(f"error while downloading: {e}") from e
    except ssl.SSLError as ssl_e:
        if "decryption failed" in str(ssl_e):
            raise unittest.SkipTest(f"SSL error while downloading: {ssl_e}") from ssl_e
    except (RuntimeError, OSError) as rt_e:
        err_str = str(rt_e)
        if any(
            k in err_str
            for k in (
                "unexpected EOF",  # incomplete download
                "network issue",
                "gdown dependency",  # gdown not installed
                "md5 check",
                "limit",  # HTTP Error 503: Egress is over the account limit
                "authenticate",
                "timed out",  # urlopen error [Errno 110] Connection timed out
                "HTTPError",  # HTTPError: 429 Client Error: Too Many Requests for huggingface hub
            )
        ):
            raise unittest.SkipTest(f"error while downloading: {rt_e}") from rt_e  # incomplete download

        raise rt_e


def test_pretrained_networks(network, input_param, device):
    with skip_if_downloading_fails():
        return network(**input_param).to(device)


def test_is_quick():
    return os.environ.get(quick_test_var, "").lower() == "true"


def is_tf32_env():
    """
    When we may be using TF32 mode, check the precision of matrix operation.
    If the checking result is greater than the threshold 0.001,
    set _tf32_enabled=True (and relax _rtol for tests).
    """
    global _tf32_enabled
    if _tf32_enabled is None:
        _tf32_enabled = False
        if torch.cuda.is_available() and (detect_default_tf32() or torch.backends.cuda.matmul.allow_tf32):
            try:
                # with TF32 enabled, the speed is ~8x faster, but the precision has ~2 digits less in the result
                g_gpu = torch.Generator(device="cuda")
                g_gpu.manual_seed(2147483647)
                a_full = torch.randn(1024, 1024, dtype=torch.double, device="cuda", generator=g_gpu)
                b_full = torch.randn(1024, 1024, dtype=torch.double, device="cuda", generator=g_gpu)
                _tf32_enabled = (a_full.float() @ b_full.float() - a_full @ b_full).abs().max().item() > 0.001  # 0.1713
            except BaseException:
                pass
        print(f"tf32 enabled: {_tf32_enabled}")
    return _tf32_enabled


def skip_if_quick(obj):
    """
    Skip the unit tests if environment variable `quick_test_var=true`.
    For example, the user can skip the relevant tests by setting ``export QUICKTEST=true``.
    """
    is_quick = test_is_quick()

    return unittest.skipIf(is_quick, "Skipping slow tests")(obj)


class SkipIfNoModule:
    """Decorator to be used if test should be skipped
    when optional module is not present."""

    def __init__(self, module_name):
        self.module_name = module_name
        self.module_missing = not optional_import(self.module_name)[1]

    def __call__(self, obj):
        return unittest.skipIf(self.module_missing, f"optional module not present: {self.module_name}")(obj)


class SkipIfModule:
    """Decorator to be used if test should be skipped
    when optional module is present."""

    def __init__(self, module_name):
        self.module_name = module_name
        self.module_avail = optional_import(self.module_name)[1]

    def __call__(self, obj):
        return unittest.skipIf(self.module_avail, f"Skipping because optional module present: {self.module_name}")(obj)


def skip_if_no_cpp_extension(obj):
    """
    Skip the unit tests if the cpp extension is not available.
    """
    return unittest.skipUnless(USE_COMPILED, "Skipping cpp extension tests")(obj)


def skip_if_no_cuda(obj):
    """
    Skip the unit tests if torch.cuda.is_available is False.
    """
    return unittest.skipUnless(torch.cuda.is_available(), "Skipping CUDA-based tests")(obj)


def skip_if_windows(obj):
    """
    Skip the unit tests if platform is win32.
    """
    return unittest.skipIf(sys.platform == "win32", "Skipping tests on Windows")(obj)


def skip_if_darwin(obj):
    """
    Skip the unit tests if platform is macOS (Darwin).
    """
    return unittest.skipIf(sys.platform == "darwin", "Skipping tests on macOS/Darwin")(obj)


class SkipIfBeforePyTorchVersion:
    """Decorator to be used if test should be skipped
    with PyTorch versions older than that given."""

    def __init__(self, pytorch_version_tuple):
        self.min_version = pytorch_version_tuple
        self.version_too_old = not pytorch_after(*pytorch_version_tuple)

    def __call__(self, obj):
        return unittest.skipIf(
            self.version_too_old, f"Skipping tests that fail on PyTorch versions before: {self.min_version}"
        )(obj)


class SkipIfAtLeastPyTorchVersion:
    """Decorator to be used if test should be skipped
    with PyTorch versions newer than or equal to that given."""

    def __init__(self, pytorch_version_tuple):
        self.max_version = pytorch_version_tuple
        self.version_too_new = pytorch_after(*pytorch_version_tuple)

    def __call__(self, obj):
        return unittest.skipIf(
            self.version_too_new, f"Skipping tests that fail on PyTorch versions at least: {self.max_version}"
        )(obj)


class SkipIfBeforeComputeCapabilityVersion:
    """Decorator to be used if test should be skipped
    with Compute Capability older than that given."""

    def __init__(self, compute_capability_tuple):
        self.min_version = compute_capability_tuple
        self.version_too_old = not compute_capabilities_after(*compute_capability_tuple)

    def __call__(self, obj):
        return unittest.skipIf(
            self.version_too_old, f"Skipping tests that fail on Compute Capability versions before: {self.min_version}"
        )(obj)


def is_main_test_process():
    ps = torch.multiprocessing.current_process()
    if not ps or not hasattr(ps, "name"):
        return False
    return ps.name.startswith("Main")


def has_cupy():
    """
    Returns True if the user has installed a version of cupy.
    """
    cp, has_cp = optional_import("cupy")
    if not is_main_test_process():
        return has_cp  # skip the check if we are running in subprocess
    if not has_cp:
        return False
    try:  # test cupy installation with a basic example
        x = cp.arange(6, dtype="f").reshape(2, 3)
        y = cp.arange(3, dtype="f")
        kernel = cp.ElementwiseKernel(
            "float32 x, float32 y", "float32 z", """ if (x - 2 > y) { z = x * y; } else { z = x + y; } """, "my_kernel"
        )
        flag = kernel(x, y)[0, 0] == 0
        del x, y, kernel
        cp.get_default_memory_pool().free_all_blocks()
        return flag
    except Exception:
        return False


HAS_CUPY = has_cupy()


def make_nifti_image(
    array: NdarrayOrTensor, affine=None, dir=None, fname=None, suffix=".nii.gz", verbose=False, dtype=float
):
    """
    Create a temporary nifti image on the disk and return the image name.
    User is responsible for deleting the temporary file when done with it.
    """
    if isinstance(array, torch.Tensor):
        array, *_ = convert_data_type(array, np.ndarray)
    if isinstance(affine, torch.Tensor):
        affine, *_ = convert_data_type(affine, np.ndarray)
    if affine is None:
        affine = np.eye(4)
    test_image = nib.Nifti1Image(array.astype(dtype), affine)  # type: ignore

    # if dir not given, create random. Else, make sure it exists.
    if dir is None:
        dir = tempfile.mkdtemp()
    else:
        os.makedirs(dir, exist_ok=True)

    # If fname not given, get random one. Else, concat dir, fname and suffix.
    if fname is None:
        temp_f, fname = tempfile.mkstemp(suffix=suffix, dir=dir)
        os.close(temp_f)
    else:
        fname = os.path.join(dir, fname + suffix)

    nib.save(test_image, fname)
    if verbose:
        print(f"File written: {fname}.")
    return fname


def make_rand_affine(ndim: int = 3, random_state: np.random.RandomState | None = None):
    """Create random affine transformation (with values == -1, 0 or 1)."""
    rs = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    vals = rs.choice([-1, 1], size=ndim)
    positions = rs.choice(range(ndim), size=ndim, replace=False)
    af = np.zeros([ndim + 1, ndim + 1])
    af[ndim, ndim] = 1
    for i, (v, p) in enumerate(zip(vals, positions)):
        af[i, p] = v
    return af


def get_arange_img(size, dtype=np.float32, offset=0):
    """
    Returns an image as a numpy array (complete with channel as dim 0)
    with contents that iterate like an arange.
    """
    n_elem = np.prod(size)
    img = np.arange(offset, offset + n_elem, dtype=dtype).reshape(size)
    return np.expand_dims(img, 0)


class DistTestCase(unittest.TestCase):
    """
    testcase without _outcome, so that it's picklable.
    """

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["_outcome"]
        return self_dict

    def __setstate__(self, data_dict):
        self.__dict__.update(data_dict)


class DistCall:
    """
    Wrap a test case so that it will run in multiple processes on a single machine using `torch.distributed`.
    It is designed to be used with `tests.utils.DistTestCase`.

    Usage:

        decorate a unittest testcase method with a `DistCall` instance::

            class MyTests(unittest.TestCase):
                @DistCall(nnodes=1, nproc_per_node=3, master_addr="localhost")
                def test_compute(self):
                ...

        the `test_compute` method should trigger different worker logic according to `dist.get_rank()`.

    Multi-node tests require a fixed master_addr:master_port, with node_rank set manually in multiple scripts
    or from environment variable "NODE_RANK".
    """

    def __init__(
        self,
        nnodes: int = 1,
        nproc_per_node: int = 1,
        master_addr: str = "localhost",
        master_port: int | None = None,
        node_rank: int | None = None,
        timeout=60,
        init_method=None,
        backend: str | None = None,
        daemon: bool | None = None,
        method: str | None = "spawn",
        verbose: bool = False,
    ):
        """

        Args:
            nnodes: The number of nodes to use for distributed call.
            nproc_per_node: The number of processes to call on each node.
            master_addr: Master node (rank 0)'s address, should be either the IP address or the hostname of node 0.
            master_port: Master node (rank 0)'s free port.
            node_rank: The rank of the node, this could be set via environment variable "NODE_RANK".
            timeout: Timeout for operations executed against the process group.
            init_method: URL specifying how to initialize the process group.
                Default is "env://" or "file:///d:/a_temp" (windows) if unspecified.
                If ``"no_init"``, the `dist.init_process_group` must be called within the code to be tested.
            backend: The backend to use. Depending on build-time configurations,
                valid values include ``mpi``, ``gloo``, and ``nccl``.
            daemon: the process’s daemon flag.
                When daemon=None, the initial value is inherited from the creating process.
            method: set the method which should be used to start a child process.
                method can be 'fork', 'spawn' or 'forkserver'.
            verbose: whether to print NCCL debug info.
        """
        self.nnodes = int(nnodes)
        self.nproc_per_node = int(nproc_per_node)
        if self.nnodes < 1 or self.nproc_per_node < 1:
            raise ValueError(
                f"number of nodes and processes per node must be >= 1, got {self.nnodes} and {self.nproc_per_node}"
            )
        self.node_rank = int(os.environ.get("NODE_RANK", "0")) if node_rank is None else int(node_rank)
        self.master_addr = master_addr
        self.master_port = np.random.randint(10000, 20000) if master_port is None else master_port

        if backend is None:
            self.backend = "nccl" if torch.distributed.is_nccl_available() and torch.cuda.is_available() else "gloo"
        else:
            self.backend = backend
        self.init_method = init_method
        if self.init_method is None and sys.platform == "win32":
            self.init_method = "file:///d:/a_temp"
        self.timeout = datetime.timedelta(0, timeout)
        self.daemon = daemon
        self.method = method
        self.verbose = verbose

    def run_process(self, func, local_rank, args, kwargs, results):
        _env = os.environ.copy()  # keep the original system env
        try:
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port)
            os.environ["LOCAL_RANK"] = str(local_rank)
            if self.verbose:
                os.environ["NCCL_DEBUG"] = "INFO"
                os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = str(1)
            os.environ["OMP_NUM_THREADS"] = str(1)
            os.environ["WORLD_SIZE"] = str(self.nproc_per_node * self.nnodes)
            os.environ["RANK"] = str(self.nproc_per_node * self.node_rank + local_rank)

            if torch.cuda.is_available():
                torch.cuda.set_device(int(local_rank))  # using device ids from CUDA_VISIBILE_DEVICES

            if self.init_method != "no_init":
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    timeout=self.timeout,
                    world_size=int(os.environ["WORLD_SIZE"]),
                    rank=int(os.environ["RANK"]),
                )
            func(*args, **kwargs)
            # the primary node lives longer to
            # avoid _store_based_barrier, RuntimeError: Broken pipe
            # as the TCP store daemon is on the rank 0
            if int(os.environ["RANK"]) == 0:
                time.sleep(0.1)
            results.put(True)
        except Exception as e:
            results.put(False)
            raise e
        finally:
            os.environ.clear()
            os.environ.update(_env)
            try:
                dist.destroy_process_group()
            except RuntimeError as e:
                warnings.warn(f"While closing process group: {e}.")

    def __call__(self, obj):
        if not torch.distributed.is_available():
            return unittest.skipIf(True, "Skipping distributed tests because not torch.distributed.is_available()")(obj)
        if torch.cuda.is_available() and torch.cuda.device_count() < self.nproc_per_node:
            return unittest.skipIf(
                True,
                f"Skipping distributed tests because it requires {self.nnodes} devices "
                f"but got {torch.cuda.device_count()}",
            )(obj)

        _cache_original_func(obj)

        @functools.wraps(obj)
        def _wrapper(*args, **kwargs):
            tmp = torch.multiprocessing.get_context(self.method)
            processes = []
            results = tmp.Queue()
            func = _call_original_func
            args = [obj.__name__, obj.__module__] + list(args)
            for proc_rank in range(self.nproc_per_node):
                p = tmp.Process(
                    target=self.run_process, args=(func, proc_rank, args, kwargs, results), daemon=self.daemon
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                assert results.get(), "Distributed call failed."
            _del_original_func(obj)

        return _wrapper


class TimedCall:
    """
    Wrap a test case so that it will run in a new process, raises a TimeoutError if the decorated method takes
    more than `seconds` to finish. It is designed to be used with `tests.utils.DistTestCase`.
    """

    def __init__(
        self,
        seconds: float = 60.0,
        daemon: bool | None = None,
        method: str | None = "spawn",
        force_quit: bool = True,
        skip_timing=False,
    ):
        """

        Args:
            seconds: timeout seconds.
            daemon: the process’s daemon flag.
                When daemon=None, the initial value is inherited from the creating process.
            method: set the method which should be used to start a child process.
                method can be 'fork', 'spawn' or 'forkserver'.
            force_quit: whether to terminate the child process when `seconds` elapsed.
            skip_timing: whether to skip the timing constraint.
                this is useful to include some system conditions such as
                `torch.cuda.is_available()`.
        """
        self.timeout_seconds = seconds
        self.daemon = daemon
        self.force_quit = force_quit
        self.skip_timing = skip_timing
        self.method = method

    @staticmethod
    def run_process(func, args, kwargs, results):
        try:
            output = func(*args, **kwargs)
            results.put(output)
        except Exception as e:
            e.traceback = traceback.format_exc()
            results.put(e)

    def __call__(self, obj):
        if self.skip_timing:
            return obj

        _cache_original_func(obj)

        @functools.wraps(obj)
        def _wrapper(*args, **kwargs):
            tmp = torch.multiprocessing.get_context(self.method)
            func = _call_original_func
            args = [obj.__name__, obj.__module__] + list(args)
            results = tmp.Queue()
            p = tmp.Process(target=TimedCall.run_process, args=(func, args, kwargs, results), daemon=self.daemon)
            p.start()

            p.join(timeout=self.timeout_seconds)

            timeout_error = None
            try:
                if p.is_alive():
                    # create an Exception
                    timeout_error = torch.multiprocessing.TimeoutError(
                        f"'{obj.__name__}' in '{obj.__module__}' did not finish in {self.timeout_seconds}s."
                    )
                    if self.force_quit:
                        p.terminate()
                    else:
                        warnings.warn(
                            f"TimedCall: deadline ({self.timeout_seconds}s) "
                            f"reached but waiting for {obj.__name__} to finish."
                        )
            finally:
                p.join()

            _del_original_func(obj)
            res = None
            try:
                res = results.get(block=False)
            except queue.Empty:  # no result returned, took too long
                pass
            if isinstance(res, Exception):  # other errors from obj
                if hasattr(res, "traceback"):
                    raise RuntimeError(res.traceback) from res
                raise res
            if timeout_error:  # no force_quit finished
                raise timeout_error
            return res

        return _wrapper


_original_funcs = {}


def _cache_original_func(obj) -> None:
    """cache the original function by name, so that the decorator doesn't shadow it."""
    _original_funcs[obj.__name__] = obj


def _del_original_func(obj):
    """pop the original function from cache."""
    _original_funcs.pop(obj.__name__, None)
    if torch.cuda.is_available():  # clean up the cached function
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _call_original_func(name, module, *args, **kwargs):
    if name not in _original_funcs:
        _original_module = importlib.import_module(module)  # reimport, refresh _original_funcs
        if not hasattr(_original_module, name):
            # refresh module doesn't work
            raise RuntimeError(f"Could not recover the original {name} from {module}: {_original_funcs}.")
    f = _original_funcs[name]
    return f(*args, **kwargs)


class NumpyImageTestCase2D(unittest.TestCase):
    im_shape = (128, 64)
    input_channels = 1
    output_channels = 4
    num_classes = 3

    def setUp(self):
        im, msk = create_test_image_2d(
            self.im_shape[0], self.im_shape[1], num_objs=4, rad_max=20, noise_max=0.0, num_seg_classes=self.num_classes
        )

        self.imt = im[None, None]
        self.seg1 = (msk[None, None] > 0).astype(np.float32)
        self.segn = msk[None, None]


class TorchImageTestCase2D(NumpyImageTestCase2D):

    def setUp(self):
        NumpyImageTestCase2D.setUp(self)
        self.imt = torch.tensor(self.imt)
        self.seg1 = torch.tensor(self.seg1)
        self.segn = torch.tensor(self.segn)


class NumpyImageTestCase3D(unittest.TestCase):
    im_shape = (64, 48, 80)
    input_channels = 1
    output_channels = 4
    num_classes = 3

    def setUp(self):
        im, msk = create_test_image_3d(
            self.im_shape[0],
            self.im_shape[1],
            self.im_shape[2],
            num_objs=4,
            rad_max=20,
            noise_max=0.0,
            num_seg_classes=self.num_classes,
        )

        self.imt = im[None, None]
        self.seg1 = (msk[None, None] > 0).astype(np.float32)
        self.segn = msk[None, None]


class TorchImageTestCase3D(NumpyImageTestCase3D):

    def setUp(self):
        NumpyImageTestCase3D.setUp(self)
        self.imt = torch.tensor(self.imt)
        self.seg1 = torch.tensor(self.seg1)
        self.segn = torch.tensor(self.segn)


def test_script_save(net, *inputs, device=None, rtol=1e-4, atol=0.0):
    """
    Test the ability to save `net` as a Torchscript object, reload it, and apply inference. The value `inputs` is
    forward-passed through the original and loaded copy of the network and their results returned.
    The forward pass for both is done without gradient accumulation.

    The test will be performed with CUDA if available, else CPU.
    """
    # TODO: would be nice to use GPU if available, but it currently causes CI failures.
    device = "cpu"
    with tempfile.TemporaryDirectory() as tempdir:
        convert_to_torchscript(
            model=net,
            filename_or_obj=os.path.join(tempdir, "model.ts"),
            verify=True,
            inputs=inputs,
            device=device,
            rtol=rtol,
            atol=atol,
        )


def test_onnx_save(net, *inputs, device=None, rtol=1e-4, atol=0.0):
    """
    Test the ability to save `net` in ONNX format, reload it and validate with runtime.
    The value `inputs` is forward-passed through the `net` without gradient accumulation
    to do onnx export and PyTorch inference.
    PyTorch model inference is performed with CUDA if available, else CPU.
    Saved ONNX model is validated with onnxruntime, if available, else ONNX native implementation.
    """
    # TODO: would be nice to use GPU if available, but it currently causes CI failures.
    device = "cpu"
    _, has_onnxruntime = optional_import("onnxruntime")
    with tempfile.TemporaryDirectory() as tempdir:
        convert_to_onnx(
            model=net,
            filename=os.path.join(tempdir, "model.onnx"),
            verify=True,
            inputs=inputs,
            device=device,
            use_ort=has_onnxruntime,
            rtol=rtol,
            atol=atol,
        )


def download_url_or_skip_test(*args, **kwargs):
    """``download_url`` and skip the tests if any downloading error occurs."""
    with skip_if_downloading_fails():
        download_url(*args, **kwargs)


def query_memory(n=2):
    """
    Find best n idle devices and return a string of device ids using the `nvidia-smi` command.
    """
    bash_string = "nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used --format=csv,noheader,nounits"

    try:
        print(f"query memory with n={n}")
        p1 = Popen(bash_string.split(), stdout=PIPE)
        output, error = p1.communicate()
        free_memory = [x.split(",") for x in output.decode("utf-8").split("\n")[:-1]]
        free_memory = np.asarray(free_memory, dtype=float).T
        free_memory[1] += free_memory[0]  # combine 0/1 column measures
        ids = np.lexsort(free_memory)[:n]
    except (TypeError, ValueError, IndexError, OSError):
        ids = range(n) if isinstance(n, int) else []
    return ",".join(f"{int(x)}" for x in ids)


def test_local_inversion(invertible_xform, to_invert, im, dict_key=None):
    """test that invertible_xform can bring to_invert back to im"""
    im_item = im if dict_key is None else im[dict_key]
    if not isinstance(im_item, MetaTensor):
        return
    im_ref = copy.deepcopy(im)
    im_inv = invertible_xform.inverse(to_invert)
    if dict_key:
        im_inv = im_inv[dict_key]
        im_ref = im_ref[dict_key]
    np.testing.assert_array_equal(im_inv.applied_operations, [])
    assert_allclose(im_inv.shape, im_ref.shape)
    assert_allclose(im_inv.affine, im_ref.affine, atol=1e-3, rtol=1e-3)


def command_line_tests(cmd, copy_env=True):
    test_env = os.environ.copy() if copy_env else os.environ
    print(f"CUDA_VISIBLE_DEVICES in {__file__}", test_env.get("CUDA_VISIBLE_DEVICES"))
    try:
        normal_out = subprocess.run(cmd, env=test_env, check=True, capture_output=True)
        print(repr(normal_out).replace("\\n", "\n").replace("\\t", "\t"))
        return repr(normal_out)
    except subprocess.CalledProcessError as e:
        output = repr(e.stdout).replace("\\n", "\n").replace("\\t", "\t")
        errors = repr(e.stderr).replace("\\n", "\n").replace("\\t", "\t")
        raise RuntimeError(f"subprocess call error {e.returncode}: {errors}, {output}") from e


def equal_state_dict(st_1, st_2):
    """
    assert equal state_dict (for the shared keys between st_1 and st_2).
    """
    for key_st_1, val_st_1 in st_1.items():
        if key_st_1 in st_2:
            val_st_2 = st_2.get(key_st_1)
            assert_allclose(val_st_1, val_st_2)


TEST_TORCH_TENSORS: tuple = (torch.as_tensor,)
if torch.cuda.is_available():
    gpu_tensor: Callable = partial(torch.as_tensor, device="cuda")
    TEST_TORCH_TENSORS = TEST_TORCH_TENSORS + (gpu_tensor,)

DEFAULT_TEST_AFFINE = torch.tensor(
    [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
_metatensor_creator = partial(MetaTensor, meta={"a": "b", "affine": DEFAULT_TEST_AFFINE})
TEST_NDARRAYS_NO_META_TENSOR: tuple[Callable] = (np.array,) + TEST_TORCH_TENSORS
TEST_NDARRAYS: tuple[Callable] = TEST_NDARRAYS_NO_META_TENSOR + (_metatensor_creator,)  # type: ignore
TEST_TORCH_AND_META_TENSORS: tuple[Callable] = TEST_TORCH_TENSORS + (_metatensor_creator,)
# alias for branch tests
TEST_NDARRAYS_ALL = TEST_NDARRAYS

TEST_DEVICES = [[torch.device("cpu")]]
if torch.cuda.is_available():
    TEST_DEVICES.append([torch.device("cuda")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="util")
    parser.add_argument("-c", "--count", default=2, help="max number of gpus")
    args = parser.parse_args()
    print("\n", query_memory(int(args.count)), sep="\n")  # print to stdout
    sys.exit(0)
