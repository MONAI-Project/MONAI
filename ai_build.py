from __future__ import annotations

import glob
import os
import re
import sys
import warnings
import platform
from packaging import version
from setuptools import find_packages, setup

import versioneer
import logging
from datetime import datetime

# Initialize logging for AI-driven insights
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AI-Driven Feature: Log system information for better environment understanding
logger.info(f"Running setup on {platform.system()} ({platform.release()}) with Python {platform.python_version()}")

# AI-Driven Feature: Automatic debug mode detection
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"
if DEBUG_MODE:
    logger.info("Debug mode enabled. Compiling with -O0 and including test cases.")
else:
    logger.info("Debug mode disabled.")

RUN_BUILD = os.getenv("BUILD_MONAI", "0") == "1"
FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"  # flag ignored if BUILD_MONAI is False

BUILD_CPP = BUILD_CUDA = False
TORCH_VERSION = 0
try:
    import torch

    logger.info(f"setup.py with torch {torch.__version__}")
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    BUILD_CPP = True
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    BUILD_CUDA = FORCE_CUDA or (torch.cuda.is_available() and (CUDA_HOME is not None))

    _pt_version = version.parse(torch.__version__).release
    if _pt_version is None or len(_pt_version) < 3:
        raise AssertionError("unknown torch version")
    TORCH_VERSION = int(_pt_version[0]) * 10000 + int(_pt_version[1]) * 100 + int(_pt_version[2])

    # AI-Driven Feature: Suggest optimization flags based on PyTorch version
    if TORCH_VERSION >= 11000:
        logger.info("Torch version is 1.10.0 or higher. Consider enabling advanced optimizations.")

except (ImportError, TypeError, AssertionError, AttributeError) as e:
    warnings.warn(f"extension build skipped: {e}")
    logger.warning("Falling back to non-Cpp/CUDA build.")
finally:
    if not RUN_BUILD:
        BUILD_CPP = BUILD_CUDA = False
        logger.info("Please set environment variable `BUILD_MONAI=1` to enable Cpp/CUDA extension build.")
    logger.info(f"BUILD_MONAI_CPP={BUILD_CPP}, BUILD_MONAI_CUDA={BUILD_CUDA}, TORCH_VERSION={TORCH_VERSION}.")


def torch_parallel_backend():
    try:
        match = re.search("^ATen parallel backend: (?P<backend>.*)$", torch._C._parallel_info(), re.MULTILINE)
        if match is None:
            return None
        backend = match.group("backend")
        if backend == "OpenMP":
            return "AT_PARALLEL_OPENMP"
        if backend == "native thread pool":
            return "AT_PARALLEL_NATIVE"
        if backend == "native thread pool and TBB":
            return "AT_PARALLEL_NATIVE_TBB"
    except (NameError, AttributeError):  # no torch or no binaries
        warnings.warn("Could not determine torch parallel_info.")
        logger.warning("Torch parallel_info could not be determined.")
    return None


def omp_flags():
    if sys.platform == "win32":
        return ["/openmp"]
    if sys.platform == "darwin":
        return []
    return ["-fopenmp"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(this_dir, "monai", "csrc")
    include_dirs = [ext_dir]

    source_cpu = glob.glob(os.path.join(ext_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)

    extension = None
    define_macros = [(f"{torch_parallel_backend()}", 1), ("MONAI_TORCH_VERSION", TORCH_VERSION)]
    extra_compile_args = {}
    extra_link_args = []
    sources = source_cpu

    # AI-Driven Feature: Dynamic suggestion of compilation strategy
    if not sources:
        logger.info("No source files found. Skipping compilation.")
        return []

    if BUILD_CPP:
        extension = CppExtension
        extra_compile_args.setdefault("cxx", [])
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
        extra_link_args = omp_flags()
        logger.info("Building with C++ extensions.")
    if BUILD_CUDA:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args = {"cxx": [], "nvcc": []}
        if torch_parallel_backend() == "AT_PARALLEL_OPENMP":
            extra_compile_args["cxx"] += omp_flags()
        logger.info("Building with CUDA extensions.")
    if extension is None or not sources:
        logger.info("No Cpp/CUDA sources available, no extensions will be built.")
        return []  # compile nothing

    ext_modules = [
        extension(
            name="monai._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


def get_cmds():
    cmds = versioneer.get_cmdclass()

    if not (BUILD_CPP or BUILD_CUDA):
        return cmds

    cmds.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})
    return cmds


# Gathering source used for JIT extensions to include in package_data.
jit_extension_source = []

for ext in ["cpp", "cu", "h", "cuh"]:
    glob_path = os.path.join("monai", "_extensions", "**", f"*.{ext}")
    jit_extension_source += glob.glob(glob_path, recursive=True)

jit_extension_source = [os.path.join("..", path) for path in jit_extension_source]

setup(
    version=versioneer.get_version(),
    cmdclass=get_cmds(),
    packages=find_packages(exclude=("docs", "examples", "tests")),
    zip_safe=False,
    package_data={"monai": ["py.typed", *jit_extension_source]},
    ext_modules=get_extensions(),
)

# AI-Driven Feature: Post-setup analysis
logger.info("Setup script completed successfully.")
logger.info("If you encountered any issues, please consult the AI-driven suggestions provided above.")
