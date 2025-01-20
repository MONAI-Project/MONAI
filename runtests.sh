#! /bin/bash

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

# script for running all tests
set -e

# output formatting
separator=""
blue=""
green=""
red=""
noColor=""

if [[ -t 1 ]] # stdout is a terminal
then
    separator=$'--------------------------------------------------------------------------------\n'
    blue="$(tput bold; tput setaf 4)"
    green="$(tput bold; tput setaf 2)"
    red="$(tput bold; tput setaf 1)"
    noColor="$(tput sgr0)"
fi

# configuration values
doCoverage=false
doQuickTests=false
doMinTests=false
doNetTests=false
doDryRun=false
doZooTests=false
doUnitTests=false
doBuild=false
doBlackFormat=false
doBlackFix=false
doIsortFormat=false
doIsortFix=false
doFlake8Format=false
doPylintFormat=false
doRuffFormat=false
doRuffFix=false
doClangFormat=false
doCopyRight=false
doPytypeFormat=false
doMypyFormat=false
doCleanup=false
doDistTests=false
doPrecommit=false

NUM_PARALLEL=1

PY_EXE=${MONAI_PY_EXE:-$(which python)}

function print_usage {
    echo "runtests.sh [--codeformat] [--autofix] [--black] [--isort] [--flake8] [--pylint] [--ruff]"
    echo "            [--clangformat] [--precommit] [--pytype] [-j number] [--mypy]"
    echo "            [--unittests] [--disttests] [--coverage] [--quick] [--min] [--net] [--build] [--list_tests]"
    echo "            [--dryrun] [--copyright] [--clean] [--help] [--version] [--path] [--formatfix]"
    echo ""
    echo "MONAI unit testing utilities."
    echo ""
    echo "Examples:"
    echo "./runtests.sh -f -u --net --coverage  # run style checks, full tests, print code coverage (${green}recommended for pull requests${noColor})."
    echo "./runtests.sh -f -u                   # run style checks and unit tests."
    echo "./runtests.sh -f                      # run coding style and static type checking."
    echo "./runtests.sh --quick --unittests     # run minimal unit tests, for quick verification during code developments."
    echo "./runtests.sh --autofix               # run automatic code formatting using \"isort\" and \"black\"."
    echo "./runtests.sh --clean                 # clean up temporary files and run \"${PY_EXE} setup.py develop --uninstall\"."
    echo "./runtests.sh --formatfix -p /my/code # run automatic code formatting using \"isort\" and \"black\" in specified path."
    echo ""
    echo "Code style check options:"
    echo "    --autofix         : format code using \"isort\" and \"black\""
    echo "    --black           : perform \"black\" code format checks"
    echo "    --isort           : perform \"isort\" import sort checks"
    echo "    --flake8          : perform \"flake8\" code format checks"
    echo "    --pylint          : perform \"pylint\" code format checks"
    echo "    --ruff            : perform \"ruff\" code format checks"
    echo "    --clangformat     : format csrc code using \"clang-format\""
    echo "    --precommit       : perform source code format check and fix using \"pre-commit\""
    echo ""
    echo "Python type check options:"
    echo "    --pytype          : perform \"pytype\" static type checks"
    echo "    -j, --jobs        : number of parallel jobs to run \"pytype\" (default $NUM_PARALLEL)"
    echo "    --mypy            : perform \"mypy\" static type checks"
    echo ""
    echo "MONAI unit testing options:"
    echo "    -u, --unittests   : perform unit testing"
    echo "    --disttests       : perform distributed unit testing"
    echo "    --coverage        : report testing code coverage, to be used with \"--net\", \"--unittests\""
    echo "    -q, --quick       : skip long running unit tests and integration tests"
    echo "    -m, --min         : only run minimal unit tests which do not require optional packages"
    echo "    --net             : perform integration testing"
    echo "    -b, --build       : compile and install the source code folder an editable release."
    echo "    --list_tests      : list unit tests and exit"
    echo ""
    echo "Misc. options:"
    echo "    --dryrun          : display the commands to the screen without running"
    echo "    --copyright       : check whether every source code has a copyright header"
    echo "    -f, --codeformat  : shorthand to run all code style and static analysis tests"
    echo "    -c, --clean       : clean temporary files from tests and exit"
    echo "    -h, --help        : show this help message and exit"
    echo "    -v, --version     : show MONAI and system version information and exit"
    echo "    -p, --path        : specify the path used for formatting, default is the current dir if unspecified"
    echo "    --formatfix       : format code using \"isort\" and \"black\" for user specified directories"
    echo ""
    echo "${separator}For bug reports and feature requests, please file an issue at:"
    echo "    https://github.com/Project-MONAI/MONAI/issues/new/choose"
    echo ""
    echo "To choose an alternative python executable, set the environmental variable, \"MONAI_PY_EXE\"."
    exit 1
}

# FIXME: https://github.com/Project-MONAI/MONAI/issues/4354
protobuf_major_version=$("${PY_EXE}" -m pip list | grep '^protobuf ' | tr -s ' ' | cut -d' ' -f2 | cut -d'.' -f1)
if [ "$protobuf_major_version" -ge "4" ]
then
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
fi

function check_import {
    echo "Python: "${PY_EXE}""
    ${cmdPrefix}"${PY_EXE}" -W error -W ignore::DeprecationWarning -W ignore::ResourceWarning -c "import monai"
}

function print_version {
    ${cmdPrefix}"${PY_EXE}" -c 'import monai; monai.config.print_config()'  # project-monai/monai#6167
}

function install_deps {
    echo "Pip installing MONAI development dependencies and compile MONAI cpp extensions..."
    ${cmdPrefix}"${PY_EXE}" -m pip install -r requirements-dev.txt
}

function compile_cpp {
    echo "Compiling and installing MONAI cpp extensions..."
    # depends on setup.py behaviour for building
    # currently setup.py uses environment variables: BUILD_MONAI and FORCE_CUDA
    ${cmdPrefix}"${PY_EXE}" setup.py develop --user --uninstall
    if [[ "$OSTYPE" == "darwin"* ]];
    then  # clang for mac os
        CC=clang CXX=clang++ ${cmdPrefix}"${PY_EXE}" setup.py develop --user
    else
        ${cmdPrefix}"${PY_EXE}" setup.py develop --user
    fi
}

function clang_format {
    echo "Running clang-format..."
    ${cmdPrefix}"${PY_EXE}" -m tests.clang_format_utils
    clang_format_tool='.clang-format-bin/clang-format'
    # Verify .
    if ! type -p "$clang_format_tool" >/dev/null; then
        echo "'clang-format' not found, skipping the formatting."
        exit 1
    fi
    find monai/csrc -type f | while read i; do $clang_format_tool -style=file -i $i; done
    find monai/_extensions -type f -name "*.cpp" -o -name "*.h" -o -name "*.cuh" -o -name "*.cu" |\
        while read i; do $clang_format_tool -style=file -i $i; done
}

function is_pip_installed() {
	return $("${PY_EXE}" -c "import sys, importlib.util; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)" $1)
}

function clean_py {
    if is_pip_installed coverage
    then
      # remove coverage history
      ${cmdPrefix}"${PY_EXE}" -m coverage erase
    fi

    # uninstall the development package
    echo "Uninstalling MONAI development files..."
    ${cmdPrefix}"${PY_EXE}" setup.py develop --user --uninstall

    # remove temporary files (in the directory of this script)
    TO_CLEAN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    echo "Removing temporary files in ${TO_CLEAN}"

    find ${TO_CLEAN}/monai -type f -name "*.py[co]" -delete
    find ${TO_CLEAN}/monai -type f -name "*.so" -delete
    find ${TO_CLEAN}/monai -type d -name "__pycache__" -delete
    find ${TO_CLEAN} -maxdepth 1 -type f -name ".coverage.*" -delete

    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".eggs" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "monai.egg-info" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "build" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "dist" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".mypy_cache" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".pytype" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name ".coverage" -exec rm -r "{}" +
    find ${TO_CLEAN} -depth -maxdepth 1 -type d -name "__pycache__" -exec rm -r "{}" +
}

function torch_validate {
    ${cmdPrefix}"${PY_EXE}" -c 'import torch; print(torch.__version__); print(torch.rand(5,3))'
}

function print_error_msg() {
    echo "${red}Error: $1.${noColor}"
    echo ""
}

function print_style_fail_msg() {
    echo "${red}Check failed!${noColor}"
    if [ "$homedir" = "$currentdir" ]
    then
        echo "Please run auto style fixes: ${green}./runtests.sh --autofix${noColor}"
    else :
    fi
}

function list_unittests() {
    "${PY_EXE}" - << END
import unittest
def print_suite(suite):
    if hasattr(suite, "__iter__"):
        for x in suite:
            print_suite(x)
    else:
        print(suite)
print_suite(unittest.defaultTestLoader.discover('./tests'))
END
    exit 0
}

if [ -z "$1" ]
then
    print_error_msg "Too few arguments to $0"
    print_usage
fi

# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --coverage)
            doCoverage=true
        ;;
        -q|--quick)
            doQuickTests=true
        ;;
        -m|--min)
            doMinTests=true
        ;;
        --net)
            doNetTests=true
        ;;
        --list_tests)
            list_unittests
        ;;
        --dryrun)
            doDryRun=true
        ;;
        -u|--u*)  # allow --unittest | --unittests | --unittesting  etc.
            doUnitTests=true
        ;;
        -f|--codeformat)
            doBlackFormat=true
            doIsortFormat=true
            doFlake8Format=true
            # doPylintFormat=true  # https://github.com/Project-MONAI/MONAI/issues/7094
            doRuffFormat=true
            doCopyRight=true
        ;;
        --disttests)
            doDistTests=true
        ;;
        --black)
            doBlackFormat=true
        ;;
        --autofix)
            doIsortFix=true
            doBlackFix=true
            doRuffFix=true
            doIsortFormat=true
            doBlackFormat=true
            doRuffFormat=true
            doCopyRight=true
        ;;
        --formatfix)
            doIsortFix=true
            doBlackFix=true
            doIsortFormat=true
            doBlackFormat=true
        ;;
        --clangformat)
            doClangFormat=true
        ;;
        --isort)
            doIsortFormat=true
        ;;
        --flake8)
            doFlake8Format=true
        ;;
        --pylint)
            doPylintFormat=true
        ;;
        --ruff)
            doRuffFormat=true
        ;;
        --precommit)
            doPrecommit=true
        ;;
        --pytype)
            doPytypeFormat=true
        ;;
        --mypy)
            doMypyFormat=true
        ;;
        -j|--jobs)
            NUM_PARALLEL=$2
            shift
        ;;
        --copyright)
            doCopyRight=true
        ;;
        -b|--build)
            doBuild=true
        ;;
        -c|--clean)
            doCleanup=true
        ;;
        -h|--help)
            print_usage
        ;;
        -v|--version)
            print_version
            exit 1
        ;;
        --nou*)  # allow --nounittest | --nounittests | --nounittesting  etc.
            print_error_msg "nounittest option is deprecated, no unit tests is the default setting"
            print_usage
        ;;
        -p|--path)
            testdir=$2
            shift
        ;;
        *)
            print_error_msg "Incorrect commandline provided, invalid key: $key"
            print_usage
        ;;
    esac
    shift
done

# home directory
currentdir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -e "$testdir" ]
then
    homedir=$testdir
else
    homedir=$currentdir
fi
echo "Run tests under $homedir"
cd "$homedir"

# python path
export PYTHONPATH="$homedir:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# by default do nothing
cmdPrefix=""

if [ $doDryRun = true ]
then
    echo "${separator}${blue}dryrun${noColor}"

    # commands are echoed instead of ran
    cmdPrefix="dryrun "
    function dryrun { echo "    " "$@"; }
else
    check_import
fi

if [ $doBuild = true ]
then
    echo "${separator}${blue}compile and install${noColor}"
    # try to compile MONAI cpp
    compile_cpp

    echo "${green}done! (to uninstall and clean up, please use \"./runtests.sh --clean\")${noColor}"
fi

if [ $doCleanup = true ]
then
    echo "${separator}${blue}clean${noColor}"

    clean_py

    echo "${green}done!${noColor}"
    exit
fi

if [ $doClangFormat = true ]
then
    echo "${separator}${blue}clang-formatting${noColor}"

    clang_format

    echo "${green}done!${noColor}"
fi

# unconditionally report on the state of monai
print_version

if [ $doCopyRight = true ]
then
    # check copyright headers
    copyright_bad=0
    copyright_all=0
    while read -r fname; do
        copyright_all=$((copyright_all + 1))
        if ! grep "http://www.apache.org/licenses/LICENSE-2.0" "$fname" > /dev/null; then
            print_error_msg "Missing the license header in file: $fname"
            copyright_bad=$((copyright_bad + 1))
        fi
    done <<< "$(find "$(pwd)/monai" "$(pwd)/tests" -type f \
        ! -wholename "*_version.py" -and -name "*.py" -or -name "*.cpp" -or -name "*.cu" -or -name "*.h")"
    if [[ ${copyright_bad} -eq 0 ]];
    then
        echo "${green}Source code copyright headers checked ($copyright_all).${noColor}"
    else
        echo "Please add the licensing header to the file ($copyright_bad of $copyright_all files)."
        echo "  See also: https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md#checking-the-coding-style"
        echo ""
        exit 1
    fi
fi


if [ $doPrecommit = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pre-commit${noColor}"

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed pre_commit
    then
        install_deps
    fi
    ${cmdPrefix}"${PY_EXE}" -m pre_commit run --all-files

    pre_commit_status=$?
    if [ ${pre_commit_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${pre_commit_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doIsortFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doIsortFix = true ]
    then
        echo "${separator}${blue}isort-fix${noColor}"
    else
        echo "${separator}${blue}isort${noColor}"
    fi

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed isort
    then
        install_deps
    fi
    ${cmdPrefix}"${PY_EXE}" -m isort --version

    if [ $doIsortFix = true ]
    then
        ${cmdPrefix}"${PY_EXE}" -m isort "$homedir"
    else
        ${cmdPrefix}"${PY_EXE}" -m isort --check "$homedir"
    fi

    isort_status=$?
    if [ ${isort_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${isort_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doBlackFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doBlackFix = true ]
    then
        echo "${separator}${blue}black-fix${noColor}"
    else
        echo "${separator}${blue}black${noColor}"
    fi

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed black
    then
        install_deps
    fi
    ${cmdPrefix}"${PY_EXE}" -m black --version

    if [ $doBlackFix = true ]
    then
        ${cmdPrefix}"${PY_EXE}" -m black --skip-magic-trailing-comma "$homedir"
    else
        ${cmdPrefix}"${PY_EXE}" -m black --skip-magic-trailing-comma --check "$homedir"
    fi

    black_status=$?
    if [ ${black_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${black_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doFlake8Format = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}flake8${noColor}"

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed flake8
    then
        install_deps
    fi
    ${cmdPrefix}"${PY_EXE}" -m flake8 --version

    ${cmdPrefix}"${PY_EXE}" -m flake8 "$homedir" --count --statistics

    flake8_status=$?
    if [ ${flake8_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${flake8_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi

if [ $doPylintFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pylint${noColor}"

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed pylint
    then
        echo "Pip installing pylint ..."
        ${cmdPrefix}"${PY_EXE}" -m pip install "pylint>2.16,!=3.0.0"
    fi
    ${cmdPrefix}"${PY_EXE}" -m pylint --version

    ignore_codes="C,R,W,E1101,E1102,E0601,E1130,E1123,E0102,E1120,E1137,E1136"
    ${cmdPrefix}"${PY_EXE}" -m pylint monai tests --disable=$ignore_codes -j $NUM_PARALLEL
    pylint_status=$?

    if [ ${pylint_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${pylint_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doRuffFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    if [ $doRuffFix = true ]
    then
        echo "${separator}${blue}ruff-fix${noColor}"
    else
        echo "${separator}${blue}ruff${noColor}"
    fi

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed ruff
    then
        install_deps
    fi
    ruff --version

    if [ $doRuffFix = true ]
    then
        ruff check --fix "$homedir"
    else
        ruff check "$homedir"
    fi

    ruff_status=$?
    if [ ${ruff_status} -ne 0 ]
    then
        print_style_fail_msg
        exit ${ruff_status}
    else
        echo "${green}passed!${noColor}"
    fi
    set -e # enable exit on failure
fi


if [ $doPytypeFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}pytype${noColor}"
    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed pytype
    then
        install_deps
    fi
    pytype_ver=$(${cmdPrefix}"${PY_EXE}" -m pytype --version)
    if [[ "$OSTYPE" == "darwin"* && "$pytype_ver" == "2021."* ]]; then
        echo "${red}pytype not working on macOS 2021 (https://github.com/Project-MONAI/MONAI/issues/2391). Please upgrade to 2022*.${noColor}"
        exit 1
    else
        ${cmdPrefix}"${PY_EXE}" -m pytype --version

        ${cmdPrefix}"${PY_EXE}" -m pytype -j ${NUM_PARALLEL} --python-version="$(${PY_EXE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")" "$homedir"

        pytype_status=$?
        if [ ${pytype_status} -ne 0 ]
        then
            echo "${red}failed!${noColor}"
            exit ${pytype_status}
        else
            echo "${green}passed!${noColor}"
        fi
    fi
    set -e # enable exit on failure
fi


if [ $doMypyFormat = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}mypy${noColor}"

    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed mypy
    then
        install_deps
    fi
    ${cmdPrefix}"${PY_EXE}" -m mypy --version
    ${cmdPrefix}"${PY_EXE}" -m mypy "$homedir"

    mypy_status=$?
    if [ ${mypy_status} -ne 0 ]
    then
        : # mypy output already follows format
        exit ${mypy_status}
    else
        : # mypy output already follows format
    fi
    set -e # enable exit on failure
fi


# testing command to run
cmd="${PY_EXE}"

# When running --quick, require doCoverage as well and set QUICKTEST environmental
# variable to disable slow unit tests from running.
if [ $doQuickTests = true ]
then
    echo "${separator}${blue}quick${noColor}"
    doCoverage=true
    export QUICKTEST=True
fi

if [ $doMinTests = true ]
then
    echo "${separator}${blue}min${noColor}"
    doCoverage=false
    ${cmdPrefix}"${PY_EXE}" -m tests.min_tests
fi

# set coverage command
if [ $doCoverage = true ]
then
    echo "${separator}${blue}coverage${noColor}"
    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed coverage
    then
        install_deps
    fi
    cmd=""${PY_EXE}" -m coverage run --append"
fi

# # download test data if needed
# if [ ! -d testing_data ] && [ "$doDryRun" != 'true' ]
# then
# fi

# unit tests
if [ $doUnitTests = true ]
then
    echo "${separator}${blue}unittests${noColor}"
    torch_validate
    ${cmdPrefix}${cmd} ./tests/runner.py -p "^(?!test_integration).*(?<!_dist)$"  # excluding integration/dist tests
fi

# distributed test only
if [ $doDistTests = true ]
then
    echo "${separator}${blue}run distributed unit test cases${noColor}"
    torch_validate
    for i in tests/test_*_dist.py
    do
        echo "$i"
        ${cmdPrefix}${cmd} "$i"
    done
fi

# network training/inference/eval integration tests
if [ $doNetTests = true ]
then
    set +e  # disable exit on failure so that diagnostics can be given on failure
    echo "${separator}${blue}integration${noColor}"
    for i in tests/integration/*.py
    do
        echo "$i"
        ${cmdPrefix}${cmd} "$i"
    done
    set -e # enable exit on failure
fi

# run model zoo tests
if [ $doZooTests = true ]
then
    echo "${separator}${blue}zoo${noColor}"
    print_error_msg "--zoo option not yet implemented"
    exit 255
fi

# report on coverage
if [ $doCoverage = true ]
then
    echo "${separator}${blue}coverage${noColor}"
    # ensure that the necessary packages for code format testing are installed
    if ! is_pip_installed coverage
    then
        install_deps
    fi
    ${cmdPrefix}"${PY_EXE}" -m coverage combine --append .coverage/
    ${cmdPrefix}"${PY_EXE}" -m coverage report --ignore-errors
fi
