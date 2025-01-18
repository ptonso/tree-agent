#!/bin/bash
set -e  # Exit on any error

echo "=== DeepMind Lab Initial Setup Script ==="

# Step 1: Install dependencies
echo "Installing dependencies..."
sudo apt update
sudo apt install -y build-essential libosmesa6-dev libgl1-mesa-dev \
    libxi-dev libxcursor-dev libxinerama-dev libxrandr-dev libopenal-dev \
    mesa-utils python3-dev python3-venv git wget unzip zlib1g-dev g++ pip

# Step 2: Install Bazelisk
echo "Installing Bazelisk..."
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 -O bazelisk
chmod +x bazelisk
sudo mv bazelisk /usr/local/bin/bazel

# Verify Bazel installation
bazel --version

# Step 3: Clone DeepMind Lab
echo "Cloning DeepMind Lab repository..."
git clone https://github.com/google-deepmind/lab.git
cd lab

# Step 4: Configure bazelrc
echo "Configuring .bazelrc..."
echo "build --enable_workspace" > .bazelrc
echo "build --python_version=PY3" >> .bazelrc
echo "build --action_env=PYTHON_BIN_PATH=$(which python3)" >> .bazelrc

# Step 5: Modify BUILD file
echo "Modifying BUILD file..."
sed -i '/\[py_binary(/,/\]]/c\
py_binary(\
    name = "python_game_py3",\
    srcs = ["examples/game_main.py"],\
    data = ["//:deepmind_lab.so"],\
    main = "examples/game_main.py",\
    python_version = "PY3",\
    srcs_version = "PY3",\
    tags = ["manual"],\
    deps = ["@six_archive//:six"],\
)' BUILD

# Step 6: Replace content of python_system.bzl
echo "Replacing python_system.bzl content..."
cat > python_system.bzl << 'EOL'
# Copyright 2021 DeepMind Technologies Limited.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# ============================================================================

"""Generates a local repository that points at the system's Python installation."""

_BUILD_FILE = '''# Description:
#   Build rule for Python

exports_files(["defs.bzl"])


cc_library(
    name = "python_headers",
    hdrs = glob(["python3/**/*.h", "numpy3/**/*.h"]),
    includes = ["python3", "numpy3"],
    visibility = ["//visibility:public"],
)
'''

_GET_PYTHON_INCLUDE_DIR = """
import sys
from distutils.sysconfig import get_python_inc
from numpy import get_include
sys.stdout.write("{}\\n{}\\n".format(get_python_inc(), get_include()))
""".strip()

def _python_repo_impl(repository_ctx):
    """Creates external/<reponame>/BUILD, a python3 symlink, and other files."""

    repository_ctx.file("BUILD", _BUILD_FILE)

    result = repository_ctx.execute(["python3", "-c", _GET_PYTHON_INCLUDE_DIR])
    if result.return_code:
        fail("Failed to run local Python3 interpreter: %s" % result.stderr)
    pypath, nppath = result.stdout.splitlines()
    repository_ctx.symlink(pypath, "python3")
    repository_ctx.symlink(nppath, "numpy3")


python_repo = repository_rule(
    implementation = _python_repo_impl,
    configure = True,
    local = True,
    attrs = {"py_version": attr.string(default = "PY3", values = ["PY3"])},
)
EOL

echo "Initial setup complete! Now run build_venv.sh to create virtual environment and build the project."