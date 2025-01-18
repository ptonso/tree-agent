#!/bin/bash
set -e  # Exit on any error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENVPATH="$SCRIPT_DIR/venv"

if [ ! -d "lab" ]; then
    echo "Error: 'lab' directory not found. Please run setup_lab.sh first."
    exit 1
fi

cd lab

echo "=== DeepMind Lab Virtual Environment Setup ==="
echo "Using virtual environment path: $VENVPATH"

echo "Setting up Python virtual environment at $VENVPATH..."
python3 -m venv "$VENVPATH"
source "$VENVPATH/bin/activate"

pip install --upgrade pip setuptools wheel
pip install numpy dm_env opencv-python

export PYTHON_BIN_PATH=$(which python3)

echo "Building the project..."
bazel clean --expunge
bazel build -c opt //python/pip_package:build_pip_package

echo "Generating wheel file..."
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg

echo "Installing package..."
pip install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl

echo "Moving files to correct locations..."
SITE_PACKAGES="$VENVPATH/lib/python3.10/site-packages/deepmind_lab"
mv "$SITE_PACKAGES/_main/"* "$SITE_PACKAGES/"
rmdir "$SITE_PACKAGES/_main"

cd ..
if [ -d "lab" ]; then
    rm -rf "lab"
fi
echo "Installation complete!"
