#!/bin/bash
# Installation script for CUDA-enabled JAX and dependencies

set -e

# Activate virtual environment if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/thesis/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/thesis/bin/activate"
elif [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "Warning: No virtual environment found. Installing to user site-packages."
fi

echo "Installing dependencies with CUDA support..."
echo "Using Python: $(which python)"
echo "Using pip: $(which pip)"

# Install numpy and numpyro first
pip install "numpy>=1.21.0" "numpyro>=0.13.0"

# Detect CUDA version (if available)
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Detected CUDA version: $CUDA_VERSION"
    
    if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 1 ]]; then
        echo "Installing JAX with CUDA 12.x support..."
        pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    elif [[ $(echo "$CUDA_VERSION >= 11.0" | bc -l) -eq 1 ]]; then
        echo "Installing JAX with CUDA 11.x support..."
        pip install "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        echo "CUDA version $CUDA_VERSION not supported. Installing CPU version."
        pip install jax jaxlib
    fi
else
    echo "nvcc not found. Please install CUDA toolkit first, or install CPU version:"
    echo "  pip install jax jaxlib"
    exit 1
fi

echo ""
echo "Installation complete!"
echo ""
echo "To use the virtual environment, run:"
echo "  source activate_venv.sh"
echo ""
echo "Or manually activate and set PYTHONPATH:"
if [ -f "$SCRIPT_DIR/thesis/bin/activate" ]; then
    echo "  source thesis/bin/activate"
    echo "  export PYTHONPATH=\"\$SCRIPT_DIR/thesis/lib/python3.13/site-packages:\$PYTHONPATH\""
fi
echo ""
echo "To verify CUDA support:"
echo "  python check_install.py"
echo "  # or"
echo "  python -c 'import jax; print(\"JAX version:\", jax.__version__); print(\"Devices:\", jax.devices())'"
echo "  python -c 'import numpyro; print(\"NumPyro version:\", numpyro.__version__)'"

