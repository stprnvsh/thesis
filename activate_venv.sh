#!/bin/bash
# Activation script that ensures the venv works correctly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/thesis/bin/activate" ]; then
    source "$SCRIPT_DIR/thesis/bin/activate"
    # Ensure site-packages is in Python path
    export PYTHONPATH="$SCRIPT_DIR/thesis/lib/python3.13/site-packages:$PYTHONPATH"
    echo "✓ Virtual environment activated"
    echo "✓ Python: $(which python)"
    echo "✓ PYTHONPATH includes venv site-packages"
else
    echo "Error: Virtual environment not found at $SCRIPT_DIR/thesis/bin/activate"
    return 1 2>/dev/null || exit 1
fi






