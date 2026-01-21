#!/usr/bin/env python3
"""Quick script to verify JAX and NumPyro installation with CUDA support."""

import sys
import os

# Add venv site-packages to path if not already there
venv_site_packages = os.path.join(os.path.dirname(__file__), 'thesis', 'lib', 'python3.13', 'site-packages')
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    print(f"✓ JAX devices: {jax.devices()}")
    
    import numpyro
    print(f"✓ NumPyro version: {numpyro.__version__}")
    
    print("\n✅ All packages installed correctly with CUDA support!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"\nPython path: {sys.path[:3]}")
    sys.exit(1)






