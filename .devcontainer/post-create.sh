#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Step 1: Install corels BEFORE upgrading pip.
#   corels' setup.py uses legacy numpy.distutils which breaks
#   with pip>=24. The base image ships pip 23.x which works fine.
# ---------------------------------------------------------------
echo "=== Installing numpy + corels (with base pip) ==="
pip install numpy==1.26.4
pip install corels==1.1.29

# ---------------------------------------------------------------
# Step 2: Upgrade pip, then install everything else.
# ---------------------------------------------------------------
echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing PyTorch (CPU-only) ==="
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing project requirements ==="
pip install -r requirements.txt

echo "=== Installing DI-engine (editable, no-deps to avoid conflicts) ==="
pip install --no-deps -e ./DI-engine/

echo "=== Installing minimal DI-engine runtime deps ==="
pip install easydict pyyaml cloudpickle tabulate \
    'DI-treetensor>=0.4.0' 'DI-toolkit>=0.1.0'

# ---------------------------------------------------------------
# Step 3: Verify
# ---------------------------------------------------------------
echo "=== Verifying installation ==="
python -c "import gymnasium; env = gymnasium.make('CartPole-v1'); print('Gym OK')"
python -c "import torch; print(f'Torch {torch.__version__} OK')"
python -c "from interpret.glassbox import ExplainableBoostingClassifier; print('EBM OK')"
python -c "from corels import CorelsClassifier; print('CORELS OK')"
python -c "import ding; print('DI-engine OK')"

echo "=== All checks passed ==="
