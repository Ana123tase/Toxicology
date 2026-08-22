### 1. Mount Drive + Go to project
```python
from google.colab import drive
drive.mount('/content/drive')

%cd "/content/drive/.shortcut-targets-by-id/1J41KUs-iq0gZzzOE4WVU3dRAczQMJ8O0/Toxicology"
!pwd
!ls -lh
```

### 2. Setup Pythons (only needed once per new VM, skip if `/content/py310` exists)
# Check
!ls /content/py310/bin/python /content/py311/bin/python

# If missing, install:
!apt update -qq
!apt install -y python3.10-venv python3.11-venv -qq
!python3.10 -m venv /content/py310
!python3.11 -m venv /content/py311
!/content/py310/bin/pip install -U pip rdkit chemprop torch --quiet
!/content/py311/bin/pip install -U pip chemprop torch lightning scikit-learn pandas numpy --quiet
!/content/py310/bin/python -c "import rdkit; print('py310 OK')"
!/content/py311/bin/python -c "import chemprop; print('py311 chemprop v2 OK')"
```

### 3. Export chemprop CSVs (if you update raw data)
```python
!/content/py311/bin/python export_chemprop.py
!ls -lh data/*chemprop.csv
```

### 4. Prepare Conformers - ALL endpoints
```python
# This takes hours - runs on CPU
!PYTHONUNBUFFERED=1 /content/py310/bin/python -u prepare_conformers.py --all 2>&1 | tee /content/drive/MyDrive/Toxicology/prepare_conformers.log

# OR per endpoint (safer, if VM dies you keep finished ones):
!PYTHONUNBUFFERED=1 /content/py310/bin/python -u prepare_conformers.py --endpoint DILI
!PYTHONUNBUFFERED=1 /content/py310/bin/python -u prepare_conformers.py --endpoint hERG
!PYTHONUNBUFFERED=1 /content/py310/bin/python -u prepare_conformers.py --endpoint CYP3A4
!PYTHONUNBUFFERED=1 /content/py310/bin/python -u prepare_conformers.py --endpoint Ames
!PYTHONUNBUFFERED=1 /content/py310/bin/python -u prepare_conformers.py --endpoint Teratogenicity
```

### 5. Train D-MPNN Chemeleon - ALL endpoints
```python
# GPU T4 required: Runtime > Change runtime type > T4
!nvidia-smi

# All 5 with ensemble 3 (best)
!PYTHONUNBUFFERED=1 MPLBACKEND=Agg /content/py311/bin/python -u train_dmpnn.py --all --ensemble-size 3 --gpu-batch-size 256 2>&1 | tee /content/drive/MyDrive/Toxicology/training.log

# Per endpoint if you want:
!PYTHONUNBUFFERED=1 MPLBACKEND=Agg /content/py311/bin/python -u train_dmpnn.py --endpoint DILI --ensemble-size 3 --gpu-batch-size 256
```

### 6. Monitor if tab says "Resuming execution"
```python
!uptime
!ps aux | grep -E "prepare_conformers|train_dmpnn" | grep -v grep
!ls -lht checkpoints_dmpnn_v2/ | head -n 20
!ls -lht modules/*metrics.json | tail -n 10
!nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

### 7. Results
```python
!cat modules/*metrics.json
!ls -lh checkpoints_dmpnn_v2/
!ls -lh modules/
```

**Your 2 main running commands:**
- Conformers: `!/content/py310/bin/python prepare_conformers.py --all`
- DMPNN: `!MPLBACKEND=Agg /content/py311/bin/python train_dmpnn.py --all --ensemble-size 3 --gpu-batch-size 256`

Always add `PYTHONUNBUFFERED=1` and `-u` + `| tee /content/drive/MyDrive/Toxicology/*.log` to see live output.