import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Building the E-Brain: High-Speed GPU Mass Training on Kaggle\n",
    "\n",
    "**Author:** Tijil Das  \n",
    "**Project:** E-Brain Handcrafted Neural Architecture  \n",
    "\n",
    "---\n",
    "\n",
    "## Kaggle Environment & Operational Checklist\n",
    "1. **GPU Accelerator:** In the right sidebar, verify `Settings -> Accelerator -> GPU T4 x2` (or `GPU P100`) is selected.\n",
    "2. **Internet Access:** In the right sidebar, verify `Settings -> Internet -> On` (required to clone repo or download libraries).\n",
    "3. **Persistent Output:** Everything in `/kaggle/working` is automatically preserved when using **`Save Version -> Save & Run All (Commit)`**.\n",
    "4. **Resumption:** If a session times out (9–12 hr limit), save the previous run's output as a Kaggle Dataset, mount it in `/kaggle/input/`, and the notebook will automatically resume without losing progress.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Run Configuration (Bump `RUN_ID` before changing any setting)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =====================================================================\n",
    "# MASTER RUN CONFIGURATION\n",
    "# =====================================================================\n",
    "CONFIG = {\n",
    "    'RUN_ID': 'KAGGLE_RUN_001',\n",
    "    'PURPOSE': 'Kaggle Tier 2 (1 MB) GPU Mass Training with Pre-LayerNorm',\n",
    "    'DATASET_PATH': '/kaggle/input/ebrain-tier2-1mb',\n",
    "    'T': 16,               # Sequence / context length\n",
    "    'D': 64,               # Embedding dimension (scaled up from 32)\n",
    "    'N_HEADS': 4,          # Attention heads\n",
    "    'H_MLP': 128,          # Feed-forward hidden dimension\n",
    "    'L': 4,                # Number of transformer blocks (scaled up from 2)\n",
    "    'BATCH_SIZE': 64,      # Parallel GPU mini-batch size\n",
    "    'LR': 0.003,           # Peak learning rate\n",
    "    'WEIGHT_DECAY': 1e-4,  # Weight decay regularization\n",
    "    'DROPOUT': 0.0,        # Zero dropout for compact capacity\n",
    "    'EPOCHS': 30,          # Total epochs\n",
    "    'PATIENCE': 5,         # Early stopping patience\n",
    "    'USE_AMP': True        # Mixed precision (FP16) acceleration\n",
    "}\n",
    "\n",
    "print(\"Active Configuration:\")\n",
    "for k, v in CONFIG.items():\n",
    "    print(f\"  {k:15s}: {v}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Hardware Verification & Repository Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvidia-smi\n",
    "import os, torch\n",
    "print(\"PyTorch Version:\", torch.__version__)\n",
    "print(\"CUDA Available:\", torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print(\"Device Name:   \", torch.cuda.get_device_name(0))\n",
    "    print(\"Total VRAM:    \", f\"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
    "\n",
    "# Working directory structure\n",
    "os.makedirs(\"/kaggle/working/checkpoints\", exist_ok=True)\n",
    "os.makedirs(\"/kaggle/working/logs\", exist_ok=True)\n",
    "print(\"Storage verified: /kaggle/working is ready.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Clone Repository or Sync Core Engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# If using GitHub repository:\n",
    "# !git clone https://github.com/your-username/E-Brain.git\n",
    "# %cd E-Brain/Stage\\ 11\\ Production\\ Engine\n",
    "# !git rev-parse HEAD\n",
    "\n",
    "# Inspect input datasets\n",
    "!ls -la /kaggle/input"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Automated Pre-Flight Verification Checklist\n",
    "Verifies strict causal isolation, initial cross-entropy loss $\\approx \\ln(V)$, and state restoration before starting the GPU run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python -c \"import ebrain_torch; print('E-Brain Engine Core verified!')\"\n",
    "# Execute pre-flight suite\n",
    "!python preflight_checklist.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Launch Training (With Auto-Checkpointing & 15-Minute Saves)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python train_kaggle.py \\\n",
    "    --run_id \"{CONFIG['RUN_ID']}\" \\\n",
    "    --data_dir \"{CONFIG['DATASET_PATH']}\" \\\n",
    "    --T {CONFIG['T']} \\\n",
    "    --D {CONFIG['D']} \\\n",
    "    --n_heads {CONFIG['N_HEADS']} \\\n",
    "    --H_mlp {CONFIG['H_MLP']} \\\n",
    "    --L {CONFIG['L']} \\\n",
    "    --batch_size {CONFIG['BATCH_SIZE']} \\\n",
    "    --lr {CONFIG['LR']} \\\n",
    "    --epochs {CONFIG['EPOCHS']} \\\n",
    "    --patience {CONFIG['PATIENCE']}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Plot Learning Curves (Train Loss vs. Validation Loss)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "csv_path = f\"/kaggle/working/logs/{CONFIG['RUN_ID']}_metrics.csv\"\n",
    "if os.path.exists(csv_path):\n",
    "    df = pd.read_csv(csv_path)\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')\n",
    "    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')\n",
    "    plt.title(f\"E-Brain Loss Curves — {CONFIG['RUN_ID']}\")\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.ylabel('Cross-Entropy Loss')\n",
    "    plt.grid(True, linestyle='--', alpha=0.6)\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "else:\n",
    "    print(f\"Metrics CSV not found at {csv_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Interactive Sample Generation from Best Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json, torch\n",
    "from ebrain_torch import EBrainTransformer\n",
    "\n",
    "# Locate vocab\n",
    "vocab_file = os.path.join(CONFIG['DATASET_PATH'], 'vocab.json')\n",
    "with open(vocab_file) as f:\n",
    "    vocab_data = json.load(f)\n",
    "\n",
    "class Vocab:\n",
    "    def __init__(self, data):\n",
    "        self.itos = data['itos']\n",
    "        self.stoi = data['stoi']\n",
    "        self.unk = data['unk_char']\n",
    "    def encode(self, c): return self.stoi.get(c, self.stoi.get(self.unk, 0))\n",
    "    def decode(self, i): return self.itos[i] if 0 <= i < len(self.itos) else self.unk\n",
    "\n",
    "vocab = Vocab(vocab_data)\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "\n",
    "model = EBrainTransformer(\n",
    "    V=len(vocab.itos), T=CONFIG['T'], D=CONFIG['D'],\n",
    "    n_heads=CONFIG['N_HEADS'], H_mlp=CONFIG['H_MLP'],\n",
    "    L=CONFIG['L'], use_layernorm=True\n",
    ").to(device)\n",
    "\n",
    "best_path = f\"/kaggle/working/checkpoints/{CONFIG['RUN_ID']}_best.pt\"\n",
    "ckpt = torch.load(best_path, map_location=device)\n",
    "model.load_state_dict(ckpt['model_state_dict'])\n",
    "print(f\"Loaded best checkpoint from Epoch {ckpt['epoch']} (Val Loss: {ckpt['val_loss']:.4f})\")\n",
    "\n",
    "prompt = \"Arthur Pendelton \"\n",
    "sample = model.generate(vocab, prompt, gen_len=300, temperature=0.8, device=device)\n",
    "print(\"\\n--- Generated Text (Temperature 0.8) ---\\n\")\n",
    "print(sample)"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "kaggle": {
   "accelerator": "gpu",
   "dataSources": [],
   "isGpuEnabled": True,
   "isInternetEnabled": True,
   "language": "python"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

with open("T:/Coding/E-Brain/Stage 11 Production Engine/EBrain_Kaggle_Master.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("EBrain_Kaggle_Master.ipynb generated successfully.")
