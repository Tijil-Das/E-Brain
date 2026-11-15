import json

null = None

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Building the E-Brain: High-Speed GPU Mass Training in Google Colab\n",
    "\n",
    "**Author:** Tijil Das  \n",
    "**Project:** E-Brain Handcrafted Neural Architecture  \n",
    "\n",
    "---\n",
    "\n",
    "### Why are we running in Google Colab?\n",
    "In Stage 1 through Stage 10, we derived and built the entire neural network engine from scratch in bare-metal Java. We touched every weight, manually calculated every analytical gradient, verified causality bitwise, and built a 2-block causal Transformer that learns English.\n",
    "\n",
    "However, when scaling the data from 170 KB to **1 MB, 5 MB, and 20 MB+**, CPU training on single threads takes dozens of hours. Colab provides NVIDIA GPUs (T4, V100, A100). Plain Java cannot natively utilize CUDA tensor cores without complex native C/JNI wrappers. \n",
    "\n",
    "To solve this **without sacrificing architectural ownership**, we ported our exact mathematical equations into PyTorch. In our cross-language parity verification (`verify_parity.py`), our PyTorch model produced the **exact same logits and loss down to the 7th decimal place** ($1.24 \\times 10^{-7}$) as our handcrafted Java engine. \n",
    "\n",
    "The architecture remains 100% ours; PyTorch merely acts as our high-speed hardware accelerator.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Verify GPU Hardware Acceleration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvidia-smi\n",
    "import torch\n",
    "print(\"PyTorch Version:\", torch.__version__)\n",
    "print(\"CUDA Available:\", torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print(\"Device Name:   \", torch.cuda.get_device_name(0))\n",
    "    print(\"Total Memory:  \", f\"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Mount Google Drive for Persistent Checkpointing\n",
    "Google Colab sessions can disconnect after inactivity. We automatically mirror every checkpoint and the optimizer state directly to Google Drive so training resumes seamlessly without losing progress."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "!mkdir -p /content/drive/MyDrive/EBrain_Checkpoints"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Clone or Sync the E-Brain Repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# If using GitHub:\n",
    "# !git clone https://github.com/your-username/E-Brain.git\n",
    "# %cd E-Brain/Stage\\ 11\\ Production\\ Engine\n",
    "\n",
    "# Or ensure files are present in working directory\n",
    "!ls -la"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Prepare Corpus & Build Fixed Vocabulary\n",
    "Cleans the text, strips Gutenberg boilerplate, normalizes typography, and creates Train (85%), Val (10%), and Test (5%) splits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python prepare_data.py Text.txt data_tier1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Automated Pre-Flight Checklist & Training Execution\n",
    "Runs the 5-point automated verification checklist before launching GPU training:\n",
    "1. Hardware verification\n",
    "2. Initial loss check (approx ln(V))\n",
    "3. Causality verification\n",
    "4. Overfit-one-batch check\n",
    "5. Checkpoint roundtrip test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python train_colab.py \\\n",
    "    --run_id \"COLAB_GPU_001\" \\\n",
    "    --data_dir \"data_tier1\" \\\n",
    "    --T 16 \\\n",
    "    --D 64 \\\n",
    "    --n_heads 4 \\\n",
    "    --H_mlp 128 \\\n",
    "    --L 4 \\\n",
    "    --batch_size 64 \\\n",
    "    --lr 0.003 \\\n",
    "    --epochs 30 \\\n",
    "    --checkpoint_dir \"/content/drive/MyDrive/EBrain_Checkpoints\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Interactive Text Generation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json, torch\n",
    "from ebrain_torch import EBrainTransformer\n",
    "from train_colab import VocabWrapper\n",
    "\n",
    "with open(\"data_tier1/vocab.json\") as f:\n",
    "    vocab = VocabWrapper(json.load(f))\n",
    "\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "model = EBrainTransformer(V=vocab.size(), T=16, D=64, n_heads=4, H_mlp=128, L=4, use_layernorm=True).to(device)\n",
    "\n",
    "# Load best weights from Google Drive\n",
    "ckpt = torch.load(\"/content/drive/MyDrive/EBrain_Checkpoints/COLAB_GPU_001_best.pt\", map_location=device)\n",
    "model.load_state_dict(ckpt['model_state_dict'])\n",
    "\n",
    "prompt = \"Arthur Pendelton \"\n",
    "sample = model.generate(vocab, prompt, gen_len=300, temperature=0.8, device=device)\n",
    "print(\"\\n--- Generated Text ---\\n\")\n",
    "print(sample)"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
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

with open("T:/Coding/E-Brain/Stage 11 Production Engine/EBrain_Colab_Master.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("EBrain_Colab_Master.ipynb created successfully.")
