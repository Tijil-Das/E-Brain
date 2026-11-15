import os
import struct
import math
import numpy as np
import torch

from ebrain_numpy import EBrainNumpy, Vocab
from ebrain_torch import EBrainTransformer

def read_doubles(f, n):
    raw = f.read(8 * n)
    return struct.unpack(f">{n}d", raw)

def load_java_checkpoint_to_numpy(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 0xEB8A1101, f"Invalid magic: {hex(magic)}"
        V, T, D, n_heads, dk, H_mlp, L = struct.unpack(">7i", f.read(28))
        use_layernorm = struct.unpack(">?", f.read(1))[0]
        adam_t = struct.unpack(">q", f.read(8))[0]

        model = EBrainNumpy(V, T, D, n_heads, H_mlp, L, use_layernorm=use_layernorm)

        # Read C [V, D]
        model.C[:] = np.array(read_doubles(f, V * D)).reshape(V, D)
        # Read P [T, D]
        model.P[:] = np.array(read_doubles(f, T * D)).reshape(T, D)

        for l in range(L):
            b = model.blocks[l]
            b['Wq'][:] = np.array(read_doubles(f, n_heads * D * dk)).reshape(n_heads, D, dk)
            b['Wk'][:] = np.array(read_doubles(f, n_heads * D * dk)).reshape(n_heads, D, dk)
            b['Wv'][:] = np.array(read_doubles(f, n_heads * D * dk)).reshape(n_heads, D, dk)
            b['Wo'][:] = np.array(read_doubles(f, D * D)).reshape(D, D)
            b['W1'][:] = np.array(read_doubles(f, D * H_mlp)).reshape(D, H_mlp)
            b['b1'][:] = np.array(read_doubles(f, H_mlp))
            b['W2'][:] = np.array(read_doubles(f, H_mlp * D)).reshape(H_mlp, D)
            b['b2'][:] = np.array(read_doubles(f, D))
            b['gamma1'][:] = np.array(read_doubles(f, D))
            b['beta1'][:] = np.array(read_doubles(f, D))
            b['gamma2'][:] = np.array(read_doubles(f, D))
            b['beta2'][:] = np.array(read_doubles(f, D))

        model.gammaF[:] = np.array(read_doubles(f, D))
        model.betaF[:] = np.array(read_doubles(f, D))
        model.Wout[:] = np.array(read_doubles(f, D * V)).reshape(D, V)
        model.bout[:] = np.array(read_doubles(f, V))

        return model

def load_numpy_to_torch(np_model):
    torch_model = EBrainTransformer(
        V=np_model.V, T=np_model.T, D=np_model.D,
        n_heads=np_model.n_heads, H_mlp=np_model.H_mlp,
        L=np_model.L, use_layernorm=np_model.use_layernorm, dropout=0.0
    )

    with torch.no_grad():
        torch_model.tok_emb.weight.copy_(torch.from_numpy(np_model.C))
        torch_model.pos_emb.weight.copy_(torch.from_numpy(np_model.P))

        for l in range(np_model.L):
            nb = np_model.blocks[l]
            tb = torch_model.blocks[l]

            # In PyTorch Linear weight is [Out, In].
            # In our handcrafted code Wq[h] is [D, dk], so for full projection [D, D]
            # concat heads along columns: Wq_full = [D, D]
            Wq_full = np.concatenate([nb['Wq'][h] for h in range(np_model.n_heads)], axis=-1)
            Wk_full = np.concatenate([nb['Wk'][h] for h in range(np_model.n_heads)], axis=-1)
            Wv_full = np.concatenate([nb['Wv'][h] for h in range(np_model.n_heads)], axis=-1)

            tb.attn.Wq.weight.copy_(torch.from_numpy(Wq_full.T))
            tb.attn.Wk.weight.copy_(torch.from_numpy(Wk_full.T))
            tb.attn.Wv.weight.copy_(torch.from_numpy(Wv_full.T))
            tb.attn.Wo.weight.copy_(torch.from_numpy(nb['Wo'].T))

            if np_model.use_layernorm:
                tb.ln1.weight.copy_(torch.from_numpy(nb['gamma1']))
                tb.ln1.bias.copy_(torch.from_numpy(nb['beta1']))
                tb.ln2.weight.copy_(torch.from_numpy(nb['gamma2']))
                tb.ln2.bias.copy_(torch.from_numpy(nb['beta2']))

            tb.mlp.W1.weight.copy_(torch.from_numpy(nb['W1'].T))
            tb.mlp.W1.bias.copy_(torch.from_numpy(nb['b1']))
            tb.mlp.W2.weight.copy_(torch.from_numpy(nb['W2'].T))
            tb.mlp.W2.bias.copy_(torch.from_numpy(nb['b2']))

        if np_model.use_layernorm:
            torch_model.ln_f.weight.copy_(torch.from_numpy(np_model.gammaF))
            torch_model.ln_f.bias.copy_(torch.from_numpy(np_model.betaF))

        torch_model.head.weight.copy_(torch.from_numpy(np_model.Wout.T))
        torch_model.head.bias.copy_(torch.from_numpy(np_model.bout))

    return torch_model

def run_parity_verification():
    print("==========================================================================")
    print("PHASE 6: CROSS-LANGUAGE PARITY VERIFICATION (JAVA vs. NUMPY vs. PYTORCH)")
    print("==========================================================================")

    # 1. Read Java reference logits & loss
    with open("java_logits.bin", "rb") as f:
        java_loss = struct.unpack(">d", f.read(8))[0]
        raw_logits = struct.unpack(f">{16 * 68}d", f.read(16 * 68 * 8))
        java_logits = np.array(raw_logits).reshape(16, 68)

    print(f"Java Reference Loss: {java_loss:.8f}")

    # 2. Test NumPy Model
    np_model = load_java_checkpoint_to_numpy("parity_ref.bin")
    test_tokens = [1, 5, 12, 24, 3, 19, 8, 33, 42, 10, 15, 2, 7, 28, 4, 11]
    test_targets = [5, 12, 24, 3, 19, 8, 33, 42, 10, 15, 2, 7, 28, 4, 11, 1]

    np_cache = np_model.forward(test_tokens, test_targets, is_training=False)
    numpy_loss = np_cache['avg_loss']
    numpy_logits = np_cache['logits']

    diff_loss_np = abs(java_loss - numpy_loss)
    max_logit_diff_np = np.max(np.abs(java_logits - numpy_logits))

    print(f"NumPy Model Loss:    {numpy_loss:.8f} (Diff vs Java: {diff_loss_np:.2e})")
    print(f"NumPy Max Logit Diff: {max_logit_diff_np:.2e}")
    np_pass = max_logit_diff_np < 1e-6 and diff_loss_np < 1e-6
    print(f">> Java vs. NumPy Parity: {'PASSED [OK]' if np_pass else 'FAILED'}")

    # 3. Test PyTorch Model
    torch_model = load_numpy_to_torch(np_model)
    torch_model.eval()
    idx_tensor = torch.tensor(test_tokens, dtype=torch.long).unsqueeze(0)
    tgt_tensor = torch.tensor(test_targets, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        torch_logits, torch_loss = torch_model(idx_tensor, tgt_tensor)
        torch_loss_val = torch_loss.item()
        torch_logits_val = torch_logits.squeeze(0).numpy()

    diff_loss_torch = abs(java_loss - torch_loss_val)
    max_logit_diff_torch = np.max(np.abs(java_logits - torch_logits_val))

    print(f"PyTorch Model Loss:  {torch_loss_val:.8f} (Diff vs Java: {diff_loss_torch:.2e})")
    print(f"PyTorch Max Logit Diff: {max_logit_diff_torch:.2e}")
    torch_pass = max_logit_diff_torch < 1e-5 and diff_loss_torch < 1e-5
    print(f">> Java vs. PyTorch Parity: {'PASSED [OK]' if torch_pass else 'FAILED'}")

    print("==========================================================================")
    if np_pass and torch_pass:
        print(">>> MATHEMATICAL PARITY VERIFIED ACROSS JAVA, NUMPY, AND PYTORCH! <<<")
        print("The GPU PyTorch implementation is an exact 1:1 clone of our handcrafted engine.")
    else:
        print(">>> PARITY CHECK FAILED! DIFFERENCE EXCEEDS THRESHOLD. <<<")
    print("==========================================================================")

if __name__ == "__main__":
    run_parity_verification()
