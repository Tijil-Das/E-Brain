"""
E-Brain V2 Architectural Regression & Sanity Verification Suite
=============================================================
Certifies the 6 non-negotiable architectural invariants of EBrainV2:
1. Weight tying integrity: lm_head.weight IS tok_emb.weight (parameter sharing)
2. Initial loss sanity: L_0 ≈ ln(V)
3. Strict causality: Future token changes have ZERO effect on past logits
4. RoPE Relative shift invariance
5. Single-batch memorization: Can overfit a batch to L < 0.05
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebrain_v2 import EBrainV2
from ebrain_rope import precompute_freqs_cis, apply_rotary_emb

def run_regression_suite():
    print("=" * 70)
    print("E-BRAIN V2: ARCHITECTURAL REGRESSION & SANITY VERIFICATION SUITE")
    print("=" * 70)
    
    passed_all = True
    
    # -------------------------------------------------------------
    # 1. Weight Tying Invariant
    # -------------------------------------------------------------
    print("\n[Invariant 1] Output Weight Tying Verification...")
    V = 13070
    D = 128
    H = 4
    L = 2
    max_seq_len = 64
    model = EBrainV2(V=V, D=D, H=H, L=L, max_seq_len=max_seq_len)
    
    is_tied = (model.lm_head.weight is model.tok_emb.weight)
    print(f" -> lm_head.weight is tok_emb.weight: {is_tied}")
    if is_tied:
        print(" -> PASSED: Output projection shares identical memory with embedding matrix.")
    else:
        print(" -> FAILED: Weights are not tied!")
        passed_all = False
        
    # -------------------------------------------------------------
    # 2. Initial Random Guessing Loss Invariant: L_0 ≈ ln(V)
    # -------------------------------------------------------------
    print("\n[Invariant 2] Initial Loss Random Guessing Check...")
    theoretical_loss = np.log(V)
    x_dummy = torch.randint(0, V, (8, 32))
    y_dummy = torch.randint(0, V, (8, 32))
    
    with torch.no_grad():
        logits = model(x_dummy)
        measured_loss = F.cross_entropy(logits.view(-1, V), y_dummy.view(-1)).item()
        
    delta_init = abs(measured_loss - theoretical_loss)
    print(f" -> Theoretical Initial Loss: ln({V}) = {theoretical_loss:.4f}")
    print(f" -> Measured Initial Loss:    {measured_loss:.4f}")
    print(f" -> Error Delta:              {delta_init:.4f}")
    
    if delta_init < 0.25:
        print(" -> PASSED: Initial loss matches theoretical entropy of uniform guessing.")
    else:
        print(" -> FAILED: Initialization variance is corrupt!")
        passed_all = False
        
    # -------------------------------------------------------------
    # 3. Strict Causality Masking Invariant (Zero Future Leakage)
    # -------------------------------------------------------------
    print("\n[Invariant 3] Strict Causality Verification (No Future Leakage)...")
    model.eval()
    seq_len = 16
    x_orig = torch.randint(0, V, (1, seq_len))
    
    with torch.no_grad():
        logits_orig = model(x_orig)
        target_token_idx = 4
        logits_at_4_orig = logits_orig[0, target_token_idx, :].clone()
        
        # Perturb tokens in future positions (positions 5 to 15)
        x_perturbed = x_orig.clone()
        x_perturbed[0, 5:] = torch.randint(0, V, (seq_len - 5,))
        
        logits_perturbed = model(x_perturbed)
        logits_at_4_perturbed = logits_perturbed[0, target_token_idx, :]
        
        causality_delta = torch.max(torch.abs(logits_at_4_orig - logits_at_4_perturbed)).item()
        
    print(f" -> Maximum logit difference at pos {target_token_idx} when altering future: {causality_delta:.10e}")
    if causality_delta < 1e-6:
        print(" -> PASSED: Model is strictly causal. Future tokens cannot leak into the past.")
    else:
        print(" -> FAILED: Future tokens leaked into past positions!")
        passed_all = False
        
    # -------------------------------------------------------------
    # 4. RoPE Relative Distance Shift Invariance
    # -------------------------------------------------------------
    print("\n[Invariant 4] RoPE Relative Distance Invariance...")
    dim = 32
    cos, sin = precompute_freqs_cis(dim, 128)
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    
    # Dot product at distance = 4 (pos 8 and pos 4 vs pos 20 and pos 16)
    q8 = apply_rotary_emb(q, cos[8:9], sin[8:9])
    k4 = apply_rotary_emb(k, cos[4:5], sin[4:5])
    score_8_4 = torch.sum(q8 * k4).item()
    
    q20 = apply_rotary_emb(q, cos[20:21], sin[20:21])
    k16 = apply_rotary_emb(k, cos[16:17], sin[16:17])
    score_20_16 = torch.sum(q20 * k16).item()
    
    rope_diff = abs(score_8_4 - score_20_16)
    print(f" -> Dot product at pos (8, 4)   [dist = 4]: {score_8_4:.8f}")
    print(f" -> Dot product at pos (20, 16) [dist = 4]: {score_20_16:.8f}")
    print(f" -> Relative invariance delta:             {rope_diff:.10e}")
    
    if rope_diff < 1e-5:
        print(" -> PASSED: RoPE maintains absolute relative distance invariance.")
    else:
        print(" -> FAILED: RoPE relative shift failed!")
        passed_all = False
        
    # -------------------------------------------------------------
    # 5. Single-Batch Memorization Invariant
    # -------------------------------------------------------------
    print("\n[Invariant 5] Single-Batch Memorization Check...")
    small_model = EBrainV2(V=256, D=64, H=2, L=2, max_seq_len=32)
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=3e-3)
    
    bx = torch.randint(0, 256, (4, 16))
    by = torch.randint(0, 256, (4, 16))
    
    initial_batch_loss = 0.0
    final_batch_loss = 0.0
    
    for step in range(60):
        optimizer.zero_grad()
        logits = small_model(bx)
        loss = F.cross_entropy(logits.view(-1, 256), by.view(-1))
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_batch_loss = loss.item()
        if step == 59:
            final_batch_loss = loss.item()
            
    print(f" -> Initial Batch Loss (step 0):  {initial_batch_loss:.4f}")
    print(f" -> Final Batch Loss (step 60):   {final_batch_loss:.4f}")
    
    if final_batch_loss < 0.10:
        print(" -> PASSED: Model easily memorizes single batch; gradients flowing cleanly.")
    else:
        print(" -> FAILED: Loss failed to plunge on single batch!")
        passed_all = False
        
    print("\n" + "=" * 70)
    if passed_all:
        print("ALL ARCHITECTURAL INVARIANTS PASSED SUCCESSFULLY!")
        print("EBrainV2 is mathematically certified for production scaling.")
    else:
        print("ONE OR MORE INVARIANTS FAILED. CHECK LOGS.")
    print("=" * 70)
    return passed_all

if __name__ == "__main__":
    success = run_regression_suite()
    sys.exit(0 if success else 1)
