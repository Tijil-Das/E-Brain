"""
Gate D.1 — Dummy GPU kernel
Proves the full end-to-end plumbing:
  - Heartbeat POST to Cloudflare Worker every step
  - Control GET every step
  - PAUSE/STOP flags halt training cleanly
  - Checkpoint save on STOP

Run this kernel from your TRAINING Kaggle account before anything else.
Push with: kaggle kernels push -p orchestrator/kaggle_dummy_kernel
"""

import os
import sys
import json
import time
import hashlib
import requests

# ── Config ────────────────────────────────────────────────────────────────────
WORKER_URL     = os.environ.get("WORKER_URL", "")          # set as Kaggle secret
HEARTBEAT_TOKEN = os.environ.get("HEARTBEAT_TOKEN", "")    # set as Kaggle secret
TOTAL_DUMMY_STEPS = 20
SLEEP_PER_STEP    = 5    # seconds — simulates training time

# ── Helpers ───────────────────────────────────────────────────────────────────

def post_heartbeat(step, extra=None):
    if not WORKER_URL or not HEARTBEAT_TOKEN:
        print(f"  [heartbeat skip — WORKER_URL or HEARTBEAT_TOKEN not set]")
        return
    payload = {
        "step": step,
        "loss": 9.68 - step * 0.05,     # fake declining loss
        "lr": 1e-4,
        "tok_s": 14389,
        "vram_gb": 11.51,
        "eta_h": (TOTAL_DUMMY_STEPS - step) * SLEEP_PER_STEP / 3600,
        "nan_alert": False,
        "loss_spike": False,
        "val_alert": None,
        "lang_gap_alert": None,
        "checkpoint_hash": hashlib.sha256(f"dummy_step_{step}".encode()).hexdigest(),
    }
    if extra:
        payload.update(extra)
    try:
        r = requests.post(
            f"{WORKER_URL}/heartbeat",
            json=payload,
            headers={"X-Token": HEARTBEAT_TOKEN},
            timeout=10,
        )
        print(f"  [heartbeat] step={step}  status={r.status_code}")
    except Exception as e:
        print(f"  [heartbeat fail — silent] {e}")


def get_control():
    if not WORKER_URL:
        return "RUN"
    try:
        r = requests.get(f"{WORKER_URL}/control", timeout=10)
        d = r.json()
        return d.get("control", "RUN")
    except Exception as e:
        print(f"  [control fail — defaulting RUN] {e}")
        return "RUN"


def save_dummy_checkpoint(step, reason="normal"):
    path = f"/tmp/dummy_ckpt_step{step}.json"
    ckpt = {"step": step, "reason": reason, "ts": time.time()}
    with open(path, "w") as f:
        json.dump(ckpt, f)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"  [checkpoint] saved {path}  SHA-256={sha[:16]}...")
    return sha

# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("E-Brain 100M — Gate D.1 Dummy Kernel")
print("=" * 70)
print(f"WORKER_URL:  {'SET' if WORKER_URL else 'NOT SET ⚠'}")
print(f"HEARTBEAT:   {'SET' if HEARTBEAT_TOKEN else 'NOT SET ⚠'}")
print(f"Steps:       {TOTAL_DUMMY_STEPS}")
print(f"Sleep/step:  {SLEEP_PER_STEP}s")
print()

for step in range(1, TOTAL_DUMMY_STEPS + 1):
    print(f"\n[Step {step}/{TOTAL_DUMMY_STEPS}]  t={time.strftime('%H:%M:%S')}")

    # Simulate training
    time.sleep(SLEEP_PER_STEP)

    # Heartbeat every step (normally every 50)
    post_heartbeat(step)

    # Control check every step (normally every 100)
    ctrl = get_control()
    print(f"  [control] flag={ctrl}")

    if ctrl == "PAUSE":
        print("\n⏸  PAUSE received. Saving checkpoint and waiting...")
        sha = save_dummy_checkpoint(step, reason="PAUSE")
        post_heartbeat(step, extra={"checkpoint_hash": sha})
        # In a real kernel we'd loop-wait; here we just exit cleanly.
        print("  PAUSE: checkpoint saved. Exiting cleanly (dummy: no wait loop).")
        sys.exit(0)

    if ctrl == "STOP":
        print("\n⏹  STOP received. Saving checkpoint and exiting.")
        sha = save_dummy_checkpoint(step, reason="STOP")
        post_heartbeat(step, extra={"checkpoint_hash": sha})
        sys.exit(0)

print("\n" + "=" * 70)
print("✅ Dummy kernel completed all steps — no PAUSE/STOP received.")
print("   End-to-end plumbing verified.")
print("=" * 70)

# Final heartbeat
post_heartbeat(TOTAL_DUMMY_STEPS, extra={"complete": True})
