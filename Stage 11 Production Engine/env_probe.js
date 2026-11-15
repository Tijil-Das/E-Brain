const { callKaggle } = require('./kaggle_client');

async function probeEnv() {
  const code = `
import urllib.request
import torch

print("=== KAGGLE ENVIRONMENT PROBE ===")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())

try:
    with urllib.request.urlopen("https://www.google.com", timeout=5) as r:
        print("Internet Access: SUCCESS (Status:", r.status, ")")
except Exception as e:
    print("Internet Access: FAILED (", e, ")")
`;

  const req = {
    request: {
      newTitle: "EBrain Full Env Probe",
      slug: "tijildas/ebrain-full-env-probe",
      text: code,
      language: "python",
      kernelType: "script",
      isPrivate: true,
      enableGpu: true,
      hasEnableGpu: true,
      enableInternet: true,
      hasEnableInternet: true,
      kernelExecutionType: "SaveAndRunAll"
    }
  };
  const res = await callKaggle('save_notebook', req);
  console.log('Dispatched:', JSON.stringify(res, null, 2));
}

probeEnv();
