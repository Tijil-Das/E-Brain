const { callKaggle } = require('./kaggle_client');

async function launchRegressionKernel() {
  const kernelSlug = `ebrain-character-regression-seeds-${Date.now()}`;
  console.log(`Launching Kaggle Dual-Seed Regression Kernel: ${kernelSlug}...`);

  const pythonScript = `
import subprocess, sys, os

print("=" * 80)
print("E-BRAIN: KAGGLE GPU DUAL-SEED REGRESSION RUNNER")
print("Target Band: 1.4957 to 1.5357 on Common Val (Target: 1.5157 ± 0.02)")
print("=" * 80)

# Clone latest codebase
cmd = "git clone https://github.com/Tijil-Das/E-Brain.git /kaggle/working/ebrain_repo"
subprocess.run(cmd, shell=True, check=True)

os.chdir("/kaggle/working/ebrain_repo")
sys.path.insert(0, "/kaggle/working/ebrain_repo")

# Run Seed 42
print("\n>>> LAUNCHING SEED 42 <<<")
cmd_42 = "python 'Stage 12 Multilingual BPE/run_character_regression.py' --seed 42"
subprocess.run(cmd_42, shell=True, check=True)

# Run Seed 123
print("\n>>> LAUNCHING SEED 123 <<<")
cmd_123 = "python 'Stage 12 Multilingual BPE/run_character_regression.py' --seed 123"
subprocess.run(cmd_123, shell=True, check=True)
`;

  const req = {
    request: {
      newTitle: `EBrain Regression Dual Seed ${Date.now()}`,
      text: pythonScript,
      language: "python",
      kernelType: "script",
      isPrivate: true,
      enableGpu: true,
      enableInternet: true,
      kernelExecutionType: "SaveAndRunAll"
    }
  };

  const res = await callKaggle('save_notebook', req);
  console.log('Kaggle Launch Result:', JSON.stringify(res, null, 2));
}

launchRegressionKernel();
