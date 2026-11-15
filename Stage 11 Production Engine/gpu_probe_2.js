const { callKaggle } = require('./kaggle_client');

async function testGPU() {
  const req = {
    request: {
      newTitle: "EBrain GPU Probe 2",
      slug: "tijildas/ebrain-gpu-probe-2",
      text: "import torch\nprint('CUDA Available:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('Device:', torch.cuda.get_device_name(0))\n",
      language: "python",
      kernelType: "script",
      isPrivate: true,
      enableGpu: true,
      hasEnableGpu: true,
      machineShape: "Gpu",
      hasMachineShape: true,
      enableInternet: true,
      hasEnableInternet: true,
      kernelExecutionType: "SaveAndRunAll"
    }
  };
  const res = await callKaggle('save_notebook', req);
  console.log('Result:', JSON.stringify(res, null, 2));
}

testGPU();
