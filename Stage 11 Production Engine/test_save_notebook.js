const { callKaggle } = require('./kaggle_client');

async function test() {
  const req = {
    request: {
      newTitle: "EBrain 3B Probe",
      slug: "tijilgupta/ebrain-3b-probe",
      text: "import torch\nprint('CUDA Available:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('Device:', torch.cuda.get_device_name(0))\n",
      language: "python",
      kernelType: "script",
      isPrivate: true,
      enableGpu: true,
      enableInternet: true,
      kernelExecutionType: "SaveAndRunAll"
    }
  };
  const res = await callKaggle('save_notebook', req);
  console.log('Save notebook response:', JSON.stringify(res, null, 2));
}

test();
