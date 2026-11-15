const fs = require('fs');
const path = require('path');
const { callKaggle } = require('./kaggle_client');

async function launch() {
  const kernelPath = path.join(__dirname, 'dual_seed_regression_kernel.py');
  const pythonScript = fs.readFileSync(kernelPath, 'utf-8');
  const timestamp = Date.now();
  const title = `EBrain Dual Seed Regression GPU ${timestamp}`;

  console.log(`Submitting Dual-Seed Regression to Kaggle GPU with Dataset tijildas/ebrain-canonical-datasets-v1...`);

  const req = {
    request: {
      newTitle: title,
      text: pythonScript,
      language: 'python',
      kernelType: 'script',
      isPrivate: true,
      hasEnableGpu: true,
      enableGpu: true,
      hasEnableInternet: false,
      enableInternet: false,
      datasetDataSources: ["tijildas/ebrain-canonical-datasets-v1"],
      kernelExecutionType: 'SaveAndRunAll'
    }
  };

  const res = await callKaggle('save_notebook', req);
  console.log('Launch Response:', JSON.stringify(res, null, 2));

  try {
    const textContent = res.result.content[0].text;
    const resultObj = JSON.parse(textContent);
    console.log(`Kernel Successfully Launched!`);
    console.log(`URL: ${resultObj.url}`);
    console.log(`Ref: ${resultObj.ref}`);
    console.log(`Kernel ID: ${resultObj.kernel_id}`);

    fs.writeFileSync(path.join(__dirname, 'active_regression_run.json'), JSON.stringify(resultObj, null, 2));
  } catch(e) {
    console.error('Error parsing response:', e);
  }
}

launch();
