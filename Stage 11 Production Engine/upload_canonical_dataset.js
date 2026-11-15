const fs = require('fs');
const https = require('https');
const { callKaggle } = require('./kaggle_client');

async function uploadDataset() {
  const zipPath = 'Stage 11 Production Engine/canonical_data.zip';
  const fileBytes = fs.readFileSync(zipPath);
  const key = fs.readFileSync('Stage 11 Production Engine/kaggle_key.local', 'utf-8').trim();

  console.log(`Requesting GCS upload URL for ${zipPath} (${fileBytes.length} bytes)...`);
  const initRes = await callKaggle('upload_dataset_file', {
    request: {
      fileName: 'canonical_data.zip',
      contentLength: fileBytes.length,
      lastModifiedEpochSeconds: Math.floor(Date.now() / 1000)
    }
  });

  const parsedInit = JSON.parse(initRes.result.content[0].text);
  console.log('GCS Token obtained.');

  // PUT bytes to GCS create_url
  console.log('Uploading bytes to Google Cloud Storage...');
  await new Promise((resolve, reject) => {
    const uploadUrl = new URL(parsedInit.create_url);
    const putReq = https.request(uploadUrl, {
      method: 'PUT',
      headers: {
        'Content-Length': fileBytes.length,
        'Content-Type': 'application/octet-stream'
      }
    }, (res) => {
      console.log('GCS Upload HTTP Status:', res.statusCode);
      res.on('data', () => {});
      res.on('end', resolve);
    });
    putReq.on('error', reject);
    putReq.write(fileBytes);
    putReq.end();
  });

  // Call Kaggle create dataset
  console.log('Registering dataset on Kaggle...');
  await new Promise((resolve, reject) => {
    const postReq = https.request('https://www.kaggle.com/api/v1/datasets/create/new', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json'
      }
    }, (res) => {
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        console.log('Kaggle Register Status:', res.statusCode);
        console.log('Response:', body);
        resolve();
      });
    });
    postReq.on('error', reject);
    postReq.write(JSON.stringify({
      title: 'EBrain Canonical Datasets V1',
      slug: 'ebrain-canonical-datasets-v1',
      ownerSlug: 'tijildas',
      isPrivate: true,
      files: [
        {
          token: parsedInit.token
        }
      ]
    }));
    postReq.end();
  });
}

uploadDataset();
