const https = require('https');
const fs = require('fs');
const { callKaggle } = require('./kaggle_client');

async function uploadFile(filePath, fileName) {
  const stat = fs.statSync(filePath);
  const initRes = await callKaggle('upload_dataset_file', {
    request: {
      fileName: fileName,
      contentLength: stat.size,
      lastModifiedEpochSeconds: Math.floor(stat.mtimeMs / 1000)
    }
  });
  
  const uploadData = JSON.parse(initRes.result.content[0].text);
  console.log('Got upload URL for', fileName, uploadData.create_url.slice(0, 80));
  
  // PUT file to create_url
  return new Promise((resolve, reject) => {
    const url = new URL(uploadData.create_url);
    const req = https.request(url, {
      method: 'PUT',
      headers: {
        'Content-Length': stat.size,
        'Content-Type': 'application/octet-stream'
      }
    }, (res) => {
      let b = '';
      res.on('data', d => b += d);
      res.on('end', () => {
        console.log(`Upload status for ${fileName}:`, res.statusCode);
        resolve({ token: uploadData.token, status: res.statusCode, body: b });
      });
    });
    req.on('error', reject);
    fs.createReadStream(filePath).pipe(req);
  });
}

async function run() {
  const testFile = 'Stage 11 Production Engine/canonical_datasets/vocab.json';
  const r = await uploadFile(testFile, 'vocab.json');
  console.log('Upload result:', r);
}

run();
