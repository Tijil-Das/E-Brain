const https = require('https');
const fs = require('fs');
const path = require('path');

function getApiKey() {
  if (process.env.KAGGLE_API_KEY) return process.env.KAGGLE_API_KEY;
  const localKeyPath = path.join(__dirname, 'kaggle_key.local');
  if (fs.existsSync(localKeyPath)) {
    return fs.readFileSync(localKeyPath, 'utf8').trim();
  }
  return "";
}

const API_KEY = getApiKey();

function callKaggle(toolName, args) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      jsonrpc: '2.0',
      id: Date.now(),
      method: 'tools/call',
      params: { name: toolName, arguments: args }
    });
    const req = https.request('https://www.kaggle.com/mcp', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Length': Buffer.byteLength(data)
      }
    }, (res) => {
      let body = '';
      res.on('data', d => body += d);
      res.on('end', () => {
        try {
          const lines = body.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              return resolve(JSON.parse(line.slice(6)));
            }
          }
          resolve(body);
        } catch(e) { resolve(body); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

module.exports = { callKaggle };

if (require.main === module) {
  (async () => {
    const tool = process.argv[2] || 'get_accelerator_quota';
    const args = process.argv[3] ? JSON.parse(process.argv[3]) : { request: {} };
    console.log(`Calling Kaggle MCP tool: ${tool}...`);
    const res = await callKaggle(tool, args);
    console.log('Result:\n', JSON.stringify(res, null, 2));
  })();
}
