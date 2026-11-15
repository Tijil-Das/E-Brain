const { callKaggle } = require('./kaggle_client');

async function testUser(un) {
  const r = await callKaggle('get_user_profile', {
    request: {
      userName: un,
      hasUserName: true
    }
  });
  console.log(`Profile for ${un}:`, JSON.stringify(r));
}

async function run() {
  await testUser('tijil');
  await testUser('tijilgupta');
}
run();
