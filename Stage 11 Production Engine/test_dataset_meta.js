const { callKaggle } = require('./kaggle_client');

async function testUpdateMeta() {
  const req = {
    request: {
      ownerSlug: "tijildas",
      datasetSlug: "ebrain-corpus",
      settings: {
        title: "EBrain Corpus",
        isPrivate: true,
        hasTitle: true
      }
    }
  };
  const res = await callKaggle('update_dataset_metadata', req);
  console.log('Update meta response:', JSON.stringify(res, null, 2));
}

testUpdateMeta();
