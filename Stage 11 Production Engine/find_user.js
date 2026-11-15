const { callKaggle } = require('./kaggle_client');

async function findUser() {
  // Test search_content
  try {
    const r1 = await callKaggle('search_content', {
      request: {
        query: 'ebrain',
        type: 'Kernels',
        filterType: 'User'
      }
    });
    console.log('Search content:', JSON.stringify(r1).slice(0, 300));
  } catch(e) { console.error('r1 error', e); }

  // Test search_datasets
  try {
    const r2 = await callKaggle('search_datasets', {
      request: {
        group: 'My',
        sortBy: 'Updated'
      }
    });
    console.log('Search datasets My:', JSON.stringify(r2));
  } catch(e) { console.error('r2 error', e); }
}

findUser();
