// 测试 API 是否正常工作
const API_URL = 'http://106.39.200.227:8000';

async function test() {
  try {
    console.log('Testing API...');
    const res = await fetch(`${API_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_history: [1, 2, 3, 4, 5],
        top_k: 10,
        exclude_history: true
      })
    });
    
    if (!res.ok) {
      console.error('HTTP Error:', res.status);
      const text = await res.text();
      console.error('Response:', text);
      return;
    }
    
    const data = await res.json();
    console.log('Success!');
    console.log('Recommendations count:', data.recommendations?.length);
    console.log('First item:', data.recommendations?.[0]);
  } catch (err) {
    console.error('Error:', err.message);
  }
}

test();
