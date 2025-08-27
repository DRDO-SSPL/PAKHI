import { useState } from 'react';

const CodeCell = ({ content, onChange , onDelete }) => {
  const [output, setOutput] = useState('...');

  const runCode = async () => {
    try {
      const res = await fetch('http://localhost:5000/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: content }),
      });

      const data = await res.json();
      setOutput(data.result); // Shows output in <pre>
    } catch (err) {
      setOutput('❌ Error running code');
    }
  };

  return (
    <div style={{ border: '1px solid #ccc', padding: '1rem', marginBottom: '1rem' }}>
     
     
     <button
  onClick={onDelete}
  style={{ float: 'right', backgroundColor: '#dc3545' }}
>
  🗑 Delete
</button>

     
      <textarea
        rows={5}
        cols={60}
        value={content}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Write Python code here..."
      />
      <div style={{ marginTop: '0.5rem' }}>
        <button onClick={runCode}>▶️ Run</button>
      </div>
      <div style={{ marginTop: '0.5rem' }}>
        <strong>Output:</strong>
        <pre>{output}</pre>
      </div>
    </div>
  );
};

export default CodeCell;
