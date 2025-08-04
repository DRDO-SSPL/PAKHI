import { useState } from 'react';
import Toolbar from './Toolbar';
import CodeCell from './CodeCell';
import MarkdownCell from './MarkdownCell';

const Notebook = () => {
  const [cells, setCells] = useState([
    { id: 1, type: 'code', content: '' }
  ]);

  const addCell = (type) => {
    setCells([...cells, { id: Date.now(), type, content: '' }]);
  };

  const updateCell = (id, newContent) => {
    setCells((prev) =>
      prev.map((cell) =>
        cell.id === id ? { ...cell, content: newContent } : cell
      )
    );
  };


  const deleteCell = (id) => {
    setCells((prevCells) => prevCells.filter((cell) => cell.id !== id));
  };


// Save notebook to backend
const saveNotebook = async () => {
  try {
    const res = await fetch("http://localhost:5000/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notebook: cells }),
    });
    const data = await res.json();
    alert("✅ Notebook saved!");
  } catch (err) {
    console.error("Save failed:", err);
  }
};

// Load notebook from backend
const loadNotebook = async () => {
  try {
    const res = await fetch("http://localhost:5000/load");
    const data = await res.json();
    setCells(data.notebook);
  } catch (err) {
    console.error("Failed to load notebook:", err);
  }
};




  return (
    <div style={{ padding: '1rem' }}>
     <Toolbar
  onAddCell={addCell}
  onSaveNotebook={saveNotebook}
  onLoadNotebook={loadNotebook}
/>

      {cells.map((cell) =>
        cell.type === 'code' ? (
          <CodeCell
            key={cell.id}
            id={cell.id}
            content={cell.content}
            onChange={(newContent) => updateCell(cell.id, newContent)}
            onDelete={() => deleteCell(cell.id)}
            />
        ) : (
          <MarkdownCell
            key={cell.id}
            id={cell.id} 
            content={cell.content}
            onChange={(newContent) => updateCell(cell.id, newContent)}
            onDelete={() => deleteCell(cell.id)}
            />
        )
      )}
    </div>
  );
};

export default Notebook;
