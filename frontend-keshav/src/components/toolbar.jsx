// src/components/Toolbar.jsx
const Toolbar = ({ onAddCell, onSaveNotebook, onLoadNotebook }) => {
  return (
    <div style={{ marginBottom: '1rem' }}>
      <button onClick={() => onAddCell('code')}>+ Code Cell</button>
      <button onClick={() => onAddCell('markdown')}>+ Markdown Cell</button>
      
      <button onClick={onSaveNotebook}>💾 Save Notebook</button>
      <button onClick={onLoadNotebook}>📂 Load Notebook</button>

      <button disabled>Upload</button>
      <button disabled>Run</button>
    </div>
  );
};

export default Toolbar;
