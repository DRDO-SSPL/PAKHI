import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const MarkdownCell = ({ id, content, onChange, onDelete }) => {
  const [preview, setPreview] = useState(false);

  return (
    <div className="cell">
      <button
        onClick={onDelete}
        className="delete"
        style={{ float: 'right', marginLeft: '0.5rem' }}
      >
        🗑 Delete
      </button>

      <button onClick={() => setPreview(!preview)} style={{ marginBottom: '0.75rem' }}>
        {preview ? 'Edit' : 'Preview'}
      </button>

      {preview ? (
        <div className="output">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      ) : (
        <textarea
          rows={4}
          value={content}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Write markdown here..."
        />
      )}
    </div>
  );
};

export default MarkdownCell;
