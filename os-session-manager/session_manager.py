# session_manager.py
import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
BASE_DIR = "user_sessions"
os.makedirs(BASE_DIR, exist_ok=True)

@app.route('/api/session/create', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    session_path = os.path.join(BASE_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, "uploads"), exist_ok=True)
    return jsonify({"session_id": session_id})

@app.route('/api/session/save', methods=['POST'])
def save_notebook():
    data = request.json
    session_id = data.get("session_id")
    notebook_data = data.get("notebook")
    if not session_id or not notebook_data:
        return jsonify({"error": "Missing session_id or notebook data"}), 400
    path = os.path.join(BASE_DIR, session_id, "notebook.json")
    with open(path, "w") as f:
        json.dump(notebook_data, f)
    return jsonify({"status": "saved"})

@app.route('/api/session/load/<session_id>', methods=['GET'])
def load_notebook(session_id):
    path = os.path.join(BASE_DIR, session_id, "notebook.json")
    if not os.path.exists(path):
        return jsonify({"error": "Notebook not found"}), 404
    with open(path, "r") as f:
        notebook_data = json.load(f)
    return jsonify({"notebook": notebook_data})

@app.route('/api/session/<session_id>/upload', methods=['POST'])
def upload_file(session_id):
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    upload_dir = os.path.join(BASE_DIR, session_id, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)
    return jsonify({"status": "uploaded", "filename": file.filename})

@app.route('/api/session/<session_id>/file/<filename>', methods=['GET'])
def get_file(session_id, filename):
    upload_dir = os.path.join(BASE_DIR, session_id, "uploads")
    return send_from_directory(upload_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
