const mongoose = require("mongoose");

const notebookSchema = new mongoose.Schema({
  userId: String,
  sessionId: String,
  filename: String,
  content: Object,
  uploadedAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model("Notebook", notebookSchema);
