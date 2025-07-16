const { IncomingForm } = require("formidable");
const fs = require("fs");
const Notebook = require("../models/Notebook");
const logger = require("../utils/logger");
const { validationResult } = require("express-validator");
const path = require("path");
const { exec } = require("child_process");
const { writeFileSync, unlinkSync } = require("fs");

exports.runNotebook = async (req, res) => {
  try {
    const notebookId = req.params.id;

    const notebook = await Notebook.findOne({
      _id: notebookId,
      userId: req.user.id,
    });

    if (!notebook) {
      logger.warn("⚠️ Notebook not found for execution", { notebookId });
      return res.status(404).json({ error: "Notebook not found" });
    }

    const tmpDir = path.join(__dirname, "../tmp");
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir);

    // 1. Write notebook to temp file
    const tmpPath = path.join(__dirname, "../tmp", `${notebookId}.json`);
    writeFileSync(tmpPath, JSON.stringify(notebook.content, null, 2));

    // 2. Run Docker container with mounted volume
    const dockerCmd = `docker run --rm -v ${path.resolve(__dirname, "../tmp")}:/data notebook-runner python runner.py /data/${notebookId}.json`;

    exec(dockerCmd, (error, stdout, stderr) => {
      // 3. Clean up temp file
      unlinkSync(tmpPath);

      if (error) {
        logger.error("❌ Docker execution error", {
          message: error.message,
          stderr,
        });
        return res.status(500).json({
          error: "Execution failed",
          details: stderr || error.message,
        });
      }

      logger.info("🚀 Notebook executed successfully", { notebookId });
      return res.status(200).json({
        output: stdout,
      });
    });

  } catch (err) {
    logger.error("❌ Run notebook failed", {
      message: err.message,
      stack: err.stack,
    });
    res.status(500).json({ error: err.message || "Run failed" });
  }
};


// 📥 Upload Notebook
exports.uploadNotebook = async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.warn("📥 Upload validation failed", { errors: errors.array() });
    return res.status(400).json({ errors: errors.array() });
  }

  const form = new IncomingForm({ multiples: false });

  form.parse(req, async (err, fields, files) => {
    if (err) {
      logger.error("❌ Form parsing error", {
        message: err.message,
        stack: err.stack,
      });
      return res.status(500).json({ error: "Form parsing failed" });
    }

    const file = files.jsonFile?.[0];
    if (!file) {
      logger.error("❌ No file found in 'files.jsonFile'");
      return res.status(400).json({ error: "No file uploaded" });
    }

    try {
      const rawData = fs.readFileSync(file.filepath, "utf-8");
      const jsonContent = JSON.parse(rawData);

      const newNotebook = new Notebook({
        userId: req.user.id,
        sessionId: fields.sessionId?.[0] || "default",
        filename: file.originalFilename,
        content: jsonContent,
      });

      await newNotebook.save();

      logger.info("✅ Notebook saved to DB", {
        filename: file.originalFilename,
        userId: req.user.id,
        sessionId: fields.sessionId?.[0] || "default",
      });

      res.status(201).json({ message: "Notebook uploaded successfully" });
    } catch (e) {
      logger.error("❌ UploadNotebook Exception", {
        message: e.message,
        stack: e.stack,
      });
      res.status(500).json({ error: e.message || "Unknown error" });
    }
  });
};

// 📤 Fetch Notebooks (by user and optional session)
exports.fetchNotebooks = async (req, res) => {
  try {
    const { sessionId } = req.query;
    const query = { userId: req.user.id };
    if (sessionId) query.sessionId = sessionId;

    const notebooks = await Notebook.find(query).sort({ uploadedAt: -1 });

    logger.info("📤 Notebooks fetched", {
      userId: req.user.id,
      count: notebooks.length,
    });

    res.status(200).json({
      count: notebooks.length,
      notebooks,
    });
  } catch (e) {
    logger.error("❌ Fetch Notebooks Failed", {
      message: e.message,
      stack: e.stack,
    });
    res.status(500).json({ error: e.message || "Fetch failed" });
  }
};

// 📥 Download Notebook
exports.downloadNotebook = async (req, res) => {
  try {
    const notebook = await Notebook.findOne({
      _id: req.params.id,
      userId: req.user.id,
    });

    if (!notebook) {
      logger.warn("⚠️ Notebook not found for download", {
        notebookId: req.params.id,
        userId: req.user.id,
      });
      return res.status(404).json({ error: "Notebook not found" });
    }

    const fileContent = JSON.stringify(notebook.content, null, 2);

    logger.info("📥 Notebook download initiated", {
      filename: notebook.filename,
      userId: req.user.id,
    });

    res.setHeader("Content-Disposition", `attachment; filename="${notebook.filename}"`);
    res.setHeader("Content-Type", "application/json");
    res.status(200).send(fileContent);
  } catch (e) {
    logger.error("❌ Download Notebook Failed", {
      message: e.message,
      stack: e.stack,
    });
    res.status(500).json({ error: e.message || "Download failed" });
  }
};

// 🗑️ Delete Notebook
exports.deleteNotebook = async (req, res) => {
  try {
    const deleted = await Notebook.findOneAndDelete({
      _id: req.params.id,
      userId: req.user.id,
    });

    if (!deleted) {
      logger.warn("⚠️ Notebook not found for deletion", {
        notebookId: req.params.id,
        userId: req.user.id,
      });
      return res.status(404).json({ error: "Notebook not found" });
    }

    logger.info("🗑️ Notebook deleted", {
      notebookId: req.params.id,
      userId: req.user.id,
    });

    res.status(200).json({ message: "Notebook deleted successfully" });
  } catch (e) {
    logger.error("❌ Delete Notebook Failed", {
      message: e.message,
      stack: e.stack,
    });
    res.status(500).json({ error: e.message || "Deletion failed" });
  }
};
