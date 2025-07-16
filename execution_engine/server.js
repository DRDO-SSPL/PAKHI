const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const { exec } = require("child_process");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 8000;

app.use(cors());
app.use(bodyParser.json());

app.post("/execute", (req, res) => {
  const { code } = req.body;

  // ✅ Define filename and filepath inside the POST handler
  const filename = `temp-${Date.now()}.py`;
  const filepath = path.join(__dirname, "temp", filename);

  fs.writeFileSync(filepath, code);

  exec(`python3 "${filepath}"`, (err, stdout, stderr) => {
    fs.unlinkSync(filepath); // delete the file after execution

    if (err) {
      return res.status(500).json({ output: stderr || err.message });
    }

    res.json({ output: stdout });
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Execution Engine running on port ${PORT}`);
});

