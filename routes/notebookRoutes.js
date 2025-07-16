const { uploadNotebook, fetchNotebooks, downloadNotebook, deleteNotebook} = require("../controllers/notebookController");
const { verifyToken } = require("../middleware/authMiddleware");
const express = require("express");
const router = express.Router();
const { body } = require("express-validator");
const { uploadNotebookValidator } = require("../middleware/notebookValidators");
const { runNotebook } = require("../controllers/notebookController");


// router.post("/upload-json", uploadNotebook);
router.get("/fetch", fetchNotebooks);
router.get("/run/:id", verifyToken, runNotebook);
router.get("/download/:id",verifyToken, downloadNotebook);
router.delete("/delete/:id", deleteNotebook);
router.post("/upload-json", verifyToken, uploadNotebook, uploadNotebookValidator    );
router.get("/notebooks", verifyToken, fetchNotebooks);
router.delete("/delete/:id", verifyToken, deleteNotebook);
// router.get("/run/:id", verifyToken, runNotebook);
router.post(
  "/upload-json",
  verifyToken,
  body("sessionId").optional().isString(),
  uploadNotebook
);


module.exports = router;
