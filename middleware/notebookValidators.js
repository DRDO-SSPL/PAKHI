const { body } = require("express-validator");

exports.uploadNotebookValidator = [
  body("sessionId")
    .optional()
    .isString()
    .withMessage("Session ID must be a string"),
];