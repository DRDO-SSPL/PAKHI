const logger = require("../utils/logger");

const errorHandler = (err, req, res, next) => {
  logger.error("Unhandled Error", {
    message: err.message,
    stack: err.stack
  });

  res.status(err.statusCode || 500).json({
    error: err.message || "Server Error"
  });
};

module.exports = errorHandler;
