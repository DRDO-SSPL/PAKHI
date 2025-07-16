const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  username: String,
  password: String, // For mock/demo, not used in real auth yet
});

module.exports = mongoose.model("User", userSchema);
