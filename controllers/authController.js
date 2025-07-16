const jwt = require("jsonwebtoken");
const User = require("../models/user");

// Hardcoded user for demo
const DEMO_USER = {
  username: "testuser",
  password: "123456", // in real app, use bcrypt-hashed password
  _id: "64f6cfed0bc5ea35a7f12345" // fake Mongo ID or create one in DB
};

exports.login = (req, res) => {
  const { username, password } = req.body;

  if (username === DEMO_USER.username && password === DEMO_USER.password) {
    const token = jwt.sign({ id: DEMO_USER._id, username }, "your_jwt_secret", {
      expiresIn: "1h",
    });

    return res.json({ token });
  }

  res.status(401).json({ error: "Invalid credentials" });
};
