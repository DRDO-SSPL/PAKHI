const jwt = require("jsonwebtoken");

exports.verifyToken = (req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1]; // Bearer <token>

  if (!token) {
    return res.status(401).json({ error: "Token required" });
  }

  try {
    // const decoded = jwt.verify(token, "your_jwt_secret");
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded; // now you can access req.user.id in controllers
    next();
  } catch (err) {
    res.status(403).json({ error: "Invalid token" });
  }
};
