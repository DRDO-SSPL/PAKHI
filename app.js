require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();


app.use(express.json());
app.use(cors());


const authRoutes = require("./routes/authRoutes");
const notebookRoutes = require("./routes/notebookRoutes");

app.use("/api/auth", authRoutes);
app.use("/api", notebookRoutes);

mongoose.connect(process.env.MONGO_URI)
  .then(() => {
    console.log("✅ MongoDB connected");
    app.listen(process.env.PORT || 5000, () => {
      console.log(`🚀 Server running on port ${process.env.PORT || 5000}`);
    });
  })
  .catch(err => console.error("❌ DB connection error:", err));
const errorHandler = require("./middleware/errorMiddleware");
app.use(errorHandler); // after all routes
