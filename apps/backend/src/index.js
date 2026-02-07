import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import chatRouter from "./routes/chat.js";
import datasetsRouter from "./routes/datasets.js";
import { requireApiKey } from "./utils/auth.js";

dotenv.config();

const app = express();

app.use(helmet());
app.use(
  cors({
    origin: process.env.CORS_ORIGIN
      ? process.env.CORS_ORIGIN.split(",").map((value) => value.trim())
      : "*",
  })
);
app.use(express.json({ limit: "6mb" }));

app.get("/api/health", (req, res) => {
  res.json({ status: "healthy", message: "Fresh Flow API is running" });
});

if (process.env.API_KEY) {
  app.use(requireApiKey(process.env.API_KEY));
}

app.use("/api/chat", chatRouter);
app.use("/api/datasets", datasetsRouter);

app.use((err, req, res, next) => {
  const status = err.status || 500;
  res.status(status).json({
    error: "Request failed",
    detail: err.message || "Unexpected error",
  });
});

const port = Number(process.env.PORT || 4000);
app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Fresh Flow API listening on ${port}`);
});
