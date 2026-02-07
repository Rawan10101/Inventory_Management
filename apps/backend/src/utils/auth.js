export const requireApiKey = (expectedKey) => (req, res, next) => {
  const header = req.headers.authorization || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : header;

  if (!token || token !== expectedKey) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  return next();
};
