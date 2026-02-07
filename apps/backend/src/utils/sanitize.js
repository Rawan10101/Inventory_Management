export const sanitizeDatasetName = (value = "") =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);

export const clampNumber = (value, min, max) =>
  Math.min(max, Math.max(min, Number(value)));
