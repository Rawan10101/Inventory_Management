import { parseCsv } from "../utils/csv.js";

const tryParseJson = (text) => {
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
};

const parseKeyValueLines = (text) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return { columns: [], rows: [] };
  }

  const rows = lines.map((line) => {
    const pairs = line.split(/\s*,\s*/);
    return pairs.reduce((acc, pair) => {
      const [rawKey, ...rest] = pair.split(":");
      if (!rawKey || rest.length === 0) {
        return acc;
      }
      const key = rawKey.trim();
      const value = rest.join(":").trim();
      acc[key] = value;
      return acc;
    }, {});
  });

  const columns = Array.from(
    rows.reduce((set, row) => {
      Object.keys(row).forEach((key) => set.add(key));
      return set;
    }, new Set())
  );

  return { columns, rows };
};

const looksLikeCsv = (text) => {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return false;
  return lines[0].includes(",") && lines[1].includes(",");
};

export const parseDataFromInput = ({ message = "", attachment }) => {
  if (attachment?.type === "json") {
    const json = tryParseJson(attachment.content);
    if (Array.isArray(json)) {
      return { format: "json", columns: Object.keys(json[0] || {}), rows: json };
    }
  }

  if (attachment?.type === "csv") {
    const parsed = parseCsv(attachment.content || "");
    return { format: "csv", ...parsed };
  }

  const jsonMatch = message.match(/\[[\s\S]*\]/);
  if (jsonMatch) {
    const json = tryParseJson(jsonMatch[0]);
    if (Array.isArray(json)) {
      return { format: "json", columns: Object.keys(json[0] || {}), rows: json };
    }
  }

  if (looksLikeCsv(message)) {
    const parsed = parseCsv(message);
    return { format: "csv", ...parsed };
  }

  const keyValueParsed = parseKeyValueLines(message);
  if (keyValueParsed.rows.length > 0 && keyValueParsed.columns.length > 0) {
    return { format: "kv", ...keyValueParsed };
  }

  return { format: "none", columns: [], rows: [] };
};
