const splitCsvLine = (line) => {
  const cells = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells;
};

export const parseCsv = (text) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return { columns: [], rows: [] };
  }

  const columns = splitCsvLine(lines[0]).map((value) => value.replace(/^"|"$/g, ""));
  const rows = lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return columns.reduce((acc, column, index) => {
      acc[column] = values[index] ?? "";
      return acc;
    }, {});
  });

  return { columns, rows };
};

export const toCsv = (columns, rows) => {
  const escapeCell = (value) => {
    const stringValue = value === null || value === undefined ? "" : String(value);
    if (stringValue.includes(",") || stringValue.includes('"') || stringValue.includes("\n")) {
      return `"${stringValue.replace(/"/g, '""')}"`;
    }
    return stringValue;
  };

  const header = columns.map(escapeCell).join(",");
  const lines = rows.map((row) => columns.map((col) => escapeCell(row[col])).join(","));
  return [header, ...lines].join("\n");
};
