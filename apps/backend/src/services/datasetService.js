import path from "path";
import { promises as fs } from "fs";
import { pool } from "../db/pool.js";
import { sanitizeDatasetName } from "../utils/sanitize.js";
import { toCsv } from "../utils/csv.js";
import { parseCsv } from "../utils/csv.js";
import { resolveDatasetMeta, resolveDataDir } from "./datasetRegistry.js";

const resolveRoot = () =>
  process.env.FRESHFLOW_ROOT
    ? path.resolve(process.env.FRESHFLOW_ROOT)
    : path.resolve(process.cwd(), "..", "..");

const ensureDataDir = async () => {
  const dir = path.join(resolveRoot(), "data", "assistant");
  await fs.mkdir(dir, { recursive: true });
  return dir;
};

const normalizeNumber = (value) => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeDate = (value) => {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString().slice(0, 10);
};

const writeCsvSnapshot = async ({ dataset, columns, rows }) => {
  const dir = await ensureDataDir();
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `${sanitizeDatasetName(dataset)}_${timestamp}.csv`;
  const outputPath = path.join(dir, filename);
  const csv = toCsv(columns, rows);
  await fs.writeFile(outputPath, csv, "utf8");
  return outputPath;
};

const resolveFilePath = async (datasetName) => {
  const dataDir = resolveDataDir();
  const meta = await resolveDatasetMeta({ dataset: datasetName });
  if (!meta) return null;
  return { ...meta, filePath: path.join(dataDir, meta.file || `${meta.name}.csv`) };
};

const readCsvFile = async (filePath) => {
  try {
    const content = await fs.readFile(filePath, "utf8");
    return parseCsv(content);
  } catch (error) {
    return { columns: [], rows: [] };
  }
};

const ensureBackup = async (filePath) => {
  try {
    const dataDir = resolveDataDir();
    const backupDir = path.join(dataDir, "backups");
    await fs.mkdir(backupDir, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${path.basename(filePath, ".csv")}_${timestamp}.csv`;
    await fs.copyFile(filePath, path.join(backupDir, filename));
  } catch (error) {
    // ignore backup failure
  }
};

const writeCsvFile = async (filePath, columns, rows) => {
  const csv = toCsv(columns, rows);
  await fs.writeFile(filePath, csv, "utf8");
};

const insertGenericRows = async (dataset, rows) => {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const datasetResult = await client.query(
      "INSERT INTO datasets (name) VALUES ($1) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
      [dataset]
    );
    const datasetId = datasetResult.rows[0].id;

    for (const row of rows) {
      await client.query(
        "INSERT INTO dataset_rows (dataset_id, data) VALUES ($1, $2)",
        [datasetId, row]
      );
    }

    await client.query("COMMIT");
    return rows.length;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

const insertInventoryRows = async (rows) => {
  const text =
    "INSERT INTO inventory (item_id, item_name, location_id, quantity, unit, expiry_date) VALUES ($1, $2, $3, $4, $5, $6)";
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    for (const row of rows) {
      await client.query(text, [
        row.item_id || row.sku || row.id || null,
        row.item_name || row.name || null,
        row.location_id || row.place_id || null,
        normalizeNumber(row.quantity || row.qty || row.on_hand),
        row.unit || row.uom || null,
        normalizeDate(row.expiry_date || row.expiration_date || row.expiry),
      ]);
    }
    await client.query("COMMIT");
    return rows.length;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

const insertSalesRows = async (rows) => {
  const text =
    "INSERT INTO sales (item_id, location_id, sale_date, quantity, revenue) VALUES ($1, $2, $3, $4, $5)";
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    for (const row of rows) {
      await client.query(text, [
        row.item_id || row.sku || row.id || null,
        row.location_id || row.place_id || null,
        normalizeDate(row.sale_date || row.date || row.day),
        normalizeNumber(row.quantity || row.qty),
        normalizeNumber(row.revenue || row.sales || row.total),
      ]);
    }
    await client.query("COMMIT");
    return rows.length;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

const updateSalesRows = async (rows) => {
  const client = await pool.connect();
  let updated = 0;
  try {
    await client.query("BEGIN");
    for (const row of rows) {
      if (!row.id && !row.item_id && !row.sku) continue;
      const result = await client.query(
        "UPDATE sales SET quantity = COALESCE($1, quantity), revenue = COALESCE($2, revenue), sale_date = COALESCE($3, sale_date), updated_at = NOW() WHERE id = COALESCE($4, id) AND item_id = COALESCE($5, item_id)",
        [
          normalizeNumber(row.quantity || row.qty),
          normalizeNumber(row.revenue || row.sales || row.total),
          normalizeDate(row.sale_date || row.date || row.day),
          row.id || null,
          row.item_id || row.sku || null,
        ]
      );
      updated += result.rowCount;
    }
    await client.query("COMMIT");
    return updated;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

const deleteSalesRows = async (rows) => {
  const client = await pool.connect();
  let deleted = 0;
  try {
    await client.query("BEGIN");
    for (const row of rows) {
      if (!row.id && !row.item_id && !row.sku) continue;
      const result = await client.query(
        "DELETE FROM sales WHERE id = COALESCE($1, id) AND item_id = COALESCE($2, item_id)",
        [row.id || null, row.item_id || row.sku || null]
      );
      deleted += result.rowCount;
    }
    await client.query("COMMIT");
    return deleted;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

const updateInventoryRows = async (rows) => {
  const client = await pool.connect();
  let updated = 0;
  try {
    await client.query("BEGIN");
    for (const row of rows) {
      if (!row.id && !row.item_id && !row.sku) continue;
      const result = await client.query(
        "UPDATE inventory SET quantity = COALESCE($1, quantity), expiry_date = COALESCE($2, expiry_date), updated_at = NOW() WHERE id = COALESCE($3, id) AND item_id = COALESCE($4, item_id)",
        [
          normalizeNumber(row.quantity || row.qty),
          normalizeDate(row.expiry_date || row.expiration_date),
          row.id || null,
          row.item_id || row.sku || null,
        ]
      );
      updated += result.rowCount;
    }
    await client.query("COMMIT");
    return updated;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

const deleteInventoryRows = async (rows) => {
  const client = await pool.connect();
  let deleted = 0;
  try {
    await client.query("BEGIN");
    for (const row of rows) {
      if (!row.id && !row.item_id && !row.sku) continue;
      const result = await client.query(
        "DELETE FROM inventory WHERE id = COALESCE($1, id) AND item_id = COALESCE($2, item_id)",
        [row.id || null, row.item_id || row.sku || null]
      );
      deleted += result.rowCount;
    }
    await client.query("COMMIT");
    return deleted;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

export const ingestDataset = async ({ dataset, columns, rows }) => {
  const resolvedColumns = columns?.length ? columns : Object.keys(rows[0] || {});
  const inferredMeta = await resolveDatasetMeta({ dataset, columns: resolvedColumns });
  const sanitized = sanitizeDatasetName(dataset || inferredMeta?.name || "dataset");

  if (inferredMeta) {
    const filePath = path.join(resolveDataDir(), inferredMeta.file || `${inferredMeta.name}.csv`);
    if (inferredMeta.required?.length) {
      const missing = inferredMeta.required.filter((col) => !resolvedColumns.includes(col));
      if (missing.length) {
        throw new Error(`Missing required columns: ${missing.join(", ")}`);
      }
    }
    const existing = await readCsvFile(filePath);
    const combinedColumns = Array.from(
      new Set([...(existing.columns || []), ...(resolvedColumns || [])])
    );
    const normalizedRows = rows.map((row) =>
      combinedColumns.reduce((acc, col) => {
        acc[col] = row[col] ?? "";
        return acc;
      }, {})
    );
    const combinedRows = [...existing.rows, ...normalizedRows];

    if (existing.rows.length) {
      await ensureBackup(filePath);
    }
    await writeCsvFile(filePath, combinedColumns, combinedRows);

    const snapshotPath = await writeCsvSnapshot({
      dataset: inferredMeta.name,
      columns: combinedColumns,
      rows: normalizedRows,
    });

    return { count: rows.length, snapshotPath, dataset: inferredMeta.name, filePath };
  }

  const snapshotPath = await writeCsvSnapshot({
    dataset: sanitized,
    columns: resolvedColumns,
    rows,
  });

  let count = 0;
  if (sanitized === "inventory") {
    count = await insertInventoryRows(rows);
  } else if (sanitized === "sales") {
    count = await insertSalesRows(rows);
  } else {
    count = await insertGenericRows(sanitized, rows);
  }

  return { count, snapshotPath, dataset: sanitized };
};

export const modifyDataset = async ({ dataset, operation, rows }) => {
  const inferredMeta = await resolveDatasetMeta({
    dataset,
    columns: Object.keys(rows[0] || {}),
  });
  const sanitized = sanitizeDatasetName(dataset || inferredMeta?.name || "dataset");

  if (inferredMeta) {
    const filePath = path.join(resolveDataDir(), inferredMeta.file || `${inferredMeta.name}.csv`);
    const existing = await readCsvFile(filePath);
    const incomingColumns = Object.keys(rows[0] || {});
    const columns = Array.from(
      new Set([...(existing.columns || []), ...incomingColumns])
    );
    const keys = inferredMeta.keys?.length ? inferredMeta.keys : [];
    const required = inferredMeta.required || [];

    if (operation === "insert" && required.length) {
      const missing = required.filter((col) => !incomingColumns.includes(col));
      if (missing.length) {
        throw new Error(`Missing required columns: ${missing.join(", ")}`);
      }
    }

    if (operation !== "insert" && !keys.length) {
      throw new Error("No key columns found to modify this dataset.");
    }

    const makeKey = (row) => keys.map((key) => row[key]).join("|");
    const existingMap = new Map(existing.rows.map((row) => [makeKey(row), row]));
    let modifiedRows = [...existing.rows];
    let count = 0;

    if (operation === "insert") {
      const normalizedRows = rows.map((row) =>
        columns.reduce((acc, col) => {
          acc[col] = row[col] ?? "";
          return acc;
        }, {})
      );
      modifiedRows = [...existing.rows, ...normalizedRows];
      count = rows.length;
    } else if (operation === "update") {
      const updateMap = new Map(rows.map((row) => [makeKey(row), row]));
      modifiedRows = existing.rows.map((row) => {
        const key = makeKey(row);
        const updateRow = updateMap.get(key);
        if (updateRow) {
          count += 1;
          return { ...row, ...updateRow };
        }
        return row;
      });
    } else if (operation === "delete") {
      const deleteKeys = new Set(rows.map((row) => makeKey(row)));
      const before = existing.rows.length;
      modifiedRows = existing.rows.filter((row) => !deleteKeys.has(makeKey(row)));
      count = before - modifiedRows.length;
    }

    if (existing.rows.length) {
      await ensureBackup(filePath);
    }
    await writeCsvFile(filePath, columns, modifiedRows);
    return { count, operation, dataset: inferredMeta.name, filePath };
  }

  const isInventory = sanitized === "inventory";
  const isSales = sanitized === "sales";

  if (!isInventory && !isSales) {
    throw new Error("Modify operations are supported for inventory and sales only.");
  }

  if (operation === "update") {
    const count = isInventory
      ? await updateInventoryRows(rows)
      : await updateSalesRows(rows);
    return { count, operation };
  }

  if (operation === "delete") {
    const count = isInventory
      ? await deleteInventoryRows(rows)
      : await deleteSalesRows(rows);
    return { count, operation };
  }

  const count = isInventory
    ? await insertInventoryRows(rows)
    : await insertSalesRows(rows);
  return { count, operation: "insert" };
};

export const queryExpiringInventory = async (days) => {
  const client = await pool.connect();
  try {
    const result = await client.query(
      "SELECT id, item_id, item_name, location_id, quantity, unit, expiry_date FROM inventory WHERE expiry_date IS NOT NULL AND expiry_date <= CURRENT_DATE + ($1 || ' days')::interval ORDER BY expiry_date ASC",
      [days]
    );
    return result.rows;
  } finally {
    client.release();
  }
};

export const insertPredictions = async ({ period, predictions }) => {
  if (!predictions?.length) return 0;
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    for (const prediction of predictions) {
      await client.query(
        "INSERT INTO predictions (item_id, location_id, period, predicted_demand, confidence) VALUES ($1, $2, $3, $4, $5)",
        [
          prediction.item_id || null,
          prediction.location_id || null,
          period || null,
          normalizeNumber(prediction.predicted_demand),
          normalizeNumber(prediction.confidence),
        ]
      );
    }
    await client.query("COMMIT");
    return predictions.length;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
};

export const insertPromotion = async ({ name, startDate, endDate, items, strategy }) => {
  const result = await pool.query(
    "INSERT INTO promotions (name, start_date, end_date, items, strategy) VALUES ($1, $2, $3, $4, $5) RETURNING id",
    [name, startDate, endDate, items, strategy]
  );
  return result.rows[0].id;
};
