import path from "path";
import { promises as fs } from "fs";
import { parseCsv } from "../utils/csv.js";

const DATASET_DEFS = {
  fct_orders: {
    file: "fct_orders.csv",
    required: ["id", "place_id", "created", "status"],
    keys: ["id"],
  },
  fct_order_items: {
    file: "fct_order_items.csv",
    required: ["order_id", "item_id", "quantity", "price"],
    keys: ["order_id", "item_id"],
  },
  dim_items: {
    file: "dim_items.csv",
    required: ["id", "title", "manage_inventory"],
    keys: ["id"],
  },
  dim_places: {
    file: "dim_places.csv",
    required: ["id", "title"],
    keys: ["id"],
  },
  fct_inventory_reports: {
    file: "fct_inventory_reports.csv",
    required: ["report_date", "item_id", "quantity_on_hand"],
    keys: ["report_date", "item_id"],
  },
  dim_bill_of_materials: {
    file: "dim_bill_of_materials.csv",
    required: [
      "menu_item_id",
      "ingredient_id",
      "ingredient_name",
      "quantity_per_serving",
      "stock_unit",
      "unit_cost",
      "shelf_life_days",
    ],
    keys: ["menu_item_id", "ingredient_id"],
  },
};

const DATASET_ALIASES = {
  orders: "fct_orders",
  "sales orders": "fct_orders",
  "sales items": "fct_order_items",
  "order items": "fct_order_items",
  inventory: "fct_inventory_reports",
  "inventory reports": "fct_inventory_reports",
  items: "dim_items",
  products: "dim_items",
  places: "dim_places",
  merchants: "dim_places",
  "bill of materials": "dim_bill_of_materials",
  bom: "dim_bill_of_materials",
  "menu items": "dim_menu_items",
  "menu item add ons": "dim_menu_item_add_ons",
  "add ons": "dim_add_ons",
  campaigns: "dim_campaigns",
  "bonus codes": "fct_bonus_codes",
  "invoice items": "fct_invoice_items",
  "cash balances": "fct_cash_balances",
  users: "dim_users",
  "taxonomy terms": "dim_taxonomy_terms",
  "stock categories": "dim_stock_categories",
  skus: "dim_skus",
  "most ordered": "most_ordered",
};

const resolveRoot = () =>
  process.env.FRESHFLOW_ROOT
    ? path.resolve(process.env.FRESHFLOW_ROOT)
    : path.resolve(process.cwd(), "..", "..");

export const resolveDataDir = () => {
  if (process.env.FRESHFLOW_DATA_DIR) {
    return path.resolve(process.env.FRESHFLOW_DATA_DIR);
  }
  return path.join(resolveRoot(), "data");
};

const readCsvHeader = async (filePath) => {
  const handle = await fs.open(filePath, "r");
  try {
    const buffer = Buffer.alloc(65536);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    const snippet = buffer.toString("utf8", 0, bytesRead);
    const firstLine = snippet.split(/\r?\n/)[0];
    const parsed = parseCsv(`${firstLine}\n`);
    return parsed.columns || [];
  } catch (error) {
    return [];
  } finally {
    await handle.close();
  }
};

const inferKeysFromColumns = (columns) => {
  const columnSet = new Set(columns);
  if (columnSet.has("id")) return ["id"];
  if (columnSet.has("order_id") && columnSet.has("item_id")) return ["order_id", "item_id"];
  if (columnSet.has("report_date") && columnSet.has("item_id")) return ["report_date", "item_id"];
  if (columnSet.has("menu_item_id") && columnSet.has("ingredient_id")) {
    return ["menu_item_id", "ingredient_id"];
  }
  if (columnSet.has("menu_item_id") && columnSet.has("add_on_id")) {
    return ["menu_item_id", "add_on_id"];
  }
  if (columnSet.has("code")) return ["code"];
  if (columnSet.has("campaign_id")) return ["campaign_id"];
  return [];
};

let registryCache = null;
let registryTimestamp = 0;

export const loadDatasetRegistry = async () => {
  const now = Date.now();
  if (registryCache && now - registryTimestamp < 30000) {
    return registryCache;
  }

  const dataDir = resolveDataDir();
  const registry = { ...DATASET_DEFS };

  try {
    const files = await fs.readdir(dataDir);
    const csvFiles = files.filter((file) => file.toLowerCase().endsWith(".csv"));

    for (const file of csvFiles) {
      const base = file.replace(/\.csv$/i, "");
      if (!registry[base]) {
        registry[base] = { file, required: [], keys: [] };
      }
      const filePath = path.join(dataDir, file);
      const columns = await readCsvHeader(filePath);
      registry[base].columns = columns;
      registry[base].keys = registry[base].keys?.length ? registry[base].keys : inferKeysFromColumns(columns);
    }
  } catch (error) {
    // ignore
  }

  registryCache = registry;
  registryTimestamp = now;
  return registry;
};

const normalizeDatasetKey = (value) =>
  (value || "")
    .toLowerCase()
    .replace(/\.csv$/i, "")
    .replace(/[^a-z0-9_\s-]+/g, "")
    .trim();

export const resolveDatasetMeta = async ({ dataset, columns }) => {
  const registry = await loadDatasetRegistry();
  const normalized = normalizeDatasetKey(dataset);

  if (normalized) {
    const aliasKey = DATASET_ALIASES[normalized];
    if (aliasKey && registry[aliasKey]) {
      const entry = registry[aliasKey];
      return {
        name: aliasKey,
        file: entry.file || `${aliasKey}.csv`,
        required: entry.required || [],
        keys: entry.keys || [],
        columns: entry.columns || [],
      };
    }

    if (registry[normalized]) {
      const entry = registry[normalized];
      return {
        name: normalized,
        file: entry.file || `${normalized}.csv`,
        required: entry.required || [],
        keys: entry.keys || [],
        columns: entry.columns || [],
      };
    }
  }

  if (columns && columns.length) {
    let best = null;
    let bestScore = 0;
    const columnSet = new Set(columns.map((value) => value.toLowerCase()));

    for (const [name, entry] of Object.entries(registry)) {
      const required = entry.required || [];
      if (required.length) {
        const requiredMatches = required.filter((col) => columnSet.has(col.toLowerCase())).length;
        if (requiredMatches === required.length && requiredMatches > bestScore) {
          best = {
            name,
            file: entry.file || `${name}.csv`,
            required,
            keys: entry.keys || [],
            columns: entry.columns || [],
          };
          bestScore = requiredMatches;
        }
      } else if (entry.columns && entry.columns.length) {
        const overlap = entry.columns.filter((col) => columnSet.has(col.toLowerCase())).length;
        const score = overlap / Math.max(1, columns.length);
        if (score > 0.6 && score > bestScore) {
          best = {
            name,
            file: entry.file || `${name}.csv`,
            required,
            keys: entry.keys || [],
            columns: entry.columns || [],
          };
          bestScore = score;
        }
      }
    }

    if (best) return best;
  }

  return null;
};

export const listAvailableDatasets = async () => {
  const registry = await loadDatasetRegistry();
  return Object.keys(registry);
};
