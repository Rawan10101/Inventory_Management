CREATE TABLE IF NOT EXISTS inventory (
  id SERIAL PRIMARY KEY,
  item_id TEXT,
  item_name TEXT,
  location_id TEXT,
  quantity NUMERIC,
  unit TEXT,
  expiry_date DATE,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sales (
  id SERIAL PRIMARY KEY,
  item_id TEXT,
  location_id TEXT,
  sale_date DATE,
  quantity NUMERIC,
  revenue NUMERIC,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  item_id TEXT,
  location_id TEXT,
  period TEXT,
  predicted_demand NUMERIC,
  confidence NUMERIC,
  generated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS promotions (
  id SERIAL PRIMARY KEY,
  name TEXT,
  start_date DATE,
  end_date DATE,
  items JSONB,
  strategy TEXT,
  status TEXT DEFAULT 'draft',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS datasets (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  schema JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_rows (
  id SERIAL PRIMARY KEY,
  dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chatbot_logs (
  id SERIAL PRIMARY KEY,
  conversation_id TEXT,
  user_message TEXT,
  assistant_message TEXT,
  action_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inventory_expiry ON inventory (expiry_date);
CREATE INDEX IF NOT EXISTS idx_sales_item_date ON sales (item_id, sale_date);
