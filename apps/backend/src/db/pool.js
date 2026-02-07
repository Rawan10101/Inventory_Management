import pg from "pg";

const { Pool } = pg;

const buildConfig = () => {
  if (process.env.DATABASE_URL) {
    const sslRequired = process.env.PGSSLMODE === "require" || process.env.DATABASE_SSL === "true";
    return {
      connectionString: process.env.DATABASE_URL,
      ssl: sslRequired ? { rejectUnauthorized: false } : false,
    };
  }

  return {
    host: process.env.PGHOST || "localhost",
    port: Number(process.env.PGPORT || 5432),
    user: process.env.PGUSER || "postgres",
    password: process.env.PGPASSWORD || "postgres",
    database: process.env.PGDATABASE || "fresh_flow",
  };
};

export const pool = new Pool(buildConfig());
