import express from "express";
import { pool } from "../db/pool.js";
import { ingestDataset, modifyDataset, queryExpiringInventory } from "../services/datasetService.js";

const router = express.Router();

router.post("/ingest", async (req, res, next) => {
  try {
    const result = await ingestDataset(req.body || {});
    return res.json(result);
  } catch (error) {
    return next(error);
  }
});

router.post("/modify", async (req, res, next) => {
  try {
    const result = await modifyDataset(req.body || {});
    return res.json(result);
  } catch (error) {
    return next(error);
  }
});

router.get("/inventory/expiring", async (req, res, next) => {
  try {
    const days = Number(req.query.days || 3);
    const items = await queryExpiringInventory(days);
    return res.json({ days, items });
  } catch (error) {
    return next(error);
  }
});

router.get("/predictions", async (req, res, next) => {
  try {
    const result = await pool.query(
      "SELECT id, item_id, location_id, period, predicted_demand, confidence, generated_at FROM predictions ORDER BY generated_at DESC LIMIT 50"
    );
    return res.json({ predictions: result.rows });
  } catch (error) {
    return next(error);
  }
});

router.get("/promotions", async (req, res, next) => {
  try {
    const result = await pool.query(
      "SELECT id, name, start_date, end_date, items, strategy, status, created_at FROM promotions ORDER BY created_at DESC LIMIT 50"
    );
    return res.json({ promotions: result.rows });
  } catch (error) {
    return next(error);
  }
});

router.get("/analytics/summary", async (req, res, next) => {
  try {
    const [inventoryCount, salesCount, predictionCount, promotionCount] = await Promise.all([
      pool.query("SELECT COUNT(*) FROM inventory"),
      pool.query("SELECT COUNT(*) FROM sales"),
      pool.query("SELECT COUNT(*) FROM predictions"),
      pool.query("SELECT COUNT(*) FROM promotions"),
    ]);

    return res.json({
      inventoryRows: Number(inventoryCount.rows[0].count),
      salesRows: Number(salesCount.rows[0].count),
      predictions: Number(predictionCount.rows[0].count),
      promotions: Number(promotionCount.rows[0].count),
    });
  } catch (error) {
    return next(error);
  }
});

export default router;
