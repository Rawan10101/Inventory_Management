import {
  ingestDataset,
  modifyDataset,
  queryExpiringInventory,
  insertPredictions,
  insertPromotion,
} from "./datasetService.js";
import { runDemandForecast } from "./mlPipeline.js";
import { generatePrepPlan, generatePromotionPlan } from "./recommendations.js";
import { pool } from "../db/pool.js";

const logAction = async ({ conversationId, userMessage, assistantMessage, action }) => {
  try {
    await pool.query(
      "INSERT INTO chatbot_logs (conversation_id, user_message, assistant_message, action_json) VALUES ($1, $2, $3, $4)",
      [conversationId, userMessage, assistantMessage, action]
    );
  } catch (error) {
    // Logging should never break user-facing flows.
  }
};

export const executeAction = async ({ action, conversationId, userMessage }) => {
  if (!action) {
    return { assistantMessage: "I need more detail to proceed.", structuredOutput: null };
  }

  let result = null;
  let assistantMessage = "";

  switch (action.type) {
    case "ingest_dataset": {
      const ingest = await ingestDataset(action);
      assistantMessage = `${ingest.count} rows added to ${ingest.dataset}. Snapshot saved.`;
      result = ingest;
      break;
    }
    case "modify_dataset": {
      const change = await modifyDataset(action);
      assistantMessage = `${change.count} rows ${change.operation}d in ${action.dataset}.`;
      result = change;
      break;
    }
    case "forecast_demand": {
      result = await runDemandForecast({ horizon: action.horizon });
      if (result?.predictions?.length) {
        await insertPredictions({
          period: action.horizon,
          predictions: result.predictions,
        });
      }
      assistantMessage = `Demand forecast pipeline ${result.status} for the next ${action.horizon}.`;
      break;
    }
    case "query_expiring": {
      const items = await queryExpiringInventory(action.days);
      assistantMessage = `${items.length} items are expiring within ${action.days} days.`;
      result = { items };
      break;
    }
    case "create_promotion": {
      result = await generatePromotionPlan({
        expiringDays: action.expiringDays,
        discountPercent: action.discountPercent,
        scope: action.scope,
      });
      const discountLabel = action.discountPercent ? ` ${action.discountPercent}%` : "";
      const name =
        action.scope === "all"
          ? `Global promo${discountLabel}`.trim()
          : `Near-expiry promo (${action.expiringDays} days${discountLabel})`.trim();
      const promotionId = await insertPromotion({
        name,
        startDate: new Date().toISOString().slice(0, 10),
        endDate: new Date(Date.now() + 7 * 86400000).toISOString().slice(0, 10),
        items: result.items,
        strategy: result.strategy,
      });
      result.promotionId = promotionId;
      if (action.scope === "all") {
        assistantMessage = `Promotion prepared for all products${action.discountPercent ? ` at ${action.discountPercent}%` : ""}.`;
      } else {
        assistantMessage = `Promotion strategy prepared for ${result.items.length} near-expiry items${action.discountPercent ? ` at ${action.discountPercent}%` : ""}.`;
      }
      break;
    }
    case "recommend_prep": {
      result = await generatePrepPlan({ horizon: action.horizon });
      assistantMessage = `Prep recommendations ready for the next ${action.horizon}.`;
      break;
    }
    default: {
      assistantMessage = "Action not supported yet.";
      result = null;
    }
  }

  await logAction({ conversationId, userMessage, assistantMessage, action });

  return { assistantMessage, structuredOutput: result };
};
