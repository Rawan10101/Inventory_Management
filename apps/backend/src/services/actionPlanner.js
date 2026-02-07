const detectOperation = (message = "") => {
  if (/\b(delete|remove|wipe|overwrite)\b/i.test(message)) return "delete";
  if (/\b(update|modify|edit|change)\b/i.test(message)) return "update";
  if (/\b(add|append|insert)\b/i.test(message)) return "insert";
  return "insert";
};

export const planAction = ({ message, context, intentResult, parsedData }) => {
  const { intent, entities } = intentResult;
  const promotionKeyword = /\b(promo|promotion|promotions|discount|markdown|percent)\b/i.test(message || "");
  const resolvedIntent =
    intent === "general_help" && parsedData.rows.length
      ? "data_ingestion"
      : intent === "dataset_modification" && promotionKeyword
        ? "promotion_planning"
        : intent;
  const dataset = entities.dataset || context?.lastDataset || "sales";
  const horizon = entities.horizon || (entities.period === "weekly" ? "week" : "day") || "week";
  const expiringDays = entities.expiringDays || 3;
  const discountPercent = entities.discountPercent ?? null;
  const scopeAll = entities.scopeAll === true;

  let action = null;
  let requiresConfirmation = false;
  let assistantMessage = "";
  let actionPreview = null;

  switch (resolvedIntent) {
    case "data_ingestion": {
      if (!parsedData.rows.length) {
        assistantMessage = "Please include the rows to ingest as CSV, JSON, or key:value lines.";
        break;
      }
      action = {
        type: "ingest_dataset",
        dataset,
        format: parsedData.format,
        columns: parsedData.columns,
        rows: parsedData.rows,
      };
      assistantMessage = `Ready to ingest ${parsedData.rows.length || 0} rows into the ${dataset} dataset.`;
      actionPreview = {
        dataset,
        rows: parsedData.rows.slice(0, 5),
        totalRows: parsedData.rows.length,
      };
      break;
    }
    case "dataset_modification": {
      if (!parsedData.rows.length) {
        assistantMessage = "Please include the rows to modify so I can apply the change.";
        break;
      }
      const operation = detectOperation(message);
      action = {
        type: "modify_dataset",
        dataset,
        operation,
        rows: parsedData.rows,
      };
      requiresConfirmation = operation === "delete" || /overwrite/i.test(message);
      assistantMessage = `Prepared a ${operation} operation on ${dataset}.`;
      actionPreview = {
        dataset,
        operation,
        rows: parsedData.rows.slice(0, 5),
        totalRows: parsedData.rows.length,
      };
      break;
    }
    case "demand_forecast": {
      action = { type: "forecast_demand", dataset, horizon };
      assistantMessage = `Forecasting demand for the next ${horizon}.`;
      actionPreview = { dataset, horizon };
      break;
    }
    case "promotion_planning": {
      const scope = scopeAll ? "all" : "expiring";
      action = {
        type: "create_promotion",
        dataset: "inventory",
        scope,
        expiringDays: scopeAll ? null : expiringDays,
        discountPercent,
      };
      if (scopeAll) {
        assistantMessage = `Drafting promotion for all products${discountPercent ? ` at ${discountPercent}%` : ""}.`;
      } else {
        assistantMessage = `Drafting promotion strategy for items expiring in ${expiringDays} days${discountPercent ? ` at ${discountPercent}%` : ""}.`;
      }
      actionPreview = { scope, expiringDays: scopeAll ? null : expiringDays, discountPercent };
      requiresConfirmation = true;
      break;
    }
    case "query_expiring": {
      action = { type: "query_expiring", days: expiringDays };
      assistantMessage = `Fetching items expiring in ${expiringDays} days.`;
      actionPreview = { expiringDays };
      break;
    }
    case "inventory_decision": {
      action = { type: "recommend_prep", horizon, dataset };
      assistantMessage = `Preparing inventory recommendations for the next ${horizon}.`;
      actionPreview = { horizon, dataset };
      break;
    }
    default: {
      assistantMessage = "Tell me what you want to do with inventory, sales, forecasts, or promotions.";
    }
  }

  return {
    action,
    requiresConfirmation,
    assistantMessage,
    actionPreview,
    resolvedIntent,
  };
};
