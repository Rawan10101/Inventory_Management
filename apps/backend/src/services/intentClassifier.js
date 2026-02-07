const intentPatterns = [
  { intent: "promotion_planning", patterns: [/\b(promo|promotion|promotions|discount|bundle|markdown)\b/i] },
  { intent: "data_ingestion", patterns: [/\b(upload|import|ingest|load)\b/i] },
  { intent: "dataset_modification", patterns: [/\b(update|modify|edit|change|append|add)\b/i] },
  {
    intent: "demand_forecast",
    patterns: [/\b(predict|forecast|demand|train|retrain|model|pipeline)\b/i],
  },
  { intent: "query_expiring", patterns: [/\b(expire|expiry|expiring)\b/i] },
  { intent: "inventory_decision", patterns: [/\b(prep|prepare|restock|reorder|prioritize)\b/i] },
];

const datasetKeywords = [
  { name: "sales", patterns: [/\bsales\b/i, /order items?/i] },
  { name: "inventory", patterns: [/\binventory\b/i, /stock/i] },
  { name: "promotions", patterns: [/\bpromotions?\b/i, /campaign/i] },
];

export const classifyIntent = (message = "") => {
  const matched = intentPatterns.find((entry) => entry.patterns.some((pattern) => pattern.test(message)));
  let intent = matched ? matched.intent : "general_help";

  const datasetMatch = datasetKeywords.find((entry) =>
    entry.patterns.some((pattern) => pattern.test(message))
  );

  const horizonMatch = message.match(/next\s+(day|week|month|quarter|year)/i);
  const expiringMatch = message.match(/expire\w*\s+in\s+(\d+)\s+days?/i);
  const periodMatch = message.match(/(daily|weekly|monthly)/i);
  const percentMatch = message.match(/(\d+(?:\.\d+)?)\s*(%|percent|percent\s+off|off)/i);
  const scopeAllMatch = message.match(
    /\b(all products|all the products|all items|all the items|all inventory|everything|entire catalog|whole catalog)\b/i
  );
  const promotionKeyword = /\b(promo|promotion|promotions|discount|markdown|percent)\b/i.test(message);
  if (promotionKeyword) {
    intent = "promotion_planning";
  }

  return {
    intent,
    confidence: matched ? 0.82 : 0.4,
    entities: {
      dataset: datasetMatch ? datasetMatch.name : null,
      horizon: horizonMatch ? horizonMatch[1].toLowerCase() : null,
      period: periodMatch ? periodMatch[1].toLowerCase() : null,
      expiringDays: expiringMatch ? Number(expiringMatch[1]) : null,
      discountPercent: percentMatch ? Number(percentMatch[1]) : null,
      scopeAll: Boolean(scopeAllMatch),
    },
  };
};
