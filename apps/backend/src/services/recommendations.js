import { queryExpiringInventory } from "./datasetService.js";

export const generatePrepPlan = async ({ horizon }) => ({
  horizon,
  recommendations: [
    { item_id: "SKU-001", suggested_prep: 120, rationale: "High velocity item" },
    { item_id: "SKU-002", suggested_prep: 75, rationale: "Stable baseline" },
  ],
});

export const generatePromotionPlan = async ({ expiringDays, discountPercent, scope }) => {
  const discountText = discountPercent ? `${discountPercent}%` : "a promotional";
  if (scope === "all") {
    return {
      expiringDays: null,
      items: [],
      strategy: `Apply ${discountText} discount across all products and monitor margin impact.`,
    };
  }

  const expiring = await queryExpiringInventory(expiringDays);
  return {
    expiringDays,
    items: expiring.slice(0, 10),
    strategy: `Bundle near-expiry items with top sellers and apply ${discountText} markdown.`,
  };
};
