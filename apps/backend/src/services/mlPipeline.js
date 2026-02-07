import path from "path";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const resolveRoot = () =>
  process.env.FRESHFLOW_ROOT
    ? path.resolve(process.env.FRESHFLOW_ROOT)
    : path.resolve(process.cwd(), "..", "..");

export const runDemandForecast = async ({ horizon }) => {
  if (process.env.PIPELINE_CMD) {
    const { stdout } = await execAsync(process.env.PIPELINE_CMD, {
      cwd: resolveRoot(),
    });
    return {
      status: "triggered",
      horizon,
      detail: stdout.trim() || "Pipeline executed.",
    };
  }

  return {
    status: "simulated",
    horizon,
    predictions: [
      { item_id: "SKU-001", location_id: "LOC-01", predicted_demand: 120 },
      { item_id: "SKU-002", location_id: "LOC-01", predicted_demand: 85 },
    ],
  };
};
