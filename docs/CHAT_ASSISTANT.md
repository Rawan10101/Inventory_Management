# Fresh Flow AI Assistant

## What's Included

- React chat panel with action previews, confirmations, and file upload.
- Node/Express API that classifies intents, sanitizes input, and executes actions.
- PostgreSQL schema for inventory, sales, predictions, promotions, datasets, and chatbot logs.

## Backend Setup

1. Install dependencies in `apps/backend`.
2. Configure `.env` (see `apps/backend/.env.example`).
3. Apply schema from `apps/backend/sql/schema.sql` to your PostgreSQL database.
4. Start the API with `npm run dev`.

## Frontend Setup

1. Install dependencies in `apps/frontend`.
2. Set `VITE_API_BASE` (and optionally `VITE_API_KEY`).
3. Run `npm run dev`.

## Notes

- Destructive operations require confirmation.
- The assistant stores CSV snapshots in `data/assistant`.
- Forecasting can be wired to the Python pipeline with `PIPELINE_CMD`.
