# API Specification

These endpoints are defined in `src/api/routes.py`. The current implementation returns placeholder values for demonstration.

| Method | Path | Description |
| --- | --- | --- |
| GET | /api/health | Health check. |
| POST | /api/inventory/predict | Predict demand for an item. |
| POST | /api/menu/analyze | Menu engineering analysis. |
| POST | /api/shifts/optimize | Shift optimization based on demand. |

## POST /api/inventory/predict

Request body:

```json
{
  "item_id": "string",
  "period": "daily"
}
```

Response body:

```json
{
  "item_id": "string",
  "predicted_demand": 150.5,
  "period": "daily",
  "confidence": 0.85,
  "timestamp": "2026-02-02T12:00:00Z"
}
```

## POST /api/menu/analyze

Request body:

```json
{
  "place_id": "string",
  "analysis_type": "profitability"
}
```

Response body:

```json
{
  "place_id": "string",
  "analysis_type": "profitability",
  "stars": [],
  "plowhorses": [],
  "puzzles": [],
  "dogs": [],
  "timestamp": "2026-02-02T12:00:00Z"
}
```

## POST /api/shifts/optimize

Request body:

```json
{
  "place_id": "string",
  "date": "YYYY-MM-DD",
  "constraints": {}
}
```

Response body:

```json
{
  "place_id": "string",
  "date": "YYYY-MM-DD",
  "shifts": [],
  "total_staff_hours": 120,
  "estimated_coverage": 0.95,
  "timestamp": "2026-02-02T12:00:00Z"
}
```
