"""
Run the Flask API defined in src/api/routes.py.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
API_DIR = ROOT_DIR / "src" / "api"
if str(API_DIR) not in sys.path:
    sys.path.append(str(API_DIR))

from routes import app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
