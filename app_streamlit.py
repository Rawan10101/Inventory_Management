from __future__ import annotations

from pathlib import Path
import sys
import json
import io
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_PDF = True
except Exception:
    HAS_PDF = False

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = ROOT_DIR / "data" / "Inventory_Management"
REPORTS_DIR_DEFAULT = ROOT_DIR / "reports"
STATE_FILE = DATA_DIR_DEFAULT.parent / ".system_state.json"
BACKUP_DIR = DATA_DIR_DEFAULT.parent / "backups"

PIPELINE_DIR = ROOT_DIR / "src" / "pipeline"
MODELS_DIR = ROOT_DIR / "src" / "models"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

try:
    from pipeline_runner import run_pipeline
    HAS_PIPELINE = True
except Exception:
    HAS_PIPELINE = False
    print("Pipeline not available. Using mock mode.")

DATASETS = {
    "Sales orders": {
        "file": "fct_orders.csv",
        "required": ["id", "place_id", "created", "status"],
        "key": ["id"],
        "date_cols": ["created"],
    },
    "Sales items": {
        "file": "fct_order_items.csv",
        "required": ["order_id", "item_id", "quantity", "price"],
        "key": ["order_id", "item_id"],
        "numeric_cols": ["quantity", "price"],
    },
    "Inventory reports": {
        "file": "fct_inventory_reports.csv",
        "required": ["report_date", "item_id", "quantity_on_hand"],
        "key": ["report_date", "item_id"],
        "date_cols": ["report_date"],
        "numeric_cols": ["quantity_on_hand", "unit_cost", "total_value"],
    },
    "Product list": {
        "file": "dim_items.csv",
        "required": ["id", "title", "manage_inventory"],
        "key": ["id"],
        "numeric_cols": ["manage_inventory", "price", "unit_cost"],
    },
}
DATASET_OPTIONS = list(DATASETS.keys()) + ["Other dataset"]

st.set_page_config(
    page_title="Fresh Flow Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Black & Green CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    }

    /* Prevent dimming during rerun/loading */
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewContainer"] > .main,
    div[data-testid="stAppViewContainer"] > .main > div {
        opacity: 1 !important;
        filter: none !important;
    }

    div[data-testid="stSpinner"] {
        background: transparent !important;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #4caf50 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 25px 50px rgba(46, 125, 50, 0.4);
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        text-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.25rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 60px rgba(46, 125, 50, 0.3);
        border-color: rgba(76, 175, 80, 0.5);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2e7d32, #4caf50, #81c784);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
        font-weight: 500;
        margin: 0;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Section Headers */
    .section-header {
        background: rgba(46, 125, 50, 0.2);
        backdrop-filter: blur(15px);
        padding: 1.8rem 2.5rem;
        border-radius: 16px;
        margin: 3rem 0 2rem 0;
        border: 1px solid rgba(76, 175, 80, 0.3);
        display: flex;
        align-items: center;
    }
    
    .section-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* Alert System */
    .alert-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 5px solid;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .alert-critical {
        background: rgba(244, 67, 54, 0.15);
        border-color: #f44336;
        color: #ffffff;
    }
    
    .alert-info {
        background: rgba(33, 150, 243, 0.15);
        border-color: #2196f3;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.4);
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(46, 125, 50, 0.5);
        background: linear-gradient(135deg, #388e3c 0%, #66bb6a 100%);
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(46, 125, 50, 0.2);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 8px;
        gap: 8px;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.15) !important;
        border-radius: 12px;
        padding: 1.2rem 2rem !important;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
        color: white !important;
        font-weight: 500;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
    }
    
    /* Footer */
    .footer {
        background: rgba(17, 17, 17, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-top: 4rem;
        text-align: center;
        border: 1px solid rgba(46, 125, 50, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# All helper functions remain exactly the same (unchanged functionality)
def _latest_report(report_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        report_dir.glob(f"{prefix}_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

def _read_csv_with_fallback(source) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(source, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    try:
        return pd.read_csv(source, encoding="latin1", encoding_errors="ignore", low_memory=False)
    except Exception:
        return pd.DataFrame()

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return _read_csv_with_fallback(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _read_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return _read_csv_with_fallback(uploaded_file)
    if name.endswith(".xlsx"):
        try:
            return pd.read_excel(uploaded_file)
        except Exception as exc:
            raise ValueError("Excel support requires openpyxl.") from exc
    raise ValueError("Unsupported file type. Use .csv or .xlsx")

def _load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_state(state: Dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def _load_latest_reports(report_dir: Path) -> Dict[str, pd.DataFrame]:
    report_dir = Path(report_dir)
    reports = {}
    for prefix in ["summary", "forecast", "recommendations", "promotions", "prep_plan", "model_metrics"]:
        path = _latest_report(report_dir, prefix)
        if path:
            df = _safe_read_csv(path)
            if not df.empty:
                column_mapping = {
                    'item_name': 'item_name',
                    'title': 'item_name',
                    'name': 'item_name',
                    'product': 'item_name',
                }
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df = df.rename(columns={old_col: new_col})
            reports[prefix] = df
    return reports

def _create_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    if forecast_df.empty:
        return go.Figure()
    
    pred_col = None
    for col in ['predicted_daily_demand', 'predicted_demand', 'demand', 'quantity']:
        if col in forecast_df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        return go.Figure()
    
    top_items = forecast_df.nlargest(15, pred_col)
    name_col = 'item_name' if 'item_name' in top_items.columns else 'item_id'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_items[name_col].astype(str),
        y=top_items[pred_col],
        marker=dict(color='#4caf50', line=dict(color='#2e7d32', width=2)),
        text=top_items[pred_col].round(1),
        textposition='outside',
        textfont=dict(size=12, color='white')
    ))
    
    fig.update_layout(
        title="Top 15 Products - Demand Forecast",
        xaxis_title="Product",
        yaxis_title="Expected Daily Demand",
        height=500,
        template='plotly_dark',
        xaxis_tickangle=-45,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title_font=dict(size=20, color='white')
    )
    return fig

def _create_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or 'date' not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    if 'quantity_sold' in df.columns:
        qty_col = 'quantity_sold'
    elif 'quantity' in df.columns:
        qty_col = 'quantity'
    else:
        return go.Figure()
    
    trend_df = df.groupby('date')[qty_col].sum().reset_index()
    trend_df.columns = ['date', 'total']
    trend_df = trend_df.sort_values('date')
    
    fig.add_trace(go.Scatter(
        x=trend_df['date'],
        y=trend_df['total'],
        mode='lines+markers',
        line=dict(color='#4caf50', width=4),
        marker=dict(size=8, color='#2e7d32'),
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.2)'
    ))
    
    if len(trend_df) >= 7:
        trend_df['ma7'] = trend_df['total'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=trend_df['date'],
            y=trend_df['ma7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#81c784', width=3, dash='dash')
        ))
    
    fig.update_layout(
        title="Sales Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=450,
        template='plotly_dark',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def _create_risk_distribution(recommendations_df: pd.DataFrame) -> go.Figure:
    if recommendations_df.empty or 'risk_category' not in recommendations_df.columns:
        return go.Figure()
    
    risk_counts = recommendations_df['risk_category'].value_counts()
    
    colors = {
        'Critical': '#f44336', 'critical': '#f44336',
        'High': '#ff9800', 'high': '#ff9800',
        'Medium': '#ffeb3b', 'medium': '#ffeb3b',
        'Low': '#4caf50', 'low': '#4caf50'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker=dict(colors=[colors.get(cat, '#757575') for cat in risk_counts.index]),
        textinfo='label+percent+value',
        textposition='inside',
        textfont=dict(color='white', size=11)
    )])
    
    fig.update_layout(
        title="Inventory Risk Distribution",
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=18, color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0.7)')
    )
    return fig

def _load_daily_sales(data_dir: Path) -> pd.DataFrame:
    orders_path = Path(data_dir) / "fct_orders.csv"
    items_path = Path(data_dir) / "fct_order_items.csv"
    
    items = _safe_read_csv(items_path)
    orders = _safe_read_csv(orders_path)

    if items.empty or orders.empty:
        return pd.DataFrame()

    if 'created' in orders.columns:
        orders["order_created_at"] = pd.to_datetime(orders["created"], unit="s", errors="coerce")
    elif 'order_created_at' not in orders.columns:
        return pd.DataFrame()
    
    if 'status' in orders.columns:
        orders = orders[orders["status"] == "Closed"]

    merged = items.merge(
        orders[["id", "place_id", "order_created_at"]],
        left_on="order_id",
        right_on="id",
        how="left",
    )
    
    if 'order_created_at' not in merged.columns:
        return pd.DataFrame()
    
    merged["date"] = pd.to_datetime(merged["order_created_at"], errors="coerce").dt.date

    qty_col = 'quantity' if 'quantity' in merged.columns else 'quantity_sold'
    daily = (
        merged.groupby(["item_id", "date"], dropna=True)[qty_col]
        .sum()
        .reset_index()
        .rename(columns={qty_col: "quantity_sold"})
    )

    items_dim = _safe_read_csv(Path(data_dir) / "dim_items.csv")
    if not items_dim.empty and "id" in items_dim.columns and "title" in items_dim.columns:
        daily = daily.merge(items_dim[["id", "title"]], left_on="item_id", right_on="id", how="left")
        daily = daily.rename(columns={"title": "item_name"})

    return daily

def _trend_label(series: pd.Series) -> str:
    series = series.dropna()
    if len(series) < 14:
        return "Stable"
    last = series.tail(7).mean()
    prev = series.iloc[-14:-7].mean()
    if prev == 0:
        return "Increasing" if last > 0 else "Stable"
    change = (last - prev) / prev
    if change > 0.05:
        return "Increasing"
    if change < -0.05:
        return "Decreasing"
    return "Stable"


def _confidence_label(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "Low"
    mean = series.mean()
    if mean == 0:
        return "Low"
    cv = series.std() / mean if mean else 1.0
    if cv < 0.25:
        return "High"
    if cv < 0.5:
        return "Medium"
    return "Low"


def _load_inventory_snapshot(data_dir: Path) -> pd.DataFrame:
    inv = _safe_read_csv(Path(data_dir) / "fct_inventory_reports.csv")
    required_cols = {"item_id", "quantity_on_hand"}

    if not inv.empty and required_cols.issubset(inv.columns):
        if "report_date" in inv.columns:
            inv["report_date"] = pd.to_datetime(inv["report_date"], errors="coerce")
            latest_date = inv["report_date"].max()
            inv = inv[inv["report_date"] == latest_date]

        items_dim = _safe_read_csv(Path(data_dir) / "dim_items.csv")
        if not items_dim.empty and "id" in items_dim.columns and "title" in items_dim.columns:
            inv = inv.merge(items_dim[["id", "title"]], left_on="item_id", right_on="id", how="left")
            inv = inv.rename(columns={"title": "item_name"})
        return inv

    return pd.DataFrame()


def _inventory_schema_issue(data_dir: Path) -> str | None:
    inv_path = Path(data_dir) / "fct_inventory_reports.csv"
    inv = _safe_read_csv(inv_path)
    if inv.empty:
        return "Inventory report is missing or empty. Upload an inventory report to show ordering needs."
    missing = {"item_id", "quantity_on_hand"} - set(inv.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        return f"Inventory report missing columns: {missing_cols}"
    return None


def _style_risk(df: pd.DataFrame):
    if "Risk" not in df.columns:
        return df

    def _color(value: str) -> str:
        text = str(value).lower()
        if "high" in text:
            return "background-color: #ffcccc"
        if "medium" in text:
            return "background-color: #fff2cc"
        if "low" in text:
            return "background-color: #d9ead3"
        return ""

    return df.style.applymap(_color, subset=["Risk"])
def _to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()
    except Exception:
        return None


def _assistant_init_state() -> None:
    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": (
                    "I can ingest data, update inventory, run forecasts, and plan promotions. "
                    "Tell me what you need or attach a CSV/JSON file."
                ),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    if "assistant_conversation_id" not in st.session_state:
        st.session_state.assistant_conversation_id = None
    if "assistant_action_preview" not in st.session_state:
        st.session_state.assistant_action_preview = None
    if "assistant_structured_output" not in st.session_state:
        st.session_state.assistant_structured_output = None
    if "assistant_pending_confirmation" not in st.session_state:
        st.session_state.assistant_pending_confirmation = None
    if "assistant_api_base" not in st.session_state:
        st.session_state.assistant_api_base = os.environ.get("ASSISTANT_API_BASE", "http://localhost:4000")
    if "assistant_api_key" not in st.session_state:
        st.session_state.assistant_api_key = os.environ.get("ASSISTANT_API_KEY", "")
    if "assistant_mode" not in st.session_state:
        st.session_state.assistant_mode = "API"
    if "assistant_last_decision" not in st.session_state:
        st.session_state.assistant_last_decision = None
    if "assistant_last_outcome" not in st.session_state:
        st.session_state.assistant_last_outcome = None
    if "assistant_action_log" not in st.session_state:
        st.session_state.assistant_action_log = []


def _assistant_add_message(role: str, content: str) -> None:
    st.session_state.assistant_messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def _assistant_log_event(event: dict) -> None:
    st.session_state.assistant_action_log.insert(0, event)
    st.session_state.assistant_action_log = st.session_state.assistant_action_log[:10]


def _assistant_post(path: str, payload: dict) -> dict:
    api_base = st.session_state.assistant_api_base.rstrip("/")
    api_key = st.session_state.assistant_api_key
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(
            f"{api_base}{path}",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Assistant API is unreachable. Start the backend (apps/backend) or switch to Local mode."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Assistant API timed out. Try again or switch to Local mode.") from exc
    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = {}
        detail = error_body.get("detail") or error_body.get("error") or response.text
        raise RuntimeError(detail or "Assistant request failed.")
    return response.json()


def _assistant_check_health() -> tuple[bool, str]:
    api_base = st.session_state.assistant_api_base.rstrip("/")
    try:
        response = requests.get(f"{api_base}/api/health", timeout=3)
        if response.ok:
            return True, "Connected"
        return False, f"API error ({response.status_code})"
    except Exception:
        return False, "API unreachable"


def _assistant_sanitize_dataset(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", (name or "").lower()).strip("-")
    return cleaned[:64] if cleaned else "dataset"


def _assistant_parse_data(message: str, attachment: dict | None) -> dict:
    if attachment and attachment.get("type") == "json":
        try:
            parsed = json.loads(attachment.get("content", ""))
            if isinstance(parsed, list):
                columns = list(parsed[0].keys()) if parsed else []
                return {"format": "json", "columns": columns, "rows": parsed}
        except Exception:
            pass

    if attachment and attachment.get("type") == "csv":
        df = pd.read_csv(io.StringIO(attachment.get("content", "")))
        return {"format": "csv", "columns": list(df.columns), "rows": df.fillna("").to_dict("records")}

    json_match = re.search(r"\[[\s\S]*\]", message)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                columns = list(parsed[0].keys()) if parsed else []
                return {"format": "json", "columns": columns, "rows": parsed}
        except Exception:
            pass

    if "," in message and "\n" in message:
        try:
            df = pd.read_csv(io.StringIO(message))
            return {"format": "csv", "columns": list(df.columns), "rows": df.fillna("").to_dict("records")}
        except Exception:
            pass

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if lines:
        rows = []
        for line in lines:
            parts = [part.strip() for part in line.split(",") if part.strip()]
            row = {}
            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    row[key.strip()] = value.strip()
            if row:
                rows.append(row)
        if rows:
            columns = sorted({key for row in rows for key in row.keys()})
            return {"format": "kv", "columns": columns, "rows": rows}

    return {"format": "none", "columns": [], "rows": []}


def _assistant_classify_intent(message: str) -> dict:
    patterns = [
        ("promotion_planning", [r"\b(promo|promotion|promotions|discount|bundle|markdown)\b"]),
        ("data_ingestion", [r"\b(upload|import|ingest|load)\b"]),
        ("dataset_modification", [r"\b(update|modify|edit|change|append|add|delete|remove|overwrite)\b"]),
        ("demand_forecast", [r"\b(predict|forecast|demand|train|retrain|model|pipeline)\b"]),
        ("query_expiring", [r"\b(expire|expiry|expiring)\b"]),
        ("inventory_decision", [r"\b(prep|prepare|restock|reorder|prioritize)\b"]),
    ]
    intent = "general_help"
    for name, pats in patterns:
        if any(re.search(pat, message, re.IGNORECASE) for pat in pats):
            intent = name
            break

    dataset = None
    if re.search(r"\bsales\b|order items?", message, re.IGNORECASE):
        dataset = "sales"
    elif re.search(r"\binventory\b|stock", message, re.IGNORECASE):
        dataset = "inventory"
    elif re.search(r"\bpromotion\b|campaign", message, re.IGNORECASE):
        dataset = "promotions"

    horizon_match = re.search(r"next\s+(day|week|month|quarter|year)", message, re.IGNORECASE)
    expiring_match = re.search(r"expire\w*\s+in\s+(\d+)\s+days?", message, re.IGNORECASE)
    period_match = re.search(r"(daily|weekly|monthly)", message, re.IGNORECASE)
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|percent\s+off|off)", message, re.IGNORECASE)
    scope_all_match = re.search(
        r"\b(all products|all the products|all items|all the items|all inventory|everything|entire catalog|whole catalog)\b",
        message,
        re.IGNORECASE,
    )
    promotion_keyword = re.search(r"\b(promo|promotion|promotions|discount|markdown|percent)\b", message, re.IGNORECASE)
    if promotion_keyword:
        intent = "promotion_planning"

    return {
        "intent": intent,
        "confidence": 0.8 if intent != "general_help" else 0.4,
        "entities": {
            "dataset": dataset,
            "horizon": horizon_match.group(1).lower() if horizon_match else None,
            "period": period_match.group(1).lower() if period_match else None,
            "expiringDays": int(expiring_match.group(1)) if expiring_match else None,
            "discountPercent": float(percent_match.group(1)) if percent_match else None,
            "scopeAll": bool(scope_all_match),
        },
    }


def _assistant_plan_action(message: str, context: dict, intent_result: dict, parsed: dict) -> dict:
    intent = intent_result["intent"]
    entities = intent_result.get("entities", {})
    promotion_keyword = bool(re.search(r"\b(promo|promotion|promotions|discount|markdown|percent)\b", message, re.IGNORECASE))
    if intent == "dataset_modification" and promotion_keyword:
        resolved_intent = "promotion_planning"
    else:
        resolved_intent = "data_ingestion" if intent == "general_help" and parsed["rows"] else intent
    dataset = entities.get("dataset") or context.get("lastDataset") or "sales"
    horizon = entities.get("horizon") or ("week" if entities.get("period") == "weekly" else "day")
    expiring_days = entities.get("expiringDays") or 3
    discount_percent = entities.get("discountPercent")
    scope_all = entities.get("scopeAll") is True

    action = None
    requires_confirmation = False
    assistant_message = ""
    action_preview = None

    if resolved_intent == "data_ingestion":
        if not parsed["rows"]:
            assistant_message = "Please include rows to ingest as CSV, JSON, or key:value lines."
        else:
            action = {
                "type": "ingest_dataset",
                "dataset": dataset,
                "format": parsed["format"],
                "columns": parsed["columns"],
                "rows": parsed["rows"],
            }
            assistant_message = f"Ready to ingest {len(parsed['rows'])} rows into the {dataset} dataset."
            action_preview = {"dataset": dataset, "rows": parsed["rows"][:5], "totalRows": len(parsed["rows"])}
    elif resolved_intent == "dataset_modification":
        if not parsed["rows"]:
            assistant_message = "Please include the rows to modify so I can apply the change."
        else:
            operation = "delete" if re.search(r"\b(delete|remove|overwrite)\b", message, re.IGNORECASE) else (
                "update" if re.search(r"\b(update|modify|edit|change)\b", message, re.IGNORECASE) else "insert"
            )
            action = {
                "type": "modify_dataset",
                "dataset": dataset,
                "operation": operation,
                "rows": parsed["rows"],
            }
            requires_confirmation = operation == "delete" or "overwrite" in message.lower()
            assistant_message = f"Prepared a {operation} operation on {dataset}."
            action_preview = {
                "dataset": dataset,
                "operation": operation,
                "rows": parsed["rows"][:5],
                "totalRows": len(parsed["rows"]),
            }
    elif resolved_intent == "demand_forecast":
        action = {"type": "forecast_demand", "dataset": dataset, "horizon": horizon}
        assistant_message = f"Forecasting demand for the next {horizon}."
        action_preview = {"dataset": dataset, "horizon": horizon}
    elif resolved_intent == "promotion_planning":
        scope = "all" if scope_all else "expiring"
        action = {
            "type": "create_promotion",
            "dataset": "inventory",
            "scope": scope,
            "expiringDays": None if scope_all else expiring_days,
            "discountPercent": discount_percent,
        }
        if scope_all:
            assistant_message = (
                f"Drafting promotion for all products{f' at {discount_percent}%' if discount_percent else ''}."
            )
        else:
            assistant_message = (
                f"Drafting promotion strategy for items expiring in {expiring_days} days"
                f"{f' at {discount_percent}%' if discount_percent else ''}."
            )
        action_preview = {
            "scope": scope,
            "expiringDays": None if scope_all else expiring_days,
            "discountPercent": discount_percent,
        }
        requires_confirmation = True
    elif resolved_intent == "query_expiring":
        action = {"type": "query_expiring", "days": expiring_days}
        assistant_message = f"Fetching items expiring in {expiring_days} days."
        action_preview = {"expiringDays": expiring_days}
    elif resolved_intent == "inventory_decision":
        action = {"type": "recommend_prep", "horizon": horizon, "dataset": dataset}
        assistant_message = f"Preparing inventory recommendations for the next {horizon}."
        action_preview = {"horizon": horizon, "dataset": dataset}
    else:
        assistant_message = "Tell me what you want to do with inventory, sales, forecasts, or promotions."

    return {
        "resolved_intent": resolved_intent,
        "action": action,
        "requires_confirmation": requires_confirmation,
        "assistant_message": assistant_message,
        "action_preview": action_preview,
    }


def _assistant_local_dataset_path(dataset: str) -> Path:
    base_dir = ROOT_DIR / "data" / "assistant"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{_assistant_sanitize_dataset(dataset)}.csv"


def _assistant_local_snapshot(dataset: str, rows: list[dict]) -> str:
    base_dir = ROOT_DIR / "data" / "assistant"
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_assistant_sanitize_dataset(dataset)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    snapshot_path = base_dir / filename
    pd.DataFrame(rows).to_csv(snapshot_path, index=False)
    return str(snapshot_path)


def _assistant_local_ingest(action: dict) -> dict:
    dataset = action.get("dataset", "dataset")
    rows = action.get("rows", [])
    if not rows:
        return {"count": 0, "dataset": dataset}
    path = _assistant_local_dataset_path(dataset)
    df_new = pd.DataFrame(rows)
    if path.exists():
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    snapshot = _assistant_local_snapshot(dataset, rows)
    return {"count": len(rows), "dataset": dataset, "snapshotPath": snapshot}


def _assistant_local_modify(action: dict) -> dict:
    dataset = action.get("dataset", "dataset")
    operation = action.get("operation", "insert")
    rows = action.get("rows", [])
    path = _assistant_local_dataset_path(dataset)
    df_existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df_updates = pd.DataFrame(rows)
    key_col = None
    for candidate in ["id", "item_id", "sku"]:
        if candidate in df_existing.columns and candidate in df_updates.columns:
            key_col = candidate
            break

    if operation == "insert":
        df = pd.concat([df_existing, df_updates], ignore_index=True)
        df.to_csv(path, index=False)
        return {"count": len(df_updates), "operation": "insert"}

    if key_col is None:
        return {"count": 0, "operation": operation, "error": "No key column found to modify."}

    if operation == "delete":
        keys = df_updates[key_col].dropna().unique().tolist()
        if not keys:
            return {"count": 0, "operation": "delete"}
        mask = df_existing[key_col].isin(keys)
        deleted = int(mask.sum())
        df_existing.loc[~mask].to_csv(path, index=False)
        return {"count": deleted, "operation": "delete"}

    if operation == "update":
        updated = 0
        df = df_existing.copy()
        for _, update in df_updates.iterrows():
            key = update.get(key_col)
            if pd.isna(key):
                continue
            mask = df[key_col] == key
            if not mask.any():
                continue
            for col, value in update.items():
                if pd.notna(value):
                    df.loc[mask, col] = value
            updated += int(mask.sum())
        df.to_csv(path, index=False)
        return {"count": updated, "operation": "update"}

    return {"count": 0, "operation": operation}


def _assistant_local_query_expiring(days: int) -> list[dict]:
    path = _assistant_local_dataset_path("inventory")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    date_col = None
    for candidate in ["expiry_date", "expiration_date", "expiry"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if not date_col:
        return []
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    cutoff = datetime.now().date() + timedelta(days=days)
    filtered = df[df[date_col].notna() & (df[date_col] <= cutoff)]
    return filtered.to_dict("records")


def _assistant_local_execute(action: dict) -> dict:
    action_type = action.get("type")
    if action_type == "ingest_dataset":
        result = _assistant_local_ingest(action)
        message = f"{result.get('count', 0)} rows added to {result.get('dataset')}."
        return {"assistantMessage": message, "structuredOutput": result}
    if action_type == "modify_dataset":
        result = _assistant_local_modify(action)
        message = f"{result.get('count', 0)} rows {result.get('operation')}d in {action.get('dataset')}."
        return {"assistantMessage": message, "structuredOutput": result}
    if action_type == "forecast_demand":
        result = {
            "status": "simulated",
            "horizon": action.get("horizon", "week"),
            "predictions": [
                {"item_id": "SKU-001", "location_id": "LOC-01", "predicted_demand": 120},
                {"item_id": "SKU-002", "location_id": "LOC-01", "predicted_demand": 85},
            ],
        }
        message = f"Demand forecast simulated for the next {action.get('horizon', 'week')}."
        return {"assistantMessage": message, "structuredOutput": result}
    if action_type == "query_expiring":
        items = _assistant_local_query_expiring(int(action.get("days", 3)))
        message = f"{len(items)} items are expiring within {action.get('days', 3)} days."
        return {"assistantMessage": message, "structuredOutput": {"items": items}}
    if action_type == "create_promotion":
        discount_percent = action.get("discountPercent")
        scope = action.get("scope") or "expiring"
        if scope == "all":
            result = {
                "expiringDays": None,
                "items": [],
                "strategy": f"Apply {discount_percent or 10}% discount across all products and monitor margin impact.",
            }
            message = (
                f"Promotion prepared for all products{f' at {discount_percent}%' if discount_percent else ''}."
            )
        else:
            items = _assistant_local_query_expiring(int(action.get("expiringDays", 3)))
            result = {
                "expiringDays": action.get("expiringDays", 3),
                "items": items[:10],
                "strategy": f"Bundle near-expiry items with top sellers and apply {discount_percent or 10}% markdown.",
            }
            message = (
                f"Promotion strategy prepared for {len(result['items'])} near-expiry items"
                f"{f' at {discount_percent}%' if discount_percent else ''}."
            )
        return {"assistantMessage": message, "structuredOutput": result}
    if action_type == "recommend_prep":
        result = {
            "horizon": action.get("horizon", "week"),
            "recommendations": [
                {"item_id": "SKU-001", "suggested_prep": 120, "rationale": "High velocity item"},
                {"item_id": "SKU-002", "suggested_prep": 75, "rationale": "Stable baseline"},
            ],
        }
        message = f"Prep recommendations ready for the next {action.get('horizon', 'week')}."
        return {"assistantMessage": message, "structuredOutput": result}
    return {"assistantMessage": "Action not supported yet.", "structuredOutput": None}

# Main Header
_assistant_init_state()
st.markdown("""
<div class="main-header">
    <h1>Fresh Flow Dashboard</h1>
    <p>Professional inventory management with AI forecasting<br>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h3 style="color: white; margin-bottom: 2rem;">Configuration</h3>', unsafe_allow_html=True)
    
    data_dir = st.text_input("Data Folder", str(DATA_DIR_DEFAULT))
    output_dir = st.text_input("Reports Folder", str(REPORTS_DIR_DEFAULT))

    st.markdown("---")
    st.markdown('<h4 style="color: rgba(255,255,255,0.9);">Assistant Mode</h4>', unsafe_allow_html=True)
    st.session_state.assistant_mode = st.selectbox(
        "Mode",
        ["API", "Local"],
        index=0 if st.session_state.assistant_mode == "API" else 1,
        help="API mode calls the Node backend. Local mode runs a lightweight in-app engine.",
    )
    if st.session_state.assistant_mode == "API":
        st.session_state.assistant_api_base = st.text_input(
            "API Base URL",
            st.session_state.assistant_api_base,
        )
        st.session_state.assistant_api_key = st.text_input(
            "API Key (optional)",
            st.session_state.assistant_api_key,
            type="password",
        )
    else:
        st.caption("Local mode stores assistant data in data/assistant.")

    st.markdown("---")
    st.markdown('<h4 style="color: rgba(255,255,255,0.9);">Auto Refresh</h4>', unsafe_allow_html=True)
    
    auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
    refresh_seconds = st.slider("Refresh Interval (seconds)", 30, 600, 120)
    
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")
    elif auto_refresh:
        st.info("Install streamlit-autorefresh for auto-refresh")
    
    st.markdown("---")
    
    if st.button("Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    state = _load_state()
    if state.get("last_refresh"):
        try:
            last_refresh = datetime.fromisoformat(state.get('last_refresh'))
            st.caption(f"Last update: {last_refresh.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    st.markdown("---")
    st.caption("Version 4.0 - Production Ready")

# Main tabs (no emojis)
tabs = st.tabs(["Dashboard", "Forecasts", "Ingredients", "Upload Data", "Exports", "Assistant"])

with tabs[0]:
    col1, col2 = st.columns([1, 3])
    with col1:
        if HAS_PIPELINE:
            if st.button("Update Insights Now", use_container_width=True):
                with st.spinner("Updating insights..."):
                    try:
                        run_pipeline(data_dir=str(data_dir), output_dir=str(output_dir))
                        state = _load_state()
                        state["last_refresh"] = datetime.now().isoformat(timespec="seconds")
                        _save_state(state)
                        st.success("System updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Pipeline not available")
    
    reports = _load_latest_reports(Path(output_dir))
    summary_df = reports.get("summary", pd.DataFrame())
    forecast_df = reports.get("forecast", pd.DataFrame())
    recommendations_df = reports.get("recommendations", pd.DataFrame())
    prep_df = reports.get("prep_plan", pd.DataFrame())
    
    if summary_df.empty:
        st.markdown("""
        <div class="alert-box alert-info">
            <h4 style="margin: 0 0 0.5rem 0; color: white;">Welcome to Fresh Flow</h4>
            <p style="margin: 0; color: rgba(255,255,255,0.9);">No reports found yet. Click Update Insights Now to generate your first forecast.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header"><h2 class="section-title">Key Performance Indicators</h2></div>', unsafe_allow_html=True)
        
        summary = summary_df.iloc[0].to_dict()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Expected Demand</div>
            </div>
            """.format(f"{summary.get('total_predicted_demand', 0):,.0f}"), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">At-Risk Items</div>
            </div>
            """.format(f"{summary.get('at_risk_items_count', 0):,.0f}"), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">${:,.0f}</div>
                <div class="metric-label">Potential Waste</div>
            </div>
            """.format(summary.get('potential_waste_value', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">${:,.0f}</div>
                <div class="metric-label">Recovery Opportunity</div>
            </div>
            """.format(summary.get('potential_recovery_value', 0)), unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h2 class="section-title">Forecast Overview</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if not forecast_df.empty:
                st.plotly_chart(_create_forecast_chart(forecast_df), use_container_width=True)
        
        with col2:
            if not recommendations_df.empty:
                st.plotly_chart(_create_risk_distribution(recommendations_df), use_container_width=True)
        
        st.markdown('<div class="section-header"><h2 class="section-title">Sales Trends</h2></div>', unsafe_allow_html=True)
        
        daily_sales = _load_daily_sales(Path(data_dir))
        if not daily_sales.empty:
            st.plotly_chart(_create_trend_chart(daily_sales), use_container_width=True)

# Rest of tabs remain identical to original functionality (no emojis, black/green theme)
with tabs[1]:
    st.markdown('<div class="section-header"><h2 class="section-title">Product Forecasts</h2></div>', unsafe_allow_html=True)
    
    if not forecast_df.empty:
        search = st.text_input("Search Products", "")
        
        # Column normalization and display logic (unchanged)
        display_df = forecast_df.copy()
        col_mapping = {
            'item_name': ['item_name', 'title', 'name'],
            'item_id': ['item_id', 'id'],
            'predicted_daily_demand': ['predicted_daily_demand', 'predicted_demand', 'demand'],
        }
        
        for target_col, possible_cols in col_mapping.items():
            for col in possible_cols:
                if col in display_df.columns:
                    if target_col not in display_df.columns:
                        display_df[target_col] = display_df[col]
                    break
        
        if search and 'item_name' in display_df.columns:
            display_df = display_df[display_df['item_name'].astype(str).str.contains(search, case=False, na=False)]
        
        if 'item_name' in display_df.columns and 'predicted_daily_demand' in display_df.columns:
            show_cols = ['item_name', 'predicted_daily_demand']
            if 'item_id' in display_df.columns:
                show_cols.insert(0, 'item_id')
            
            display_table = display_df[show_cols].copy()
            display_table.columns = ['ID', 'Product', 'Expected Daily Demand'][:len(show_cols)]
            
            if 'Expected Daily Demand' in display_table.columns:
                display_table = display_table.sort_values('Expected Daily Demand', ascending=False)
            
            st.dataframe(display_table, use_container_width=True, height=600)
            
            csv = display_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Forecasts",
                data=csv,
                file_name=f"forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No forecast data available yet.")

with tabs[2]:
    st.markdown('<div class="section-header"><h2 class="section-title">Ingredient Requirements</h2></div>', unsafe_allow_html=True)
    
    if not prep_df.empty:
        col_mapping = {
            'ingredient_name': 'Ingredient',
            'ingredient_id': 'ID',
            'quantity_needed': 'Quantity Needed',
            'current_stock': 'Current Stock',
            'net_to_order': 'Order Quantity',
            'unit': 'Unit',
            'estimated_cost': 'Estimated Cost'
        }
        
        display_prep = prep_df.copy()
        for old_col, new_col in col_mapping.items():
            if old_col in display_prep.columns:
                display_prep = display_prep.rename(columns={old_col: new_col})
        
        total_items = len(display_prep)
        items_to_order = (display_prep.get('Order Quantity', pd.Series([0])) > 0).sum()
        total_cost = display_prep.get('Estimated Cost', pd.Series([0])).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Ingredients", f"{total_items}")
        with col2: st.metric("Items to Order", f"{items_to_order}")
        with col3: st.metric("Estimated Cost", f"${total_cost:,.2f}")
        
        st.dataframe(display_prep, use_container_width=True, height=500)
        
        csv = display_prep.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Shopping List",
            data=csv,
            file_name=f"shopping_list_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No ingredient data available.")

    st.subheader("What to Order")
    if not forecast_df.empty:
        schema_issue = _inventory_schema_issue(Path(data_dir))
        inventory_df = _load_inventory_snapshot(Path(data_dir))
        if schema_issue:
            st.info(schema_issue)
        elif not inventory_df.empty:
            order_days = 3
            order_df = forecast_df.merge(inventory_df, on="item_id", how="left")
            if "item_name" not in order_df.columns:
                if "title" in order_df.columns:
                    order_df["item_name"] = order_df["title"]
                else:
                    order_df["item_name"] = order_df["item_id"].astype(str)
            order_df["item_name"] = order_df["item_name"].fillna(order_df["item_id"].astype(str))
            order_df["current_stock"] = order_df.get("quantity_on_hand", 0).fillna(0)
            order_df["required_qty"] = (order_df["predicted_daily_demand"] * order_days).round(0)
            order_df["qty_to_order"] = (order_df["required_qty"] - order_df["current_stock"]).clip(lower=0)
            order_df = order_df[order_df["qty_to_order"] > 0]
            st.dataframe(
                order_df[["item_name", "current_stock", "required_qty", "qty_to_order"]]
                .rename(columns={
                    "item_name": "Product",
                    "current_stock": "Current stock",
                    "required_qty": "Needed",
                    "qty_to_order": "Order quantity",
                })
                .head(50)
            )
        else:
            st.info("Inventory data not available.")
    else:
        st.info("Forecast not available yet.")

    st.subheader("Alerts and Risks")
    if not recommendations_df.empty:
        alert_view = recommendations_df[["item_name", "current_stock", "risk_category", "recommended_action"]]
        alert_view = alert_view.rename(columns={
            "item_name": "Product",
            "current_stock": "Current stock",
            "risk_category": "Risk",
            "recommended_action": "Action",
        })
        styled = _style_risk(alert_view)
        if hasattr(styled, "hide_index"):
            styled = styled.hide_index()
        st.dataframe(styled)
    else:
        st.info("No alerts available.")

    st.subheader("Preparation Plan")
    if not prep_df.empty:
        st.dataframe(prep_df.head(50))
    else:
        st.info("Preparation plan not available.")

    st.subheader("Trends")
    daily_sales = _load_daily_sales(Path(data_dir))
    if not daily_sales.empty:
        trend_df = daily_sales.groupby("date", as_index=False)["quantity_sold"].sum()
        trend_df = trend_df.rename(columns={"date": "Date", "quantity_sold": "Total sales"})
        st.line_chart(trend_df, x="Date", y="Total sales")
    else:
        st.info("Sales history not available.")

with tabs[3]:
    st.markdown('<div class="section-header"><h2 class="section-title">Upload Data</h2></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select File", type=["csv", "xlsx"])
    dataset_choice = st.selectbox("Data Type", DATASET_OPTIONS)
    
    if uploaded_file is not None:
        try:
            incoming_df = _read_upload(uploaded_file)
            st.success(f"Loaded {len(incoming_df):,} rows")
            st.dataframe(incoming_df.head(50), use_container_width=True)
            
            if st.button("Save to Data Folder", type="primary"):
                target_file = Path(data_dir) / uploaded_file.name
                target_file.parent.mkdir(parents=True, exist_ok=True)
                incoming_df.to_csv(target_file, index=False)
                st.success(f"Saved to {target_file}")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[4]:
    st.markdown('<div class="section-header"><h2 class="section-title">Export Reports</h2></div>', unsafe_allow_html=True)
    
    reports = _load_latest_reports(Path(output_dir))
    
    export_options = {
        "Demand Forecast": reports.get("forecast", pd.DataFrame()),
        "Risk Alerts": reports.get("recommendations", pd.DataFrame()),
        "Ingredient List": reports.get("prep_plan", pd.DataFrame()),
    }
    
    for report_name, report_df in export_options.items():
        if not report_df.empty:
            st.markdown(f"### {report_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = report_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{report_name.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                excel_bytes = _to_excel_bytes(report_df)
                if excel_bytes:
                    st.download_button(
                        label="Download Excel",
                        data=excel_bytes,
                        file_name=f"{report_name.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            st.markdown("---")

with tabs[5]:
    st.markdown('<div class="section-header"><h2 class="section-title">Assistant Console</h2></div>', unsafe_allow_html=True)
    st.caption("Send natural language commands to update data, run forecasts, and plan promotions.")

    if st.session_state.assistant_mode == "API":
        ok, status_msg = _assistant_check_health()
        if ok:
            st.success(f"API status: {status_msg}")
        else:
            st.error(f"API status: {status_msg}. Start the backend or switch to Local mode.")

    for message in st.session_state.assistant_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message.get("timestamp", ""))

    pending = st.session_state.assistant_pending_confirmation
    if pending:
        st.info(pending.get("message", "Confirmation required."))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel Action", key="assistant_cancel"):
                try:
                    if pending.get("local"):
                        result = {"status": "cancelled", "assistantMessage": "Action cancelled."}
                    else:
                        result = _assistant_post(
                            "/api/chat/confirm",
                            {"confirmationId": pending.get("confirmation_id"), "approve": False},
                        )
                    _assistant_add_message("assistant", result.get("assistantMessage", "Action cancelled."))
                    st.session_state.assistant_structured_output = result.get("structuredOutput")
                    st.session_state.assistant_last_outcome = {
                        "status": result.get("status", "cancelled"),
                        "message": result.get("assistantMessage", "Action cancelled."),
                        "structuredOutput": result.get("structuredOutput"),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                except Exception as exc:
                    _assistant_add_message("assistant", f"Error: {exc}")
                st.session_state.assistant_pending_confirmation = None
                st.rerun()
        with col2:
            if st.button("Confirm Action", key="assistant_confirm"):
                try:
                    if pending.get("local"):
                        exec_result = _assistant_local_execute(pending.get("action", {}))
                        result = {
                            "status": "executed",
                            "assistantMessage": exec_result.get("assistantMessage"),
                            "structuredOutput": exec_result.get("structuredOutput"),
                        }
                    else:
                        result = _assistant_post(
                            "/api/chat/confirm",
                            {"confirmationId": pending.get("confirmation_id"), "approve": True},
                        )
                    _assistant_add_message("assistant", result.get("assistantMessage", "Action confirmed."))
                    st.session_state.assistant_structured_output = result.get("structuredOutput")
                    st.session_state.assistant_last_outcome = {
                        "status": result.get("status", "executed"),
                        "message": result.get("assistantMessage", "Action executed."),
                        "structuredOutput": result.get("structuredOutput"),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    _assistant_log_event(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": result.get("status", "executed"),
                            "message": result.get("assistantMessage", "Action executed."),
                            "action": pending.get("action"),
                        }
                    )
                except Exception as exc:
                    _assistant_add_message("assistant", f"Error: {exc}")
                st.session_state.assistant_pending_confirmation = None
                st.rerun()

    with st.form("assistant_form", clear_on_submit=True):
        user_text = st.text_area(
            "Message",
            placeholder="e.g., Upload this sales data, Predict demand for next week, Show items expiring in 3 days",
            height=120,
        )
        upload = st.file_uploader("Attach CSV or JSON", type=["csv", "json"])
        submitted = st.form_submit_button("Send")

    if submitted:
        if not user_text.strip() and upload is None:
            st.warning("Please enter a message or attach a file.")
        else:
            attachment = None
            if upload is not None:
                file_bytes = upload.read()
                content = file_bytes.decode("utf-8", errors="ignore")
                extension = upload.name.split(".")[-1].lower()
                attachment = {
                    "name": upload.name,
                    "type": "json" if extension == "json" else "csv",
                    "content": content,
                }

            _assistant_add_message("user", user_text.strip() or "[File uploaded]")

            try:
                if st.session_state.assistant_mode == "Local":
                    intent = _assistant_classify_intent(user_text.strip())
                    parsed = _assistant_parse_data(user_text.strip(), attachment)
                    plan = _assistant_plan_action(
                        user_text.strip(),
                        {"lastDataset": st.session_state.assistant_last_decision.get("action", {}).get("dataset")
                         if st.session_state.assistant_last_decision else None},
                        intent,
                        parsed,
                    )
                    if plan["action"] and not plan["requires_confirmation"]:
                        exec_result = _assistant_local_execute(plan["action"])
                        result = {
                            "conversationId": st.session_state.assistant_conversation_id or "local",
                            "intent": intent,
                            "resolvedIntent": plan["resolved_intent"],
                            "action": plan["action"],
                            "actionPreview": plan["action_preview"],
                            "requiresConfirmation": False,
                            "assistantMessage": exec_result.get("assistantMessage"),
                            "structuredOutput": exec_result.get("structuredOutput"),
                        }
                    else:
                        result = {
                            "conversationId": st.session_state.assistant_conversation_id or "local",
                            "intent": intent,
                            "resolvedIntent": plan["resolved_intent"],
                            "action": plan["action"],
                            "actionPreview": plan["action_preview"],
                            "requiresConfirmation": plan["requires_confirmation"],
                            "assistantMessage": plan["assistant_message"],
                            "confirmationId": f"local-{datetime.now().strftime('%H%M%S')}",
                            "structuredOutput": {"action": plan["action"]} if plan["action"] else None,
                        }
                else:
                    payload = {"message": user_text.strip()}
                    if st.session_state.assistant_conversation_id:
                        payload["conversationId"] = st.session_state.assistant_conversation_id
                    if attachment:
                        payload["attachment"] = attachment
                    result = _assistant_post("/api/chat/interpret", payload)
                st.session_state.assistant_conversation_id = result.get("conversationId")
                st.session_state.assistant_action_preview = result.get("actionPreview")
                st.session_state.assistant_structured_output = result.get("structuredOutput")
                st.session_state.assistant_last_decision = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "intent": result.get("resolvedIntent") or result.get("intent", {}).get("intent"),
                    "confidence": result.get("intent", {}).get("confidence"),
                    "entities": result.get("intent", {}).get("entities"),
                    "requiresConfirmation": result.get("requiresConfirmation"),
                    "action": result.get("action"),
                    "actionPreview": result.get("actionPreview"),
                }
                if result.get("requiresConfirmation"):
                    st.session_state.assistant_pending_confirmation = {
                        "confirmation_id": result.get("confirmationId"),
                        "message": result.get("assistantMessage"),
                        "action": result.get("action"),
                        "local": st.session_state.assistant_mode == "Local",
                    }
                    st.session_state.assistant_last_outcome = {
                        "status": "pending_confirmation",
                        "message": result.get("assistantMessage"),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                else:
                    st.session_state.assistant_pending_confirmation = None
                    st.session_state.assistant_last_outcome = {
                        "status": "executed",
                        "message": result.get("assistantMessage"),
                        "structuredOutput": result.get("structuredOutput"),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    _assistant_log_event(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "executed",
                            "message": result.get("assistantMessage", "Action executed."),
                            "action": result.get("action"),
                        }
                    )
                _assistant_add_message("assistant", result.get("assistantMessage", "Done."))
            except Exception as exc:
                _assistant_add_message("assistant", f"Error: {exc}")

            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Action Preview**")
        if st.session_state.assistant_action_preview:
            st.json(st.session_state.assistant_action_preview)
        else:
            st.caption("No action planned yet.")
    with col2:
        st.markdown("**Structured Output**")
        if st.session_state.assistant_structured_output:
            st.json(st.session_state.assistant_structured_output)
        else:
            st.caption("No output yet.")

    st.markdown("**Decision Trace**")
    last_decision = st.session_state.assistant_last_decision
    if last_decision:
        intent_value = last_decision.get("intent") or "unknown"
        confidence_value = last_decision.get("confidence")
        confidence_text = f"{confidence_value:.2f}" if isinstance(confidence_value, (int, float)) else "n/a"
        st.write(f"Intent: `{intent_value}` (confidence {confidence_text}).")
        st.write(f"Requires confirmation: {last_decision.get('requiresConfirmation')}.")
        entities = last_decision.get("entities")
        if entities:
            st.json(entities)
        if last_decision.get("action"):
            st.write("Planned action:")
            st.json(last_decision.get("action"))
    else:
        st.caption("No decision captured yet.")

    st.markdown("**Outcome**")
    last_outcome = st.session_state.assistant_last_outcome
    if last_outcome:
        st.write(f"Status: `{last_outcome.get('status')}`")
        st.write(last_outcome.get("message", ""))
        if last_outcome.get("structuredOutput"):
            st.json(last_outcome.get("structuredOutput"))
    else:
        st.caption("No outcome yet.")

    if st.session_state.assistant_action_log:
        st.markdown("**Recent Actions**")
        log_df = pd.DataFrame(st.session_state.assistant_action_log)
        st.dataframe(log_df, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.95); margin-bottom: 1rem;">
        Fresh Flow Dashboard 
    </div>
    <div style="color: rgba(255,255,255,0.8); font-size: 0.95rem;">
        Professional inventory management with AI forecasting
    </div>
    <div style="margin-top: 1.5rem; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
        Production Ready System
    </div>
</div>
""", unsafe_allow_html=True)
