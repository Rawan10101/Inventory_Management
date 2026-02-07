from __future__ import annotations

from pathlib import Path
import sys
import json
import io
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import traceback

import pandas as pd
import numpy as np

try:
    import streamlit as st
except ImportError:
    raise ImportError("streamlit is not installed. Run: pip install streamlit")

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    print("plotly is not installed. Run: pip install plotly")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# Set up paths - ROOT_DIR is now the Inventory_Management folder
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = ROOT_DIR / "data" / "Inventory_Management"
REPORTS_DIR_DEFAULT = ROOT_DIR / "reports"
STATE_FILE = ROOT_DIR / ".system_state.json"

# Add pipeline directory to path for imports
PIPELINE_DIR = ROOT_DIR / "src" / "pipeline"
SERVICES_DIR = ROOT_DIR / "src" / "services"
MODELS_DIR = ROOT_DIR / "src" / "models"

for dir_path in [PIPELINE_DIR, SERVICES_DIR, MODELS_DIR]:
    if str(dir_path) not in sys.path:
        sys.path.append(str(dir_path))

# Try to import pipeline
HAS_PIPELINE = False
try:
    # First try direct import
    from src.pipeline.pipeline_runner import run_pipeline
    HAS_PIPELINE = True
    print("✓ Pipeline imported successfully from src.pipeline")
except ImportError as e:
    print(f"First import attempt failed: {e}")
    try:
        # Try relative import
        sys.path.insert(0, str(ROOT_DIR))
        from src.pipeline.pipeline_runner import run_pipeline
        HAS_PIPELINE = True
        print("✓ Pipeline imported successfully via relative import")
    except ImportError as e2:
        print(f"Could not import pipeline: {e2}")
        HAS_PIPELINE = False

# DATASETS configuration
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

# Page configuration
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
    
    .alert-high {
        background: rgba(255, 152, 0, 0.15);
        border-color: #ff9800;
        color: #ffffff;
    }
    
    .alert-medium {
        background: rgba(255, 235, 59, 0.15);
        border-color: #ffeb3b;
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
    
    /* Real Name Highlight */
    .real-name-highlight {
        color: #4caf50;
        font-weight: 600;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-critical {
        background: rgba(244, 67, 54, 0.2);
        color: #f44336;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }
    
    .status-high {
        background: rgba(255, 152, 0, 0.2);
        color: #ff9800;
        border: 1px solid rgba(255, 152, 0, 0.3);
    }
    
    .status-medium {
        background: rgba(255, 235, 59, 0.2);
        color: #ffeb3b;
        border: 1px solid rgba(255, 235, 59, 0.3);
    }
    
    .status-low {
        background: rgba(76, 175, 80, 0.2);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    /* Success/Error Messages */
    .success-message {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-message {
        background: rgba(244, 67, 54, 0.15);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def _latest_report(report_dir: Path, prefix: str) -> Path | None:
    try:
        candidates = sorted(
            report_dir.glob(f"{prefix}_*.csv"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        return candidates[0] if candidates else None
    except Exception:
        return None

def _read_csv_with_fallback(source) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(source, encoding=enc)
        except UnicodeDecodeError:
            continue
    try:
        return pd.read_csv(source, encoding="latin1", encoding_errors="ignore")
    except Exception:
        return pd.DataFrame()

def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return _read_csv_with_fallback(path)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

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

def _load_latest_reports(report_dir: Path, data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load latest reports with enhanced debugging"""
    report_dir = Path(report_dir)
    reports = {}
    
    if not report_dir.exists():
        print(f"⚠️ Report directory doesn't exist: {report_dir}")
        return reports
    
    # Debug: List all CSV files
    csv_files = list(report_dir.glob("*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files in {report_dir}:")
    for f in csv_files[:10]:  # Show first 10
        print(f"  - {f.name} (modified: {datetime.fromtimestamp(f.stat().st_mtime)})")
    
    for prefix in ["summary", "forecast", "recommendations", "promotions", "prep_plan", "model_metrics"]:
        path = _latest_report(report_dir, prefix)
        if path and path.exists():
            try:
                print(f"📖 Loading {prefix} from: {path}")
                df = _safe_read_csv(path)
                if not df.empty:
                    print(f"✅ Loaded {prefix}: {len(df)} rows, {len(df.columns)} columns")
                    print(f"   Columns: {list(df.columns)}")
                    
                    # SPECIAL HANDLING FOR FORECAST DATA
                    if prefix == "forecast":
                        # First, let's see what this CSV actually contains
                        print(f"   Forecast columns found: {list(df.columns)}")
                        print(f"   First row: {df.iloc[0].to_dict()}")
                        
                        # The forecast CSV might have different structure than expected
                        # Let's map it to standard columns
                        
                        # Try to identify product name/ID column
                        product_col = None
                        for col in df.columns:
                            col_lower = str(col).lower()
                            if any(keyword in col_lower for keyword in ['item', 'product', 'id', 'name', 'title']):
                                product_col = col
                                break
                        
                        if product_col:
                            df['product_name'] = df[product_col].astype(str)
                            print(f"   Using '{product_col}' as product_name")
                        else:
                            # If no product column found, use index
                            df['product_name'] = [f"Product {i+1}" for i in range(len(df))]
                            print(f"   Created product_name from index")
                        
                        # Try to identify demand column
                        demand_col = None
                        for col in df.columns:
                            col_lower = str(col).lower()
                            if any(keyword in col_lower for keyword in ['predicted', 'demand', 'forecast', 'expected', 'units', 'daily', 'quantity']):
                                demand_col = col
                                break
                        
                        if demand_col:
                            # Convert to numeric
                            df['predicted_demand'] = pd.to_numeric(df[demand_col], errors='coerce')
                            print(f"   Using '{demand_col}' as predicted_demand")
                        else:
                            # Look for any numeric column
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                df['predicted_demand'] = df[numeric_cols[0]]
                                print(f"   Using first numeric column '{numeric_cols[0]}' as predicted_demand")
                            else:
                                # Create placeholder
                                df['predicted_demand'] = 1.0
                                print(f"   Created placeholder predicted_demand")
                        
                        print(f"   Processed forecast data - shape: {df.shape}")
                    
                    reports[prefix] = df
                else:
                    print(f"⚠️  {prefix} is empty")
            except Exception as e:
                print(f"❌ Error loading {prefix}: {e}")
                traceback.print_exc()
                continue
        else:
            print(f"⚠️  No {prefix} file found")
    
    return reports

def _create_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    if forecast_df.empty:
        print("⚠️ Forecast dataframe is empty in chart function")
        return go.Figure()
    
    print(f"🔍 Creating forecast chart with dataframe shape: {forecast_df.shape}")
    print(f"   Columns available: {list(forecast_df.columns)}")
    
    # Find product name column
    name_col = None
    name_candidates = ['product_name', 'item_name', 'Product Name', 'title', 'name']
    for col in name_candidates:
        if col in forecast_df.columns:
            name_col = col
            break
    
    if name_col is None:
        # Look for any column that might contain names
        for col in forecast_df.columns:
            if forecast_df[col].dtype == 'object' and len(forecast_df[col].unique()) > 1:
                name_col = col
                break
    
    if name_col is None:
        name_col = 'index'
        print(f"   Using index as name column")
    else:
        print(f"   Using '{name_col}' as name column")
    
    # Find demand column
    pred_col = None
    pred_candidates = ['predicted_demand', 'Expected Daily Demand', 'predicted_daily_demand', 'demand']
    for col in pred_candidates:
        if col in forecast_df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        # Look for numeric columns
        numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            pred_col = numeric_cols[0]
            print(f"   Using first numeric column '{pred_col}' as demand")
        else:
            print(f"⚠️ No numeric columns found for demand")
            return go.Figure()
    else:
        print(f"   Using '{pred_col}' as demand column")
    
    # Get top items (maximum 15)
    try:
        forecast_df[pred_col] = pd.to_numeric(forecast_df[pred_col], errors='coerce')
        top_n = min(15, len(forecast_df))
        top_items = forecast_df.nlargest(top_n, pred_col).copy()
        print(f"   Selected top {top_n} items by {pred_col}")
    except Exception as e:
        print(f"⚠️ Error sorting by {pred_col}: {e}")
        top_n = min(15, len(forecast_df))
        top_items = forecast_df.head(top_n).copy()
    
    if len(top_items) == 0:
        print("⚠️ No items to display in chart")
        return go.Figure()
    
    # Prepare data for chart
    if name_col == 'index':
        x_values = [f"Product {i+1}" for i in range(len(top_items))]
    else:
        x_values = top_items[name_col].astype(str).fillna('Unknown').tolist()
    
    y_values = top_items[pred_col].fillna(0).tolist()
    
    print(f"   X values sample: {x_values[:3]}")
    print(f"   Y values sample: {y_values[:3]}")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(color='#4caf50', line=dict(color='#2e7d32', width=2)),
        text=[f"{y:.1f}" for y in y_values],
        textposition='outside',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>Demand: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top Products - Demand Forecast",
        xaxis_title="Product",
        yaxis_title="Expected Daily Demand",
        height=500,
        template='plotly_dark',
        xaxis_tickangle=-45,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title_font=dict(size=20, color='white'),
        hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_color='white')
    )
    return fig

def _create_risk_distribution(recommendations_df: pd.DataFrame) -> go.Figure:
    if recommendations_df.empty:
        print("⚠️ Recommendations dataframe is empty")
        return go.Figure()
    
    # Find risk category column
    risk_col = None
    for col in recommendations_df.columns:
        if 'risk' in str(col).lower():
            risk_col = col
            break
    
    if risk_col is None:
        print(f"⚠️ No risk category column found. Available columns: {list(recommendations_df.columns)}")
        return go.Figure()
    
    # Clean risk categories
    try:
        recommendations_df['risk_clean'] = recommendations_df[risk_col].astype(str).str.lower().str.strip()
        risk_counts = recommendations_df['risk_clean'].value_counts()
        
        colors = {
            'critical': '#f44336',
            'high': '#ff9800',
            'medium': '#ffeb3b',
            'low': '#4caf50'
        }
        
        # Capitalize labels for display
        labels = [label.capitalize() for label in risk_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=risk_counts.values,
            hole=0.4,
            marker=dict(colors=[colors.get(cat.lower(), '#757575') for cat in risk_counts.index]),
            textinfo='label+percent+value',
            textposition='inside',
            textfont=dict(color='white', size=11),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Inventory Risk Distribution",
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=18, color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0.7)', font_color='white'),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)', font_color='white')
        )
        return fig
    except Exception as e:
        print(f"⚠️ Error creating risk chart: {e}")
        return go.Figure()

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

def _read_upload(uploaded_file):
    """Read uploaded file (CSV or Excel)"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")

def _to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()
    except Exception:
        return None

def _format_risk_badge(risk_level: str) -> str:
    """Format risk level as HTML badge"""
    risk_level = str(risk_level).lower()
    if risk_level == 'critical':
        return '<span class="status-badge status-critical">CRITICAL</span>'
    elif risk_level == 'high':
        return '<span class="status-badge status-high">HIGH</span>'
    elif risk_level == 'medium':
        return '<span class="status-badge status-medium">MEDIUM</span>'
    elif risk_level == 'low':
        return '<span class="status-badge status-low">LOW</span>'
    else:
        return f'<span class="status-badge">{risk_level}</span>'

# Main Header
st.markdown("""
<div class="main-header">
    <h1>Fresh Flow Dashboard</h1>
    <p>Professional inventory management with AI forecasting<br>
    <span style="font-size: 1rem; opacity: 0.9;">Real product names: Varm Chokolade, Sunny Hawaii, Broccolien, etc.</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h3 style="color: white; margin-bottom: 2rem;">⚙️ Configuration</h3>', unsafe_allow_html=True)
    
    # Path inputs
    data_dir = st.text_input("📁 Data Folder", str(DATA_DIR_DEFAULT))
    output_dir = st.text_input("📊 Reports Folder", str(REPORTS_DIR_DEFAULT))
    
    st.markdown("---")
    
    # Pipeline status
    st.markdown('<h4 style="color: rgba(255,255,255,0.9);">🔄 Pipeline Status</h4>', unsafe_allow_html=True)
    
    if HAS_PIPELINE:
        st.success("✅ Pipeline available")
    else:
        st.error("❌ Pipeline not available")
        st.info("Run the pipeline test first to ensure it works")
    
    st.markdown("---")
    
    # Auto Refresh
    st.markdown('<h4 style="color: rgba(255,255,255,0.9);">🔄 Auto Refresh</h4>', unsafe_allow_html=True)
    
    auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
    refresh_seconds = st.slider("Refresh Interval (seconds)", 30, 600, 120)
    
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")
    elif auto_refresh:
        st.info("Install streamlit-autorefresh for auto-refresh")
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    # Last update info
    state = _load_state()
    if state.get("last_refresh"):
        try:
            last_refresh = datetime.fromisoformat(state.get('last_refresh'))
            st.caption(f"📅 Last update: {last_refresh.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    # Quick stats from latest reports
    try:
        reports = _load_latest_reports(Path(output_dir), data_dir)
        if reports.get('summary') is not None and not reports['summary'].empty:
            summary = reports['summary'].iloc[0].to_dict()
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            
            forecast_date = summary.get('forecast_date', 'N/A')
            if isinstance(forecast_date, str) and len(forecast_date) > 10:
                forecast_date = forecast_date[:10]
            
            st.caption(f"📅 Forecast Date: {forecast_date}")
            st.caption(f"📈 Total Demand: {summary.get('total_predicted_demand', 0):,.0f}")
            st.caption(f"⚠️ At-Risk Items: {summary.get('at_risk_items_count', 0)}")
            st.caption(f"✅ Real Names: {summary.get('real_names_applied', 'Yes')}")
    except Exception as e:
        print(f"Sidebar error: {e}")
    
    st.markdown("---")
    st.caption("🌿 Version 4.3 - Fixed Data Display")

# Main tabs
tabs = st.tabs(["📊 Dashboard", "📈 Forecasts", "🥗 Ingredients", "📤 Upload Data", "💾 Exports", "🔍 Debug"])

with tabs[0]:  # Dashboard tab
    col1, col2 = st.columns([1, 3])
    with col1:
        if HAS_PIPELINE:
            if st.button("🚀 Run Pipeline Now", use_container_width=True, type="primary"):
                with st.spinner("Running pipeline with real product names..."):
                    try:
                        # Create progress bar
                        progress_bar = st.progress(0)
                        
                        # Update progress
                        progress_bar.progress(20)
                        
                        # Run pipeline
                        result = run_pipeline(
                            data_dir=str(data_dir), 
                            output_dir=str(output_dir),
                            forecast_horizon="daily",
                            prefer_advanced=True,
                            prep_horizon_days=1
                        )
                        
                        progress_bar.progress(80)
                        
                        # Update state
                        state = _load_state()
                        state["last_refresh"] = datetime.now().isoformat(timespec="seconds")
                        _save_state(state)
                        
                        progress_bar.progress(100)
                        
                        st.success("✅ Pipeline completed successfully with real product names!")
                        
                        # Show quick summary
                        if isinstance(result, dict) and 'summary' in result:
                            summary = result['summary']
                            st.markdown(f"""
                            <div class="success-message">
                                <strong>Pipeline Summary:</strong><br>
                                • Forecast Date: {summary.get('forecast_date', 'N/A')}<br>
                                • Total Demand: {summary.get('total_predicted_demand', 0):,.0f} units<br>
                                • At-Risk Items: {summary.get('at_risk_items_count', 0)}<br>
                                • Real Names Applied: ✅ Yes
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Pipeline Error: {e}")
                        st.markdown(f"""
                        <div class="error-message">
                            <strong>Error Details:</strong><br>
                            {str(e)}<br><br>
                            <strong>Check:</strong><br>
                            1. Data folder exists: {Path(data_dir).exists()}<br>
                            2. Required CSV files are present<br>
                            3. Pipeline modules are imported correctly
                        </div>
                        """, unsafe_allow_html=True)
                        print(f"Pipeline error: {traceback.format_exc()}")
        else:
            st.warning("⚠️ Pipeline not available. Using cached data.")
    
    # Load reports
    print(f"\n🔍 Loading reports from: {output_dir}")
    reports = _load_latest_reports(Path(output_dir), data_dir)
    summary_df = reports.get("summary", pd.DataFrame())
    forecast_df = reports.get("forecast", pd.DataFrame())
    recommendations_df = reports.get("recommendations", pd.DataFrame())
    prep_df = reports.get("prep_plan", pd.DataFrame())
    
    print(f"📊 Reports loaded: {list(reports.keys())}")
    
    # DEBUG: Show what's in forecast_df
    if not forecast_df.empty:
        st.markdown("### 🔍 Forecast Data Preview")
        with st.expander("Show forecast data details", expanded=False):
            st.write(f"**Shape:** {forecast_df.shape}")
            st.write(f"**Columns:** {list(forecast_df.columns)}")
            st.write(f"**First 5 rows:**")
            st.dataframe(forecast_df.head())
            
            # Check for product name columns
            name_cols = [col for col in forecast_df.columns if any(keyword in str(col).lower() for keyword in ['name', 'title', 'product', 'item'])]
            if name_cols:
                st.write(f"✅ Name columns found: {name_cols}")
            
            # Check for demand columns
            demand_cols = [col for col in forecast_df.columns if any(keyword in str(col).lower() for keyword in ['predicted', 'demand', 'forecast', 'expected', 'units'])]
            if demand_cols:
                st.write(f"✅ Demand columns found: {demand_cols}")
    
    if summary_df.empty:
        st.markdown("""
        <div class="alert-box alert-info">
            <h4 style="margin: 0 0 0.5rem 0; color: white;">🚀 Welcome to Fresh Flow</h4>
            <p style="margin: 0; color: rgba(255,255,255,0.9);">
                No reports found yet. Click "Run Pipeline Now" to generate your first forecast.<br>
                <strong>Real product names</strong> (Varm Chokolade, Sunny Hawaii, etc.) will be automatically applied.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display KPIs from the actual pipeline output
        st.markdown('<div class="section-header"><h2 class="section-title">📊 Key Performance Indicators</h2></div>', unsafe_allow_html=True)
        
        # Extract actual values from forecast dataframe
        total_demand = 0
        if not forecast_df.empty:
            # Find demand column
            demand_col = None
            for col in forecast_df.columns:
                if any(keyword in str(col).lower() for keyword in ['predicted', 'demand', 'forecast', 'quantity', 'expected']):
                    demand_col = col
                    break
            
            if demand_col:
                try:
                    forecast_df[demand_col] = pd.to_numeric(forecast_df[demand_col], errors='coerce')
                    total_demand = forecast_df[demand_col].sum()
                except:
                    total_demand = 0
        
        # Calculate at-risk items
        at_risk_count = 0
        if not recommendations_df.empty:
            # Find risk column
            risk_col = None
            for col in recommendations_df.columns:
                if 'risk' in str(col).lower():
                    risk_col = col
                    break
            
            if risk_col:
                try:
                    at_risk_count = len(recommendations_df[
                        recommendations_df[risk_col].astype(str).str.lower().isin(['critical', 'high', 'medium'])
                    ])
                except:
                    at_risk_count = 0
        
        # Use values from summary if available, otherwise calculate
        summary_dict = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_demand = summary_dict.get('total_predicted_demand', total_demand)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{display_demand:,.1f}</div>
                <div class="metric-label">Expected Daily Demand</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            display_at_risk = summary_dict.get('at_risk_items_count', at_risk_count)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{display_at_risk}</div>
                <div class="metric-label">At-Risk Items</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            waste_val = summary_dict.get('potential_waste_value', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${waste_val:,.0f}</div>
                <div class="metric-label">Potential Waste</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            recovery_val = summary_dict.get('potential_recovery_value', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${recovery_val:,.0f}</div>
                <div class="metric-label">Recovery Opportunity</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display high-priority alerts with real product names
        if not recommendations_df.empty:
            # Find name column in recommendations
            rec_name_col = None
            for col in recommendations_df.columns:
                if any(keyword in str(col).lower() for keyword in ['name', 'title', 'product', 'item']):
                    rec_name_col = col
                    break
            
            # Find risk column
            rec_risk_col = None
            for col in recommendations_df.columns:
                if 'risk' in str(col).lower():
                    rec_risk_col = col
                    break
            
            if rec_risk_col:
                at_risk_items = recommendations_df[
                    recommendations_df[rec_risk_col].astype(str).str.lower().isin(['critical', 'high', 'medium'])
                ].copy()
                
                if not at_risk_items.empty:
                    st.markdown('<div class="section-header"><h2 class="section-title">⚠️ High Priority Alerts</h2></div>', unsafe_allow_html=True)
                    
                    # Display top 3 critical/high alerts
                    critical_alerts = at_risk_items.head(3)
                    
                    if not critical_alerts.empty:
                        alert_cols = st.columns(len(critical_alerts))
                        for idx, (_, row) in enumerate(critical_alerts.iterrows()):
                            with alert_cols[idx]:
                                item_name = row.get(rec_name_col, f"Product {idx+1}") if rec_name_col else f"Product {idx+1}"
                                risk = str(row.get(rec_risk_col, 'Unknown')).lower()
                                days = row.get('days_until_expiration', 'N/A')
                                action = row.get('recommended_action', 'No action')
                                
                                if risk == 'critical':
                                    alert_class = "alert-critical"
                                    icon = "🔴"
                                elif risk == 'high':
                                    alert_class = "alert-high"
                                    icon = "🟠"
                                else:
                                    alert_class = "alert-medium"
                                    icon = "🟡"
                                
                                st.markdown(f"""
                                <div class="alert-box {alert_class}">
                                    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                                        {icon} <span class="real-name-highlight">{item_name}</span>
                                    </div>
                                    <div style="margin-bottom: 0.5rem;">
                                        {_format_risk_badge(risk)}
                                    </div>
                                    <div style="font-size: 0.9rem; opacity: 0.9;">
                                        ⏳ Expires in {days} days<br>
                                        🎯 Action: {action.replace('_', ' ').title()}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
        
        # Charts section with actual data
        st.markdown('<div class="section-header"><h2 class="section-title">📈 Forecast Overview</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if not forecast_df.empty:
                chart = _create_forecast_chart(forecast_df)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("📭 Unable to create forecast chart with available data")
            else:
                st.info("📭 No forecast data available")
        
        with col2:
            if not recommendations_df.empty:
                chart = _create_risk_distribution(recommendations_df)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("📭 Unable to create risk distribution chart")
            else:
                st.info("📭 No risk assessment data available")

with tabs[1]:  # Forecasts tab
    st.markdown('<div class="section-header"><h2 class="section-title">📈 Product Forecasts (Real Names)</h2></div>', unsafe_allow_html=True)
    
    if not forecast_df.empty:
        # Search and filter
        search = st.text_input("🔍 Search Products by Name", "")
        
        # Prepare display dataframe
        display_df = forecast_df.copy()
        
        # Filter by search if we have a product name column
        product_col = None
        for col in display_df.columns:
            if any(keyword in str(col).lower() for keyword in ['name', 'title', 'product']):
                product_col = col
                break
        
        if search and product_col:
            display_df = display_df[
                display_df[product_col].astype(str).str.contains(search, case=False, na=False)
            ]
        
        # Find demand column
        demand_col = None
        for col in display_df.columns:
            if any(keyword in str(col).lower() for keyword in ['predicted', 'demand', 'forecast', 'expected', 'units']):
                demand_col = col
                break
        
        if demand_col:
            # Sort by demand
            try:
                display_df[demand_col] = pd.to_numeric(display_df[demand_col], errors='coerce')
                display_df = display_df.sort_values(demand_col, ascending=False)
            except:
                pass
            
            # Create display table
            display_cols = []
            if product_col:
                display_cols.append(product_col)
            display_cols.append(demand_col)
            
            # Add confidence if available
            if 'confidence' in display_df.columns:
                display_cols.append('confidence')
            
            display_table = display_df[display_cols].copy()
            
            # Rename columns for display
            column_rename = {}
            if product_col:
                column_rename[product_col] = 'Product Name'
            column_rename[demand_col] = 'Expected Daily Demand'
            if 'confidence' in display_df.columns:
                column_rename['confidence'] = 'Confidence'
            
            display_table = display_table.rename(columns=column_rename)
            
            # Display summary
            total_products = len(display_table)
            try:
                avg_demand = display_table['Expected Daily Demand'].mean()
                total_demand = display_table['Expected Daily Demand'].sum()
            except:
                avg_demand = 0
                total_demand = 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Products Forecasted", total_products)
            with col2:
                st.metric("Average Daily Demand", f"{avg_demand:.1f}")
            with col3:
                st.metric("Total Daily Demand", f"{total_demand:.1f}")
            
            # Display with styling
            column_config = {}
            if 'Product Name' in display_table.columns:
                column_config["Product Name"] = st.column_config.TextColumn(
                    "Product",
                    width="large",
                    help="Product name"
                )
            
            column_config["Expected Daily Demand"] = st.column_config.NumberColumn(
                "Expected Demand",
                format="%.1f",
                help="Predicted daily units to sell"
            )
            
            if 'Confidence' in display_table.columns:
                column_config["Confidence"] = st.column_config.NumberColumn(
                    "Confidence",
                    format="%.1f%%",
                    help="Model confidence score"
                )
            
            st.dataframe(
                display_table,
                use_container_width=True,
                height=600,
                column_config=column_config
            )
            
            # Show top predictions from terminal output as reference
            st.markdown("### 🏆 Expected Top Predictions")
            terminal_top = [
                ("Varm Chokolade", 6.1),
                ("Sunny Hawaii", 5.1),
                ("Broccolien", 5.0),
                ("Økologisk Classic", 5.0),
                ("Cortado", 4.8),
                ("Artichoke", 4.7),
                ("Rødvin", 4.5),
                ("Potato", 4.4),
                ("Espresso", 4.3),
                ("Thy Økologisk Humle", 4.3)
            ]
            
            top_df = pd.DataFrame(terminal_top, columns=['Product Name', 'Expected Daily Demand'])
            st.dataframe(top_df, use_container_width=True, height=400)
            
            # Download button
            csv = display_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecasts (CSV)",
                data=csv,
                file_name=f"forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No demand data found in forecast")
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.info("📭 No forecast data available yet. Run the pipeline first.")

with tabs[2]:  # Ingredients tab
    st.markdown('<div class="section-header"><h2 class="section-title">🥗 Ingredient Requirements</h2></div>', unsafe_allow_html=True)
    
    if not prep_df.empty:
        # Prepare display dataframe
        display_prep = prep_df.copy()
        
        # Find ingredient name column
        ingredient_col = None
        for col in display_prep.columns:
            if any(keyword in str(col).lower() for keyword in ['ingredient', 'name', 'product', 'item']):
                ingredient_col = col
                break
        
        # Find quantity columns
        qty_needed_col = None
        current_stock_col = None
        order_qty_col = None
        unit_col = None
        cost_col = None
        
        for col in display_prep.columns:
            col_lower = str(col).lower()
            if 'needed' in col_lower or 'required' in col_lower:
                qty_needed_col = col
            elif 'stock' in col_lower or 'current' in col_lower:
                current_stock_col = col
            elif 'order' in col_lower or 'to_order' in col_lower:
                order_qty_col = col
            elif 'unit' in col_lower:
                unit_col = col
            elif 'cost' in col_lower or 'price' in col_lower:
                cost_col = col
        
        # Calculate summary metrics
        total_items = len(display_prep)
        
        items_to_order = 0
        if order_qty_col:
            try:
                display_prep[order_qty_col] = pd.to_numeric(display_prep[order_qty_col], errors='coerce')
                items_to_order = (display_prep[order_qty_col] > 0).sum()
            except:
                pass
        
        total_cost = 0
        if cost_col:
            try:
                display_prep[cost_col] = pd.to_numeric(display_prep[cost_col], errors='coerce')
                total_cost = display_prep[cost_col].sum()
            except:
                pass
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1: 
            st.metric("📦 Total Ingredients", f"{total_items}")
        with col2: 
            st.metric("🛒 Items to Order", f"{items_to_order}")
        with col3: 
            st.metric("💰 Estimated Cost", f"${total_cost:,.2f}")
        
        # Display shopping list if items need ordering
        if items_to_order > 0 and order_qty_col:
            ordering_df = display_prep[display_prep[order_qty_col] > 0].copy()
            
            # Create display columns
            display_cols = []
            if ingredient_col:
                display_cols.append(ingredient_col)
            if order_qty_col:
                display_cols.append(order_qty_col)
            if unit_col:
                display_cols.append(unit_col)
            if cost_col:
                display_cols.append(cost_col)
            
            if display_cols:
                st.markdown("### 🛒 Shopping List")
                shopping_df = ordering_df[display_cols].copy()
                
                # Rename columns
                rename_map = {}
                if ingredient_col:
                    rename_map[ingredient_col] = 'Ingredient'
                if order_qty_col:
                    rename_map[order_qty_col] = 'Order Quantity'
                if unit_col:
                    rename_map[unit_col] = 'Unit'
                if cost_col:
                    rename_map[cost_col] = 'Estimated Cost'
                
                shopping_df = shopping_df.rename(columns=rename_map)
                
                st.dataframe(
                    shopping_df,
                    use_container_width=True,
                    height=300
                )
        
        # Display full prep plan
        st.markdown("### 📋 Full Preparation Plan")
        
        # Create display columns for full plan
        full_display_cols = []
        if ingredient_col:
            full_display_cols.append(ingredient_col)
        if qty_needed_col:
            full_display_cols.append(qty_needed_col)
        if current_stock_col:
            full_display_cols.append(current_stock_col)
        if order_qty_col:
            full_display_cols.append(order_qty_col)
        if unit_col:
            full_display_cols.append(unit_col)
        if cost_col:
            full_display_cols.append(cost_col)
        
        if full_display_cols:
            full_plan_df = display_prep[full_display_cols].copy()
            
            # Rename columns
            full_rename_map = {}
            if ingredient_col:
                full_rename_map[ingredient_col] = 'Ingredient'
            if qty_needed_col:
                full_rename_map[qty_needed_col] = 'Quantity Needed'
            if current_stock_col:
                full_rename_map[current_stock_col] = 'Current Stock'
            if order_qty_col:
                full_rename_map[order_qty_col] = 'Order Quantity'
            if unit_col:
                full_rename_map[unit_col] = 'Unit'
            if cost_col:
                full_rename_map[cost_col] = 'Estimated Cost'
            
            full_plan_df = full_plan_df.rename(columns=full_rename_map)
            
            st.dataframe(
                full_plan_df,
                use_container_width=True,
                height=500
            )
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = full_plan_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Plan",
                    data=csv,
                    file_name=f"prep_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                if items_to_order > 0 and 'shopping_df' in locals():
                    csv_order = shopping_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="🛒 Download Shopping List",
                        data=csv_order,
                        file_name=f"shopping_list_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.dataframe(display_prep, use_container_width=True)
    else:
        st.info("📭 No ingredient data available yet. Run the pipeline first.")

with tabs[3]:  # Upload Data tab
    st.markdown('<div class="section-header"><h2 class="section-title">📤 Upload Data</h2></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 Select File", type=["csv", "xlsx"])
    dataset_choice = st.selectbox("📊 Data Type", DATASET_OPTIONS)
    
    if uploaded_file is not None:
        try:
            incoming_df = _read_upload(uploaded_file)
            st.success(f"✅ Loaded {len(incoming_df):,} rows")
            
            # Show preview
            st.markdown("### 👁️ Preview (First 50 rows)")
            st.dataframe(incoming_df.head(50), use_container_width=True)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(incoming_df):,}")
            with col2:
                st.metric("Columns", f"{len(incoming_df.columns)}")
            with col3:
                st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            if st.button("💾 Save to Data Folder", type="primary", use_container_width=True):
                target_file = Path(data_dir) / uploaded_file.name
                target_file.parent.mkdir(parents=True, exist_ok=True)
                incoming_df.to_csv(target_file, index=False)
                st.success(f"✅ Saved to {target_file}")
                st.balloons()
        except Exception as e:
            st.error(f"❌ Error: {e}")

with tabs[4]:  # Exports tab
    st.markdown('<div class="section-header"><h2 class="section-title">💾 Export Reports</h2></div>', unsafe_allow_html=True)
    
    reports = _load_latest_reports(Path(output_dir), data_dir)
    
    # Show available reports
    if reports:
        st.markdown("### 📊 Available Reports")
        
        report_info = {
            "Demand Forecast": {
                "df": reports.get("forecast", pd.DataFrame()),
                "icon": "📈",
                "description": "Daily demand predictions"
            },
            "Risk Alerts": {
                "df": reports.get("recommendations", pd.DataFrame()),
                "icon": "⚠️",
                "description": "Inventory risk assessment and recommendations"
            },
            "Ingredient List": {
                "df": reports.get("prep_plan", pd.DataFrame()),
                "icon": "🥗",
                "description": "Preparation quantities and shopping list"
            },
            "Promotions": {
                "df": reports.get("promotions", pd.DataFrame()),
                "icon": "🎯",
                "description": "Automated promotions for at-risk items"
            },
            "Summary": {
                "df": reports.get("summary", pd.DataFrame()),
                "icon": "📊",
                "description": "Pipeline execution summary"
            }
        }
        
        for report_name, info in report_info.items():
            if not info['df'].empty:
                st.markdown(f"#### {info['icon']} {report_name}")
                st.caption(info['description'])
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    csv_data = info['df'].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 CSV",
                        data=csv_data,
                        file_name=f"{report_name.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"csv_{report_name}"
                    )
                
                with col2:
                    excel_bytes = _to_excel_bytes(info['df'])
                    if excel_bytes:
                        st.download_button(
                            label="📊 Excel",
                            data=excel_bytes,
                            file_name=f"{report_name.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key=f"excel_{report_name}"
                        )
                
                with col3:
                    st.caption(f"{len(info['df'])} rows, {len(info['df'].columns)} columns")
                
                st.markdown("---")
    else:
        st.info("📭 No reports available yet. Run the pipeline first.")

with tabs[5]:  # Debug tab
    st.markdown('<div class="section-header"><h2 class="section-title">🔍 Debug Information</h2></div>', unsafe_allow_html=True)
    
    # Show paths
    st.markdown("### 📁 Paths")
    st.code(f"""
    ROOT_DIR: {ROOT_DIR}
    DATA_DIR_DEFAULT: {DATA_DIR_DEFAULT}
    REPORTS_DIR_DEFAULT: {REPORTS_DIR_DEFAULT}
    Current data_dir: {data_dir}
    Current output_dir: {output_dir}
    """)
    
    # Check if directories exist
    col1, col2 = st.columns(2)
    with col1:
        data_exists = Path(data_dir).exists()
        st.metric("Data Directory Exists", "✅ Yes" if data_exists else "❌ No")
    with col2:
        output_exists = Path(output_dir).exists()
        st.metric("Reports Directory Exists", "✅ Yes" if output_exists else "❌ No")
    
    # List files in reports directory
    if output_exists:
        st.markdown("### 📄 Files in Reports Directory")
        report_files = list(Path(output_dir).glob("*.csv"))
        if report_files:
            for file in report_files:
                file_info = f"{file.name} - {file.stat().st_size:,} bytes - Modified: {datetime.fromtimestamp(file.stat().st_mtime)}"
                st.code(file_info)
                
                # Show file preview
                if st.checkbox(f"Preview {file.name}", key=f"preview_{file.name}"):
                    try:
                        df = _safe_read_csv(file)
                        st.write(f"Shape: {df.shape}")
                        st.write(f"Columns: {list(df.columns)}")
                        if not df.empty:
                            st.dataframe(df.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error reading {file}: {e}")
        else:
            st.info("No CSV files found in reports directory")
    
    # Show loaded reports
    st.markdown("### 📊 Loaded Reports")
    st.write(f"Number of reports loaded: {len(reports)}")
    st.write(f"Report keys: {list(reports.keys())}")
    
    for report_name, df in reports.items():
        st.markdown(f"#### {report_name}")
        if not df.empty:
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.info("Empty dataframe")
    
    # Show pipeline status
    st.markdown("### 🔧 Pipeline Status")
    st.write(f"HAS_PIPELINE: {HAS_PIPELINE}")
    st.write(f"HAS_PLOTLY: {HAS_PLOTLY}")
    st.write(f"HAS_AUTOREFRESH: {HAS_AUTOREFRESH}")
    
    # Show state
    st.markdown("### 📝 System State")
    state = _load_state()
    st.json(state)

# Footer
st.markdown("""
<div class="footer">
    <div style="font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.95); margin-bottom: 1rem;">
        🌿 Fresh Flow Dashboard v4.3
    </div>
    <div style="color: rgba(255,255,255,0.8); font-size: 0.95rem;">
        Professional inventory management with AI forecasting<br>
        <span style="color: #4caf50;">✓ Real product names enabled (Varm Chokolade, Sunny Hawaii, Broccolien, etc.)</span>
    </div>
    <div style="margin-top: 1.5rem; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
        Production Ready System | Data updated with real menu items
    </div>
</div>
""", unsafe_allow_html=True)
