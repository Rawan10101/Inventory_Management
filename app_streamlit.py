"""
Business dashboard for Fresh Flow.
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import io
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

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
DATA_DIR_DEFAULT = ROOT_DIR / "data"
REPORTS_DIR_DEFAULT = ROOT_DIR / "reports"
STATE_FILE = DATA_DIR_DEFAULT / ".system_state.json"
BACKUP_DIR = DATA_DIR_DEFAULT / "backups"

PIPELINE_DIR = ROOT_DIR / "src" / "pipeline"
MODELS_DIR = ROOT_DIR / "src" / "models"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

from pipeline_runner import run_pipeline
from data_loader import DataLoader

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
    "Ingredient list (recipes)": {
        "file": "dim_bill_of_materials.csv",
        "required_any": [
            ["menu_item_id", "ingredient_id", "quantity_per_serving"],
            ["parent_sku_id", "sku_id", "quantity"],
        ],
        "key_options": [
            ["menu_item_id", "ingredient_id"],
            ["parent_sku_id", "sku_id"],
        ],
        "numeric_cols": ["quantity_per_serving", "quantity"],
    },
    "Places": {
        "file": "dim_places.csv",
        "required": ["id", "title"],
        "key": ["id"],
    },
}
DATASET_OPTIONS = list(DATASETS.keys()) + ["Other dataset"]


st.set_page_config(page_title="Fresh Flow Dashboard", layout="wide")


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
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _get_key_columns(meta: Dict, df: pd.DataFrame) -> List[str]:
    if "key" in meta:
        return meta["key"]
    if "key_options" in meta:
        for option in meta["key_options"]:
            if all(col in df.columns for col in option):
                return option
    return []


def _validate_upload(df: pd.DataFrame, meta: Dict) -> List[str]:
    errors = []

    if "required_any" in meta:
        valid = any(all(col in df.columns for col in group) for group in meta["required_any"])
        if not valid:
            options = [", ".join(group) for group in meta["required_any"]]
            errors.append("Your file must include columns: " + " OR ".join(options))
    else:
        missing = [col for col in meta.get("required", []) if col not in df.columns]
        if missing:
            for col in missing:
                errors.append(f"Your file is missing the column: {col}")

    for col in meta.get("numeric_cols", []):
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            bad = converted.isna() & df[col].notna()
            if bad.any():
                errors.append(f"Column '{col}' must be numeric")

    for col in meta.get("date_cols", []):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            bad = parsed.isna() & df[col].notna()
            if bad.any():
                errors.append(f"Column '{col}' must be a date or timestamp")

    return errors


def _merge_data(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "added": 0,
        "modified": 0,
        "ignored": 0,
        "duplicates": 0,
    }

    if incoming.empty:
        return existing.copy(), stats

    if key_cols:
        stats["duplicates"] = int(incoming.duplicated(subset=key_cols).sum())
        incoming = incoming.drop_duplicates(subset=key_cols, keep="last")

    if mode == "Replace file":
        stats["added"] = len(incoming)
        return incoming.copy(), stats

    if existing.empty:
        if mode == "Modify existing records":
            stats["ignored"] = len(incoming)
            return existing.copy(), stats
        stats["added"] = len(incoming)
        return incoming.copy(), stats

    existing = existing.copy()
    incoming = incoming.copy()

    if mode == "Add new records":
        existing_idx = existing.set_index(key_cols, drop=False)
        incoming_idx = incoming.set_index(key_cols, drop=False)
        new_index = incoming_idx.index.difference(existing_idx.index)
        new_rows = incoming_idx.loc[new_index].reset_index(drop=True)
        stats["added"] = len(new_rows)
        merged = pd.concat([existing, new_rows], ignore_index=True)
        return merged, stats

    if mode == "Modify existing records":
        existing_idx = existing.set_index(key_cols)
        incoming_idx = incoming.set_index(key_cols)
        common = existing_idx.index.intersection(incoming_idx.index)
        stats["modified"] = len(common)
        stats["ignored"] = len(incoming_idx.index.difference(existing_idx.index))

        updated = existing_idx.copy()
        for col in incoming_idx.columns:
            if col not in updated.columns:
                updated[col] = pd.NA
            updated.loc[common, col] = incoming_idx.loc[common, col].combine_first(updated.loc[common, col])

        merged = updated.reset_index()
        return merged, stats

    return existing.copy(), stats


def _change_ratio(stats: Dict[str, int], base_count: int, mode: str) -> float:
    if mode == "Replace file":
        return 1.0
    if base_count <= 0:
        return 1.0 if (stats["added"] + stats["modified"]) > 0 else 0.0
    return (stats["added"] + stats["modified"]) / base_count


def _load_latest_reports(report_dir: Path) -> Dict[str, pd.DataFrame]:
    report_dir = Path(report_dir)
    reports = {}
    for prefix in ["summary", "forecast", "recommendations", "promotions", "prep_plan", "model_metrics"]:
        path = _latest_report(report_dir, prefix)
        if path:
            reports[prefix] = _safe_read_csv(path)
    return reports


def _load_daily_sales(data_dir: Path) -> pd.DataFrame:
    orders_path = Path(data_dir) / "fct_orders.csv"
    items_path = Path(data_dir) / "fct_order_items.csv"
    items = _safe_read_csv(items_path)
    orders = _safe_read_csv(orders_path)

    if items.empty or orders.empty:
        return pd.DataFrame()

    orders["order_created_at"] = pd.to_datetime(orders["created"], unit="s", errors="coerce")
    orders = orders[orders.get("status", "Closed") == "Closed"]

    merged = items.merge(
        orders[["id", "place_id", "order_created_at"]],
        left_on="order_id",
        right_on="id",
        how="left",
    )
    merged["date"] = pd.to_datetime(merged["order_created_at"], errors="coerce").dt.date

    daily = (
        merged.groupby(["item_id", "date"], dropna=True)["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "quantity_sold"})
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


def _to_excel_bytes(df: pd.DataFrame) -> bytes | None:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()
    except Exception:
        return None


def _to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes | None:
    if not HAS_PDF:
        return None
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, title)
    y -= 20
    pdf.setFont("Helvetica", 9)
    for line in df.head(60).to_string(index=False).splitlines():
        if y < 40:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 9)
        pdf.drawString(40, y, line[:110])
        y -= 12
    pdf.save()
    return buffer.getvalue()


def _style_risk(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_risk(val: str) -> str:
        if str(val).lower() == "critical":
            return "background-color: #ffcccc"
        if str(val).lower() == "high":
            return "background-color: #ffe5b4"
        if str(val).lower() == "medium":
            return "background-color: #fff3cd"
        return "background-color: #d4edda"

    if "risk_category" in df.columns:
        return df.style.applymap(color_risk, subset=["risk_category"])
    return df.style


st.title("Fresh Flow Business Dashboard")
st.caption("Simple, real-time decisions for sales and inventory.")

with st.sidebar:
    st.subheader("Settings")
    data_dir = st.text_input("Data folder", str(DATA_DIR_DEFAULT))
    output_dir = st.text_input("Reports folder", str(REPORTS_DIR_DEFAULT))

    auto_refresh = st.checkbox("Auto refresh dashboard", value=False)
    refresh_seconds = st.slider("Refresh interval (seconds)", 30, 600, 120)
    if auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=refresh_seconds * 1000, key="auto_refresh")
    elif auto_refresh:
        st.info("Auto refresh needs streamlit-autorefresh. Use Refresh now for manual updates.")

    if st.button("Refresh now"):
        st.rerun()

    state = _load_state()
    if state.get("last_refresh"):
        st.caption(f"Last system update: {state.get('last_refresh')}")


tabs = st.tabs(["Dashboard", "Upload Data", "Exports"])

with tabs[0]:
    if st.button("Update insights now"):
        with st.spinner("Updating insights..."):
            run_pipeline(data_dir=str(data_dir), output_dir=str(output_dir))
        state = _load_state()
        state["last_refresh"] = datetime.now().isoformat(timespec="seconds")
        _save_state(state)
        st.success("System updated with latest data.")

    reports = _load_latest_reports(Path(output_dir))
    summary_df = reports.get("summary", pd.DataFrame())
    forecast_df = reports.get("forecast", pd.DataFrame())
    recommendations_df = reports.get("recommendations", pd.DataFrame())
    prep_df = reports.get("prep_plan", pd.DataFrame())

    if summary_df.empty:
        st.info("No reports found yet. Use 'Update insights now' to generate reports.")
    else:
        summary = summary_df.iloc[0].to_dict()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected demand", f"{summary.get('total_predicted_demand', 0):,.0f}")
        col2.metric("At-risk items", f"{summary.get('at_risk_items_count', 0):,.0f}")
        col3.metric("Potential waste", f"{summary.get('potential_waste_value', 0):,.0f}")
        col4.metric("Potential recovery", f"{summary.get('potential_recovery_value', 0):,.0f}")

    st.subheader("Today’s Forecast")
    if not forecast_df.empty:
        daily_sales = _load_daily_sales(Path(data_dir))
        if not daily_sales.empty and "item_id" in daily_sales.columns:
            trends = daily_sales.groupby("item_id")["quantity_sold"].apply(_trend_label)
            confidence = daily_sales.groupby("item_id")["quantity_sold"].apply(_confidence_label)
            forecast_df = forecast_df.copy()
            forecast_df["trend"] = forecast_df["item_id"].map(trends).fillna("Stable")
            forecast_df["confidence"] = forecast_df["item_id"].map(confidence).fillna("Low")
        st.dataframe(
            forecast_df[["item_name", "predicted_daily_demand", "trend", "confidence"]]
            .rename(columns={
                "item_name": "Product",
                "predicted_daily_demand": "Expected sales",
                "trend": "Trend",
                "confidence": "Confidence",
            })
            .head(50)
        )
    else:
        st.info("Forecast not available yet.")

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

with tabs[1]:
    st.subheader("Upload Data File")
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])
    dataset_choice = st.selectbox("Which data file do you want to update?", DATASET_OPTIONS)
    update_mode = st.radio(
        "How should the data be applied?",
        ["Add new records", "Modify existing records", "Replace file"],
        horizontal=True,
    )

    target_file = None
    key_cols = []

    if dataset_choice != "Other dataset":
        meta = DATASETS[dataset_choice]
        target_file = Path(data_dir) / meta["file"]
    else:
        file_name = st.text_input("File name in the data folder", "custom.csv")
        target_file = Path(data_dir) / file_name
        meta = {"required": [], "key": []}

    if uploaded_file is not None:
        try:
            incoming_df = _read_upload(uploaded_file)
        except Exception as exc:
            st.error(str(exc))
            incoming_df = pd.DataFrame()

        if not incoming_df.empty:
            st.write("Preview of uploaded data")
            st.dataframe(incoming_df.head(50))

            errors = _validate_upload(incoming_df, meta)
            if errors:
                for err in errors:
                    st.error(err)
            else:
                existing_df = _safe_read_csv(target_file) if target_file else pd.DataFrame()
                key_cols = _get_key_columns(meta, incoming_df)
                if update_mode != "Replace file" and not key_cols:
                    st.error("Key columns are required for add/modify updates.")
                else:
                    merged_df, stats = _merge_data(existing_df, incoming_df, key_cols, update_mode)
                    st.write("Update summary")
                    st.write({
                        "rows_added": stats["added"],
                        "rows_modified": stats["modified"],
                        "rows_ignored": stats["ignored"],
                        "duplicates_in_upload": stats["duplicates"],
                    })
                    if st.button("Apply changes"):
                        Path(data_dir).mkdir(parents=True, exist_ok=True)
                        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                        if target_file and target_file.exists():
                            backup_path = BACKUP_DIR / f"{target_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            target_file.replace(backup_path)

                        merged_df.to_csv(target_file, index=False)

                        base_count = len(existing_df)
                        ratio = _change_ratio(stats, base_count, update_mode)

                        state = _load_state()
                        state.setdefault("datasets", {})
                        state["datasets"][str(target_file.name)] = {
                            "rows": int(len(merged_df)),
                            "change_ratio": round(ratio, 4),
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        _save_state(state)

                        st.success("Data updated. System will refresh if needed.")

                        if ratio >= 0.05:
                            with st.spinner("Updating insights..."):
                                run_pipeline(data_dir=str(data_dir), output_dir=str(output_dir))
                            state["last_refresh"] = datetime.now().isoformat(timespec="seconds")
                            _save_state(state)
                            st.success("System updated with latest data.")
                        else:
                            st.info("Data changes were small. No full refresh was needed.")

with tabs[2]:
    st.subheader("Exports")
    reports = _load_latest_reports(Path(output_dir))
    forecast_df = reports.get("forecast", pd.DataFrame())
    recommendations_df = reports.get("recommendations", pd.DataFrame())
    prep_df = reports.get("prep_plan", pd.DataFrame())

    if not forecast_df.empty:
        st.write("Download Today’s Forecast")
        st.download_button(
            "Download CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast.csv",
            mime="text/csv",
        )
        excel_bytes = _to_excel_bytes(forecast_df)
        if excel_bytes:
            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name="forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        pdf_bytes = _to_pdf_bytes(forecast_df, "Today's Forecast")
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="forecast.pdf",
                mime="application/pdf",
            )

    if not recommendations_df.empty:
        st.write("Download Alerts and Risks")
        st.download_button(
            "Download CSV",
            data=recommendations_df.to_csv(index=False).encode("utf-8"),
            file_name="alerts.csv",
            mime="text/csv",
        )
        excel_bytes = _to_excel_bytes(recommendations_df)
        if excel_bytes:
            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name="alerts.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        pdf_bytes = _to_pdf_bytes(recommendations_df, "Alerts and Risks")
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="alerts.pdf",
                mime="application/pdf",
            )

    if not prep_df.empty:
        st.write("Download Preparation Plan")
        st.download_button(
            "Download CSV",
            data=prep_df.to_csv(index=False).encode("utf-8"),
            file_name="prep_plan.csv",
            mime="text/csv",
        )
        excel_bytes = _to_excel_bytes(prep_df)
        if excel_bytes:
            st.download_button(
                "Download Excel",
                data=excel_bytes,
                file_name="prep_plan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        pdf_bytes = _to_pdf_bytes(prep_df, "Preparation Plan")
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name="prep_plan.pdf",
                mime="application/pdf",
            )
