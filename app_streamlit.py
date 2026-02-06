"""
Streamlit dashboard for Fresh Flow.
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "src" / "pipeline"
SCRIPTS_DIR = ROOT_DIR / "scripts"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from pipeline_runner import run_pipeline
from generate_synthetic_data import generate_synthetic_data


st.title("Fresh Flow Inventory Intelligence")

st.write("Run the full demand forecasting and inventory optimization pipeline.")

data_dir = st.text_input("Data directory", str(ROOT_DIR / "data" / "Inventory_Management"))
output_dir = st.text_input("Output directory", str(ROOT_DIR / "reports"))

if st.button("Generate sample data"):
    generate_synthetic_data(data_dir)
    st.success("Sample data generated.")

if st.button("Run pipeline"):
    report = run_pipeline(data_dir=data_dir, output_dir=output_dir)

    st.subheader("Summary")
    st.json(report["summary"])

    st.subheader("Demand Forecast")
    st.dataframe(report["demand_forecast"].head(50))

    if not report["at_risk_items"].empty:
        st.subheader("At-Risk Items")
        st.dataframe(report["at_risk_items"].head(50))

    if report["promotions"]:
        st.subheader("Promotions")
        st.dataframe(report["promotions"])

    if not report["prep_plan"].empty:
        st.subheader("Prep Plan")
        st.dataframe(report["prep_plan"].head(50))
