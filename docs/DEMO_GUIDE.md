# Demo Guide

## Generate Sample Data

1. Run `python scripts/generate_synthetic_data.py --output data/Inventory_Management`.

## Run Full Pipeline

1. Run `python scripts/run_full_pipeline.py --data-dir data/Inventory_Management --output-dir reports`.
2. Open the CSVs generated in the `reports` directory.

## Run Streamlit Dashboard

1. Run `streamlit run app_streamlit.py`.
2. Select the data directory and run the pipeline from the UI.

## Run API Server

1. Run `python scripts/run_api.py`.
2. Test the health endpoint at `http://localhost:5000/api/health`.
