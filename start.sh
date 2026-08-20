#!/usr/bin/env bash
set -e

# Local helper to create venv and run the app (for developers)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
