# app.py
import streamlit as st
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent
SCRIPT = APP_DIR / "stock_scoring_live.py"

st.set_page_config(page_title="Live Stock Scoring", layout="wide")
st.title("📈 Live Stock Scoring")

# Diagnostics to verify files/paths on Streamlit Cloud
st.write("Working directory:", str(APP_DIR))
st.write("Script exists:", SCRIPT.exists(), "| Path:", str(SCRIPT))

tickers_input = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA")
run = st.button("Run scoring")

if run:
    if not SCRIPT.exists():
        st.error("stock_scoring_live.py not found next to app.py. "
                 "Commit/push the file to the repo root.")
    else:
        args = [sys.executable, str(SCRIPT), "--tickers"] + [
            t.strip() for t in tickers_input.split(",") if t.strip()
        ]
        with st.status("Running…", expanded=True) as s:
            st.write("Command:", " ".join(args))
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode != 0:
                s.update(label="❌ Error", state="error")
                st.error(proc.stderr or "Unknown error")
            else:
                s.update(label="✅ Done", state="complete")
                st.code(proc.stdout, language="text")
