
# app.py
import streamlit as st
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).parent
SCRIPT = APP_DIR / "stock_scoring_live.py"

st.title("📈 Live Stock Scoring")

# Sanity check: show where we are and whether the script exists
st.write("Working directory:", str(APP_DIR))
st.write("Script path:", str(SCRIPT))
if not SCRIPT.exists():
    st.error("`stock_scoring_live.py` not found next to `app.py`. "
             "Please add/commit it to your repo root.")
else:
    default = "AAPL, MSFT, NVDA"
    tickers_input = st.text_input("Tickers (comma-separated)", default)
    run = st.button("Run scoring")

    if run:
        args = [sys.executable, str(SCRIPT), "--tickers"] + [
            t.strip() for t in tickers_input.split(",") if t.strip()
        ]
        with st.status("Running...", expanded=True) as s:
            st.write("Command:", " ".join(args))
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode != 0:
                s.update(label="❌ Error", state="error")
                st.error(proc.stderr or "Unknown error")
            else:
                s.update(label="✅ Done", state="complete")
                st.code(proc.stdout, language="text")
