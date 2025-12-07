
import streamlit as st
import subprocess
import sys
from typing import List

st.set_page_config(page_title="Live Stock Scoring", layout="wide")

st.title("📈 Live Stock Scoring")
st.write("Tip: start with a few tickers or set a low S&P 500 limit, then scale up.")

# Controls
default = "AAPL, MSFT, NVDA"
tickers_input = st.text_input("Tickers (comma-separated)", default)
limit = st.slider("Limit (for S&P 500 mode)", 10, 503, 50)
period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)
interval = st.selectbox("Interval", ["1d", "1h"], index=0)
csv_out = st.text_input("Optional CSV output filename", "scores.csv")

col1, col2 = st.columns(2)
run_custom = col1.button("Run Custom Tickers")
run_sp500 = col2.button("Run S&P 500 (Wikipedia)")

def run_cli(args: List[str]):
    with st.status("Running scoring…", expanded=True) as status:
        cmd = [sys.executable, "stock_scoring_live.py"] + args
        st.write("Command:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            status.update(label="❌ Finished with error", state="error")
            st.error(proc.stderr or "Unknown error")
        else:
            status.update(label="✅ Finished successfully", state="complete")
            st.success("Done.")
            st.code(proc.stdout, language="text")

if run_custom:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    args = ["--tickers"] + tickers + ["--period", period, "--interval", interval]
    if csv_out:
        args += ["--out", csv_out]
    run_cli(args)

if run_sp500:
    args = ["--sp500", "--limit", str(limit), "--period", period, "--interval", interval]
    if csv_out:
        args += ["--out", csv_out]
    run_cli(args)
