import streamlit as st
import subprocess
import sys
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Live Stock Scoring", layout="wide")
st.title("📈 Live Stock Scoring")

# Resolve script path relative to this app file
APP_DIR = Path(__file__).parent
SCRIPT = APP_DIR / "stock_scoring_live.py"  # adjust if your script lives elsewhere

# We'll always save results here (safe on Streamlit Cloud)
OUT_PATH = Path("/tmp/scores.csv")

# Diagnostics to help debug on Streamlit Cloud
with st.expander("Diagnostics", expanded=False):
    st.write("App directory:", str(APP_DIR))
    st.write("Script path:", str(SCRIPT))
    st.write("Script exists:", SCRIPT.exists())

# UI controls
default = "AAPL, MSFT, NVDA"
tickers_input = st.text_input("Tickers (comma-separated)", default)
limit = st.slider("Limit (for S&P 500 mode)", 10, 503, 50)
period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)
interval = st.selectbox("Interval", ["1d", "1h"], index=0)
download_name = st.text_input("Download CSV filename", "scores.csv")

col1, col2 = st.columns(2)
run_custom = col1.button("Run Custom Tickers")
run_sp500 = col2.button("Run S&P 500 (Wikipedia)")

def run_cli(args):
    """Run the CLI script with robust path and show status/logs, then display CSV."""
    if not SCRIPT.exists():
        st.error("`stock_scoring_live.py` not found next to `app.py`.\n"
                 "Move/commit the file to the repo root or update SCRIPT path.")
        return

    # ensure we always overwrite the same CSV
    if OUT_PATH.exists():
        try:
            OUT_PATH.unlink()
        except Exception:
            pass

    cmd = [sys.executable, str(SCRIPT)] + args + ["--out", str(OUT_PATH)]
    with st.status("Running scoring…", expanded=True) as status:
        st.write("Command:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Show raw output (helpful for quick debugging)
        with st.expander("Raw output log", expanded=False):
            st.code(proc.stdout or "(no stdout)", language="text")
            if proc.stderr:
                st.code(proc.stderr, language="text")

        if proc.returncode != 0:
            status.update(label="❌ Finished with error", state="error")
            st.error(proc.stderr or "Unknown error")
            return

        status.update(label="✅ Finished successfully", state="complete")
        st.success("Done.")

    # Load and display results
    if OUT_PATH.exists():
        try:
            df = pd.read_csv(OUT_PATH, index_col=0)
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=OUT_PATH.read_bytes(),
                file_name=download_name,
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        st.warning("No CSV found after run. Did the script skip all tickers?")

# Actions
if run_custom:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    args = ["--tickers"] + tickers + ["--period", period, "--interval", interval]
    run_cli(args)

if run_sp500:
    args = ["--sp500", "--limit", str(limit), "--period", period, "--interval", interval]
    run_cli(args)
