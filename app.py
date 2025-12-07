import streamlit as st

st.set_page_config(page_title="Health Check", layout="centered")
st.title("✅ Streamlit is running")
st.write("If you can see this, the app passed the health check and is alive.")

st.divider()
st.write("Next steps:")
st.markdown("1. Replace this file with your full UI or keep it as a quick diagnostics page.")
st.markdown("2. Add buttons to run your scoring script only after the page loads.")
