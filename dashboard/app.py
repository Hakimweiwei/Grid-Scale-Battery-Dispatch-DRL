import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Grid-Scale Battery Dispatch DRL", layout="wide")

st.title("🔋 Grid-Scale Battery Dispatch Dashboard")
st.markdown("Interactive visualization of DRL agent performance in the AEMO market.")

@st.cache_data
def load_results():
    try:
        return pd.read_csv("reports/backtest_results.csv")
    except FileNotFoundError:
        return None

results = load_results()

if results is not None:
    st.success("Results loaded successfully.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Profit (AUD)", f"${results['profit'].sum():,.2f}")
    col2.metric("Final SOC", f"{results['soc'].iloc[-1]*100:.1f}%")
    col3.metric("Final SOH", f"{results['soh'].iloc[-1]*100:.3f}%")
    
    st.subheader("Cumulative Profit Over Time")
    results['cumulative_profit'] = results['profit'].cumsum()
    fig1 = px.line(results, x='step', y='cumulative_profit', title="Cumulative Profit (AUD)")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Battery Action vs SOC")
    fig2 = px.line(results[:288], x='step', y=['actual_arb_power', 'soc'], title="First 24 Hours: Dispatch vs SOC")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No backtest results found. Please run the backtester first.")
