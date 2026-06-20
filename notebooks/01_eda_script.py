# EDA 1: Price Volatility
# Analyzing AEMO price volatility.

import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg') # Prevent display errors
import matplotlib.pyplot as plt
import os

os.makedirs('../reports/figures', exist_ok=True)

# Load the data
df = pd.read_parquet('../data/processed/train.parquet')

# View data summary
print(df[['RRP', 'TOTALDEMAND', 'TOTALINTERMITTENTGENERATION']].describe())

# Plot RRP Histogram
fig_hist = px.histogram(df, x='RRP', title='Distribution of Regional Reference Price (RRP)', nbins=100)
fig_hist.write_html('../reports/figures/01_rrp_histogram.html')
try:
    fig_hist.write_image('../reports/figures/01_rrp_histogram.png')
except ValueError as e:
    print("Could not write png:", e)

# Plot Price vs Demand Scatter
fig_scatter = px.scatter(df, x='TOTALDEMAND', y='RRP', title='Price vs Total Demand', opacity=0.5)
fig_scatter.write_html('../reports/figures/01_price_vs_demand.html')
try:
    fig_scatter.write_image('../reports/figures/01_price_vs_demand.png')
except ValueError as e:
    print("Could not write png:", e)

# Rolling Volatility
df['RRP_Volatility'] = df['RRP'].rolling(window=12*24).std()
fig_volatility = px.line(df, x='SETTLEMENTDATE_UTC', y='RRP_Volatility', title='RRP 24-Hour Rolling Volatility')
fig_volatility.write_html('../reports/figures/01_rrp_volatility.html')
try:
    fig_volatility.write_image('../reports/figures/01_rrp_volatility.png')
except ValueError as e:
    print("Could not write png:", e)
