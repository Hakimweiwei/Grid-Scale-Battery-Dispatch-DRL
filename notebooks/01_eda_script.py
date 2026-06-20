# EDA 1: Price Volatility
# Analyzing AEMO price volatility.

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Load the data
df = pd.read_parquet('../data/processed/train.parquet')

# View data summary
print(df[['RRP', 'TOTALDEMAND', 'TOTALINTERMITTENTGENERATION']].describe())

# Plot RRP Histogram
fig_hist = px.histogram(df, x='RRP', title='Distribution of Regional Reference Price (RRP)', nbins=100)
# fig_hist.show()

# Plot Price vs Demand Scatter
fig_scatter = px.scatter(df, x='TOTALDEMAND', y='RRP', title='Price vs Total Demand', opacity=0.5)
# fig_scatter.show()

# Rolling Volatility
df['RRP_Volatility'] = df['RRP'].rolling(window=12*24).std()
fig_volatility = px.line(df, x='SETTLEMENTDATE_UTC', y='RRP_Volatility', title='RRP 24-Hour Rolling Volatility')
# fig_volatility.show()
