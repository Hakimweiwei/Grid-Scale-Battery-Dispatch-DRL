# EDA 2: Duck Curve and FCAS
# Analyzing solar penetration and frequency control markets.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_parquet('../data/processed/train.parquet')

# Calculate average profile per hour of the day
hourly_profile = df.groupby('hour_of_day')[['TOTALDEMAND', 'TOTALINTERMITTENTGENERATION']].mean().reset_index()
hourly_profile['Net Demand'] = hourly_profile['TOTALDEMAND'] - hourly_profile['TOTALINTERMITTENTGENERATION']

# Plot Duck Curve
fig_duck = go.Figure()
fig_duck.add_trace(go.Scatter(x=hourly_profile['hour_of_day'], y=hourly_profile['TOTALDEMAND'], name='Total Demand'))
fig_duck.add_trace(go.Scatter(x=hourly_profile['hour_of_day'], y=hourly_profile['Net Demand'], name='Net Demand', line=dict(dash='dash')))
fig_duck.update_layout(title='The Duck Curve: Average Daily Demand Profile', xaxis_title='Hour of Day', yaxis_title='MW')
# fig_duck.show()

# Average FCAS Prices
fcas_cols = ['RAISE6SECRRP', 'LOWER6SECRRP', 'RAISE60SECRRP', 'LOWER60SECRRP', 'RAISE5MINRRP', 'LOWER5MINRRP']
fcas_means = df[fcas_cols].mean().reset_index()
fcas_means.columns = ['Market', 'Average Price (AUD)']
fig_fcas = px.bar(fcas_means, x='Market', y='Average Price (AUD)', title='Average FCAS Prices')
# fig_fcas.show()
