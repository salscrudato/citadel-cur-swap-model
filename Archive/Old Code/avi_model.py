#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:18:46 2024

@author: salscrudato
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from utils import *

pio.renderers.default = 'browser'

# True turns off print/plots
toggle_print(0)
toggle_plot(1)

# Risk Aversion Parameter - not sure what this is doing
RISK_AVERSION = 1.0

# Target volatility
TARGET_VOLATILITY = 15000.00

# File Path
file_path = '/Users/salscrudato/Finance Tests/Front End Data.csv'
bbg_file_path = '/Users/salscrudato/Finance Tests/bbg_codes.xlsx'

# Set bbg tickers
bbg_tickers = set_bbg_ticker_mapping(bbg_file_path)

# Read the raw data
df_raw = load_raw_dataframe(file_path)

# Define the header row and skip rows
bbg_header_row = 2
bbg_skip_rows = bbg_header_row + 3

# Create ticker mapping
ticker_mapping = create_ticker_mapping(bbg_tickers)

# Generate new headers
new_headers = generate_new_headers(df_raw, bbg_header_row, ticker_mapping)

# Load the final dataframe with new headers
df = load_dataframe_with_new_headers(file_path, bbg_skip_rows, new_headers)

# Clean the dataframe | fillna, bfill/ffill, set Dates as index, sort ascending
df = preprocess_dataframe(df)

# Define column suffixes
suffixes = ['_2Y', '_3M2Y', '_ToT', 'USD', '_IV', '_CPI', '_MSCI']

# Creates column groups based on suffixes
grouped_columns = group_columns_by_suffix(df, suffixes)

for col_3m2y in grouped_columns['_3M2Y']:
    for col_2y in grouped_columns['_2Y']:
        currency_3m2y = col_3m2y.split('_')[0]
        currency_2y = col_2y.split('_')[0]
        if currency_3m2y == currency_2y:
            calculate_cnr_metrics(df, col_3m2y, col_2y, currency_3m2y, hl_lookback=174)

# Calculate rolling 5-day total returns and related metrics
df, df_rolling_5d_tr = calculate_rolling_returns(df, grouped_columns['_3M2Y'])

# Plot the rolling 5-day total returns
plot_time_series(df_rolling_5d_tr, title='Rolling 5-Day Total Returns', ylabel='5-Day Total Return', legend_title='Currencies')

# Use this function to plot a series of data (column) from a dataframe
# plot_single_column(df, 'CLP_2Y', title='CLP 2Y', ylabel='Price')

# Calculate the exponentially weighted covariance matrix and correlation matrix
ewm_cov_matrix, ewm_corr_matrix = calculate_cov_corr_matrices(df_rolling_5d_tr)

# Define expected returns, currency names, and rolling 5-day total return standard deviations
expected_returns, currencies, rolling_5d_tr_std_values = define_expected_returns_and_metrics(df, grouped_columns['_3M2Y'])

# Convert the list to a NumPy array for compatibility with CVXPY
expected_returns = np.array(expected_returns)

# Solve the quadratic utility optimization problem
optimal_weights = quadratic_utility_optimizer2(expected_returns, ewm_corr_matrix, RISK_AVERSION, 15000)
print("Optimized Weights:", optimal_weights)
            
# Calculate the portfolio variance
portfolio_variance = np.dot(optimal_weights.T, np.dot(ewm_corr_matrix, optimal_weights))

# Calculate the portfolio volatility (standard deviation)
portfolio_volatility = np.sqrt(portfolio_variance)

# Calculate the scaling factor
scaling_factor = TARGET_VOLATILITY / portfolio_volatility

# Rescale the optimal weights
scaled_optimal_weights = optimal_weights * scaling_factor

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Currency': currencies,
    'Optimal Risk Weight': [round(weight, 2) for weight in scaled_optimal_weights],
    'Std Dev': [round(value) for value in rolling_5d_tr_std_values],
    'Expected Return': [round(ret, 2) for ret in expected_returns]
})

results_df['Optimal DV01'] = (results_df['Optimal Risk Weight'] / results_df['Std Dev']).round(2)  # Round for display

# Print the results
print("\nDetailed Results:")
print(results_df)
create_professional_colored_table(results_df)

# Recalculate the portfolio variance with scaled weights
scaled_portfolio_variance = np.dot(scaled_optimal_weights.T, np.dot(ewm_corr_matrix, scaled_optimal_weights))

# Recalculate the portfolio volatility (standard deviation)
scaled_portfolio_volatility = np.sqrt(scaled_portfolio_variance)

#print(f"\nOriginal Portfolio Expected Volatility: {portfolio_volatility:.2f}")
print(f"Scaled Portfolio Expected Volatility: {scaled_portfolio_volatility:.2f}")
