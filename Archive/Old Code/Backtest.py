#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:29:13 2024

@author: salscrudato
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cvxpy as cp
#from tqdm import tqdm
import time
def load_and_process_data(file_path):
    # Load the CSV file into a DataFrame with the first column as the index and parse dates
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Function to safely merge cumulative return into the DataFrame
    def safe_merge(df, currency_df, col_name, retries=3):
        for i in range(retries):
            try:
                if col_name in df.columns:
                    df.drop(columns=[col_name], inplace=True)  # Remove the column if it already exists
                df = df.merge(currency_df[[col_name]], how='left', left_index=True, right_index=True)
                return df
            except Exception as e:
                print(f"Error during merge: {e}. Retrying {i + 1}/{retries}...")
                time.sleep(1)  # Wait for 1 second before retrying
        raise Exception(f"Failed to merge after {retries} retries.")

    # Loop through each unique currency in the DataFrame and calculate the cumulative gross and net returns
    for currency in df['Currency'].unique():
        # Filter the DataFrame for the current currency
        currency_df = df[df['Currency'] == currency].copy()

        # Calculate the cumulative gross return for the current currency
        gross_col_name = f'{currency}_CUM_GROSS_RETURN'
        currency_df[gross_col_name] = currency_df['1D Gross Return'].cumsum()

        # Calculate the cumulative net return for the current currency
        net_col_name = f'{currency}_CUM_NET_RETURN'
        currency_df[net_col_name] = currency_df['1D Net Return'].cumsum()

        # Merge the cumulative gross return back into the original DataFrame
        df = safe_merge(df, currency_df, gross_col_name)

        # Merge the cumulative net return back into the original DataFrame
        df = safe_merge(df, currency_df, net_col_name)

    # Create the Portfolio_CUM_GROSS_RETURN and Portfolio_CUM_NET_RETURN by summing the respective cumulative returns of all currencies
    cumulative_gross_return_cols = [f'{currency}_CUM_GROSS_RETURN' for currency in df['Currency'].unique()]
    cumulative_net_return_cols = [f'{currency}_CUM_NET_RETURN' for currency in df['Currency'].unique()]
    df['Portfolio_CUM_GROSS_RETURN'] = df[cumulative_gross_return_cols].sum(axis=1)
    df['Portfolio_CUM_NET_RETURN'] = df[cumulative_net_return_cols].sum(axis=1)

    return df

def consolidate_returns(df, return_col):
    # Consolidate the cumulative returns into a new DataFrame with one row per date
    consolidated_df = df.groupby(df.index.date).first()
    consolidated_df.index = pd.to_datetime(consolidated_df.index)
    consolidated_df = consolidated_df.sort_index()
    consolidated_df[f'{return_col}_5D_Return'] = consolidated_df[return_col].diff(5)
    return consolidated_df

def calculate_statistics(df, return_type='gross'):  #NEW: Added return_type parameter
    return_col = 'Portfolio_CUM_GROSS_RETURN' if return_type == 'gross' else 'Portfolio_CUM_NET_RETURN'

    # Consolidate the returns for proper calculation of 5-day changes
    consolidated_df = consolidate_returns(df, return_col)

    # Calculate the average number of non-zero "1-Day Change" entries per calendar day since 2013
    df_filtered = df[df.index >= pd.Timestamp('2013-01-01')]
    df_non_zero = df_filtered[df_filtered['1-Day Change'] != 0]
    daily_counts = df_non_zero.groupby(df_non_zero.index.date)['1-Day Change'].count()
    average_trades_per_day = daily_counts.mean()

    # Calculate the mean and standard deviation of the 5-day returns
    returns_5d = consolidated_df[f'{return_col}_5D_Return'].dropna()
    mean_return = returns_5d.mean()
    std_dev_return = returns_5d.std()
    
    # Manually calculate the Sharpe Ratio
    sharpe_ratio_5d = (mean_return / std_dev_return) * np.sqrt(252 / 5)

    # Calculate the average annual return since inception for the portfolio cumulative return series
    total_years = (consolidated_df.index[-1] - consolidated_df.index[0]).days / 365.25
    most_recent_cum_return = consolidated_df[return_col].iloc[-1]
    average_annual_nominal_change = most_recent_cum_return / total_years

    # Calculate the annualized volatility of the 5-day changes in the cumulative return series
    vol_5d_annualized = std_dev_return * np.sqrt(252 / 5)

    # Calculate the maximum drawdown since inception of the cumulative return
    def calculate_max_drawdown(series):
        peak = series.cummax()
        drawdown = peak - series
        max_drawdown = drawdown.max()
        return max_drawdown

    max_drawdown = calculate_max_drawdown(consolidated_df[return_col])  #NEW: Updated column name
    max_drawdown_percentage = (max_drawdown / average_annual_nominal_change) * 100

    # Calculate the maximum number of months underwater for the portfolio cumulative return series
    def calculate_max_months_underwater(series):
        series = series.sort_index()
        peak = series.cummax()
        underwater = series < peak
        groups = (underwater != underwater.shift()).cumsum()
        df_underwater = pd.DataFrame({'series': series, 'peak': peak, 'underwater': underwater, 'group': groups})
        df_underwater['month'] = df_underwater.index.to_period('M')
        underwater_duration = df_underwater[df_underwater['underwater']].groupby('group')['month'].nunique()
        max_months_underwater = underwater_duration.max()
        return max_months_underwater

    max_months_underwater = calculate_max_months_underwater(consolidated_df[return_col])

    return {
        f"Average Trades per Day ({return_type})": average_trades_per_day,
        f"Sharpe Ratio ({return_type})": sharpe_ratio_5d,
        f"Average Annual Return ({return_type})": average_annual_nominal_change,
        f"Annualized Volatility of 5-Day Changes ({return_type})": vol_5d_annualized,
        f"Maximum Drawdown ({return_type})": max_drawdown,
        f"Max Drawdown % of Average Annual Return ({return_type})": max_drawdown_percentage,
        f"Max Months Underwater ({return_type})": max_months_underwater
    }

def compare_portfolios(file_path1, file_path2):
    # Load and process the data for both portfolios
    df1 = load_and_process_data(file_path1)
    df2 = load_and_process_data(file_path2)

    # Calculate statistics for both portfolios for gross and net returns
    stats1_gross = calculate_statistics(df1, return_type='gross')
    stats1_net = calculate_statistics(df1, return_type='net')
    stats2_gross = calculate_statistics(df2, return_type='gross')
    stats2_net = calculate_statistics(df2, return_type='net')

    # Combine the statistics into a single dictionary for each portfolio
    stats1 = {**stats1_gross, **stats1_net}
    stats2 = {**stats2_gross, **stats2_net}

    # Calculate historical correlation between the 5-day changes of the two portfolios
    df1['Portfolio_5D_Return'] = df1['Portfolio_CUM_GROSS_RETURN'].diff(5)
    df2['Portfolio_5D_Return'] = df2['Portfolio_CUM_GROSS_RETURN'].diff(5)
    combined_df = pd.DataFrame({
        'Portfolio_1_5D_Return': df1['Portfolio_5D_Return'],
        'Portfolio_2_5D_Return': df2['Portfolio_5D_Return']
    }).dropna()
    historical_correlation = combined_df['Portfolio_1_5D_Return'].corr(combined_df['Portfolio_2_5D_Return'])

    # Add the historical correlation to the stats dictionary
    stats1['Historical Correlation'] = historical_correlation
    stats2['Historical Correlation'] = historical_correlation

    # Create a DataFrame to display the statistics side-by-side
    comparison_df = pd.DataFrame([stats1, stats2], index=["Portfolio 1", "Portfolio 2"])

    return comparison_df

# Define the file paths for the two portfolios. So far use ToT 313, CnR Mom 154, Mom 153, FX 324, MSCI 212, IV 4/242/3
file_path1 = '/Users/salscrudato/Finance Tests/Outputs/output_cnr_hl_63.csv'
file_path2 = '/Users/salscrudato/Finance Tests/Outputs/output_cnr_hl_130.csv'

# Compare the two portfolios
comparison_df = compare_portfolios(file_path1, file_path2)

# Print the comparison DataFrame
print(comparison_df)

# Compare the two portfolios and get the processed DataFrames
df1 = load_and_process_data(file_path1)
df2 = load_and_process_data(file_path2)

# Plot the net cumulative return over time for both portfolios
plt.figure(figsize=(10, 6))
plt.plot(df1['Portfolio_CUM_NET_RETURN'], label='Net Cumulative Return (Portfolio 1)')
plt.plot(df2['Portfolio_CUM_NET_RETURN'], label='Net Cumulative Return (Portfolio 2)')
plt.title('Net Cumulative Return Over Time')
plt.xlabel('Date')
plt.ylabel('Net Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()

def plot_rolling_correlation(df1, df2, window=756):  # 3 years * 252 trading days/year
    df1['Portfolio_5D_Return'] = df1['Portfolio_CUM_GROSS_RETURN'].diff(5)
    df2['Portfolio_5D_Return'] = df2['Portfolio_CUM_GROSS_RETURN'].diff(5)

    combined_df = pd.DataFrame({
        'Portfolio_1_5D_Return': df1['Portfolio_5D_Return'],
        'Portfolio_2_5D_Return': df2['Portfolio_5D_Return']
    }).dropna()

    rolling_corr = combined_df['Portfolio_1_5D_Return'].rolling(window=window).corr(combined_df['Portfolio_2_5D_Return'])

    plt.figure(figsize=(14, 7))
    plt.plot(rolling_corr, label='3-Year Rolling Correlation', color='blue')
    plt.title('3-Year Rolling Correlation between Portfolio 1 and Portfolio 2')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()

# Load and process the data (assuming load_and_process_data function is defined as above)
df1 = load_and_process_data(file_path1)
df2 = load_and_process_data(file_path2)

# Calculate and plot the rolling correlation
plot_rolling_correlation(df1, df2)

