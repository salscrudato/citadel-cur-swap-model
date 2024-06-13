import pandas as pd
import numpy as np
import time
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

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

def compare_portfolios(directory_path):
    # Get a list of all CSV files in the directory
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]

    all_stats = []
    all_files = []

    for file_path in file_paths:
        df = load_and_process_data(file_path)

        # Calculate statistics for the portfolio for gross and net returns
        stats_gross = calculate_statistics(df, return_type='gross')
        stats_net = calculate_statistics(df, return_type='net')

        # Combine the statistics into a single dictionary for the portfolio
        stats = {**stats_gross, **stats_net}
        
        # Extract the number at the end of the file name for indexing
        file_number = re.search(r'\d+$', os.path.splitext(os.path.basename(file_path))[0])
        if file_number:
            index_value = int(file_number.group())
        else:
            index_value = os.path.basename(file_path)  # Use file name if no number found
        
        all_files.append(index_value)
        stats['File'] = os.path.basename(file_path)
        
        all_stats.append((index_value, stats))

    # Create a DataFrame to display the statistics for all portfolios
    comparison_df = pd.DataFrame([stats for index, stats in all_stats], index=[index for index, stats in all_stats])

    return comparison_df

# Define the directory path for the portfolios
directory_path = '/Users/salscrudato/Finance Tests/CNR/'

# Compare the portfolios
comparison_df = compare_portfolios(directory_path)

# Specify the columns to include
columns_to_include = ['Sharpe Ratio (gross)', 'Average Trades per Day (gross)', 'Sharpe Ratio (net)']

# Create a new DataFrame with only the specified columns
aaa_comparison_summary = comparison_df[columns_to_include]

# Print the comparison DataFrame
print(comparison_df)

# Plot the 'Sharpe Ratio (net)' and 'Sharpe Ratio (gross)' using Seaborn
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

ax = sns.lineplot(x=comparison_df.index, y=comparison_df['Sharpe Ratio (net)'], label='Sharpe Ratio (net)', marker='o')
ax = sns.lineplot(x=comparison_df.index, y=comparison_df['Sharpe Ratio (gross)'], label='Sharpe Ratio (gross)', marker='o')

ax.set_title('Sharpe Ratios (net and gross) for 3M2Y Momentum', fontsize=14)
ax.set_xlabel('Half Life', fontsize=12)
ax.set_ylabel('Sharpe Ratio', fontsize=12)

# Set x-axis ticks to show labels at increments of 10
ax.set_xticks(comparison_df.index[::20])
ax.set_xticklabels(comparison_df.index[::20], rotation=90)

plt.legend()
plt.tight_layout()
plt.show()

# Make the plot show up in a separate window
plt.show(block=True)
