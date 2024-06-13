# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from datetime import datetime, timedelta
# import cvxpy as cp
# import time
# import os, re
# import warnings
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.offline as pyo

# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually")

# # Define the directory path for the portfolios
# directory_path = '/Users/salscrudato/Finance Tests/3M2Y/'
# file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]

# def load_and_process_data(file_path):
#     df = pd.read_csv(file_path, index_col=[0, 1], parse_dates=True)

#     def safe_merge(df, currency_df, col_name, retries=3):
#         for i in range(retries):
#             try:
#                 if col_name in df.columns:
#                     df.drop(columns=[col_name], inplace=True)
#                 df = df.merge(currency_df[[col_name]], how='left', left_index=True, right_index=True)
#                 return df
#             except Exception as e:
#                 print(f"Error during merge: {e}. Retrying {i + 1}/{retries}...")
#                 time.sleep(1)
#         raise Exception(f"Failed to merge after {retries} retries.")

#     for currency in df.index.get_level_values('Currency').unique():
#         currency_df = df.xs(currency, level='Currency').copy()
#         gross_col_name = f'{currency}_CUM_GROSS_RETURN'
#         currency_df[gross_col_name] = currency_df['1D Gross Return'].cumsum()
#         net_col_name = f'{currency}_CUM_NET_RETURN'
#         currency_df[net_col_name] = currency_df['1D Net Return'].cumsum()
#         df = safe_merge(df, currency_df, gross_col_name)
#         df = safe_merge(df, currency_df, net_col_name)

#     cumulative_gross_return_cols = [f'{currency}_CUM_GROSS_RETURN' for currency in df.index.get_level_values('Currency').unique()]
#     cumulative_net_return_cols = [f'{currency}_CUM_NET_RETURN' for currency in df.index.get_level_values('Currency').unique()]
#     df['Portfolio_CUM_GROSS_RETURN'] = df[cumulative_gross_return_cols].sum(axis=1)
#     df['Portfolio_CUM_NET_RETURN'] = df[cumulative_net_return_cols].sum(axis=1)

#     return df

# def consolidate_returns(df, return_col):
#     consolidated_df = df.groupby(df.index.get_level_values('Date')).first()
#     consolidated_df.index = pd.to_datetime(consolidated_df.index)
#     consolidated_df = consolidated_df.sort_index()
#     consolidated_df[f'{return_col}_5D_Return'] = consolidated_df[return_col].diff(5)
#     return consolidated_df

# def calculate_statistics(df, return_type='gross'):
#     return_col = 'Portfolio_CUM_GROSS_RETURN' if return_type == 'gross' else 'Portfolio_CUM_NET_RETURN'
#     consolidated_df = consolidate_returns(df, return_col)
#     df_filtered = df[df.index.get_level_values('Date') >= pd.Timestamp('2013-01-01')]
#     df_non_zero = df_filtered[df_filtered['1-Day Change'] != 0]
#     daily_counts = df_non_zero.groupby(df_non_zero.index.get_level_values('Date'))['1-Day Change'].count()
#     average_trades_per_day = daily_counts.mean()
#     returns_5d = consolidated_df[f'{return_col}_5D_Return'].dropna()
#     mean_return = returns_5d.mean()
#     std_dev_return = returns_5d.std()
#     sharpe_ratio_5d = (mean_return / std_dev_return) * np.sqrt(252 / 5)
#     total_years = (consolidated_df.index[-1] - consolidated_df.index[0]).days / 365.25
#     most_recent_cum_return = consolidated_df[return_col].iloc[-1]
#     average_annual_nominal_change = most_recent_cum_return / total_years
#     vol_5d_annualized = std_dev_return * np.sqrt(252 / 5)

#     def calculate_max_drawdown(series):
#         peak = series.cummax()
#         drawdown = peak - series
#         max_drawdown = drawdown.max()
#         return max_drawdown

#     max_drawdown = calculate_max_drawdown(consolidated_df[return_col])
#     max_drawdown_percentage = (max_drawdown / average_annual_nominal_change) * 100

#     def calculate_max_months_underwater(series):
#         series = series.sort_index()
#         peak = series.cummax()
#         underwater = series < peak
#         groups = (underwater != underwater.shift()).cumsum()
#         df_underwater = pd.DataFrame({'series': series, 'peak': peak, 'underwater': underwater, 'group': groups})
#         df_underwater['month'] = df_underwater.index.to_period('M')
#         underwater_duration = df_underwater[df_underwater['underwater']].groupby('group')['month'].nunique()
#         max_months_underwater = underwater_duration.max()
#         return max_months_underwater

#     max_months_underwater = calculate_max_months_underwater(consolidated_df[return_col])

#     return {
#         f"Average Trades per Day ({return_type})": average_trades_per_day,
#         f"Sharpe Ratio ({return_type})": sharpe_ratio_5d,
#         f"Average Annual Return ({return_type})": average_annual_nominal_change,
#         f"Annualized Volatility of 5-Day Changes ({return_type})": vol_5d_annualized,
#         f"Maximum Drawdown ({return_type})": max_drawdown,
#         f"Max Drawdown % of Average Annual Return ({return_type})": max_drawdown_percentage,
#         f"Max Months Underwater ({return_type})": max_months_underwater
#     }

# def compare_portfolios(directory_path):
#     # Get a list of all CSV files in the directory
#     file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]

#     all_stats = []
#     all_files = []

#     for file_path in file_paths:
#         df = load_and_process_data(file_path)

#         # Calculate statistics for the portfolio for gross and net returns
#         stats_gross = calculate_statistics(df, return_type='gross')
#         stats_net = calculate_statistics(df, return_type='net')

#         # Combine the statistics into a single dictionary for the portfolio
#         stats = {**stats_gross, **stats_net}
        
#         # Extract the number at the end of the file name for indexing
#         file_number = re.search(r'\d+$', os.path.splitext(os.path.basename(file_path))[0])
#         if file_number:
#             index_value = int(file_number.group())
#         else:
#             index_value = os.path.basename(file_path)  # Use file name if no number found
        
#         all_files.append(index_value)
#         stats['File'] = os.path.basename(file_path)
        
#         all_stats.append((index_value, stats))

#     # Create a DataFrame to display the statistics for all portfolios
#     comparison_df = pd.DataFrame([stats for index, stats in all_stats], index=[index for index, stats in all_stats])

#     return comparison_df

# comparison_df = compare_portfolios(directory_path)
# # Specify the columns to include
# columns_to_include = ['Sharpe Ratio (gross)', 'Average Trades per Day (gross)', 'Sharpe Ratio (net)']

# # Create a new DataFrame with only the specified columns
# aaa_comparison_summary = comparison_df[columns_to_include]

# # Load and process data for each portfolio
# dfs = [load_and_process_data(file_path) for file_path in file_paths]

# # Extract the cumulative net return for each portfolio
# cumulative_net_returns = [df[['Portfolio_CUM_NET_RETURN']] for df in dfs]

# # Concatenate the cumulative net returns into a single DataFrame
# combined_df = pd.concat(cumulative_net_returns, axis=1)

# # Rename the columns to reflect the portfolio names
# combined_df.columns = [f'Portfolio_{i+1}_CUM_NET_RETURN' for i in range(len(dfs))]

# # Ensure the DataFrame contains data from 2013 onwards
# combined_df = combined_df[combined_df.index.get_level_values('Date') >= pd.Timestamp('2013-01-01')]

# # Reset the index to have the date as a column
# combined_df.reset_index(inplace=True)

# # Display the column names to find the correct date column
# print(combined_df.columns)

# # Assuming the first column is the date column
# date_column = combined_df.columns[0]

# # Set the date column as the index
# combined_df.set_index(date_column, inplace=True)

# plt.figure(figsize=(14, 8))
# sns.set(style="whitegrid")

# ax = sns.lineplot(x=comparison_df.index, y=comparison_df['Sharpe Ratio (net)'], label='Sharpe Ratio (net)', marker='o')
# ax = sns.lineplot(x=comparison_df.index, y=comparison_df['Sharpe Ratio (gross)'], label='Sharpe Ratio (gross)', marker='o')

# ax.set_title('Sharpe Ratios (net and gross) for 3M2Y Momentum', fontsize=14)
# ax.set_xlabel('Half Life', fontsize=12)
# ax.set_ylabel('Sharpe Ratio', fontsize=12)

# # Set x-axis ticks to show labels at increments of 10
# ax.set_xticks(range(min(comparison_df.index), max(comparison_df.index) + 1, 10))
# ax.set_xticklabels(range(min(comparison_df.index), max(comparison_df.index) + 1, 10), rotation=90)

# plt.legend()
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cvxpy as cp
import time
import os, re
import warnings
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually")

# Define the directory path for the portfolios
directory_path = '/Users/salscrudato/Finance Tests/CNR/'

file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]

def load_and_process_data(file_path):
    df = pd.read_csv(file_path, index_col=[0, 1], parse_dates=True)

    def safe_merge(df, currency_df, col_name, retries=3):
        for i in range(retries):
            try:
                if col_name in df.columns:
                    df.drop(columns=[col_name], inplace=True)
                df = df.merge(currency_df[[col_name]], how='left', left_index=True, right_index=True)
                return df
            except Exception as e:
                print(f"Error during merge: {e}. Retrying {i + 1}/{retries}...")
                time.sleep(1)
        raise Exception(f"Failed to merge after {retries} retries.")

    for currency in df.index.get_level_values('Currency').unique():
        currency_df = df.xs(currency, level='Currency').copy()
        gross_col_name = f'{currency}_CUM_GROSS_RETURN'
        currency_df[gross_col_name] = currency_df['1D Gross Return'].cumsum()
        net_col_name = f'{currency}_CUM_NET_RETURN'
        currency_df[net_col_name] = currency_df['1D Net Return'].cumsum()
        df = safe_merge(df, currency_df, gross_col_name)
        df = safe_merge(df, currency_df, net_col_name)

    cumulative_gross_return_cols = [f'{currency}_CUM_GROSS_RETURN' for currency in df.index.get_level_values('Currency').unique()]
    cumulative_net_return_cols = [f'{currency}_CUM_NET_RETURN' for currency in df.index.get_level_values('Currency').unique()]
    df['Portfolio_CUM_GROSS_RETURN'] = df[cumulative_gross_return_cols].sum(axis=1)
    df['Portfolio_CUM_NET_RETURN'] = df[cumulative_net_return_cols].sum(axis=1)

    return df

def consolidate_returns(df, return_col):
    consolidated_df = df.groupby(df.index.get_level_values('Date')).first()
    consolidated_df.index = pd.to_datetime(consolidated_df.index)
    consolidated_df = consolidated_df.sort_index()
    consolidated_df[f'{return_col}_5D_Return'] = consolidated_df[return_col].diff(5)
    return consolidated_df

def calculate_statistics(df, return_type='gross'):
    return_col = 'Portfolio_CUM_GROSS_RETURN' if return_type == 'gross' else 'Portfolio_CUM_NET_RETURN'
    consolidated_df = consolidate_returns(df, return_col)
    df_filtered = df[df.index.get_level_values('Date') >= pd.Timestamp('2013-01-01')]
    df_non_zero = df_filtered[df_filtered['1-Day Change'] != 0]
    daily_counts = df_non_zero.groupby(df_non_zero.index.get_level_values('Date'))['1-Day Change'].count()
    average_trades_per_day = daily_counts.mean()
    returns_5d = consolidated_df[f'{return_col}_5D_Return'].dropna()
    mean_return = returns_5d.mean()
    std_dev_return = returns_5d.std()
    sharpe_ratio_5d = (mean_return / std_dev_return) * np.sqrt(252 / 5)
    total_years = (consolidated_df.index[-1] - consolidated_df.index[0]).days / 365.25
    most_recent_cum_return = consolidated_df[return_col].iloc[-1]
    average_annual_nominal_change = most_recent_cum_return / total_years
    vol_5d_annualized = std_dev_return * np.sqrt(252 / 5)

    def calculate_max_drawdown(series):
        peak = series.cummax()
        drawdown = peak - series
        max_drawdown = drawdown.max()
        return max_drawdown

    max_drawdown = calculate_max_drawdown(consolidated_df[return_col])
    max_drawdown_percentage = (max_drawdown / average_annual_nominal_change) * 100

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

comparison_df = compare_portfolios(directory_path)
# Specify the columns to include
columns_to_include = ['Sharpe Ratio (gross)', 'Average Trades per Day (gross)', 'Sharpe Ratio (net)']

# Create a new DataFrame with only the specified columns
aaa_comparison_summary = comparison_df[columns_to_include]

# Load and process data for each portfolio
dfs = [load_and_process_data(file_path) for file_path in file_paths]

# Extract the cumulative net return for each portfolio
cumulative_net_returns = [df[['Portfolio_CUM_NET_RETURN']] for df in dfs]

# Concatenate the cumulative net returns into a single DataFrame
combined_df = pd.concat(cumulative_net_returns, axis=1)

# Rename the columns to reflect the portfolio names
combined_df.columns = [f'Portfolio_{i+1}_CUM_NET_RETURN' for i in range(len(dfs))]

# Ensure the DataFrame contains data from 2013 onwards
combined_df = combined_df[combined_df.index.get_level_values('Date') >= pd.Timestamp('2013-01-01')]

# Ensure the data is correctly formatted without duplicates
comparison_df = comparison_df.groupby(comparison_df.index).mean()

# Create the interactive plot
fig = go.Figure()

# Add Sharpe Ratio (net) trace
fig.add_trace(go.Scatter(
    x=comparison_df.index,
    y=comparison_df['Sharpe Ratio (net)'],
    mode='lines+markers',
    name='Sharpe Ratio (net)',
    line=dict(shape='linear', width=1),
    marker=dict(symbol='circle', size=2)
))

# Add Sharpe Ratio (gross) trace
fig.add_trace(go.Scatter(
    x=comparison_df.index,
    y=comparison_df['Sharpe Ratio (gross)'],
    mode='lines+markers',
    name='Sharpe Ratio (gross)',
    line=dict(shape='linear', width=1),
    marker=dict(symbol='circle', size=2)
))

# Customize the layout
fig.update_layout(
    title='Sharpe Ratios (net and gross) for 3ToT',
    xaxis_title='Half Life',
    # yaxis_title='Sharpe Ratio',
    xaxis=dict(
        tickmode='array',
        tickvals=list(comparison_df.index)[::25],  # Show fewer ticks to reduce clutter
        ticktext=[str(i) for i in list(comparison_df.index)[::25]]
    ),
    legend=dict(x=0.01, y=0.99),
    template='seaborn',
    width=1400,
    height=800,
    hovermode='x'
)

# Show the plot in the default web browser
pyo.plot(fig)


