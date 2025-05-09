# import pandas as pd
# import numpy as np
# import time
# import os, re
# import warnings
# import plotly.graph_objects as go
# import plotly.offline as pyo

# test_type = "CNR"
# test_sub_type = "Z Score"
# method = "EWMA"
# smoothing = "5"

# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually")

# # Define the function to load and process data
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

# def compare_portfolios(file_paths):
#     all_stats = []
#     all_files = []

#     for file_path in file_paths:
#         df = load_and_process_data(file_path)
#         stats_net = calculate_statistics(df, return_type='net')
#         file_number = re.search(r'\d+$', os.path.splitext(os.path.basename(file_path))[0])
#         if file_number:
#             index_value = int(file_number.group())
#         else:
#             index_value = os.path.basename(file_path)
#         all_files.append(index_value)
#         stats_net['File'] = os.path.basename(file_path)
#         all_stats.append((index_value, stats_net))

#     comparison_df = pd.DataFrame([stats for index, stats in all_stats], index=[index for index, stats in all_stats])
#     numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
#     comparison_df = comparison_df[numeric_cols]

#     return comparison_df

# # Initialize the plot
# fig = go.Figure()

# # Define the directory path
# directory_path = '/Users/salscrudato/Finance Tests/Outputs/CNR/ZScore/'

# # Get a list of unique z_score values from the file names
# file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]
# z_scores = sorted(set(re.search(r'zscore(\d+)', file).group(1) for file in os.listdir(directory_path) if re.search(r'zscore(\d+)', file)))

# for z_score in z_scores:
#     # Filter files for the current z_score
#     z_score_files = [file for file in file_paths if f'zscore{z_score}' in file]
    
#     # Create a comparison dataframe for the current z_score
#     comparison_df = compare_portfolios(z_score_files)
#     comparison_df = comparison_df.groupby(comparison_df.index).mean()

#     # Add trace for the net Sharpe ratio of each z_score value
#     fig.add_trace(go.Scatter(
#         x=comparison_df.index,
#         y=comparison_df['Sharpe Ratio (net)'],
#         mode='lines+markers',
#         name=f'Z Score {z_score}',
#         line=dict(shape='linear', width=1),
#         marker=dict(symbol='circle', size=2)
#     ))

# # Customize the layout
# fig.update_layout(
#     title=f'Net Sharpe Ratios for {test_type} | Various Z Scores',
#     xaxis_title='Half Life',
#     yaxis_title='Net Sharpe Ratio',
#     xaxis=dict(
#         tickmode='array',
#         tickvals=list(comparison_df.index)[::25],
#         ticktext=[str(i) for i in list(comparison_df.index)[::25]]
#     ),
#     legend=dict(x=0.01, y=0.99),
#     template='seaborn',
#     width=1400,
#     height=800,
#     hovermode='x'
# )

# # Construct the filename using the variables
# filename = f'Net_Sharpe_Ratios_{test_type}_Various_Z_Scores.html'

# # Save the plot with the constructed filename
# pyo.plot(fig, filename=filename)


import pandas as pd
import numpy as np
import time
import os, re
import warnings
import plotly.graph_objects as go
import plotly.offline as pyo

test_type = "CNR"
test_sub_type = "Z Score"
method = "EWMA"
smoothing = "5"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format, so each element will be parsed individually")

# Define the function to load and process data
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

def compare_portfolios(file_paths):
    all_stats = []
    all_files = []

    for file_path in file_paths:
        df = load_and_process_data(file_path)
        stats_net = calculate_statistics(df, return_type='net')
        file_number = re.search(r'\d+$', os.path.splitext(os.path.basename(file_path))[0])
        if file_number:
            index_value = int(file_number.group())
        else:
            index_value = os.path.basename(file_path)
        all_files.append(index_value)
        stats_net['File'] = os.path.basename(file_path)
        all_stats.append((index_value, stats_net))

    comparison_df = pd.DataFrame([stats for index, stats in all_stats], index=[index for index, stats in all_stats])
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
    comparison_df = comparison_df[numeric_cols]

    return comparison_df

# Initialize the plot
fig_sharpe = go.Figure()
fig_drawdown = go.Figure()

# Define the directory path
directory_path = '/Users/salscrudato/Finance Tests/Outputs/CNR/ZScore/'

# Get a list of unique z_score values from the file names
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]
z_scores = sorted(set(re.search(r'zscore(\d+)', file).group(1) for file in os.listdir(directory_path) if re.search(r'zscore(\d+)', file)))

for z_score in z_scores:
    # Filter files for the current z_score
    z_score_files = [file for file in file_paths if f'zscore{z_score}' in file]
    
    # Create a comparison dataframe for the current z_score
    comparison_df = compare_portfolios(z_score_files)
    comparison_df = comparison_df.groupby(comparison_df.index).mean()

    # Add trace for the net Sharpe ratio of each z_score value
    fig_sharpe.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Sharpe Ratio (net)'],
        mode='lines+markers',
        name=f'Z Score {z_score}',
        line=dict(shape='linear', width=1),
        marker=dict(symbol='circle', size=2)
    ))
    
    # Add trace for the max drawdown percentage of each z_score value
    fig_drawdown.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Max Drawdown % of Average Annual Return (net)'],
        mode='lines+markers',
        name=f'Z Score {z_score}',
        line=dict(shape='linear', width=1),
        marker=dict(symbol='circle', size=2)
    ))

# Customize the layout for the Sharpe ratio plot
fig_sharpe.update_layout(
    title=f'Net Sharpe Ratios for {test_type} | Various Z Scores',
    xaxis_title='Half Life',
    yaxis_title='Net Sharpe Ratio',
    xaxis=dict(
        tickmode='array',
        tickvals=list(comparison_df.index)[::25],
        ticktext=[str(i) for i in list(comparison_df.index)[::25]]
    ),
    legend=dict(x=0.01, y=0.99),
    template='seaborn',
    width=1400,
    height=800,
    hovermode='x'
)

# Customize the layout for the Max Drawdown Percentage plot
fig_drawdown.update_layout(
    title=f'Max Drawdown Percentage for {test_type} | Various Z Scores',
    xaxis_title='Half Life',
    yaxis_title='Max Drawdown % of Average Annual Return',
    xaxis=dict(
        tickmode='array',
        tickvals=list(comparison_df.index)[::25],
        ticktext=[str(i) for i in list(comparison_df.index)[::25]]
    ),
    legend=dict(x=0.01, y=0.99),
    template='seaborn',
    width=1400,
    height=800,
    hovermode='x'
)

# Construct the filename for the Sharpe ratio plot
filename_sharpe = f'Net_Sharpe_Ratios_{test_type}_Various_Z_Scores.html'

# Save the Sharpe ratio plot with the constructed filename
pyo.plot(fig_sharpe, filename=filename_sharpe)

# Construct the filename for the Max Drawdown Percentage plot
filename_drawdown = f'Max_Drawdown_Percentage_{test_type}_Various_Z_Scores.html'

# Save the Max Drawdown Percentage plot with the constructed filename
pyo.plot(fig_drawdown, filename=filename_drawdown)