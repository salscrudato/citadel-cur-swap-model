import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cvxpy as cp
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import sys
import argparse

# Ignore warnings
warnings.filterwarnings('ignore')

# Set base directory and update system path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def get_bbg_ticker_mapping(path):
    """
    Reads a Bloomberg ticker mapping file and returns two dictionaries:
    1. The original mapping of tickers to currency and suffix.
    2. A mapping of tickers to concatenated currency and suffix.

    Args:
    path (str): Path to the Excel file containing the ticker mapping.

    Returns:
    tuple: Two dictionaries - 
           1. Original ticker mapping where the key is the ticker and the value is a list containing the currency and suffix.
           2. Concatenated ticker mapping where the key is the ticker and the value is the concatenated currency and suffix.
    """
    df_bbg = pd.read_excel(path, index_col=0)
    original_mapping = {key: [row['Currency'], row['Suffix']] for key, row in df_bbg.iterrows()}
    concatenated_mapping = {key: value[0] + value[1] for key, value in original_mapping.items()}
    
    return original_mapping, concatenated_mapping

def process_and_convert_header_row(df, hr, new_hr, conversion_dict):
    """
    Processes and converts the header row of a DataFrame.

    Args:
    df (pd.DataFrame): DataFrame to be processed.
    hr (int): Index of the header row to be converted.
    new_hr (int): Index for the new header row after conversion.
    conversion_dict (dict): Dictionary for converting header names.

    Returns:
    pd.DataFrame: Processed DataFrame with converted headers and set index.
    """
    # Convert header row using conversion dictionary
    df.iloc[hr] = df.iloc[hr].str.split().str[0].map(conversion_dict)
    df.iloc[hr, 0] = "Dates"
    df.columns = df.iloc[hr]

    # Select relevant rows and reset index
    df = df.iloc[new_hr + 1:].copy()

    # Set 'Dates' column as the index and convert to datetime
    df.set_index(pd.to_datetime(df.pop("Dates")), inplace=True)

    # Sort index, fill missing values, and convert to numeric
    df.sort_index(inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def group_columns_by_suffix(df, bbg_tickers):
    """
    Groups columns of a DataFrame by their suffixes.

    Args:
    df (pd.DataFrame): DataFrame with columns to be grouped.
    bbg_tickers (dict): Dictionary with Bloomberg tickers as keys and lists containing currency and suffix as values.

    Returns:
    dict: Dictionary where keys are suffixes and values are lists of column names ending with those suffixes.
    """
    suffixes = {value[1] for value in bbg_tickers.values()}
    grouped_columns = {suffix: [col for col in df.columns if col.endswith(suffix)] for suffix in suffixes}

    # Change the key from '_USD' to '_FX' if it exists
    if '_USD' in grouped_columns:
        grouped_columns['_FX'] = grouped_columns.pop('_USD')

    return grouped_columns

def calculate_metrics(df, col, cur, prefix, hl_smooth, hl_lookback, z_score_lower, z_score_upper):
    """
    Calculate metrics for a given DataFrame and parameters.

    Args:
    df (pd.DataFrame): Input DataFrame.
    col (str): Column name for the metric.
    cur (str): Currency.
    prefix (str): Prefix for the new columns.
    hl_smooth (int): Half-life for smoothing.
    hl_lookback (int): Half-life for lookback.
    z_score_lower (int): Lower limit for z-score clipping.
    z_score_upper (int): Upper limit for z-score clipping.

    Returns:
    pd.DataFrame: DataFrame with calculated metrics.
    """
    assert col in df.columns, f"{col} not in dataframe columns"

    smoothed = df[col].ewm(halflife=hl_smooth).mean()
    mean = smoothed.ewm(halflife=hl_lookback).mean()
    std = smoothed.ewm(halflife=hl_lookback).std()
    z_score = (smoothed - mean) / std

    df[f'{cur}_{prefix}_SMOOTH'] = smoothed
    df[f'{cur}_{prefix}_MEAN'] = mean
    df[f'{cur}_{prefix}_STD'] = std
    df[f'{cur}_{prefix}_Z_SCORE'] = z_score
    df[f'{cur}_{prefix}_Z_SCORE_CAPPED'] = np.clip(z_score, z_score_lower, z_score_upper)

    return df

def calculate_metrics_for_all_currencies(df, grouped_columns, prefix, hl_lookback, hl_smooth, z_score_lower, z_score_upper):
    """
    Calculate metrics for all currencies.
    """
    for col in grouped_columns[f'_{prefix}']:
        currency = col.split('_')[0]
        df = calculate_metrics(df, col, currency, prefix, hl_smooth, hl_lookback, z_score_lower, z_score_upper)
    return df

def calculate_cnr_metrics(df, col_3m2y, col_2y, cur, hl_smooth, hl_lookback, z_score_lower, z_score_upper):
    """
    Calculate CNR metrics for a given DataFrame and parameters.
    """
    assert col_3m2y in df.columns, f"{col_3m2y} not in dataframe columns"
    assert col_2y in df.columns, f"{col_2y} not in dataframe columns"

    df[f'{cur}_CNR'] = (df[col_3m2y] - df[col_2y]) * 4 / 252
    return calculate_metrics(df, f'{cur}_CNR', cur, 'CNR', hl_smooth, hl_lookback, z_score_lower, z_score_upper)

def calculate_cnr_metrics_for_all_currencies(df, grouped_columns, hl_lookback, hl_smooth, z_score_lower, z_score_upper):
    """
    Calculate CNR metrics for all currencies.
    """
    for currency in set(col.split('_')[0] for col in grouped_columns['_3M2Y']):
        col_3m2y = next((col for col in grouped_columns['_3M2Y'] if col.startswith(currency)), None)
        col_2y = next((col for col in grouped_columns['_2Y'] if col.startswith(currency)), None)
        if col_3m2y and col_2y:
            df = calculate_cnr_metrics(df, col_3m2y, col_2y, currency, hl_smooth, hl_lookback, z_score_lower, z_score_upper)
    return df

def calculate_msci_metrics(df, cols_msci, cols_fx, hl_lookback, hl_smooth, z_score_lower, z_score_upper):
    """
    Calculate MSCI metrics.
    """
    fx_dict = {col.split('_')[0]: col for col in cols_fx}

    for col in cols_msci:
        currency = col.split('_')[0]
        if currency in fx_dict:
            col2 = fx_dict[currency]
            msci_lccy = f'{currency}_MSCI_LCCY'
            smoothed = f'{currency}_MSCI_LCCY_SMOOTHED'
            mean = f'{currency}_MSCI_MEAN'
            std = f'{currency}_MSCI_STD'
            z_score = f'{currency}_MSCI_Z_SCORE'
            z_score_capped = f'{currency}_MSCI_Z_SCORE_CAPPED'

            df[msci_lccy] = df[col] * df[col2]
            df[smoothed] = df[msci_lccy].ewm(halflife=hl_smooth).mean()
            df[mean] = df[smoothed].ewm(halflife=hl_lookback).mean()
            df[std] = df[smoothed].ewm(halflife=hl_lookback).std()
            df[z_score] = (df[smoothed] - df[mean]) / df[std]
            df[z_score_capped] = np.clip(df[z_score], z_score_lower, z_score_upper)
    return df

# Refactored IV Metrics Calculation
def calculate_all_metrics(df, cols_iv, hl_smooth_iv, hl_mean_iv, z_score_lower, z_score_upper):
    """
    Calculate all IV metrics using generalized function.

    Args:
    df (pd.DataFrame): Input DataFrame.
    cols_iv (list): List of IV columns.
    hl_smooth_iv (int): Half-life for IV smoothing.
    hl_mean_iv (int): Half-life for IV mean calculation.
    z_score_lower (int): Lower limit for z-score clipping.
    z_score_upper (int): Upper limit for z-score clipping.

    Returns:
    pd.DataFrame: DataFrame with calculated metrics.
    """
    for col in cols_iv:
        currency = col.split('_')[0]
        df = calculate_metrics(df, col, currency, 'IV', hl_smooth_iv, hl_mean_iv, z_score_lower, z_score_upper)
    return df

def calculate_returns(df, cols_3m2y, rolling_window=5, hl_lookback=63):
    """
    Calculate daily price changes, daily total returns, cumulative returns, and rolling statistics for a given DataFrame and parameters.

    Args:
    df (pd.DataFrame): Input DataFrame.
    cols_3m2y (list): List of columns to calculate returns for.
    rolling_window (int): Window size for rolling calculations. Default is 5.
    hl_lookback (int): Half-life for exponential moving standard deviation. Default is 63.

    Returns:
    pd.DataFrame, pd.DataFrame: DataFrame with calculated returns and rolling statistics, DataFrame with rolling total returns.
    """
    df_rolling_5d_tr = pd.DataFrame(index=df.index)
    
    for col in cols_3m2y:
        cur = col.split('_')[0]
        daily_price_change = f"{cur}_1D_CHG"
        daily_total_return = f"{cur}_1D_TR"
        rolling_total_return = f"{cur}_{rolling_window}D_TR"
        rolling_tr_std = f"{cur}_TR_STD"
        cum_total_return = f"{cur}_CUM_TR"

        # Calculate daily price change if not already calculated
        if daily_price_change not in df.columns:
            df[daily_price_change] = df[col].diff()

        # Calculate daily total return if not already calculated
        if daily_total_return not in df.columns:
            df[daily_total_return] = df[f"{cur}_CNR"] - df[daily_price_change]

        # Calculate rolling total return
        df[rolling_total_return] = df[daily_total_return].rolling(window=rolling_window).sum()

        # Calculate rolling standard deviation of total return
        df[rolling_tr_std] = df[rolling_total_return].ewm(halflife=hl_lookback).std() * np.sqrt(252 / rolling_window) * 100

        # Calculate cumulative total return
        df[cum_total_return] = df[daily_total_return].cumsum()

        # Store rolling total return in separate DataFrame
        df_rolling_5d_tr[rolling_total_return] = df[rolling_total_return]

    return df, df_rolling_5d_tr

def calculate_expected_returns(df, currencies, date, metrics_weights):
    expected_returns = []
    rolling_5d_tr_std_values = []
    for cur in currencies:
        cur = cur.split('_')[0]
        expected_return = sum(weight * df[f"{cur}_{metric}_Z_SCORE_CAPPED"].loc[date] for metric, weight in metrics_weights.items())
        expected_returns.append(expected_return)
        
        if cur + "_TR_STD" in df.columns:
            rolling_5d_tr_std_values.append(df[cur + "_TR_STD"].loc[date])
        else:
            rolling_5d_tr_std_values.append(np.nan)  # Append NaN if the column doesn't exist

    return expected_returns, rolling_5d_tr_std_values

def quadratic_utility_optimizer(expected_returns, corr_matrix, risk_aversion, portfolio_volatility):
    """
    Optimize portfolio weights using quadratic utility maximization.

    Args:
    expected_returns (list): Expected returns for each asset.
    corr_matrix (ndarray): Covariance matrix of asset returns.
    risk_aversion (float): Risk aversion parameter.
    portfolio_volatility (float): Target portfolio volatility.

    Returns:
    ndarray: Optimized portfolio weights.
    """
    n = len(expected_returns)
    weights = cp.Variable(n)
    portfolio_return = weights @ expected_returns
    portfolio_variance = cp.quad_form(weights, corr_matrix)  # Quadratic form for variance
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    constraints = [
        weights <= portfolio_volatility,  # Constraint for maximum optimal risk weight based on pre-scaling portfolio volatility
    ]
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve()
    except cp.SolverError as e:
        print(f"Optimization failed: {e}")
        return None
    
    return weights.value

def print_results(df):
    """
    Print the first and last values of various Z_SCORE_CAPPED columns for the 'CLP' currency.

    Args:
    df (pd.DataFrame): Input DataFrame containing the calculated metrics.
    """
    columns_to_print = [
        'CLP_CNR_Z_SCORE_CAPPED', 'CLP_ToT_Z_SCORE_CAPPED',
        'CLP_3M2Y_Z_SCORE_CAPPED', 'CLP_FX_Z_SCORE_CAPPED',
        'CLP_MSCI_Z_SCORE_CAPPED', 'CLP_IV_Z_SCORE_CAPPED'
    ]

    for col in columns_to_print:
        print(f"{col} first value: {df[col].iloc[0]}")
        print(f"{col} last value: {df[col].iloc[-1]}")


#-------------------- Start Coding Logic --------------------
    
# Define file paths and header rows for Bloomberg data and codes
BBG_FP = os.path.join(BASE_DIR, 'Front End Data.csv')
BBG_CODES_FP = os.path.join(BASE_DIR, 'bbg_codes.xlsx')
BBG_HEADER_ROW = 2
BBG_NEW_HEADER_ROW = BBG_HEADER_ROW + 2

# Read the raw Bloomberg data
df_raw = pd.read_csv(BBG_FP)

# Create Bloomberg ticker dictionary and mapping
bbg_tickers, bbg_ticker_dict = get_bbg_ticker_mapping(BBG_CODES_FP)

# Preprocess the raw Bloomberg data to convert and set the header row
df = process_and_convert_header_row(df_raw, BBG_HEADER_ROW, BBG_NEW_HEADER_ROW, bbg_ticker_dict)

# Define unique suffixes from Bloomberg tickers
suffixes = list({value[1] for value in bbg_tickers.values()})

# Group DataFrame columns by their suffixes
grouped_columns = group_columns_by_suffix(df, bbg_tickers)

"""

These are the variables to change for testing.

"""
test_type = 'CNR'
test_sub_type = 'ZScore'
min_trade_size = 2500
target_volatility = 15000.00
risk_aversion = 1.0

z_score_tmp = 5
z_score_upper = z_score_tmp
z_score_lower = z_score_upper * -1

z_score_upper_3m2y = 4
z_score_lower_3m2y = -4

hl_smooth = 5
default_hl_start = 5
default_hl_end = 505
default_hl_skip = 5

# Define constants as variables
rolling_window = 5
hl_lookback_returns = 63
hl_lookback_cov = 126

# Define initial metrics weights based on test type
metrics_weights = {
    "3M2Y": -1,
    "CNR": 1,
    "MSCI": -1,
    "FX": -1,
    "TOT": 1,
    "IV": -1
}

# Override metrics weights based on test type
if test_type in metrics_weights:
    metrics_weights[test_type] = 1 if metrics_weights[test_type] > 0 else -1
    updated_metrics_weights = {test_type: metrics_weights[test_type]}
else:
    updated_metrics_weights = {}
metrics_weights = updated_metrics_weights

# Define the temporary half-life lookback variable
tmp_hl = f"hl_lookback_{test_type.lower()}"

# Setup argument parser
parser = argparse.ArgumentParser(description='Run signal test with different half-life values')
parser.add_argument('--hl_start', type=int, default=default_hl_start, help='Value to start')
parser.add_argument('--hl_end', type=int, default=default_hl_end, help='Value to end')
parser.add_argument('--hl_skip', type=int, default=default_hl_skip, help='Step size for the range')

# Parse arguments
args = parser.parse_args()

# Override hl_start, hl_end, and hl_skip values with arguments if provided
hl_start = args.hl_start
hl_end = args.hl_end
hl_skip = args.hl_skip

# Define transaction costs for each currency
transaction_costs = {
    'CLP': 2, 
    'CNY': 1, 
    'COP': 3, 
    'CZK': 1.5, 
    'HUF': 3,
    'INR': 2, 
    'ILS': 2, 
    'KRW': 1.5, 
    'MXN': 1, 
    'PLN': 1.5, 
    'ZAR': 1.5
}

"""

These are the variables to change for testing.

"""

# Loop through the range of half-life values
for tmp_hl in range(hl_start, hl_end, hl_skip):
    print(f"Running for {test_type}, {test_sub_type}, Z Score = {z_score_upper} Half Life = {tmp_hl}")

    # Define half-life lookback for different metrics
    hl_lookback_cnr = tmp_hl
    hl_lookback_tot = tmp_hl
    hl_lookback_3m2y = tmp_hl
    hl_lookback_fx = tmp_hl
    hl_lookback_msci = tmp_hl
    hl_lookback_iv = tmp_hl
    
    # Define the date range starting from January 1, 2013
    dates = df.index[df.index >= pd.Timestamp('2013-01-01')]
    
    cols_3m2y = [col for col in df.columns if col.endswith('_3M2Y')]
    currencies = []  # List to store currency names
    # Define the currencies list outside the loop
    for col in cols_3m2y:
        cur = col.split('_')[0]
        if cur not in currencies:
            currencies.append(cur)
    
    # Initialize previous weights and DV01 to zero for the first iteration
    returns_len = len(grouped_columns['_3M2Y'])
    prev_optimal_weights = np.zeros(returns_len)
    prev_dv01 = np.zeros(returns_len)
    cumulative_gross_return = np.zeros(returns_len)
    cumulative_net_return = np.zeros(returns_len)

    # Initialize the results list
    results_list = []
    
    # Specify the desired output file path
    output_file = f'Outputs/{test_type}/{test_sub_type}/{test_type}_smooth{hl_smooth}_zscore{z_score_upper}_hl{tmp_hl}.csv'

    # Extract the directory path
    output_dir = os.path.dirname(output_file)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate CNR metrics
    calculate_cnr_metrics_for_all_currencies(df=df, grouped_columns=grouped_columns,
                                             hl_lookback=hl_lookback_cnr, hl_smooth=hl_smooth,
                                             z_score_lower=z_score_lower, z_score_upper=z_score_upper)

    # Calculate ToT metrics
    calculate_metrics_for_all_currencies(df=df, grouped_columns=grouped_columns, prefix='ToT',
                                         hl_lookback=hl_lookback_tot, hl_smooth=hl_smooth,
                                         z_score_lower=z_score_lower, z_score_upper=z_score_upper)

    # Calculate 3M2Y metrics
    calculate_metrics_for_all_currencies(df=df, grouped_columns=grouped_columns, prefix='3M2Y',
                                         hl_lookback=hl_lookback_3m2y, hl_smooth=hl_smooth,
                                         z_score_lower=z_score_lower_3m2y, z_score_upper=z_score_upper_3m2y)

    # Calculate FX metrics
    calculate_metrics_for_all_currencies(df=df, grouped_columns=grouped_columns, prefix='FX',
                                         hl_lookback=hl_lookback_fx, hl_smooth=hl_smooth,
                                         z_score_lower=z_score_lower, z_score_upper=z_score_upper)

    # Calculate MSCI metrics
    calculate_msci_metrics(df, grouped_columns['_MSCI'], grouped_columns['_FX'], hl_lookback=hl_lookback_msci,
                           hl_smooth=hl_smooth, z_score_lower=z_score_lower, z_score_upper=z_score_upper)

    # Calculate IV metrics
    calculate_all_metrics(df, grouped_columns['_IV'], hl_smooth_iv=hl_smooth, hl_mean_iv=hl_lookback_iv,
                          z_score_lower=z_score_lower, z_score_upper=z_score_upper)

    # Initialize DataFrame for rolling 5-day total returns
    df_rolling_5d_tr = pd.DataFrame(index=df.index)
    
    # Calculate daily price changes, daily total returns, cumulative returns, and rolling total returns
    df, df_rolling_5d_tr = calculate_returns(df, grouped_columns['_3M2Y'], rolling_window=rolling_window, hl_lookback=hl_lookback_returns)
    
    # Calculate exponentially weighted covariance and correlation matrices
    ewm_cov_matrix_all = df_rolling_5d_tr.ewm(halflife=hl_lookback_cov).cov()
    ewm_corr_matrix_all = df_rolling_5d_tr.ewm(halflife=hl_lookback_cov).corr()
    
    # Loop through each day from the first day of 2013
    for date in tqdm(dates, desc="Processing dates"):
        # Calculate the exponentially weighted covariance matrix and correlation matrix
        ewm_cov_matrix = ewm_cov_matrix_all.loc[date]
        ewm_corr_matrix = ewm_corr_matrix_all.loc[date]
        
        last_index = len(df_rolling_5d_tr.columns)
        ewm_cov_matrix = ewm_cov_matrix.iloc[-last_index:, -last_index:]
        ewm_corr_matrix = ewm_corr_matrix.iloc[-last_index:, -last_index:]
        
        # Annualize the covariance matrix for 5-day returns (since there are 52 weeks in a year)
        annualized_cov_matrix = ewm_cov_matrix * 52
        
        # Convert to numpy arrays if they are DataFrames
        if isinstance(ewm_cov_matrix, pd.DataFrame):
            ewm_cov_matrix = ewm_cov_matrix.values
        if isinstance(ewm_corr_matrix, pd.DataFrame):
            ewm_corr_matrix = ewm_corr_matrix.values

        # Define expected returns for each asset using the capped z-score as a proxy
        expected_returns = []
        rolling_5d_tr_std_values = []  # List to store rolling 5-day total return standard deviations 

        # Define expected returns for each asset using the capped z-score as a proxy
        expected_returns, rolling_5d_tr_std_values = calculate_expected_returns(df, grouped_columns['_3M2Y'], date, metrics_weights)
        expected_returns = np.array(expected_returns)
    
        # Solve the quadratic utility optimization problem without scaling
        optimal_weights = quadratic_utility_optimizer(expected_returns, ewm_corr_matrix, risk_aversion, np.sqrt(np.dot(expected_returns.T, np.dot(ewm_corr_matrix, expected_returns))))
        
        if optimal_weights is None:
            continue  # Skip this date if optimization failed
        
        # Calculate the portfolio variance and volatility
        portfolio_variance = np.dot(optimal_weights.T, np.dot(ewm_corr_matrix, optimal_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate the scaling factor and rescale the optimal weights
        scaling_factor = target_volatility / portfolio_volatility
        scaled_optimal_weights = optimal_weights * scaling_factor
        
        # Ensure no currency's optimal risk weight exceeds the target volatility
        scaled_optimal_weights = np.clip(scaled_optimal_weights, None, target_volatility)
        
        # Calculate the 1-day change in position
        position_change = scaled_optimal_weights - prev_optimal_weights
        
        # Adjust the weights to ensure minimum trade size only if the change exceeds the threshold
        for i, cur in enumerate(currencies):
            if abs(position_change[i]) < min_trade_size:
                scaled_optimal_weights[i] = prev_optimal_weights[i]
    
        # Recalculate the 1-day change in position after adjustment
        position_change = scaled_optimal_weights - prev_optimal_weights
        
        # Calculate the optimal DV01 for the current day
        current_dv01 = scaled_optimal_weights / rolling_5d_tr_std_values
        
        # Calculate the daily change in optimal DV01
        dv01_change = current_dv01 - prev_dv01
        
        # Calculate the transaction costs
        t_costs = [
            0 if position_change[i] == 0 else abs(dv01_change[i] * transaction_costs[cur])
            for i, cur in enumerate(currencies)
        ]
        
        # Calculate the 1-day realized return and net return
        realized_return = prev_dv01 * 100 * df.loc[date, [f"{cur}_1D_TR" for cur in currencies]].values
        net_return = realized_return - t_costs
        
        # Update cumulative returns
        cumulative_gross_return += realized_return
        cumulative_net_return += net_return
        
        # Calculate portfolio expected volatility using DV01 weights rounded to whole integers
        adjusted_dv01_weights = current_dv01 * 100
        dv01_portfolio_variance = np.dot(adjusted_dv01_weights.T, np.dot(annualized_cov_matrix, adjusted_dv01_weights))
        dv01_portfolio_volatility = np.sqrt(dv01_portfolio_variance)
        
        # Collect the results for the current date
        for i, cur in enumerate(currencies):
            results_list.append({
                'Date': date,
                'Currency': cur,
                'Optimal Risk Weight': round(scaled_optimal_weights[i], 2),
                'Std Dev': round(rolling_5d_tr_std_values[i], 2),
                'Expected Return': round(expected_returns[i], 2),
                'Optimal DV01': round(current_dv01[i], 2),
                'Portfolio Volatility': round(target_volatility, 2),
                '1-Day Change': round(position_change[i], 2),
                'T-Costs': round(t_costs[i], 2),
                '1D Gross Return': round(realized_return[i], 2),
                '1D Net Return': round(net_return[i], 2),
                'Cumulative Gross Return': round(cumulative_gross_return[i], 2),
                'Cumulative Net Return': round(cumulative_net_return[i], 2),
                'DV01 Portfolio Volatility': round(dv01_portfolio_volatility, 0)
            })
        
        # Update previous weights and DV01 for the next iteration
        prev_optimal_weights = scaled_optimal_weights
        prev_dv01 = current_dv01
    
    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)
    
    # Export the results to a CSV file
    results_df.to_csv(output_file, index=False)
    
    
    

    
    