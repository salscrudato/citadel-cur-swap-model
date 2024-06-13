#signal_test.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import warnings
import argparse

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from signal_test_utils import (set_bbg_ticker_mapping, create_ticker_mapping, group_columns_by_suffix,
                               process_and_convert_header_row, calculate_cnr_metrics_for_all_currencies,
                               calculate_metrics_for_all_currencies, calculate_msci_metrics,
                               print_results, calculate_returns, calculate_expected_returns,
                               quadratic_utility_optimizer)

#-------------------- File Variables --------------------
BBG_FP = os.path.join(BASE_DIR, 'Front End Data.csv')
BBG_CODES_FP = os.path.join(BASE_DIR, 'bbg_codes.xlsx')
BBG_HEADER_ROW = 2
BBG_NEW_HEADER_ROW = BBG_HEADER_ROW + 2

#-------------------- Start Coding Logic --------------------
# Read the raw Bloomberg data
df_raw = pd.read_csv(BBG_FP)

# Create Bloomberg ticker dictionary
bbg_tickers = set_bbg_ticker_mapping(BBG_CODES_FP)

# Create mapping between Bloomberg tickers and df headers
bbg_ticker_dict = create_ticker_mapping(bbg_tickers)

# Preprocess df
df = process_and_convert_header_row(df_raw, BBG_HEADER_ROW, BBG_NEW_HEADER_ROW, bbg_ticker_dict)

# Define suffixes
suffixes = list({value[1] for value in bbg_tickers.values()})

# Create dictionary of suffixes
grouped_columns = group_columns_by_suffix(df, bbg_tickers)

#-------------------- Variables to Change Signal --------------------
test_type = 'CNR'

# parser = argparse.ArgumentParser(description='Run signal test with different hhalf life values')
# parser.add_argument('start', type=int, help='Value to start')
# parser.add_argument('end', type=int, help='Value to end')
# args = parser.parse_args()
# start = args.start
# end = args.end
start = 170
end = 171
skip = 1

tmp_hl = f"hl_lookback_{test_type.lower()}"

for tmp_hl in range(start, end, skip):
    print(f"Running for {test_type} Half Life = {tmp_hl}")
    
    # Define half-life lookback
    hl_lookback_cnr = tmp_hl
    hl_lookback = tmp_hl
    hl_lookback_tot = tmp_hl
    hl_lookback_3m2y = tmp_hl
    hl_lookback_fx = tmp_hl
    hl_lookback_msci = tmp_hl
    hl_lookback_iv = tmp_hl
    hl_smooth = 5
    z_score_upper = 4
    z_score_lower = -4
    z_score_upper_3m2y = 3
    z_score_lower_3m2y = -3
    
    #-------------------- Define Optimization Variables --------------------
    results_list = []
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
    
    min_trade_size = 2500
    
    target_volatility = 15000.00
    
    risk_aversion = 1.0
    
    # Specify the desired output file path
    output_file = f'{test_type}/{test_type}_smooth_{hl_smooth}_hl_{tmp_hl}.csv'
    
    # metrics_weights = {
    #     "3M2Y": -0.13,
    #     "CNR": 0.17,
    #     "MSCI": -0.19,
    #     "FX": -0.16,
    #     "ToT": 0.2,
    #     "IV": -0.14
    # }
    metrics_weights = {
        "3M2Y": -1,
        "CNR": 1,
        "MSCI": -1,
        "FX": -1,
        "TOT": 1,
        "IV": -1
    }
    # Update the dictionary based on the test_type
    if test_type in metrics_weights:
        # Preserve the sign and set the value to 1 or -1
        if metrics_weights[test_type] < 0:
            metrics_weights[test_type] = -1
        else:
            metrics_weights[test_type] = 1
        
        # Create a new dictionary with only the updated key
        updated_metrics_weights = {test_type: metrics_weights[test_type]}
    else:
        # If test_type is not found, keep the dictionary empty
        updated_metrics_weights = {}
    
    metrics_weights = updated_metrics_weights
    
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
    calculate_metrics_for_all_currencies(df=df, grouped_columns=grouped_columns, prefix='USD',
                                             hl_lookback=hl_lookback_fx, hl_smooth=hl_smooth,
                                             z_score_lower=z_score_lower, z_score_upper=z_score_upper)
    
    # MSCI MSCI metrics
    calculate_msci_metrics(df, grouped_columns['_MSCI'], grouped_columns['_USD'], hl_lookback=hl_lookback_msci,
                           hl_smoothed=hl_smooth, z_score_lower=z_score_lower, z_score_upper=z_score_upper)
    
    # Calculate IV metrics
    calculate_metrics_for_all_currencies(df=df, grouped_columns=grouped_columns, prefix='IV',
                                             hl_lookback=hl_lookback_iv, hl_smooth=hl_smooth,
                                             z_score_lower=z_score_lower, z_score_upper=z_score_upper)

    #-------------------- TODO: Try Different Dates --------------------
    dates = df.index[df.index >= pd.Timestamp('2013-01-01')]
    
    #-------------------- TODO: Change this to remove all references that aren't generic --------------------
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
    
    df_rolling_5d_tr = pd.DataFrame(index=df.index)
    # Calculate daily price changes, daily total returns, and cumulative returns
    df, df_rolling_5d_tr = calculate_returns(df, grouped_columns['_3M2Y'], rolling_window=5, hl_lookback=63)
    ewm_cov_matrix_all = df_rolling_5d_tr.ewm(halflife=126).cov()
    ewm_corr_matrix_all = df_rolling_5d_tr.ewm(halflife=126).corr()
    
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
        if isinstance(ewm_cov_matrix, pd.DataFrame):
            ewm_cov_matrix = ewm_cov_matrix.values
        # Verify that the correlation matrix is in the right format for CVXPY
        if isinstance(ewm_corr_matrix, pd.DataFrame):
            ewm_corr_matrix = ewm_corr_matrix.values
    
        # Define expected returns for each asset using the capped z-score as a proxy
        expected_returns = []
        rolling_5d_tr_std_values = []  # List to store rolling 5-day total return standard deviations 
        
        expected_returns, rolling_5d_tr_std_values = calculate_expected_returns(df, grouped_columns['_3M2Y'], date, metrics_weights)
        # Convert the list to a NumPy array for compatibility with CVXPY
        expected_returns = np.array(expected_returns)
    
        # Solve the quadratic utility optimization problem without scaling
        optimal_weights = quadratic_utility_optimizer(expected_returns, ewm_corr_matrix, risk_aversion, np.sqrt(np.dot(expected_returns.T, np.dot(ewm_corr_matrix, expected_returns))))
    
        if optimal_weights is None:
            continue  # Skip this date if optimization failed
            
        # Calculate the portfolio variance
        portfolio_variance = np.dot(optimal_weights.T, np.dot(ewm_corr_matrix, optimal_weights))
    
        # Calculate the portfolio volatility (standard deviation)
        portfolio_volatility = np.sqrt(portfolio_variance)
    
        # Calculate the scaling factor
        scaling_factor = target_volatility / portfolio_volatility
    
        # Rescale the optimal weights
        scaled_optimal_weights = optimal_weights * scaling_factor
    
        # Ensure no currency's optimal risk weight exceeds 15000
        scaled_optimal_weights = np.clip(scaled_optimal_weights, None, target_volatility)
    
        # Calculate the 1-day change in position
        position_change = scaled_optimal_weights - prev_optimal_weights  # Calculate 1-day change in position
    
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
    
        # Calculate the transaction costs and set to zero if 1-day change is zero
        t_costs = [
            0 if position_change[i] == 0 else abs(dv01_change[i] * transaction_costs[cur])
            for i, cur in enumerate(currencies)
        ]
    
        # Calculate the 1d realized return
        realized_return = prev_dv01 * 100 * df.loc[date, [cur + "_1D_TR" for cur in currencies]].values
    
        # Calculate the 1d net return
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
                'Std Dev': round(rolling_5d_tr_std_values[i]),
                'Expected Return': round(expected_returns[i], 2),
                'Optimal DV01': round(current_dv01[i], 2),
                'Portfolio Volatility': round(target_volatility, 2),
                '1-Day Change': round(position_change[i], 2),
                'T-Costs': round(t_costs[i], 2),  # Transaction costs (absolute value)
                '1D Gross Return': round(realized_return[i], 2),  # 1d realized return renamed to 1D Gross Return
                '1D Net Return': round(net_return[i], 2),  # 1d net return
                'Cumulative Gross Return': round(cumulative_gross_return[i], 2),  # Cumulative gross return
                'Cumulative Net Return': round(cumulative_net_return[i], 2),  # Cumulative net return
                'DV01 Portfolio Volatility': round(dv01_portfolio_volatility, 0)  # Add new column for DV01 portfolio volatility
            })
    
        # Update previous weights and DV01 for the next iteration
        prev_optimal_weights = scaled_optimal_weights
        prev_dv01 = current_dv01
    
    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)
    
    # Export the results to a CSV file
    results_df.to_csv(output_file, index=False)
    
    
    
