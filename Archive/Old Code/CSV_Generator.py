#!/usr/bin/env python
# coding: utf-8

# Imports
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta



import cvxpy as cp
from tqdm import tqdm

import os, sys
import warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
BBG_FP = os.path.join(BASE_DIR, 'Front End Data.csv')
BBG_CODES_FP = os.path.join(BASE_DIR, 'bbg_codes.xlsx')
BBG_HEADER_ROW = 2
BBG_NEW_HEADER_ROW = BBG_HEADER_ROW + 2
file_path = 'Front End Data.csv'
BBG_CODES_FP = 'bbg_codes.xlsx'

#Read File
file_name = BBG_FP

#Create Dictionary
bbg_tickers = {
    'CHSWP2': ['CLP_2Y', 'CLP'],
    'CCSWNI2': ['CNY_2Y', 'CNY'],
    'CLSWIB2': ['COP_2Y', 'COP'],
    'CKSW2': ['CZK_2Y', 'CZK'],
    'HFSW2': ['HUF_2Y', 'HUF'],
    'IRSWNI2': ['INR_2Y', 'INR'],
    'ISSW2': ['ILS_2Y', 'ILS'],
    'KWSWNI2': ['KRW_2Y', 'KRW'],
    'MPSW2B': ['MXN_2Y', 'MXN'],
    'PZSW2': ['PLN_2Y', 'PLN'],
    'SASW2': ['ZAR_2Y', 'ZAR'],
    'CHFS0C02': ['CLP_3M2Y', 'CLP'],
    'S0204FS': ['CNY_3M2Y', 'CNY'],
    'S0329FS': ['COP_3M2Y', 'COP'],
    'S0320FS': ['CZK_3M2Y', 'CZK'],
    'S0325FS': ['HUF_3M2Y', 'HUF'],
    'S0266FS': ['INR_3M2Y', 'INR'],
    'ISFS0C02': ['ILS_3M2Y', 'ILS'],
    'S0205FS': ['KRW_3M2Y', 'KRW'],
    'MPFS0C02': ['MXN_3M2Y', 'MXN'],
    'S0323FS': ['PLN_3M2Y', 'PLN'],
    'SAFS0C02': ['ZAR_3M2Y', 'ZAR'],
    'USDCLP': ['CLP_USD', 'CLP'],
    'USDCNH': ['CNY_USD', 'CNY'],
    'USDCOP': ['COP_USD', 'COP'],
    'USDCZK': ['CZK_USD', 'CZK'],
    'USDHUF': ['HUF_USD', 'HUF'],
    'USDINR': ['INR_USD', 'INR'],
    'USDILS': ['ILS_USD', 'ILS'],
    'USDKRW': ['KRW_USD', 'KRW'],
    'USDMXN': ['MXN_USD', 'MXN'],
    'USDPLN': ['PLN_USD', 'PLN'],
    'USDZAR': ['ZAR_USD', 'ZAR'],
    'GSCLPTOT': ['CLP_ToT', 'CLP'],
    'GSCNYTOT': ['CNY_ToT', 'CNY'],
    'GSCOPTOT': ['COP_ToT', 'COP'],
    'GSCZKTOT': ['CZK_ToT', 'CZK'],
    'GSHUFTOT': ['HUF_ToT', 'HUF'],
    'GSINRTOT': ['INR_ToT', 'INR'],
    'GSILSTOT': ['ILS_ToT', 'ILS'],
    'GSKRWTOT': ['KRW_ToT', 'KRW'],
    'GSMXNTOT': ['MXN_ToT', 'MXN'],
    'GSPLNTOT': ['PLN_ToT', 'PLN'],
    'GSZARTOT': ['ZAR_ToT', 'ZAR'],
    'MXCL': ['CLP_MSCI', 'CLP'],
    'MXCN': ['CNY_MSCI', 'CNY'],
    'MXCO': ['COP_MSCI', 'COP'],
    'MXCZ': ['CZK_MSCI', 'CZK'],
    'MXHU': ['HUF_MSCI', 'HUF'],
    'MXIN': ['INR_MSCI', 'INR'],
    'MXIL': ['ILS_MSCI', 'ILS'],
    'MXKR': ['KRW_MSCI', 'KRW'],
    'MXMX': ['MXN_MSCI', 'MXN'],
    'MXPL': ['PLN_MSCI', 'PLN'],
    'MXZA': ['ZAR_MSCI', 'ZAR'],
    'USDCLPV3M': ['CLP_IV', 'CLP'],
    'USDCNHV3M': ['CNY_IV', 'CNY'],
    'USDCOPV3M': ['COP_IV', 'COP'],
    'USDCZKV3M': ['CZK_IV', 'CZK'],
    'USDHUFV3M': ['HUF_IV', 'HUF'],
    'USDINRV3M': ['INR_IV', 'INR'],
    'USDILSV3M': ['ILS_IV', 'ILS'],
    'USDKRWV3M': ['KRW_IV', 'KRW'],
    'USDMXNV3M': ['MXN_IV', 'MXN'],
    'USDPLNV3M': ['PLN_IV', 'PLN'],
    'USDZARV3M': ['ZAR_IV', 'ZAR'],
    'CLINNSYO': ['CLP_CPI', 'CLP'],
    'CNCPIYOY': ['CNY_CPI', 'CNY'],
    'COCPIYOY': ['COP_CPI', 'COP'],
    'CZCPYOY': ['CZK_CPI', 'CZK'],
    'HUCPIYY': ['HUF_CPI', 'HUF'],
    'INFUTOTY': ['INR_CPI', 'INR'],
    'ISCPIYYN': ['ILS_CPI', 'ILS'],
    'KOCPIYOY': ['KRW_CPI', 'KRW'],
    'MXCPYOY': ['MXN_CPI', 'MXN'],
    'POCPIYOY': ['PLN_CPI', 'PLN'],
    'SACPIYOY': ['ZAR_CPI', 'ZAR']
}

# Define variables for initial data frame clean-up
bbg_header_row = 2
bbg_headers_new = ["Dates"]
bbg_skip_rows = bbg_header_row + 3

# Read the raw file
df_raw = pd.read_csv(file_name)

# Pull the bloomberg headers from CSV
bbg_headers = df_raw.iloc[bbg_header_row].tolist()

# Create new header names
for ticker in bbg_tickers:
    bbg_headers_new.append(bbg_tickers[ticker][0])

# Read the CSV again using the modified headers
df = pd.read_csv(file_name, skiprows=bbg_skip_rows)

# Set the new headers
df.columns = bbg_headers_new

# Clean df
df = df.replace('#NAME?', np.nan)
df = df.bfill()
df = df.ffill()

# Tell Jupyter that the data type for the Dates column is Dates
df["Dates"] = pd.to_datetime(df["Dates"])

# Slice to a start date in the df (if necessary)
# df = df.loc[:end_date]
df = df.sort_values(by='Dates')

# If Dates appear in columns, set the index to "Dates"
df.set_index("Dates", inplace=True)

# Define column groups
cols_2y = [col for col in df.columns if col.endswith('_2Y')]
cols_3m2y = [col for col in df.columns if col.endswith('_3M2Y')]
cols_tot = [col for col in df.columns if col.endswith('_ToT')]
cols_fx = [col for col in df.columns if col.endswith('USD')]
cols_iv = [col for col in df.columns if col.endswith('_IV')]
cols_cpi = [col for col in df.columns if col.endswith('_CPI')]
cols_msci = [col for col in df.columns if col.endswith('_MSCI')]

from sklearn.preprocessing import MinMaxScaler

# 'cols_3m2y' and 'cols_2y' are lists of columns defined earlier
daily_cnr_columns = []

# Calculate CNR and related metrics
for col in cols_3m2y:
    for col2 in cols_2y:
        cur = col.split('_')[0]
        cur2 = col2.split('_')[0]
        if cur == cur2:
            daily_cnr = cur + "_CNR"
            daily_cnr_smoothed = cur + "_CNR_SMOOTHED"
            cnr_z_score = cur + "_CNR_Z_SCORE"
            cnr_z_score_capped = cur + "_CNR_Z_SCORE_CAPPED"
            cnr_std = cur + "_CNR_STD"
            cnr_mean = cur + "_CNR_MEAN"
            df[daily_cnr] = (df[col] - df[col2]) * 4 / 252
            daily_cnr_columns.append(daily_cnr)
            df[daily_cnr_smoothed] = df[daily_cnr].ewm(halflife=5).mean()
            df[cnr_mean] = df[daily_cnr_smoothed].ewm(halflife=174).mean()
            df[cnr_std] = df[daily_cnr_smoothed].ewm(halflife=174).std()
            df[cnr_z_score] = (df[daily_cnr_smoothed] - df[cnr_mean]) / df[cnr_std]
            df[cnr_z_score_capped] = df[cnr_z_score].clip(lower=-4, upper=4)
print(df['CLP_CNR_Z_SCORE_CAPPED'].iloc[0])
print(df['CLP_CNR_Z_SCORE_CAPPED'].iloc[-1])

for col in cols_tot:
    cur = col.split('_')[0]
    tot = cur+"_ToT"
    tot_smoothed = cur+"_ToT SMOOTHED"
    tot_z_score = cur+"_ToT_Z_SCORE"
    tot_z_score_capped = cur+"_ToT_Z_SCORE_CAPPED"
    tot_std = cur+"_ToT_STD"
    tot_mean = cur+"_ToT_MEAN"
    df[tot_smoothed] = df[tot].ewm(halflife=5).mean()
    df[tot_mean] = df[tot_smoothed].ewm(halflife=42).mean()
    df[tot_std] = df[tot_smoothed].ewm(halflife=42).std()
    df[tot_z_score] = (df[tot_smoothed]-df[tot_mean]) / df[tot_std]
    df[tot_z_score_capped] = df[tot_z_score].clip(lower=-4,upper=4) 
print(df['CLP_ToT_Z_SCORE_CAPPED'].iloc[0])
print(df['CLP_ToT_Z_SCORE_CAPPED'].iloc[-1])

for col in cols_3m2y:
    cur = col.split('_')[0]
    mom = cur+"_3M2Y"
    mom_smoothed = cur+"_3M2Y_SMOOTHED"
    mom_z_score = cur+"_3M2Y_Z_SCORE"
    mom_z_score_capped = cur+"_3M2Y_Z_SCORE_CAPPED"
    mom_std = cur+"_3M2Y_STD"
    mom_mean = cur+"_3M2Y_MEAN"
    df[mom_smoothed] = df[mom].ewm(halflife=5).mean()
    df[mom_mean] = df[mom_smoothed].ewm(halflife=504).mean()
    df[mom_std] = df[mom_smoothed].ewm(halflife=504).std()
    df[mom_z_score] = (df[mom_smoothed]-df[mom_mean]) / df[mom_std]
    df[mom_z_score_capped] = df[mom_z_score].clip(lower=-3,upper=3)
print(df['CLP_3M2Y_Z_SCORE_CAPPED'].iloc[0])
print(df['CLP_3M2Y_Z_SCORE_CAPPED'].iloc[-1])

for col in cols_fx:
    cur = col.split('_')[0]
    fx = cur+"_USD"
    fx_smoothed = cur+"_FX SMOOTHED"
    fx_z_score = cur+"_FX_Z_SCORE"
    fx_z_score_capped = cur+"_FX_Z_SCORE_CAPPED"
    fx_std = cur+"_FX_STD"
    fx_mean = cur+"_FX_MEAN"
    df[fx_smoothed] = df[fx].ewm(halflife=5).mean()
    df[fx_mean] = df[fx_smoothed].ewm(halflife=84).mean()
    df[fx_std] = df[fx_smoothed].ewm(halflife=84).std()
    df[fx_z_score] = (df[fx_smoothed]-df[fx_mean]) / df[fx_std]
    df[fx_z_score_capped] = df[fx_z_score].clip(lower=-4,upper=4) 
print(df['CLP_FX_Z_SCORE_CAPPED'].iloc[0])
print(df['CLP_FX_Z_SCORE_CAPPED'].iloc[-1])

for col in cols_msci:
    for col2 in cols_fx:
        cur = col.split('_')[0]
        cur2 = col2.split('_')[0]
        if cur == cur2:
            msci_lccy = cur+"_MSCI_LCCY"
            msci_lccy_smoothed = cur+"_MSCI_LCCY_SMOOTHED"
            msci_z_score = cur+"_MSCI_Z_SCORE"
            msci_z_score_capped = cur+"_MSCI_Z_SCORE_CAPPED"
            msci_std = cur+"_MSCI_STD"
            msci_mean = cur+"_MSCI_MEAN"
            df[msci_lccy] = (df[col] * df[col2])
            df[msci_lccy_smoothed] = df[msci_lccy].ewm(halflife=5).mean()
            df[msci_mean] = df[msci_lccy_smoothed].ewm(halflife=84).mean()
            df[msci_std] = df[msci_lccy_smoothed].ewm(halflife=84).std()
            df[msci_z_score] = (df[msci_lccy_smoothed]-df[msci_mean]) / df[msci_std]
            df[msci_z_score_capped] = df[msci_z_score].clip(lower=-4,upper=4)
print(df['CLP_MSCI_Z_SCORE_CAPPED'].iloc[0])
print(df['CLP_MSCI_Z_SCORE_CAPPED'].iloc[-1])

for col in cols_iv:
    cur = col.split('_')[0]
    iv = cur+"_IV"
    iv_smoothed = cur+"_IV_SMOOTHED"
    iv_z_score = cur+"_IV_Z_SCORE"
    iv_z_score_capped = cur+"_IV_Z_SCORE_CAPPED"
    iv_std = cur+"_IV_STD"
    iv_mean = cur+"_IV_MEAN"
    df[iv_smoothed] = df[iv].ewm(halflife=5).mean()
    df[iv_mean] = df[iv_smoothed].ewm(halflife=252).mean()
    df[iv_std] = df[iv_smoothed].ewm(halflife=252).std()
    df[iv_z_score] = (df[iv_smoothed]-df[iv_mean]) / df[iv_std]
    df[iv_z_score_capped] = df[iv_z_score].clip(lower=-4,upper=4)
print(df['CLP_IV_Z_SCORE_CAPPED'].iloc[0])
print(df['CLP_IV_Z_SCORE_CAPPED'].iloc[-1])

# Create an empty DataFrame to store the results
results_list = []

# Transaction costs for each currency (static values)
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

# Minimum trade size threshold
min_trade_size = 2500

# Define expected returns for each asset using the capped z-score as a proxy
expected_returns = []

dates = df.index[df.index >= pd.Timestamp('2013-01-01')]
currencies = []  # List to store currency names
# Define the currencies list outside the loop
for col in cols_3m2y:
    cur = col.split('_')[0]
    if cur not in currencies:
        currencies.append(cur)

rolling_5d_tr_std_values = []  # List to store rolling 5-day total return standard deviations

# Initialize previous weights and DV01 to zero for the first iteration
prev_optimal_weights = np.zeros(len(cols_3m2y))
prev_dv01 = np.zeros(len(cols_3m2y))

# Initialize cumulative returns
cumulative_gross_return = np.zeros(len(cols_3m2y))
cumulative_net_return = np.zeros(len(cols_3m2y))
    
  
    











# Ensure the covariance matrix is compatible with CVXPY
def quadratic_utility_optimizer(expected_returns, corr_matrix, risk_aversion, portfolio_volatility):
    n = len(expected_returns)
    weights = cp.Variable(n)
    portfolio_return = weights @ expected_returns
    portfolio_variance = cp.quad_form(weights, corr_matrix)  # Quadratic form for variance
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    constraints = [
        weights <= portfolio_volatility,  # Adding the constraint for maximum optimal risk weight based on pre-scaling portfolio volatility
        # weights[cop_index] == 0  # Constraint to set COP position to zero
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except Exception as e:
        print(f"Optimization failed for date {date}: {e}")
        return None
    return weights.value


# Loop through each day from the first day of 2013
for date in tqdm(dates, desc="Processing dates"):
    df_rolling_5d_tr = pd.DataFrame(index=df.index)

    # Calculate daily price changes, daily total returns, and cumulative returns
    for col in cols_3m2y:
        cur = col.split('_')[0]
        daily_total_return = cur + "_1D_TR"
        daily_price_change = cur + "_1D_CHG"
        cum_total_return = cur + "_CUM_TR"
        rolling_5d_total_return = cur + "_5D_TR"
        rolling_5d_tr_std = cur + "_TR_STD"
        
        if daily_price_change not in df.columns:
            df[daily_price_change] = df[col].diff()
        
        if daily_total_return not in df.columns:
            df[daily_total_return] = df[cur + "_CNR"] - df[daily_price_change]
        
        df[rolling_5d_total_return] = df[daily_total_return].rolling(window=5).sum()
        df[rolling_5d_tr_std] = df[rolling_5d_total_return].ewm(halflife=63).std() * np.sqrt(252/5) * 100
        df[cum_total_return] = df[daily_total_return].cumsum()
        df_rolling_5d_tr[rolling_5d_total_return] = df[rolling_5d_total_return]

    # Calculate the exponentially weighted covariance matrix and correlation matrix
    ewm_cov_matrix = df_rolling_5d_tr.ewm(halflife=126).cov().loc[date]
    ewm_corr_matrix = df_rolling_5d_tr.ewm(halflife=126).corr().loc[date]
    # Extract the last block of the matrix that corresponds to the rolling 5-day returns
    last_index = len(df_rolling_5d_tr.columns)
    ewm_cov_matrix = ewm_cov_matrix.iloc[-last_index:, -last_index:]
    ewm_corr_matrix = ewm_corr_matrix.iloc[-last_index:, -last_index:]  
    # Annualize the covariance matrix for 5-day returns (since there are 52 weeks in a year)
    annualized_cov_matrix = ewm_cov_matrix * 52   
    # Verify that the covariance matrix is in the right format for CVXPY
    if isinstance(ewm_cov_matrix, pd.DataFrame):
        ewm_cov_matrix = ewm_cov_matrix.values   
    # Verify that the correlation matrix is in the right format for CVXPY
    if isinstance(ewm_corr_matrix, pd.DataFrame):
        ewm_corr_matrix = ewm_corr_matrix.values       
    
    # Define expected returns for each asset using the capped z-score as a proxy
    expected_returns = []
    rolling_5d_tr_std_values = []  # List to store rolling 5-day total return standard deviations   

    # Risk aversion parameter
    risk_aversion = 1.0

    for col in cols_3m2y:
        cur = col.split('_')[0]
        #expected_returns.append((-.13 * df[cur + "_3M2Y_Z_SCORE_CAPPED"].loc[date]) + (.17 * df[cur + "_CNR_Z_SCORE_CAPPED"].loc[date]) - (.19 * df[cur + "_MSCI_Z_SCORE_CAPPED"].loc[date]) - (.16 * df[cur + "_FX_Z_SCORE_CAPPED"].loc[date]) + (.2 * df[cur + "_ToT_Z_SCORE_CAPPED"].loc[date]) - (.14 * df[cur + "_IV_Z_SCORE_CAPPED"].loc[date]))
        expected_returns.append(df[cur + "_CNR_Z_SCORE_CAPPED"].loc[date])
        # Check if the rolling 5D TR Std column exists and append the most recent value
        if cur + "_TR_STD" in df.columns:
            rolling_5d_tr_std_values.append(df[cur + "_TR_STD"].loc[date])
        else:
            rolling_5d_tr_std_values.append(np.nan)  # Append NaN if the column doesn't exist
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

    # Target volatility
    target_volatility = 15000.00  # Corrected to 15000.00

    # Calculate the scaling factor
    scaling_factor = target_volatility / portfolio_volatility

    # Rescale the optimal weights
    scaled_optimal_weights = optimal_weights * scaling_factor

    # Ensure no currency's optimal risk weight exceeds 15000
    scaled_optimal_weights = np.clip(scaled_optimal_weights, None, 15000)

    # Ensure the COP position is zero
    # scaled_optimal_weights[cop_index] = 0

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

















# Specify the desired output file path
#output_file = '/Users/melissamertz/Documents/Bloomberg Data/Combined 1mHL MVO weights 3000 Ex COP.csv'
output_file = f'Outputs2/output_cnr_hl_170_Avi.csv'

# Export the results to a CSV file
results_df.to_csv(output_file, index=False)

print(f"Results have been exported to {output_file}")




