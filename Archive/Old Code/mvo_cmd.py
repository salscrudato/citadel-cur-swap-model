# import warnings
# warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
#import cvxpy as cp
from tqdm import tqdm
from utils import *
import sys


# File Paths
file_path = 'Front End Data.csv'
bbg_file_path = 'bbg_codes_old.xlsx'

# Read the raw data
df_raw = load_raw_dataframe(file_path)

# Define the header row and skip rows
bbg_header_row = 2
bbg_skip_rows = bbg_header_row + 3

# Set BBG tickers
bbg_tickers = set_bbg_ticker_mapping(bbg_file_path)

# Create ticker mapping
ticker_mapping = create_ticker_mapping(bbg_tickers)












# Generate new headers
new_headers = generate_new_headers(df_raw, bbg_header_row, ticker_mapping)

# Load the final dataframe with new headers
df = load_dataframe_with_new_headers(file_path, bbg_skip_rows, new_headers)

# Clean the dataframe
df = preprocess_dataframe(df)

# Define column suffixes
suffixes = ['_2Y', '_3M2Y', '_ToT', '_USD', '_IV', '_CPI', '_MSCI']

# Create column groups based on suffixes
grouped_columns = group_columns_by_suffix(df, suffixes)

# Define transaction costs and minimum trade size
transaction_costs = {
    'CLP': 2, 'CNY': 1, 'COP': 3, 'CZK': 1.5, 'HUF': 3,
    'INR': 2, 'ILS': 2, 'KRW': 1.5, 'MXN': 1, 'PLN': 1.5, 'ZAR': 1.5
}
min_trade_size = 2500

# Filter dates and initialize variables
dates = df.index[df.index >= pd.Timestamp('2013-01-01')]
len_3m2y = len(grouped_columns['_3M2Y'])

# Half-life values to test
#half_life_values = range(182, 202, 1)

def run_backtest_for_half_life(hl_cnr, hl_lookback_tot, hl_lookback_mom, hl_mean_fx, hl_mean_msci, hl_mean_iv):
    prev_optimal_weights = np.zeros(len_3m2y)
    prev_dv01 = np.zeros(len_3m2y)
    cumulative_gross_return = np.zeros(len_3m2y)
    cumulative_net_return = np.zeros(len_3m2y)
    results_list = []
    
    # Calculate CNR metrics
    for col_3m2y in grouped_columns['_3M2Y']:
        currency_3m2y = col_3m2y.split('_')[0]
        for col_2y in grouped_columns['_2Y']:
            currency_2y = col_2y.split('_')[0]
            if currency_3m2y == currency_2y:
                calculate_cnr_metrics(df, col_3m2y, col_2y, currency_3m2y, hl_lookback=hl_cnr)
            


    # Process different column types with given half-lives
    tot_to_z_score(df, grouped_columns['_ToT'], hl_lookback_tot)
    process_mom_columns(df, grouped_columns['_3M2Y'], hl_lookback_mom)
    process_fx_columns(df, grouped_columns['_USD'], hl_mean_fx)
    process_msci_columns(df, grouped_columns['_MSCI'], grouped_columns['_USD'], hl_mean_msci)
    process_iv_columns(df, grouped_columns['_IV'], hl_mean_iv)

    # Calculate rolling metrics
    calculate_rolling_metrics(df, grouped_columns['_3M2Y'])

    for date in tqdm(dates, desc=f"Processing dates for half-life {hl_cnr}"):
        df_rolling_5d_tr = df[[f'{col.split("_")[0]}_5D_TR' for col in grouped_columns['_3M2Y']]].copy()
        
        ewm_cov_matrix = df_rolling_5d_tr.ewm(halflife=126).cov().loc[date]
        ewm_corr_matrix = df_rolling_5d_tr.ewm(halflife=126).corr().loc[date]
        
        last_index = len(df_rolling_5d_tr.columns)
        ewm_cov_matrix = ewm_cov_matrix.iloc[-last_index:, -last_index:]
        ewm_corr_matrix = ewm_corr_matrix.iloc[-last_index:, -last_index:]
        
        annualized_cov_matrix = ewm_cov_matrix * 52
        
        expected_returns = []
        currencies = []
        rolling_5d_tr_std_values = []

        for col in grouped_columns['_3M2Y']:
            cur = col.split('_')[0]
            currencies.append(cur)
            # CNR Test
            # expected_returns.append(df[cur + "_CNR_Z_SCORE_CAPPED"].loc[date])
            # 3M2Y Test
            expected_returns.append(df[cur + "_3M2Y_Z_SCORE_CAPPED"].loc[date] * -1)
            rolling_5d_tr_std_values.append(df[f'{cur}_TR_STD'].loc[date])

        expected_returns = np.array(expected_returns)
        
        risk_aversion = 1.0
        optimal_weights = quadratic_utility_optimizer(expected_returns, ewm_corr_matrix, risk_aversion, np.sqrt(np.dot(expected_returns.T, np.dot(ewm_corr_matrix, expected_returns))))

        if optimal_weights is None:
            continue

        portfolio_variance = np.dot(optimal_weights.T, np.dot(ewm_corr_matrix, optimal_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        target_volatility = 15000.00

        scaling_factor = target_volatility / portfolio_volatility
        scaled_optimal_weights = optimal_weights * scaling_factor
        scaled_optimal_weights = np.clip(scaled_optimal_weights, None, 15000)

        position_change = scaled_optimal_weights - prev_optimal_weights

        for i, cur in enumerate(currencies):
            if abs(position_change[i]) < min_trade_size:
                scaled_optimal_weights[i] = prev_optimal_weights[i]

        position_change = scaled_optimal_weights - prev_optimal_weights
        current_dv01 = scaled_optimal_weights / rolling_5d_tr_std_values
        dv01_change = current_dv01 - prev_dv01

        t_costs = [0 if position_change[i] == 0 else abs(dv01_change[i] * transaction_costs[cur]) for i, cur in enumerate(currencies)]

        realized_return = prev_dv01 * 100 * df.loc[date, [f'{cur}_1D_TR' for cur in currencies]].values
        net_return = realized_return - t_costs

        cumulative_gross_return += realized_return
        cumulative_net_return += net_return

        adjusted_dv01_weights = current_dv01 * 100
        dv01_portfolio_variance = np.dot(adjusted_dv01_weights.T, np.dot(annualized_cov_matrix, adjusted_dv01_weights))
        dv01_portfolio_volatility = np.sqrt(dv01_portfolio_variance)

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
                'T-Costs': round(t_costs[i], 2),
                '1D Gross Return': round(realized_return[i], 2),
                '1D Net Return': round(net_return[i], 2),
                'Cumulative Gross Return': round(cumulative_gross_return[i], 2),
                'Cumulative Net Return': round(cumulative_net_return[i], 2),
                'DV01 Portfolio Volatility': round(dv01_portfolio_volatility, 0)
            })

        prev_optimal_weights = scaled_optimal_weights
        prev_dv01 = current_dv01

    return results_list

if __name__ == "__main__":
    hl_start = int(sys.argv[1])
    hl_end = int(sys.argv[2])
    hl_test = 'mom'

    for i in range(hl_start, hl_end, 1):
        print(i)
        results_list = run_backtest_for_half_life(174, 63, i, 63, 63, 63)
        results_df = pd.DataFrame(results_list)
        output_path = f'Outputs/output_{hl_test}_hl_{i}.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Completed backtest for half-life {i}")


