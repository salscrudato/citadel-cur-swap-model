#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_test_utils.py

@author: salscrudato
"""

import pandas as pd
import numpy as np
import cvxpy as cp

def set_bbg_ticker_mapping(path):
    df_bbg = pd.read_excel(path, index_col=0)
    return {key: [value['Currency'], value['Suffix']] for key, value in df_bbg.to_dict(orient='index').items()}

def create_ticker_mapping(bbg_tickers):
    return {key: bbg_tickers[key][0] + bbg_tickers[key][1] for key in bbg_tickers}

def process_and_convert_header_row(df, hr, new_hr, conversion_dict):
    df.iloc[hr] = df.iloc[hr].str.split().str[0].map(conversion_dict)
    df.iloc[hr, 0] = "Dates"
    df.iloc[new_hr] = df.iloc[hr]
    df.columns = df.iloc[new_hr]
    df = df.iloc[new_hr+1:]
    df["Dates"] = pd.to_datetime(df["Dates"])
    df.set_index("Dates", inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df = df.bfill().ffill()
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def group_columns_by_suffix(df, bbg_tickers):
    suffixes = set(value[1] for value in bbg_tickers.values())
    return {suffix: [col for col in df.columns if col.endswith(suffix)] for suffix in suffixes}

def calculate_metrics(df, col, cur, prefix, hl_smooth, hl_lookback, z_score_lower, z_score_upper):
    """
    Calculate metrics for a given dataframe and parameters.

    Parameters:
    df (pandas.DataFrame): Input dataframe
    col (str): Column name for the metric
    cur (str): Currency
    prefix (str): Prefix for the new columns
    hl_smooth (int): Half-life for smoothing
    hl_lookback (int): Half-life for lookback
    z_score_lower (int): Lower limit for z-score clipping
    z_score_upper (int): Upper limit for z-score clipping

    Returns:
    pandas.DataFrame: DataFrame with calculated metrics
    """
    # Check if required columns exist in the dataframe
    assert col in df.columns, f"{col} not in dataframe columns"

    # Calculate smoothed metric
    df[f'{cur}_{prefix}_SMOOTH'] = df[col].ewm(halflife=hl_smooth).mean()

    # Calculate mean and std dev of smoothed metric over lookback period
    df[f'{cur}_{prefix}_MEAN'] = df[f'{cur}_{prefix}_SMOOTH'].ewm(halflife=hl_lookback).mean()
    df[f'{cur}_{prefix}_STD'] = df[f'{cur}_{prefix}_SMOOTH'].ewm(halflife=hl_lookback).std()

    # Calculate z-score of smoothed metric
    df[f'{cur}_{prefix}_Z_SCORE'] = (df[f'{cur}_{prefix}_SMOOTH'] - df[f'{cur}_{prefix}_MEAN']) / df[f'{cur}_{prefix}_STD']

    # Clip z-score to specified range
    df[f'{cur}_{prefix}_Z_SCORE_CAPPED'] = np.clip(df[f'{cur}_{prefix}_Z_SCORE'], z_score_lower, z_score_upper)

    return df

def calculate_cnr_metrics_for_all_currencies(df, grouped_columns, hl_lookback, hl_smooth, z_score_lower, z_score_upper):
    # Create a dictionary where the keys are the currencies and the values are the column names
    columns_dict = {}
    for col in grouped_columns['_3M2Y'] + grouped_columns['_2Y']:
        currency, suffix = col.split('_')
        if currency not in columns_dict:
            columns_dict[currency] = {'3M2Y': None, '2Y': None}
        columns_dict[currency][suffix] = col

    # Iterate over the dictionary and call calculate_cnr_metrics for each pair of columns with the same currency
    for currency, cols in columns_dict.items():
        if cols['3M2Y'] is not None and cols['2Y'] is not None:
            df[f'{currency}_CNR'] = (df[cols['3M2Y']] - df[cols['2Y']]) * 4 / 252
            df = calculate_cnr_metrics(df=df, col_3m2y=cols['3M2Y'], col_2y=cols['2Y'], cur=currency, hl_lookback=hl_lookback, hl_smooth=hl_smooth, z_score_lower=z_score_lower, z_score_upper=z_score_upper)
    return df

def calculate_cnr_metrics(df, col_3m2y, col_2y, cur, hl_smooth, hl_lookback, z_score_lower, z_score_upper):
    """
    Calculate CNR metrics for a given dataframe and parameters.
    Parameters:
    df (pandas.DataFrame): Input dataframe
    col_3m2y (str): Column name for 3M2Y
    col_2y (str): Column name for 2Y
    cur (str): Currency
    hl_smooth (int): Half-life for smoothing
    hl_lookback (int): Half-life for lookback
    z_score_lower (int): Lower limit for z-score clipping
    z_score_upper (int): Upper limit for z-score clipping

    Returns:
    pandas.DataFrame: DataFrame with calculated CNR metrics
    """
    # Check if required columns exist in the dataframe
    assert col_3m2y in df.columns, f"{col_3m2y} not in dataframe columns"
    assert col_2y in df.columns, f"{col_2y} not in dataframe columns"
    # Calculate CNR
    df[f'{cur}_CNR'] = (df[col_3m2y] - df[col_2y]) * 4 / 252
    # Use calculate_metrics to perform the remaining calculations
    df = calculate_metrics(df=df, col=f'{cur}_CNR', cur=cur, prefix='CNR', hl_smooth=hl_smooth, hl_lookback=hl_lookback, z_score_lower=z_score_lower, z_score_upper=z_score_upper)
    return df

def calculate_metrics_for_all_currencies(df, grouped_columns, prefix, hl_lookback, hl_smooth, z_score_lower, z_score_upper):
    # Create a dictionary where the keys are the currencies and the values are the column names
    columns_dict = {}
    for col in grouped_columns['_' + prefix]:
        currency, suffix = col.split('_')
        if currency not in columns_dict:
            columns_dict[currency] = None
        columns_dict[currency] = col

    # Iterate over the dictionary and call calculate_tot_metrics for each column with the same currency
    for currency, col in columns_dict.items():
        if col is not None:
            df = calculate_metrics(df=df, col=f'{currency}_{prefix}', cur=currency, prefix=prefix, hl_smooth=hl_smooth, hl_lookback=hl_lookback, z_score_lower=z_score_lower, z_score_upper=z_score_upper)
    return df

def calculate_msci_metrics(df, cols_msci, cols_fx, hl_lookback, hl_smoothed, z_score_lower, z_score_upper):
    # Create a dictionary mapping currencies to their corresponding columns in cols_fx
    fx_dict = {col.split('_')[0]: col for col in cols_fx}

    for col in cols_msci:
        currency = col.split('_')[0]
        if currency in fx_dict:
            col2 = fx_dict[currency]
            msci_lccy = f'{currency}_MSCI_LCCY'
            msci_lccy_smoothed = f'{currency}_MSCI_LCCY_SMOOTHED'
            msci_z_score = f'{currency}_MSCI_Z_SCORE'
            msci_z_score_capped = f'{currency}_MSCI_Z_SCORE_CAPPED'
            msci_std = f'{currency}_MSCI_STD'
            msci_mean = f'{currency}_MSCI_MEAN'

            df[msci_lccy] = df[col] * df[col2]
            df[msci_lccy_smoothed] = df[msci_lccy].ewm(halflife=hl_smoothed).mean()
            df[msci_mean] = df[msci_lccy_smoothed].ewm(halflife=hl_lookback).mean()
            df[msci_std] = df[msci_lccy_smoothed].ewm(halflife=hl_lookback).std()
            df[msci_z_score] = (df[msci_lccy_smoothed] - df[msci_mean]) / df[msci_std]
            df[msci_z_score_capped] = np.clip(df[msci_z_score], z_score_lower, z_score_upper)
    return df

def calculate_iv_metrics(df, cols_iv, hl_mean, z_score_lower, z_score_upper):
    for col in cols_iv:
        currency = col.split('_')[0]
        iv = f'{currency}_IV'
        iv_smoothed = f'{currency}_IV_SMOOTHED'
        iv_z_score = f'{currency}_IV_Z_SCORE'
        iv_z_score_capped = f'{currency}_IV_Z_SCORE_CAPPED'
        iv_std = f'{currency}_IV_STD'
        iv_mean = f'{currency}_IV_MEAN'

        df[iv_smoothed] = df[iv].ewm(halflife=4).mean()
        df[iv_mean] = df[iv_smoothed].ewm(halflife=hl_mean).mean()
        df[iv_std] = df[iv_smoothed].ewm(halflife=hl_mean).std()
        df[iv_z_score] = (df[iv_smoothed] - df[iv_mean]) / df[iv_std]
        df[iv_z_score_capped] = np.clip(df[iv_z_score], z_score_lower, z_score_upper)
    return df

def calculate_returns(df, cols_3m2y, rolling_window=5, hl_lookback=63):
    df_rolling_5d_tr = pd.DataFrame(index=df.index)
    # Calculate daily price changes, daily total returns, and cumulative returns
    for col in cols_3m2y:
        cur = col.split('_')[0]
        daily_total_return = f"{cur}_1D_TR"
        daily_price_change = f"{cur}_1D_CHG"
        cum_total_return = f"{cur}_CUM_TR"
        rolling_total_return = f"{cur}_{rolling_window}D_TR"
        rolling_tr_std = f"{cur}_TR_STD"
        
        if df.get(daily_price_change) is None:
            df[daily_price_change] = df[col].diff()
        
        if df.get(daily_total_return) is None:
            df[daily_total_return] = df[f"{cur}_CNR"] - df[daily_price_change]
        
        df[rolling_total_return] = df[daily_total_return].rolling(window=rolling_window).sum()
        df[rolling_tr_std] = df[rolling_total_return].ewm(halflife=hl_lookback).std() * np.sqrt(252/rolling_window) * 100
        df[cum_total_return] = df[daily_total_return].cumsum()
        df_rolling_5d_tr[rolling_total_return] = df[rolling_total_return].copy()
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
        print(f"Optimization failed: {e}")
        return None
    return weights.value

def print_results(df):
    print(df['CLP_CNR_Z_SCORE_CAPPED'].iloc[0])
    print(df['CLP_CNR_Z_SCORE_CAPPED'].iloc[-1])
    print(df['CLP_ToT_Z_SCORE_CAPPED'].iloc[0])
    print(df['CLP_ToT_Z_SCORE_CAPPED'].iloc[-1])
    print(df['CLP_3M2Y_Z_SCORE_CAPPED'].iloc[0])
    print(df['CLP_3M2Y_Z_SCORE_CAPPED'].iloc[-1])
    print(df['CLP_USD_Z_SCORE_CAPPED'].iloc[0])
    print(df['CLP_USD_Z_SCORE_CAPPED'].iloc[-1])
    print(df['CLP_MSCI_Z_SCORE_CAPPED'].iloc[0])
    print(df['CLP_MSCI_Z_SCORE_CAPPED'].iloc[-1])
    print(df['CLP_IV_Z_SCORE_CAPPED'].iloc[0])
    print(df['CLP_IV_Z_SCORE_CAPPED'].iloc[-1])
    # print(df['CLP_CUM_TR'])
    
    