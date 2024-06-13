import pandas as pd
import numpy as np
import cvxpy as cp
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#import plotly.graph_objects as go
import plotly.io as pio



def load_raw_dataframe(file_path):
    return pd.read_csv(file_path)

def set_bbg_ticker_mapping(path):
    df_bbg = pd.read_excel(path, index_col=0)
    return {key: [value['Currency'], value['Suffix']] for key, value in df_bbg.to_dict(orient='index').items()}

def create_ticker_mapping(bbg_tickers):
    return {key: bbg_tickers[key][0] + bbg_tickers[key][1] for key in bbg_tickers}

def generate_new_headers(df_raw, header_row, ticker_mapping):
    bbg_headers = df_raw.iloc[header_row].tolist()
    return ["Dates"] + [ticker_mapping.get(str(ticker).split()[0], str(ticker).split()[0]) for ticker in bbg_headers if pd.notnull(ticker)]

def load_dataframe_with_new_headers(file_path, skip_rows, new_headers):
    df = pd.read_csv(file_path, skiprows=skip_rows)
    df.columns = new_headers
    return df

def preprocess_dataframe(df):
    df = df.replace('#NAME?', np.nan).bfill().ffill()
    df["Dates"] = pd.to_datetime(df["Dates"])
    df.set_index("Dates", inplace=True)
    df.sort_index(ascending=True, inplace=True)
    return df

def group_columns_by_suffix(df, suffixes):
    return {suffix: [col for col in df.columns if col.endswith(suffix)] for suffix in suffixes}


def calculate_cnr_metrics(df, col_3m2y, col_2y, currency, hl_smooth=5, hl_lookback=174):
    daily_cnr = f"{currency}_CNR"
    daily_cnr_smoothed = f"{currency}_CNR_SMOOTHED"
    cnr_z_score = f"{currency}_CNR_Z_SCORE"
    cnr_z_score_capped = f"{currency}_CNR_Z_SCORE_CAPPED"
    cnr_std = f"{currency}_CNR_STD"
    cnr_mean = f"{currency}_CNR_MEAN"
    
    # one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
    # trading_days_past_year = len(df.loc[df.index >= one_year_ago])
    
    df[daily_cnr] = (df[col_3m2y] - df[col_2y]) * 4 / 252
    df[daily_cnr_smoothed] = df[daily_cnr].ewm(halflife=5).mean()
    df[cnr_mean] = df[daily_cnr_smoothed].ewm(halflife=hl_lookback).mean()
    df[cnr_std] = df[daily_cnr_smoothed].ewm(halflife=hl_lookback).std()
    df[cnr_z_score] = (df[daily_cnr_smoothed] - df[cnr_mean]) / df[cnr_std]
    df[cnr_z_score_capped] = df[cnr_z_score].clip(lower=-4, upper=4)


def tot_to_z_score(df, cols_tot, hl_lookback):
    for col in cols_tot:
        currency = col.split('_')[0]
        tot = f'{currency}_ToT'
        tot_smoothed = f'{currency}_ToT_SMOOTHED'
        tot_z_score = f'{currency}_ToT_Z_SCORE'
        tot_z_score_capped = f'{currency}_ToT_Z_SCORE_CAPPED'
        tot_std = f'{currency}_ToT_STD'
        tot_mean = f'{currency}_ToT_MEAN'
        
        df[tot_smoothed] = df[tot].ewm(halflife=5).mean()
        df[tot_mean] = df[tot_smoothed].ewm(halflife=hl_lookback).mean()
        df[tot_std] = df[tot_smoothed].ewm(halflife=hl_lookback).std()
        df[tot_z_score] = (df[tot_smoothed] - df[tot_mean]) / df[tot_std]
        df[tot_z_score_capped] = df[tot_z_score].clip(lower=-4, upper=4)
    return df

def process_mom_columns(df, cols_3m2y, hl_mean=63):
    for col in cols_3m2y:
        currency = col.split('_')[0]
        mom = f'{currency}_3M2Y'
        mom_smoothed = f'{currency}_3M2Y_SMOOTHED'
        mom_z_score = f'{currency}_3M2Y_Z_SCORE'
        mom_z_score_capped = f'{currency}_3M2Y_Z_SCORE_CAPPED'
        mom_std = f'{currency}_3M2Y_STD'
        mom_mean = f'{currency}_3M2Y_MEAN'

        df[mom_smoothed] = df[mom].ewm(halflife=5).mean()
        df[mom_mean] = df[mom_smoothed].ewm(halflife=hl_mean).mean()
        df[mom_std] = df[mom_smoothed].ewm(halflife=hl_mean).std()
        df[mom_z_score] = (df[mom_smoothed] - df[mom_mean]) / df[mom_std]
        df[mom_z_score_capped] = df[mom_z_score].clip(lower=-3, upper=3)
    return df

def process_fx_columns(df, cols_fx, hl_mean=63):
    for col in cols_fx:
        currency = col.split('_')[0]
        fx = f'{currency}_USD'
        fx_smoothed = f'{currency}_FX_SMOOTHED'
        fx_z_score = f'{currency}_FX_Z_SCORE'
        fx_z_score_capped = f'{currency}_FX_Z_SCORE_CAPPED'
        fx_std = f'{currency}_FX_STD'
        fx_mean = f'{currency}_FX_MEAN'

        df[fx_smoothed] = df[fx].ewm(halflife=5).mean()
        df[fx_mean] = df[fx_smoothed].ewm(halflife=hl_mean).mean()
        df[fx_std] = df[fx_smoothed].ewm(halflife=hl_mean).std()
        df[fx_z_score] = (df[fx_smoothed] - df[fx_mean]) / df[fx_std]
        df[fx_z_score_capped] = df[fx_z_score].clip(lower=-4, upper=4)
    return df

def process_msci_columns(df, cols_msci, cols_fx, hl_mean=63):
    for col in cols_msci:
        for col2 in cols_fx:
            currency = col.split('_')[0]
            currency2 = col2.split('_')[0]
            if currency == currency2:
                msci_lccy = f'{currency}_MSCI_LCCY'
                msci_lccy_smoothed = f'{currency}_MSCI_LCCY_SMOOTHED'
                msci_z_score = f'{currency}_MSCI_Z_SCORE'
                msci_z_score_capped = f'{currency}_MSCI_Z_SCORE_CAPPED'
                msci_std = f'{currency}_MSCI_STD'
                msci_mean = f'{currency}_MSCI_MEAN'

                df[msci_lccy] = df[col] * df[col2]
                df[msci_lccy_smoothed] = df[msci_lccy].ewm(halflife=5).mean()
                df[msci_mean] = df[msci_lccy_smoothed].ewm(halflife=hl_mean).mean()
                df[msci_std] = df[msci_lccy_smoothed].ewm(halflife=hl_mean).std()
                df[msci_z_score] = (df[msci_lccy_smoothed] - df[msci_mean]) / df[msci_std]
                df[msci_z_score_capped] = df[msci_z_score].clip(lower=-4, upper=4)
    return df

def process_iv_columns(df, cols_iv, hl_mean=63):
    for col in cols_iv:
        currency = col.split('_')[0]
        iv = f'{currency}_IV'
        iv_smoothed = f'{currency}_IV_SMOOTHED'
        iv_z_score = f'{currency}_IV_Z_SCORE'
        iv_z_score_capped = f'{currency}_IV_Z_SCORE_CAPPED'
        iv_std = f'{currency}_IV_STD'
        iv_mean = f'{currency}_IV_MEAN'

        df[iv_smoothed] = df[iv].ewm(halflife=5).mean()
        df[iv_mean] = df[iv_smoothed].ewm(halflife=hl_mean).mean()
        df[iv_std] = df[iv_smoothed].ewm(halflife=hl_mean).std()
        df[iv_z_score] = (df[iv_smoothed] - df[iv_mean]) / df[iv_std]
        df[iv_z_score_capped] = df[iv_z_score].clip(lower=-4, upper=4)
    return df

def calculate_rolling_metrics(df, cols_3m2y):
    for col in cols_3m2y:
        cur = col.split('_')[0]
        df[f'{cur}_1D_CHG'] = df[col].diff()
        df[f'{cur}_1D_TR'] = df[cur + "_CNR"] - df[f'{cur}_1D_CHG']
        df[f'{cur}_5D_TR'] = df[f'{cur}_1D_TR'].rolling(window=5).sum()
        df[f'{cur}_TR_STD'] = df[f'{cur}_5D_TR'].ewm(halflife=63).std() * np.sqrt(252 / 5) * 100
        df[f'{cur}_CUM_TR'] = df[f'{cur}_1D_TR'].cumsum()



















def quadratic_utility_optimizer(expected_returns, corr_matrix, risk_aversion, portfolio_volatility):
    n = len(expected_returns)
    weights = cp.Variable(n)
    portfolio_return = weights @ expected_returns
    portfolio_variance = cp.quad_form(weights, corr_matrix)
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    constraints = [weights <= portfolio_volatility]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None
    return weights.value



def toggle_print(enabled):
    global PRINT_ENABLED
    PRINT_ENABLED = enabled

def toggle_plot(enabled):
    global PLOT_ENABLED
    PLOT_ENABLED = enabled


def calculate_rolling_returns(df, cols_3m2y):
    """Calculate rolling 5-day total returns and related metrics."""
    df_rolling_5d_tr = pd.DataFrame(index=df.index)
    for col in cols_3m2y:
        cur = col.split('_')[0]
        daily_total_return = f"{cur}_1D_TR"
        daily_price_change = f"{cur}_1D_CHG"
        cum_total_return = f"{cur}_CUM_TR"
        rolling_5d_total_return = f"{cur}_5D_TR"
        rolling_5d_tr_std = f"{cur}_TR_STD"
        
        if daily_price_change not in df.columns:
            df[daily_price_change] = df[col].diff()
        
        if daily_total_return not in df.columns:
            df[daily_total_return] = df[f"{cur}_CNR"] - df[daily_price_change]
        
        df[rolling_5d_total_return] = df[daily_total_return].rolling(window=5).sum()
        """~~~~~~~~~~Can make the 252 dynamic here~~~~~~~~~~"""
        df[rolling_5d_tr_std] = df[rolling_5d_total_return].ewm(halflife=63).std() * np.sqrt(252/5) * 100
        df[cum_total_return] = df[daily_total_return].cumsum()
        df_rolling_5d_tr[rolling_5d_total_return] = df[rolling_5d_total_return]
    
    return df, df_rolling_5d_tr

def plot_time_series(data, title='Time Series Data', xlabel='Date', ylabel='Value', legend_title='Series'):
    """Plot time series data."""
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")
    
    # Plot all columns in the DataFrame
    for column in data.columns:
        sns.lineplot(data=data, x=data.index, y=column, label=column)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, title=legend_title)
    plt.tight_layout()
    plt.show()

def plot_single_column(df, column_name, title=None, xlabel='Date', ylabel=None):
    """Plot a single column from a DataFrame over time with interactive features."""
    fig = px.line(df, x=df.index, y=column_name, title=title if title else f'Time Series of {column_name}')

    # Update layout for better appearance
    fig.update_layout(
        title={
            'text': title if title else f'Time Series of {column_name}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title=xlabel,
        yaxis_title=ylabel if ylabel else column_name,
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
        plot_bgcolor='white'
    )

    # Add hover mode
    fig.update_traces(mode='lines+markers', hovertemplate='%{y}')

    # Show the plot in a separate window
    pio.show(fig)

def calculate_cov_corr_matrices(df_rolling_5d_tr):
    """Calculate the exponentially weighted covariance and correlation matrices."""
    ewm_cov_matrix = df_rolling_5d_tr.ewm(halflife=126).cov()
    ewm_corr_matrix = df_rolling_5d_tr.ewm(halflife=126).corr()
    # Extract the last block of the matrix that corresponds to the rolling 5-day returns
    last_index = len(df_rolling_5d_tr.columns)
    ewm_cov_matrix = ewm_cov_matrix.iloc[-last_index:, -last_index:]
    ewm_corr_matrix = ewm_corr_matrix.iloc[-last_index:, -last_index:]
    
    # Verify that the covariance matrix is in the right format for CVXPY
    if isinstance(ewm_cov_matrix, pd.DataFrame):
        ewm_cov_matrix = ewm_cov_matrix.values
    
    # Verify that the correlation matrix is in the right format for CVXPY
    if isinstance(ewm_corr_matrix, pd.DataFrame):
        ewm_corr_matrix = ewm_corr_matrix.values
    
    return ewm_cov_matrix, ewm_corr_matrix

def define_expected_returns_and_metrics(df, cols_3m2y):
    """Define expected returns, currency names, and rolling 5-day total return standard deviations."""
    expected_returns = []
    currencies = []
    rolling_5d_tr_std_values = []

    for col in cols_3m2y:
        cur = col.split('_')[0]
        currencies.append(cur)
        expected_returns.append(df[cur + "_CNR_Z_SCORE_CAPPED"].iloc[-1])

        if cur + "_TR_STD" in df.columns:
            rolling_5d_tr_std_values.append(df[cur + "_TR_STD"].iloc[-1])
        else:
            rolling_5d_tr_std_values.append(np.nan)

    return expected_returns, currencies, rolling_5d_tr_std_values

def quadratic_utility_optimizer2(expected_returns, corr_matrix, risk_aversion, portfolio_volatility):
    """Optimize portfolio weights using quadratic utility optimization."""
    n = len(expected_returns)
    weights = cp.Variable(n)
    portfolio_return = weights @ expected_returns
    portfolio_variance = cp.quad_form(weights, corr_matrix)  # Quadratic form for variance
    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
    constraints = [
        weights <= portfolio_volatility  # Adding the constraint for maximum optimal risk weight based on pre-scaling portfolio volatility
    ]
    problem = cp.Problem(objective)
    problem.solve()
    return weights.value

def create_professional_colored_table(df):
    # Separate the numeric columns for normalization
    numeric_df = df.drop(columns=['Currency'])

    # Normalize the data for coloring
    norm_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the heatmap
    sns.heatmap(norm_df, annot=numeric_df, fmt='.2f', cmap='RdYlGn_r', ax=ax, cbar=True, linewidths=0.5, linecolor='black')

    # Set the currency column as the index for better visualization
    df.set_index('Currency', inplace=True)

    # Customize the appearance
    ax.set_title('Optimal Risk Weights and Related Metrics', fontsize=18, weight='bold')
    ax.set_xticklabels(['Optimal Risk Weight', 'Std Dev', 'Expected Return', 'Optimal DV01'], fontsize=12, weight='bold', rotation=0)
    ax.set_yticklabels(df.index, fontsize=12, weight='bold', rotation=0)
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

def create_ticker_mapping(bbg_tickers):
    return {key: bbg_tickers[key][0] + bbg_tickers[key][1] for key in bbg_tickers}


# def normalize_smoothed_cnr(df, number):
#     scaler = MinMaxScaler(feature_range=(-number, number))
#     for col in df.columns:
#         if col.endswith('_CNR_SMOOTHED'):
#             currency = col.split('_')[0]
#             cnr_level_score = f'{currency}_CNR_LEVEL_SCORE'
#             df[cnr_level_score] = scaler.fit_transform(df[col].values.reshape(-1, 1))
#     return df