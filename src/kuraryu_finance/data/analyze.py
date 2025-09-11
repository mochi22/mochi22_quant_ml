
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.plot_utils import adjust_row_domain, set_yaxis_domains
import plotly.express as px


# -----------------------------
# 設定
# -----------------------------
CSV_PATH = "datas/tmp_df.csv"
HTML_PATH = "graph_datas/analyze/histgram.html"

NUMBER_ROW = 4
ROW_HEIGHTS = [0.6, 0.2, 0.2, 0.2]
VERTICAL_SPACING = 0.1
# TARGET_ROW = 2
# SHIFT = -0.05

SUBPLOT_TITLES = [
    "OHLCV (Candlestick)", 
    "Close Ratio",
    "Rolling 10 annual 252 Volatility",
    "Rolling 10 annual 365 Volatility"
]

# -----------------------------
# データ読み込み
# -----------------------------
def load_data(path: str):
    df = pl.read_csv(path)
    
    df = df.with_columns(
        pl.col("timestamp").str.to_date("%Y-%m-%d")
    )
    print(df)
    return df

# -----------------------------
# Plotly グラフ作成
# -----------------------------

def display_histgram(df, col):
    fig = px.histogram(df, x=col)
    # fig.show()
    return fig


# -----------------------------
# メイン処理
# -----------------------------
if __name__ == "__main__":
    df = load_data(CSV_PATH)
    fig = display_histgram(df, "log_daily_close_ratio")
    fig.write_html(HTML_PATH, auto_open=True)

