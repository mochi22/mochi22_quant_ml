
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


# -----------------------------
# 設定
# -----------------------------
CSV_PATH = "datas/tmp_df.csv"
HTML_PATH = "graph_datas/ohlcv_volatility.html"

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
def create_ohlcv_volatility_plot(df):
    # サブプロット作成
    fig = make_subplots(
        rows=NUMBER_ROW, cols=1,
        shared_xaxes=True,
        vertical_spacing=VERTICAL_SPACING,
        row_heights=ROW_HEIGHTS,
        subplot_titles=SUBPLOT_TITLES
    )

    # ローソク足
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            # name="Candlestick",
        ),
        row=1, col=1
    )

    # volume
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            marker_color="blue",
            opacity=0.5,
            # name="Close Ratio"
        ),
        row=2, col=1
    )

    # ローリングボラティリティ
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["rolling_20_annual_252_volatility"]*100,
            mode="lines",
            line=dict(color="red"),
            # name="Rolling 20 annual 252 Volatility"
        ),
        row=3, col=1
    )

    # ローリングボラティリティ
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["rolling_20_annual_365_volatility"]*100,
            mode="lines",
            line=dict(color="red"),
            # name="Rolling 20 annual 365 Volatility"
        ),
        row=4, col=1
    )

    # レイアウト設定
    # fig.update_xaxes(
    #     type="date",
    #     tickformat="%Y/%m/%d",
    #     tickangle=30,
    #     # tickfont=dict(
    #     #     size=10,
    #     #     # color="red"
    #     # )
    # )
    fig.update_layout(
        title="OHLCV and Volatility",
        template="plotly_white",
        # xaxis_rangeslider_visible=False, # rangeslider setting
        xaxis1=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        ),
        xaxis=dict(
            type="date",
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]
            ),
            rangeslider=dict(visible=True)
        ),
        legend=dict(
            x=0,   # 0=左, 1=右
            y=1,   # 0=下, 1=上
            xanchor="left",
            yanchor="top"
        )
    )




    # # 特定行のドメイン調整はめんどうでやめた。
    # fig = adjust_row_domain(ROW_HEIGHTS, VERTICAL_SPACING, TARGET_ROW, SHIFT)

    return fig

# -----------------------------
# メイン処理
# -----------------------------
if __name__ == "__main__":
    df = load_data(CSV_PATH)
    fig = create_ohlcv_volatility_plot(df)
    fig.write_html(HTML_PATH, auto_open=True)

    df=pl.read_csv("datas/fr.csv")
    # df["tmp"]=pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    # df.to_csv("tmp.csv")
    print(df)





### mpf のみで描画　ちょい見づらいけど問題はない ###
# # ローソク足 (Candlestick)
# apds = [
#     # 日次ボラティリティを棒グラフで第2パネルに表示
#     mpf.make_addplot(df["daily_volatility"], type="bar", panel=1, color="blue", alpha=0.5, ylabel="Daily Volatility"),

#     # ローリングボラティリティを折れ線で第3パネルに表示
#     mpf.make_addplot(df["rolling_volatility"], type="line", panel=2, color="red", ylabel="Rolling Volatility")
# ]

# mpf.plot(
#     df,
#     type="candle",
#     style="yahoo",
#     addplot=apds,
#     volume=False,
#     datetime_format='%Y/%m/%d',
#     title="OHLCV and Volatility",
#     figscale=1.2,
#     figratio=(16, 9),
#     panel_ratios=(3, 1, 1)  # パネルの縦比率 (ローソク足, daily vol, rolling vol)
# )

# print("=== Funding Rate ===")
# df=fetcher.get_funding_rate_history(limit=5).head()
# print(df)
# print(df.iloc[0]["info"])

# print("=== Open Interest ===")
# df=fetcher.get_open_interest_history(timeframe="4h",limit=5).head()
# print(df)
# print(df.iloc[0]["info"])

# print("=== Premium Index ===")
# print(fetcher.get_premium_index(timeframe="4h",limit=5).head())

# print("=== Long short Ratio ===")
# df=fetcher.get_long_short_ratio_history(timeframe="4h",limit=5).head()
# print(df)
# print(df.iloc[0]["info"])
# # taker volume(sell, buy volume) # ccxt onlyだと無理
