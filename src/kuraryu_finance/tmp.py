import time
import pandas as pd
import numpy as np
import ccxt
import mplfinance as mpf
import matplotlib.pyplot as plt

def get_datas():
    symbol = 'BTC/USDT'  # 通常はこちら
    timeframe = '4h'
    limit = 6000 #1500
    MONTH=12
    current_time = int(time.time() * 1000)  # 秒→ミリ秒
    start_time = current_time - 60*60*24*30*MONTH*1000  # 約90日前

    ex = ccxt.binanceusdm()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=start_time, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["unixtime", "open", "high", "low", "close", "volume"])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df['unixtime'] = df['unixtime'].astype(int)
    df['time'] = pd.to_datetime(df['unixtime'], unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df


def triple_barrier_method(df, pt=0.05, sl=0.05, max_holding_period=5):
    """
    df: OHLCV DataFrame with DatetimeIndex and 'close' column
    pt: Take Profit (percentage)
    sl: Stop Loss (percentage)
    max_holding_period: max days to hold
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have DatetimeIndex")

    labels = []
    closes = df['close'].values
    n = len(df)

    for i in range(n - max_holding_period):
        start_price = closes[i]
        upper_barrier = start_price * (1 + pt)
        lower_barrier = start_price * (1 - sl)
        window = closes[i+1:i+1+max_holding_period]

        label = 0  # default: hold to expiry
        for price in window:
            if price >= upper_barrier:
                label = 1
                break
            elif price <= lower_barrier:
                label = -1
                break
        labels.append(label)

    # 残りの日はNaNで埋める
    labels += [np.nan] * max_holding_period

    df_result = df.copy()
    df_result['label'] = labels
    return df_result


def plot_triple_barrier_candlestick(df, pt=0.05, sl=0.05, max_holding_period=5, num_events=50):
    import numpy as np
    import mplfinance as mpf

    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot.index)

    df_plot.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    df_plot = df_plot[['Open', 'High', 'Low', 'Close', 'Volume']]

    apds = []
    barrier_columns = {}

    # マーカーとバリアを作成
    for i in range(min(num_events, len(df) - max_holding_period)):
        label = df['label'].iloc[i]
        if pd.isna(label):
            continue

        start_price = df['close'].iloc[i]
        upper = start_price * (1 + pt)
        lower = start_price * (1 - sl)

        # マーカー用データ（df_plotのサイズに合わせる）
        marker_data = [np.nan] * len(df_plot)
        marker_data[i] = start_price

        color, marker = ('green', '^') if label == 1 else ('red', 'v') if label == -1 else ('blue', 'o')
        apds.append(mpf.make_addplot(marker_data, scatter=True, markersize=100, marker=marker, color=color))

        # バリアデータを格納
        for j in range(i + 1, min(i + 1 + max_holding_period, len(df_plot))):
            ts = df_plot.index[j]
            barrier_columns.setdefault(f'Upper_{i}', {})[ts] = upper
            barrier_columns.setdefault(f'Lower_{i}', {})[ts] = lower

    # バリアラインをDataFrame化して結合
    if barrier_columns:
        barrier_df = pd.DataFrame(barrier_columns)
        barrier_df = barrier_df.reindex(df_plot.index)  # 念のためindex揃え
        df_plot = pd.concat([df_plot, barrier_df], axis=1)
        for col in barrier_df.columns:
            apds.append(mpf.make_addplot(df_plot[col], color='gray', linestyle='dotted'))

    # 表示範囲を設定（最初のイベントの開始日時から最後のイベントの終わりまで）
    start_idx = 0
    end_idx = min(num_events + max_holding_period, len(df_plot) - 1)
    xlim = (df_plot.index[start_idx], df_plot.index[end_idx])

    mpf.plot(df_plot,
             type='candle',
             style='charles',
             volume=True,
             addplot=apds,
             figsize=(16, 8),
             title=f'Triple Barrier Events (first {num_events} events only)',
             xlim=xlim)


def backtest_triple_barrier(df, max_holding_period=5):
    """
    df: triple_barrier_methodでラベル付け済みのDataFrame（datetime index、label列あり）
    max_holding_period: 最大保有期間（日）

    戻り値: DataFrameに「return」「holding_period」列を追加したもの
    """

    returns = []
    holding_periods = []

    for i in range(len(df) - max_holding_period):
        label = df['label'].iloc[i]
        entry_price = df['close'].iloc[i]

        if pd.isna(label) or label == 0:
            # 保持ラベルはエントリーなしとして0リターンでスキップも可
            returns.append(0)
            holding_periods.append(0)
            continue

        # 決済期間（最大保持期間まで）
        window = df['close'].iloc[i+1:i+1+max_holding_period]

        # 決済価格・決済日決定用変数
        exit_price = None
        exit_idx = None

        if label == 1:
            # 利確バリアに当たった最初の日の価格を探す
            upper_barrier = entry_price * (1 + PT)
            for j, price in enumerate(window):
                if price >= upper_barrier:
                    exit_price = price
                    exit_idx = i + 1 + j
                    break
        elif label == -1:
            # 損切りバリアに当たった最初の日の価格を探す
            lower_barrier = entry_price * (1 - SL)
            for j, price in enumerate(window):
                if price <= lower_barrier:
                    exit_price = price
                    exit_idx = i + 1 + j
                    break

        # バリアに当たらなかったら max_holding_period目の価格で決済
        if exit_price is None:
            exit_price = window.iloc[-1]
            exit_idx = i + max_holding_period

        # 保有期間（何日ポジション持ったか）
        holding_period = exit_idx - i

        # リターン計算（買いポジション）
        ret = (exit_price - entry_price) / entry_price * label  # labelが1なら買い、-1なら売りポジション

        returns.append(ret)
        holding_periods.append(holding_period)

    # 保持期間分のNaN埋め
    for _ in range(max_holding_period):
        returns.append(0)
        holding_periods.append(0)

    df_result = df.iloc[:len(df)].copy()
    df_result['return'] = returns
    df_result['holding_period'] = holding_periods

    return df_result

def backtest_with_capital(df, pt=0.05, sl=0.05, max_holding_period=5, initial_capital=100):
    capital = initial_capital
    capital_history = [capital]
    timestamps = [df.index[0]]

    for i in range(len(df) - max_holding_period):
        label = df['label'].iloc[i]
        entry_price = df['close'].iloc[i]

        if pd.isna(label) or label == 0:
            # トレードなし
            capital_history.append(capital)
            timestamps.append(df.index[i + 1])
            continue

        # 決済価格を探す
        window = df['close'].iloc[i+1:i+1+max_holding_period]
        exit_price = None
        exit_idx = None

        if label == 1:
            # 利確バリア
            for j, price in enumerate(window):
                if price >= entry_price * (1 + pt):
                    exit_price = price
                    exit_idx = i + 1 + j
                    break
        elif label == -1:
            # 損切りバリア
            for j, price in enumerate(window):
                if price <= entry_price * (1 - sl):
                    exit_price = price
                    exit_idx = i + 1 + j
                    break

        # バリアにかからなければ最大保持期間の最後で決済
        if exit_price is None:
            exit_price = window.iloc[-1]
            exit_idx = i + max_holding_period

        # リターン（買い or 売り）
        if label == 1:
            ret = (exit_price - entry_price) / entry_price
        else:
            ret = (entry_price - exit_price) / entry_price

        capital *= (1 + ret)
        capital_history.append(capital)
        timestamps.append(df.index[exit_idx])

    # DataFrame 化
    result_df = pd.DataFrame({
        'timestamp': timestamps,
        'capital': capital_history
    }).set_index('timestamp')

    return result_df

def backtest_with_trade_log(df, pt=0.05, sl=0.05, max_holding_period=5, initial_capital=100):
    capital = initial_capital
    capital_history = [capital]
    capital_timestamps = [df.index[0]]

    trades = []

    for i in range(len(df) - max_holding_period):
        label = df['label'].iloc[i]
        entry_time = df.index[i]
        entry_price = df['close'].iloc[i]

        if pd.isna(label) or label == 0:
            capital_history.append(capital)
            capital_timestamps.append(df.index[i + 1])
            continue

        window = df['close'].iloc[i+1:i+1+max_holding_period]
        exit_price = None
        exit_idx = None

        if label == 1:
            # Long
            for j, price in enumerate(window):
                if price >= entry_price * (1 + pt):
                    exit_price = price
                    exit_idx = i + 1 + j
                    break
        elif label == -1:
            # Short
            for j, price in enumerate(window):
                if price <= entry_price * (1 - sl):
                    exit_price = price
                    exit_idx = i + 1 + j
                    break

        if exit_price is None:
            exit_price = window.iloc[-1]
            exit_idx = i + max_holding_period

        exit_time = df.index[exit_idx]

        # Return
        if label == 1:
            direction = 'Long'
            ret = (exit_price - entry_price) / entry_price
        else:
            direction = 'Short'
            ret = (entry_price - exit_price) / entry_price

        capital *= (1 + ret)
        capital_history.append(capital)
        capital_timestamps.append(exit_time)

        # 取引記録を追加
        trades.append({
            'Entry time': entry_time,
            'Entry price': entry_price,
            'Exit time': exit_time,
            'Exit price': exit_price,
            'Direction': direction,
            'Return': round(ret * 100, 2)
        })

    # 資産推移データフレーム
    capital_df = pd.DataFrame({
        'timestamp': capital_timestamps,
        'capital': capital_history
    }).set_index('timestamp')

    # トレード履歴データフレーム
    trades_df = pd.DataFrame(trades)

    return capital_df, trades_df


if __name__ == "__main__":
    ohlcv = get_datas()

    PT = 0.05
    SL = 0.05
    MAX_HOLDING_PERIOD = 10

    labeled_df = triple_barrier_method(ohlcv, pt=PT, sl=SL, max_holding_period=MAX_HOLDING_PERIOD)

    print(labeled_df)
    labeled_df.to_csv("labeled_df.csv", index=False)

    plot_triple_barrier_candlestick(labeled_df, pt=PT, sl=SL, max_holding_period=MAX_HOLDING_PERIOD, num_events=50)

    # # バックテスト実行例
    # bt_result = backtest_triple_barrier(labeled_df, max_holding_period=MAX_HOLDING_PERIOD)

    # # 累積リターン（単純リターンの積み上げ）
    # bt_result['cumulative_return'] = (1 + bt_result['return']).cumprod()

    # print(bt_result)#[['return', 'holding_period', 'cumulative_return']])

    # # 可視化（累積リターン）
    # plt.figure(figsize=(12,6))
    # plt.plot(bt_result.index, bt_result['cumulative_return'])
    # plt.title('Cumulative Return of Triple Barrier Strategy')
    # plt.xlabel('Date')
    # plt.ylabel('Cumulative Return')
    # plt.grid(True)
    # plt.show()


    # バックテスト実行
    result = backtest_with_capital(
        labeled_df,
        pt=PT,
        sl=SL,
        max_holding_period=MAX_HOLDING_PERIOD,
        initial_capital=100
    )

    # 結果表示
    print(result)
    print(f"\nInitial Capital: $100")
    print(f"Final Capital: ${result['capital'].iloc[-1]:.2f}")
    print(f"Total Return: {((result['capital'].iloc[-1] / 100) - 1) * 100:.2f}%")

    # グラフ描画
    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result['capital'], label='Capital Over Time')
    plt.title('Capital Growth with Triple Barrier Strategy')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.legend()
    plt.show()


    capital_df, trades_df = backtest_with_trade_log(
        labeled_df,
        pt=PT,
        sl=SL,
        max_holding_period=MAX_HOLDING_PERIOD,
        initial_capital=100
    )

    # 資産推移を表示
    print(capital_df.tail())

    # トレード履歴を表示
    print(trades_df.head())

    # 最終結果
    print(f"\nInitial Capital: $100")
    print(f"Final Capital: ${capital_df['capital'].iloc[-1]:.2f}")
    print(f"Total Return: {((capital_df['capital'].iloc[-1] / 100) - 1) * 100:.2f}%")

    # 資産推移をプロット
    plt.figure(figsize=(12, 6))
    plt.plot(capital_df.index, capital_df['capital'], label='Capital Over Time')
    plt.title('Capital Growth with Triple Barrier Strategy')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.legend()
    plt.show()
