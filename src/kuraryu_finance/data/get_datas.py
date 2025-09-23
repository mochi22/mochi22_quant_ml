import ccxt
import pandas as pd
import numpy as np
import ast
import json
import time
from typing import Optional, Union
from rich import print
from datetime import datetime
from dateutil.relativedelta import relativedelta

from utils.date import how_long_ago, get_unixtime

class BinanceDataFetcher:
    def __init__(self, symbol: str = "BTC/USDT"):
        self.exchange = ccxt.binance()
        # self.exchange.verbose = True # API requestも表示される
        self.symbol = symbol
        self.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    def check_all_method(self):
        print(self.exchange.has)
    def check_method(self, name):
        print(self.exchange.has[name])

    def fetch_trades(self, since: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """約定履歴を取得"""
        # fetch_tradesで取得されるtakerOrMakerは適当に入れてるだけらしい。なのでTaker volumeなどを算出することはCCXTを使用するだけだと無理。-> Binance APIもしくは、Pybottersで頑張る必要がありそう。
        trades = self.exchange.fetch_trades(self.symbol, since=since, limit=limit)
        df = pd.DataFrame(trades)
        return df

    def get_ohlcv(self, timeframe: str = "5m", since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """OHLCVデータ取得"""
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol, timeframe=timeframe, since=since, limit=limit
        )
        return pd.DataFrame(ohlcv, columns=self.columns)

    def get_funding_rate_history(self, since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """
        Funding Rate履歴取得
        binance: every 8 hour
        """
        fr = self.exchange.fetch_funding_rate_history(
            f"{self.symbol}:USDT", since=since, limit=limit
        )
        return pd.DataFrame(fr)

    def get_open_interest_history(self, timeframe: str = "5m", since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Open Interest履歴取得"""
        # binance: Only the data of the latest 1 month is available.直近一か月のみらしい。。。
        oi = self.exchange.fetch_open_interest_history(
            f"{self.symbol}:USDT", timeframe=timeframe, limit=limit,since=since #params={"startTime": since}
        )
        return pd.DataFrame(oi)

    def get_long_short_ratio_history(self, timeframe: str = "5m", since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        ls_ratio = self.exchange.fetch_long_short_ratio_history(
            f"{self.symbol}:USDT", timeframe=timeframe, since=since, limit=limit
        )
        return pd.DataFrame(ls_ratio)

    def get_volatility_history(self):
        # binanceだと、'fetchVolatilityHistory': False,なので使えないかも
        volatility = self.exchange.fetch_volatility_history(
            code="BTC"
        )
        return pd.DataFrame(volatility)

    def get_premium_index(self, timeframe: str = "5m", since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Premium Indexデータ取得"""
        premium = self.exchange.fetch_premium_index_ohlcv(
            self.symbol, timeframe=timeframe, since=since, limit=limit
        )
        return pd.DataFrame(premium, columns=self.columns)

    def get_mark_price(self, timeframe: str = "5m", since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Mark Priceデータ取得"""
        mark = self.exchange.fetch_mark_ohlcv(
            self.symbol, timeframe=timeframe, since=since, limit=limit
        )
        return pd.DataFrame(mark, columns=self.columns)

    def get_index_price(self, timeframe: str = "5m", since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Index Priceデータ取得"""
        index = self.exchange.fetch_index_ohlcv(
            self.symbol, timeframe=timeframe, since=since, limit=limit
        )
        return pd.DataFrame(index, columns=self.columns)

def create_features(df):
    # ==== timestampをdatetimeに変換 ====
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)

    df["daily_close_ratio"] =  df["close"] / df["close"].shift(1)
    df["daily_close_pct_change"] = df["close"].pct_change()
    df["log_daily_close_ratio"] = np.log(df["daily_close_ratio"])

    df["daily_open_close_ratio"] = df["close"] / df["open"]

    # ==== 一日ごとのボラ（絶対リターンの大きさ）====
    df["daily_volatility"] = df["daily_close_ratio"].rolling(window=window).std()
    df["log_daily_volatility"] = df["log_daily_close_ratio"].rolling(window=window).std()


    # ボラティリティ（標準偏差ベース）
    # volatility = df["log_day_return"].std() * np.sqrt(len(df))  # 年率換算するなら√Nを掛ける
    # print("直近データから算出したボラティリティ:", volatility)

    # 移動ボラティリティ（例: 10日ローリング標準偏差, 年率換算）。一般的にこれをボラと呼ぶのかなと思ってるが正しいか不明。
    # df['rolling_volatility'] = df['log_day_return'].rolling(window=3).std() * np.sqrt(252)

    df['rolling_20_annual_365_volatility'] = df['log_daily_close_ratio'].rolling(window=window).std() * np.sqrt(annual)
    df['rolling_20_annual_252_volatility'] = df['log_daily_close_ratio'].rolling(window=window).std() * np.sqrt(252)

    return df

def extract_value(s, col):
    try:
        s_clean = s.replace("'", '"')  # シングルクォート → ダブルクォート
        return float(json.loads(s_clean)[col])
    except Exception as e:
        print("Failed:", s)
        return None

def create_df(fetcher, init_since: int, timeframe="1d"):
    num_days = how_long_ago(init_since)

    ### OHLCV ###
    # 1000件がMax。強制的に1000件しか取得できない
    if num_days>ohlcv_limit:
        start_since=init_since
        df=pd.DataFrame()
        for i in range((num_days//ohlcv_limit)+1):
            print(i, start_since)
            tmp_df = fetcher.get_ohlcv(
                timeframe=timeframe, 
                since=start_since, 
                limit=ohlcv_limit
            )
            tmp_df.to_csv(f"datas/tmp_datas/ohlcv/ohlcv{i}.csv", index=False)

            # 取得してきたやつを縦方向にconcat
            df=pd.concat([df, tmp_df], axis=0)

            start_since=int(tmp_df.iloc[-1].timestamp)+1
            time.sleep(0.1)
    else:
        df = fetcher.get_ohlcv(
            timeframe=timeframe,
            limit=ohlcv_limit
        )
    df.to_csv("datas/ohlcv.csv", index=False)

    ### FR ###
    # frは一回で200件しか取れない。1000をちょい超えるとエラーになる。なのでlimitに合わせて繰り返し取得する
    if num_days>fr_limit:
        start_since=init_since
        fr=pd.DataFrame()
        # 8hごとで1dに対応してるコードなので、3倍してる。なんか+2だとうまくいく(頭が回ってない)。たまたまかわからんがちょうど+3だと行きすぎる
        for i in range(3*(num_days//fr_limit)+2):
            print(i, start_since)
            try:
                # timeframe 引数がない
                tmp_fr = fetcher.get_funding_rate_history(
                    since=start_since, 
                    limit=fr_limit
                )
            except Exception as e:
                print("ERROR!!!")
                print(i, e)
                pass
            # timestampで下二桁のずれが結構ある。msだしもっとずれを補正しても問題なさそうではある。
            tmp_fr["timestamp"] = (tmp_fr["timestamp"] // 100) * 100
            tmp_fr["markPrice"] = tmp_fr["info"].apply(lambda x: ast.literal_eval(str(x))["markPrice"])
            tmp_fr.to_csv(f"datas/tmp_datas/fr/fr{i}.csv", index=False)

            # 取得してきたやつを縦方向にconcat
            fr=pd.concat([fr, tmp_fr], axis=0)
            # 重複はないようにするため+1
            start_since=int(tmp_fr.iloc[-1].timestamp)+1
            time.sleep(0.1)
    else:
        tmp_fr = fetcher.get_funding_rate_history(
            # since=start_since,
            limit=fr_limit
        )
    fr.to_csv("datas/fr.csv", index=False)
    fr = fr.drop(["info", "datetime", "symbol"], axis=1)

    ### OI ###
    # oiのbaseVolume,quoteVolumeはdeprecated
    # binanceは直近一か月のみ。
    oi = fetcher.get_open_interest_history(
                    timeframe=timeframe, 
                    limit=oi_limit
                )
    oi["CMCCirculatingSupply"] = oi["info"].apply(
        lambda x: ast.literal_eval(str(x))["CMCCirculatingSupply"]
    )
    oi.to_csv("datas/oi.csv", index=False)
    oi = oi.drop(
        ["info", "datetime", "symbol", "baseVolume","quoteVolume"], 
        axis=1
    )
    
    df = df.merge(fr, on="timestamp", how="left", suffixes = ["_left", "_fr"])
    df = df.merge(oi, on="timestamp", how="left", suffixes = ["_left2", "_oi"])

    gomi=df[df["fundingRate"].isna()]
    print(gomi)
    gomi.to_csv("gomi.csv", index=False)
    print(f"fr:{fr.shape}, oi:{oi.shape}, last df:{df.shape}")
    return df


window = 10 # binanceのclose to close volatilityのperiodが10
annual = 365 # 株式とかだと252. binance見た感じ252かも
timeframe="1m"
limit=10000
fr_limit=200
oi_limit=50
ohlcv_limit=1000

fetcher = BinanceDataFetcher(symbol="BTC/USDT")

print("=== create df ===")
# ログリターンを計算
# df = fetcher.get_ohlcv(timeframe="1d",limit=500)
init_unixtime=get_unixtime(years=3)
df=create_df(fetcher, init_since=init_unixtime, timeframe=timeframe)
df=create_features(df)

print(df.tail())
print(df.info())
print(df.shape)
df.to_csv(f"datas/df_{timeframe}_{limit}limit.csv")

print("=== ALL DONE!!! ===")

# # ===== 使い方例 =====
# if __name__ == "__main__":
#     fetcher = BinanceDataFetcher(symbol="BTC/USDT")

#     print("=== OHLCV ===")
#     print(fetcher.get_ohlcv(limit=5).head())

#     print("=== Funding Rate ===")
#     print(fetcher.get_funding_rate(limit=5).head())

#     print("=== Open Interest ===")
#     print(fetcher.get_open_interest(limit=5).head())

#     print("=== Premium Index ===")
#     print(fetcher.get_premium_index(limit=5).head())


# ---

# # print (ccxt.exchanges)
# exchange = ccxt.binance()
# # print(dir(exchange)) # implicite api methods

# # ロードマーケット：取引所の市場シンボルのリスト等の情報を取得
# markets = exchange.load_markets()
# print(list(markets.keys())[:5]) # output a short list of market symbols

# # hyparameters
# limit = 100 
# symbol='BTC/USDT'
# columns=["timestamp","open", "high", "low", "close", "volume"]

# # 取引履歴の取得
# trades = exchange.fetch_trades(symbol, limit=limit)
# trades = pd.DataFrame(trades)
# print(trades.head())

# print(trades.columns)
# print(trades.iloc[0])
# print(trades.iloc[0]["info"])

# ### その他
# # tick = exchange.fetch_ticker(symbol) # 取得できた
# # print(tick)

# # print(exchange.fetch_balance()) # api keyが必須っぽい

# ### OHLCV
# print("==="*10, "OHLCV", "==="*10)
# ohlcv = pd.DataFrame(
#     exchange.fetch_ohlcv(
#         symbol, timeframe="5m", since=1756388604000, limit=10
#     ), 
#     columns=columns
# )
# print(ohlcv)

# """
# paramsとして、keyにprice、valueに下記を指定することでそれぞれを取得できる。
# 'mark'
# 'index'
# 'premiumIndex'
# """
# mark_klines = exchange.fetch_mark_ohlcv(symbol, '1h', limit=10)
# index_klines = exchange.fetch_index_ohlcv(symbol, '1h', limit=10)
# premium_klines = exchange.fetch_premium_index_ohlcv(symbol, '1h', limit=10)
# # from pprint import pprint
# # pprint(mark_klines)
# # pprint(index_klines)
# # pprint(premium_klines)
# premium_klines = pd.DataFrame(
#     premium_klines, 
#     columns=columns
# )
# print("premiumIndex")
# print(premium_klines)

# ### OI(Open Interest)
# # print("==="*10, "OI", "==="*10)
# # # print(exchange.fetch_open_interest("BTC/USDT:USDT")) # 取得できた。
# # oi = pd.DataFrame(
# #     exchange.fetch_open_interest_history("BTC/USDT:USDT", timeframe = '5m', limit=10 , params={"startTime": 1756388804000})
# # ) #, "endTime": 1756388814000})) # startTimeとかはmili secで指定する。もしくはparamsのとこはsince=時間、の形でもいける

# # print(oi)#since = undefined, limit = undefined))


# ### FR(Funding Rate)
# # print("==="*10, "FR", "==="*10)
# # fr=pd.DataFrame(
# #     exchange.fetch_funding_rate_history("BTC/USDT:USDT", since = 1756388604000, limit = 10)
# # )
# # print(fr)
