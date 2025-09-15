import pandas as pd

#簡略させた。同じ感じになるはず
def getDailyVol(df, span=100):
    df=df["close"]
    returns = df / df.shift(1) - 1
    vol = returns.ewm(span=span).std()
    return vol
# 書かれてる通りだと機能しない
# def getDailyVol(df, span=100):
#     df0 = df.index.searchsorted(df.index-pd.Timedelta(days=1))
#     print(df0)
#     df0=df0[df0>0]
#     print(df0)
#     df0=pd.Series(df.index[df0-1], index=df.index[df.shape[0]-df0.shape[0]:])

#     # daily return
#     df0=df.loc[df0.index] / df.loc[df0.values].values-1
#     df0=df0.ewm(span=span).std()
#     return df0

df=pd.read_csv("datas/df_1d_100limit.csv")
df.set_index('date', inplace=True)
print(df)
hist=getDailyVol(df)
print(hist)
hist.hist()