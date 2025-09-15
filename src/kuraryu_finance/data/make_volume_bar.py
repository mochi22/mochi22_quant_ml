import pandas as pd
import polars as pl

def make_volume_bars(df, threshold):
    """
    OHLCVデータからVolume Barを作成
    
    Parameters
    ----------
    df : pd.DataFrame
        ['timestamp', 'open', 'high', 'low', 'close', 'volume'] を含む
    threshold : float
        1バーあたりの出来高しきい値
    
    Returns
    -------
    bars : pd.DataFrame
        Volume Bars (OHLCV形式)
    """

    # 累積出来高
    df = df.with_columns(
        pl.col("volume").cum_sum().alias("cum_volume")
    )

    print(df)
    df.to_csv("gomi.csv")

df=pl.read_csv("datas/df_1h_10000limit.csv")
print(df)
make_volume_bars(df, 10)