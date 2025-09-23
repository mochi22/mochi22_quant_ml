import asyncio
import pybotters
from rich import print
import logging
from rich.logging import RichHandler
from datetime  import datetime
import pandas as pd

from datalake import save_to_s3

symbol="btcusdt"
range="1m"
# stream="trade"
stream="kline"
send_json={
    "method": "SUBSCRIBE",
    "params": {
        "returnRateLimits": True,
        # "auth": pybotters.Auth
    },
    "id": "10",
}

# for trade stream
key_map = {
    "e": "event_type",
    "E": "event_timestamp",
    "s": "symbol",
    "t": "trade_id",
    "p": "price",
    "q": "quantity",
    "T": "trade_timestamp",
    "m": "is_buyer_mm",
    "M": "ignore"
}

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(markup=True,rich_tracebacks=True),
        logging.FileHandler(filename=f"logs/log_{current_time}.txt", mode="a", encoding="utf-8")
    ]
)

def rename_keys(data, key_map):
    return { key_map.get(k, k): v for k, v in data.items() }

def check_change_day(current_day):
    """
    日付が変わったかどうかを判定する
    """
    today = datetime.now().date()
    if today == current_day:
        # 日付は変わってない
        return False
    else:
        # 日付が変わった
        return True
    

async def main():
    logging.info("start!!!")

    current_day=datetime.now().date()
    
    # 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    if stream=="kline":
        url=f"wss://stream.binance.com:9443/ws/{symbol}@{stream}_{range}"
    elif stream=="trade":
        url=f"wss://stream.binance.com:9443/ws/{symbol}@{stream}"
    logging.info(f"stream:{stream}, symbol:{symbol}, url:{url}")
    logging.info(f"send_json:{send_json}")
    df=pd.DataFrame()
    async with pybotters.Client() as client:
        wsqueue = pybotters.WebSocketQueue()

        await client.ws_connect(
            url,
            send_json=send_json,
            hdlr_json=wsqueue.onmessage,
        )
        # await client.ws_connect(
        #     "wss://ws.lightstream.bitflyer.com/json-rpc",
        #     send_json={
        #         "method": "subscribe",
        #         "params": {"channel": "lightning_ticker_BTC_JPY"},
        #         "id": 1,
        #     },
        #     hdlr_json=wsqueue.onmessage,
        # )

        async for msg in wsqueue:  # Ctrl+C to break
            if "error" in msg:
                pass
            else:
                if stream=="kline":
                    """kline
{
  "e": "kline",         // Event type
  "E": 1672515782136,   // Event time
  "s": "BNBBTC",        // Symbol
  "k": {
    "t": 1672515780000, // Kline start time
    "T": 1672515839999, // Kline close time
    "s": "BNBBTC",      // Symbol
    "i": "1m",          // Interval
    "f": 100,           // First trade ID
    "L": 200,           // Last trade ID
    "o": "0.0010",      // Open price
    "c": "0.0020",      // Close price
    "h": "0.0025",      // High price
    "l": "0.0015",      // Low price
    "v": "1000",        // Base asset volume
    "n": 100,           // Number of trades
    "x": false,         // Is this kline closed?
    "q": "1.0000",      // Quote asset volume
    "V": "500",         // Taker buy base asset volume
    "Q": "0.500",       // Taker buy quote asset volume
    "B": "123456"       // Ignore
  }
}
                    """
                    logging.info(msg)
                    event_time=datetime.fromtimestamp(msg["E"] / 1000)
                    content=msg["k"]
                    Kline_start_time=datetime.fromtimestamp(content["t"] / 1000)
                    Kline_close_time=datetime.fromtimestamp(content["T"] / 1000)
                    logging.info(f"event_time:{event_time}, Kline_start_time:{Kline_start_time}, Kline_close_time:{Kline_close_time}")
                elif stream=="trade":
                    msg=rename_keys(msg, key_map)
                    msg.pop("ignore", None) # delete the ignore key
                    logging.debug(msg)

                    msg=pd.DataFrame([msg])
                    df=pd.concat([df, msg], axis=0)
                    if df.shape[0]%2==0:
                        logging.info(f"df.shape: {df.shape}")
                        logging.info(df.tail())
                    # if check_change_day(current_day):
                        # partitionなどでdt infoは使用したい
                        # msg["event_dt"] = datetime.fromtimestamp(msg["event_timestamp"] / 1000)
                        df["dt"] = pd.to_datetime(df["event_timestamp"], unit="ms")
                        df['partition_dt'] = df['dt'].dt.floor('H')
                        # data量は減らした方がよい
                        df["event_minus_trade_timestamp"] = df["event_timestamp"] - df["trade_timestamp"]
                        logging.info(df["trade_timestamp"])
                        df=df.drop(["trade_timestamp"], axis=1)
                        # "partition_dt", "event_type", "event_timestamp", "symbol", "trade_id", "price", "quantity", "", "is_buyer_mm","event_dt", "event_minus_trade_timestamp"
                        
                        save_to_s3(
                            df=df,
                            symbol='BTCUSDT', 
                            s3_path=f's3://kuraryu-cex-test-bucket/binance', 
                            region_name="ap-northeast-1", 
                            glue_db_name='binance_trade_db', 
                            table_name='binance_trade_table',
                            reversed=True
                        )
                        df.to_csv(f"datas/df_{current_day}.csv", index=False)

                        # update process
                        df=pd.DataFrame()
                        current_day=datetime.now().date()
                    """
{
  "e": "trade",       // Event type
  "E": 1672515782136, // Event time
  "s": "BNBBTC",      // Symbol
  "t": 12345,         // Trade ID
  "p": "0.001",       // Price
  "q": "100",         // Quantity
  "T": 1672515782136, // Trade time
  "m": true,          // Is the buyer the market maker?
  "M": true           // Ignore
}
                    """
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
