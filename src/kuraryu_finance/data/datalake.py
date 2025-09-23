

import athena_timeseries
import pandas as pd
import numpy as np
import boto3

symbol='BTCUSDT'

s3_path='s3://kuraryu-cex-test-bucket/example_db_directory'
region_name="ap-northeast-1"
glue_db_name='example_db'
table_name='example_table'

def save_to_s3(
        df,
        symbol='BTCUSDT', 
        s3_path='s3://kuraryu-cex-test-bucket/example_db_directory', 
        region_name="ap-northeast-1", 
        glue_db_name='example_db', 
        table_name='example_table',
        reversed=False
        ):
    boto3_session = boto3.Session(region_name=region_name)

    tsdb = athena_timeseries.AthenaTimeSeries(
        boto3_session=boto3_session, 
        glue_db_name=glue_db_name, 
        s3_path=s3_path,
    )

    # # Prepare example data, your data need to have 3 columns named symbol, dt, partition_dt
    # df = pd.DataFrame(np.random.randn(5000, 4))

    # df.columns = ['open', 'high', 'low', 'close']

    # # symbol represent a group of data for given data columns
    # df['symbol'] = symbol

    # # timestamp should be UTC timezone but without tz info
    # df['dt'] = pd.date_range('2022-01-01', '2022-05-01', freq='15Min')[:5000]
    # print(df)

    # # partition_dt must be date, data will be updated partition by partition with use of this column.
    # # Every time, you have to upload all the data for a given partition_dt, otherwise older will be gone.
    # #日次のパーティションなら下記が正しい
    # df['partition_dt'] = df['dt'].dt.date#.map(lambda x: x.replace(day=1))
    # print(df)

    # reveresed make change the partition from `partition_df/symbol` to `symbol/partition_dt`
    tsdb.upload(table_name=table_name, df=df, reversed=reversed)




# ========

# # Query for raw data.
# raw_close = tsdb.query(
#     table_name='example_table',
#     field='close',
#     start_dt='2022-02-01 00:00:00', # yyyy-mm-dd HH:MM:SS, inclusive
#     end_dt='2022-02-05 23:59:59', # yyyy-mm-dd HH:MM:SS, inclusive
#     symbols=['BTCUSDT'],
# )

# # Query for raw data with resampling
# resampeld_daily_close = tsdb.resample_query(
#     table_name='example_table',
#     field='close',
#     start_dt='2022-01-01 00:00:00', # yyyy-mm-dd HH:MM:SS, inclusive
#     end_dt='2022-01-31 23:59:59', # yyyy-mm-dd HH:MM:SS, inclusive
#     symbols=['BTCUSDT'],
#     interval='day', # month | week | day | hour | {1,2,3,4,6,8,12}hour | minute | {5,15,30}minute
#     op='last', # last | first | min | max | sum
# )
