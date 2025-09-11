from datetime import datetime
from dateutil.relativedelta import relativedelta

def how_long_ago(unixtime_ms: int) -> str:
    """
    ミリ秒のUnixTimeが現在から何年前・何か月前・何日前かを返す

    Parameters
    ----------
    unixtime_ms : int
        ミリ秒単位のUnixTime

    Returns
    -------
    str
        "X年前 Yか月前 Z日前" のような文字列
    """
    dt = datetime.fromtimestamp(unixtime_ms / 1000)
    now = datetime.now()

    # diff = relativedelta(now, dt)

    # parts = {}
    # if diff.years > 0:
    #     parts["years"]=diff.years
    # if diff.months > 0:
    #     parts["months"]=diff.months
    # if diff.days > 0:
    #     parts["days"]=diff.days
    # if diff.hours > 0:
    #     parts["hours"]=diff.hours

    # return parts

    diff = now - dt
    return diff.days  # 日数だけ返す


def get_unixtime(years=0, months=0, days=0, hours=0, millis=True):
    """
    n年前、nか月前のUnixTimeを返す関数
    下記のような入力も可能
    years=1, months=16

    Parameters
    ----------
    years : int
        遡る年数（例: 3なら3年前）
    months : int
        遡る月数（例: 6なら6か月前, 16なら16か月前）
    millis : bool
        Trueならミリ秒単位、Falseなら秒単位

    Returns
    -------
    int : UnixTime（秒 or ミリ秒）
    """
    now = datetime.now()
    target = now - relativedelta(years=years, months=months, days=days, hours=hours)
    timestamp = int(target.timestamp())
    return timestamp * 1000 if millis else timestamp