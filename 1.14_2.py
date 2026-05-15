import tushare as ts
import pandas as pd
import numpy as np

ts.set_token("1720b38d64284be11ecf869292740df2a78339ff1c3060bef1b942f6")
pro = ts.pro_api()
universe = [
    "600897.SH",
    "002594.SZ",
    "300750.SZ",
    "002230.SZ",
    "300058.SZ",
    "002131.SZ",
    "601088.SH",
    "600477.SH",
    "600797.SH",
    "600570.SH",
    "600118.SH",
    "600893.SH",
    # 继续加到 20~30 只更好
]

def get_daily(ts_code, start="20250901", end="20260113"):
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    if df is None or len(df) == 0:
        return None

    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")

    return df[["open", "high", "low", "close", "vol"]]


def get_many_daily(stock_list, start="20250901", end="20260113", min_len=60):
    data = {}
    for code in stock_list:
        try:
            df = get_daily(code, start, end)
            if df is not None and len(df) >= min_len:
                data[code] = df
        except Exception as e:
            print("skip", code, e)
    return data

data = get_many_daily(universe)
print("loaded:", len(data))
def add_features(df, short=20, long=60, mom=30):
    out = df.copy()
    out["ma_s"] = out["close"].rolling(short).mean()
    out["ma_l"] = out["close"].rolling(long).mean()
    out["signal"] = (out["ma_s"] > out["ma_l"]).astype(int)
    out["ret"] = out["close"].pct_change()
    out["mom"] = out["close"].pct_change(mom)  # 动量分数
    return out

signals = {code: add_features(df) for code, df in data.items()}

def portfolio_backtest(signals_dict, topn=5, min_hold=3):
    all_dates = sorted(set().union(*[df.index for df in signals_dict.values()]))

    port_ret = []
    hold_cnt = []

    for i in range(1, len(all_dates)):
        date = all_dates[i]
        prev = all_dates[i-1]

        candidates = []
        for code, df in signals_dict.items():
            if prev in df.index and date in df.index:
                if df.loc[prev, "signal"] == 1 and pd.notna(df.loc[prev, "mom"]):
                    candidates.append((code, float(df.loc[prev, "mom"])))

        # 动量排序选 TopN
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [c[0] for c in candidates[:topn]]

        # 最小持仓数过滤
        if len(selected) < min_hold:
            port_ret.append(0.0)
            hold_cnt.append(0)
            continue

        # 等权收益
        rets = []
        for code in selected:
            df = signals_dict[code]
            rets.append(df.loc[date, "ret"])

        port_ret.append(float(np.mean(rets)))
        hold_cnt.append(len(selected))

    port_ret = pd.Series(port_ret, index=all_dates[1:])
    nav = (1 + port_ret).cumprod()
    hold_cnt = pd.Series(hold_cnt, index=all_dates[1:], name="holdings")
    return port_ret, nav, hold_cnt

def performance(nav: pd.Series, daily_ret: pd.Series):
    daily_ret = daily_ret.dropna()
    nav = nav.dropna()

    ann_ret = nav.iloc[-1] ** (252/len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else np.nan

    # 最大回撤
    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = dd.min()

    return {
        "年化收益": float(ann_ret),
        "年化波动": float(ann_vol),
        "夏普": float(sharpe),
        "最大回撤": float(mdd),
    }
port_ret, port_nav, hold_cnt = portfolio_backtest(signals, topn=5, min_hold=3)
print(performance(port_nav, port_ret),hold_cnt.describe())