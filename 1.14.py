import tushare as ts
import pandas as pd
import numpy as np

ts.set_token("1720b38d64284be11ecf869292740df2a78339ff1c3060bef1b942f6")
pro = ts.pro_api()

def get_daily(ts_code: str, start: str="20230101", end: str="20260113") -> pd.DataFrame:
    df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
    # TuShare 返回一般是倒序，这里按日期升序
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    # 只留常用列
    df = df[["open","high","low","close","vol","amount"]]
    return df

df = get_daily("000981.SZ", start="20230101", end="20260113")
df.tail()
def ma_crossover_signals(df: pd.DataFrame, short=20, long=60) -> pd.DataFrame:
    out = df.copy()
    out["ma_s"] = out["close"].rolling(short).mean()
    out["ma_l"] = out["close"].rolling(long).mean()
    out["pos"] = (out["ma_s"] > out["ma_l"]).astype(int)  # 1=持仓, 0=空仓
    return out

sig = ma_crossover_signals(df, short=20, long=60)
sig[["close","ma_s","ma_l","pos"]].tail()
def backtest_long_only(df: pd.DataFrame, fee_buy=0.0003, fee_sell=0.0003, stamp_sell=0.0010):
    out = df.copy()

    # 标的日收益（用收盘到收盘）
    out["ret"] = out["close"].pct_change()

    # 持仓用昨天的pos（避免未来函数）
    out["pos_lag"] = out["pos"].shift(1).fillna(0)

    # 策略毛收益
    out["strat_gross"] = out["pos_lag"] * out["ret"]

    # 交易发生：pos 变化的那天（用昨天->今天）
    out["trade"] = out["pos"].diff().fillna(0)  # 1=买入开仓，-1=卖出平仓

    # 成本：买入收佣金；卖出收佣金+印花税
    out["cost"] = 0.0
    out.loc[out["trade"] == 1, "cost"] = fee_buy
    out.loc[out["trade"] == -1, "cost"] = (fee_sell + stamp_sell)

    # 策略净收益（用收益减成本，成本按“当日一次性扣除”简化）
    out["strat_net"] = out["strat_gross"] - out["cost"]

    # 净值曲线
    out["nav"] = (1 + out["strat_net"]).cumprod()
    out["bh_nav"] = (1 + out["ret"]).cumprod()  # 买入持有对照
    return out

bt = backtest_long_only(sig)
bt[["nav","bh_nav","strat_net","cost","trade"]].tail()
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

perf = performance(bt["nav"], bt["strat_net"])
perf_bh = performance(bt["bh_nav"], bt["ret"])

print(perf, perf_bh)
