import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts


TOKEN = "1720b38d64284be11ecf869292740df2a78339ff1c3060bef1b942f6"
COMPONENTS_PATH = r"C:\Users\hamil\OneDrive\Desktop\新建 文本文档.txt"
TARGET_STOCKS = 100
WINDOW_DAYS = 20
SLEEP_SEC = 0.8
SENSITIVITY_TOTALS = [60, 80, 100]

SPRING_FESTIVAL_DATES = [
    "20100126",
    "20110203",
    "20120123",
    "20130210",
    "20140131",
    "20150219",
    "20160208",
    "20170128",
    "20180216",
    "20190205",
    "20200125",
    "20210212",
    "20220201",
    "20230122",
    "20240210",
]


def _require_token() -> str:
    return os.getenv("TUSHARE_TOKEN") or TOKEN


def _load_components(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError("Failed to read components file.")

    cols = {c: c for c in df.columns}
    for c in df.columns:
        lc = str(c).lower()
        if "代码" in c or lc in {"code", "ts_code"}:
            cols[c] = "code"
        elif "简称" in c or "名称" in c or lc == "name":
            cols[c] = "name"
        elif "行业" in c or lc == "industry":
            cols[c] = "industry"
        elif "市值" in c or lc in {"market_cap", "marketcap"}:
            cols[c] = "market_cap"
    df = df.rename(columns=cols)
    required = {"code", "industry", "market_cap"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError("Missing required columns in components file.")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["code", "industry", "market_cap"])
    return df


def _select_representative(df: pd.DataFrame, total: int) -> pd.DataFrame:
    industries = sorted(df["industry"].unique())
    if len(industries) > total:
        return df.sort_values("market_cap", ascending=False).head(total)

    per_ind = max(1, total // len(industries))
    selected = []
    for _, group in df.groupby("industry"):
        selected.append(
            group.sort_values("market_cap", ascending=False).head(per_ind)
        )
    selected_df = pd.concat(selected, ignore_index=True)
    if len(selected_df) >= total:
        return selected_df.sort_values("market_cap", ascending=False).head(total)

    remaining = total - len(selected_df)
    leftovers = df.merge(
        selected_df[["code"]], on="code", how="left", indicator=True
    )
    leftovers = leftovers[leftovers["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    extras = leftovers.sort_values("market_cap", ascending=False).head(remaining)
    return pd.concat([selected_df, extras], ignore_index=True)


def build_trade_days(prices: dict) -> pd.DatetimeIndex:
    dates = set()
    for series in prices.values():
        dates.update(series.index)
    if not dates:
        raise RuntimeError("No trade dates collected from price data.")
    return pd.DatetimeIndex(sorted(dates))


def fetch_daily_prices(
    pro: ts.pro_api, codes: list, start: str, end: str, sleep_sec: float
) -> dict:
    prices = {}
    start_time = time.time()
    total = len(codes)
    for idx, code in enumerate(codes, 1):
        time.sleep(sleep_sec)
        df = pro.daily(
            ts_code=code,
            start_date=start,
            end_date=end,
            fields="trade_date,close",
        )
        if df is None or df.empty:
            continue
        df = df.sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        prices[code] = df.set_index("trade_date")["close"]
        if idx % 5 == 0 or idx == total:
            elapsed = time.time() - start_time
            filled = int((idx / total) * 24)
            bar = "#" * filled + "-" * (24 - filled)
            print(
                f"\r[FETCH] |{bar}| {idx}/{total} {elapsed:.0f}s",
                end="",
                flush=True,
            )
    if total:
        print()
    return prices


def _window_return(close: pd.Series, days: pd.DatetimeIndex) -> float | None:
    segment = close.reindex(days)
    if segment.isna().any():
        return None
    rets = segment.pct_change().dropna()
    if rets.empty:
        return None
    return (1 + rets).prod() - 1


def _analyze_with_selection(
    prices: dict,
    trade_days: pd.DatetimeIndex,
    selected: pd.DataFrame,
    festival_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    weights = selected.set_index("code")["market_cap"].astype(float).to_dict()
    rows = []
    for dt in festival_dates:
        pos = trade_days.searchsorted(dt)
        pre_days = trade_days[max(0, pos - WINDOW_DAYS) : pos]
        post_days = trade_days[pos : pos + WINDOW_DAYS]
        if len(pre_days) < WINDOW_DAYS or len(post_days) < WINDOW_DAYS:
            continue

        pre_returns = []
        post_returns = []
        used_weights = []
        for code, close in prices.items():
            if code not in weights:
                continue
            pre_ret = _window_return(close, pre_days)
            post_ret = _window_return(close, post_days)
            if pre_ret is None or post_ret is None:
                continue
            pre_returns.append(pre_ret)
            post_returns.append(post_ret)
            used_weights.append(weights[code])

        if not pre_returns:
            continue
        weight_sum = sum(used_weights)
        pre_weighted = sum(
            r * w for r, w in zip(pre_returns, used_weights)
        ) / weight_sum
        post_weighted = sum(
            r * w for r, w in zip(post_returns, used_weights)
        ) / weight_sum
        pre_win = sum(w for r, w in zip(pre_returns, used_weights) if r > 0)
        post_win = sum(w for r, w in zip(post_returns, used_weights) if r > 0)
        rows.append(
            {
                "year": dt.year,
                "pre_return_mean": pre_weighted,
                "post_return_mean": post_weighted,
                "pre_win_rate": pre_win / weight_sum,
                "post_win_rate": post_win / weight_sum,
                "sample_count": len(pre_returns),
            }
        )
    if not rows:
        raise SystemExit("No valid windows found.")
    return pd.DataFrame(rows).set_index("year").sort_index()


def _print_summary(df: pd.DataFrame, label: str) -> None:
    summary = pd.DataFrame(
        {
            "mean_pre": [df["pre_return_mean"].mean()],
            "mean_post": [df["post_return_mean"].mean()],
            "win_rate_pre": [df["pre_win_rate"].mean()],
            "win_rate_post": [df["post_win_rate"].mean()],
        }
    )
    summary_cn = summary.rename(
        columns={
            "mean_pre": "节前平均收益",
            "mean_post": "节后平均收益",
            "win_rate_pre": "节前胜率",
            "win_rate_post": "节后胜率",
        }
    )
    print(f"\n汇总（{label}）")
    print(summary_cn.to_string(index=False, float_format=lambda x: f"{x:.2%}"))


def main() -> None:
    token = _require_token()
    ts.set_token(token)
    pro = ts.pro_api()

    comp = _load_components(COMPONENTS_PATH)
    festival_dates = pd.to_datetime(SPRING_FESTIVAL_DATES)
    min_date = festival_dates.min() - timedelta(days=60)
    max_date = festival_dates.max() + timedelta(days=60)

    max_total = max(SENSITIVITY_TOTALS + [TARGET_STOCKS])
    selected_max = _select_representative(comp, max_total)
    codes = sorted(selected_max["code"].unique().tolist())
    print(
        f"[INFO] selected {len(codes)} stocks from "
        f"{selected_max['industry'].nunique()} industries"
    )

    prices = fetch_daily_prices(
        pro,
        codes=codes,
        start=min_date.strftime("%Y%m%d"),
        end=max_date.strftime("%Y%m%d"),
        sleep_sec=SLEEP_SEC,
    )
    trade_days = build_trade_days(prices)

    results_by_total = {}
    for total in SENSITIVITY_TOTALS:
        selected = _select_representative(comp, total)
        df = _analyze_with_selection(
            prices=prices,
            trade_days=trade_days,
            selected=selected,
            festival_dates=festival_dates,
        )
        results_by_total[total] = df
        _print_summary(df, f"样本数={total}")

    df_main = results_by_total.get(TARGET_STOCKS)
    if df_main is None:
        df_main = results_by_total[max(results_by_total)]

    detail_cn = df_main.rename(
        columns={
            "pre_return_mean": "节前均值收益",
            "post_return_mean": "节后均值收益",
            "pre_win_rate": "节前胜率",
            "post_win_rate": "节后胜率",
            "sample_count": "样本数",
        }
    )
    print("\n明细（行业分层 + 市值加权）")
    print(detail_cn.to_string(float_format=lambda x: f"{x:.2%}"))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    df_plot = df_main[["pre_return_mean", "post_return_mean"]].rename(
        columns={
            "pre_return_mean": "节前均值收益",
            "post_return_mean": "节后均值收益",
        }
    )
    ax = df_plot.plot(
        kind="bar",
        figsize=(10, 5),
        title="春节前后20个交易日收益（行业分层+市值加权）",
    )
    ax.set_xlabel("年份")
    ax.set_ylabel("区间累计收益均值")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
