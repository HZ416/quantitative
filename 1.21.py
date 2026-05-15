import argparse
import time
import tkinter as tk
from itertools import product
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import tushare as ts

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False

TOKEN = "1720b38d64284be11ecf869292740df2a78339ff1c3060bef1b942f6"
ts.set_token(TOKEN)
pro = ts.pro_api()


def fetch_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = pro.daily(
        ts_code=ts_code,
        start_date=start,
        end_date=end,
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.sort_values("trade_date")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    return df[["open", "high", "low", "close", "vol", "amount"]]


def zscore(series: pd.Series) -> pd.Series:
    mean = series.rolling(60, min_periods=20).mean()
    std = series.rolling(60, min_periods=20).std()
    return (series - mean) / std


def _transform_z(series: pd.Series, mode: str) -> pd.Series:
    if mode == "tanh":
        return np.tanh(series)
    if mode == "log":
        return np.sign(series) * np.log1p(np.abs(series))
    return series


def build_signals(
    df: pd.DataFrame,
    window: int,
    w_price: float,
    w_vol: float,
    threshold: float,
    min_hold: int,
    mode: str = "linear",
) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out["price_mom"] = out["close"].pct_change(window)
    out["vol_mom"] = out["vol"].pct_change(window)
    out["price_z"] = _transform_z(zscore(out["price_mom"]), mode)
    out["vol_z"] = _transform_z(zscore(out["vol_mom"]), mode)
    total_w = w_price + w_vol
    if total_w == 0:
        total_w = 1.0
    out["score"] = (
        (w_price / total_w) * out["price_z"]
        + (w_vol / total_w) * out["vol_z"]
    )

    out["raw_signal"] = 0
    out.loc[out["score"] > threshold, "raw_signal"] = 1
    out.loc[out["score"] < -threshold, "raw_signal"] = 0

    positions = []
    last_pos = 0
    hold_days = 0
    for _, row in out.iterrows():
        desired = int(row["raw_signal"])
        if last_pos == 1:
            hold_days += 1

        if desired != last_pos:
            if last_pos == 1 and hold_days < min_hold:
                positions.append(last_pos)
                continue
            last_pos = desired
            hold_days = 1 if last_pos == 1 else 0

        positions.append(last_pos)

    out["position"] = positions
    out["position_prev"] = out["position"].shift(1).fillna(0)
    out["trade"] = ""
    out.loc[(out["position_prev"] == 0) & (out["position"] == 1), "trade"] = "买入"
    out.loc[(out["position_prev"] == 1) & (out["position"] == 0), "trade"] = "卖出"

    out["strategy_ret"] = out["position_prev"] * out["ret"]
    out["nav"] = (1 + out["strategy_ret"].fillna(0)).cumprod()
    return out


def performance(nav: pd.Series, daily_ret: pd.Series) -> dict:
    daily_ret = daily_ret.dropna()
    nav = nav.dropna()
    if len(daily_ret) == 0 or len(nav) == 0:
        return {}
    ann_ret = nav.iloc[-1] ** (252 / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (
        (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        if daily_ret.std() != 0
        else np.nan
    )
    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = dd.min()
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
    }


def performance_from_ret(daily_ret: pd.Series) -> dict:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) == 0:
        return {}
    ann_ret = (1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (
        (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        if daily_ret.std() != 0
        else np.nan
    )
    nav = (1 + daily_ret).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = dd.min()
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
    }


def add_regime_labels(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].pct_change()
    trend = df["close"].pct_change(126)
    vol = ret.rolling(60, min_periods=20).std()

    trend_high = trend.quantile(0.6)
    trend_low = trend.quantile(0.4)
    vol_high = vol.quantile(0.7)

    regime = pd.Series("side_lowvol", index=df.index)
    regime.loc[trend > trend_high] = "up_lowvol"
    regime.loc[trend < trend_low] = "down_lowvol"
    regime.loc[(regime == "up_lowvol") & (vol > vol_high)] = "up_highvol"
    regime.loc[(regime == "down_lowvol") & (vol > vol_high)] = "down_highvol"
    regime.loc[(regime == "side_lowvol") & (vol > vol_high)] = "side_highvol"
    return regime


def performance_by_regime(daily_ret: pd.Series, regimes: pd.Series) -> dict:
    if regimes.name != "regime":
        regimes = regimes.rename("regime")
    aligned = pd.concat([daily_ret, regimes], axis=1).dropna()
    if aligned.empty:
        return {}
    metrics = {"ann_return": [], "ann_vol": [], "sharpe": [], "max_drawdown": []}
    for _, group in aligned.groupby("regime"):
        stats = performance_from_ret(group.iloc[:, 0])
        if not stats:
            continue
        for key in metrics:
            metrics[key].append(stats[key])
    if not metrics["sharpe"]:
        return {}
    return {key: _safe_mean(values) for key, values in metrics.items()}


def walk_forward_splits(
    index: pd.DatetimeIndex, train_years: int, test_months: int
) -> List[tuple]:
    if len(index) == 0:
        return []
    splits = []
    start = index.min()
    end = index.max()
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        splits.append((train_start, train_end, test_end))
        train_start = train_start + pd.DateOffset(months=test_months)
    return splits


def _safe_mean(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def grid_search(
    codes: List[str],
    start: str,
    end: str,
    windows: List[int],
    w_prices: List[float],
    w_vols: List[float],
    thresholds: List[float],
    min_holds: List[int],
    modes: List[str],
    metric: str = "sharpe",
    top_n: int = 10,
    sleep_sec: float = 1.3,
    balance_regimes: bool = False,
    walkforward: bool = False,
    train_years: int = 3,
    test_months: int = 6,
) -> pd.DataFrame:
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.2f}h"

    df_cache: Dict[str, pd.DataFrame] = {}
    fetch_start = time.time()
    for idx, code in enumerate(codes, 1):
        time.sleep(sleep_sec)
        df = fetch_daily(code, start, end)
        if df.empty:
            print(f"[WARN] no data for {code}")
            continue
        df_cache[code] = df
        elapsed = time.time() - fetch_start
        avg = elapsed / idx
        eta = avg * (len(codes) - idx)
        if idx % 5 == 0 or idx == len(codes):
            print(
                f"[INFO] fetch {idx}/{len(codes)} "
                f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(eta)}"
            )

    results: List[dict] = []
    combos = list(product(windows, w_prices, w_vols, thresholds, min_holds, modes))
    total = len(combos)
    run_start = time.time()
    for i, (window, w_price, w_vol, threshold, min_hold, mode) in enumerate(
        combos, 1
    ):
        stats_list = []
        for code, df in df_cache.items():
            signals = build_signals(
                df, window, w_price, w_vol, threshold, min_hold, mode
            )
            if balance_regimes:
                regime = add_regime_labels(df).rename("regime")
                signals = signals.join(regime, how="left")
            if walkforward:
                split_stats = []
                for _, train_end, test_end in walk_forward_splits(
                    df.index, train_years, test_months
                ):
                    mask = (signals.index > train_end) & (signals.index <= test_end)
                    test_ret = signals.loc[mask, "strategy_ret"]
                    if balance_regimes:
                        test_regime = signals.loc[mask, "regime"]
                        stats = performance_by_regime(test_ret, test_regime)
                    else:
                        stats = performance_from_ret(test_ret)
                    if stats:
                        split_stats.append(stats)
                if split_stats:
                    stats_list.append(
                        {
                            "ann_return": _safe_mean(
                                [s["ann_return"] for s in split_stats]
                            ),
                            "ann_vol": _safe_mean(
                                [s["ann_vol"] for s in split_stats]
                            ),
                            "sharpe": _safe_mean(
                                [s["sharpe"] for s in split_stats]
                            ),
                            "max_drawdown": _safe_mean(
                                [s["max_drawdown"] for s in split_stats]
                            ),
                        }
                    )
            else:
                if balance_regimes:
                    stats = performance_by_regime(
                        signals["strategy_ret"], signals["regime"]
                    )
                else:
                    stats = performance(signals["nav"], signals["strategy_ret"])
                if stats:
                    stats_list.append(stats)
        if not stats_list:
            continue
        mean_sharpe = _safe_mean([s["sharpe"] for s in stats_list])
        mean_return = _safe_mean([s["ann_return"] for s in stats_list])
        mean_mdd = _safe_mean([s["max_drawdown"] for s in stats_list])
        results.append(
            {
                "window": window,
                "w_price": w_price,
                "w_vol": w_vol,
                "threshold": threshold,
                "min_hold": min_hold,
                "mode": mode,
                "avg_sharpe": mean_sharpe,
                "avg_ann_return": mean_return,
                "avg_max_drawdown": mean_mdd,
                "stocks": len(stats_list),
            }
        )
        if i % 25 == 0 or i == total:
            elapsed = time.time() - run_start
            avg = elapsed / i
            eta = avg * (total - i)
            print(
                f"[INFO] progress {i}/{total} "
                f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(eta)}"
            )

    if not results:
        print("[ERROR] no valid results")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if metric not in df.columns:
        metric = "avg_sharpe"
    df = df.sort_values([metric, "avg_ann_return"], ascending=[False, False])
    print(df.head(top_n).to_string(index=False))
    return df


class BacktestApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tushare 量价回测")
        self.geometry("980x760")
        self.recommended = {
            "w_price": 0.3,
            "w_vol": 0.6,
            "threshold": 0.8,
            "min_hold": 20,
            "mode": "log",
        }
        self._build_ui()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(frm, text="股票代码").grid(row=0, column=0, sticky=tk.W)
        self.code_var = tk.StringVar(value="600021.SH")
        ttk.Entry(frm, textvariable=self.code_var, width=15).grid(
            row=0, column=1, padx=6
        )

        ttk.Label(frm, text="开始日期 (YYYYMMDD)").grid(
            row=0, column=2, sticky=tk.W
        )
        self.start_var = tk.StringVar(value="20220101")
        ttk.Entry(frm, textvariable=self.start_var, width=12).grid(
            row=0, column=3, padx=6
        )

        ttk.Label(frm, text="结束日期 (YYYYMMDD)").grid(
            row=0, column=4, sticky=tk.W
        )
        self.end_var = tk.StringVar(value=datetime.today().strftime("%Y%m%d"))
        ttk.Entry(frm, textvariable=self.end_var, width=12).grid(
            row=0, column=5, padx=6
        )

        ttk.Label(frm, text="动量窗口").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        self.window_var = tk.IntVar(value=20)
        ttk.Entry(frm, textvariable=self.window_var, width=8).grid(
            row=1, column=1, padx=6, pady=(8, 0)
        )

        ttk.Label(frm, text="价格权重").grid(
            row=1, column=2, sticky=tk.W, pady=(8, 0)
        )
        self.w_price_var = tk.DoubleVar(value=0.3)
        ttk.Entry(frm, textvariable=self.w_price_var, width=8).grid(
            row=1, column=3, padx=6, pady=(8, 0)
        )

        ttk.Label(frm, text="成交量权重").grid(
            row=1, column=4, sticky=tk.W, pady=(8, 0)
        )
        self.w_vol_var = tk.DoubleVar(value=0.6)
        ttk.Entry(frm, textvariable=self.w_vol_var, width=8).grid(
            row=1, column=5, padx=6, pady=(8, 0)
        )

        ttk.Label(frm, text="阈值").grid(
            row=1, column=6, sticky=tk.W, pady=(8, 0)
        )
        self.th_var = tk.DoubleVar(value=0.8)
        ttk.Entry(frm, textvariable=self.th_var, width=8).grid(
            row=1, column=7, padx=6, pady=(8, 0)
        )

        ttk.Label(frm, text="最短持有天数").grid(
            row=2, column=0, sticky=tk.W, pady=(8, 0)
        )
        self.min_hold_var = tk.IntVar(value=20)
        ttk.Entry(frm, textvariable=self.min_hold_var, width=8).grid(
            row=2, column=1, padx=6, pady=(8, 0)
        )

        ttk.Label(frm, text="组合方式").grid(
            row=2, column=2, sticky=tk.W, pady=(8, 0)
        )
        self.mode_var = tk.StringVar(value="log")
        ttk.Combobox(
            frm,
            textvariable=self.mode_var,
            values=("linear", "tanh", "log"),
            width=8,
            state="readonly",
        ).grid(row=2, column=3, padx=6, pady=(8, 0))

        self.run_btn = ttk.Button(frm, text="开始回测", command=self.run_backtest)
        self.run_btn.grid(row=0, column=7, padx=6)

        rec_text = "推荐参数：价格权重 0.3 / 成交量权重 0.6 / 阈值 0.8 / 最短持有 20 / 组合方式 log"
        self.rec_var = tk.StringVar(value=rec_text)
        rec_frame = ttk.Frame(self, padding=(10, 6, 10, 0))
        rec_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(rec_frame, textvariable=self.rec_var).pack(side=tk.LEFT)
        ttk.Button(
            rec_frame, text="使用推荐参数", command=self.apply_recommended
        ).pack(side=tk.RIGHT)

        self.info_var = tk.StringVar(value="就绪。")
        ttk.Label(self, textvariable=self.info_var).pack(
            side=tk.TOP, anchor=tk.W, padx=10
        )

        table_frame = ttk.Frame(self, padding=(10, 0, 10, 6))
        table_frame.pack(side=tk.TOP, fill=tk.X)
        self.trade_table = ttk.Treeview(
            table_frame,
            columns=("date", "action", "price"),
            show="headings",
            height=6,
        )
        self.trade_table.heading("date", text="日期")
        self.trade_table.heading("action", text="操作")
        self.trade_table.heading("price", text="价格")
        self.trade_table.column("date", width=120, anchor=tk.W)
        self.trade_table.column("action", width=80, anchor=tk.CENTER)
        self.trade_table.column("price", width=100, anchor=tk.E)
        self.trade_table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        scroll.config(command=self.trade_table.yview)
        self.trade_table.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax_price = self.fig.add_subplot(2, 1, 1)
        self.ax_nav = self.fig.add_subplot(2, 1, 2)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def run_backtest(self) -> None:
        code = self.code_var.get().strip()
        start = self.start_var.get().strip()
        end = self.end_var.get().strip()
        try:
            window = int(self.window_var.get())
            w_price = float(self.w_price_var.get())
            w_vol = float(self.w_vol_var.get())
            threshold = float(self.th_var.get())
            min_hold = int(self.min_hold_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "参数需要为数字。")
            return

        if not code or len(start) != 8 or len(end) != 8:
            messagebox.showerror("输入错误", "请检查股票代码与日期格式。")
            return
        if min_hold < 1:
            messagebox.showerror("输入错误", "最短持有天数需大于等于 1。")
            return

        self.info_var.set("正在加载数据...")
        self.update_idletasks()

        df = fetch_daily(code, start, end)
        if df.empty:
            messagebox.showerror("数据错误", "未获取到 Tushare 数据。")
            self.info_var.set("无数据。")
            return

        mode = self.mode_var.get().strip() or "linear"
        signals = build_signals(
            df, window, w_price, w_vol, threshold, min_hold, mode=mode
        )
        stats = performance(signals["nav"], signals["strategy_ret"])
        if not stats:
            messagebox.showwarning("结果提示", "数据量不足，无法回测。")
            return

        self._plot_result(signals)
        self._populate_trades(signals)
        self.info_var.set(
            "年化收益={:.2%}  年化波动={:.2%}  夏普={:.2f}  最大回撤={:.2%}".format(
                stats["ann_return"],
                stats["ann_vol"],
                stats["sharpe"],
                stats["max_drawdown"],
            )
        )

    def apply_recommended(self) -> None:
        self.w_price_var.set(self.recommended["w_price"])
        self.w_vol_var.set(self.recommended["w_vol"])
        self.th_var.set(self.recommended["threshold"])
        self.min_hold_var.set(self.recommended["min_hold"])
        self.mode_var.set(self.recommended["mode"])

    def _populate_trades(self, signals: pd.DataFrame) -> None:
        for item in self.trade_table.get_children():
            self.trade_table.delete(item)
        trades = signals[signals["trade"] != ""]
        for idx, row in trades.iterrows():
            self.trade_table.insert(
                "",
                tk.END,
                values=(idx.strftime("%Y-%m-%d"), row["trade"], f"{row['close']:.2f}"),
            )

    def _plot_result(self, signals: pd.DataFrame) -> None:
        self.ax_price.clear()
        self.ax_nav.clear()

        self.ax_price.plot(signals.index, signals["close"], label="收盘价", lw=1.2)
        buy_idx = signals.index[signals["trade"] == "买入"]
        sell_idx = signals.index[signals["trade"] == "卖出"]
        self.ax_price.scatter(
            buy_idx,
            signals.loc[buy_idx, "close"],
            marker="^",
            color="green",
            s=30,
            label="买入点",
        )
        self.ax_price.scatter(
            sell_idx,
            signals.loc[sell_idx, "close"],
            marker="v",
            color="red",
            s=30,
            label="卖出点",
        )

        self.ax_price.set_title("价格与交易信号")
        self.ax_price.legend(loc="best")

        self.ax_nav.plot(signals.index, signals["nav"], color="blue", label="净值")
        self.ax_nav.set_title("策略净值")
        self.ax_nav.legend(loc="best")

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="run batch grid search")
    parser.add_argument("--start", default="20220101")
    parser.add_argument("--end", default=datetime.today().strftime("%Y%m%d"))
    parser.add_argument("--metric", default="avg_sharpe")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--modes", default="linear,tanh,log")
    parser.add_argument("--windows", default="10,20,30")
    parser.add_argument("--w_prices", default="0.2,0.4,0.6")
    parser.add_argument("--w_vols", default="0.2,0.4,0.6")
    parser.add_argument("--thresholds", default="0.5,0.7,0.9")
    parser.add_argument("--min_holds", default="5,10,20")
    parser.add_argument("--sleep", type=float, default=1.3)
    parser.add_argument("--balance_regimes", action="store_true")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--train_years", type=int, default=3)
    parser.add_argument("--test_months", type=int, default=6)
    args = parser.parse_args()

    if args.batch:
        DEFAULT_CODES = [
            "601398.SH",
            "601288.SH",
            "601939.SH",
            "600036.SH",
            "601318.SH",
            "601628.SH",
            "601601.SH",
            "600030.SH",
            "600941.SH",
            "601728.SH",
            "600519.SH",
            "000858.SZ",
            "000568.SZ",
            "600887.SH",
            "600886.SH",
            "600276.SH",
            "600104.SH",
            "002594.SZ",
            "300750.SZ",
            "601012.SH",
            "600438.SH",
            "600196.SH",
            "002475.SZ",
            "002241.SZ",
            "601138.SH",
            "688981.SH",
            "002371.SZ",
            "603501.SH",
            "600570.SH",
            "688111.SH",
            "601857.SH",
            "600028.SH",
            "600938.SH",
            "601899.SH",
            "601088.SH",
            "600900.SH",
            "601668.SH",
            "000002.SZ",
            "000333.SZ",
            "000651.SZ",
            "600690.SH",
            "002415.SZ",
            "300760.SZ",
            "002714.SZ",
            "601888.SH",
            "600009.SH",
            "601111.SH",
            "600019.SH",
            "601360.SH",
            "300308.SZ",
        ]
        grid_search(
            DEFAULT_CODES,
            args.start,
            args.end,
            windows=[int(x) for x in args.windows.split(",") if x.strip()],
            w_prices=[float(x) for x in args.w_prices.split(",") if x.strip()],
            w_vols=[float(x) for x in args.w_vols.split(",") if x.strip()],
            thresholds=[float(x) for x in args.thresholds.split(",") if x.strip()],
            min_holds=[int(x) for x in args.min_holds.split(",") if x.strip()],
            modes=[m.strip() for m in args.modes.split(",") if m.strip()],
            metric=args.metric,
            top_n=args.top,
            sleep_sec=args.sleep,
            balance_regimes=args.balance_regimes,
            walkforward=args.walkforward,
            train_years=args.train_years,
            test_months=args.test_months,
        )
    else:
        app = BacktestApp()
        app.mainloop()
