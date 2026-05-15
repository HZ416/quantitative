# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime, time as dtime
import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -------------------------
# Matplotlib 中文显示
# -------------------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------
# 配置区
# -------------------------
TUSHARE_TOKEN = "1720b38d64284be11ecf869292740df2a78339ff1c3060bef1b942f6"
TS_CODE = "600000.SH"          # 浦发银行
MULTI_CODE = "002475.SZ \n 600276.SH \n 002594.SZ \n 601398.SH \n 601318.SH \n 002050.SZ \n 600150.SH \n 000568.SZ \n 601888.SH \n 603993.SH \n 002558.SZ \n 601360.SH \n 603131.SH"  # 多股票输入示例（实盘预测模式）
START_DATE = "20230101"
END_DATE = "20250201"

N = 10                         # history_len：回溯窗口
M = 5                          # forecast_len：预测窗口
TRAIN_LEN = 90                 # training_len：训练样本长度（交易日）
TAKE_PROFIT = 0.10             # 止盈 10%
FRI_EXIT_TH = 0.02             # 周五涨幅<2%：卖出

INITIAL_CASH = 10_000_000
COMMISSION = 0.0001            # 佣金
SLIPPAGE = 0.0001              # 滑点（按成交价乘(1+/-slippage)近似）

# -------------------------
# 数据获取（TuShare Pro）
# -------------------------
def fetch_daily(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    使用 TuShare Pro daily 接口拉取A股日线行情。
    daily 文档说明字段包含 trade_date/open/high/low/close/vol 等。:contentReference[oaicite:1]{index=1}
    """
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                   fields="ts_code,trade_date,open,high,low,close,vol")
    if df is None or df.empty:
        raise RuntimeError("TuShare daily 返回为空：请检查 token、积分权限、ts_code、日期范围。")

    # trade_date 是 YYYYMMDD，TuShare 返回通常是倒序
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # vol 单位通常是“手”，这里保持原样
    return df

# -------------------------
# 特征工程（复刻你 gm 版本的 7 维特征）
# -------------------------
def make_features(window: pd.DataFrame) -> np.ndarray:
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    vol = window["vol"].values

    close_mean = close[-1] / np.mean(close)          # 1 收盘价/均值
    volume_mean = vol[-1] / np.mean(vol)             # 2 现量/均量
    max_mean = high[-1] / np.mean(high)              # 3 最高价/均价
    min_mean = low[-1] / np.mean(low)                # 4 最低价/均价
    v_now = vol[-1]                                   # 5 现量（绝对值）
    ret_now = close[-1] / close[0]                   # 6 区间收益率
    std = np.std(close)                              # 7 区间标准差

    return np.array([close_mean, volume_mean, max_mean, min_mean, v_now, ret_now, std], dtype=float)

def build_dataset(df: pd.DataFrame, t_end_idx: int, n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用 [t_end_idx-TRAIN_LEN ... t_end_idx] 这一段历史构建训练集：
      X: 每个样本取过去 n 天窗口特征
      y: 未来 m 天最后收盘 > 第一天收盘 => 1 else 0
    同时返回 t_end_idx 当天对应的“最新一期特征”用于预测。
    """
    x_train, y_train = [], []

    # 训练样本区间的起点：保证能取到 n 天窗口
    start = max(0, t_end_idx - TRAIN_LEN - n)
    end = t_end_idx

    for i in range(start + n - 1, end + 1):
        win = df.iloc[i - n + 1:i + 1]
        x = make_features(win)

        # 标签：未来 m 天
        if i + m < len(df):
            y0 = df.iloc[i + 1]["close"]
            y1 = df.iloc[i + m]["close"]
            y = 1 if y1 > y0 else 0
            x_train.append(x)
            y_train.append(y)

    if len(x_train) <= 10:
        raise RuntimeError("训练样本太少：请扩大日期范围或调小 TRAIN_LEN/N/M。")

    # 最新一期特征：用 t_end_idx 结束的 n 日窗口
    latest_win = df.iloc[t_end_idx - n + 1:t_end_idx + 1]
    latest_x = make_features(latest_win)

    return np.asarray(x_train), np.asarray(y_train), latest_x

# -------------------------
# 回测主循环（周一预测开仓；止盈；周五弱势退出）
# -------------------------
def backtest(df: pd.DataFrame) -> dict:
    cash = INITIAL_CASH
    position = 0.0   # 持股数量
    entry_price = None

    equity_curve = []
    dates = []
    trades = []  # (date, "BUY"/"SELL", price)

    # 用 Pipeline：标准化 + RBF SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", gamma="scale"))
    ])

    for i in range(N - 1 + TRAIN_LEN + M, len(df)):
        row = df.iloc[i]
        date = row["trade_date"]
        close = float(row["close"])

        # 记录净值
        equity = cash + position * close
        equity_curve.append(equity)
        dates.append(date)

        weekday = date.isoweekday()  # 周一=1...周日=7

        # 已持仓：止盈 or 周五弱势退出
        if position > 0:
            pnl = close / entry_price - 1.0

            # 止盈：>=10%
            if pnl >= TAKE_PROFIT:
                sell_price = close * (1 - SLIPPAGE)
                cash = position * sell_price * (1 - COMMISSION)
                position = 0.0
                entry_price = None
                trades.append((date, "SELL_TP", sell_price))
                continue

            # 周五收盘：涨幅 < 2% 则退出
            if weekday == 5 and pnl < FRI_EXIT_TH:
                sell_price = close * (1 - SLIPPAGE)
                cash = position * sell_price * (1 - COMMISSION)
                position = 0.0
                entry_price = None
                trades.append((date, "SELL_FRI", sell_price))
                continue

        # 未持仓：周一开仓信号（用“周一收盘”近似 gm 的 9:31）
        if position == 0 and weekday == 1:
            X, y, latest_x = build_dataset(df, t_end_idx=i, n=N, m=M)

            model.fit(X, y)
            pred = int(model.predict(latest_x.reshape(1, -1))[0])

            if pred == 1:
                buy_price = close * (1 + SLIPPAGE)
                position = (cash * (1 - COMMISSION)) / buy_price
                cash = 0.0
                entry_price = buy_price
                trades.append((date, "BUY", buy_price))

    # 最后一天强制平仓（方便统计）
    if position > 0:
        last_close = float(df.iloc[-1]["close"])
        sell_price = last_close * (1 - SLIPPAGE)
        cash = position * sell_price * (1 - COMMISSION)
        position = 0.0
        entry_price = None
        trades.append((df.iloc[-1]["trade_date"], "SELL_END", sell_price))
        equity_curve.append(cash)
        dates.append(df.iloc[-1]["trade_date"])

    res = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity_curve}).drop_duplicates("date").set_index("date")
    res["ret"] = res["equity"].pct_change().fillna(0.0)
    res["cummax"] = res["equity"].cummax()
    res["dd"] = res["equity"] / res["cummax"] - 1.0

    return {"perf": res, "trades": trades}

# -------------------------
# 可视化
# -------------------------

def _require_token() -> str:
    return os.getenv("TUSHARE_TOKEN") or TUSHARE_TOKEN

def build_gui():

    root = tk.Tk()
    root.title("SVM 策略回测可视化")
    root.geometry("1100x800")

    frm = ttk.Frame(root, padding=10)
    frm.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(frm, text="股票代码(ts_code):").grid(row=0, column=0, sticky="w")
    code_var = tk.StringVar(value=TS_CODE)
    code_entry = ttk.Entry(frm, textvariable=code_var, width=12)
    code_entry.grid(row=0, column=1, padx=5)

    ttk.Label(frm, text="Token:").grid(row=0, column=2, sticky="w")
    token_var = tk.StringVar(value=_require_token())
    token_entry = ttk.Entry(frm, textvariable=token_var, width=26)
    token_entry.grid(row=0, column=3, padx=5)

    ttk.Label(frm, text="开始日期(YYYYMMDD):").grid(row=1, column=0, sticky="w")
    start_var = tk.StringVar(value=START_DATE)
    start_entry = ttk.Entry(frm, textvariable=start_var, width=10)
    start_entry.grid(row=1, column=1, padx=5, sticky="w")

    ttk.Label(frm, text="结束日期(YYYYMMDD):").grid(row=1, column=2, sticky="w")
    end_var = tk.StringVar(value=END_DATE)
    end_entry = ttk.Entry(frm, textvariable=end_var, width=10)
    end_entry.grid(row=1, column=3, padx=5, sticky="w")

    mode_var = tk.StringVar(value="backtest")
    ttk.Radiobutton(frm, text="回测", variable=mode_var, value="backtest").grid(
        row=1, column=4, padx=5, sticky="w"
    )
    ttk.Radiobutton(frm, text="实盘预测(周一收盘)", variable=mode_var, value="live").grid(
        row=1, column=5, padx=5, sticky="w"
    )

    run_btn = ttk.Button(frm, text="运行")
    run_btn.grid(row=1, column=6, padx=10)

    reset_btn = ttk.Button(frm, text="重置缩放")
    reset_btn.grid(row=1, column=7, padx=5)

    metrics_var = tk.StringVar(value="")
    metrics_label = ttk.Label(root, textvariable=metrics_var, padding=(10, 0))
    metrics_label.pack(side=tk.TOP, anchor="w")

    signal_var = tk.StringVar(value="")
    signal_label = ttk.Label(root, textvariable=signal_var, padding=(10, 0))
    signal_label.pack(side=tk.TOP, anchor="w")

    # 右侧交易记录栏
    right_frame = ttk.Frame(root, padding=(5, 5))
    right_frame.pack(side=tk.RIGHT, fill=tk.Y)
    ttk.Label(right_frame, text="交易记录").pack(anchor="w")
    trades_list = tk.Listbox(right_frame, width=32, height=30)
    trades_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=trades_list.yview)
    trades_list.configure(yscrollcommand=trades_scroll.set)
    trades_list.pack(side=tk.LEFT, fill=tk.Y, expand=False)
    trades_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # 左侧主区域（图表/实盘预测输入）
    main_frame = ttk.Frame(root)
    main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Matplotlib 图形区域
    fig = Figure(figsize=(10, 6), dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1r = ax1.twinx()
    locator = AutoDateLocator()
    formatter = ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    fig.tight_layout(pad=2.0)

    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 实盘预测输入区（替代图表区域）
    live_frame = ttk.Frame(main_frame, padding=(10, 10))
    ttk.Label(live_frame, text="实盘预测 - 多股票输入（每行一个或用逗号分隔）").pack(anchor="w")
    live_text = tk.Text(live_frame, height=10, width=60)
    live_text.insert("1.0", MULTI_CODE)
    live_text.pack(fill=tk.X, expand=False)
    live_btn = ttk.Button(live_frame, text="运行实盘预测")
    live_btn.pack(anchor="w", pady=(6, 4))
    ttk.Label(live_frame, text="预测结果").pack(anchor="w")
    live_results = tk.Listbox(live_frame, height=18)
    live_scroll = ttk.Scrollbar(live_frame, orient=tk.VERTICAL, command=live_results.yview)
    live_results.configure(yscrollcommand=live_scroll.set)
    live_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    live_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def parse_codes(text: str) -> list[str]:
        raw = [c.strip() for c in text.replace("，", ",").replace("\n", ",").split(",")]
        return [c for c in raw if c]

    def show_mode(mode: str):
        if mode == "live":
            canvas.get_tk_widget().pack_forget()
            live_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            live_frame.pack_forget()
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def render(ts_code: str, start_date: str, end_date: str, token: str, mode: str):
        try:
            if not token:
                raise RuntimeError("Tushare Token 为空，请填写有效 Token。")
            ts.set_token(token)
            pro = ts.pro_api()
            # 实盘预测模式：忽略输入时间，使用真实时间窗口
            if mode == "live":
                today = datetime.now().date()
                # 拉取足够长历史，覆盖 TRAIN_LEN + N + M
                start_date = (today - pd.Timedelta(days=400)).strftime("%Y%m%d")
                end_date = today.strftime("%Y%m%d")
            df_stock = fetch_daily(pro, ts_code, start_date, end_date)
            trades = []
            perf = None
            signal_msg = ""
            if mode == "backtest":
                out = backtest(df_stock)
                perf = out["perf"]
                trades = out["trades"]
            else:
                # 实盘预测：根据真实时间决定使用哪一次“周一收盘”作为信号
                now = datetime.now()
                is_monday_after_close = (now.weekday() == 0 and now.time() >= dtime(16, 0))
                monday_idx = df_stock.index[df_stock["trade_date"].dt.weekday == 0]
                if len(monday_idx) == 0:
                    raise RuntimeError("数据中未找到周一交易日，无法进行实盘预测。")
                if is_monday_after_close:
                    i = len(df_stock) - 1
                    if df_stock.iloc[i]["trade_date"].weekday() != 0:
                        i = monday_idx[-1]
                else:
                    i = monday_idx[-1]
                X, y, latest_x = build_dataset(df_stock, t_end_idx=i, n=N, m=M)
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svc", SVC(C=1.0, kernel="rbf", gamma="scale"))
                ])
                model.fit(X, y)
                pred = int(model.predict(latest_x.reshape(1, -1))[0])
                last_date = df_stock.iloc[i]["trade_date"].strftime("%Y-%m-%d")
                next_open = (df_stock.iloc[i]["trade_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if pred == 1:
                    signal_msg = f"实盘预测：{last_date} 收盘信号 -> {next_open} 开盘【买入】"
                else:
                    signal_msg = f"实盘预测：{last_date} 收盘信号 -> {next_open} 开盘【不买入】"
                if now.weekday() == 4:
                    signal_msg += "    提醒：今日周五，请人工检查止盈/弱势退出规则"
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return

        if mode == "live":
            metrics_var.set("")
            signal_var.set(signal_msg)
            return

        # 个股实际行情 + 买卖点（左轴股价，右轴区间涨跌幅%）
        ax1.clear()
        ax1r.clear()
        stock_close = df_stock.set_index("trade_date")["close"]
        ax1.plot(stock_close.index, stock_close.values, color="#2ca02c", label="价格")
        ax1.set_title(f"个股行情线 ({ts_code})")
        ax1.set_ylabel("股价")

        stock_pct = (stock_close / float(stock_close.iloc[0]) - 1.0) * 100.0
        ax1r.plot(stock_pct.index, stock_pct.values, color="#1f77b4", linestyle="--", label="区间涨跌幅%")
        ax1r.set_ylabel("区间涨跌幅(%)")
        for d, typ, _ in trades:
            if d in stock_close.index:
                if "BUY" in typ:
                    ax1.scatter(d, stock_close.loc[d], marker="^", color="red", s=40)
                else:
                    ax1.scatter(d, stock_close.loc[d], marker="v", color="blue", s=40)

        # 策略收益线 + 买卖点（仅显示净值）
        ax2.clear()
        if perf is not None:
            ax2.plot(perf.index, perf["equity"], color="#d62728", label="净值")
            ax2.set_title("策略收益线")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("净值")
            for d, typ, _ in trades:
                if d in perf.index:
                    if "BUY" in typ:
                        ax2.scatter(d, perf.loc[d, "equity"], marker="^", color="red", s=40)
                    else:
                        ax2.scatter(d, perf.loc[d, "equity"], marker="v", color="blue", s=40)
        else:
            ax2.set_title("实盘预测模式（不显示回测收益线）")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("净值")

        # 关键指标
        if perf is not None:
            equity0 = float(perf["equity"].iloc[0])
            equity1 = float(perf["equity"].iloc[-1])
            total_ret = equity1 / equity0 - 1.0
            n_days = max(1, len(perf))
            ann_ret = (equity1 / equity0) ** (252 / n_days) - 1.0
            max_dd = float(perf["dd"].min())
            metrics_var.set(
                f"总收益: {total_ret*100:.2f}%    "
                f"年化收益: {ann_ret*100:.2f}%    "
                f"最大回撤: {max_dd*100:.2f}%    "
                f"交易次数: {len(trades)}"
            )
            signal_var.set("")
        else:
            metrics_var.set("")
            signal_var.set(signal_msg)

        # 更新右侧交易记录
        trades_list.delete(0, tk.END)
        if trades:
            for d, typ, p in trades:
                trades_list.insert(tk.END, f"{d.strftime('%Y-%m-%d')} {typ} {p:.2f}")
        elif signal_msg:
            trades_list.insert(tk.END, signal_msg)

        fig.tight_layout(pad=2.0)
        canvas.draw_idle()

    def on_select_ax1(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        canvas.draw_idle()

    def on_select_ax2(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        canvas.draw_idle()

    RectangleSelector(
        ax1, on_select_ax1, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords="pixels", interactive=True
    )
    RectangleSelector(
        ax2, on_select_ax2, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords="pixels", interactive=True
    )

    def on_run():
        token = token_var.get().strip()
        mode = mode_var.get()
        show_mode(mode)
        if mode == "backtest":
            ts_code = code_var.get().strip()
            start_date = start_var.get().strip()
            end_date = end_var.get().strip()
            if not ts_code or not start_date or not end_date:
                messagebox.showwarning("提示", "请完整填写股票代码、开始日期、结束日期。")
                return
            render(ts_code, start_date, end_date, token, mode)
        else:
            codes = parse_codes(live_text.get("1.0", tk.END))
            if not codes:
                messagebox.showwarning("提示", "请在左侧区域输入股票代码列表。")
                return
            # 清空结果并逐一预测
            live_results.delete(0, tk.END)
            metrics_var.set("")
            signal_var.set("")
            buy_results = []
            no_buy_results = []
            other_results = []
            for c in codes:
                try:
                    render(c, start_var.get().strip(), end_var.get().strip(), token, mode)
                    # signal_var 会被更新为本次信号
                    result_text = f"{c}  {signal_var.get()}"
                    if "【买入】" in result_text:
                        buy_results.append(result_text)
                    elif "【不买入】" in result_text:
                        no_buy_results.append(result_text)
                    else:
                        other_results.append(result_text)
                except Exception as e:
                    other_results.append(f"{c}  错误: {e}")

            for item in buy_results + no_buy_results + other_results:
                live_results.insert(tk.END, item)

    run_btn.configure(command=on_run)
    live_btn.configure(command=on_run)
    reset_btn.configure(command=lambda: render(
        code_var.get().strip(),
        start_var.get().strip(),
        end_var.get().strip(),
        token_var.get().strip(),
        mode_var.get(),
    ))

    # 初次渲染
    show_mode(mode_var.get())
    render(TS_CODE, START_DATE, END_DATE, token_var.get().strip(), mode_var.get())

    root.mainloop()

def main():
    build_gui()

if __name__ == "__main__":
    main()
