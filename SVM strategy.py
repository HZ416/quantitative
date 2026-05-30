# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
import queue
import threading
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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

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
TRAIN_LEN = 250                # training_len：训练样本长度（交易日），由 90 扩大至 250
TAKE_PROFIT = 0.10             # 周五检查：盈利 > 10% 卖出

# 优化新增参数
K_ATR_LABEL = 0.5              # 标签阈值倍数：|未来M日收益| > K_ATR_LABEL * ATR% * sqrt(M) 才计入训练
K_ATR_STOP = 2.0               # ATR 动态止损倍数：close < entry - K_ATR_STOP * ATR_entry 任意日卖出
MAX_HOLD_DAYS = 15             # 最长持仓交易日数（避免信号失效后僵持）
PROBA_THRESH = 0.55            # 概率阈值：proba(class=1) > 阈值才买入
CV_REFRESH = 12                # CV 网格搜索刷新周期（按已训练次数计）

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
# 技术指标预计算（一次性，避免每个样本重复算）
# -------------------------
_FEATURE_INDICATOR_COLS = [
    "vol_ratio", "rsi14", "macd_hist", "ma5_slope", "ma20_slope",
    "boll_pos", "atr_pct", "amp",
]

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在 df 上追加技术指标列。所有列均做归一化处理，避免量纲混杂。"""
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["vol"]

    # 均线 & 斜率
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    df["ma5_slope"] = (ma5 - ma5.shift(3)) / ma5.shift(3)
    df["ma20_slope"] = (ma20 - ma20.shift(5)) / ma20.shift(5)

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD hist（按价格归一）
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - signal) / c

    # 布林位置：(close - MA20) / (2*std20) ∈ ~[-1, 1]
    std20 = c.rolling(20).std()
    df["boll_pos"] = (c - ma20) / (2 * std20)

    # ATR(14) 及其归一化
    prev_close = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / c

    # 振幅、成交量比
    df["amp"] = (h - l) / c
    df["vol_ratio"] = v / v.rolling(20).mean()

    return df

# -------------------------
# 特征工程（13 维，全部归一化）
# -------------------------
def make_features(df: pd.DataFrame, i: int, n: int) -> np.ndarray:
    """构建以 df 第 i 行为终点的特征向量。窗口特征 + 预计算指标。"""
    win = df.iloc[i - n + 1:i + 1]
    close = win["close"].values
    high = win["high"].values
    low = win["low"].values
    last = df.iloc[i]
    close_mean = close.mean()

    return np.array([
        # —— 窗口比值（保留原有思路，但 std 改为变异系数）——
        close[-1] / close_mean,
        high[-1] / high.mean(),
        low[-1] / low.mean(),
        close[-1] / close[0],
        np.std(close) / close_mean,       # CV，替代原 std 绝对值
        # —— 预计算技术指标 ——
        float(last["vol_ratio"]),
        float(last["rsi14"]) / 100.0,
        float(last["macd_hist"]),
        float(last["ma5_slope"]),
        float(last["ma20_slope"]),
        float(last["boll_pos"]),
        float(last["atr_pct"]),
        float(last["amp"]),
    ], dtype=float)


def build_dataset(df: pd.DataFrame, t_end_idx: int, n: int, m: int,
                  k_atr_label: float = K_ATR_LABEL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用 [t_end_idx - TRAIN_LEN ... t_end_idx] 区间构建训练集：
      X: 每个样本取过去 n 天窗口 + 当日指标
      y: 未来 m 天收益用 ATR 阈值化 —— 显著上涨=1，显著下跌=0，区间内丢弃
    同时返回 t_end_idx 当天对应的"最新一期特征"用于预测。
    """
    x_train, y_train = [], []
    start = max(0, t_end_idx - TRAIN_LEN - n)
    end = t_end_idx

    for i in range(start + n - 1, end + 1):
        # 指标未就绪则跳过（开头若干日存在 NaN）
        if df.iloc[i][_FEATURE_INDICATOR_COLS].isna().any():
            continue
        x = make_features(df, i, n)
        if np.isnan(x).any():
            continue

        # 阈值化标签
        if i + m >= len(df):
            continue
        y0 = float(df.iloc[i + 1]["close"])
        y1 = float(df.iloc[i + m]["close"])
        ret = y1 / y0 - 1.0
        atr_pct = float(df.iloc[i]["atr_pct"])
        thresh = k_atr_label * atr_pct * np.sqrt(m)
        if ret > thresh:
            y = 1
        elif ret < -thresh:
            y = 0
        else:
            continue  # 中间地带丢弃，减少标签噪声

        x_train.append(x)
        y_train.append(y)

    if len(x_train) <= 20:
        raise RuntimeError(
            f"训练样本太少（仅 {len(x_train)} 条）：扩大日期范围、降低 K_ATR_LABEL 或缩小 TRAIN_LEN。"
        )

    # 最新一期特征
    latest_x = make_features(df, t_end_idx, n)
    if np.isnan(latest_x).any():
        raise RuntimeError("最新特征含 NaN，技术指标未就绪（历史长度不足）。")

    return np.asarray(x_train), np.asarray(y_train), latest_x


# -------------------------
# 策略模型：CV 网格搜索 + 概率输出 + 类别均衡
# -------------------------
class StrategyModel:
    """封装 Pipeline，提供延迟刷新的 GridSearchCV 与基于概率的预测。"""

    def __init__(self, cv_refresh: int = CV_REFRESH):
        self.best_params: dict | None = None
        self.refresh_counter: int = 0
        self.cv_refresh: int = cv_refresh
        self.model: Pipeline | None = None

    def _make_pipeline(self, params: dict | None) -> Pipeline:
        kwargs = {k.split("__")[1]: v for k, v in (params or {}).items()}
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", **kwargs))
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # 单类样本（极端情况）退化为 None，调用方应跳过
        if len(np.unique(y)) < 2:
            self.model = None
            self.refresh_counter += 1
            return

        need_search = (self.best_params is None) or (self.refresh_counter % self.cv_refresh == 0)
        if need_search:
            param_grid = {
                "svc__C": [0.5, 1.0, 2.0, 5.0],
                "svc__gamma": ["scale", 0.1, 0.5],
            }
            base = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", class_weight="balanced")),  # 搜索时关 probability，更快
            ])
            n_splits = min(4, max(2, len(y) // 40))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            try:
                search = GridSearchCV(base, param_grid, cv=tscv, scoring="f1", n_jobs=1)
                search.fit(X, y)
                self.best_params = search.best_params_
            except Exception:
                self.best_params = {"svc__C": 1.0, "svc__gamma": "scale"}

        self.model = self._make_pipeline(self.best_params)
        self.model.fit(X, y)
        self.refresh_counter += 1

    def predict_proba_pos(self, x: np.ndarray) -> float:
        if self.model is None:
            return 0.0
        proba = self.model.predict_proba(x.reshape(1, -1))[0]
        classes = list(self.model.named_steps["svc"].classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0


# -------------------------
# 实盘批量预测：跨股票池化训练 + 概率排序 + 建议止损价
# -------------------------
def _select_signal_index(df_stock: pd.DataFrame) -> int:
    """挑选用于产生信号的"周一"在 df 中的行号。"""
    now = datetime.now()
    is_monday_after_close = (now.weekday() == 0 and now.time() >= dtime(16, 0))
    monday_idx = df_stock.index[df_stock["trade_date"].dt.weekday == 0]
    if len(monday_idx) == 0:
        raise RuntimeError("无周一交易日")
    if is_monday_after_close and df_stock.iloc[-1]["trade_date"].weekday() == 0:
        return len(df_stock) - 1
    return int(monday_idx[-1])


def run_live_batch(codes: list[str], token: str, emit) -> None:
    """跨股票池化训练的实盘批量预测（后台线程调用）。
    emit(kind, payload)：kind ∈ {'progress','result','done','error'}，由主线程消费。
    """
    try:
        if not token:
            raise RuntimeError("Tushare Token 为空")
        ts.set_token(token)
        pro = ts.pro_api()

        today = datetime.now().date()
        start_date = (today - pd.Timedelta(days=400)).strftime("%Y%m%d")
        end_date = today.strftime("%Y%m%d")

        # —— 阶段 1：逐只拉取 + 算指标 + 组样本 ——
        stocks = []  # 成功加载的票
        for idx, code in enumerate(codes):
            emit("progress", f"[{idx + 1}/{len(codes)}] 拉取 {code} ...")
            try:
                df = fetch_daily(pro, code, start_date, end_date)
                df = add_indicators(df)
                i = _select_signal_index(df)
                X, y, latest_x = build_dataset(df, t_end_idx=i, n=N, m=M)
                stocks.append({
                    "code": code, "df": df, "i": i,
                    "X": X, "y": y, "latest_x": latest_x,
                })
            except Exception as e:
                emit("result", {"code": code, "error": str(e)})

        if not stocks:
            emit("done", None)
            return

        # —— 阶段 2：池化训练（一次 CV 搜索）——
        X_pool = np.vstack([s["X"] for s in stocks])
        y_pool = np.concatenate([s["y"] for s in stocks])
        pos_rate = float(y_pool.mean()) if len(y_pool) else 0.0
        emit("progress",
             f"池化训练：{len(stocks)} 只票合并 {len(y_pool)} 样本（阳性 {pos_rate:.1%}）...")

        sm = StrategyModel(cv_refresh=CV_REFRESH)
        sm.fit(X_pool, y_pool)

        # —— 阶段 3：逐只预测 + 排序输出 ——
        results = []
        for s in stocks:
            proba = sm.predict_proba_pos(s["latest_x"])
            last_row = s["df"].iloc[s["i"]]
            ref_price = float(last_row["close"])
            atr_abs = float(last_row["atr14"]) if not pd.isna(last_row["atr14"]) else 0.0
            stop_loss = ref_price - K_ATR_STOP * atr_abs
            last_date = last_row["trade_date"].strftime("%Y-%m-%d")
            next_open = (last_row["trade_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            results.append({
                "code": s["code"],
                "proba": proba,
                "action": "买入" if proba > PROBA_THRESH else "观望",
                "last_date": last_date,
                "next_open": next_open,
                "ref_price": ref_price,
                "stop_loss": stop_loss,
                "atr_abs": atr_abs,
                "max_hold": MAX_HOLD_DAYS,
            })

        results.sort(key=lambda r: -r["proba"])
        for r in results:
            emit("result", r)

        emit("progress",
             f"完成：{sum(1 for r in results if r['action']=='买入')} 买入 / {sum(1 for r in results if r['action']=='观望')} 观望  阈值={PROBA_THRESH:.0%}")
        emit("done", None)

    except Exception as e:
        emit("error", str(e))


# -------------------------
# 回测主循环（与实盘对齐：信号当日收盘产生 → 次日开盘成交）
#   - 周一收盘：模型预测；预测=1 排队，次日（周二）开盘买入
#   - 周五收盘：持仓且 pnl<0 或 pnl>10% 排队，次日（周一）开盘卖出
#   - [0%, 10%] 区间：继续持有
# -------------------------
def backtest(df: pd.DataFrame) -> dict:
    # 预计算技术指标（一次）
    df = add_indicators(df)

    cash = INITIAL_CASH
    position = 0.0          # 持股数量
    entry_price = None
    entry_atr_abs = None    # 进场时刻的绝对 ATR（用于止损判定）
    days_held = 0

    pending_buy = False
    pending_sell = None     # None / "TP" / "LOSS" / "ATR" / "MAXHOLD"

    equity_curve = []
    dates = []
    trades = []             # (date, tag, price)
    predictions = []        # (date, y_pred, y_true)
    trade_pnls = []

    sm = StrategyModel(cv_refresh=CV_REFRESH)

    # 起点须覆盖 (a) 训练样本所需历史 (b) 技术指标 warmup (~35 天)
    start_i = max(N - 1 + TRAIN_LEN + M, 40)

    for i in range(start_i, len(df)):
        row = df.iloc[i]
        date = row["trade_date"]
        open_p = float(row["open"])
        close = float(row["close"])
        atr_abs_today = float(row["atr14"]) if not pd.isna(row["atr14"]) else 0.0
        weekday = date.isoweekday()  # 周一=1...周五=5

        # ---- 1. 开盘：执行昨日产生的待执行订单 ----
        if pending_sell is not None and position > 0:
            sell_price = open_p * (1 - SLIPPAGE)
            cash = position * sell_price * (1 - COMMISSION)
            trade_pnls.append(sell_price / entry_price - 1.0)
            trades.append((date, f"SELL_{pending_sell}", sell_price))
            position = 0.0
            entry_price = None
            entry_atr_abs = None
            days_held = 0
            pending_sell = None

        if pending_buy and position == 0:
            buy_price = open_p * (1 + SLIPPAGE)
            position = (cash * (1 - COMMISSION)) / buy_price
            cash = 0.0
            entry_price = buy_price
            entry_atr_abs = atr_abs_today if atr_abs_today > 0 else None
            days_held = 0
            trades.append((date, "BUY", buy_price))
            pending_buy = False

        # ---- 2. 当日收盘盯市 ----
        equity = cash + position * close
        equity_curve.append(equity)
        dates.append(date)

        # ---- 3. 收盘后产生"下一日开盘"待执行订单 ----
        if position > 0:
            days_held += 1
            pnl = close / entry_price - 1.0

            # (a) ATR 动态止损：任意交易日触发
            if entry_atr_abs is not None and close < entry_price - K_ATR_STOP * entry_atr_abs:
                pending_sell = "ATR"
            # (b) 最长持仓
            elif days_held >= MAX_HOLD_DAYS:
                pending_sell = "MAXHOLD"
            # (c) 周五止盈 / 弱势退出
            elif weekday == 5:
                if pnl > TAKE_PROFIT:
                    pending_sell = "TP"
                elif pnl < 0.0:
                    pending_sell = "LOSS"

        # 周一空仓：训练 + 预测（带概率阈值）
        if position == 0 and weekday == 1 and not pending_buy:
            try:
                X, y, latest_x = build_dataset(df, t_end_idx=i, n=N, m=M)
            except RuntimeError:
                continue
            sm.fit(X, y)
            proba_pos = sm.predict_proba_pos(latest_x)
            pred = 1 if proba_pos > PROBA_THRESH else 0

            # 记录该次预测的 ground truth（沿用原始定义便于横向对照）
            if i + M < len(df):
                y0 = float(df.iloc[i + 1]["close"])
                y1 = float(df.iloc[i + M]["close"])
                y_true = 1 if y1 > y0 else 0
                predictions.append((date, pred, y_true))

            if pred == 1:
                pending_buy = True

    # 最后一天强制平仓（方便统计）
    if position > 0:
        last_close = float(df.iloc[-1]["close"])
        sell_price = last_close * (1 - SLIPPAGE)
        cash = position * sell_price * (1 - COMMISSION)
        trade_pnls.append(sell_price / entry_price - 1.0)
        position = 0.0
        entry_price = None
        trades.append((df.iloc[-1]["trade_date"], "SELL_END", sell_price))
        equity_curve.append(cash)
        dates.append(df.iloc[-1]["trade_date"])

    res = pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity_curve}).drop_duplicates("date").set_index("date")
    res["ret"] = res["equity"].pct_change().fillna(0.0)
    res["cummax"] = res["equity"].cummax()
    res["dd"] = res["equity"] / res["cummax"] - 1.0

    return {"perf": res, "trades": trades,
            "predictions": predictions, "trade_pnls": trade_pnls}


# -------------------------
# 模型与策略评估
# -------------------------
def evaluate_model(predictions: list) -> dict:
    """评估 SVM 分类质量。predictions: list of (date, y_pred, y_true)"""
    if not predictions:
        return {}
    y_pred = np.array([p[1] for p in predictions])
    y_true = np.array([p[2] for p in predictions])

    n = len(y_true)
    pos_rate = float(y_true.mean())   # 正样本占比（基线对照）
    pred_pos_rate = float(y_pred.mean())

    acc = accuracy_score(y_true, y_pred)
    # 只有一个类别时 precision/recall 会报警，做兜底
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "n": n,
        "pos_rate": pos_rate,
        "pred_pos_rate": pred_pos_rate,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cm": cm,
    }


def evaluate_strategy(perf: pd.DataFrame, trade_pnls: list, df: pd.DataFrame) -> dict:
    """补充策略层面指标：Sharpe、胜率、盈亏比、买入持有基准。"""
    daily_ret = perf["ret"]
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = 0.0
        pl_ratio = 0.0

    # 买入持有基准（同区间）
    bh_ret = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)

    return {
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pl_ratio": pl_ratio,
        "bh_ret": bh_ret,
    }

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

    model_metrics_var = tk.StringVar(value="")
    model_metrics_label = ttk.Label(
        root, textvariable=model_metrics_var, padding=(10, 0),
        justify="left", foreground="#444"
    )
    model_metrics_label.pack(side=tk.TOP, anchor="w")

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
    ax2 = fig.add_subplot(212, sharex=ax1)
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
            predictions = []
            trade_pnls = []
            if mode == "backtest":
                out = backtest(df_stock)
                perf = out["perf"]
                trades = out["trades"]
                predictions = out.get("predictions", [])
                trade_pnls = out.get("trade_pnls", [])
            else:
                # 实盘预测：根据真实时间决定使用哪一次"周一收盘"作为信号
                df_stock = add_indicators(df_stock)
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
                sm = StrategyModel(cv_refresh=CV_REFRESH)
                sm.fit(X, y)
                proba_pos = sm.predict_proba_pos(latest_x)
                pred = 1 if proba_pos > PROBA_THRESH else 0
                last_date = df_stock.iloc[i]["trade_date"].strftime("%Y-%m-%d")
                next_open = (df_stock.iloc[i]["trade_date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                action = "【买入】" if pred == 1 else "【不买入】"
                signal_msg = (
                    f"实盘预测：{last_date} 收盘信号 -> {next_open} 开盘{action}"
                    f"    P(涨)={proba_pos:.2%}  阈值={PROBA_THRESH:.0%}"
                )
                if now.weekday() == 4:
                    signal_msg += "    提醒：今日周五，请人工检查止盈/弱势退出规则"
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return

        if mode == "live":
            metrics_var.set("")
            model_metrics_var.set("")
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

            strat = evaluate_strategy(perf, trade_pnls, df_stock)
            metrics_var.set(
                f"总收益: {total_ret*100:.2f}%    "
                f"年化: {ann_ret*100:.2f}%    "
                f"最大回撤: {max_dd*100:.2f}%    "
                f"Sharpe: {strat['sharpe']:.2f}    "
                f"胜率: {strat['win_rate']*100:.1f}%    "
                f"盈亏比: {strat['pl_ratio']:.2f}    "
                f"基准(买入持有): {strat['bh_ret']*100:.2f}%    "
                f"交易次数: {len(trades)}"
            )

            mm = evaluate_model(predictions)
            if mm:
                cm = mm["cm"]  # [[TN,FP],[FN,TP]]
                model_metrics_var.set(
                    f"[模型] 样本数: {mm['n']}    "
                    f"准确率: {mm['accuracy']*100:.2f}%    "
                    f"Precision: {mm['precision']*100:.2f}%    "
                    f"Recall: {mm['recall']*100:.2f}%    "
                    f"F1: {mm['f1']:.3f}    "
                    f"正样本率: {mm['pos_rate']*100:.1f}%    "
                    f"预测为1占比: {mm['pred_pos_rate']*100:.1f}%    "
                    f"混淆矩阵 TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}"
                )
            else:
                model_metrics_var.set("[模型] 无可评估预测样本")

            signal_var.set("")
        else:
            metrics_var.set("")
            model_metrics_var.set("")
            signal_var.set(signal_msg)

        # 更新右侧交易记录
        trades_list.delete(0, tk.END)
        if trades:
            for d, typ, p in trades:
                trades_list.insert(tk.END, f"{d.strftime('%Y-%m-%d')} {typ} {p:.2f}")
        elif signal_msg:
            trades_list.insert(tk.END, signal_msg)

        # 对齐两图的日期横轴：以股价完整区间为准
        if len(stock_close.index) > 0:
            ax1.set_xlim(stock_close.index.min(), stock_close.index.max())

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

    ui_queue: queue.Queue = queue.Queue()

    def _format_result(r: dict) -> str:
        if "error" in r:
            return f"{r['code']:<11}  错误: {r['error']}"
        tag = "[BUY]" if r["action"] == "买入" else "[----]"
        return (
            f"{tag} {r['code']:<11}  P={r['proba']:.1%}  【{r['action']}】  "
            f"{r['next_open']}开盘≈¥{r['ref_price']:.2f}  "
            f"止损¥{r['stop_loss']:.2f}  最长{r['max_hold']}日"
        )

    def _poll_ui_queue():
        try:
            while True:
                kind, payload = ui_queue.get_nowait()
                if kind == "progress":
                    signal_var.set(payload)
                elif kind == "result":
                    live_results.insert(tk.END, _format_result(payload))
                elif kind == "done":
                    run_btn.configure(state="normal")
                    live_btn.configure(state="normal")
                    return
                elif kind == "error":
                    messagebox.showerror("错误", payload)
                    signal_var.set("批量预测失败")
                    run_btn.configure(state="normal")
                    live_btn.configure(state="normal")
                    return
        except queue.Empty:
            pass
        root.after(100, _poll_ui_queue)

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
            # 清空结果，禁用按钮，启动后台线程
            live_results.delete(0, tk.END)
            metrics_var.set("")
            model_metrics_var.set("")
            signal_var.set("准备拉取数据 ...")
            run_btn.configure(state="disabled")
            live_btn.configure(state="disabled")

            def _emit(kind, payload):
                ui_queue.put((kind, payload))

            worker = threading.Thread(
                target=run_live_batch, args=(codes, token, _emit), daemon=True
            )
            worker.start()
            root.after(100, _poll_ui_queue)

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
