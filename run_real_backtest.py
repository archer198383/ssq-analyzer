#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 ssq_history.db 中的真实历史数据跑回测（替代脚本自带的随机数据兜底）。"""
import sqlite3
import pandas as pd
from backtest_ssq import run_large_scale_backtest

conn = sqlite3.connect("ssq_history.db")
rows = conn.execute(
    "SELECT issue, r1, r2, r3, r4, r5, r6, blue FROM lottery_records ORDER BY issue ASC"
).fetchall()
conn.close()

records = [
    {"issue": str(r[0]), "reds": sorted([r[1], r[2], r[3], r[4], r[5], r[6]]), "blue": r[7]}
    for r in rows
]
df = pd.DataFrame(records)
print(f"[i] 载入真实历史数据 {len(df)} 期（{df.iloc[0]['issue']} ~ {df.iloc[-1]['issue']}）")
run_large_scale_backtest(df, lookback_window=50)
