#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上期推荐结果汇总：对最近 N 期，分别用其之前的数据生成推荐并与实际开奖核对。"""
import random
import sqlite3
import pandas as pd
from ssq_analyzer import QuantitativeEngine
from backtest_ssq import calculate_prize

N = 5

conn = sqlite3.connect("ssq_history.db")
df = pd.read_sql_query("SELECT * FROM lottery_records ORDER BY issue ASC", conn)
conn.close()

print("共 %d 期数据，对最近 %d 期做推荐结果验证" % (len(df), N))
print("=" * 72)
total_money = 0
for k in range(1, N + 1):
    idx = len(df) - k
    df_before = df.iloc[:idx].reset_index(drop=True)
    actual = df.iloc[idx]
    act_reds = sorted(int(actual["r%d" % i]) for i in range(1, 7))
    act_blue = int(actual["blue"])
    random.seed(20260000 + idx)
    anchor, candidates = QuantitativeEngine.generate_enhanced_portfolio(df_before)
    reds_str = " ".join("%02d" % x for x in act_reds)
    print("")
    print("【第 %s 期】实际开奖：红球 %s 蓝球 %02d" % (actual["issue"], reds_str, act_blue))
    period_money = 0
    for c in candidates:
        rh = len(set(c["reds"]) & set(act_reds))
        bh = c["blue"] == act_blue
        tier, money = calculate_prize(rh, bh)
        period_money += money
        mark = "→ %d等奖 %d元" % (tier, money) if tier else "→ 未中奖"
        cn = " ".join("%02d" % x for x in sorted(c["reds"]))
        bmark = "蓝中" if bh else "蓝错"
        print("  第%02d注 %s+%02d (红中%d %s) %s" % (c["id"], cn, c["blue"], rh, bmark, mark))
    total_money += period_money
    print("  小计：投入10元，中奖 %d 元" % period_money)
print("")
print("=" * 72)
print("【%d 期汇总】总投入 %d 元，总中奖 %d 元，回报率 %.1f%%" % (N, N*10, total_money, total_money/(N*10)*100))
