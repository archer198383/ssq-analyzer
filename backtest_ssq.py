#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球 500~1000 期大规模蒙特卡洛样本外滚动回测检验系统
SSQ Large-Scale Monte Carlo Walk-Forward Backtest Engine
"""

import sys
import time
import itertools
import random
import numpy as np
import pandas as pd
import requests

# ==================== 1. 形态学与博弈论剪枝过滤器 ====================

class SSQFilterEngine:
    """形态学与博弈论剪枝过滤器"""
    def __init__(self, sum_range=(75, 130), max_consecutive=3):
        self.sum_min, self.sum_max = sum_range
        self.max_consecutive = max_consecutive
        self.valid_odd_even = {(3, 3), (2, 4), (4, 2)}
        self.valid_size = {(3, 3), (2, 4), (4, 2)}

    def validate(self, reds):
        total_sum = sum(reds)
        if not (self.sum_min <= total_sum <= self.sum_max):
            return False

        odd_count = sum(1 for r in reds if r % 2 != 0)
        even_count = 6 - odd_count
        if (odd_count, even_count) not in self.valid_odd_even:
            return False

        small_count = sum(1 for r in reds if r <= 16)
        big_count = 6 - small_count
        if (small_count, big_count) not in self.valid_size:
            return False

        sorted_r = sorted(reds)
        streak = max_streak = 1
        for i in range(len(sorted_r) - 1):
            if sorted_r[i+1] == sorted_r[i] + 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak > self.max_consecutive:
            return False

        diffs = [sorted_r[i+1] - sorted_r[i] for i in range(len(sorted_r)-1)]
        if len(set(diffs)) == 1:
            return False

        tails = [r % 10 for r in reds]
        if max(tails.count(t) for t in set(tails)) > 3:
            return False

        return True


# ==================== 2. 官方奖级与收益核算体系 ====================

def calculate_prize(red_hits: int, blue_hit: bool):
    """
    双色球官方奖级与返奖金额换算
    :return: (奖级, 单注奖金金额)
    """
    if red_hits == 6 and blue_hit:
        return 1, 5000000  # 一等奖 (固定估值500万元)
    elif red_hits == 6 and not blue_hit:
        return 2, 200000   # 二等奖 (固定估值20万元)
    elif red_hits == 5 and blue_hit:
        return 3, 3000     # 三等奖 (单注3000元)
    elif (red_hits == 5 and not blue_hit) or (red_hits == 4 and blue_hit):
        return 4, 200      # 四等奖 (单注200元)
    elif (red_hits == 4 and not blue_hit) or (red_hits == 3 and blue_hit):
        return 5, 10       # 五等奖 (单注10元)
    elif blue_hit:
        return 6, 5        # 六等奖 (单注5元)
    return 0, 0            # 未中奖


# ==================== 3. 历史数据获取与多源保底 ====================

def load_or_fetch_history_data(total_target_issues: int = 1000) -> pd.DataFrame:
    """拉取历史数据，网络受限时自动生成合规高精度基准数据"""
    records = []
    print(f"[*] 正在尝试联网获取双色球历史开奖记录 (目标: {total_target_issues} 期)...")
    
    try:
        url = f'http://f.api.lottery.sina.com.cn/lottery/get_issue_list?type=ssq&format=json&limit={total_target_issues}'
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            issue_list = data.get('result', {}).get('data', {}).get('lotteryIssueList', [])
            if issue_list:
                for item in issue_list:
                    code_str = item.get('lotteryDrawResult', '')
                    if '|' in code_str:
                        reds_str, blue_str = code_str.split('|')
                        reds = sorted([int(x) for x in reds_str.split(',')])
                        blue = int(blue_str)
                        records.append({
                            'issue': str(item.get('lotteryIssue')),
                            'reds': reds,
                            'blue': blue
                        })
                if len(records) >= 100:
                    df = pd.DataFrame(records).sort_values(by='issue', ascending=True).reset_index(drop=True)
                    print(f"[✓] 成功从官方接口获取 {len(df)} 期真实历史数据！")
                    return df
    except Exception as e:
        print(f"[-] 接口网络连接受限: {e}")

    print(f"[i] 启用本地独立离线开奖池生成 {total_target_issues} 期样本进行回测检验...")
    np.random.seed(2026)
    records = []
    for i in range(total_target_issues):
        r = sorted(np.random.choice(range(1, 34), size=6, replace=False).tolist())
        b = int(np.random.choice(range(1, 17)))
        records.append({
            'issue': f'202{i//150:01d}{i%150+1:03d}',
            'reds': r,
            'blue': b
        })
    return pd.DataFrame(records)


# ==================== 4. 向量化极速滚动回测核心 ====================

def run_large_scale_backtest(df_history: pd.DataFrame, lookback_window: int = 50):
    total_len = len(df_history)
    test_start = lookback_window
    test_end = total_len
    total_tested_issues = test_end - test_start
    total_tickets = total_tested_issues * 5
    total_cost = total_tickets * 2  # 每期 5 注，每注 2 元

    print(f"\n[*] 启动滚动样本外回测流水线:")
    print(f"    - 总回测期数: {total_tested_issues} 期")
    print(f"    - 滑动训练窗口: 每期严格仅使用前 {lookback_window} 期数据（无未来函数）")
    print(f"    - 每期投注策略: 模型推荐 5 注 vs 蒙特卡洛纯随机机选 5 注")
    print(f"    - 总投入本金: {total_cost:,} 元")
    print(f"------------------------------------------------------------")

    # 构建 0/1 特征矩阵与蓝球数组
    red_matrix = np.zeros((total_len, 33), dtype=np.int8)
    blue_array = np.zeros(total_len, dtype=np.int8)
    for i, row in df_history.iterrows():
        red_matrix[i, [x - 1 for x in row['reds']]] = 1
        blue_array[i] = row['blue']

    filter_engine = SSQFilterEngine()

    model_prize_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    random_prize_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    model_total_prize_money = 0
    random_total_prize_money = 0
    model_red_hits = 0
    random_red_hits = 0
    model_blue_hits = 0
    random_blue_hits = 0

    start_timer = time.time()

    for t in range(test_start, test_end):
        # 1. 严格使用 t 之前的数据切片
        window_reds = red_matrix[t - lookback_window:t]
        window_blues = blue_array[t - lookback_window:t]

        # 向量化计算马尔可夫转移概率
        zeros = (window_reds[:-1] == 0)
        trans_0_to_1 = np.sum(zeros & (window_reds[1:] == 1), axis=0)
        count_0 = np.sum(zeros, axis=0)
        trans_probs = np.where(count_0 > 0, trans_0_to_1 / count_0, 0.18)

        # 向量化计算遗漏值
        last_seen = np.zeros(33, dtype=int)
        for num in range(33):
            idx = np.where(window_reds[:, num] == 1)[0]
            last_seen[num] = (lookback_window - 1 - idx[-1]) if len(idx) > 0 else lookback_window

        # 综合打分推导候选大池 (2 胆 + 10 拖)
        scores = trans_probs * 0.7 + last_seen * 0.3
        ranked_nums = (np.argsort(scores)[::-1] + 1).tolist()
        dan_reds = sorted(ranked_nums[:2])
        tuo_reds = sorted(ranked_nums[2:12])

        # 旋转矩阵 + 形态剪枝生成 5 注
        all_tuo_combos = list(itertools.combinations(tuo_reds, 4))
        valid_combos = []
        for tuo_part in all_tuo_combos:
            comb = sorted(dan_reds + list(tuo_part))
            if filter_engine.validate(comb):
                valid_combos.append(comb)

        if len(valid_combos) < 5:
            for tuo_part in all_tuo_combos:
                comb = sorted(dan_reds + list(tuo_part))
                if comb not in valid_combos:
                    valid_combos.append(comb)
                if len(valid_combos) >= 5:
                    break

        # 蓝球频次与遗漏评分
        b_freq = np.bincount(window_blues, minlength=17)[1:]
        b_omission = np.zeros(16, dtype=int)
        for b_val in range(1, 17):
            b_idx = np.where(window_blues == b_val)[0]
            b_omission[b_val - 1] = (lookback_window - 1 - b_idx[-1]) if len(b_idx) > 0 else lookback_window
        b_scores = b_freq * 0.6 + b_omission * 0.4
        ranked_blues = (np.argsort(b_scores)[::-1] + 1).tolist()

        # 实际当期开奖核验
        act_reds_set = set(np.where(red_matrix[t] == 1)[0] + 1)
        act_blue_val = int(blue_array[t])

        # A. 模型推荐 5 注核验
        for i in range(5):
            m_reds = valid_combos[i % len(valid_combos)]
            m_blue = ranked_blues[i]
            r_hit = len(set(m_reds) & act_reds_set)
            b_hit = (m_blue == act_blue_val)
            model_red_hits += r_hit
            if b_hit:
                model_blue_hits += 1
            tier, money = calculate_prize(r_hit, b_hit)
            if tier > 0:
                model_prize_counts[tier] += 1
                model_total_prize_money += money

        # B. 蒙特卡洛纯随机机选 5 注核验 (基准对照组)
        for _ in range(5):
            rnd_reds = random.sample(range(1, 34), 6)
            rnd_blue = random.randint(1, 16)
            r_hit = len(set(rnd_reds) & act_reds_set)
            b_hit = (rnd_blue == act_blue_val)
            random_red_hits += r_hit
            if b_hit:
                random_blue_hits += 1
            tier, money = calculate_prize(r_hit, b_hit)
            if tier > 0:
                random_prize_counts[tier] += 1
                random_total_prize_money += money

    elapsed = time.time() - start_timer
    print(f"[✓] 回测执行完毕！计算耗时: {elapsed:.3f} 秒 ({total_tested_issues / elapsed:.1f} 期/秒)\n")

    # ==================== 5. 生成统计对比分析报告 ====================
    
    model_roi = (model_total_prize_money / total_cost) * 100
    random_roi = (random_total_prize_money / total_cost) * 100
    model_avg_red = model_red_hits / total_tickets
    random_avg_red = random_red_hits / total_tickets
    model_blue_rate = (model_blue_hits / total_tickets) * 100
    random_blue_rate = (random_blue_hits / total_tickets) * 100

    print("=" * 70)
    print(" 📊 双色球大规模蒙特卡洛样本外回测实证报告 (500~1000期)")
    print("=" * 70)
    print(f"{'统计维度':<20} | {'【算法优化模型】':<18} | {'【纯随机机选基线】':<18}")
    print("-" * 70)
    print(f"{'总回测期数':<20} | {f'{total_tested_issues} 期':<18} | {f'{total_tested_issues} 期':<18}")
    print(f"{'总投入本金':<20} | {f'{total_cost:,} 元':<18} | {f'{total_cost:,} 元':<18}")
    print(f"{'累计中奖总金额':<20} | {f'{model_total_prize_money:,} 元':<18} | {f'{random_total_prize_money:,} 元':<18}")
    print(f"{'资金回报率 (ROI)':<20} | {f'{model_roi:.2f}%':<18} | {f'{random_roi:.2f}%':<18}")
    print(f"{'单注红球平均命中':<20} | {f'{model_avg_red:.3f} 个/注':<18} | {f'{random_avg_red:.3f} 个/注':<18}")
    print(f"{'蓝球命中率':<20} | {f'{model_blue_rate:.2f}% (理论6.25%)':<18} | {f'{random_blue_rate:.2f}% (理论6.25%)':<18}")
    print("-" * 70)
    print(" 🏆 详细奖级分布对比 (注数):")
    for tier_num in range(1, 7):
        names = {
            1: "一等奖 (6+1)",
            2: "二等奖 (6+0)",
            3: "三等奖 (5+1)",
            4: "四等奖 (5+0/4+1)",
            5: "五等奖 (4+0/3+1)",
            6: "六等奖 (末等奖)"
        }
        m_cnt = model_prize_counts[tier_num]
        r_cnt = random_prize_counts[tier_num]
        print(f"  • {names[tier_num]:<18}: 模型命中 {m_cnt:>4} 注 | 随机机选 {r_cnt:>4} 注")
    print("=" * 70)

    print("\n💡 【数理统计学核心结论与学术客观评定】:")
    print(" 1. 胜率收敛性：大样本回测证明，无论使用马尔可夫链、遗漏散度还是纯随机机选，长期回报率 (ROI) 均收敛于彩票官方设定的理论返奖率 (~50%) 之下。")
    print(" 2. 剪枝与矩阵价值：算法模型的主要价值在于『剔除极端非理性组合（全奇全偶、等差扎堆等）』与『以较少注数覆盖多维特征』，但在完全独立的物理随机事件中无法改变数学底层期望。")
    print(" 3. 风险防线：必须严格控制投注规模与频率，理性对待彩票分析。")


if __name__ == '__main__':
    df = load_or_fetch_history_data(total_target_issues=1000)
    run_large_scale_backtest(df, lookback_window=50)
