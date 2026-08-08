import os
import time
import numpy as np
import pandas as pd
import requests


def fetch_ssq_history(limit=50):
  """获取最新 limit 期双色球开奖数据"""
  url = f'http://f.api.lottery.sina.com.cn/lottery/get_issue_list?type=ssq&format=json&limit={limit}'
  headers = {
      'User-Agent': (
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      )
  }

  try:
    response = requests.get(url, headers=headers, timeout=12)
    data = response.json()
    issue_list = (
        data.get('result', {}).get('data', {}).get('lotteryIssueList', [])
    )

    records = []
    for item in issue_list:
      code_str = item.get('lotteryDrawResult', '')
      if '|' in code_str:
        reds_str, blue_str = code_str.split('|')
        reds = sorted([int(x) for x in reds_str.split(',')])
        blue = int(blue_str)
        sum_red = sum(reds)
        span_red = max(reds) - min(reds)
        odd_count = sum(1 for x in reds if x % 2 != 0)
        big_count = sum(1 for x in reds if x >= 17)

        records.append({
            'issue': str(item.get('lotteryIssue')),
            'date': item.get('lotteryDrawTime'),
            'r1': reds[0],
            'r2': reds,
            'r3': reds,
            'r4': reds,
            'r5': reds,
            'r6': reds,
            'blue': blue,
            'sum_red': sum_red,
            'span_red': span_red,
            'odd_even': f'{odd_count}:{6 - odd_count}',
            'big_small': f'{big_count}:{6 - big_count}',
        })

    df = pd.DataFrame(records)
    return df.sort_values(by='issue', ascending=True).reset_index(drop=True)
  except Exception as e:
    print(f'数据获取异常: {e}')
    return pd.DataFrame()


def analyze_trends(df):
  """分析红蓝球冷热与遗漏趋势"""
  total_issues = len(df)

  # 1. 红球分析
  red_stats = {}
  for num in range(1, 34):
    appeared = df[
        (df['r1'] == num)
        | (df['r2'] == num)
        | (df['r3'] == num)
        | (df['r4'] == num)
        | (df['r5'] == num)
        | (df['r6'] == num)
    ]
    freq = len(appeared)
    omission = (
        (total_issues - 1 - appeared.index[-1]) if not appeared.empty else total_issues
    )
    red_stats[num] = {'freq': freq, 'omission': omission}

  # 2. 蓝球分析
  blue_stats = {}
  for num in range(1, 17):
    appeared = df[df['blue'] == num]
    freq = len(appeared)
    omission = (
        (total_issues - 1 - appeared.index[-1]) if not appeared.empty else total_issues
    )
    blue_stats[num] = {'freq': freq, 'omission': omission}

  return red_stats, blue_stats


def generate_recommendation(red_stats, blue_stats):
  """概率打分算法（结合遗漏均值与冷热权重预测）"""
  red_scores = {}
  for num, stat in red_stats.items():
    score = (stat['freq'] * 0.6) + (stat['omission'] * 0.4)
    red_scores[num] = score

  top_reds = sorted(red_scores.keys(), key=lambda x: red_scores[x], reverse=True)[
      :10
  ]
  recommended_reds = sorted(top_reds[:6])

  blue_scores = {
      num: (stat['freq'] * 0.5 + stat['omission'] * 0.5)
      for num, stat in blue_stats.items()
  }
  top_blues = sorted(
      blue_scores.keys(), key=lambda x: blue_scores[x], reverse=True
  )[:2]

  return recommended_reds, top_blues


def generate_readme_report(df, red_stats, blue_stats, rec_reds, rec_blues):
  """生成 README.md 展示 Markdown 报告"""
  latest = df.iloc[-1]
  now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

  cold_reds = sorted(
      red_stats.keys(), key=lambda x: red_stats[x]['omission'], reverse=True
  )[:5]
  hot_reds = sorted(
      red_stats.keys(), key=lambda x: red_stats[x]['freq'], reverse=True
  )[:5]

  rec_reds_str = ' '.join([f'`{n:02d}`' for n in rec_reds])
  rec_blues_str = ' '.join([f'`{n:02d}`' for n in rec_blues])

  markdown_content = f"""# 🎱 双色球数据分析与趋势推算系统

> **自动更新时间**：`{now_str}` （云端自动监测运行）

---

### 📌 上期开奖结果回顾（第 {latest['issue']} 期 | {latest['date']}）
* **开奖号码**：`{latest['r1']:02d}` `{latest['r2']:02d}` `{latest['r3']:02d}` `{latest['r4']:02d}` `{latest['r5']:02d}` `{latest['r6']:02d}`  +  **蓝球**：`{latest['blue']:02d}`
* **关键指标**：和值 `{latest['sum_red']}` | 跨度 `{latest['span_red']}` | 奇偶比 `{latest['odd_even']}` | 大小比 `{latest['big_small']}`

---

### 📊 遗漏与冷热趋势（统计近 {len(df)} 期）
* **🔥 最热红球**：{', '.join([f'`{n:02d}`号({red_stats[n]["freq"]}次)' for n in hot_reds])}
* **🧊 遗漏最长红球（冷号）**：{', '.join([f'`{n:02d}`号(遗漏{red_stats[n]["omission"]}期)' for n in cold_reds])}

---

### 🔮 下期概率推算与参考组合
* **精选推荐红球（6码）**：{rec_reds_str}
* **推荐参考蓝球（2码）**：{rec_blues_str}

---

### 📋 历史开奖记录（最近 10 期）

| 期号 | 开奖日期 | 红球 1~6 | 蓝球 | 和值 | 跨度 | 奇偶比 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

  for _, row in df.tail(10).iloc[::-1].iterrows():
    reds_str = f"{row['r1']:02d} {row['r2']:02d} {row['r3']:02d} {row['r4']:02d} {row['r5']:02d} {row['r6']:02d}"
    markdown_content += f"| {row['issue']} | {row['date']} | {reds_str} | `{row['blue']:02d}` | {row['sum_red']} | {row['span_red']} | {row['odd_even']} |\n"

  with open('README.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)


def main():
  print('开始抓取双色球最新数据...')
  df = fetch_ssq_history(limit=50)
  if df.empty:
    print('数据抓取失败，退出运行')
    return

  red_stats, blue_stats = analyze_trends(df)
  rec_reds, rec_blues = generate_recommendation(red_stats, blue_stats)
  generate_readme_report(df, red_stats, blue_stats, rec_reds, rec_blues)
  print('报告生成完毕！')


if __name__ == '__main__':
  main()
