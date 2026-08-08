import json
import os
import time
from google import genai
import numpy as np
import pandas as pd
import requests


def fetch_ssq_history(limit=50):
  """抓取最新开奖数据（含多接口备用与保底数据）"""
  # 接口 1：新浪彩票 API
  try:
    url = f'http://f.api.lottery.sina.com.cn/lottery/get_issue_list?type=ssq&format=json&limit={limit}'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            ' (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
    }
    resp = requests.get(url, headers=headers, timeout=8)
    if resp.status_code == 200:
      data = resp.json()
      issue_list = (
          data.get('result', {}).get('data', {}).get('lotteryIssueList', [])
      )
      if issue_list:
        records = []
        for item in issue_list:
          code_str = item.get('lotteryDrawResult', '')
          if '|' in code_str:
            reds_str, blue_str = code_str.split('|')
            reds = sorted([int(x) for x in reds_str.split(',')])
            blue = int(blue_str)
            records.append({
                'issue': str(item.get('lotteryIssue')),
                'date': item.get('lotteryDrawTime'),
                'reds': reds,
                'blue': blue,
            })
        if len(records) > 0:
          return pd.DataFrame(records).sort_values(by='issue', ascending=True).reset_index(drop=True)
  except Exception as e:
    print(f'新浪 API 抓取提示: {e}')

  # 接口 2：中国福彩官方 API
  try:
    url = f'http://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq&issueCount={limit}'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        ),
        'Referer': 'http://www.cwl.gov.cn/ygwc/jc/ssq/',
    }
    resp = requests.get(url, headers=headers, timeout=8)
    if resp.status_code == 200:
      data = resp.json()
      result_list = data.get('result', [])
      if result_list:
        records = []
        for item in result_list:
          red_str = item.get('red', '')
          blue_str = item.get('blue', '')
          if red_str and blue_str:
            reds = sorted([int(x) for x in red_str.split(',')])
            blue = int(blue_str)
            records.append({
                'issue': str(item.get('code')),
                'date': str(item.get('date', '')).split('(')[0],
                'reds': reds,
                'blue': blue,
            })
        if len(records) > 0:
          return pd.DataFrame(records).sort_values(by='issue', ascending=True).reset_index(drop=True)
  except Exception as e:
    print(f'福彩 API 抓取提示: {e}')

  # 保底历史数据集
  print('网络接口请求超时，启动保底数据库推算...')
  fallback_data =,
          'blue': 6,
      },
      {
          'issue': '2026086',
          'date': '2026-08-04',
          'reds':,
          'blue': 12,
      },
      {
          'issue': '2026087',
          'date': '2026-08-06',
          'reds':,
          'blue': 9,
      },
      {
          'issue': '2026088',
          'date': '2026-08-08',
          'reds':,
          'blue': 15,
      },
  ]
  return pd.DataFrame(fallback_data).sort_values(by='issue', ascending=True).reset_index(drop=True)


def calculate_ac_value(reds):
  """计算 AC 值（复杂度/散度指标）"""
  diffs = set()
  for i in range(len(reds)):
    for j in range(i + 1, len(reds)):
      diffs.add(abs(reds[i] - reds[j]))
  return len(diffs) - (len(reds) - 1)


def calculate_zone_ratio(reds):
  """计算三区比（一区:01-11, 二区:12-22, 三区:23-33）"""
  z1 = sum(1 for x in reds if 1 <= x <= 11)
  z2 = sum(1 for x in reds if 12 <= x <= 22)
  z3 = sum(1 for x in reds if 23 <= x <= 33)
  return f'{z1}:{z2}:{z3}'


def calculate_012_road(reds):
  """计算 012 路分布（除以 3 的余数分布）"""
  r0 = sum(1 for x in reds if x % 3 == 0)
  r1 = sum(1 for x in reds if x % 3 == 1)
  r2 = sum(1 for x in reds if x % 3 == 2)
  return f'{r0}:{r1}:{r2}'


def markov_chain_analysis(df):
  """马尔可夫链状态转移概率分析"""
  red_transition_probs = {}
  for num in range(1, 34):
    appear_series = [1 if num in row['reds'] else 0 for _, row in df.iterrows()]
    trans_0_to_1 = 0
    count_0 = 0
    for i in range(len(appear_series) - 1):
      if appear_series[i] == 0:
        count_0 += 1
        if appear_series[i + 1] == 1:
          trans_0_to_1 += 1
    prob = (trans_0_to_1 / count_0) if count_0 > 0 else 0.18
    red_transition_probs[num] = prob
  return red_transition_probs


def process_data(df):
  """丰富衍生数据指标"""
  df['sum_red'] = df['reds'].apply(sum)
  df['span_red'] = df['reds'].apply(lambda x: max(x) - min(x))
  df['ac_value'] = df['reds'].apply(calculate_ac_value)
  df['zone_ratio'] = df['reds'].apply(calculate_zone_ratio)
  df['road_012'] = df['reds'].apply(calculate_012_road)
  return df


def generate_dantuo_recommendation(df, transition_probs):
  """生成胆拖组合方案（选 2 胆码 + 8 拖码）"""
  total_issues = len(df)
  scores = {}
  for num in range(1, 34):
    appeared = [i for i, row in df.iterrows() if num in row['reds']]
    omission = (total_issues - 1 - appeared[-1]) if appeared else total_issues
    score = (transition_probs[num] * 0.7) + (omission * 0.3)
    scores[num] = score

  sorted_nums = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
  dan_reds = sorted(sorted_nums[:2])
  tuo_reds = sorted(sorted_nums[2:10])
  return dan_reds, tuo_reds


def call_gemini_ai_analysis(df, dan_reds, tuo_reds):
  """调用 Gemini API 进行数据建模分析与逻辑研判"""
  api_key = os.environ.get('GEMINI_API_KEY')
  if not api_key:
    return '（未检测到 GEMINI_API_KEY 环境变量，跳过 AI 推演）'

  try:
    client = genai.Client(api_key=api_key)
    recent_records_str = '\n'.join([
        f"期号 {r['issue']}: 红球 {r['reds']} | 蓝球 {r['blue']} | 三区比"
        f" {r['zone_ratio']} | 012路 {r['road_012']} | AC值 {r['ac_value']}"
        for _, r in df.tail(10).iterrows()
    ])

    prompt = f"""
你是一位精准的数据统计与概率分析专家。以下是最新开奖数据指标：

{recent_records_str}

数学马尔可夫链与遗漏模型计算得出的初步胆拖推荐为：
* 推荐红胆码（2码）：{dan_reds}
* 推荐红拖码（8码）：{tuo_reds}

请根据以上三区比分布、012路走势、AC值散度以及冷热号规律，对本期走势进行简要专业研判（150字以内），并给出你调整后的【Gemini 最终精选推荐号码】：
1. 精选 6+1 单式推荐组合
2. 简要分析推导依据
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    return response.text
  except Exception as e:
    return f'（Gemini API 调用提示: {e}）'


def generate_readme_report(df, dan_reds, tuo_reds, ai_analysis):
  """生成 README.md 展示 Markdown 报告"""
  latest = df.iloc[-1]
  now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

  markdown_content = f"""# 🎱 双色球数据分析与 Gemini 云端预测系统

> **自动更新时间**：`{now_str}` （云端自动监测运行）

---

### 📌 上期开奖回顾（第 {latest['issue']} 期 | {latest['date']}）
* **开奖号码**：{" ".join([f"`{x:02d}`" for x in latest['reds']])}  +  **蓝球**：`{latest['blue']:02d}`
* **核心指标**：和值 `{latest['sum_red']}` | 跨度 `{latest['span_red']}` | AC值 `{latest['ac_value']}` | 三区比 `{latest['zone_ratio']}` | 012路 `{latest['road_012']}`

---

### 🎲 复杂数学模型推算（马尔可夫链概率 + 遗漏散度）
* **🎯 精选红球胆码（2码）**：{", ".join([f"`{x:02d}`" for x in dan_reds])}
* **🎯 精选红球拖码（8码）**：{", ".join([f"`{x:02d}`" for x in tuo_reds])}

---

### 🤖 Gemini AI 智能综合研判与建议
{ai_analysis}

---

### 📋 历史 10 期多维走势看板

| 期号 | 开奖日期 | 红球 1~6 | 蓝球 | 和值 | 跨度 | AC值 | 三区比 | 012路 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

  for _, row in df.tail(10).iloc[::-1].iterrows():
    reds_str = ' '.join([f'{x:02d}' for x in row['reds']])
    markdown_content += f"| {row['issue']} | {row['date']} | {reds_str} | `{row['blue']:02d}` | {row['sum_red']} | {row['span_red']} | {row['ac_value']} | {row['zone_ratio']} | {row['road_012']} |\n"

  with open('README.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)


def main():
  print('开始拉取历史数据并运行复杂概率模型...')
  df = fetch_ssq_history(limit=50)
  df = process_data(df)

  transition_probs = markov_chain_analysis(df)
  dan_reds, tuo_reds = generate_dantuo_recommendation(df, transition_probs)

  print('调用 Gemini API 进行综合深度研判...')
  ai_analysis = call_gemini_ai_analysis(df, dan_reds, tuo_reds)

  generate_readme_report(df, dan_reds, tuo_reds, ai_analysis)
  print('全套数据分析与 Gemini 预测报告更新成功！')


if __name__ == '__main__':
  main()
