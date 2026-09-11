import os
import sqlite3
import random
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from google import genai

# ----------------------------------------------------------------------
# 1. 数据库管理与官方数据同步模块 (SQLite + CWL API + Pandas)
# ----------------------------------------------------------------------
class SSQDataManager:
    def __init__(self, db_path: str = "ssq_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lottery_records (
                    issue TEXT PRIMARY KEY,
                    date TEXT,
                    r1 INTEGER, r2 INTEGER, r3 INTEGER,
                    r4 INTEGER, r5 INTEGER, r6 INTEGER,
                    blue INTEGER,
                    sales TEXT,
                    pool TEXT
                )
            """)
            conn.commit()

    def sync_official_data(self, fetch_count: int = 100):
        """从中国福彩网官方接口增量抓取最新开奖并持久化到 SQLite"""
        url = f"https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq&issueCount={fetch_count}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.cwl.gov.cn/"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json().get("result", [])
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    for item in data:
                        issue = item.get("code")
                        date = item.get("date", "")[:10]
                        reds = [int(x) for x in item.get("red", "").split(",") if x.isdigit()]
                        blue = int(item.get("blue", 0))
                        if len(reds) == 6 and blue > 0:
                            cursor.execute("""
                                INSERT OR IGNORE INTO lottery_records 
                                (issue, date, r1, r2, r3, r4, r5, r6, blue, sales, pool)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (issue, date, reds[0], reds, reds, reds, reds[4], reds[5], 
                                  blue, item.get("sales", "0"), item.get("poolmoney", "0")))
                    conn.commit()
                print("[OK] 官方最新开奖数据已同步入库。")
            else:
                print(f"[Warn] 官方接口返回异常代码: {resp.status_code}")
        except Exception as e:
            print(f"[Warn] 官方接口同步跳过: {e}")

    def load_dataframe(self) -> pd.DataFrame:
        """加载历史数据为结构化 Pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM lottery_records ORDER BY issue ASC", conn)
        return df

# ----------------------------------------------------------------------
# 2. 量化特征与博弈论（EV优化）引擎 (NumPy / Pandas)
# ----------------------------------------------------------------------
class QuantitativeEngine:
    PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    @staticmethod
    def calculate_ac_value(reds: List[int]) -> int:
        """计算红球数字复杂度 AC 值"""
        diffs = {abs(reds[j] - reds[i]) for i in range(len(reds)) for j in range(i + 1, len(reds))}
        return len(diffs) - (len(reds) - 1)

    @classmethod
    def extract_features(cls, reds: List[int], blue: int) -> Dict:
        sorted_red = sorted(reds)
        red_arr = np.array(sorted_red)
        
        red_sum = int(np.sum(red_arr))
        span = int(red_arr[-1] - red_arr[0])
        odd_count = int(np.sum(red_arr % 2 != 0))
        prime_count = sum(1 for x in sorted_red if x in cls.PRIME_NUMBERS)
        
        # 三区比 (01-11, 12-22, 23-33)
        z1 = int(np.sum((red_arr >= 1) & (red_arr <= 11)))
        z2 = int(np.sum((red_arr >= 12) & (red_arr <= 22)))
        z3 = int(np.sum((red_arr >= 23) & (red_arr <= 33)))
        
        # 连号统计
        consecutive_pairs = int(np.sum(np.diff(red_arr) == 1))
        ac_val = cls.calculate_ac_value(sorted_red)
        
        # 博弈论指标：1-31 大众生日号密集度
        birthday_count = int(np.sum(red_arr <= 31))

        return {
            "sum": red_sum,
            "span": span,
            "odd_even": f"{odd_count}:{6 - odd_count}",
            "prime_comp": f"{prime_count}:{6 - prime_count}",
            "zone_ratio": f"{z1}:{z2}:{z3}",
            "consecutive": consecutive_pairs,
            "ac_value": ac_val,
            "birthday_count": birthday_count,
            "blue": blue
        }

    @classmethod
    def filter_and_ev_optimize(cls, reds: List[int], blue: int) -> bool:
        """科学形态过滤 + 博弈论去热门化"""
        feats = cls.extract_features(reds, blue)
        
        # 1. 过滤偏离正态分布的极端和值（核心概率区间 80-130，仅作为内部过滤）
        if not (80 <= feats["sum"] <= 130):
            return False
            
        # 2. 跨度极差过滤（合理范围 18-30）
        if not (18 <= feats["span"] <= 30):
            return False
            
        # 3. 奇偶形态过滤（排除极端全奇全偶）
        if feats["odd_even"] in ["0:6", "6:0"]:
            return False
            
        # 4. AC值过滤（过滤低复杂度/等差规律组合）
        if feats["ac_value"] < 6:
            return False
            
        # 5. 限制长连号（最多允许2组2连号）
        if feats["consecutive"] > 2:
            return False
            
        # 6. 博弈论/期望收益（EV）去热门化：
        # 适度规避全由 1-31 生日号构成的组合，配置高位号码以降低均分头奖风险
        if feats["birthday_count"] == 6 and random.random() < 0.65:
            return False
            
        return True

    @classmethod
    def generate_candidate_pool(cls, n_picks: int = 5) -> List[Tuple[List[int], int, Dict]]:
        picks = []
        pool_reds = list(range(1, 34))
        while len(picks) < n_picks:
            reds = sorted(random.sample(pool_reds, 6))
            blue = random.randint(1, 16)
            if cls.filter_and_ev_optimize(reds, blue):
                feats = cls.extract_features(reds, blue)
                picks.append((reds, blue, feats))
        return picks

# ----------------------------------------------------------------------
# 3. 基于 google-genai 的 Gemini AI 研判模块
# ----------------------------------------------------------------------
def generate_gemini_analysis(df: pd.DataFrame, candidates: List[Tuple[List[int], int, Dict]]) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "> ⚠️ 未检测到 `GEMINI_API_KEY` 环境变量，跳过 AI 研判生成。"

    try:
        client = genai.Client(api_key=api_key)
        recent_df = df.tail(5)[["issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"]]
        recent_records = recent_df.to_dict(orient="records")
        
        summary_prompt = f"""
你是一位专业的彩票量化精算分析师。请结合双色球历史客观数据与最新候选组合，提供简要的数理推演点评（150-250字）。

【最新5期开奖数据】：
{recent_records}

【候选组合】：
{[{"reds": c[0], "blue": c} for c in candidates]}

【要求】：
1. 语言简练客观，严禁绝对化预测。
2. 简评上期形态偏离（如三区、连号）对当期均值回归的指引。
3. 从博弈论角度简述避开大众集中选号对期望收益（EV）的意义。
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"> ⚠️ Gemini AI 研判生成提示: {str(e)}"

# ----------------------------------------------------------------------
# 4. 主流程：极简自适应卡片看板输出（无表格，防左右拉动）
# ----------------------------------------------------------------------
def main():
    print("=== 开始运行双色球量化分析工作流 ===")
    data_mgr = SSQDataManager()
    data_mgr.sync_official_data(fetch_count=100)
    df = data_mgr.load_dataframe()
    
    if df.empty:
        print("[Error] 未获取到历史数据，终止流程。")
        return
        
    latest = df.iloc[-1]
    latest_reds = [int(latest[f"r{i}"]) for i in range(1, 7)]
    latest_blue = int(latest["blue"])
    latest_feats = QuantitativeEngine.extract_features(latest_reds, latest_blue)
    
    print(f"最新一期: {latest['issue']} ({latest['date']}) | 红球: {latest_reds} | 蓝球: {latest_blue}")
    
    # 筛选 5 组精选候选组合
    candidates = QuantitativeEngine.generate_candidate_pool(n_picks=5)
    
    # 调用 Gemini AI 分析
    ai_commentary = generate_gemini_analysis(df, candidates)
    
    # 组装极简 Markdown 看板（方案二：卡片式列表）
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    next_issue = int(latest['issue']) + 1 if str(latest['issue']).isdigit() else "下期"
    
    lines = [
        "# 🔴🔵 双色球量化分析看板",
        "",
        f"> 🕒 **更新时间**：`{current_time_str}` ｜ **期号**：第 `{next_issue}` 期推演",
        "",
        "---",
        "",
        f"### 📋 上期开奖总结（第 `{latest['issue']}` 期）",
        f"- **开奖号码**：红球 {' '.join(f'`{x:02d}`' for x in latest_reds)} ｜ 蓝球 `{latest_blue:02d}`",
        f"- **奖池滚存**：约 `{latest['pool']}` 元",
        f"- **形态特征**：三区比 `{latest_feats['zone_ratio']}` ｜ 奇偶比 `{latest_feats['odd_even']}` ｜ 连号组数 `{latest_feats['consecutive']}` ｜ 跨度 `{latest_feats['span']}` ｜ AC值 `{latest_feats['ac_value']}`",
        "",
        "---",
        "",
        "### 🎯 本期推荐组合（卡片式）",
        ""
    ]
    
    # 方案二：生成纯自适应卡片式列表（不产生任何横向滑动条）
    for i, (reds, blue, _) in enumerate(candidates, 1):
        red_str = " ".join(f"`{x:02d}`" for x in reds)
        lines.append(f"* 🔴 **第 {i:02d} 注**：{red_str} ＋ 🔵 `{blue:02d}`")
        
    lines.extend([
        "",
        "---",
        "",
        "### 💡 核心推演要点",
        ai_commentary,
        "",
        "---",
        "",
        "<sub>*免责声明：彩票为独立随机事件，量化模型旨在科学缩水与优化期望收益，请理性看待。*</sub>"
    ])
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("[Done] 方案二卡片式 README.md 已生成完毕。")

if __name__ == "__main__":
    main()
