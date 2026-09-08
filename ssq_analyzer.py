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
                            """, (issue, date, reds[0], reds, reds, reds[3], reds[4], reds[5], 
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
        
        # 1. 过滤偏离正态分布的极端和值（双色球理论期望均值 102，核心概率区间 80-130）
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
        # 大众偏好 1-31 的生日号码，若开出容易造成数百人均分奖池。
        # 适度偏好包含 32、33 号码的高熵组合，提升独占大奖期望。
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
        # 使用最新的 google-genai 客户端接口
        client = genai.Client(api_key=api_key)
        
        # 提取近 5 期开奖简要
        recent_df = df.tail(5)[["issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"]]
        recent_records = recent_df.to_dict(orient="records")
        
        summary_prompt = f"""
你是一位拥有深厚数理统计与博弈论背景的彩票量化精算分析师。
请根据以下双色球真实历史开奖数据和量化模型筛选出的候选组合，撰写一份客观、专业的推演研判报告（250-400字）。

【最新5期历史开奖】：
{recent_records}

【量化缩水与博弈去热门化筛选出的候选组合】：
{[{"reds": c[0], "blue": c, "features": c} for c in candidates]}

【要求】：
1. 明确指出独立随机摇奖的无记忆性与客观概率特性，杜绝虚假预测。
2. 从和值均值回归、奇偶平衡、三区分布等数理角度简析。
3. 从博弈论视角阐释：避开1-31大众生日号高密集组合对于提高独占头奖期望收益（EV）的数学意义。
4. 语言客观严谨，排版清晰。
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"> ⚠️ Gemini AI 研判生成提示: {str(e)}"

# ----------------------------------------------------------------------
# 4. 主流程：计算、汇总并生成 README.md
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
    print(f"最新一期: {latest['issue']} ({latest['date']}) | 红球: {latest_reds} | 蓝球: {latest['blue']}")
    
    # 筛选候选注码
    candidates = QuantitativeEngine.generate_candidate_pool(n_picks=5)
    
    # 调用 Gemini AI 分析
    ai_commentary = generate_gemini_analysis(df, candidates)
    
    # 组装 Markdown 看板
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# 🔴🔵 双色球专业量化推演与博弈分析看板",
        "",
        f"> 🕒 **最后更新时间**：`{current_time_str}`  ",
        f"> 📊 **历史样本库**：已收录 `{len(df)}` 期历史官方开奖数据（SQLite + Pandas 高速计算）",
        "",
        "---",
        "",
        "### 📌 官方上一期开奖结果",
        f"- **开奖期号**：`{latest['issue']}` 期（{latest['date']}）",
        f"- **开奖红球**：{' '.join(f'`{x:02d}`' for x in latest_reds)}",
        f"- **开奖蓝球**：`{int(latest['blue']):02d}`",
        f"- **奖池累积**：约 `{latest['pool']}` 元",
        "",
        "---",
        "",
        "### 🎯 下期量化缩水与 EV 优化推荐组合",
        "| 编号 | 推荐红球组合 (6码) | 蓝球 | 和值 | 跨度 | 奇偶比 | AC值 | 三区比 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for i, (reds, blue, feats) in enumerate(candidates, 1):
        red_str = " ".join(f"`{x:02d}`" for x in reds)
        lines.append(
            f"| {i} | {red_str} | `{blue:02d}` | {feats['sum']} | {feats['span']} | {feats['odd_even']} | {feats['ac_value']} | {feats['zone_ratio']} |"
        )
        
    lines.extend([
        "",
        "> **筛选准则**：和值正态分布区间（80-130）、跨度适中（18-30）、AC复杂度约束（≥6）、博弈去热门化（规避1-31全生日号密集区，降低头奖平分风险）。",
        "",
        "---",
        "",
        "### 🧠 Gemini AI 专家量化研判",
        "",
        ai_commentary,
        "",
        "---",
        "",
        "<sub>*免责声明：本系统基于数理统计与博弈论模型生成，旨在提供科学量化分析视角，随机摇奖请理性对待。*</sub>"
    ])
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("[Done] README.md 生成完毕，工作流就绪。")

if __name__ == "__main__":
    main()
