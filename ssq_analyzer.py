import sqlite3
import os
import requests
import json
import random
from typing import List, Dict, Tuple

# -------------------------------------------------------------
# 1. 数据库适配层 (Database Layer)
# -------------------------------------------------------------
class SSQDatabase:
    def __init__(self, db_path: str = "ssq_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化标准双色球历史开奖表结构"""
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

    def load_all_history(self) -> List[Dict]:
        """从数据库读取全量历史开奖记录（按期号升序）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT issue, date, r1, r2, r3, r4, r5, r6, blue 
                FROM lottery_records 
                ORDER BY issue ASC
            """)
            rows = cursor.fetchall()
            
        records = []
        for r in rows:
            records.append({
                "issue": r[0],
                "date": r[1],
                "reds": [r[2], r[3], r[4], r[5], r[6], r[7]],
                "blue": r[8]
            })
        return records

    def sync_from_cwl(self, page_size: int = 30):
        """增量抓取福彩官方数据并同步入库"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url = f"https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq&issueCount={page_size}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
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
                            """, (issue, date, reds[0], reds[1], reds[2], reds[3], reds[4], reds[5], 
                                  blue, item.get("sales", ""), item.get("poolmoney", "")))
                    conn.commit()
        except Exception as e:
            print(f"数据抓取同步提示: {e}")

# -------------------------------------------------------------
# 2. 专业量化与博弈论分析核心 (Analytics & Game Theory Core)
# -------------------------------------------------------------
class SSQProfessionalAnalyzer:
    @staticmethod
    def calculate_ac_value(reds: List[int]) -> int:
        """计算红球 AC 值（复杂度指标）"""
        diffs = {abs(reds[j] - reds[i]) for i in range(len(reds)) for j in range(i + 1, len(reds))}
        return len(diffs) - (len(reds) - 1)

    @staticmethod
    def get_combination_features(reds: List[int], blue: int) -> Dict:
        """提取一组号码的核心数学属性"""
        sorted_red = sorted(reds)
        red_sum = sum(sorted_red)
        span = sorted_red[-1] - sorted_red[0]
        odd_count = sum(1 for x in sorted_red if x % 2 != 0)
        
        # 三区比 (01-11, 12-22, 23-33)
        z1 = sum(1 for x in sorted_red if 1 <= x <= 11)
        z2 = sum(1 for x in sorted_red if 12 <= x <= 22)
        z3 = sum(1 for x in sorted_red if 23 <= x <= 33)
        
        consecutive_pairs = sum(1 for i in range(len(sorted_red) - 1) if sorted_red[i+1] - sorted_red[i] == 1)
        ac_value = SSQProfessionalAnalyzer.calculate_ac_value(sorted_red)
        
        # 博弈论指标：大众生日号偏好（1-31过度聚集度）
        birthday_count = sum(1 for x in sorted_red if x <= 31)

        return {
            "sum": red_sum,
            "span": span,
            "odd_even": f"{odd_count}:{6 - odd_count}",
            "zone_ratio": f"{z1}:{z2}:{z3}",
            "consecutive": consecutive_pairs,
            "ac_value": ac_value,
            "birthday_count": birthday_count,
            "blue": blue
        }

    @classmethod
    def filter_and_optimize(cls, reds: List[int], blue: int) -> bool:
        """科学缩水与期望值优化过滤"""
        feats = cls.get_combination_features(reds, blue)
        
        # 1. 过滤偏离正态分布均值的极端和值（核心概率区间: 80 - 130）
        if not (80 <= feats["sum"] <= 130):
            return False
            
        # 2. 跨度过滤
        if not (18 <= feats["span"] <= 30):
            return False
            
        # 3. 奇偶形态（排除全奇全偶）
        if feats["odd_even"] in ["0:6", "6:0"]:
            return False
            
        # 4. AC值过滤过简单组合
        if feats["ac_value"] < 6:
            return False
            
        # 5. 过滤长连号
        if feats["consecutive"] > 2:
            return False
            
        # 6. 博弈论去热门化：降低全由 1-31 生日号构成的聚集组合，提高大奖独占期望
        if feats["birthday_count"] == 6 and random.random() < 0.65:
            return False
            
        return True

    @classmethod
    def generate_recommendations(cls, n: int = 5) -> List[Tuple[List[int], int, Dict]]:
        """生成符合专业量化与 EV 优化的推荐注码"""
        picks = []
        pool_reds = list(range(1, 34))
        while len(picks) < n:
            reds = sorted(random.sample(pool_reds, 6))
            blue = random.randint(1, 16)
            if cls.filter_and_optimize(reds, blue):
                feats = cls.get_combination_features(reds, blue)
                picks.append((reds, blue, feats))
        return picks

# -------------------------------------------------------------
# 3. 执行迁移与生成 Markdown 报告
# -------------------------------------------------------------
def run_pipeline():
    # 1. 实例化数据库并同步最新开奖
    db = SSQDatabase("ssq_history.db")
    db.sync_from_cwl(page_size=30)
    history = db.load_all_history()
    
    print(f"数据库加载成功，历史总期数: {len(history)}")
    
    # 2. 获取最新一期
    latest = history[-1] if history else None
    
    # 3. 执行推演与缩水
    picks = SSQProfessionalAnalyzer.generate_recommendations(n=5)
    
    # 4. 生成兼容现有 README 的报告格式
    report_lines = [
        "# 双色球专业量化分析与推演系统",
        "",
        f"> **数据状态**：已接入历史数据库（当前共收录 {len(history)} 期）",
    ]
    
    if latest:
        report_lines.append(f"> **最新开奖 ({latest['issue']} 期 - {latest['date']})**：红球 `{' '.join(f'{x:02d}' for x in latest['reds'])}` + 蓝球 `{latest['blue']:02d}`")
    
    report_lines.extend([
        "",
        "## 🎯 下期量化缩水与博弈优化推荐 (EV-Optimized)",
        "| 序号 | 推荐红球组合 | 蓝球 | 和值 | 跨度 | 奇偶比 | AC值 | 三区比 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])
    
    for i, (reds, blue, feats) in enumerate(picks, 1):
        red_str = " ".join(f"`{x:02d}`" for x in reds)
        report_lines.append(
            f"| {i} | {red_str} | `{blue:02d}` | {feats['sum']} | {feats['span']} | {feats['odd_even']} | {feats['ac_value']} | {feats['zone_ratio']} |"
        )
        
    report_lines.append("\n*注：模型基于和值正态分布区间、AC复杂度和博弈论去大众偏好（防分奖稀释）进行筛选。*")
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("README.md 报告已更新完成。")

if __name__ == "__main__":
    run_pipeline()
