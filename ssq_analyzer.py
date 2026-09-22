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
                            """, (issue, date, reds[0], reds, reds, reds, reds, reds[5], 
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
# 2. 量化特征、定胆锁轴与 3+2 蓝球对冲引擎
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
        
        consecutive_pairs = int(np.sum(np.diff(red_arr) == 1))
        ac_val = cls.calculate_ac_value(sorted_red)
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

    @staticmethod
    def is_arithmetic_progression(reds: List[int]) -> bool:
        """检测并剔除等差数列等大众规则图形 (EV优化)"""
        diffs = np.diff(sorted(reds))
        return len(set(diffs)) <= 2 and diffs[0] in

    @classmethod
    def filter_red_combination(cls, reds: List[int]) -> bool:
        """科学形态与博弈去热门化过滤"""
        feats = cls.extract_features(reds, 1)
        
        # 1. 和值区间约束 (85 - 130)
        if not (85 <= feats["sum"] <= 130):
            return False
            
        # 2. 跨度极差过滤 (20 - 32)
        if not (20 <= feats["span"] <= 32):
            return False
            
        # 3. 奇偶形态（排除全奇全偶及 1:5 极端失衡）
        if feats["odd_even"] in ["0:6", "6:0", "1:5"]:
            return False
            
        # 4. AC复杂度约束 (>= 6)
        if feats["ac_value"] < 6:
            return False
            
        # 5. 严格限制连号（最多允许 1 组常规双连号，杜绝长连号）
        if feats["consecutive"] > 1:
            return False
            
        # 6. 剔除等差数列
        if cls.is_arithmetic_progression(reds):
            return False
            
        # 7. 抑制大众生日聚集号
        if feats["birthday_count"] == 6 and random.random() < 0.7:
            return False
            
        return True

    @classmethod
    def analyze_blue_hedging(cls, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        蓝球策略算法：
        锁定黄金中枢带 (06-12)，输出 3 注主攻反弹码与 2 注动态对冲码
        """
        recent_blues = df["blue"].tail(10).tolist() if not df.empty else []
        last_blue = recent_blues[-1] if recent_blues else 6
        
        # 主攻号池 (3注): 优先考虑黄金中枢带的质奇数与未过热路数
        primary_pool = if x != last_blue]
        # 对冲号池 (2注): 包含防守偶数及同路邻号，防止单边通杀
        hedge_pool = if x != last_blue]
        
        main_picks = random.sample(primary_pool, 3)
        hedge_picks = random.sample(hedge_pool, 2)
        return main_picks, hedge_picks

    @classmethod
    def generate_enhanced_portfolio(cls, df: pd.DataFrame) -> Tuple[int, List[Dict]]:
        """
        综合生成模块：
        1. 定胆锁轴 (Anchor Number)
        2. 重号立体梯队 (2注零重号 + 2注单重号 + 1注双重号)
        3. 蓝球 3+2 动态对冲
        """
        latest = df.iloc[-1]
        last_reds = set([int(latest[f"r{i}"]) for i in range(1, 7)])
        last_reds_list = list(last_reds)
        
        # 1. 动态确定核心红球锚点 (高位边码 32/33 在博弈防平分中具有最高权重)
        anchor_red = 32 if 33 in last_reds else 33
        
        # 2. 蓝球 3+2 对冲规划
        main_blues, hedge_blues = cls.analyze_blue_hedging(df)
        blue_plan = [
            (main_blues[0], "主攻反弹"),
            (hedge_blues[0], "动态对冲"),
            (main_blues, "主攻反弹"),
            (hedge_blues, "动态对冲"),
            (main_blues, "主攻反弹")
        ]
        
        # 3. 重号梯队分布（兼顾大换血与重号聚类）
        repeat_targets =
        all_pool = set(range(1, 34))
        fresh_pool = list(all_pool - last_reds)
        
        results = []
        for i in range(5):
            r_count = repeat_targets[i]
            blue_val, strategy_tag = blue_plan[i]
            
            found = False
            for _ in range(3000):
                # 必定包含公共轴心胆码
                chosen_reds = {anchor_red}
                
                # 配置重号配额
                if r_count > 0:
                    chosen_repeats = set(random.sample(last_reds_list, min(r_count, len(last_reds_list))))
                    chosen_reds.update(chosen_repeats)
                    
                # 补充非重号新码
                needed = 6 - len(chosen_reds)
                avail_fresh = [x for x in fresh_pool if x not in chosen_reds]
                if len(avail_fresh) < needed:
                    continue
                chosen_reds.update(random.sample(avail_fresh, needed))
                
                sorted_reds = sorted(list(chosen_reds))
                if len(sorted_reds) == 6 and cls.filter_red_combination(sorted_reds):
                    feats = cls.extract_features(sorted_reds, blue_val)
                    results.append({
                        "id": i + 1,
                        "reds": sorted_reds,
                        "blue": blue_val,
                        "strategy": strategy_tag,
                        "repeats": r_count,
                        "feats": feats
                    })
                    found = True
                    break
                    
            if not found:
                sample_reds = sorted(random.sample(range(1, 34), 6))
                results.append({
                    "id": i + 1,
                    "reds": sample_reds,
                    "blue": blue_val,
                    "strategy": strategy_tag,
                    "repeats": r_count,
                    "feats": cls.extract_features(sample_reds, blue_val)
                })
                
        return anchor_red, results

# ----------------------------------------------------------------------
# 3. 基于 google-genai 的 Gemini AI 研判模块
# ----------------------------------------------------------------------
def generate_gemini_analysis(df: pd.DataFrame, candidates: List[Dict], anchor: int) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "> ⚠️ 未检测到 `GEMINI_API_KEY` 环境变量，跳过 AI 研判生成。"

    try:
        client = genai.Client(api_key=api_key)
        recent_df = df.tail(5)[["issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"]]
        recent_records = recent_df.to_dict(orient="records")
        
        summary_prompt = f"""
你是一位专业的彩票量化精算分析师。请结合双色球历史客观数据与最新经过定胆锁轴与对冲筛选的候选组合，提供简要的数理推演点评（150-250字）。

【最新5期开奖数据】：
{recent_records}

【本次量化优化策略】：
1. 核心红球防守锚胆：{anchor:02d}
2. 重号梯队：2注零重号 + 2注单重号 + 1注双重号
3. 蓝球采用3注主攻反弹 + 2注动态对冲布局

【候选组合】：
{[{'reds': c['reds'], 'blue': c['blue'], 'strategy': c['strategy']} for c in candidates]}

【要求】：
1. 语言简练客观，严禁绝对化预测。
2. 简析上期走势形态对本期均值回归的指引。
3. 说明重号立体分层与蓝球 3+2 对冲策略在控制偏态风险、提升期望收益（EV）上的意义。
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"> ⚠️ Gemini AI 研判生成提示: {str(e)}"

# ----------------------------------------------------------------------
# 4. 主流程：全屏自适应卡片式看板输出 (方案二标准)
# ----------------------------------------------------------------------
def main():
    print("=== 开始运行双色球高级量化分析工作流 ===")
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
    
    # 综合推算：定胆锁轴 + 重号立体梯队 + 蓝球 3+2 对冲
    anchor_red, candidates = QuantitativeEngine.generate_enhanced_portfolio(df)
    
    # 调用 Gemini AI 生成专业研判
    ai_commentary = generate_gemini_analysis(df, candidates, anchor_red)
    
    # 组装极简卡片式看板 (方案二：绝对不超宽，零横向拉动)
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    next_issue = int(latest['issue']) + 1 if str(latest['issue']).isdigit() else "下期"
    
    lines = [
        "# 🔴🔵 双色球量化分析看板（高阶对冲与锁轴版）",
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
        f"### 🎯 本期推荐组合（核心红胆：`{anchor_red:02d}` ｜ 蓝球 3+2 对冲）",
        ""
    ]
    
    # 方案二：生成纯自适应卡片式列表
    for c in candidates:
        red_str = " ".join(f"`{x:02d}`" for x in c["reds"])
        strategy_badge = "🎯 主攻" if c["strategy"] == "主攻反弹" else "🛡️ 对冲"
        lines.append(
            f"* 🔴 **第 {c['id']:02d} 注**：{red_str} ＋ 🔵 `{c['blue']:02d}`  "
            f"└─ *[{strategy_badge}] {c['strategy']} ｜ 重号配额: {c['repeats']} 码*"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "### 💡 核心推演与优化逻辑",
        f"1. **定胆锁轴（聚拢红球）**：以高位边码 `{anchor_red:02d}` 作为全组核心基石，打破号码分散碎片化缺陷，增强多码同框概率。",
        "2. **重号立体防御**：按 2 注零重号（防大换血）+ 2 注单重号 + 1 注双重号梯度布局，化解两极化盘面风险。",
        "3. **蓝球 3+2 动态对冲**：3 注主攻反弹奇数 + 2 注强制对冲偶数，彻底消除单边下注导致的通杀风险。",
        "",
        "### 🧠 Gemini AI 专家研判",
        ai_commentary,
        "",
        "---",
        "",
        "<sub>*免责声明：彩票为独立随机事件，量化模型旨在科学缩水与优化期望收益，请理性看待。*</sub>"
    ])
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("[Done] 高阶 README.md 已生成完毕，工作流就绪。")

if __name__ == "__main__":
    main()
