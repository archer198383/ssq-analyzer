"""诚实版"自学"：不学预测号码（数学上不可能），只学"不骗自己"。

包含两个部件：
1. StrategyGate（策略门禁）：任何新策略/新参数在采用前，必须先通过滚动样本外
   回测——在测试期之前的数据上生成推荐，与测试期真实开奖核对；平均红球命中与
   ROI 必须同时不低于纯随机基线，否则自动驳回并出具报告。
2. SelfScoreboard（自我记分板）：长期跟踪本软件已验证推荐的真实 ROI，
   与同注数随机基线并排对比，只记录事实，不作预测。
"""
import random
import sqlite3
from typing import Callable, Dict, List, Tuple

import pandas as pd

from backtest_ssq import calculate_prize


# ----------------------------------------------------------------------
# 1. 策略门禁
# ----------------------------------------------------------------------
class StrategyGate:
    """新策略上岗前的强制回测门禁。

    strategy_fn(history_df, rng) -> List[(reds: List[int], blue: int)]：
    只能使用测试期之前的历史数据，rng 为按 draw 配对的随机源（与随机基线同种子，保证公平对照）。
    """

    def __init__(self, n_test: int = 60, tickets_per_draw: int = 5, seed: int = 20261008):
        self.n_test = n_test
        self.tickets_per_draw = tickets_per_draw
        self.seed = seed

    @staticmethod
    def _draw_reds(row) -> List[int]:
        return [int(row[f"r{i}"]) for i in range(1, 7)]

    def _score_tickets(self, tickets: List[Tuple[List[int], int]],
                       act_reds: set, act_blue: int) -> Dict:
        n = len(tickets)
        red_hits_total = 0
        prize_total = 0
        for reds, blue in tickets:
            rh = len(set(reds) & act_reds)
            bh = int(blue) == act_blue
            _, money = calculate_prize(rh, bh)
            red_hits_total += rh
            prize_total += money
        cost = n * 2
        return {
            "n_tickets": n,
            "mean_red_hits": round(red_hits_total / n, 4) if n else 0.0,
            "total_prize": prize_total,
            "total_cost": cost,
            "roi": round(prize_total / cost, 4) if cost else 0.0,
        }

    def _random_tickets(self, rng: random.Random) -> List[Tuple[List[int], int]]:
        return [(sorted(rng.sample(range(1, 34), 6)), rng.randint(1, 16))
                for _ in range(self.tickets_per_draw)]

    def run_backtest(self, strategy_fn: Callable, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """滚动样本外回测：返回 (挑战者指标, 随机基线指标)。"""
        if len(df) < self.n_test + 1:
            raise ValueError(f"历史数据不足：需要至少 {self.n_test + 1} 期，实际 {len(df)} 期")
        chal_agg = {"red_hits": 0, "prize": 0, "n": 0}
        rand_agg = {"red_hits": 0, "prize": 0, "n": 0}
        start = len(df) - self.n_test
        for t in range(start, len(df)):
            history = df.iloc[:t]
            row = df.iloc[t]
            act_reds = set(self._draw_reds(row))
            act_blue = int(row["blue"])
            rng = random.Random(self.seed + t)
            chal_tickets = strategy_fn(history, rng)
            rand_tickets = self._random_tickets(random.Random(self.seed + t))
            for agg, tickets in ((chal_agg, chal_tickets), (rand_agg, rand_tickets)):
                for reds, blue in tickets:
                    rh = len(set(reds) & act_reds)
                    bh = int(blue) == act_blue
                    _, money = calculate_prize(rh, bh)
                    agg["red_hits"] += rh
                    agg["prize"] += money
                    agg["n"] += 1
        def finalize(agg):
            n = agg["n"]
            cost = n * 2
            return {
                "n_tickets": n,
                "mean_red_hits": round(agg["red_hits"] / n, 4),
                "total_prize": agg["prize"],
                "total_cost": cost,
                "roi": round(agg["prize"] / cost, 4),
            }
        return finalize(chal_agg), finalize(rand_agg)

    @staticmethod
    def decide(chal: Dict, rand: Dict) -> bool:
        """采用标准：挑战者平均红球命中与 ROI 必须同时不低于随机基线。"""
        return chal["mean_red_hits"] > rand["mean_red_hits"] and chal["roi"] >= rand["roi"]

    def compare(self, name: str, strategy_fn: Callable, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """运行门禁并出具中文报告。返回 (是否通过, 报告行)。"""
        chal, rand = self.run_backtest(strategy_fn, df)
        passed = self.decide(chal, rand)
        verdict = "✅ 通过门禁" if passed else "❌ 未通过门禁（驳回）"
        lines = [
            f"### 🚦 策略门禁报告：{name}",
            f"- 样本外期数：`{self.n_test}` 期，每期 `{self.tickets_per_draw}` 注",
            f"- 挑战者：平均红球命中 `{chal['mean_red_hits']}`/注 ｜ ROI `{chal['roi']}`"
            f"（{chal['n_tickets']} 注，中奖 {chal['total_prize']} 元）",
            f"- 随机基线：平均红球命中 `{rand['mean_red_hits']}`/注 ｜ ROI `{rand['roi']}`"
            f"（{rand['n_tickets']} 注，中奖 {rand['total_prize']} 元）",
            f"- 结论：**{verdict}**",
            "- 门禁标准：挑战者平均红球命中与 ROI 须同时不低于随机基线；"
            "通过门禁只是最低要求，不构成策略有效的证明（更大样本、独立时间段仍需复核）。",
            "",
        ]
        return passed, lines


def engine_strategy(history_df: pd.DataFrame, rng: random.Random):
    """把当前线上引擎包装成门禁可测的策略函数（严格只用测试期之前的数据）。"""
    import ssq_analyzer
    import random as pyrandom
    state = pyrandom.getstate()
    pyrandom.seed(rng.randint(0, 2 ** 31 - 1))
    try:
        _, cands = ssq_analyzer.QuantitativeEngine.generate_enhanced_portfolio(history_df)
    finally:
        pyrandom.setstate(state)
    return [(c["reds"], c["blue"]) for c in cands]


# ----------------------------------------------------------------------
# 2. 自我记分板
# ----------------------------------------------------------------------
class SelfScoreboard:
    """跟踪本软件已验证推荐的真实战绩，与同注数随机基线并排对比。"""

    @staticmethod
    def _find_draw(df: pd.DataFrame, target_issue: str, target_date: str):
        draw = df[df["issue"].astype(str) == str(target_issue)]
        if draw.empty and target_date:
            draw = df[df["date"].astype(str).str[:10] == str(target_date)[:10]]
        return draw.iloc[0] if not draw.empty else None

    @classmethod
    def compute(cls, conn: sqlite3.Connection, df: pd.DataFrame,
                seed: int = 20261008) -> List[Dict]:
        rows = conn.execute(
            "SELECT target_issue, target_date, note_id, reds, blue FROM recommendations"
            " WHERE verified_at IS NOT NULL AND verified_at != ''"
            " ORDER BY target_issue ASC").fetchall()
        by_issue: Dict[str, Dict] = {}
        for tgt, tgt_date, _nid, reds_str, blue in rows:
            reds = sorted(int(x) for x in str(reds_str).split(",") if x.strip().isdigit())
            if len(reds) != 6:
                continue
            by_issue.setdefault(str(tgt), {"date": tgt_date, "tickets": []})
            by_issue[str(tgt)]["tickets"].append((reds, int(blue)))

        results = []
        for tgt in sorted(by_issue.keys()):
            info = by_issue[tgt]
            draw = cls._find_draw(df, tgt, info["date"])
            if draw is None:
                continue
            act_reds = {int(draw[f"r{i}"]) for i in range(1, 7)}
            act_blue = int(draw["blue"])
            tickets = info["tickets"]
            n = len(tickets)
            prize = hits = 0
            for reds, blue in tickets:
                rh = len(set(reds) & act_reds)
                bh = int(blue) == act_blue
                _, money = calculate_prize(rh, bh)
                hits += rh
                prize += money
            # 同注数随机基线（种子由期号派生，固定可复现）
            issue_digits = "".join(ch for ch in tgt if ch.isdigit())
            rng = random.Random(seed + (int(issue_digits[-6:]) if issue_digits else 0))
            r_prize = r_hits = 0
            for _ in range(n):
                r_reds = rng.sample(range(1, 34), 6)
                r_blue = rng.randint(1, 16)
                rh = len(set(r_reds) & act_reds)
                bh = r_blue == act_blue
                _, money = calculate_prize(rh, bh)
                r_hits += rh
                r_prize += money
            cost = n * 2
            results.append({
                "issue": tgt,
                "date": str(draw["date"])[:10],
                "n": n,
                "cost": cost,
                "prize": prize,
                "roi": round(prize / cost, 4),
                "mean_hits": round(hits / n, 4),
                "rand_prize": r_prize,
                "rand_roi": round(r_prize / cost, 4),
                "rand_hits": round(r_hits / n, 4),
            })
        return results

    @classmethod
    def render_markdown(cls, rows: List[Dict]) -> List[str]:
        if not rows:
            return ["> 暂无已验证推荐，记分板待开奖数据到位后生成。", ""]
        lines = [
            "> 🧾 本记分板只记录已验证推荐的**事实**；随机基线为同注数模拟（种子固定，可复现）；"
            "样本量较小时结论仅供参考，**不构成任何预测依据**。",
            "",
            "| 期号 | 投入 | 中奖 | ROI | 红球命中/注 | 随机基线命中/注 | 随机基线ROI |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        tot_cost = tot_prize = tot_rand_prize = 0
        tot_hits = tot_rand_hits = tot_n = 0
        for r in rows:
            lines.append(
                f"| `{r['issue']}` | {r['cost']}元 | {r['prize']}元 | {r['roi']:.1%} "
                f"| {r['mean_hits']} | {r['rand_hits']} | {r['rand_roi']:.1%} |")
            tot_cost += r["cost"]
            tot_prize += r["prize"]
            tot_rand_prize += r["rand_prize"]
            tot_hits += r["mean_hits"] * r["n"]
            tot_rand_hits += r["rand_hits"] * r["n"]
            tot_n += r["n"]
        lines.extend([
            "",
            f"- **累计**（{len(rows)} 期，{tot_n} 注）：实际投入 `{tot_cost}` 元，"
            f"中奖 `{tot_prize}` 元，ROI **{tot_prize / tot_cost:.1%}**；"
            f"随机基线中奖 `{tot_rand_prize}` 元，ROI **{tot_rand_prize / tot_cost:.1%}**；"
            f"平均红球命中 `{tot_hits / tot_n:.4f}`/注 vs 随机基线 `{tot_rand_hits / tot_n:.4f}`/注。",
            "",
        ])
        return lines


def expert_mimic_strategy(history_df: pd.DataFrame, rng: random.Random,
                          tickets_per_draw: int = 5):
    """实验性挑战者：模仿彩票专家的选号思路。

    - 胆码：近30期热号 Top3，每注必含其中 2 个
    - 拖码：按近30期出现频率加权抽样（追热）
    - 蓝球：80% 追近10期最热蓝球，20% 随机（模仿专家"主追热、防冷"）

    仅用于策略门禁测试，验证"专家思路"有无样本外优势；
    无论结果如何，都不直接用于线上推荐（线上推荐逻辑变更必须经用户批准）。
    """
    import ssq_analyzer  # 延迟导入，避免循环依赖
    E = ssq_analyzer.ExpertStyleAnalysis
    a = E.analyze(history_df, window=10, hot_window=30)
    danma = a.get("danma") or [1, 2, 3]

    hot_pool = history_df.tail(30)
    freq: Dict[int, int] = {}
    for _, row in hot_pool.iterrows():
        for x in E._reds(row):
            freq[x] = freq.get(x, 0) + 1
    weights = [freq.get(x, 0) + 1 for x in range(1, 34)]  # +1 平滑，避免零权重

    blues = [int(b) for b in history_df.tail(10)["blue"].tolist()]
    bfreq: Dict[int, int] = {}
    for b in blues:
        bfreq[b] = bfreq.get(b, 0) + 1
    hot_blue = max(sorted(bfreq.items()), key=lambda kv: kv[1])[0]

    tickets: List[Tuple[List[int], int]] = []
    seen = set()
    attempts = 0
    while len(tickets) < tickets_per_draw and attempts < 200:
        attempts += 1
        picks = set(rng.sample(danma, min(2, len(danma))))
        while len(picks) < 6:
            picks.add(rng.choices(range(1, 34), weights=weights, k=1)[0])
        reds = sorted(picks)
        if tuple(reds) in seen:
            continue
        seen.add(tuple(reds))
        blue = hot_blue if rng.random() < 0.8 else rng.randint(1, 16)
        tickets.append((reds, blue))
    while len(tickets) < tickets_per_draw:  # 兜底：纯随机补齐
        tickets.append((sorted(rng.sample(range(1, 34), 6)), rng.randint(1, 16)))
    return tickets


if __name__ == "__main__":
    # 门禁首跑：当前线上引擎 vs 纯随机（最近 60 期，严格样本外）
    import os
    import ssq_analyzer
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssq_history.db")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM lottery_records ORDER BY issue ASC", conn)
    conn.close()
    print(f"历史数据 {len(df)} 期，运行策略门禁（60 期样本外）……")
    gate = StrategyGate(n_test=60, tickets_per_draw=5)
    for name, fn in (("当前线上引擎", engine_strategy),
                     ("专家思路模仿者", expert_mimic_strategy)):
        passed, report = gate.compare(name, fn, df)
        print("\n".join(report))
