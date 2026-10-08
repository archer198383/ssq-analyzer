"""自学习模块单元测试：策略门禁 + 自我记分板。"""
import random
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, "/home/hatch/workspace/ssq-analyzer")
from self_learning import StrategyGate, SelfScoreboard, expert_mimic_strategy


def make_df(n=25, seed=11):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        reds = sorted(rng.sample(range(1, 34), 6))
        rows.append({"issue": f"2026{i:03d}", "date": "2026-01-01",
                     "r1": reds[0], "r2": reds[1], "r3": reds[2],
                     "r4": reds[3], "r5": reds[4], "r6": reds[5],
                     "blue": rng.randint(1, 16)})
    return pd.DataFrame(rows)


def fixed_strategy(history_df, rng):
    # 固定票：每期都买 1-6 + 蓝1（仅用于测试门禁流程，不代表任何真实策略）
    return [([1, 2, 3, 4, 5, 6], 1) for _ in range(5)]


def test_gate_decide_logic():
    good = {"mean_red_hits": 1.2, "roi": 0.25}
    bad = {"mean_red_hits": 1.0, "roi": 0.15}
    assert StrategyGate.decide(good, bad) is True
    assert StrategyGate.decide(bad, good) is False
    # 命中更高但 ROI 更低 -> 驳回（必须同时不低于）
    assert StrategyGate.decide({"mean_red_hits": 1.3, "roi": 0.10}, bad) is False
    # 命中持平 -> 驳回（必须"高于"随机）
    assert StrategyGate.decide({"mean_red_hits": 1.0, "roi": 0.20}, bad) is False
    print("PASS gate_decide_logic")


def test_gate_runs_end_to_end():
    df = make_df(25)
    gate = StrategyGate(n_test=20, tickets_per_draw=5, seed=99)
    chal, rand = gate.run_backtest(fixed_strategy, df)
    for m in (chal, rand):
        assert m["n_tickets"] == 100, m
        assert m["mean_red_hits"] >= 0
        assert m["total_cost"] == 200
    passed, report = gate.compare("固定票策略", fixed_strategy, df)
    text = "\n".join(report)
    assert "策略门禁报告" in text and ("通过门禁" in text or "驳回" in text)
    assert isinstance(passed, bool)
    print(f"PASS gate_runs_end_to_end (verdict={'通过' if passed else '驳回'})")


def test_gate_rejects_cheating_check():
    # 策略函数只能看到历史数据：若某"策略"试图偷看未来，门禁签名层面无法获得未来数据
    df = make_df(25)
    seen = {}

    def spy_strategy(history_df, rng):
        seen["max_issue"] = history_df["issue"].max()
        return fixed_strategy(history_df, rng)

    gate = StrategyGate(n_test=20, seed=99)
    gate.run_backtest(spy_strategy, df)
    # 最后一次调用时 history 只到倒数第2期，绝不能看到最后一期
    assert seen["max_issue"] == df["issue"].iloc[-2], seen
    print("PASS gate_no_peeking")


def _scoreboard_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE lottery_records (issue TEXT PRIMARY KEY, date TEXT,
        r1 INTEGER, r2 INTEGER, r3 INTEGER, r4 INTEGER, r5 INTEGER, r6 INTEGER, blue INTEGER)""")
    conn.execute("""CREATE TABLE recommendations (target_issue TEXT, note_id INTEGER, reds TEXT,
        blue INTEGER, strategy TEXT, repeats INTEGER, target_date TEXT, created_at TEXT, verified_at TEXT)""")
    conn.execute("INSERT INTO lottery_records VALUES ('2026001','2026-01-01',1,2,3,4,5,6,7)")
    conn.execute("INSERT INTO lottery_records VALUES ('2026002','2026-01-04',10,11,12,13,14,15,16)")
    # 第1期：票1=6红+蓝错(二等奖20万)；票2=3红+蓝中(五等奖10元)
    conn.execute("INSERT INTO recommendations VALUES ('2026001',1,'1,2,3,4,5,6',8,'t',0,'2026-01-01','x','2026-01-02')")
    conn.execute("INSERT INTO recommendations VALUES ('2026001',2,'1,2,3,10,11,12',7,'t',0,'2026-01-01','x','2026-01-02')")
    # 第2期：6+1 全中(一等奖500万)
    conn.execute("INSERT INTO recommendations VALUES ('2026002',1,'10,11,12,13,14,15',16,'t',0,'2026-01-04','x','2026-01-05')")
    # 未验证的不应计入
    conn.execute("INSERT INTO recommendations VALUES ('2026003',1,'1,2,3,4,5,6',7,'t',0,'2026-01-06','x','')")
    conn.commit()
    df = pd.read_sql_query("SELECT * FROM lottery_records ORDER BY issue ASC", conn)
    return conn, df


def test_scoreboard_known_outcome():
    conn, df = _scoreboard_conn()
    rows = SelfScoreboard.compute(conn, df)
    conn.close()
    assert len(rows) == 2, rows  # 未验证的 2026003 不计入
    r1 = rows[0]
    assert r1["issue"] == "2026001" and r1["n"] == 2
    assert r1["cost"] == 4 and r1["prize"] == 200010, r1
    assert r1["roi"] == 200010 / 4
    assert r1["mean_hits"] == 4.5, r1
    r2 = rows[1]
    assert r2["prize"] == 5000000 and r2["roi"] == 2500000.0
    for r in rows:
        assert r["rand_roi"] >= 0 and 0 <= r["rand_hits"] <= 6
    print("PASS scoreboard_known_outcome")


def test_scoreboard_render():
    conn, df = _scoreboard_conn()
    rows = SelfScoreboard.compute(conn, df)
    conn.close()
    text = "\n".join(SelfScoreboard.render_markdown(rows))
    assert "自我记分板" not in text  # 标题由主流程加，此处只断言内容
    assert "2026001" in text and "累计" in text and "随机基线" in text
    assert "不构成任何预测依据" in text, "必须带诚实声明"
    print("PASS scoreboard_render")


def test_scoreboard_empty():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE recommendations (target_issue TEXT, note_id INTEGER, reds TEXT,"
                 " blue INTEGER, target_date TEXT, verified_at TEXT)")
    conn.commit()
    rows = SelfScoreboard.compute(conn, pd.DataFrame())
    conn.close()
    assert rows == []
    assert any("暂无已验证推荐" in l for l in SelfScoreboard.render_markdown(rows))
    print("PASS scoreboard_empty")


def test_expert_mimic_valid_tickets():
    df = make_df(40)
    tickets = expert_mimic_strategy(df, random.Random(5))
    assert len(tickets) == 5
    for reds, blue in tickets:
        assert len(reds) == 6 and len(set(reds)) == 6
        assert all(1 <= x <= 33 for x in reds)
        assert 1 <= blue <= 16
    print("PASS expert_mimic_valid_tickets")


if __name__ == "__main__":
    test_gate_decide_logic()
    test_gate_runs_end_to_end()
    test_gate_rejects_cheating_check()
    test_scoreboard_known_outcome()
    test_scoreboard_render()
    test_scoreboard_empty()
    test_expert_mimic_valid_tickets()
    print("ALL 7 SELF-LEARNING TESTS PASSED")
