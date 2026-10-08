"""专家式走势分析模块单元测试（ExpertStyleAnalysis）。"""
import sys
import pandas as pd

sys.path.insert(0, "/home/hatch/workspace/ssq-analyzer")
from ssq_analyzer import ExpertStyleAnalysis as E


def make_df(n=40, seed=7):
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        reds = sorted(rng.sample(range(1, 34), 6))
        rows.append({
            "issue": f"2026{i:03d}", "date": "2026-01-01",
            "r1": reds[0], "r2": reds[1], "r3": reds[2],
            "r4": reds[3], "r5": reds[4], "r6": reds[5],
            "blue": rng.randint(1, 16),
        })
    return pd.DataFrame(rows)


def test_zone_boundaries():
    assert E.zone_of(1) == 1 and E.zone_of(11) == 1
    assert E.zone_of(12) == 2 and E.zone_of(22) == 2
    assert E.zone_of(23) == 3 and E.zone_of(33) == 3
    print("PASS zone_boundaries")


def test_counts_sum():
    df = make_df(40)
    a = E.analyze(df, window=10)
    assert sum(a["zone_counts"]) == 60, a["zone_counts"]
    assert sum(a["route012"]) == 60, a["route012"]
    assert sum(a["odd_even"]) == 60
    assert sum(a["big_small"]) == 60
    assert sum(a["prime_comp"]) == 60
    assert sum(a["blue_odd_even"]) == 10
    assert sum(a["blue_prime_comp"]) == 10
    print("PASS counts_sum")


def test_hot_cold_valid():
    df = make_df(40)
    a = E.analyze(df, window=10, hot_window=30)
    assert len(a["hot6"]) == 6 and all(1 <= x <= 33 for x, _ in a["hot6"])
    assert len(a["cold6"]) == 6 and all(1 <= x <= 33 for x, _ in a["cold6"])
    omits = [o for _, o in a["cold6"]]
    assert omits == sorted(omits, reverse=True), "遗漏榜应按遗漏期数降序"
    assert all(o >= 0 for o in omits)
    print("PASS hot_cold_valid")


def test_danma_shahao_disjoint():
    df = make_df(40)
    a = E.analyze(df)
    assert len(a["danma"]) == 3
    assert len(a["shahao"]) == 3
    assert set(a["danma"]).isdisjoint(set(a["shahao"])), "胆码与杀号不应重叠"
    hot_nums = {x for x, _ in a["hot6"]}
    assert set(a["danma"]) <= hot_nums, "胆码应来自热号榜"
    print("PASS danma_shahao_disjoint")


def test_render_content_and_disclaimer():
    df = make_df(40)
    a = E.analyze(df)
    lines = E.render_markdown(a)
    text = "\n".join(lines)
    for kw in ["三区走势", "012路", "奇偶比", "大小比", "质合比", "蓝球", "热号", "遗漏", "胆码参考", "杀号参考"]:
        assert kw in text, f"缺少关键词: {kw}"
    assert "不改变任何号码的中奖概率" in text, "必须带诚实声明"
    print("PASS render_content_and_disclaimer")


def test_empty_df():
    a = E.analyze(pd.DataFrame())
    assert a == {}
    lines = E.render_markdown(a)
    assert any("暂无足够历史数据" in l for l in lines)
    print("PASS empty_df")


def test_small_df():
    df = make_df(5)  # 少于默认 window=10
    a = E.analyze(df, window=10)
    assert a["window"] == 5
    assert sum(a["zone_counts"]) == 30
    lines = E.render_markdown(a)
    assert "三区走势" in "\n".join(lines)
    print("PASS small_df")


if __name__ == "__main__":
    test_zone_boundaries()
    test_counts_sum()
    test_hot_cold_valid()
    test_danma_shahao_disjoint()
    test_render_content_and_disclaimer()
    test_empty_df()
    test_small_df()
    print("ALL 7 EXPERT-ANALYSIS TESTS PASSED")
