"""开奖数据同步修复测试：重试诊断 + 手动录入。"""
import sqlite3
import sys
import tempfile
import os

sys.path.insert(0, "/home/hatch/workspace/ssq-analyzer")
from ssq_analyzer import SSQDataManager


def make_mgr():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return SSQDataManager(db_path=path), path


def test_manual_import_valid():
    mgr, path = make_mgr()
    ok = mgr.import_manual_draw("2026115", "2026-10-08", [3, 11, 18, 22, 29, 33], 9)
    assert ok is True
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT issue, date, r1, r2, r3, r4, r5, r6, blue FROM lottery_records WHERE issue='2026115'").fetchone()
    conn.close()
    assert row == ("2026115", "2026-10-08", 3, 11, 18, 22, 29, 33, 9), row
    os.unlink(path)
    print("PASS manual_import_valid")


def test_manual_import_idempotent():
    mgr, path = make_mgr()
    assert mgr.import_manual_draw("2026115", "2026-10-08", [3, 11, 18, 22, 29, 33], 9) is True
    # 重复录入不同号码：必须幂等跳过，不覆盖
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 6], 1) is True
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT r1, blue FROM lottery_records WHERE issue='2026115'").fetchone()
    conn.close()
    assert row == (3, 9), f"重复录入覆盖了原始数据: {row}"
    os.unlink(path)
    print("PASS manual_import_idempotent")


def test_manual_import_invalid():
    mgr, path = make_mgr()
    assert mgr.import_manual_draw("20261", "2026-10-08", [1, 2, 3, 4, 5, 6], 1) is False      # 期号位数错
    assert mgr.import_manual_draw("2026115", "10-08", [1, 2, 3, 4, 5, 6], 1) is False          # 日期格式错
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5], 1) is False        # 红球不足6个
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 5], 1) is False     # 红球重复
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 34], 1) is False    # 红球越界
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 6], 17) is False    # 蓝球越界
    conn = sqlite3.connect(path)
    n = conn.execute("SELECT COUNT(*) FROM lottery_records").fetchone()[0]
    conn.close()
    assert n == 0, "非法录入不应写库"
    os.unlink(path)
    print("PASS manual_import_invalid")


def test_sync_failure_diagnostics():
    # 官方接口当前被拦截：同步应返回 False，且 last_sync 有诊断信息（仅 1 次尝试，保持测试快）
    mgr, path = make_mgr()
    ok = mgr.sync_official_data(fetch_count=2, retries=1)
    info = mgr.last_sync
    assert ok is False, "当前环境官方接口应失败"
    assert info["attempts"] == 1 and info["ok"] is False
    assert info["error"], "失败时必须记录诊断信息"
    print(f"PASS sync_failure_diagnostics (error={info['error'][:60]})")
    os.unlink(path)


if __name__ == "__main__":
    test_manual_import_valid()
    test_manual_import_idempotent()
    test_manual_import_invalid()
    test_sync_failure_diagnostics()
    print("ALL 4 SYNC-FIX TESTS PASSED")
