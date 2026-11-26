"""双阈值峰判定的单元测试（合成确定性剖面，不依赖地形数据）。

验证：
  1. 高度阈值：低于阈值的局部极大被否决；
  2. 间距阈值：两个都过高阈值、但间隔小于阈值的峰，高者保留、近者被间距否决；
  3. 两条件必须同时满足：仅过高度但间距不足、或间距够但高度不足都不成峰。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_core import pick_peaks, BINS

ANG = np.arange(BINS, dtype=float)
HEIGHT_THR = 1.25
SEP_THR = 15.0


def gaussian(center, width=4.0, amp=2.0, base=1.0):
    d = np.abs((ANG - center + 90) % 180 - 90)
    return base + amp * np.exp(-0.5 * (d / width) ** 2)


def test_height_threshold():
    # 一个强峰（2.0）+ 一个弱局部极大（1.1，低于高度阈值）
    p = np.full(BINS, 1.0)
    p = np.maximum(p, gaussian(120, amp=1.2))   # 峰高 2.2
    p = np.maximum(p, gaussian(40, amp=0.1))    # 峰高 1.1
    acc, rej_h, rej_s = pick_peaks(p, HEIGHT_THR, SEP_THR)
    assert len(acc) == 1 and abs(acc[0]["angle"] - 120) <= 1
    assert any(abs(r["angle"] - 40) <= 1 for r in rej_h), "弱峰应被高度阈值否决"
    print("PASS test_height_threshold")


def test_spacing_threshold():
    # 两个都过高阈值、相隔 10°（< 间距阈值 15°）的峰
    p = np.full(BINS, 1.0)
    p = np.maximum(p, gaussian(60, amp=1.0))    # 峰高 2.0
    p = np.maximum(p, gaussian(70, amp=0.6))    # 峰高 1.6，较弱
    acc, rej_h, rej_s = pick_peaks(p, HEIGHT_THR, SEP_THR)
    assert len(acc) == 1 and abs(acc[0]["angle"] - 60) <= 1, "高者应保留"
    assert any(abs(r["angle"] - 70) <= 1 for r in rej_s), "近峰应被间距阈值否决"
    print("PASS test_spacing_threshold")


def test_both_conditions_required():
    # 三个候选：A 强且孤立（成峰）；B 强但贴近 A（间距否决）；C 孤立但弱（高度否决）
    p = np.full(BINS, 1.0)
    p = np.maximum(p, gaussian(120, amp=1.2))   # A: 2.2，成峰
    p = np.maximum(p, gaussian(128, amp=0.8))   # B: 1.8，过高但距 A 仅 8°
    p = np.maximum(p, gaussian(30, amp=0.15))   # C: 1.15，孤立但高度不足
    acc, rej_h, rej_s = pick_peaks(p, HEIGHT_THR, SEP_THR)
    acc_angles = [r["angle"] for r in acc]
    assert acc_angles == [120.0], f"仅 A 应成峰，实际 {acc_angles}"
    assert any(abs(r["angle"] - 128) <= 1 for r in rej_s)
    assert any(abs(r["angle"] - 30) <= 1 for r in rej_h)
    print("PASS test_both_conditions_required")


def test_wide_separation_keeps_both():
    # 两个都过高阈值、相隔 60°（> 间距阈值），都应保留
    p = np.full(BINS, 1.0)
    p = np.maximum(p, gaussian(40, amp=1.0))
    p = np.maximum(p, gaussian(100, amp=1.0))
    acc, rej_h, rej_s = pick_peaks(p, HEIGHT_THR, SEP_THR)
    assert len(acc) == 2, f"两远峰都应保留，实际 {acc}"
    assert not rej_s, "远峰不应被间距否决"
    print("PASS test_wide_separation_keeps_both")


if __name__ == "__main__":
    test_height_threshold()
    test_spacing_threshold()
    test_both_conditions_required()
    test_wide_separation_keeps_both()
    print("全部双阈值测试通过")
