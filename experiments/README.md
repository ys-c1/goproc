# experiments

种子扩散实验的代码与产物说明，分析与证明见 [REPORT.md](REPORT.md)。

## Go 实验程序

- `../cmd/seedexp/` — 蒙特卡洛实验主体
  - `permutations.go`：三类置换表派生（production / weak / hash）、值噪声与指标
  - `main.go`：三档种子间隔（1、16、256）与 q 扫描，输出 `../artifacts/seedexp-report.json`
- `../cmd/terrainpairs/` — 导出代表性 PGM 高度图对（通过符号链接复用 `permutations.go`）

指标定义：

- 改动条目数：两张置换表的汉明距离（满分 256）
- 相关面积：两幅归一化高度图逐点差小于 τ=0.10 的像素占比
- Pearson：两幅高度图的皮尔逊相关系数
- 独立判据：互不相关散列种子对的同协议实测基准（相关面积约 0.217）

## Python 绘图

系统 Python 受 PEP 668 管理且未预装 matplotlib，使用虚拟环境：

```
python3 -m venv /tmp/vizvenv
/tmp/vizvenv/bin/pip install matplotlib numpy
make PYTHON=/tmp/vizvenv/bin/python figures
```

绘图脚本按 dataviz 技能流程选色，调色板（蓝/橙/青/红）已通过
`validate_palette.js` 校验；中文字体显式注册本机
`~/.local/share/fonts/NotoSansCJKsc-Regular.ttf`。

## 产物

- `../artifacts/seedexp-report.json` — 全部实验均值与配置（表格视图）
- `../artifacts/figures/correlated_area_weak.png` — 弱派生相关面积 vs 独立判据
- `../artifacts/figures/correlated_area_hash.png` — 散列扰动相关面积 vs 独立判据
- `../artifacts/figures/terrain_pairs.png` — 相邻种子高度图目视对照
- `../artifacts/figures/terrain/*.pgm` — 地形对照图的原始灰度数据
