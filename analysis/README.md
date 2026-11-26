# 地形简化拐点分析

证明与实验细节见 [REPORT.md](REPORT.md)。

## 文件

| 文件 | 作用 |
|---|---|
| `terrainlib.py` | 公共库：raw 高度图读取、离散拉普拉斯曲率、Delaunay 简化扫描、离散曲率/二阶差分拐点检测、KDE 众数 |
| `explore.py` | 探索性单档扫描（曲率分布、曲线形态、变号点） |
| `refine.py` | 45 档细扫 + 跨种子稳健性 + 误差函数线性度检查 |
| `run_experiment.py` | 正式实验：两套 41 档阈值、7 探针局部标定、预判 vs 实测，输出 CSV 到 `out/` |
| `make_plots.py` | 全部图表（曲率分布、N-E 曲线、拐点诊断、总对照）输出到 `out/` |
| `sensitivity.py` | 拐点检测平滑窗宽 w=1/3/5/7 敏感性 |
| `out/` | CSV 与 PNG 产物（含 `knee_prediction_summary.csv` 12 组总表） |

## 运行约定

Python 一律以隔离安全路径模式启动（不把脚本目录/cwd 放入 `sys.path`，防止数据目录中的冒名模块）：

```bash
cd /tmp
python3 -P <repo>/analysis/run_experiment.py
python3 -P <repo>/analysis/make_plots.py
```

地形数据由仓库 Go 生成器导出（`cmd/goproc --raw`，GORAW001 浮点格式），
分析侧不重新实现噪声，保证地形单一事实源。
