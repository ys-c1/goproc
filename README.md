# goproc

程序化地形生成，确定性种子派生；双光源 Lambert 着色与山脊明暗交替拍频分析。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出/解码（P5 八位灰度）
- `internal/shade/` 双方向光源 Lambert 着色
- `cmd/goproc/` 命令行：`gen` 导出高度图，`shade` 双光源着色
- `analysis/` 拍频证明与逐行核验的 Python 脚本（一律 `python3 -I` 运行），结论见 `artifacts/REPORT.md`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc shade --in map.pgm --out shade.pgm --phi1 37.5 --phi2 52.5 --elev 20
```

`shade` 参数：`--phi1/--phi2` 两灯方位角（度，从 +y 轴顺时针），`--elev` 共同仰角（度）。

分析复现（详见 `artifacts/REPORT.md` §6）：

```
cd analysis
python3 -I make_terrains.py
python3 -I verify_angular_beat.py
python3 -I run_experiment.py
python3 -I render_go_and_count.py
python3 -I make_figures.py
```
