# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/caliper/` 凸包（Andrew 单调链）与旋转卡壳，求最远点对的预判与暴力对照
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/caliper/` 两个画幅的最远点对预判/实测比对与存档
- `docs/caliper-proof.md` 最远点对必在凸包上、必为对踵点对、卡壳可预判的论证
- `analysis/caliper_verify.py` 独立复算（numpy），与 Go 产物逐项对拍

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
make caliper
```

## 最远点对：凸包 + 旋转卡壳 vs 暴力遍历

`make caliper` 等价于

```bash
go run ./cmd/caliper --out artifacts/caliper
python3 -I analysis/caliper_verify.py artifacts/caliper
```

候选点定义为归一化高度 `>= 0.5` 的整数格点（该阈值与 PGM 灰度 `>= 128` 严格等价）。两个画幅并排结论：

| 项目 | 画幅 A | 画幅 B |
|---|---|---|
| 画幅 | 128×128 | 192×192 |
| 种子 / 层数 | 7 / 4 | 9 / 4 |
| 候选点数 | 9773 | 11357 |
| **凸包顶点数** | **7** | **13** |
| **卡壳轮数** | **7** | **13** |
| 游标推进步数 | 7 | 14 |
| 对踵点对数 | 9 | 15 |
| **预判距离（像素）** | **177** | **256** |
| 预判点坐标 | (3,127)–(127,0) | (0,170)–(191,0) |
| **暴力距离（像素）** | **177** | **256** |
| 暴力点坐标 | (3,127)–(127,0) | (0,170)–(191,0) |
| 取整距离一致 | 是 | 是 |

两画幅一致，无「不一致画幅」，故无需报告不一致坐标。完整的逐对对踵点表、
耗时与凸包顶点清单见 `artifacts/caliper/report.txt`，机器可读汇总见
`summary.csv` / `summary.json`。最远点连线图各留一张：`link_A.pgm`、`link_B.pgm`
（灰度分层：0–114 地形、150 凸包边、235 最远点连线、255 端点）。

## 缺陷修复

`internal/hmap` 的 PGM 文件头此前误写成字面量 `\n`（反斜杠 + n），
标准解码器会把尺寸读错。已改为真实换行，并由 `TestEncodePGMHeader` 锁定。
