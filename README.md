# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生；另含边界槽位追踪与条目对齐原语
- `internal/fbm/` 分形叠加，含置换表槽位追踪
- `internal/hmap/` 归一化与 PGM 导出
- `internal/seam/` 接缝高度差实验：拼接、度量、带标注 PNG/PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/seam/` 命令行，构造两块地形并验证接缝差对齐后恒为零
- `verify/verify_pgm.py` 独立 Python 复算脚本（仅标准库，`python3 -I` 运行）

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

## 接缝实验

两块地形各自由种子派生置换表；让两块的共享边界落在同一世界坐标线上，
则对齐前接缝两侧高度差唯一来源于两表边界条目的差。把 A 表在边界槽位上的
条目复制进 B 表（两表在边界处取同一批条目），B 重采样后边界高度逐位相等，
接缝差恒为零。

```
go run ./cmd/seam --size 128 --seed-a 11 --seed-b 47 --outdir artifacts/seam
python3 -I verify/verify_pgm.py artifacts/seam
```

接缝差定义为接缝两侧相邻像素灰度差的最大值：竖拼按同列，横拼按同行
（图旋转九十度后等价于同列，报告中给出旋转交叉验证值）。输出每个方向
对齐前后各一张带红色接缝标注的 PNG 与一张未标注 PGM，度量写入
`metrics.txt`。全部图共用同一灰度量化尺度，同一原始高度灰度处处相同。
