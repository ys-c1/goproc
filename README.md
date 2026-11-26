# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/divide/` 逐格流向、平地处理、分水线格边与连通描线、独立脊线、PNG 渲染
- `cmd/goproc/` 命令行，`gen` 导出高度图，`divide` 导出分水线全套产物
- `tools/crest_vs_divide.py` 分水线与独立脊线的逐像素双向偏差比对
- `docs/divide.md` 分水线两条证明、平地处理与复杂度说明

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc divide --size 128 --seed 19 --octaves 2 --freq 0.02 --scale 5 \
  --f64 frame.f64 --prefix frame
python3 -I tools/crest_vs_divide.py --heights frame.f64 --edges frame_divide.txt \
  --crest frame_crest.txt --tol 2 --out frame_compare.txt
bash artifacts/build/measure.sh   # 两个画幅端到端复现
```

`divide` 子命令产物（前缀 `frame`）：

| 文件 | 内容 |
|---|---|
| `frame.f64` | 四字节小端边长 + float64 高度 |
| `frame_divide.txt` | 完整分水线格边 `H x y` / `V x y`，按水平/竖直分类 |
| `frame_opposite.txt` | 仅“两侧流向严格相反”的格边子集 |
| `frame_crest.txt` | 独立脊线像素 `x y` |
| `frame_gray.png` / `frame_overlay.png` | 灰度图 / 叠加图（蓝=水平边，红=竖直边） |
| `frame_compare.txt` | 两组方向的偏差像素数、段落数与超 2 格段落坐标 |

