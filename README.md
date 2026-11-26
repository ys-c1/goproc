# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
```

## LOD 接缝：整除判据、缝宽系数与零缝修复

`internal/lod/` 给出相邻 LOD 边界顶点的坐标整除判据、缝宽—格距差比例定律
（一般系数 $C=\rho$；整数截断模型 $C=1$，像素中心半格登记 $C=1/2$），
并用 Hausdorff 距离实测缝宽与最大高程误差。LOD 二/三/四级（格距 4/8/16 px）
经“整除对齐 + 共享边界顶点(zipper)”修复后缝宽严格为 0。

```
make lod       # = go run ./cmd/goproc lodcheck --out artifacts/lod
```

产物（证明、成组指标、每级分别存档的拼接图）在
[`artifacts/lod/`](artifacts/lod/PROOF.md)：`PROOF.md`、`metrics.json`、`table.md`、
`lod{2,3,4}_stitched.png`、`lod{2,3,4}_compare.png`。

