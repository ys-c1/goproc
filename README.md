# goproc

程序化地形生成，确定性种子派生。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/slope/` 坡度估计，含边缘一圈的边界延拓补偿
- `cmd/goproc/` 命令行，`gen` 子命令导出高度图
- `cmd/slopeprobe/` 坡度图与补偿前后逐像素比对，产物写入 `artifacts/slope/`

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
make slope
```

## 坡度图的边缘偏置

内部像素用中心差（双侧差）估坡度，边缘一圈只有单侧差可用。把两者在
同一像素展开，单侧差相对双侧差的偏量为

```
Δ = F − C = (dx/2)·κ
```

即「格距 × 曲率 / 2」。`dx=1` 时 `Δ=κ/2`，等于该格二阶差分的一半。
偏量是一阶的，所以未补偿时边缘坡度整体偏移、与内侧不连续（`internal/slope`
的 `TestConvergenceOrder` 给出未补偿边缘收敛阶约 1、内部约 2）。

补偿方式是在画幅外补一圈鬼点，取 `h(-1)=2h(0)-h(1)+κ_est`，再在延拓场上
做中心差。这样扣掉的正是上面的 `Δ`：`TestCompensationEqualsBias` 逐像素验证
「未补偿 − 补偿后 = Δ」，`TestConvergenceOrder` 验证补偿后边缘收敛阶回到约 2、
误差降至与内部同量级。补偿只改边缘一圈（一格宽），内部坡度逐位不变。

`make slope` 会对 128 与 96 两个画幅各出补偿前、补偿后两张坡度图，并逐像素比对：
色阶改变像素数、补偿后最大偏差（补偿前后逐像素坡度模长的最大差），且校验全部
改变像素都落在边缘一圈内。
产物见 `artifacts/slope/`（含 `report.csv` 两栏表）。
