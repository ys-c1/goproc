# goproc

程序化地形生成，确定性种子派生；在高度图之上做水面镜面高光的离线标定与在线预判。

- `api/` 对外入口，`Options` 是参数声明处
- `internal/noise/` 值噪声，置换表由 splitmix64 派生
- `internal/fbm/` 分形叠加
- `internal/hmap/` 归一化与 PGM 导出
- `internal/water/` 水面法线、镜面反射着色与白像素统计
- `internal/calibrate/` 阈值二次曲线拟合、临界夹角反解、在线预判与偏差对照
- `cmd/goproc/` 命令行，`gen` 导出高度图，`highlight` 跑标定—预判—对照全链路

```
make test
go run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out map.pgm
go run ./cmd/goproc highlight --calib 20,40,60 --check 30,50
```

## 高光标定与预判

反射系数模型（gamma 为像素法线与半向量 H 的夹角）：

```
F(v) = F0 + (1-F0) * v^Q          v = cos gamma，对齐增益
rho  = K * F(v) * v^P
```

rho 经八位灰度量化后顶格（灰度 255，即 rho >= 254.5/255）的像素记为白像素。

1. **离线标定**：在三个已知光照仰角下完整渲染，取每个角度白像素中的最小反射
   系数作为该角度的阈值 T。
2. **二次曲线拟合**：三个标定点唯一解出 T(t) = a*t^2 + b*t + c，t 为光照角（度）。
3. **临界夹角反解**：反射系数随 gamma 单调递减，在 rho = T(t) 处二分求根，得到
   该光照角下的临界夹角 gamma*（法线与半向量夹角的上界）。
4. **在线预判**：只输入光照角，逐像素比较几何夹角与 gamma*，不跑任何渲染，
   输出高光掩码与像素数。
5. **对照留档**：另取两个光照角，预判结果与渲染后的白像素统计对照。偏差按到
   对方集合的最近格距（3-4 倒角距离）计，严格超过两格的像素在偏差掩码中标为
   二五五，两格内的不一致标为一二八。

`highlight` 在留档目录（默认 `artifacts/highlight/`）输出：

- `calib_angleXXX_white.pgm` / `_render.pgm`：标定点的白像素掩码与渲染灰度图
- `check_angleXXX_pred.pgm`：在线预判高光掩码
- `check_angleXXX_render_white.pgm` / `_render.pgm`：渲染白像素掩码与灰度图
- `check_angleXXX_deviation.pgm`：偏差标注图（0 一致 / 128 两格内 / 255 超两格）
- `report.json`：阈值、二次曲线系数、临界夹角、计数对照与全部留档文件名
