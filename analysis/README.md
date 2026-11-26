# 八位量化直方图空隙分析

## 复现步骤

```bash
# 1. 导出两个画幅（种子 7，参数与 README 示例一致：octaves=4 lacunarity=2 gain=0.5 freq=0.02）
cd analysis/godump && go build -o /tmp/godump . && cd ../..
/tmp/godump 64  artifacts
/tmp/godump 256 artifacts

# 2. 多种子批处理（60 个种子，用于标定空间相关导致的有效样本量）
cd analysis/gobatch && go build -o /tmp/gobatch . && cd ../..
/tmp/gobatch 64  $(seq 1 60 | tr '\n' ' ') > artifacts/batch_64.jsonl
/tmp/gobatch 256 $(seq 1 60 | tr '\n' ' ') > artifacts/batch_256.jsonl

# 3. 分析、标定、出图（venv 内含 numpy/matplotlib）
.venv/bin/python -I analysis/gap_analysis.py
```

产物：

| 文件 | 内容 |
|---|---|
| `artifacts/floats_{64,256}.bin` | 归一化浮点高度（float64 小端） |
| `artifacts/map_{64,256}.pgm` | 八位量化结果（注意：仓库 `EncodePGM` 头含字面 `\n`，非标准 P5） |
| `artifacts/batch_{64,256}.jsonl` | 60 种子各自的 256 档计数与空隙档 |
| `artifacts/metrics.json` | 两个画幅的全部指标与三种预测值 |
| `artifacts/histogram_{64,256}.png` | 量化直方图各一张（红三角=空隙档） |

量化与空隙口径：

- 量化规则与 `internal/hmap.EncodePGM` 完全一致：`g = clamp(round(x*255), 0, 255)`，Python 侧结果与 PGM 负载逐字节核对。
- 空隙 = 灰度档 k∈{1,…,254} 中计数为零的档；**首尾两档（0、255）不计**。归一化保证 min/max 恰好为 0/1，故 0、255 档恒非空。

## 理论与期望表达式

设 n 个像素独立同分布地量化到 256 档，像素落入档 k 的概率为 p_k。

1. 指示变量 I_k = 1{档 k 计数为零}，空隙数 G = Σ_{k=1}^{254} I_k。
   空隙"个数"按定义就是未被任何像素落入的内部灰度档数（同一计数的两种表述，恒等）。
2. P(I_k=1) = (1−p_k)^n（n 次抽样全部避开档 k）。
3. 期望（线性期望，不要求各 I_k 独立）：

   E[G] = Σ_{k=1}^{254} (1−p_k)^n

   - 均匀占位近似：支撑宽度 W 档且各档等概率 p_k=1/W，则
     E[G] ≈ W·(1−1/W)^n ≈ **W·e^(−n/W)**。
   - 非均匀形状：p_k 由该画幅的高斯 KDE（Scott 带宽）在各档区间积分得到，直接代入求和式。
   - 空间相关修正：FBM 像素不独立，以有效样本量 n_eff 替换 n；n_eff 由 59 个标定种子（留一种子 7 验证）的实测空隙均值反解。

## 结果（种子 7）

| 画幅 | n | W（参与率宽度，档） | 均匀 W·e^(−n/W) | 形状感知 E[G]（iid） | 相关性修正 E[G]（n_eff） | 实测空隙 | 偏差（修正预测−实测） |
|---|---|---|---|---|---|---|---|
| 64×64 | 4096 | 185.5 | ≈0（4.8e-8） | 2.230 | 2.574（n_eff=3848，0.94n） | 3（档 6,9,12） | −0.43 档 |
| 256×256 | 65536 | 185.4 | ≈0（5.4e-152） | 0.023 | 0.038（n_eff=58762，0.90n） | 1（档 253） | −0.96 档 |

两个画幅偏差均不超过两档，无超差画幅。均匀模型失效是因为分布明显非均匀（参与率宽度仅 ~185 档而非 254 档），空隙集中在低概率尾部档。
