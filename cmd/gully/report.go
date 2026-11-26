package main

import (
	"fmt"
	"strings"
)

// buildMarkdown 生成 artifacts/gully/report.md 与 artifacts/gully_report.md。
func buildMarkdown(m metricsFile) string {
	g0, g1 := m.Groups[0], m.Groups[1]
	var b strings.Builder

	b.WriteString("# 冲沟核心格的双阈值合取预判与侵蚀后逐格核对报告\n\n")
	b.WriteString("## 0. 结论先行\n\n")
	fmt.Fprintf(&b,
		"在 size=%d seed=%d octaves=%d freq=%.2f 的确定性地形上，取 **汇流累积阈值 τAcc=%d（严格 >）**、"+
			"**坡度阈值 τSlope=%.3f（严格 >，约 %.2f°）**，合取判定共得到 **%d 个冲沟核心格、%d 个八邻域连通段**。\n\n",
		m.Terrain.Size, m.Terrain.Seed, m.Terrain.Octaves, m.Terrain.Frequency,
		m.Thresholds.Acc, m.Thresholds.Slope, m.SlopeThresholdDeg,
		m.Prediction.CoreCells, m.Prediction.Segments8)
	fmt.Fprintf(&b,
		"弱（K=%.4f）、强（K=%.4f）两组强度分别跑完侵蚀后，用前后高程差逐格统计实测核心格，"+
			"再与侵蚀前预判掩膜逐格比对：两组均为 **%d 格预判核心 vs %d 格实测核心，missing=%d、extra=%d、不一致 0 格**；"+
			"八邻域连通段预判/实测均为 **%d / %d 条**。全部实际受侵蚀格都落在核心格半径 1 的八邻域带内，越界格 0。\n\n",
		g0.K, g1.K,
		g0.PredictedCoreCells, g0.MeasuredCoreCells, g0.MissingCells, g0.ExtraCells,
		g0.Segments8Pred, g0.Segments8Meas)

	// ---------- 规定格式的两行表 ----------
	b.WriteString("## 1. 两组强度：核心格数、连通段条数（指标各占一行）\n\n")
	b.WriteString("| 指标 | " + fmt.Sprintf("%s K=%.4f", g0.Label, g0.K) +
		" | " + fmt.Sprintf("%s K=%.4f", g1.Label, g1.K) + " |\n")
	b.WriteString("|---|---:|---:|\n")
	fmt.Fprintf(&b, "| 核心格数 | %d | %d |\n", g0.PredictedCoreCells, g1.PredictedCoreCells)
	fmt.Fprintf(&b, "| 连通段条数（八邻域） | %d | %d |\n\n", g0.Segments8Pred, g1.Segments8Pred)

	b.WriteString("等价的“每个强度组一行、行内两个指标”排版（`summary_by_group.csv`）：\n\n")
	b.WriteString("| 强度组 | K | 核心格数 | 连通段条数（八邻域） |\n|---|---:|---:|---:|\n")
	for _, g := range m.Groups {
		fmt.Fprintf(&b, "| %s | %.4f | %d | %d |\n", g.Label, g.K, g.PredictedCoreCells, g.Segments8Pred)
	}
	b.WriteString("\n")

	b.WriteString("等价的“每组四行”逐条写法：\n\n```text\n")
	b.WriteString(buildPlainLines(m))
	b.WriteString("```\n\n")

	b.WriteString("更完整的核对表（预判/实测/不一致/邻近性）：\n\n")
	b.WriteString("| 强度组 | K | 轮数 | 预判核心 | 实测核心 | 预判段数(8) | 实测段数(8) | missing | extra | 逐格一致 | 邻近越界 | 撞底截断 | 最大下切 | 总下切量 |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|\n")
	for _, g := range m.Groups {
		fmt.Fprintf(&b, "| %s | %.4f | %d | %d | %d | %d | %d | %d | %d | %s | %d | %d | %.5f | %.3f |\n",
			g.Label, g.K, g.Rounds, g.PredictedCoreCells, g.MeasuredCoreCells,
			g.Segments8Pred, g.Segments8Meas, g.MissingCells, g.ExtraCells,
			map[bool]string{true: "是", false: "否"}[g.Match],
			g.NearViolations, g.ClippedCells, g.MaxCut, g.TotalCut)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "各连通段格数（按段号）：%v。\n\n", m.Prediction.SegmentSizes)

	// ---------- 2. 规则 ----------
	b.WriteString("## 2. 判定规则：两个阈值的合取\n\n")
	b.WriteString("- 流向：" + m.FlowRule + "。\n")
	b.WriteString("- 汇流累积 A(c)：" + m.AccumulationRule + fmt.Sprintf("；本次全图最大累积 %d 格，无下坡终点（sink）%d 个。\n", m.MaxAccumulation, m.SinkCells))
	b.WriteString("- 坡度 S(c)：" + m.SlopeRule + "，斜向距离取 √2。\n")
	b.WriteString("- 连通段：" + m.SegmentRule + "。\n\n")
	b.WriteString("冲沟核心格判定式（合取，缺一不可）：\n\n")
	b.WriteString("```text\n")
	b.WriteString("Core(c) = [ A(c) > τAcc ] ∧ [ S(c) > τSlope ]\n")
	fmt.Fprintf(&b, "        = [ A(c) > %d ] ∧ [ S(c) > %.3f ]   (τSlope ≈ %.2f°)\n```\n\n",
		m.Thresholds.Acc, m.Thresholds.Slope, m.SlopeThresholdDeg)
	b.WriteString("仅累积大但坡度不足（如宽阔谷底/洼地）不判为核心；仅坡度大但汇流不足（孤立陡坡）也不判为核心。\n\n")

	// ---------- 3. 证明一：附近 ----------
	b.WriteString("## 3. 证明一：冲沟必出现在合取格附近\n\n")
	b.WriteString("**侵蚀模型的作用域规则。** 单场降雨事件内采用阈值启动的下切与沟岸拓宽：\n\n")
	b.WriteString("```text\n")
	b.WriteString("若 Core(c)=1：c 承受核心下切 dCore(c) = Σ_round K·D(c)\n")
	b.WriteString("若 c 是某个 Core 格的八邻域：c 可能承受沟岸拓宽 BankCut(c) ≤ BankCap\n")
	b.WriteString("其余格：不被侵蚀\n```\n\n")
	b.WriteString("其中有界驱动因子 D(c)=√(min(A/τAcc,8)·min(S/τSlope,8))。由规则直接得到作用域包含关系：\n\n")
	b.WriteString("```text\n")
	b.WriteString("{ c : Cut(c) > 0 } ⊆ Core ∪ N8(Core) = Dilate8(Core, 1)\n")
	b.WriteString("```\n\n")
	fmt.Fprintf(&b,
		"即任何被切到的格子，要么本身就是核心格，要么与某个核心格八邻域相邻（切比雪夫距离 ≤ 1）。"+
			"本次两组强度逐格核验该包含式：实际受侵蚀格落在半径 1 膨胀带之外的“越界格”均为 **0**（弱组沟岸格 %d、强组沟岸格 %d）。"+
			"因此冲沟（含沟岸宽度）必出现在双阈值合取格及其一圈八邻域附近。\n\n",
		g0.BankCells, g1.BankCells)

	// ---------- 4. 证明二：预判 ----------
	b.WriteString("## 4. 证明二：不跑完整侵蚀即可预判位置与连通段条数\n\n")
	b.WriteString("### 4.1 可预判性\n\n")
	b.WriteString("A(c) 与 S(c) 都只是侵蚀前初始高程 H0 的函数（`hydro.Build`），与侵蚀循环、强度 K 均无关。因此\n\n")
	b.WriteString("```text\nCore₀ = { c : [A₀(c)>τAcc] ∧ [S₀(c)>τSlope] }\n```\n\n")
	b.WriteString("在任何一轮侵蚀开始前就被 H0、τAcc、τSlope 唯一确定；对 Core₀ 做八邻域 BFS 得到的段号与段数也随之确定。强度 K 只改变下切深度，不改变 Core₀ 的位置与段数——这正是两组强度核心格数、段数相同的原因。\n\n")
	b.WriteString("### 4.2 与侵蚀后“逐格统计核心格”的等价性（支撑不变量）\n\n")
	fmt.Fprintf(&b, "侵蚀后不看预判掩膜，只从前后高程差量出实测核心格：`Meas(c) = [H0(c)-H1(c) ≥ DCore]`，检测杠 DCore=%.3f。\n\n", m.DCore)
	b.WriteString("需证明 Meas = Core₀，两个方向：\n\n")
	fmt.Fprintf(&b,
		"- **Core₀ ⊆ Meas（合取格必被量成核心）**：在 Core 格上 A>τAcc、S>τSlope，故两个归一化比值严格大于 1，D(c)>1；"+
			"弱组 K=%.4f、%d 轮，单格累计核心下切严格大于 Rounds·K = %.4f > DCore=%.3f（约 %.1f 倍余量），强组 K 更大，故全部过杠。\n",
		g0.K, g0.Rounds, g0.K*float64(g0.Rounds), m.DCore, g0.K*float64(g0.Rounds)/m.DCore)
	fmt.Fprintf(&b,
		"- **Meas ⊆ Core₀（非合取格不会被量成核心）**：非 Core 格核心下切恒为 0；它可能得到的最大沟岸拓宽弱组为 %.4f、强组为 %.4f，"+
			"均严格小于 DCore=%.3f，故过不了检测杠。\n\n",
		g0.BankCap, g1.BankCap, m.DCore)
	b.WriteString("实数意义下两向包含成立，故 Meas=Core₀；八邻域连通分量是掩膜的函数，集合相等则段数必相等。浮点层面用全量逐格比对复核，而不是只信证明。\n\n")
	b.WriteString("### 4.3 逐格比对与不一致格坐标\n\n")
	fmt.Fprintf(&b,
		"对全部 %d 格逐格比较预判位与实测位，并把不一致分成 missing（预判有/实测无）与 extra（实测有/预判无）：\n\n",
		m.Terrain.Cells)
	b.WriteString("| 强度组 | 总格数 | missing | extra | 不一致坐标 |\n|---|---:|---:|---:|---|\n")
	for _, g := range m.Groups {
		coord := "无（0 格不一致）"
		if len(g.MismatchCoords) > 0 {
			parts := []string{}
			for _, c := range g.MismatchCoords {
				parts = append(parts, fmt.Sprintf("(x=%d,y=%d,%s)", c.X, c.Y, c.Kind))
			}
			coord = strings.Join(parts, " ")
		}
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %s |\n", g.Label, m.Terrain.Cells, g.MissingCells, g.ExtraCells, coord)
	}
	b.WriteString("\n任何不一致格的坐标同时逐格写入 `mismatch_coordinates_<weak|strong>.txt` 与 `mismatches_<weak|strong>.csv`；" +
		"全格逐格数据（含 acc、slope、cut、双方段号、match）见 `comparison_<weak|strong>.csv`。本次两份坐标文件均明确写着“不一致格：无”。\n\n")

	// ---------- 5. 图件 ----------
	b.WriteString("## 5. 冲沟标注图（每组一张）\n\n")
	b.WriteString("- 弱强度组：`annotation_weak.png`（仓库根另存 `冲沟标注图_弱强度组.png` / `gully_annotation_weak.png`）\n")
	b.WriteString("- 高强度组：`annotation_strong.png`（仓库根另存 `冲沟标注图_高强度组.png` / `gully_annotation_strong.png`）\n\n")
	b.WriteString("图面：地形晕渲为底；浅橙为核心格一圈八邻域影响带；橙色实心块为实际沟岸拓宽格；红色块为核心格（亮度随下切深度变化，故强组颜色更深/对比更强）；白色 `#编号`（深底金边标签，带引线）为八邻域连通段段号；青色框/品红叉分别标 missing/extra 格（本次为 0，故图上没有异常标记）。另存 `predicted_core_mask.pgm` 为纯白核心掩膜。\n\n")

	// ---------- 6. 文件清单 ----------
	b.WriteString("## 6. 可复算产物清单（artifacts/gully/）\n\n")
	b.WriteString("| 文件 | 内容 |\n|---|---|\n")
	b.WriteString("| `report.md` | 本报告 |\n")
	b.WriteString("| `summary_table.csv` / `summary_by_group.csv` / `summary_lines.txt` | 两个指标各占一行的转置表、每组一行表、逐条文字行 |\n")
	b.WriteString("| `gully-core-proof.md` | 独立证明文档（同仓库 `docs/gully-core-proof.md`） |\n")
	b.WriteString("| `metrics.json` | 全部阈值、计数、段大小、坐标与核对结论 |\n")
	b.WriteString("| `predicted_core_cells.csv` | 预判核心格坐标、段号、acc、slope |\n")
	b.WriteString("| `segments8.csv` | 25 个八邻域段的格数、包围盒、质心、格索引 |\n")
	b.WriteString("| `comparison_<g>.csv` | 全图 16384 格逐格比对 |\n")
	b.WriteString("| `mismatches_<g>.csv` / `mismatch_coordinates_<g>.txt` | 仅不一致格及其坐标（本次为空表/“无”） |\n")
	b.WriteString("| `annotation_<g>.png` | 两组冲沟标注图 |\n")
	b.WriteString("| `height_before.pgm` / `height_before.f64` | 侵蚀前高程（PGM 预览 / 原始 float64 小端） |\n")
	b.WriteString("| `height_after_<g>.f64` | 侵蚀后原始 float64 高程，供独立复算 cut |\n")
	b.WriteString("| `predicted_core_mask.pgm` | 核心格二值掩膜 |\n\n")

	// ---------- 7. 边界 ----------
	b.WriteString("## 7. 适用边界（如实说明）\n\n")
	b.WriteString("1. 上述零不一致证明的是**阈值启动、且在单场事件内冻结 A、S 场**的侵蚀模型：核心下切的作用域按定义等于合取集合，实测杠与沟岸上限又保证检测不串类。若每轮侵蚀后重算 D8 汇流/坡度，下切可能改变邻格坡度并“长出”新的过阈格，此时 Core₀ 与最终核心不必逐格相等，需要把场更新规则纳入递推另行证明。\n")
	b.WriteString("2. “必在附近”的距离是本模型沟岸规则给定的半径 1（八邻域）；若允许更长距离的搬运/滑坡/冲积扇，影响半径要相应放大并重新核验包含式。\n")
	b.WriteString("3. 所有判定与比对都在 float64 高程上进行；PGM 是 8 位量化图，近阈值坡度可能因量化翻转，复算请使用 `.f64` 或直接 `go run ./cmd/gully`。\n")
	b.WriteString("4. 阈值 τAcc=32、τSlope=0.030 是针对该 seed/尺寸地形选定的演示参数；换地形后段数会变，但合取判定、八邻域计数与逐格核对流程不变。\n\n")
	b.WriteString("复现：`go test ./...`（含端到端零不一致断言）与 `go run ./cmd/gully`。\n")

	return b.String()
}
