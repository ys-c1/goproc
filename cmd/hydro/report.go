package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"goproc/internal/hydro"
)

type reportInput struct {
	size, octaves, bands        int
	seed                        uint64
	an                          *hydro.Analysis
	rows                        []bandRow
	eventBand                   []int
	scanDur, bfsDur             time.Duration
	mismatchAll, mismatchShow   int
	boundaryMinima, geomSaddles int
	saddleLocalMax, maxK        int
	sumDelta, sumKM1            int
	changedBands                int
}

func writeReport(path string, in reportInput) error {
	var b strings.Builder
	an := in.an

	fmt.Fprintf(&b, "# 水陆连通分量：扫描预测与逐档实测核对\n\n")
	fmt.Fprintf(&b, "参数：size=%d seed=%d octaves=%d；展示水位 %d 档（0 到 1 等距）。\n\n",
		in.size, in.seed, in.octaves, in.bands)
	fmt.Fprintf(&b, "约定：水 W(w)={p: h(p)≤w}；连通=四邻域；**网格外既不是水域也不是陆地，不参与连接、不环绕、无外部海洋**，边界格点按普通格点处理（只是有效邻格更少）。\n\n")

	fmt.Fprintf(&b, "## 一、各级水位预测与实测逐一比对\n\n")
	b.WriteString("| 档 | 水位 | 预测分量 | 实测分量(BFS) | 一致 | 本档极小台地 | 本档鞍点台地 | 净变化 |\n")
	b.WriteString("|---:|---:|---:|---:|:---:|---:|---:|---:|\n")
	for _, r := range in.rows {
		fmt.Fprintf(&b, "| %d | %.4f | %d | %d | %s | %d | %d | %+d |\n",
			r.index, r.level, r.predicted, r.measured,
			mark(r.predicted == r.measured), r.births, r.merges, r.net)
	}

	fmt.Fprintf(&b, "\n## 二、跳变点逐行（只列引起分量数变化的临界台地：极小与鞍点，共 %d 个；普通点 %d 个见 jumps.csv）\n\n",
		an.Minima+an.Saddles, an.Regular)
	b.WriteString("`K` = 台地入水前四邻域已入水的不同分量数，`Δ=1-K`；")
	b.WriteString("`pattern` 按 U D L R 给代表格邻格高低（`<`低 `>`高 `=`同高 `#`网格外）。")
	b.WriteString("`geom_saddle` 是八邻域几何鞍点判定，仅作对照，不参与预测。\n\n")
	b.WriteString("| # | 高度档 | Y | X | 高度 | 类型 | 格点数 | K | Δ | pattern | 几何鞍点 | 局部极大 |\n")
	b.WriteString("|---:|---:|---:|---:|---:|:---|---:|---:|---:|:---:|:---:|:---:|\n")
	n := 0
	for i, e := range an.Events {
		if e.Kind == hydro.KindRegular {
			continue
		}
		n++
		fmt.Fprintf(&b, "| %d | %d | %d | %d | %.6f | %s | %d | %d | %+d | `%s` | %s | %s |\n",
			n, in.eventBand[i], e.Y, e.X, e.Height, zhKind(e.Kind), len(e.Cells),
			e.K, e.Delta, e.Pattern, mark(e.GeomSaddle), mark(e.LocalMax))
	}

	fmt.Fprintf(&b, "\n## 三、结构核对\n\n")
	b.WriteString("- 格点总数 ")
	fmt.Fprintf(&b, "%d；不同高度 %d；四邻域同高相邻对 %d。\n",
		in.size*in.size, len(an.Records), an.EqualNeighborPairs)
	fmt.Fprintf(&b, "- 临界台地：极小 %d 个、鞍点 %d 个、普通 %d 个；鞍点 K 最大值 %d。\n",
		an.Minima, an.Saddles, an.Regular, in.maxK)
	fmt.Fprintf(&b, "- 分量数发生变化的高度档 %d 个；发生变化的临界台地 %d 个（=极小+鞍点=%d+%d）。\n",
		in.changedBands, an.Minima+an.Saddles, an.Minima, an.Saddles)
	fmt.Fprintf(&b, "- 恒等式：ΣΔ = %d，等于 极小数 − Σ(K−1) = %d − %d = %d；全域淹没末态分量数 %d。\n",
		in.sumDelta, an.Minima, in.sumKM1, an.Minima-in.sumKM1,
		an.Records[len(an.Records)-1].Components)
	fmt.Fprintf(&b, "- 全部 %d 个不同高度上，预测与独立 BFS 实测不一致 **%d** 处；%d 个展示档不一致 **%d** 处。\n",
		len(an.Records), in.mismatchAll, in.bands, in.mismatchShow)
	fmt.Fprintf(&b, "- 鞍点台地中代表格为四邻域局部极大的 %d 个；八邻域环几何鞍点 %d 个（对照量）。\n",
		in.saddleLocalMax, in.geomSaddles)
	b.WriteString("  - 几何鞍点口径：沿顺时针八邻环读低/高符号，内点环上符号交替 4 次、边界点在有效邻格环上交替 2 次；该判定**不参与分量预测**，与合并台地（K>=2）不等价。\n")
	fmt.Fprintf(&b, "- 位于边界的极小台地 %d 个（边界不引入外部海洋，边界低点各自成域）。\n",
		in.boundaryMinima)
	fmt.Fprintf(&b, "- 一次扫描耗时 %s；把全部 %d 个不同高度逐档重做 BFS 耗时 %s。\n",
		in.scanDur, len(an.Records), in.bfsDur)
	b.WriteString("- 预测只需一次 O(n log n) 扫描，之后任意水位查询 O(log n)；上表 BFS 仅供独立核对，不参与预测。\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func mark(ok bool) string {
	if ok {
		return "是"
	}
	return "否"
}

func zhKind(k hydro.EventKind) string {
	switch k {
	case hydro.KindMinimum:
		return "极小"
	case hydro.KindSaddle:
		return "合并"
	default:
		return "普通"
	}
}
