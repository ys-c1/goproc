// lodstudy 命令对一张确定性 FBM 高度图做分块 LOD 层级分配与验证。
//
// 输出（每档允许误差一份）：
//   - 预测层级与试渲染最低层级的逐行列矩阵（.txt）
//   - 偏低块坐标清单（.txt，空集也明确写出）
//   - 两套层级色块图（.ppm，分别对应预测与最低可行层级）
//   - 汇总统计（stdout 与 summary.txt）
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/lod"
)

func main() {
	size := flag.Int("size", 128, "高度图边长（需能被块边长整除）")
	seed := flag.Uint64("seed", 7, "噪声种子")
	octaves := flag.Int("octaves", 4, "FBM 层数")
	tile := flag.Int("tile", 8, "每块像素边长")
	lmax := flag.Int("lmax", 4, "最细层级（L=0 步长 2^lmax）")
	outDir := flag.String("out", "artifacts/lodstudy", "输出目录")
	flag.Parse()

	// 两档允许误差（高度已归一化到 [0,1]）。
	tols := []float64{0.02, 0.01}

	hm, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: 0.02,
	})
	if err != nil {
		fatal("generate: %v", err)
	}
	if *size%*tile != 0 {
		fatal("size %d 必须能被 tile %d 整除", *size, *tile)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal("mkdir: %v", err)
	}

	var summary strings.Builder
	fmt.Fprintf(&summary, "# LOD 层级分配汇总\n\n")
	fmt.Fprintf(&summary, "高度图 size=%d seed=%d octaves=%d | 块 tile=%d | 层级 L=0..%d（步长 %d..1）\n\n",
		*size, *seed, *octaves, *tile, *lmax, 1<<*lmax)

	for _, tol := range tols {
		g := lod.Build(hm.Data, hm.Size, *tile, *lmax, tol)
		tag := fmt.Sprintf("tol%g", tol)

		// 逐行列层级矩阵。
		pred := g.LevelGrid(true)
		mins := g.LevelGrid(false)
		writeMatrix(filepath.Join(*outDir, "levels_pred_"+tag+".txt"), "预测层级（多尺度曲率对数公式）", pred)
		writeMatrix(filepath.Join(*outDir, "levels_min_"+tag+".txt"), "试渲染最低可行层级（基准）", mins)

		// 色块图：预测与最低可行各一份。
		ppmPred := lod.EncodeLevelPPM(pred, *lmax, 24)
		ppmMin := lod.EncodeLevelPPM(mins, *lmax, 24)
		writeBytes(filepath.Join(*outDir, "levels_pred_"+tag+".ppm"), ppmPred)
		writeBytes(filepath.Join(*outDir, "levels_min_"+tag+".ppm"), ppmMin)

		// 偏低块：多尺度（应空）与朴素单尺度（对照）。
		under := g.Underpredicted()
		naive := g.NaiveUnderpredicted()
		writeUnder(filepath.Join(*outDir, "underpredicted_"+tag+".txt"), g, under, naive)

		// 层级分布与逐块误差统计。
		distPred, distMin := [16]int{}, [16]int{}
		var maxErrRatio float64
		for _, t := range g.Tiles {
			distPred[t.LevelPred]++
			distMin[t.LevelMin]++
			bound := t.Kappa / 8.0
			if !t.Border && bound > 0 {
				if r := t.ErrPred / bound; r > maxErrRatio {
					maxErrRatio = r
				}
			}
		}
		fmt.Fprintf(&summary, "## 允许误差 tol=%g\n\n", tol)
		fmt.Fprintf(&summary, "- 总块数 %d，边界块 %d（直接赋 L%d）\n", len(g.Tiles), countBorder(g), *lmax)
		fmt.Fprintf(&summary, "- 预测层级分布: %s\n", distString(distPred[:], *lmax))
		fmt.Fprintf(&summary, "- 最低可行分布: %s\n", distString(distMin[:], *lmax))
		fmt.Fprintf(&summary, "- 多尺度公式偏低块: %d（判定：实际误差 > tol）\n", len(under))
		fmt.Fprintf(&summary, "- 朴素单尺度偏低块: %d（对照，暴露高频失效）\n", len(naive))
		fmt.Fprintf(&summary, "- 内部块 L=0 实际误差/曲率界 最大比值: %.4f（<=1 即界成立）\n\n", maxErrRatio)

		fmt.Printf("tol=%g: 多尺度偏低=%d 朴素偏低=%d 分布pred=%s\n",
			tol, len(under), len(naive), distString(distPred[:], *lmax))
	}

	writeBytes(filepath.Join(*outDir, "legend.svg"), []byte(lod.LegendSVG(*lmax)))
	writeBytes(filepath.Join(*outDir, "summary.txt"), []byte(summary.String()))
	fmt.Println("产物目录:", *outDir)
}

func writeMatrix(path, title string, m [][]int) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "# %s\n# 行=块行 列=块列，层级 0=最粗 ... 最大=最细\n", title)
	for r, row := range m {
		fmt.Fprintf(&sb, "r%02d:", r)
		for _, v := range row {
			fmt.Fprintf(&sb, " %d", v)
		}
		sb.WriteByte('\n')
	}
	writeBytes(path, []byte(sb.String()))
}

func writeUnder(path string, g *lod.Grid, under, naive []*lod.Tile) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "# 偏低块清单（判定：该层级实际最大误差 > 允许误差 %.4f）\n", g.Tol)
	fmt.Fprintf(&sb, "# 坐标为 (块行,块列)，左上角为 (0,0)\n")
	fmt.Fprintf(&sb, "## 多尺度曲率公式（严格）偏低块: %d 块\n", len(under))
	if len(under) == 0 {
		sb.WriteString("# 无：所有块预测层级的实际误差均不超过允许误差\n")
	}
	for _, t := range under {
		fmt.Fprintf(&sb, "(%d,%d) predL=%d minL=%d errPred=%.6e\n", t.Row, t.Col, t.LevelPred, t.LevelMin, t.ErrPred)
	}
	fmt.Fprintf(&sb, "## 朴素单尺度（仅 L=0 曲率）偏低块: %d 块\n", len(naive))
	for _, t := range naive {
		fmt.Fprintf(&sb, "(%d,%d) naiveL=%d minL=%d kappa0=%.4e kappa=%.4e\n",
			t.Row, t.Col, t.LevelNaive, t.LevelMin, t.Kappa0, t.Kappa)
	}
	writeBytes(path, []byte(sb.String()))
}

func countBorder(g *lod.Grid) int {
	n := 0
	for _, t := range g.Tiles {
		if t.Border {
			n++
		}
	}
	return n
}

func distString(d []int, lmax int) string {
	parts := make([]string, 0, lmax+1)
	for l := 0; l <= lmax; l++ {
		parts = append(parts, fmt.Sprintf("L%d:%d", l, d[l]))
	}
	return strings.Join(parts, " ")
}

func writeBytes(path string, b []byte) {
	if err := os.WriteFile(path, b, 0o644); err != nil {
		fatal("write %s: %v", path, err)
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "lodstudy: "+format+"\n", args...)
	os.Exit(1)
}
