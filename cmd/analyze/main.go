// Command analyze 对照两种水深→色阶映射，并由左右两块主方向角反解旋转角。
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/bathy"
	"goproc/internal/orient"
)

const (
	size     = 128
	seed     = 7
	octaves  = 4
	Q        = 256 // 水深量化档数，对应八位 PGM 的 256 级
	seaLevel = 0.5
)

func main() {
	outDir := "artifacts"
	if len(os.Args) > 1 {
		outDir = os.Args[1]
	}
	if err := run(outDir); err != nil {
		fmt.Fprintln(os.Stderr, "analyze failed:", err)
		os.Exit(1)
	}
}

func run(outDir string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	h, err := api.Generate(api.Options{Size: size, Octaves: octaves, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02})
	if err != nil {
		return err
	}
	depth, mask := bathy.Depth(h.Data, seaLevel)
	qbins := bathy.Quantize(depth, Q)

	var b strings.Builder
	fmt.Fprintf(&b, "地形 size=%d seed=%d octaves=%d；水位 seaLevel=%.2f；水深量化档数 Q=%d\n\n",
		size, seed, octaves, seaLevel, Q)

	water := 0
	for _, m := range mask {
		if m {
			water++
		}
	}
	fmt.Fprintf(&b, "水下属点 %d / %d\n\n", water, size*size)

	fmt.Fprintln(&b, "== 两组色阶数下的对照（色阶差按档数计，水深按量化后的档位算）==")
	reports := make([]bathy.LevelReport, 0, 2)
	for _, L := range []int{16, 10} {
		r := bathy.Compare(size, depth, mask, qbins, Q, L)
		reports = append(reports, r)
		fmt.Fprintf(&b, "L=%-2d  Q mod L=%-3d  差异像素=%-4d  最大色阶差=%d  失配测度=%.6f\n",
			r.L, r.Remainder, r.DiffPixels, r.MaxAbsDelta, bathy.MismatchMeasure(Q, L))
	}
	fmt.Fprintln(&b)

	for _, r := range reports {
		fmt.Fprintf(&b, "-- L=%d 各档像素数（A=连续直映, B=先量化再映色）\n", r.L)
		fmt.Fprintln(&b, "level\tA\tB\tB-A")
		for k := 0; k < r.L; k++ {
			fmt.Fprintf(&b, "%d\t%d\t%d\t%d\n", k, r.HistA[k], r.HistB[k], r.HistB[k]-r.HistA[k])
		}
		fmt.Fprintln(&b)
	}

	non := reports[1] // L=10，不整除组
	fmt.Fprintf(&b, "== L=10 不整除：颜色错的区域（B 恒比真实水深浅一档）==\n")
	if non.BBox != nil {
		fmt.Fprintf(&b, "错色区域包围盒（像素坐标，左上原点，闭区间）: x∈[%d,%d], y∈[%d,%d]\n",
			non.BBox.X0, non.BBox.X1, non.BBox.Y0, non.BBox.Y1)
	}
	fmt.Fprintln(&b, "逐点清单（x,y, 水深量化档q, A档, B档, 色阶差B-A）:")
	for _, p := range non.Mismatch {
		fmt.Fprintf(&b, "(%d,%d)\tq=%d\tA=%d\tB=%d\tdelta=%d\n", p.X, p.Y, p.QBin, p.LevelA, p.LevelB, p.Delta())
	}

	// 旋转角：左右两块的结构张量主方向
	left, right := orient.SplitVertical(size)
	tL := orient.GradientTensor(h.Data, size, left[0], left[1], left[2], left[3])
	tR := orient.GradientTensor(h.Data, size, right[0], right[1], right[2], right[3])
	gL, gR := tL.GradientAngle(), tR.GradientAngle()
	rL, rR := tL.RidgeAngle(), tR.RidgeAngle()
	rotG := orient.Rotation(gL, gR)
	rotR := orient.Rotation(rL, rR)
	fmt.Fprintf(&b, "\n== 旋转角（结构张量，角度制；0°=东，顺时针为正，卷绕到 (-90°,90°]）==\n")
	fmt.Fprintf(&b, "左块 x∈[%d,%d]  右块 x∈[%d,%d]\n", left[0], left[2], right[0], right[2])
	fmt.Fprintf(&b, "梯度主方向(法向): θL=%.4f° θR=%.4f°  旋转角=%.4f°\n", gL, gR, rotG)
	fmt.Fprintf(&b, "山脊/沟谷走向  : θL=%.4f° θR=%.4f°  旋转角=%.4f°（与法向结果同值，轴线整体旋转）\n", rL, rR, rotR)

	text := b.String()
	fmt.Print(text)

	if err := os.WriteFile(filepath.Join(outDir, "mismatch_L10.csv"),
		[]byte(csv(non)), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outDir, "report.txt"), []byte(text), 0o644)
}

func csv(r bathy.LevelReport) string {
	var b strings.Builder
	b.WriteString("x,y,qbin,levelA,levelB,deltaB-A\n")
	for _, p := range r.Mismatch {
		fmt.Fprintf(&b, "%d,%d,%d,%d,%d,%d\n", p.X, p.Y, p.QBin, p.LevelA, p.LevelB, p.Delta())
	}
	return b.String()
}
