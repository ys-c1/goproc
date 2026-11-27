// bevel 是陡崖倒角双约束实验：在合成的一维陡崖剖面上，
// 并排给出几何约束 w_g、量化约束 w_q、min 预测与量化后的实测带宽。
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/internal/cliff"
	"goproc/internal/hmap"
)

type Row struct {
	Bits     int     `json:"bits"`
	H        float64 `json:"H"`
	S        float64 `json:"s"`
	BetaDeg  float64 `json:"betaDeg"`
	WG       float64 `json:"wGeom"`
	WQ       float64 `json:"wQuant"`
	WMin     float64 `json:"wMin"`
	Regime   string  `json:"regime"`
	BClamped int     `json:"BClamped"`
	DClamped int     `json:"DClamped"`
	BNatural int     `json:"BNatural"`
	DNatural int     `json:"DNatural"`
	NClamped int     `json:"nClamped"`
	NNatural int     `json:"nNatural"`
}

type Crossover struct {
	Bits      int     `json:"bits"`
	Levels    int     `json:"levels"`
	StarS     float64 `json:"starS"`
	BetaDeg   float64 `json:"betaDeg"`
	WStarH1   float64 `json:"wStarH1"`
	WStarHalf float64 `json:"wStarHalf"`
}

func regimeOf(H, s float64, levels int) string {
	if s > cliff.CrossoverSteepness(levels) {
		return "几何主导"
	}
	if s < cliff.CrossoverSteepness(levels) {
		return "量化主导"
	}
	return "交点"
}

func measure(H, s float64, levels int) Row {
	// 裁剪倒角：宽度取 min(w_g, w_q)。
	cr := cliff.ClampedRamp(H, s, levels)
	cLine := cliff.Profile1D(H, cr, 8)
	_, _, bC, dC := cliff.BandStats(cliff.QuantizeLine(cLine, levels))

	// 自然线性倒角：只按几何陡度铺 w_g 个像素，不做裁剪。
	nat := int(float64(math.Round(cliff.GeomLimit(H, s))))
	nr := cliff.UnclampedRamp(H, nat)
	nLine := cliff.Profile1D(H, nr, 8)
	_, _, bN, dN := cliff.BandStats(cliff.QuantizeLine(nLine, levels))

	return Row{
		Bits:     int(float64(math.Log2(float64(levels)))),
		H:        H,
		S:        s,
		BetaDeg:  math.Atan(s) * 180 / math.Pi,
		WG:       cliff.GeomLimit(H, s),
		WQ:       cliff.QuantLimit(H, levels),
		WMin:     cliff.UsedWidth(H, s, levels),
		Regime:   regimeOf(H, s, levels),
		BClamped: bC,
		DClamped: dC,
		BNatural: bN,
		DNatural: dN,
		NClamped: len(cr),
		NNatural: nat,
	}
}

func writePGM(path string, line []float64, bits int) error {
	size, grid := cliff.Extrude(line)
	return os.WriteFile(path, hmap.EncodePGMBits(size, grid, bits), 0o644)
}

func main() {
	dir := "artifacts"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		panic(err)
	}

	type sweep struct {
		bits   int
		levels int
		ss     []float64
	}
	sweeps := []sweep{
		{8, 256, []float64{0.1, 0.01, 0.005, 0.00392156862745098, 0.003, 0.001}},
		{16, 65536, []float64{1e-3, 1e-4, 2e-5, 1.5259021896696422e-5, 1e-5, 5e-6}},
	}
	const H1, Hhalf = 1.0, 0.5

	var rows []Row
	for _, sw := range sweeps {
		for _, s := range sw.ss {
			rows = append(rows, measure(H1, s, sw.levels))
		}
		// 半落差只在代表陡度上取样，验证 H 的作用。
		rows = append(rows, measure(Hhalf, sw.ss[1], sw.levels))
		rows = append(rows, measure(Hhalf, sw.ss[len(sw.ss)-1], sw.levels))
	}

	var crosses []Crossover
	for _, sw := range sweeps {
		sStar := cliff.CrossoverSteepness(sw.levels)
		crosses = append(crosses, Crossover{
			Bits:      sw.bits,
			Levels:    sw.levels,
			StarS:     sStar,
			BetaDeg:   math.Atan(sStar) * 180 / math.Pi,
			WStarH1:   cliff.CrossoverWidth(1.0, sw.levels),
			WStarHalf: cliff.CrossoverWidth(0.5, sw.levels),
		})
	}

	tsv, err := os.Create(filepath.Join(dir, "bevel_rows.tsv"))
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(tsv, "bits\tH\ts\tbeta_deg\tw_geom\tw_quant\tw_min\tregime\tB_clamped\tD_clamped\tn_clamped\tB_natural\tD_natural\tn_natural")
	for _, r := range rows {
		fmt.Fprintf(tsv, "%d\t%g\t%g\t%.6f\t%.4f\t%.4f\t%.4f\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n",
			r.Bits, r.H, r.S, r.BetaDeg, r.WG, r.WQ, r.WMin, r.Regime,
			r.BClamped, r.DClamped, r.NClamped, r.BNatural, r.DNatural, r.NNatural)
	}
	tsv.Close()

	ct, err := os.Create(filepath.Join(dir, "bevel_crossover.tsv"))
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(ct, "bits\tlevels\ts_star\tbeta_deg\tw_star_H1\tw_star_H0.5")
	for _, c := range crosses {
		fmt.Fprintf(ct, "%d\t%d\t%.15g\t%.9f\t%.1f\t%.1f\n",
			c.Bits, c.Levels, c.StarS, c.BetaDeg, c.WStarH1, c.WStarHalf)
	}
	ct.Close()

	jb, err := json.MarshalIndent(struct {
		Crossover []Crossover `json:"crossover"`
		Rows      []Row       `json:"rows"`
	}{crosses, rows}, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bevel_results.json"), jb, 0o644); err != nil {
		panic(err)
	}

	// 演示 PGM：八位几何主导与量化主导（裁剪/自然各一），以及十六位量化主导（小落差）。
	demos := []struct {
		name   string
		bits   int
		levels int
		line   []float64
	}{
		{"bevel_8bit_geom.pgm", 8, 256, cliff.Profile1D(1.0, cliff.ClampedRamp(1.0, 0.01, 256), 16)},
		{"bevel_8bit_quant_clamped.pgm", 8, 256, cliff.Profile1D(1.0, cliff.ClampedRamp(1.0, 0.001, 256), 16)},
		{"bevel_8bit_quant_natural.pgm", 8, 256, cliff.Profile1D(1.0, cliff.UnclampedRamp(1.0, 1000), 16)},
		{"bevel_16bit_quant.pgm", 16, 65536, cliff.Profile1D(0.01, cliff.ClampedRamp(0.01, 1e-6, 65536), 16)},
	}
	for _, d := range demos {
		if err := writePGM(filepath.Join(dir, d.name), d.line, d.bits); err != nil {
			panic(err)
		}
	}
	fmt.Printf("wrote %d rows, %d crossover points and %d PGMs to %s\n", len(rows), len(crosses), len(demos), dir)
}
