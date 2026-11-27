// combexp 构造多层叠加梳齿的反例与对照组，输出 PGM、JSON 与文本报告。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/internal/comb"
)

type sweepPoint struct {
	DeltaDeg int `json:"delta_deg"`
	RawRuns  int `json:"raw_runs"`
	Teeth    int `json:"teeth"`
}

type groupJSON struct {
	Name           string    `json:"name"`
	RFactor        float64   `json:"r"`
	DeltaDeg       float64   `json:"delta_deg"`
	RawRuns        int       `json:"raw_runs"`
	Teeth          int       `json:"teeth"`
	Predicted      int       `json:"predicted"`
	Diff           int       `json:"diff"`
	SpacingPred    float64   `json:"spacing_pred_px,omitempty"`
	SpacingMeas    []float64 `json:"spacing_meas_px,omitempty"`
	MinResidualDeg float64   `json:"min_residual_deg,omitempty"`
	ToothCentersPx []float64 `json:"tooth_centers_px"`
}

type report struct {
	Params     comb.Params  `json:"params"`
	GeomEpsDeg float64      `json:"geom_epsilon_deg"`
	PhiTolDeg  float64      `json:"phi_tol_deg"`
	Groups     []groupJSON  `json:"groups"`
	SumAbsDiff int          `json:"sum_abs_diff"`
	PhaseSweep []sweepPoint `json:"phase_sweep_r4"`
}

func main() {
	outDir := flag.String("out", "artifacts/comb", "输出目录")
	flag.Parse()
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail(err)
	}

	p := comb.DefaultParams()
	groups := comb.StandardGroups()
	rep := report{
		Params:     p,
		GeomEpsDeg: p.GeomEpsilon() * 180 / math.Pi,
		PhiTolDeg:  p.PhiTol * 180 / math.Pi,
	}

	fmt.Printf("参数: N=%d f1=%.4f(周期%.0fpx) A=%.2f w=%.2f T=%.2f HMin=%d MergeGap=%d PhiTol=%.2fdeg\n",
		p.N, p.F1, 1/p.F1, p.A, p.W, p.T, p.HMin, p.MergeGap, rep.PhiTolDeg)
	fmt.Printf("几何相位容差 eps_geom=2*acos((T+w)/(2A))=%.2fdeg；分类容差 PhiTol=%.2fdeg\n\n",
		rep.GeomEpsDeg, rep.PhiTolDeg)

	for i, g := range groups {
		f := comb.Build(p, g.R, g.Delta)
		res := comb.RunGroup(p, g)

		pgmPath := filepath.Join(*outDir, fmt.Sprintf("group%d.pgm", i))
		if err := os.WriteFile(pgmPath, f.EncodePGM(), 0o644); err != nil {
			fail(err)
		}

		gj := groupJSON{
			Name: g.Name, RFactor: g.R, DeltaDeg: g.Delta * 180 / math.Pi,
			RawRuns: res.RawRuns, Teeth: len(res.Teeth), Predicted: res.Predicted,
			Diff: res.Diff(), SpacingPred: res.SpacingPred, SpacingMeas: res.SpacingMeas,
			MinResidualDeg: res.MinResidualDeg,
		}
		for _, t := range res.Teeth {
			gj.ToothCentersPx = append(gj.ToothCentersPx, t.Center)
		}
		rep.Groups = append(rep.Groups, gj)
		rep.SumAbsDiff += res.Diff()

		fmt.Println(res.String())
		if len(res.Teeth) > 0 {
			fmt.Printf("  齿中心: %v\n", gj.ToothCentersPx)
		}
		if len(res.SpacingMeas) > 0 {
			fmt.Printf("  实测间距: %v  理论间距: %.1fpx\n", res.SpacingMeas, res.SpacingPred)
		}
		if !comb.IsRational(g.R) {
			fmt.Printf("  c>=1 最小相位残差: %.2fdeg > PhiTol=%.2fdeg，%d 个几何亮段全部被相位判据拒绝\n",
				res.MinResidualDeg, rep.PhiTolDeg, res.RawRuns)
		}
		fmt.Printf("  -> %s\n\n", pgmPath)
	}

	// 相位扫描：r=4 固定，δ 从 0 到 360 度全程 1 度步进，
	// 既能定位 δ≈0 主族，也能捕捉 δ≈67.5°、292.5° 的空间补偿对齐族。
	degs := make([]int, 0, 361)
	for d := 0; d <= 360; d++ {
		degs = append(degs, d)
	}
	fmt.Println("相位扫描 r=4/1：")
	for _, d := range degs {
		f := comb.Build(p, 4.0, float64(d)*math.Pi/180)
		raw, teeth := f.Detect()
		rep.PhaseSweep = append(rep.PhaseSweep, sweepPoint{d, raw, len(teeth)})
		fmt.Printf("  delta=%3ddeg raw=%3d teeth=%2d\n", d, raw, len(teeth))
	}

	jb, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fail(err)
	}
	jsonPath := filepath.Join(*outDir, "report.json")
	if err := os.WriteFile(jsonPath, jb, 0o644); err != nil {
		fail(err)
	}
	fmt.Printf("\n逐图差值绝对值之和 S=Σ|diff|=%d，JSON: %s\n", rep.SumAbsDiff, jsonPath)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "combexp failed:", err)
	os.Exit(1)
}
