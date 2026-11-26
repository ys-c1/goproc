// seedexp 量化种子在置换表与高度图两个层面的扩散程度, 输出 JSON 供绘图与存档。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

// GapResult 是一档种子间隔的汇总。
type GapResult struct {
	Gap              int     `json:"gap"`
	ChangedEntries   float64 `json:"changedEntries"`   // 平均改动条目数(满分 256)
	ChangedFraction  float64 `json:"changedFraction"`  // 平均改动条目占比
	CorrelatedArea   float64 `json:"correlatedArea"`   // 平均相关面积(零到一)
	Pearson          float64 `json:"pearson"`          // 平均皮尔逊相关系数
}

// SchemeResult 是一种派生方案在三档间隔下的结果。
type SchemeResult struct {
	Scheme string      `json:"scheme"`
	Q      float64     `json:"q,omitempty"`
	Gaps   []GapResult `json:"gaps"`
}

// Report 是完整实验报告。
type Report struct {
	Config map[string]any `json:"config"`
	// 独立判据: 两张互不相关置换表渲染出的高度图基准。
	Independence struct {
		ChangedEntries   float64 `json:"changedEntries"`
		ChangedFraction  float64 `json:"changedFraction"`
		CorrelatedArea   float64 `json:"correlatedArea"`
		Pearson          float64 `json:"pearson"`
	} `json:"independence"`
	// 扩散充分条件: q > 1 - 1/256。
	DiffusionThreshold float64        `json:"diffusionThresholdQ"`
	Weak               SchemeResult   `json:"weak"`
	Hash               []SchemeResult `json:"hash"`
	Production         SchemeResult   `json:"production"`
}

type averages struct {
	entries int
	area    float64
	pearson float64
}

// evaluatePair 对一组置换表对求三项指标的累计。
func evaluatePair(p1, p2 [256]uint8, t1, t2 []float64, acc *averages) {
	acc.entries += hammingDistance(p1, p2)
	acc.area += correlatedArea(t1, t2, tau)
	acc.pearson += pearson(t1, t2)
}

func finalize(gap int, acc averages) GapResult {
	n := float64(trials)
	e := float64(acc.entries) / n
	return GapResult{
		Gap:             gap,
		ChangedEntries:  e,
		ChangedFraction: e / 256,
		CorrelatedArea:  acc.area / n,
		Pearson:         acc.pearson / n,
	}
}

func runWeak() SchemeResult {
	res := SchemeResult{Scheme: "weak"}
	for _, g := range gaps {
		k := weakK[g]
		var acc averages
		for s := 0; s < trials; s++ {
			p1 := weakPerm(uint64(s), k)
			p2 := weakPerm(uint64(s+g), k)
			evaluatePair(p1, p2, terrain(p1, mapSize, frequency), terrain(p2, mapSize, frequency), &acc)
		}
		res.Gaps = append(res.Gaps, finalize(g, acc))
	}
	return res
}

func runHash(q float64) SchemeResult {
	res := SchemeResult{Scheme: "hash", Q: q}
	for _, g := range gaps {
		var acc averages
		for s := 0; s < trials; s++ {
			p1 := hashPerm(uint64(s), q)
			p2 := hashPerm(uint64(s+g), q)
			evaluatePair(p1, p2, terrain(p1, mapSize, frequency), terrain(p2, mapSize, frequency), &acc)
		}
		res.Gaps = append(res.Gaps, finalize(g, acc))
	}
	return res
}

func runProduction() SchemeResult {
	res := SchemeResult{Scheme: "production"}
	for _, g := range gaps {
		var acc averages
		for s := 0; s < trials; s++ {
			p1 := productionPerm(uint64(s))
			p2 := productionPerm(uint64(s + g))
			evaluatePair(p1, p2, terrain(p1, mapSize, frequency), terrain(p2, mapSize, frequency), &acc)
		}
		res.Gaps = append(res.Gaps, finalize(g, acc))
	}
	return res
}

// runIndependence 用互不相关的散列种子估计独立判据。
func runIndependence() (float64, float64, float64, float64) {
	var acc averages
	for m := 0; m < trials; m++ {
		p1 := hashPerm(uint64(m)*7919+1, 1.0)
		p2 := hashPerm(uint64(m)*104729+3, 1.0)
		evaluatePair(p1, p2, terrain(p1, mapSize, frequency), terrain(p2, mapSize, frequency), &acc)
	}
	n := float64(trials)
	e := float64(acc.entries) / n
	return e, e / 256, acc.area / n, acc.pearson / n
}

func main() {
	out := flag.String("out", "", "JSON 输出路径(缺省输出到标准输出)")
	flag.Parse()

	report := Report{
		Config: map[string]any{
			"gaps":      gaps,
			"weakK":     weakK,
			"qLevels":   qLevels,
			"mapSize":   mapSize,
			"frequency": frequency,
			"tau":       tau,
			"trials":    trials,
		},
		DiffusionThreshold: 1 - 1.0/256.0,
	}
	e, frac, area, r := runIndependence()
	report.Independence.ChangedEntries = e
	report.Independence.ChangedFraction = frac
	report.Independence.CorrelatedArea = area
	report.Independence.Pearson = r

	report.Weak = runWeak()
	for _, q := range qLevels {
		report.Hash = append(report.Hash, runHash(q))
	}
	report.Production = runProduction()

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "marshal failed:", err)
		os.Exit(1)
	}
	if *out == "" {
		fmt.Println(string(data))
		return
	}
	if err := os.WriteFile(*out, append(data, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Println("wrote", *out)
}
