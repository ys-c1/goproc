// Command zeroreport 导出水平/竖直剖面的逐层符号翻转计数与绘图数据。
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/noise"
)

type Crossing struct {
	Edge  int     `json:"edge"`
	XZero float64 `json:"x_zero_linear"`
	V0    float64 `json:"v0"`
	V1    float64 `json:"v1"`
}

type OctaveStep struct {
	Octave       int     `json:"octave"`
	Frequency    float64 `json:"frequency_grid_cycles_per_cell_equiv"`
	Weight       float64 `json:"normalized_weight"`
	Crossings    int     `json:"crossings_after_adding"`
	AddedEdges   []int   `json:"added_edges"`
	RemovedEdges []int   `json:"removed_edges"`
	NetExtra     int     `json:"net_extra"`
}

type Profile struct {
	ID                 string       `json:"id"`
	Orientation        string       `json:"orientation"`
	FixedAxis          string       `json:"fixed_axis"`
	FixedIndex         int          `json:"fixed_index"`
	SampleAxis         string       `json:"sample_axis"`
	N                  int          `json:"n_samples"`
	SpacingCells       float64      `json:"spacing_cells"`
	Layers             [][]float64  `json:"layers"`
	Low                []float64    `json:"low"`
	Total              []float64    `json:"total_raw_fbm"`
	NormalizedMidpoint []float64    `json:"normalized_minmax_centered_0_5"`
	Measured           int          `json:"measured_sign_flips"`
	ZeroSamples        int          `json:"exact_zero_samples"`
	Crossings          []Crossing   `json:"crossings"`
	LowCrossings       int          `json:"low_crossings"`
	LowEdges           []int        `json:"low_edges"`
	Ratio              float64      `json:"adjacent_frequency_ratio"`
	TotalRatio         float64      `json:"highest_to_lowest_ratio"`
	RatioPrediction    int          `json:"ratio_only_prediction"`
	RatioOvercount     int          `json:"ratio_overcount"`
	RatioUndercount    int          `json:"ratio_undercount"`
	ExactPrediction    int          `json:"exact_increment_prediction"`
	ExactEdgeFalse     int          `json:"exact_edge_false_alarms"`
	ExactEdgeMiss      int          `json:"exact_edge_misses"`
	Steps              []OctaveStep `json:"steps"`
}

type CounterExample struct {
	Description  string `json:"description"`
	Orientation  string `json:"orientation"`
	FixedIndex   int    `json:"fixed_index"`
	LowCrossings int    `json:"low_crossings"`
	Measured     int    `json:"measured_sign_flips"`
}

type Report struct {
	Parameters      api.Options      `json:"parameters"`
	ZeroLevel       string           `json:"zero_level"`
	SignRule        string           `json:"sign_rule"`
	Profiles        []Profile        `json:"profiles"`
	CounterExamples []CounterExample `json:"counterexamples_same_seed_and_ratios"`
}

func crossingEdges(v []float64) ([]Crossing, int) {
	var out []Crossing
	zeros := 0
	for _, x := range v {
		if x == 0 {
			zeros++
		}
	}
	for i := 0; i+1 < len(v); i++ {
		a, b := v[i], v[i+1]
		if (a < 0 && b > 0) || (a > 0 && b < 0) {
			out = append(out, Crossing{
				Edge:  i,
				XZero: float64(i) + a/(a-b),
				V0:    a,
				V1:    b,
			})
		}
	}
	return out, zeros
}

func edgeSet(cs []Crossing) map[int]bool {
	m := make(map[int]bool, len(cs))
	for _, c := range cs {
		m[c.Edge] = true
	}
	return m
}

func sortedKeys(m map[int]bool) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func diffSets(a, b map[int]bool) []int {
	var out []int
	for k := range a {
		if !b[k] {
			out = append(out, k)
		}
	}
	sort.Ints(out)
	return out
}

func buildProfile(id, orientation string, fixed int, opt api.Options, hm *api.Heightmap, src *noise.Source) Profile {
	n := opt.Size
	p := Profile{
		ID: id, Orientation: orientation, FixedIndex: fixed, N: n, SpacingCells: 1,
		Ratio:      opt.Lacunarity,
		TotalRatio: math.Pow(opt.Lacunarity, float64(opt.Octaves-1)),
	}
	if orientation == "horizontal" {
		p.FixedAxis = "y"
		p.SampleAxis = "x"
	} else {
		p.FixedAxis = "x"
		p.SampleAxis = "y"
	}

	p.Layers = make([][]float64, opt.Octaves)
	for j := range p.Layers {
		p.Layers[j] = make([]float64, n)
	}
	p.Low = make([]float64, n)
	p.Total = make([]float64, n)
	p.NormalizedMidpoint = make([]float64, n)

	for t := 0; t < n; t++ {
		var x, y int
		if orientation == "horizontal" {
			x, y = t, fixed
			p.NormalizedMidpoint[t] = hm.At(x, y) - 0.5
		} else {
			x, y = fixed, t
			p.NormalizedMidpoint[t] = hm.At(x, y) - 0.5
		}
		layers := fbm.EvalLayers(src, float64(x)*opt.Frequency, float64(y)*opt.Frequency,
			opt.Octaves, opt.Lacunarity, opt.Gain)
		for j, v := range layers {
			p.Layers[j][t] = v
			p.Total[t] += v
		}
		p.Low[t] = p.Layers[0][t]
	}

	cs, zeros := crossingEdges(p.Total)
	p.Crossings = cs
	p.Measured = len(cs)
	p.ZeroSamples = zeros
	lowCS, _ := crossingEdges(p.Low)
	p.LowCrossings = len(lowCS)
	p.LowEdges = make([]int, len(lowCS))
	for i, c := range lowCS {
		p.LowEdges[i] = c.Edge
	}

	p.RatioPrediction = int(math.Round(float64(p.LowCrossings) * p.TotalRatio))
	p.RatioOvercount = max(p.RatioPrediction-p.Measured, 0)
	p.RatioUndercount = max(p.Measured-p.RatioPrediction, 0)

	// Exact incremental sign-set identity.
	current := make([]float64, n)
	predSet := map[int]bool{}
	prevSet := map[int]bool{}
	for j := 0; j < opt.Octaves; j++ {
		for t := range current {
			current[t] += p.Layers[j][t]
		}
		cs, _ := crossingEdges(current)
		nowSet := edgeSet(cs)
		if j == 0 {
			p.Steps = append(p.Steps, OctaveStep{
				Octave: j, Frequency: opt.Frequency * math.Pow(opt.Lacunarity, float64(j)),
				Weight:    math.Pow(opt.Gain, float64(j)) / normalizedWeightSum(opt),
				Crossings: len(cs), AddedEdges: sortedKeys(nowSet), RemovedEdges: []int{},
				NetExtra: len(cs),
			})
			predSet = nowSet
		} else {
			added := diffSets(nowSet, prevSet)
			removed := diffSets(prevSet, nowSet)
			step := OctaveStep{
				Octave: j, Frequency: opt.Frequency * math.Pow(opt.Lacunarity, float64(j)),
				Weight:    math.Pow(opt.Gain, float64(j)) / normalizedWeightSum(opt),
				Crossings: len(cs), AddedEdges: added, RemovedEdges: removed,
				NetExtra: len(added) - len(removed),
			}
			p.Steps = append(p.Steps, step)
			for _, e := range added {
				predSet[e] = true
			}
			for _, e := range removed {
				delete(predSet, e)
			}
		}
		prevSet = nowSet
	}
	p.ExactPrediction = p.LowCrossings
	for j := 1; j < opt.Octaves; j++ {
		p.ExactPrediction += p.Steps[j].NetExtra
	}
	actualSet := edgeSet(p.Crossings)
	p.ExactEdgeFalse = len(diffSets(predSet, actualSet))
	p.ExactEdgeMiss = len(diffSets(actualSet, predSet))
	if p.ExactPrediction != p.Measured || p.ExactEdgeFalse+p.ExactEdgeMiss != 0 {
		panic(fmt.Sprintf("exact crossing identity failed for %s: pred=%d actual=%d false=%d miss=%d",
			id, p.ExactPrediction, p.Measured, p.ExactEdgeFalse, p.ExactEdgeMiss))
	}
	return p
}

func normalizedWeightSum(opt api.Options) float64 {
	s, a := 0.0, 1.0
	for i := 0; i < opt.Octaves; i++ {
		s += a
		a *= opt.Gain
	}
	return s
}

func summaryProfile(orientation string, fixed int, opt api.Options, src *noise.Source) CounterExample {
	n := opt.Size
	var low, total []float64
	for t := 0; t < n; t++ {
		var x, y int
		if orientation == "horizontal" {
			x, y = t, fixed
		} else {
			x, y = fixed, t
		}
		layers := fbm.EvalLayers(src, float64(x)*opt.Frequency, float64(y)*opt.Frequency,
			opt.Octaves, opt.Lacunarity, opt.Gain)
		if low == nil {
			low = make([]float64, n)
			total = make([]float64, n)
		}
		low[t] = layers[0]
		for _, v := range layers {
			total[t] += v
		}
	}
	lc, _ := crossingEdges(low)
	tc, _ := crossingEdges(total)
	return CounterExample{
		Description: "同一 seed、N、频率比与基础过零数，仅剖面位置不同，合成过零数不同",
		Orientation: orientation, FixedIndex: fixed,
		LowCrossings: len(lc), Measured: len(tc),
	}
}

func writeCSV(path string, r Report) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write([]string{
		"profile", "orientation", "fixed", "n", "spacing_cells", "z_low",
		"frequency_ratio", "total_ratio", "ratio_pred", "measured",
		"overcount", "undercount", "exact_pred", "exact_false_edges", "exact_miss_edges",
		"crossing_edges",
	}); err != nil {
		return err
	}
	for _, p := range r.Profiles {
		edges := ""
		for i, e := range p.Crossings {
			if i > 0 {
				edges += ";"
			}
			edges += fmt.Sprintf("%d", e.Edge)
		}
		if err := w.Write([]string{
			p.ID, p.Orientation, fmt.Sprint(p.FixedIndex), fmt.Sprint(p.N), "1",
			fmt.Sprint(p.LowCrossings), fmt.Sprint(p.Ratio), fmt.Sprint(p.TotalRatio),
			fmt.Sprint(p.RatioPrediction), fmt.Sprint(p.Measured),
			fmt.Sprint(p.RatioOvercount), fmt.Sprint(p.RatioUndercount),
			fmt.Sprint(p.ExactPrediction), fmt.Sprint(p.ExactEdgeFalse),
			fmt.Sprint(p.ExactEdgeMiss), edges,
		}); err != nil {
			return err
		}
	}
	return w.Error()
}

func main() {
	size := flag.Int("size", api.DefaultOptions().Size, "生成分辨率边长")
	seed := flag.Uint64("seed", api.DefaultOptions().Seed, "种子")
	octaves := flag.Int("octaves", api.DefaultOptions().Octaves, "层数")
	lac := flag.Float64("lacunarity", api.DefaultOptions().Lacunarity, "频率倍增")
	gain := flag.Float64("gain", api.DefaultOptions().Gain, "振幅衰减")
	freq := flag.Float64("frequency", api.DefaultOptions().Frequency, "基础频率")
	fixed := flag.Int("fixed", -1, "水平剖面的 y / 竖直剖面的 x；默认 size/2")
	out := flag.String("out", "artifacts", "输出目录")
	flag.Parse()

	opt := api.Options{Size: *size, Octaves: *octaves, Lacunarity: *lac, Gain: *gain, Seed: *seed, Frequency: *freq}
	if *fixed < 0 {
		*fixed = opt.Size / 2
	}
	hm, err := api.Generate(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	src := noise.New(opt.Seed)
	r := Report{
		Parameters: opt,
		ZeroLevel:  "fbm.Eval 的未归一化有符号零电平；api 归一化只用于补充的 0.5 中值口径",
		SignRule:   "仅当相邻一格样本严格异号时计一次；乘积 < 0，精确零样本单列",
		Profiles: []Profile{
			buildProfile("horizontal-center", "horizontal", *fixed, opt, hm, src),
			buildProfile("vertical-center", "vertical", *fixed, opt, hm, src),
		},
		CounterExamples: []CounterExample{
			summaryProfile("horizontal", *fixed, opt, src),
			summaryProfile("horizontal", 0, opt, src),
		},
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}
	jb, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(*out, "zero_profile_data.json"), jb, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write json failed:", err)
		os.Exit(1)
	}
	if err := writeCSV(filepath.Join(*out, "zero_profile_table.csv"), r); err != nil {
		fmt.Fprintln(os.Stderr, "write csv failed:", err)
		os.Exit(1)
	}
	for _, p := range r.Profiles {
		fmt.Printf("%s fixed=%d z0=%d ratio_pred=%d measured=%d over=%d under=%d exact=%d\n",
			p.Orientation, p.FixedIndex, p.LowCrossings, p.RatioPrediction, p.Measured,
			p.RatioOvercount, p.RatioUndercount, p.ExactPrediction)
	}
}
