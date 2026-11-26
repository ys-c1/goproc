package erosion

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"goproc/internal/hmap"
)

// Snapshot 是某个迭代步数下一条轨迹的全部存档材料。
type Snapshot struct {
	Steps   int
	Field   []float64 // 侵蚀后高度场
	Delta   []float64 // 相对初始场的逐格变化量（新 - 旧）
	Height  []byte    // 高度图 PGM，与初始场共用灰度基准
	Erosion []byte    // 侵蚀量图 PGM，零变化对应中灰
	Result  VariantResult
}

// VariantReport 是一条轨迹在所有目标步数下的结果。
type VariantReport struct {
	Name      string          `json:"name"`
	Quantized bool            `json:"quantized"`
	Results   []VariantResult `json:"results"`
}

// Report 是一次完整守恒实验的机器可读记录。
type Report struct {
	Size            int           `json:"size"`
	CellSize        float64       `json:"cellSize"`
	K               float64       `json:"k"`
	Quantum         float64       `json:"quantum"`
	Threshold       float64       `json:"threshold"`
	InitialTotal    float64       `json:"initialTotal"`
	Steps           []int         `json:"steps"`
	Fixed           VariantReport `json:"fixed"`
	Naive           VariantReport `json:"naive"`
	FixedAbsByStep  []float64     `json:"fixedAbsByStep,omitempty"`
	NaiveLossByStep []float64     `json:"naiveLossByStep,omitempty"`
	AllFixedPass    bool          `json:"allFixedPass"`
	AllNaivePass    bool          `json:"allNaivePass"`
}

// Study 持有两条轨迹在各目标步数的快照。
type Study struct {
	Report   Report
	Fixed    map[int]Snapshot
	Naive    map[int]Snapshot
	lo, hi   float64
	maxDelta float64
}

func extrema(h []float64) (lo, hi float64) {
	lo, hi = h[0], h[0]
	for _, v := range h {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return lo, hi
}

// RunStudy 在同一初始场与同一步数清单上跑守恒版与定点量化版，
// 并为每个目标步数生成高度图、侵蚀量图与核对结果。
func RunStudy(base []float64, size int, stepList []int, c Config) (*Study, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	if len(stepList) == 0 {
		return nil, fmt.Errorf("%w: 步数清单为空", ErrBadConfig)
	}
	targets := append([]int(nil), stepList...)
	sort.Ints(targets)
	for _, s := range targets {
		if s <= 0 {
			return nil, fmt.Errorf("%w: steps=%d 必须为正", ErrBadConfig, s)
		}
	}
	maxSteps := targets[len(targets)-1]

	fixed := cloneField(base)
	naive := cloneField(base)
	initialTotal := Total(base)
	lo, hi := extrema(base)

	st := &Study{
		Fixed: make(map[int]Snapshot, len(targets)),
		Naive: make(map[int]Snapshot, len(targets)),
		lo:    lo,
		hi:    hi,
	}
	fixedStep, fixedAbs := 0, 0.0
	naiveStep, naiveAbs := 0, 0.0
	fixedAbsSeries := make([]float64, 0, maxSteps)
	naiveLossSeries := make([]float64, 0, maxSteps)

	targetSet := make(map[int]bool, len(targets))
	for _, s := range targets {
		targetSet[s] = true
	}

	for k := 1; k <= maxSteps; k++ {
		if err := Step(fixed, size, c); err != nil {
			return nil, err
		}
		lost, err := NaiveStep(naive, size, c)
		if err != nil {
			return nil, err
		}
		fixedAbsSeries = append(fixedAbsSeries, math.Abs(Total(fixed)-initialTotal))
		naiveLossSeries = append(naiveLossSeries, lost)
		if fixedStep == 0 {
			if d := math.Abs(Total(fixed) - initialTotal); d != 0 {
				fixedStep, fixedAbs = k, d
			}
		}
		if naiveStep == 0 {
			if d := math.Abs(Total(naive) - initialTotal); d != 0 {
				naiveStep, naiveAbs = k, d
			}
		}
		if targetSet[k] {
			st.snapshot(base, fixed, size, k, "fixed", initialTotal, fixedStep, fixedAbs)
			st.snapshot(base, naive, size, k, "naive", initialTotal, naiveStep, naiveAbs)
		}
	}

	st.Report = Report{
		Size:            size,
		CellSize:        c.CellSize,
		K:               c.K,
		Quantum:         c.Quantum,
		Threshold:       ConservationThreshold,
		InitialTotal:    initialTotal,
		Steps:           targets,
		FixedAbsByStep:  fixedAbsSeries,
		NaiveLossByStep: naiveLossSeries,
		Fixed:           VariantReport{Name: "fixed", Results: collectResults(st.Fixed, targets)},
		Naive:           VariantReport{Name: "naive", Quantized: true, Results: collectResults(st.Naive, targets)},
	}
	st.finalizeMaps(size)
	st.Report.AllFixedPass = allPass(st.Report.Fixed.Results)
	st.Report.AllNaivePass = allPass(st.Report.Naive.Results)
	return st, nil
}

func collectResults(m map[int]Snapshot, targets []int) []VariantResult {
	out := make([]VariantResult, 0, len(targets))
	for _, s := range targets {
		out = append(out, m[s].Result)
	}
	return out
}

func allPass(rs []VariantResult) bool {
	for _, r := range rs {
		if !r.Pass {
			return false
		}
	}
	return true
}

func (st *Study) snapshot(base, field []float64, size, steps int, variant string,
	initialTotal float64, firstStep int, firstAbs float64) {
	delta := make([]float64, len(field))
	peak := 0.0
	for i := range field {
		delta[i] = field[i] - base[i]
		if a := math.Abs(delta[i]); a > peak {
			peak = a
		}
	}
	if peak > st.maxDelta {
		st.maxDelta = peak
	}
	snap := Snapshot{
		Steps:  steps,
		Field:  cloneField(field),
		Delta:  delta,
		Height: hmap.EncodePGMRange(size, field, st.lo, st.hi),
		Result: newResult(steps, initialTotal, Total(field), firstStep, firstAbs),
	}
	if variant == "fixed" {
		st.Fixed[steps] = snap
	} else {
		st.Naive[steps] = snap
	}
}

// finalizeMaps 在全部步数跑完后用统一的差值峰值渲染侵蚀量图，
// 保证不同步数的侵蚀量图灰度互相可比。
func (st *Study) finalizeMaps(size int) {
	peak := st.maxDelta
	if peak == 0 {
		peak = 1
	}
	for k, snap := range st.Fixed {
		snap.Erosion = hmap.EncodePGMRange(size, snap.Delta, -peak, peak)
		st.Fixed[k] = snap
	}
	for k, snap := range st.Naive {
		snap.Erosion = hmap.EncodePGMRange(size, snap.Delta, -peak, peak)
		st.Naive[k] = snap
	}
}

// Archive 把侵蚀图与 JSON 报告写入 dir（侵蚀图在 RunStudy 结束时
// 已按统一差值峰值渲染完毕）：
//
//	dir/fixed_NNN_height.pgm   守恒版第 NNN 步高度图
//	dir/fixed_NNN_erosion.pgm  守恒版第 NNN 步侵蚀量图
//	dir/naive_NNN_height.pgm   定点版第 NNN 步高度图
//	dir/naive_NNN_erosion.pgm  定点版第 NNN 步侵蚀量图
//	dir/report.json            总高差、首次偏差步等全部数值
func (st *Study) Archive(dir string, size int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	write := func(name string, data []byte) error {
		return os.WriteFile(filepath.Join(dir, name), data, 0o644)
	}
	for _, s := range st.Report.Steps {
		f := st.Fixed[s]
		n := st.Naive[s]
		if err := write(fmt.Sprintf("fixed_%03d_height.pgm", s), f.Height); err != nil {
			return err
		}
		if err := write(fmt.Sprintf("fixed_%03d_erosion.pgm", s), f.Erosion); err != nil {
			return err
		}
		if err := write(fmt.Sprintf("naive_%03d_height.pgm", s), n.Height); err != nil {
			return err
		}
		if err := write(fmt.Sprintf("naive_%03d_erosion.pgm", s), n.Erosion); err != nil {
			return err
		}
	}
	raw, err := json.MarshalIndent(st.Report, "", "  ")
	if err != nil {
		return err
	}
	return write("report.json", append(raw, '\n'))
}
