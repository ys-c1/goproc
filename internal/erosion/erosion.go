// Package erosion 实现给定坡度与汇流量后的闭式降雨侵蚀量。
//
// 本包的等价性证明针对“同一份坡度/汇流场”：这正是题目中“给定坡度与汇流量”
// 的含义。若每轮侵蚀后重新计算坡度或汇流，输入场已经改变，N*K*Q*S 不再是
// 该动态过程的总闭式解。
package erosion

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Fields 是侵蚀公式使用的给定地形水文场。
type Fields struct {
	Size                int       // 边长
	Receiver            []int     // D8 最陡下坡邻格；无下游时为 -1
	Slope               []float64 // 沿 Receiver 方向的无量纲坡度 S
	Flow                []float64 // 汇流量 Q，这里用上游格数（含本格）表示
	Distance            []float64 // 到 Receiver 的实际距离（像元单位）
	MaxFlowSlopeProduct float64
}

var (
	ErrBadSize   = errors.New("网格尺寸非法")
	ErrBadLength = errors.New("数据长度与尺寸不符")
	ErrBadValue  = errors.New("高度含 NaN/Inf")
	ErrBadCoeff  = errors.New("侵蚀系数非法")
	ErrBadRounds = errors.New("迭代轮数非法")
)

// BuildFields 从高度场构造确定性的 D8 坡度、流向和汇流量。
// cellSize 是一个像元对应的水平距离；坡度按高差/实际水平距离计算。
func BuildFields(size int, height []float64, cellSize float64) (*Fields, error) {
	if size <= 0 {
		return nil, fmt.Errorf("%w: %d", ErrBadSize, size)
	}
	n := size * size
	if len(height) != n {
		return nil, fmt.Errorf("%w: len=%d want=%d", ErrBadLength, len(height), n)
	}
	if cellSize <= 0 || math.IsNaN(cellSize) || math.IsInf(cellSize, 0) {
		return nil, fmt.Errorf("%w: cellSize=%v", ErrBadValue, cellSize)
	}
	for _, v := range height {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, ErrBadValue
		}
	}

	f := &Fields{
		Size:     size,
		Receiver: make([]int, n),
		Slope:    make([]float64, n),
		Distance: make([]float64, n),
		Flow:     make([]float64, n),
	}
	for i := range f.Receiver {
		f.Receiver[i] = -1
		f.Flow[i] = 1
	}

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			bestSlope := 0.0
			bestDistance := 0.0
			receiver := -1

			// 固定遍历顺序保证并列坡降时结果确定。
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx, ny := x+dx, y+dy
					if nx < 0 || nx >= size || ny < 0 || ny >= size {
						continue
					}
					j := ny*size + nx
					distance := cellSize * math.Hypot(float64(dx), float64(dy))
					slope := (height[i] - height[j]) / distance
					if slope > bestSlope {
						bestSlope = slope
						bestDistance = distance
						receiver = j
					}
				}
			}
			f.Slope[i] = bestSlope
			f.Distance[i] = bestDistance
			f.Receiver[i] = receiver
		}
	}

	// 下游格高度严格更低，因此按高度降序可保证汇流累加先收齐上游来水。
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		ia, ib := order[a], order[b]
		if height[ia] != height[ib] {
			return height[ia] > height[ib]
		}
		return ia < ib
	})
	for _, i := range order {
		if r := f.Receiver[i]; r >= 0 {
			f.Flow[r] += f.Flow[i]
		}
		p := f.Flow[i] * f.Slope[i]
		if p > f.MaxFlowSlopeProduct {
			f.MaxFlowSlopeProduct = p
		}
	}
	return f, nil
}

func (f *Fields) check() error {
	if f == nil || f.Size <= 0 {
		return ErrBadSize
	}
	n := f.Size * f.Size
	if len(f.Slope) != n || len(f.Flow) != n || len(f.Receiver) != n {
		return ErrBadLength
	}
	return nil
}

func validateCoefficient(k float64) error {
	if math.IsNaN(k) || math.IsInf(k, 0) || k < 0 {
		return fmt.Errorf("%w: K=%v", ErrBadCoeff, k)
	}
	return nil
}

// StepAmount 返回一轮单格侵蚀深度：epsilon_i = K * Q_i * S_i。
// 这是题目要求的“坡度、流量与系数的乘积”闭式表达。
func StepAmount(f *Fields, k float64) ([]float64, error) {
	if err := f.check(); err != nil {
		return nil, err
	}
	if err := validateCoefficient(k); err != nil {
		return nil, err
	}
	out := make([]float64, f.Size*f.Size)
	for i := range out {
		out[i] = k * f.Flow[i] * f.Slope[i]
	}
	return out, nil
}

// TotalErosionClosed 一次算出 N 轮后的总侵蚀深度：E_i = N*K*Q_i*S_i。
func TotalErosionClosed(f *Fields, k float64, rounds int) ([]float64, error) {
	if err := f.check(); err != nil {
		return nil, err
	}
	if err := validateCoefficient(k); err != nil {
		return nil, err
	}
	if rounds < 0 {
		return nil, fmt.Errorf("%w: %d", ErrBadRounds, rounds)
	}
	out := make([]float64, f.Size*f.Size)
	scale := float64(rounds) * k
	for i := range out {
		out[i] = scale * f.Flow[i] * f.Slope[i]
	}
	return out, nil
}

// ClosedFormFixed 用闭式总侵蚀量一次性更新高程。
func ClosedFormFixed(initial []float64, f *Fields, k float64, rounds int) ([]float64, error) {
	if err := f.check(); err != nil {
		return nil, err
	}
	if len(initial) != f.Size*f.Size {
		return nil, fmt.Errorf("%w: initial len=%d", ErrBadLength, len(initial))
	}
	erosion, err := TotalErosionClosed(f, k, rounds)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(initial))
	for i := range initial {
		out[i] = initial[i] - erosion[i]
	}
	return out, nil
}

// IterateFixed 逐步执行 rounds 轮。坡度和汇流量始终使用调用方给定的同一份 f。
// 它是闭式公式的直接迭代参照实现。
func IterateFixed(initial []float64, f *Fields, k float64, rounds int) ([]float64, error) {
	if err := f.check(); err != nil {
		return nil, err
	}
	if len(initial) != f.Size*f.Size {
		return nil, fmt.Errorf("%w: initial len=%d", ErrBadLength, len(initial))
	}
	if err := validateCoefficient(k); err != nil {
		return nil, err
	}
	if rounds < 0 {
		return nil, fmt.Errorf("%w: %d", ErrBadRounds, rounds)
	}
	out := append([]float64(nil), initial...)
	for t := 0; t < rounds; t++ {
		for i := range out {
			out[i] -= k * f.Flow[i] * f.Slope[i]
		}
	}
	return out, nil
}

// GrayCode 与 hmap.EncodePGM 的八位灰度量化规则保持一致。
func GrayCode(v float64) int {
	g := int(math.Round(v * 255))
	if g < 0 {
		return 0
	}
	if g > 255 {
		return 255
	}
	return g
}

// CellDiff 保存一个格子的闭式/迭代逐格比对结果。
type CellDiff struct {
	Index            int     `json:"index"`
	X                int     `json:"x"`
	Y                int     `json:"y"`
	ClosedErosion    float64 `json:"closed_erosion"`
	IteratedErosion  float64 `json:"iterated_erosion"`
	ClosedHeight     float64 `json:"closed_height"`
	IteratedHeight   float64 `json:"iterated_height"`
	AbsHeightError   float64 `json:"abs_height_error"`
	GrayDistance     float64 `json:"gray_distance"`
	ClosedGray       int     `json:"closed_gray"`
	IteratedGray     int     `json:"iterated_gray"`
	EncodedGrayDelta int     `json:"encoded_gray_delta"`
	OverOneBin       bool    `json:"over_one_bin"`
	EncodedOverOne   bool    `json:"encoded_over_one_level"`
}

// Comparison 是某一轮数下的逐格比对汇总。
type Comparison struct {
	Rounds                   int        `json:"rounds"`
	Cells                    int        `json:"cells"`
	ThresholdHeight          float64    `json:"threshold_height"`
	OverOneBinCount          int        `json:"over_one_bin_count"`
	EncodedOverOneLevelCount int        `json:"encoded_over_one_level_count"`
	EncodedNonzeroDeltaCount int        `json:"encoded_nonzero_delta_count"`
	MaxAbsHeightError        float64    `json:"max_abs_height_error"`
	MaxGrayDistance          float64    `json:"max_gray_distance"`
	MaxEncodedGrayDelta      int        `json:"max_encoded_gray_delta"`
	MinClosedHeight          float64    `json:"min_closed_height"`
	MinIteratedHeight        float64    `json:"min_iterated_height"`
	MaxClosedErosionDepth    float64    `json:"max_closed_erosion_depth"`
	Cells_                   []CellDiff `json:"-"`
}

// Compare 逐格比较闭式和迭代后的高程。
// “超过一档灰度”的主判据为 255*|h1-h2| > 1，即高度差严格大于 1/255。
func Compare(size, rounds int, closed, iterated, initial []float64) (*Comparison, error) {
	n := size * size
	if len(closed) != n || len(iterated) != n || len(initial) != n {
		return nil, fmt.Errorf("%w: compare lengths", ErrBadLength)
	}
	cmp := &Comparison{
		Rounds:            rounds,
		Cells:             n,
		ThresholdHeight:   1.0 / 255.0,
		MinClosedHeight:   math.Inf(1),
		MinIteratedHeight: math.Inf(1),
		Cells_:            make([]CellDiff, 0, n),
	}
	for i := 0; i < n; i++ {
		absErr := math.Abs(closed[i] - iterated[i])
		grayDistance := absErr * 255
		cg, ig := GrayCode(closed[i]), GrayCode(iterated[i])
		encodedDelta := cg - ig
		if encodedDelta < 0 {
			encodedDelta = -encodedDelta
		}
		d := CellDiff{
			Index:            i,
			X:                i % size,
			Y:                i / size,
			ClosedErosion:    initial[i] - closed[i],
			IteratedErosion:  initial[i] - iterated[i],
			ClosedHeight:     closed[i],
			IteratedHeight:   iterated[i],
			AbsHeightError:   absErr,
			GrayDistance:     grayDistance,
			ClosedGray:       cg,
			IteratedGray:     ig,
			EncodedGrayDelta: encodedDelta,
			OverOneBin:       grayDistance > 1.0,
			EncodedOverOne:   encodedDelta > 1,
		}
		if d.OverOneBin {
			cmp.OverOneBinCount++
		}
		if d.EncodedOverOne {
			cmp.EncodedOverOneLevelCount++
		}
		if encodedDelta != 0 {
			cmp.EncodedNonzeroDeltaCount++
		}
		if absErr > cmp.MaxAbsHeightError {
			cmp.MaxAbsHeightError = absErr
		}
		if grayDistance > cmp.MaxGrayDistance {
			cmp.MaxGrayDistance = grayDistance
		}
		if encodedDelta > cmp.MaxEncodedGrayDelta {
			cmp.MaxEncodedGrayDelta = encodedDelta
		}
		if closed[i] < cmp.MinClosedHeight {
			cmp.MinClosedHeight = closed[i]
		}
		if iterated[i] < cmp.MinIteratedHeight {
			cmp.MinIteratedHeight = iterated[i]
		}
		if depth := initial[i] - closed[i]; depth > cmp.MaxClosedErosionDepth {
			cmp.MaxClosedErosionDepth = depth
		}
		cmp.Cells_ = append(cmp.Cells_, d)
	}
	return cmp, nil
}

// MismatchMask 返回超过一档灰度判据的二值掩膜（违规格为 1，其余为 0）。
func MismatchMask(cmp *Comparison) []float64 {
	out := make([]float64, len(cmp.Cells_))
	for i, d := range cmp.Cells_ {
		if d.OverOneBin {
			out[i] = 1
		}
	}
	return out
}
