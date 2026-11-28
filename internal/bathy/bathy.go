// Package bathy 负责水深量化，以及两种“水深→色阶”映射的逐点对照。
//
// 设归一化水深 d∈[0,1)，色阶数 L，水深量化档数 Q：
//
//	映射 A（连续直映）：    c_A(d) = floor(d·L)
//	映射 B（先量化再映色）：q = floor(d·Q)，c_B(d) = floor(q·L/Q)
//
// 两映射逐点相同当且仅当 L≤Q 且 Q 能被 L 整除（Q=mL）；
// 水深分布连续的作用是：不整除时令失配区间以概率 1 被实际像素命中。
package bathy

import "math"

// Pixel 是一个两映射给色不一致的水下属点。
type Pixel struct {
	X, Y   int
	QBin   int // 量化后的水深档位 q∈[0,Q-1]
	LevelA int // 映射 A 给的色阶
	LevelB int // 映射 B 给的色阶
}

// Delta 是色阶差（按档数计），B 相对 A 的偏移，不整除时恒为 -1。
func (p Pixel) Delta() int { return p.LevelB - p.LevelA }

// BBox 是错色区域的包围盒，坐标为像素行列号（左上原点）。
type BBox struct {
	X0, Y0, X1, Y1 int
}

// LevelReport 是一组色阶数下两映射的对照结果。
type LevelReport struct {
	L           int
	Q           int
	Remainder   int // Q mod L，为零即整除
	DiffPixels  int // 差异像素数
	TotalPixels int // 水下属点总数
	MaxAbsDelta int // 最大色阶差（按档数计）
	HistA       []int
	HistB       []int
	Mismatch    []Pixel
	BBox        *BBox
}

// Depth 由归一化高度 h∈[0,1] 计算相对水深。
// h<seaLevel 为水下，d=(seaLevel-h)/seaLevel 归一化到 [0,1]；陆地点 mask=false。
func Depth(h []float64, seaLevel float64) (depth []float64, mask []bool) {
	depth = make([]float64, len(h))
	mask = make([]bool, len(h))
	for i, v := range h {
		if v >= seaLevel {
			continue
		}
		mask[i] = true
		d := (seaLevel - v) / seaLevel
		if d > 1 {
			d = 1
		}
		depth[i] = d
	}
	return depth, mask
}

// Quantize 把连续水深量化到 Q 个等宽档：q=min(Q-1, floor(d·Q))。
func Quantize(depth []float64, Q int) []int {
	q := make([]int, len(depth))
	fq := float64(Q)
	for i, d := range depth {
		v := int(math.Floor(d * fq))
		if v >= Q {
			v = Q - 1
		}
		q[i] = v
	}
	return q
}

// MapContinuous 是映射 A：连续水深直接定色阶 c=min(L-1, floor(d·L))。
func MapContinuous(depth []float64, L int) []int {
	fl := float64(L)
	c := make([]int, len(depth))
	for i, d := range depth {
		v := int(math.Floor(d * fl))
		if v >= L {
			v = L - 1
		}
		c[i] = v
	}
	return c
}

// MapQuantized 是映射 B：用量化档位定色阶 c=min(L-1, floor(q·L/Q))。
func MapQuantized(qbins []int, Q, L int) []int {
	fl := float64(L)
	fq := float64(Q)
	c := make([]int, len(qbins))
	for i, q := range qbins {
		v := int(math.Floor(float64(q) * fl / fq))
		if v >= L {
			v = L - 1
		}
		c[i] = v
	}
	return c
}

// Compare 在给定 L 下逐点对照两映射，只统计水下属点。
func Compare(size int, depth []float64, mask []bool, qbins []int, Q, L int) LevelReport {
	ca := MapContinuous(depth, L)
	cb := MapQuantized(qbins, Q, L)
	r := LevelReport{L: L, Q: Q, Remainder: Q % L, HistA: make([]int, L), HistB: make([]int, L)}
	var box *BBox
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			if !mask[i] {
				continue
			}
			r.TotalPixels++
			r.HistA[ca[i]]++
			r.HistB[cb[i]]++
			if ca[i] == cb[i] {
				continue
			}
			r.DiffPixels++
			if d := ca[i] - cb[i]; d > r.MaxAbsDelta {
				r.MaxAbsDelta = d
			}
			r.Mismatch = append(r.Mismatch, Pixel{X: x, Y: y, QBin: qbins[i], LevelA: ca[i], LevelB: cb[i]})
			if box == nil {
				box = &BBox{X0: x, Y0: y, X1: x, Y1: y}
			} else {
				if x < box.X0 {
					box.X0 = x
				}
				if x > box.X1 {
					box.X1 = x
				}
				if y < box.Y0 {
					box.Y0 = y
				}
				if y > box.Y1 {
					box.Y1 = y
				}
			}
		}
	}
	r.BBox = box
	return r
}

// MismatchMeasure 返回连续水深分布下失配区间在 [0,1) 上的总长度（测度）。
// 逐档累加：B 的第 q 档 [q/Q,(q+1)/Q) 内，A 在 d≥(b+1)/L 处升档而 B 不升档。
func MismatchMeasure(Q, L int) float64 {
	if Q%L == 0 {
		return 0
	}
	var sum float64
	fq, fl := float64(Q), float64(L)
	for q := 0; q < Q; q++ {
		b := int(math.Floor(float64(q) * fl / fq))
		cut := float64(b+1) / fl
		right := float64(q+1) / fq
		if right > cut {
			sum += right - cut
		}
	}
	return sum
}
