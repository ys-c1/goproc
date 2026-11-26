// Package seam 度量平铺接缝。全部量都以归一化后的高度计，
// 一档灰度固定为 1/255。推导见 artifacts/seam_report.md。
package seam

import (
	"math"

	"goproc/api"
)

// GrayStep 是一档灰度对应的归一化高度。
const GrayStep = 1.0 / 255.0

// Axis 选择接缝方向。
type Axis int

const (
	X Axis = iota // 左右拼接，考察右边界
	Y             // 上下拼接，考察下边界
)

// Jump 返回相位跳变：贴图在边界外一步的应然值与贴图自身首像素之差。
//
//	Jump = |h(Size) − h(0)|
//
// 无缝平铺时它恒为零；不满足整除时它是相位余数的函数，
// 见 artifacts/seam_report.md 式 (2.3)。
func Jump(h *api.Heightmap, ax Axis) float64 {
	n := h.Size
	max := 0.0
	for k := 0; k < n; k++ {
		var d float64
		if ax == X {
			d = h.Sample(float64(n), float64(k)) - h.At(0, k)
		} else {
			d = h.Sample(float64(k), float64(n)) - h.At(k, 0)
		}
		if a := math.Abs(d); a > max {
			max = a
		}
	}
	return max
}

// SeamStep 返回拼接时接缝处的相邻像素差：
// 贴图最后一列/行与贴图自身第一列/行。无缝时它等于内部同位置的差分。
func SeamStep(h *api.Heightmap, ax Axis) float64 {
	n := h.Size
	max := 0.0
	for k := 0; k < n; k++ {
		var d float64
		if ax == X {
			d = h.At(n-1, k) - h.At(0, k)
		} else {
			d = h.At(k, n-1) - h.At(k, 0)
		}
		if a := math.Abs(d); a > max {
			max = a
		}
	}
	return max
}

// Interior 返回内部相邻像素差的最大值，作为正常起伏的基线。
func Interior(h *api.Heightmap, ax Axis) float64 {
	n := h.Size
	max := 0.0
	for k := 0; k < n; k++ {
		for i := 1; i < n; i++ {
			var d float64
			if ax == X {
				d = h.At(i, k) - h.At(i-1, k)
			} else {
				d = h.At(k, i) - h.At(k, i-1)
			}
			if a := math.Abs(d); a > max {
				max = a
			}
		}
	}
	return max
}

// OctaveJump 返回第 octave 层单独造成的相位跳变，单位归一化到零到一。
// 它对应 seam_report.md 的式 (2.4)：J_i 只是该层相位余数的函数。
func OctaveJump(h *api.Heightmap, ax Axis, octave int) float64 {
	n := h.Size
	lo, hi := h.Bounds()
	scale := 1.0
	if hi != lo {
		scale = 1 / (hi - lo)
	}
	max := 0.0
	for k := 0; k < n; k++ {
		var x0, y0, x1, y1 float64
		if ax == X {
			x0, y0 = float64(n), float64(k)
			x1, y1 = 0, float64(k)
		} else {
			x0, y0 = float64(k), float64(n)
			x1, y1 = float64(k), 0
		}
		d := (h.OctaveRaw(x0, y0, octave) - h.OctaveRaw(x1, y1, octave)) * scale
		if a := math.Abs(d); a > max {
			max = a
		}
	}
	return max
}

// Metrics 是一次接缝度量的全部数字。
type Metrics struct {
	Size   int
	Period int

	JumpX, JumpY         float64 // 相位跳变 |h(Size)-h(0)|，见 Jump
	StepX, StepY         float64 // 接缝相邻像素差，见 SeamStep
	InteriorX, InteriorY float64 // 内部相邻像素差最大值，见 Interior
	OctaveJumpX          []float64
	OctaveJumpY          []float64
}

// Measure 度量一张高度图在两个方向上的接缝，并逐层拆分跳变。
func Measure(h *api.Heightmap) Metrics {
	opt := h.Options()
	m := Metrics{
		Size:      h.Size,
		Period:    opt.Period,
		JumpX:     Jump(h, X),
		JumpY:     Jump(h, Y),
		StepX:     SeamStep(h, X),
		StepY:     SeamStep(h, Y),
		InteriorX: Interior(h, X),
		InteriorY: Interior(h, Y),
	}
	for i := 0; i < opt.Octaves; i++ {
		m.OctaveJumpX = append(m.OctaveJumpX, OctaveJump(h, X, i))
		m.OctaveJumpY = append(m.OctaveJumpY, OctaveJump(h, Y, i))
	}
	return m
}

// JumpMax 返回两个方向里较大的相位跳变。
func (m Metrics) JumpMax() float64 { return math.Max(m.JumpX, m.JumpY) }

// StepMax 返回两个方向里较大的接缝相邻像素差。
func (m Metrics) StepMax() float64 { return math.Max(m.StepX, m.StepY) }

// InteriorMax 返回两个方向里较大的内部相邻像素差。
func (m Metrics) InteriorMax() float64 { return math.Max(m.InteriorX, m.InteriorY) }

// BelowOneGray 报告相位跳变是否低于一档灰度。
func (m Metrics) BelowOneGray() bool { return m.JumpMax() < GrayStep }

// Fault 报告接缝跳变是否超出内部相邻像素差——超出即算断层。
func (m Metrics) Fault() bool {
	return m.JumpX > m.InteriorX || m.JumpY > m.InteriorY
}

// StepExceeds 报告接缝处的实际相邻像素差是否超出内部基线。
func (m Metrics) StepExceeds() bool {
	return m.StepX > m.InteriorX || m.StepY > m.InteriorY
}
