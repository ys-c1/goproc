// Package api 是 goproc 的对外入口，提供确定性高度图生成。
package api

import (
	"errors"
	"fmt"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

// Options 是全部可调参数，新增参数必须加在这里并注明默认值。
type Options struct {
	Size       int     // 边长像素数，默认 128
	Octaves    int     // 分形层数，默认 4
	Lacunarity float64 // 频率倍增，默认 2.0
	Gain       float64 // 振幅衰减，默认 0.5
	Seed       uint64  // 种子，默认 1
	Frequency  float64 // 基础频率，默认 0.02
}

// DefaultOptions 返回默认参数。
func DefaultOptions() Options {
	return Options{Size: 128, Octaves: 4, Lacunarity: 2.0, Gain: 0.5, Seed: 1, Frequency: 0.02}
}

var (
	ErrBadSize  = errors.New("尺寸非法")
	ErrBadParam = errors.New("参数越界")
)

func (o Options) validate() error {
	if o.Size <= 0 || o.Size > 4096 {
		return fmt.Errorf("%w: Size=%d", ErrBadSize, o.Size)
	}
	if o.Octaves <= 0 || o.Octaves > 12 {
		return fmt.Errorf("%w: Octaves=%d", ErrBadParam, o.Octaves)
	}
	if o.Lacunarity <= 1 || o.Gain <= 0 || o.Gain >= 1 {
		return fmt.Errorf("%w: Lacunarity=%v Gain=%v", ErrBadParam, o.Lacunarity, o.Gain)
	}
	if o.Frequency <= 0 {
		return fmt.Errorf("%w: Frequency=%v", ErrBadParam, o.Frequency)
	}
	return nil
}

// Heightmap 是生成结果，值域归一化到零到一。
type Heightmap struct {
	Size int
	Data []float64
}

// At 取某点高度。
func (h *Heightmap) At(x, y int) float64 { return h.Data[y*h.Size+x] }

// RectOptions 描述一个各向异性采样的矩形高度图。
//
// 物理域为半开正方形 [0,L)×[0,L)。场点按 (RX*x, RY*y) 送入噪声，
// RX、RY 即两轴相对基础场的坐标缩放比；(NX, NY) 是两轴采样数，
// 由 internal/sampling 按奈奎斯特边反解得到。
type RectOptions struct {
	NX, NY     int     // 两轴采样数
	L          float64 // 物理域边长
	RX, RY     float64 // 两轴缩放比（场点 = (RX*x, RY*y)）
	Octaves    int
	Lacunarity float64
	Gain       float64
	Seed       uint64
}

// RectHeightmap 是矩形（通常各向异性）高度图，行优先，Data 长度 NX*NY。
type RectHeightmap struct {
	NX, NY int
	Data   []float64 // 归一化前的原始分形值，便于离线做频谱分析
}

func (o RectOptions) validate() error {
	if o.NX <= 0 || o.NY <= 0 || o.NX > 8192 || o.NY > 8192 {
		return fmt.Errorf("%w: NX=%d NY=%d", ErrBadSize, o.NX, o.NY)
	}
	if o.L <= 0 || o.RX <= 0 || o.RY <= 0 {
		return fmt.Errorf("%w: L=%v RX=%v RY=%v", ErrBadParam, o.L, o.RX, o.RY)
	}
	if o.Octaves <= 0 || o.Octaves > 12 {
		return fmt.Errorf("%w: Octaves=%d", ErrBadParam, o.Octaves)
	}
	if o.Lacunarity <= 1 || o.Gain <= 0 || o.Gain >= 1 {
		return fmt.Errorf("%w: Lacunarity=%v Gain=%v", ErrBadParam, o.Lacunarity, o.Gain)
	}
	return nil
}

// GenerateRect 按给定网格与缩放比生成矩形高度图，返回未归一化的原始值。
func GenerateRect(opt RectOptions) (*RectHeightmap, error) {
	if err := opt.validate(); err != nil {
		return nil, err
	}
	src := noise.New(opt.Seed)
	out := make([]float64, opt.NX*opt.NY)
	for j := 0; j < opt.NY; j++ {
		py := float64(j) * opt.L / float64(opt.NY)
		for i := 0; i < opt.NX; i++ {
			px := float64(i) * opt.L / float64(opt.NX)
			out[j*opt.NX+i] = fbm.Eval(src, opt.RX*px, opt.RY*py,
				opt.Octaves, opt.Lacunarity, opt.Gain)
		}
	}
	return &RectHeightmap{NX: opt.NX, NY: opt.NY, Data: out}, nil
}

// Normalized 返回零到一归一化后的副本，原图保持不变。
func (h *RectHeightmap) Normalized() []float64 {
	v := append([]float64(nil), h.Data...)
	hmap.Normalize(v)
	return v
}

// ToPGM 导出为八位灰度 PGM。
func (h *RectHeightmap) ToPGM() []byte {
	return hmap.EncodePGMRect(h.NX, h.NY, h.Normalized())
}

// Generate 生成高度图。
func Generate(opt Options) (*Heightmap, error) {
	if err := opt.validate(); err != nil {
		return nil, err
	}
	src := noise.New(opt.Seed)
	out := make([]float64, opt.Size*opt.Size)
	for y := 0; y < opt.Size; y++ {
		for x := 0; x < opt.Size; x++ {
			out[y*opt.Size+x] = fbm.Eval(src, float64(x)*opt.Frequency, float64(y)*opt.Frequency,
				opt.Octaves, opt.Lacunarity, opt.Gain)
		}
	}
	hmap.Normalize(out)
	return &Heightmap{Size: opt.Size, Data: out}, nil
}

// ToPGM 导出为八位灰度 PGM。
func (h *Heightmap) ToPGM() []byte { return hmap.EncodePGM(h.Size, h.Data) }
