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

// Generate 生成高度图（结果归一化到零到一）。
func Generate(opt Options) (*Heightmap, error) {
	data, err := GenerateRaw(opt)
	if err != nil {
		return nil, err
	}
	hmap.Normalize(data)
	return &Heightmap{Size: opt.Size, Data: data}, nil
}

// GenerateRaw 生成未归一化的原始 fBm 值，用于需要逐点精确比较的数值实验。
func GenerateRaw(opt Options) ([]float64, error) {
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
	return out, nil
}

// ToPGM 导出为八位灰度 PGM。
func (h *Heightmap) ToPGM() []byte { return hmap.EncodePGM(h.Size, h.Data) }

// Gradients 返回单位像素前向差分梯度 (gx, gy)，边缘像素退回后向差分。
func (h *Heightmap) Gradients() (gx, gy []float64) {
	return hmap.Gradient(h.Size, h.Data, 1)
}

// ShadeLinear 返回未截断的点积亮度 L=l·n，可正可负。
func (h *Heightmap) ShadeLinear(l hmap.Light) []float64 {
	gx, gy := h.Gradients()
	return hmap.ShadeLinear(gx, gy, l)
}

// Shade 返回完整朗伯光照亮度（零到一）及背光（点积为负）像素占比。
func (h *Heightmap) Shade(l hmap.Light) (b []float64, zeroFrac float64) {
	gx, gy := h.Gradients()
	return hmap.Shade(gx, gy, l)
}
