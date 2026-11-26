// Package api 是 goproc 的对外入口，提供确定性高度图生成。
package api

import (
	"errors"
	"fmt"
	"math"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

// DefaultPeriod 是默认置换表周期（格点）。无缝平铺的充要条件是每一层的
// 相位（频率 × 边长）都能被该周期整除，证明见 artifacts/seam_report.md。
const DefaultPeriod = noise.DefaultPeriod

// Options 是全部可调参数，新增参数必须加在这里并注明默认值。
type Options struct {
	Size       int     // 边长像素数，默认 128
	Octaves    int     // 分形层数，默认 4
	Lacunarity float64 // 频率倍增，默认 2.0
	Gain       float64 // 振幅衰减，默认 0.5
	Seed       uint64  // 种子，默认 1
	Frequency  float64 // 基础频率，默认 0.02
	Period     int     // 置换表周期（格点），0 表示 DefaultPeriod
	TileCells  int     // 平铺时第零层横跨的格点数，0 表示不平铺；非零时覆盖 Frequency 为 TileCells/Size
}

// DefaultOptions 返回默认参数。
func DefaultOptions() Options {
	return Options{Size: 128, Octaves: 4, Lacunarity: 2.0, Gain: 0.5, Seed: 1, Frequency: 0.02}
}

var (
	ErrBadSize     = errors.New("尺寸非法")
	ErrBadParam    = errors.New("参数越界")
	ErrNotTileable = errors.New("无法无缝平铺")
)

// Divisible 判断 a 能否被 b 整除。判定就是两数取余为零。
func Divisible(a, b int) bool {
	if b == 0 {
		return false
	}
	return a%b == 0
}

// Resolve 返回补齐默认值后的参数副本：
// Period 为零时取 DefaultPeriod；TileCells 为正时由它推出频率。
func (o Options) Resolve() Options {
	if o.Period == 0 {
		o.Period = DefaultPeriod
	}
	if o.TileCells > 0 {
		o.Frequency = float64(o.TileCells) / float64(o.Size)
	}
	return o
}

// period 返回实际生效的置换表周期。
func (o Options) period() int {
	if o.Period == 0 {
		return DefaultPeriod
	}
	return o.Period
}

// Phase 返回第 octave 层在接缝处的相位，换算按频率与边长的乘积给：
//
//	Phase(i) = Frequency * Lacunarity^i * Size
//
// 它等于一份贴图在该层横跨的格点数。
func (o Options) Phase(octave int) float64 {
	o = o.Resolve()
	return o.Frequency * math.Pow(o.Lacunarity, float64(octave)) * float64(o.Size)
}

// PhaseCells 把 Phase 取整，报告它离整数有多远。值以格点计。
func (o Options) PhaseCells(octave int) (cells int, ok bool) {
	p := o.Phase(octave)
	r := math.Round(p)
	if math.Abs(p-r) > 1e-9 || r < 0 || r > math.MaxInt32 {
		return 0, false
	}
	return int(r), true
}

// Remainder 返回该层相位对置换表周期取余的浮点结果。
// 余数为零等价于首尾落在同一格点上。
func (o Options) Remainder(octave int) float64 {
	o = o.Resolve()
	p := o.Phase(octave)
	rem := math.Mod(p, float64(o.period()))
	if rem < 0 {
		rem += float64(o.period())
	}
	return rem
}

// RemainderCells 返回整数余数；相位不是整数时 ok 为 false。
func (o Options) RemainderCells(octave int) (rem int, ok bool) {
	cells, ok := o.PhaseCells(octave)
	if !ok {
		return 0, false
	}
	return cells % o.period(), true
}

// PhaseDivisible 报告第 octave 层是否满足「相位被置换表周期整除」。
// 由 Divisible 给出，即两数取余为零；相位不是整数时直接判否。
func (o Options) PhaseDivisible(octave int) bool {
	cells, ok := o.PhaseCells(octave)
	if !ok {
		return false
	}
	return Divisible(cells, o.period())
}

// Tileable 报告每一层是否都能首尾相接。
func (o Options) Tileable() bool {
	o = o.Resolve()
	for i := 0; i < o.Octaves; i++ {
		if !o.PhaseDivisible(i) {
			return false
		}
	}
	return true
}

// TileReport 返回逐层的相位、余数与整除结论，供命令行与留档使用。
func (o Options) TileReport() (phase []float64, remainder []float64, divisible []bool) {
	o = o.Resolve()
	for i := 0; i < o.Octaves; i++ {
		phase = append(phase, o.Phase(i))
		remainder = append(remainder, o.Remainder(i))
		divisible = append(divisible, o.PhaseDivisible(i))
	}
	return
}

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
	if o.Frequency <= 0 || math.IsInf(o.Frequency, 0) || math.IsNaN(o.Frequency) {
		return fmt.Errorf("%w: Frequency=%v", ErrBadParam, o.Frequency)
	}
	if o.Period < 2 || o.Period > 1<<20 {
		return fmt.Errorf("%w: Period=%d", ErrBadParam, o.Period)
	}
	if o.TileCells < 0 || o.TileCells > 1<<24 {
		return fmt.Errorf("%w: TileCells=%d", ErrBadParam, o.TileCells)
	}
	if o.TileCells > 0 {
		for i := 0; i < o.Octaves; i++ {
			cells, ok := o.PhaseCells(i)
			if !ok || !Divisible(cells, o.period()) {
				return fmt.Errorf("%w: 第 %d 层相位 %v 格点，对置换表周期 %d 取余 %d",
					ErrNotTileable, i, o.Phase(i), o.period(), int(o.Remainder(i)))
			}
		}
	}
	return nil
}

// Heightmap 是生成结果，值域归一化到零到一。
type Heightmap struct {
	Size int
	Data []float64

	opt Options
	src *noise.Source
	lo  float64
	hi  float64
}

// At 取某点高度。
func (h *Heightmap) At(x, y int) float64 { return h.Data[y*h.Size+x] }

// Options 返回实际生效的参数（已补齐默认值）。
func (h *Heightmap) Options() Options { return h.opt }

// Sample 在任意像素坐标处求值，并按生成时同一组上下界归一化。
// 坐标可以取到 Size（即贴图右侧的延续位置），用于度量接缝两侧的值。
func (h *Heightmap) Sample(x, y float64) float64 {
	return h.Normalized(h.Raw(x, y))
}

// Raw 在任意像素坐标处返回未归一化的 fbm 值。
func (h *Heightmap) Raw(x, y float64) float64 {
	return fbm.Eval(h.src, x*h.opt.Frequency, y*h.opt.Frequency,
		h.opt.Octaves, h.opt.Lacunarity, h.opt.Gain)
}

// Normalized 用生成时的上下界把原始值映射到零到一。
func (h *Heightmap) Normalized(v float64) float64 {
	if h.hi == h.lo {
		return 0
	}
	return (v - h.lo) / (h.hi - h.lo)
}

// Bounds 返回本次生成时原始值的上下界。
func (h *Heightmap) Bounds() (lo, hi float64) { return h.lo, h.hi }

// OctaveRaw 返回第 octave 层在任意像素坐标处的未归一化加权贡献。
// 各层贡献相加就是 Raw；用它可以把接缝跳变逐层拆开。
func (h *Heightmap) OctaveRaw(x, y float64, octave int) float64 {
	return fbm.Octave(h.src, x*h.opt.Frequency, y*h.opt.Frequency,
		octave, h.opt.Octaves, h.opt.Lacunarity, h.opt.Gain)
}

// OctaveValue 返回第 octave 层贡献按生成时上下界归一化后的值。
func (h *Heightmap) OctaveValue(x, y float64, octave int) float64 {
	return h.Normalized(h.OctaveRaw(x, y, octave))
}

// Generate 生成高度图。
//
// 置换表以 Period 个格点为周期，因此只要每一层的
// 频率 × 边长都能被该周期整除，贴图右边界就与左边界落在同一格点上，
// 拼接时首尾相接。参数不满足时仍会生成，但 Tileable 返回 false，
// 接缝处会留下相位差。
func Generate(opt Options) (*Heightmap, error) {
	opt = opt.Resolve()
	if err := opt.validate(); err != nil {
		return nil, err
	}
	src := noise.NewPeriod(opt.Seed, opt.Period)
	out := make([]float64, opt.Size*opt.Size)
	for y := 0; y < opt.Size; y++ {
		for x := 0; x < opt.Size; x++ {
			out[y*opt.Size+x] = fbm.Eval(src, float64(x)*opt.Frequency, float64(y)*opt.Frequency,
				opt.Octaves, opt.Lacunarity, opt.Gain)
		}
	}
	lo, hi := hmap.Normalize(out)
	return &Heightmap{Size: opt.Size, Data: out, opt: opt, src: src, lo: lo, hi: hi}, nil
}

// ToPGM 导出为八位灰度 PGM。
func (h *Heightmap) ToPGM() []byte { return hmap.EncodePGM(h.Size, h.Data) }

// Repeat 返回按 reps×reps 拼接后的高度数据，拼接顺序与贴图相同。
// 用于留档，也用于在拼好的图上直接量接缝。
func (h *Heightmap) Repeat(reps int) []float64 {
	if reps < 1 {
		reps = 1
	}
	n := h.Size
	side := n * reps
	out := make([]float64, side*side)
	for ry := 0; ry < reps; ry++ {
		for rx := 0; rx < reps; rx++ {
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					out[(ry*n+y)*side+rx*n+x] = h.Data[y*n+x]
				}
			}
		}
	}
	return out
}

// ToPGMRepeat 导出按 reps×reps 拼接后的八位灰度 PGM。
func (h *Heightmap) ToPGMRepeat(reps int) []byte {
	return hmap.EncodePGM(h.Size*reps, h.Repeat(reps))
}

// JoinReport 判定能否无缝平铺，不能时给出逐层余数。
// 判定的写法就是两数取余为零；余数非零的层按式 (2.4) 留下相位差。
func (o Options) JoinReport() error {
	o = o.Resolve()
	for i := 0; i < o.Octaves; i++ {
		cells, ok := o.PhaseCells(i)
		if !ok {
			return fmt.Errorf("%w: 第 %d 层相位 %v 不是整数格点", ErrNotTileable, i, o.Phase(i))
		}
		if rem := cells % o.period(); rem != 0 {
			return fmt.Errorf("%w: 第 %d 层相位 %d 对置换表周期 %d 取余 %d",
				ErrNotTileable, i, cells, o.period(), rem)
		}
	}
	return nil
}
