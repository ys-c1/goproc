// Package seam 验证两块地形按边拼接时的接缝高度差，
// 以及按边界置换表条目对齐后接缝差恒为零的结论。
package seam

import (
	"image"
	"image/color"
	"image/png"
	"io"
	"math"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

// Config 是拼接实验的全部参数，默认值与 api.DefaultOptions 保持同一组噪声设置。
type Config struct {
	N          int // 单块边长像素数
	Octaves    int
	Lacunarity float64
	Gain       float64
	Frequency  float64
	SeedA      uint64 // A 块种子
	SeedB      uint64 // 对齐前 B 块种子（与 A 不同，边界条目批次不同）
}

// DefaultConfig 返回默认实验参数。
func DefaultConfig() Config {
	return Config{
		N:          128,
		Octaves:    4,
		Lacunarity: 2.0,
		Gain:       0.5,
		Frequency:  0.02,
		SeedA:      11,
		SeedB:      47,
	}
}

// Direction 是拼接方向。
type Direction int

const (
	// Horizontal 左右拼接，接缝为竖线，按同行相邻像素度量（旋转九十度后即同列）。
	Horizontal Direction = iota
	// Vertical 上下拼接，接缝为横线，直接按同列相邻两像素度量。
	Vertical
)

func (d Direction) String() string {
	if d == Horizontal {
		return "horizontal"
	}
	return "vertical"
}

// Tile 是一块带世界坐标偏移的地形，Data 存未归一化的 fbm 原始值。
type Tile struct {
	Size   int
	OffX   float64 // 块左上角在世界坐标中的 x（像素单位，采样前再乘 Frequency）
	OffY   float64
	Source *noise.Source
	Data   []float64
	cfg    Config
}

// Generate 在给定世界偏移与种子下生成一块地形原始高度。
func Generate(cfg Config, seed uint64, offX, offY float64) *Tile {
	src := noise.New(seed)
	data := make([]float64, cfg.N*cfg.N)
	for y := 0; y < cfg.N; y++ {
		for x := 0; x < cfg.N; x++ {
			wx := (offX + float64(x)) * cfg.Frequency
			wy := (offY + float64(y)) * cfg.Frequency
			data[y*cfg.N+x] = fbm.Eval(src, wx, wy, cfg.Octaves, cfg.Lacunarity, cfg.Gain)
		}
	}
	return &Tile{Size: cfg.N, OffX: offX, OffY: offY, Source: src, Data: data, cfg: cfg}
}

// EdgeOffset 返回让 B 的边界像素与 A 的边界像素落在同一世界坐标线上时 B 的偏移。
// 对齐前后 B 都用这个偏移：对齐前两块坐标相同、仅置换表条目不同，
// 因此接缝差唯一来源于边界条目之差，对齐动作只改表条目这一个变量。
func EdgeOffset(cfg Config, dir Direction) (offX, offY float64) {
	e := float64(cfg.N - 1)
	if dir == Horizontal {
		return e, 0 // B 局部零列对齐 A 的右边界世界 x=N-1，行坐标一致
	}
	return 0, e // B 局部零行对齐 A 的下边界世界 y=N-1，列坐标一致
}

// boundarySlots 以参照块 srcRef 的表为准，追踪某条共享边界在给定世界坐标
// 线上、全部边界像素与全部八度求值时实际访问的置换表槽位。
// verticalSeam=true：竖缝，世界 x 固定，沿 y 扫描；否则横缝，世界 y 固定。
func boundarySlots(cfg Config, srcRef *noise.Source, fixed, start float64, verticalSeam bool) map[int]struct{} {
	slots := make(map[int]struct{})
	for k := 0; k < cfg.N; k++ {
		var wx, wy float64
		if verticalSeam {
			wx, wy = fixed*cfg.Frequency, (start+float64(k))*cfg.Frequency
		} else {
			wx, wy = (start+float64(k))*cfg.Frequency, fixed*cfg.Frequency
		}
		fbm.Trace(srcRef, wx, wy, cfg.Octaves, cfg.Lacunarity, cfg.Gain, slots)
	}
	return slots
}

// Align 按边界置换表条目对齐：把参照表在边界槽位上的条目复制进 B 的表，
// 条件即两表在边界处取同一批条目（对每个边界槽位 j 有 P_A[j]=P_B[j]），
// 然后在同一世界坐标下重采样 B，返回复制的去重槽位数。
func Align(cfg Config, ref, b *Tile, dir Direction) (entries map[int]uint8, copied int) {
	slots := BoundarySlots(cfg, ref, dir)
	entries = ref.Source.Snapshot(slots)
	b.Source.ApplyEntries(entries)
	resample(cfg, b)
	return entries, len(entries)
}

// BoundarySlots 返回共享边界在全部边界像素、全部八度求值中实际访问的去重槽位。
func BoundarySlots(cfg Config, ref *Tile, dir Direction) map[int]struct{} {
	if dir == Horizontal {
		fixed := ref.OffX + float64(cfg.N-1) // A 右边界世界 x，B 局部零列也在这条线上
		return boundarySlots(cfg, ref.Source, fixed, ref.OffY, true)
	}
	fixed := ref.OffY + float64(cfg.N-1) // A 下边界世界 y，B 局部零行也在这条线上
	return boundarySlots(cfg, ref.Source, fixed, ref.OffX, false)
}

// BoundaryEntryDiff 统计两表在边界槽位上取值不同的槽位数：
// 该数为零即“两表在边界处取同一批条目”，是接缝差为零的充要条件。
// 置换表双倍长度段互为镜像，槽位统一折算到零到二百五十五后去重。
func BoundaryEntryDiff(cfg Config, ref, other *Tile, dir Direction) (slots, differing int) {
	raw := BoundarySlots(cfg, ref, dir)
	seen := make(map[int]struct{}, len(raw))
	for k := range raw {
		seen[k&255] = struct{}{}
	}
	for j := range seen {
		if ref.Source.Entry(j) != other.Source.Entry(j) {
			differing++
		}
	}
	return len(seen), differing
}

func resample(cfg Config, b *Tile) {
	for y := 0; y < cfg.N; y++ {
		for x := 0; x < cfg.N; x++ {
			wx := (b.OffX + float64(x)) * cfg.Frequency
			wy := (b.OffY + float64(y)) * cfg.Frequency
			b.Data[y*cfg.N+x] = fbm.Eval(b.Source, wx, wy, cfg.Octaves, cfg.Lacunarity, cfg.Gain)
		}
	}
}

// Composite 是两块拼成的灰度图，两块各保留完整边界，接缝对即 A 边界与 B 边界。
// Gray 已按统一尺度量化到零到二百五十五，Raw 保留量化前浮点高度用于证明差恒为零。
type Composite struct {
	Width, Height int
	Gray          []uint8
	Raw           []float64
	SeamCol       int // 竖缝左列的 x（接缝在 SeamCol 与 SeamCol+1 之间），横缝时为 -1
	SeamRow       int // 横缝上行的 y（接缝在 SeamRow 与 SeamRow+1 之间），竖缝时为 -1
	Dir           Direction
}

// Scale 是跨全部实验图共用的灰度量化尺度，保证同一原始高度在任何图中灰度相同。
type Scale struct{ Lo, Hi float64 }

// NewScale 取若干块地形的全局最小最大值。
func NewScale(tiles ...*Tile) Scale {
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, t := range tiles {
		for _, v := range t.Data {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
	}
	return Scale{Lo: lo, Hi: hi}
}

func (s Scale) gray(v float64) uint8 {
	g := (v - s.Lo) / (s.Hi - s.Lo) * 255
	if g < 0 {
		g = 0
	}
	if g > 255 {
		g = 255
	}
	return uint8(math.Round(g))
}

// Stitch 把两块按方向拼接，每块完整保留，接缝两侧即两块各自的边界像素。
func Stitch(a, b *Tile, dir Direction, sc Scale) *Composite {
	n := a.Size
	if dir == Horizontal {
		w, h := 2*n, n
		c := &Composite{Width: w, Height: h, SeamCol: n - 1, SeamRow: -1, Dir: dir,
			Gray: make([]uint8, w*h), Raw: make([]float64, w*h)}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var v float64
				if x < n {
					v = a.Data[y*n+x]
				} else {
					v = b.Data[y*n+(x-n)]
				}
				c.Raw[y*w+x] = v
				c.Gray[y*w+x] = sc.gray(v)
			}
		}
		return c
	}
	w, h := n, 2*n
	c := &Composite{Width: w, Height: h, SeamCol: -1, SeamRow: n - 1, Dir: dir,
		Gray: make([]uint8, w*h), Raw: make([]float64, w*h)}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var v float64
			if y < n {
				v = a.Data[y*n+x]
			} else {
				v = b.Data[(y-n)*n+x]
			}
			c.Raw[y*w+x] = v
			c.Gray[y*w+x] = sc.gray(v)
		}
	}
	return c
}

// Metrics 是接缝度量结果。
type Metrics struct {
	PairCount      int     // 接缝两侧参与比较的像素对数
	MaxGrayDiff    int     // 灰度差的最大值（题面规定的接缝差）
	MeanGrayDiff   float64 // 灰度差的均值
	NonZeroPairs   int     // 灰度差非零的像素对数
	MaxRawDiff     float64 // 量化前原始高度差的最大值（对齐后应精确为零）
	RotatedMaxGray int     // 把图旋转九十度后按同列相邻像素度量的最大差，用于交叉验证
}

// Measure 按方向度量接缝。竖拼直接取同列相邻两像素灰度差的最大值；
// 横拼取同行相邻两像素灰度差的最大值，并对旋转九十度的图再跑一次同列度量。
func (c *Composite) Measure() Metrics {
	m := Metrics{}
	if c.Dir == Vertical {
		m.PairCount = c.Width
		sum := 0
		for x := 0; x < c.Width; x++ {
			g1 := int(c.Gray[c.SeamRow*c.Width+x])
			g2 := int(c.Gray[(c.SeamRow+1)*c.Width+x])
			d := abs(g1 - g2)
			sum += d
			if d > m.MaxGrayDiff {
				m.MaxGrayDiff = d
			}
			if d != 0 {
				m.NonZeroPairs++
			}
			r := math.Abs(c.Raw[c.SeamRow*c.Width+x] - c.Raw[(c.SeamRow+1)*c.Width+x])
			if r > m.MaxRawDiff {
				m.MaxRawDiff = r
			}
		}
		m.MeanGrayDiff = float64(sum) / float64(m.PairCount)
		m.RotatedMaxGray = c.measureRotated()
		return m
	}
	m.PairCount = c.Height
	sum := 0
	for y := 0; y < c.Height; y++ {
		g1 := int(c.Gray[y*c.Width+c.SeamCol])
		g2 := int(c.Gray[y*c.Width+c.SeamCol+1])
		d := abs(g1 - g2)
		sum += d
		if d > m.MaxGrayDiff {
			m.MaxGrayDiff = d
		}
		if d != 0 {
			m.NonZeroPairs++
		}
		r := math.Abs(c.Raw[y*c.Width+c.SeamCol] - c.Raw[y*c.Width+c.SeamCol+1])
		if r > m.MaxRawDiff {
			m.MaxRawDiff = r
		}
	}
	m.MeanGrayDiff = float64(sum) / float64(m.PairCount)
	m.RotatedMaxGray = c.measureRotated()
	return m
}

// measureRotated 顺时针旋转九十度后复算接缝最大灰度差，用另一条代码路径交叉验证：
// 原竖缝旋转后成横线，按同列相邻像素度量；原横缝旋转后成竖缝，按同行相邻像素度量。
func (c *Composite) measureRotated() int {
	rot := rotate90(c)
	maxDiff := 0
	if c.Dir == Horizontal {
		// 原竖缝左列 x=SeamCol 旋转后成为横缝上行 y=SeamCol。
		seamRow := c.SeamCol
		for x := 0; x < rot.Width; x++ {
			d := abs(int(rot.Gray[seamRow*rot.Width+x]) - int(rot.Gray[(seamRow+1)*rot.Width+x]))
			if d > maxDiff {
				maxDiff = d
			}
		}
		return maxDiff
	}
	// 原横缝上行 y=SeamRow、下行 SeamRow+1 旋转后分别落到右列与左列，左列编号如下。
	seamCol := rot.Width - 2 - c.SeamRow
	for y := 0; y < rot.Height; y++ {
		d := abs(int(rot.Gray[y*rot.Width+seamCol]) - int(rot.Gray[y*rot.Width+seamCol+1]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

func rotate90(c *Composite) *Composite {
	w, h := c.Width, c.Height
	r := &Composite{Width: h, Height: w, Gray: make([]uint8, w*h), Raw: make([]float64, w*h)}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			nx := h - 1 - y
			ny := x
			r.Gray[ny*r.Width+nx] = c.Gray[y*w+x]
			r.Raw[ny*r.Width+nx] = c.Raw[y*w+x]
		}
	}
	return r
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// 接缝标注色取自 dataviz 参考调色板的红 #e34948（对中灰 CVD ΔE≈9.1），
// 间隙处补深灰刻线做二次编码，灰度打印下仍可辨认。
var (
	seamRed  = color.RGBA{R: 0xe3, G: 0x49, B: 0x48, A: 0xff}
	seamTick = color.RGBA{R: 0x2c, G: 0x2c, B: 0x2a, A: 0xff}
)

// WriteMarkedPNG 输出带接缝标注的 PNG：接缝两侧像素列（行）画虚线，红段长 8、间隙 6，
// 间隙中用深灰刻线填充。度量始终在未标注的 Gray 上进行，标注只用于展示。
func (c *Composite) WriteMarkedPNG(w io.Writer) error {
	img := image.NewRGBA(image.Rect(0, 0, c.Width, c.Height))
	for y := 0; y < c.Height; y++ {
		for x := 0; x < c.Width; x++ {
			g := c.Gray[y*c.Width+x]
			img.Set(x, y, color.RGBA{R: g, G: g, B: g, A: 0xff})
		}
	}
	mark := func(x, y, along int) {
		if along%14 < 8 {
			img.Set(x, y, seamRed)
		} else {
			img.Set(x, y, seamTick)
		}
	}
	if c.Dir == Horizontal {
		for y := 0; y < c.Height; y++ {
			mark(c.SeamCol, y, y)
			mark(c.SeamCol+1, y, y)
		}
	} else {
		for x := 0; x < c.Width; x++ {
			mark(x, c.SeamRow, x)
			mark(x, c.SeamRow+1, x)
		}
	}
	return png.Encode(w, img)
}

// WritePGM 输出未标注的八位灰度 PGM（P5）。
func (c *Composite) WritePGM(w io.Writer) error {
	if _, err := io.WriteString(w, pgmHeader(c.Width, c.Height)); err != nil {
		return err
	}
	_, err := w.Write(c.Gray)
	return err
}

func pgmHeader(w, h int) string {
	return "P5\n" + itoa(w) + " " + itoa(h) + "\n255\n"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
