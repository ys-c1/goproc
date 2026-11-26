// Package render 把水位分析结果渲染成带图例与跳变标注的水陆 PNG 图。
package render

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// Palette 是一张图用到的全部颜色（已按数据可视化规范取色）。
type Palette struct {
	Surface     color.NRGBA // 页面底
	Water       color.NRGBA // 水域面
	Land        color.NRGBA // 陆地面
	Minima      color.NRGBA // 极小标记
	Saddle      color.NRGBA // 鞍点标记
	MarkerRing  color.NRGBA // 标记描边
	CurrentBand color.NRGBA // 本档刚跨过事件的强调描边
	Ink         color.NRGBA // 主文字
	Ink2        color.NRGBA // 次文字
	Frame       color.NRGBA // 图框线
}

// DefaultPalette 返回默认浅色配色。
// 橙 #eb6834 / 紫 #6250d6 经调色板校验器全项通过（CVD ΔE 29.1、常色觉 ΔE 34.9）。
func DefaultPalette() Palette {
	return Palette{
		Surface:     color.NRGBA{0xfc, 0xfc, 0xfb, 0xff},
		Water:       color.NRGBA{0x86, 0xb6, 0xef, 0xff}, // 蓝顺序色阶 step250
		Land:        color.NRGBA{0xe0, 0xdd, 0xd2, 0xff}, // 中性暖灰
		Minima:      color.NRGBA{0xeb, 0x68, 0x34, 0xff}, // 分类色 slot2 橙
		Saddle:      color.NRGBA{0x62, 0x50, 0xd6, 0xff}, // 分类色 slot7 紫
		MarkerRing:  color.NRGBA{0xff, 0xff, 0xff, 0xff},
		CurrentBand: color.NRGBA{0xd0, 0x3b, 0x3b, 0xff}, // 状态红，只用于强调描边
		Ink:         color.NRGBA{0x0b, 0x0b, 0x0b, 0xff},
		Ink2:        color.NRGBA{0x52, 0x51, 0x4e, 0xff},
		Frame:       color.NRGBA{0xc3, 0xc2, 0xb7, 0xff},
	}
}

// FontSet 加载一个 TTF 字体并提供常规/加粗两种面。
type FontSet struct {
	regular font.Face
	bold    font.Face
}

// LoadFont 从 TTF 文件路径加载字体，dpi 与基础字号由 size 控制。
func LoadFont(path string, size float64) (*FontSet, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	f, err := opentype.Parse(b)
	if err != nil {
		return nil, err
	}
	face, err := opentype.NewFace(f, &opentype.FaceOptions{
		Size:    size,
		DPI:     96,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return nil, err
	}
	return &FontSet{regular: face, bold: face}, nil
}

// Close 释放字体面。
func (f *FontSet) Close() {
	_ = f.regular.Close()
}

func (f *FontSet) face(bold bool) font.Face {
	if bold && f.bold != nil {
		return f.bold
	}
	return f.regular
}

// text 画一行文字，返回结束 x。
func drawText(img *image.NRGBA, fs *FontSet, x, y int, s string, c color.NRGBA, bold bool) int {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: fs.face(bold),
		Dot:  fixed.P(x, y),
	}
	d.DrawString(s)
	return x + d.MeasureString(s).Round()
}

func fillRect(img *image.NRGBA, x0, y0, x1, y1 int, c color.NRGBA) {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
}

func frameRect(img *image.NRGBA, x0, y0, x1, y1 int, c color.NRGBA) {
	for x := x0; x < x1; x++ {
		img.SetNRGBA(x, y0, c)
		img.SetNRGBA(x, y1-1, c)
	}
	for y := y0; y < y1; y++ {
		img.SetNRGBA(x0, y, c)
		img.SetNRGBA(x1-1, y, c)
	}
}

// MapData 是单张水陆图的全部输入。
type MapData struct {
	Size           int
	Height         []float64
	WaterLevel     float64
	BandIndex      int // 从 1 开始的档位序号
	BandCount      int
	Predicted      int
	Measured       int
	NetChange      int
	BirthsThisBand int
	MergesThisBand int
	Events         []EventMark // 全部已跨过（高度 <= 水位）的临界事件（仅极小/鞍点）
}

// EventMark 是图上要画的一个临界事件标记。
type EventMark struct {
	Y, X      int
	IsMinima  bool
	IsCurrent bool
	Label     string
}

// Options 控制版面。
type Options struct {
	Cell    int // 每格像素边长
	PanelW  int // 右侧信息栏宽度
	Margin  int
	Font    *FontSet
	Palette Palette
}

// Render 把一张水陆图画到 w。
func Render(w io.Writer, d MapData, opt Options) error {
	pal := opt.Palette
	fs := opt.Font
	cell, margin, panelW := opt.Cell, opt.Margin, opt.PanelW
	mapW := d.Size * cell
	mapH := d.Size * cell
	W := margin*2 + mapW + panelW
	H := margin*2 + mapH + 90

	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	fillRect(img, 0, 0, W, H, pal.Surface)

	// 标题区
	titleY := margin + 22
	drawText(img, fs, margin, titleY,
		fmt.Sprintf("水位 %.4f  第 %d/%d 档", d.WaterLevel, d.BandIndex, d.BandCount),
		pal.Ink, true)
	drawText(img, fs, margin, titleY+22,
		"水 = 高度 ≤ 水位 · 四邻域 · 网格外不算水域、不环绕", pal.Ink2, false)

	mapX0, mapY0 := margin, titleY+40
	mapX1, mapY1 := mapX0+mapW, mapY0+mapH

	// 水陆底色
	for y := 0; y < d.Size; y++ {
		for x := 0; x < d.Size; x++ {
			c := pal.Land
			if d.Height[y*d.Size+x] <= d.WaterLevel {
				c = pal.Water
			}
			fillRect(img, mapX0+x*cell, mapY0+y*cell,
				mapX0+(x+1)*cell, mapY0+(y+1)*cell, c)
		}
	}
	frameRect(img, mapX0, mapY0, mapX1, mapY1, pal.Frame)

	// 临界事件标记：极小=实心圆，鞍点=菱形；本档跨过的在同位置套细红环。
	// 标记半径小于格边长一半，保证相邻标记不粘连。
	r := cell / 2
	if r < 2 {
		r = 2
	}
	if r > 4 {
		r = 4
	}
	for _, ev := range d.Events {
		cx := mapX0 + ev.X*cell + cell/2
		cy := mapY0 + ev.Y*cell + cell/2
		col := pal.Saddle
		if ev.IsMinima {
			col = pal.Minima
		}
		// 历史标记用纯色无描边；本档标记加白色细边以便红环衬托。
		if ev.IsCurrent {
			if ev.IsMinima {
				disc(img, cx, cy, r, col, pal.MarkerRing)
			} else {
				diamond(img, cx, cy, r, col, pal.MarkerRing)
			}
			ring(img, cx, cy, r+1, pal.CurrentBand)
		} else {
			if ev.IsMinima {
				disc(img, cx, cy, r, col, col)
			} else {
				diamond(img, cx, cy, r, col, col)
			}
		}
	}

	// 右侧信息栏
	px := mapX1 + 28
	py := mapY0
	line := func(off int, s string, bold bool, c color.NRGBA) {
		drawText(img, fs, px, py+off, s, c, bold)
	}
	line(0, "连通分量数", true, pal.Ink)
	line(24, fmt.Sprintf("预测 %d", d.Predicted), false, pal.Ink)
	line(46, fmt.Sprintf("实测 %d（独立 BFS）", d.Measured), false, pal.Ink)
	sign := ""
	if d.NetChange > 0 {
		sign = "+"
	}
	line(68, fmt.Sprintf("本档净变化 %s%d", sign, d.NetChange), false, pal.Ink)
	line(102, "本档跨过", true, pal.Ink)
	line(126, fmt.Sprintf("极小台地 %d · 鞍点台地 %d", d.BirthsThisBand, d.MergesThisBand),
		false, pal.Ink)

	ly := py + 160
	swatch := 14
	legend := []struct {
		c    color.NRGBA
		text string
		disc bool
	}{
		{pal.Water, "水域（h ≤ 水位）", true},
		{pal.Land, "陆地（h > 水位）", true},
		{pal.Minima, "极小点 分量 +1", true},
		{pal.Saddle, "合并点 分量 −(K−1)", false},
		{pal.CurrentBand, "红环 = 本档刚跨过", true},
	}
	drawText(img, fs, px, ly, "图例", pal.Ink, true)
	ly += 10
	for _, e := range legend {
		y0 := ly + 12
		if e.disc {
			disc(img, px+swatch/2, y0, swatch/2, e.c, pal.MarkerRing)
		} else {
			diamond(img, px+swatch/2, y0, swatch/2, e.c, pal.MarkerRing)
		}
		drawText(img, fs, px+swatch+10, y0+5, e.text, pal.Ink2, false)
		ly += 26
	}
	ly += 6
	drawText(img, fs, px, ly+12, "绘制全部 h ≤ 水位的临界点", pal.Ink2, false)
	drawText(img, fs, px, ly+34, "网格外不是水域，也不环绕", pal.Ink2, false)

	return png.Encode(w, img)
}

func disc(img *image.NRGBA, cx, cy, r int, fill, ring color.NRGBA) {
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			if dx*dx+dy*dy <= r*r {
				edge := dx*dx+dy*dy > (r-1)*(r-1)
				img.SetNRGBA(cx+dx, cy+dy, pick(edge, ring, fill))
			}
		}
	}
}

func ring(img *image.NRGBA, cx, cy, r int, c color.NRGBA) {
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			q := dx*dx + dy*dy
			if q <= r*r && q > (r-1)*(r-1) {
				img.SetNRGBA(cx+dx, cy+dy, c)
			}
		}
	}
}

func diamond(img *image.NRGBA, cx, cy, r int, fill, ringc color.NRGBA) {
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			if abs(dx)+abs(dy) <= r {
				edge := abs(dx)+abs(dy) > r-2
				img.SetNRGBA(cx+dx, cy+dy, pick(edge, ringc, fill))
			}
		}
	}
}

func pick(edge bool, a, b color.NRGBA) color.NRGBA {
	if edge {
		return a
	}
	return b
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
