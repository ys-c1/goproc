package main

import (
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"sort"
	"strconv"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"

	"goproc/internal/gully"
)

const cellPx = 8 // 每格像素边长：128*8=1024

type rgb struct{ r, g, b uint8 }

func lerpRGB(a, b rgb, t float64) rgb {
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return rgb{
		r: uint8(float64(a.r) + (float64(b.r)-float64(a.r))*t),
		g: uint8(float64(a.g) + (float64(b.g)-float64(a.g))*t),
		b: uint8(float64(a.b) + (float64(b.b)-float64(a.b))*t),
	}
}

func blend(dst rgb, src rgb, alpha float64) rgb {
	return lerpRGB(dst, src, alpha)
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// hillShade 用中心差分近似地形晕渲，光源在西北偏高。
func hillShade(h []float64, n, x, y int) float64 {
	hl, hr := h[y*n+maxInt(x-1, 0)], h[y*n+minInt(x+1, n-1)]
	hu, hd := h[maxInt(y-1, 0)*n+x], h[minInt(y+1, n-1)*n+x]
	gx := (hr - hl) * 0.5
	gy := (hd - hu) * 0.5
	return clampF(1.0+((-gx)+(-gy))*14.0, 0.55, 1.18)
}

func newFace(size float64) font.Face {
	f, err := opentype.Parse(goregular.TTF)
	if err != nil {
		panic(err)
	}
	fc, err := opentype.NewFace(f, &opentype.FaceOptions{Size: size, DPI: 72})
	if err != nil {
		panic(err)
	}
	return fc
}

func textWidth(face font.Face, s string) int {
	d := font.Drawer{Face: face}
	return d.MeasureString(s).Round()
}

// RenderAnnotationPNG 输出一张冲沟标注图。
func RenderAnnotationPNG(path string, h0, cut []float64, n int,
	core, zone []bool, comp []int, segs []gully.Segment, diff gully.Diff,
	groupTag, kText, statLine string, depthScale float64) error {

	mapPx := n * cellPx
	banner := 78
	img := image.NewRGBA(image.Rect(0, 0, mapPx, mapPx+banner))

	low := rgb{38, 52, 61}
	high := rgb{223, 214, 188}
	zoneC := rgb{255, 176, 64}
	bankC := rgb{245, 158, 32}
	coreC := rgb{235, 58, 58}
	coreDeep := rgb{104, 0, 0}

	// 底图：高程设色 + 晕渲
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := y*n + x
			e := math.Pow(clampF(h0[i], 0, 1), 1.15)
			c := lerpRGB(low, high, e)
			sh := hillShade(h0, n, x, y)
			c = rgb{
				r: uint8(clampF(float64(c.r)*sh, 0, 255)),
				g: uint8(clampF(float64(c.g)*sh, 0, 255)),
				b: uint8(clampF(float64(c.b)*sh, 0, 255)),
			}
			fillCell(img, x, y, c)
		}
	}

	// 八邻域影响带（淡橙晕）
	for i, z := range zone {
		if z && !core[i] {
			x, y := i%n, i/n
			px, py := x*cellPx, y*cellPx
			for yy := py; yy < py+cellPx; yy++ {
				for xx := px; xx < px+cellPx; xx++ {
					o := img.RGBAAt(xx, yy)
					bc := blend(rgb{o.R, o.G, o.B}, zoneC, 0.22)
					img.SetRGBA(xx, yy, color.RGBA{bc.r, bc.g, bc.b, 255})
				}
			}
		}
	}

	// 沟岸拓宽格：内嵌小方块，作为核心的陪衬
	for i, c := range cut {
		if core[i] || c <= 0 {
			continue
		}
		x, y := i%n, i/n
		col := bankC
		if c >= gully.DCore {
			col = rgb{255, 226, 0} // 异常：岸切越过检测杠
		}
		fillCellInset(img, x, y, 2, col)
	}

	// 核心格：鲜红实心 + 亮边
	for i, v := range core {
		if !v {
			continue
		}
		x, y := i%n, i/n
		t := 0.0
		if depthScale > 0 {
			t = clampF(cut[i]/depthScale, 0, 1) // 两组共用同一绝对标尺
		}
		col := blend(coreC, coreDeep, 0.85*t) // 切得越深越暗红
		fillCell(img, x, y, col)
		rectCellBorder(img, x, y, color.RGBA{255, 236, 190, 255})
	}

	// 不一致格标记（正常为 0）
	for _, i := range diff.Missing {
		x, y := i%n, i/n
		rectThick(img, x, y, color.RGBA{0, 255, 255, 255}, 2)
	}
	for _, i := range diff.Extra {
		x, y := i%n, i/n
		drawX(img, x, y, color.RGBA{255, 0, 255, 255})
	}

	// 段编号：防重叠布点 + 引线 + 不透明标签底
	labelFace := newFace(14)
	type placed struct{ rect image.Rectangle }
	occupied := []image.Rectangle{}
	// 大段先占位（标签更需要靠近质心），小段后找位
	ordered := append([]gully.Segment(nil), segs...)
	sort.SliceStable(ordered, func(a, b int) bool { return ordered[a].Count > ordered[b].Count })
	for _, s := range ordered {
		text := "#" + strconv.Itoa(s.ID)
		tw := textWidth(labelFace, text)
		lw, lh := tw+8, 18
		anchor := image.Point{X: int(s.CentroidX*cellPx) + cellPx/2, Y: int(s.CentroidY*cellPx) + cellPx/2}
		r := placeLabelRect(anchor, lw, lh, mapPx, occupied)
		occupied = append(occupied, r)
		// 引线
		leadEnd := image.Point{X: r.Min.X + lw/2, Y: r.Min.Y + lh}
		if anchor.Y < r.Min.Y {
			leadEnd = image.Point{X: r.Min.X + lw/2, Y: r.Min.Y}
		} else if anchor.Y > r.Max.Y {
			leadEnd = image.Point{X: r.Min.X + lw/2, Y: r.Max.Y}
		}
		drawLine(img, anchor, leadEnd, color.RGBA{40, 30, 0, 255})
		// 标签底
		draw.Draw(img, r, &image.Uniform{C: color.RGBA{20, 24, 30, 235}}, image.Point{}, draw.Over)
		borderRect(img, r, color.RGBA{255, 209, 102, 255})
		d := &font.Drawer{Dst: img, Src: image.NewUniform(color.White), Face: labelFace,
			Dot: fixed.P(r.Min.X+4, r.Min.Y+14)}
		d.DrawString(text)
	}

	// 图注条
	draw.Draw(img, image.Rect(0, mapPx, mapPx, mapPx+banner),
		&image.Uniform{C: color.RGBA{18, 22, 27, 255}}, image.Point{}, draw.Src)
	titleFace := newFace(16)
	drawTextFace(img, titleFace, 14, mapPx+24,
		"GULLY CORE ANNOTATION  ["+groupTag+"]  K="+kText, color.RGBA{255, 214, 90, 255})
	drawTextFace(img, labelFace, 14, mapPx+46, statLine, color.RGBA{226, 233, 240, 255})
	// 图例行
	y3 := mapPx + 66
	fillPxRect(img, 14, y3-9, 12, 12, coreC)
	drawTextFace(img, labelFace, 32, y3+1, "core = [Acc>tauAcc] AND [Slope>tauSlope]",
		color.RGBA{210, 220, 228, 255})
	fillPxRect(img, 350, y3-9, 12, 12, bankC)
	drawTextFace(img, labelFace, 368, y3+1, "orange = bank cut / N8 r1 influence zone",
		color.RGBA{210, 220, 228, 255})
	fillPxRect(img, 700, y3-9, 12, 12, rgb{20, 24, 30})
	borderRect(img, image.Rect(700, y3-9, 712, y3+3), color.RGBA{255, 209, 102, 255})
	drawTextFace(img, labelFace, 718, y3+1, "#id = 8-neighbor segment tag",
		color.RGBA{210, 220, 228, 255})

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// placeLabelRect 按“质心上方→下方→左右→螺旋外移”的候选顺序找不与已放标签重叠的框。
func placeLabelRect(anchor image.Point, w, h, mapPx int, occupied []image.Rectangle) image.Rectangle {
	cands := []image.Point{
		{-w / 2, -h - 8}, {-w / 2, 10}, {12, -h / 2}, {-w - 12, -h / 2},
		{-w / 2, -h - 26}, {-w / 2, 28}, {-w/2 - 24, -h / 2}, {-w/2 + 24, -h / 2},
	}
	for step := 0; step < 8; step++ {
		extra := step * 16
		cands = append(cands,
			image.Point{X: -w / 2, Y: -h - 8 - extra},
			image.Point{X: -w / 2, Y: 10 + extra},
			image.Point{X: 12 + extra, Y: -h / 2},
			image.Point{X: -w - 12 - extra, Y: -h / 2},
			image.Point{X: -w/2 - extra, Y: -h/2 - h},
			image.Point{X: -w/2 + extra, Y: -h/2 + h},
		)
	}
	bounds := image.Rect(2, 2, mapPx-2, mapPx-2)
	best := image.Rectangle{}
	for k, off := range cands {
		r := image.Rect(anchor.X+off.X, anchor.Y+off.Y, anchor.X+off.X+w, anchor.Y+off.Y+h)
		if !r.In(bounds) {
			continue
		}
		collide := false
		for _, o := range occupied {
			if r.Intersect(o) != (image.Rectangle{}) {
				collide = true
				break
			}
		}
		if !collide {
			return r
		}
		if k == 0 {
			best = r
		}
	}
	if best.Empty() {
		best = image.Rect(anchor.X-w/2, anchor.Y-h-8, anchor.X+w/2, anchor.Y-8)
	}
	return best
}

func fillCell(img *image.RGBA, x, y int, c rgb) {
	px, py := x*cellPx, y*cellPx
	for yy := py; yy < py+cellPx; yy++ {
		for xx := px; xx < px+cellPx; xx++ {
			img.SetRGBA(xx, yy, color.RGBA{c.r, c.g, c.b, 255})
		}
	}
}

func fillCellInset(img *image.RGBA, x, y, inset int, c rgb) {
	px, py := x*cellPx+inset, y*cellPx+inset
	s := cellPx - 2*inset
	for yy := py; yy < py+s; yy++ {
		for xx := px; xx < px+s; xx++ {
			img.SetRGBA(xx, yy, color.RGBA{c.r, c.g, c.b, 255})
		}
	}
}

func fillPxRect(img *image.RGBA, x, y, w, h int, c rgb) {
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			img.SetRGBA(xx, yy, color.RGBA{c.r, c.g, c.b, 255})
		}
	}
}

func rectCellBorder(img *image.RGBA, x, y int, c color.RGBA) {
	x0, y0 := x*cellPx, y*cellPx
	for k := 0; k < cellPx; k++ {
		img.SetRGBA(x0+k, y0, c)
		img.SetRGBA(x0+k, y0+cellPx-1, c)
		img.SetRGBA(x0, y0+k, c)
		img.SetRGBA(x0+cellPx-1, y0+k, c)
	}
}

func rectThick(img *image.RGBA, x, y int, c color.RGBA, thick int) {
	x0, y0 := x*cellPx, y*cellPx
	s := cellPx
	for t := 0; t < thick; t++ {
		for k := 0; k < s; k++ {
			img.SetRGBA(x0+k, y0+t, c)
			img.SetRGBA(x0+k, y0+s-1-t, c)
			img.SetRGBA(x0+t, y0+k, c)
			img.SetRGBA(x0+s-1-t, y0+k, c)
		}
	}
}

func borderRect(img *image.RGBA, r image.Rectangle, c color.RGBA) {
	for x := r.Min.X; x < r.Max.X; x++ {
		img.SetRGBA(x, r.Min.Y, c)
		img.SetRGBA(x, r.Max.Y-1, c)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		img.SetRGBA(r.Min.X, y, c)
		img.SetRGBA(r.Max.X-1, y, c)
	}
}

func drawX(img *image.RGBA, x, y int, c color.RGBA) {
	x0, y0 := x*cellPx, y*cellPx
	for k := 0; k < cellPx; k++ {
		img.SetRGBA(x0+k, y0+k, c)
		img.SetRGBA(x0+cellPx-1-k, y0+k, c)
	}
}

// drawLine 是整数 Bresenham 直线。
func drawLine(img *image.RGBA, a, b image.Point, c color.RGBA) {
	x0, y0 := a.X, a.Y
	x1, y1 := b.X, b.Y
	dx := absInt(x1 - x0)
	dy := -absInt(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		for ox := -1; ox <= 1; ox++ { // 简易深色衬底，增强引线可见性
			img.SetRGBA(x0+ox, y0, color.RGBA{0, 0, 0, 200})
		}
		img.SetRGBA(x0, y0, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func drawTextFace(img *image.RGBA, face font.Face, x, y int, s string, c color.Color) {
	d := &font.Drawer{
		Dst: img, Src: image.NewUniform(c), Face: face,
		Dot: fixed.P(x, y),
	}
	d.DrawString(s)
}
