package lod

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// 地形分层设色（低→高）。
var ramp = [][3]float64{
	{31, 64, 74},   // 低：深水青
	{46, 96, 78},   // 低地绿
	{112, 122, 74}, // 丘陵橄榄
	{150, 116, 74}, // 山地褐
	{214, 205, 182},
	{248, 248, 248}, // 山顶白
}

func terrainColor(t float64) [3]float64 {
	t = math.Max(0, math.Min(1, t))
	z := t * float64(len(ramp)-1)
	i := int(z)
	if i >= len(ramp)-1 {
		return ramp[len(ramp)-1]
	}
	f := z - float64(i)
	a, b := ramp[i], ramp[i+1]
	return [3]float64{
		a[0] + (b[0]-a[0])*f,
		a[1] + (b[1]-a[1])*f,
		a[2] + (b[2]-a[2])*f,
	}
}

// shadeAt 计算像素 (x,y) 的晕渲颜色；z 为该像素用于显示的高程。
// gx,gy 是用于法线差分的高程取值函数。裂缝空带返回 NaN 时退化为中心 z，
// 保证边缘像素颜色仍是确定的。
func shadeAt(z float64, gx, gy func(int, int) float64, x, y int) color.RGBA {
	fin := func(v float64) float64 {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return z
		}
		return v
	}
	const zScale = 26.0
	dx := (fin(gx(x+1, y)) - fin(gx(x-1, y))) * zScale
	dy := (fin(gy(x, y+1)) - fin(gy(x, y-1))) * zScale
	nx, ny, nz := -dx, -dy, 1.0
	inv := 1 / math.Hypot(math.Hypot(nx, ny), nz)
	nx, ny, nz = nx*inv, ny*inv, nz*inv
	// 左上方来光。
	lx, ly, lz := -0.55, -0.62, 0.56
	li := 0.62 + 0.5*math.Max(0, nx*lx+ny*ly+nz*lz)
	c := terrainColor(z)
	return color.RGBA{
		R: clamp8(c[0] * li),
		G: clamp8(c[1] * li),
		B: clamp8(c[2] * li),
		A: 255,
	}
}

func clamp8(v float64) uint8 {
	if math.IsNaN(v) || v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// sceneHeight 返回 LOD-k 拼接场景在源像素 (x,y) 的显示高程。
// after=true 时最后一条粗级整格线到接缝之间向共享边界真值过渡。
func sceneHeight(ref []float64, n, k, seamY int, after bool) func(int, int) float64 {
	H := Spacing(k)
	rem := Remainder(seamY, H)
	yc := seamY - rem
	return func(x, y int) float64 {
		if x < 0 {
			x = 0
		} else if x >= n {
			x = n - 1
		}
		if y < 0 {
			y = 0
		} else if y >= n {
			y = n - 1
		}
		if y >= seamY {
			return ref[y*n+x] // 近处全分辨率
		}
		if !after && y > yc {
			return math.NaN() // 修复前的裂缝空带
		}
		if after && y > yc && seamY > yc {
			t := float64(y-yc) / float64(seamY-yc)
			lo := Bilinear(ref, n, H, float64(x), float64(yc))
			hi := ref[seamY*n+x] // 共享边界顶点的真实高程
			return lo + (hi-lo)*t
		}
		return Bilinear(ref, n, H, float64(x), float64(y))
	}
}

// encodePNG 把 RGBA 编码成 PNG 字节。
func encodePNG(img *image.RGBA) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// savePNG 编码并写文件。
func savePNG(path string, img *image.RGBA) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := encodePNG(img)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// scaleNearest 把 src 整数倍最近邻放大 s 倍，返回新 RGBA。
func scaleNearest(src *image.RGBA, s int) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, w*s, h*s))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := src.RGBAAt(b.Min.X+x, b.Min.Y+y)
			for dy := 0; dy < s; dy++ {
				for dx := 0; dx < s; dx++ {
					dst.SetRGBA(x*s+dx, y*s+dy, c)
				}
			}
		}
	}
	return dst
}

// drawText 在 dst 上用 basicfont 画放大 scale 倍的 ASCII 文本。
func drawText(dst *image.RGBA, x, y int, text string, scale int, col color.RGBA) {
	if scale < 1 {
		scale = 1
	}
	w := len(text)*7 + 1
	h := 16
	tmp := image.NewRGBA(image.Rect(0, 0, w, h))
	d := &font.Drawer{
		Dst:  tmp,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  fixed.Point26_6{X: fixed.I(1), Y: fixed.I(12)},
	}
	d.DrawString(text)
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			a := tmp.RGBAAt(px, py)
			if a.A < 100 {
				continue
			}
			for dy := 0; dy < scale; dy++ {
				for dx := 0; dx < scale; dx++ {
					dst.SetRGBA(x+px*scale+dx, y+py*scale+dy, col)
				}
			}
		}
	}
}

func setPx(img *image.RGBA, x, y int, c color.RGBA) {
	if x >= 0 && y >= 0 && x < img.Bounds().Dx() && y < img.Bounds().Dy() {
		img.SetRGBA(x, y, c)
	}
}

// drawHLine 画 1px 水平线（带裁剪）。
func drawHLine(img *image.RGBA, x0, x1, y int, c color.RGBA) {
	for x := x0; x <= x1; x++ {
		setPx(img, x, y, c)
	}
}

func drawVLine(img *image.RGBA, x, y0, y1 int, c color.RGBA) {
	if y0 > y1 {
		y0, y1 = y1, y0
	}
	for y := y0; y <= y1; y++ {
		setPx(img, x, y, c)
	}
}

// drawRect 实心矩形。
func drawRect(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	for y := y0; y <= y1; y++ {
		drawHLine(img, x0, x1, y, c)
	}
}

// drawRectFrame 矩形框。
func drawRectFrame(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	drawHLine(img, x0, x1, y0, c)
	drawHLine(img, x0, x1, y1, c)
	drawVLine(img, x0, y0, y1, c)
	drawVLine(img, x1, y0, y1, c)
}

// renderTerrainCrop 渲染接缝附近的地形裁剪（源像素坐标 crop），返回未放大的小图。
func renderTerrainCrop(ref []float64, n, k, seamY int, after bool, crop image.Rectangle) *image.RGBA {
	w, h := crop.Dx(), crop.Dy()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	zf := sceneHeight(ref, n, k, seamY, after)
	H := Spacing(k)
	yc := seamY - Remainder(seamY, H)
	for y := crop.Min.Y; y < crop.Max.Y; y++ {
		for x := crop.Min.X; x < crop.Max.X; x++ {
			z := zf(x, y)
			lx, ly := x, y
			gx := func(xx, yy int) float64 { return zf(xx, yy) }
			gy := gx
			var c color.RGBA
			if math.IsNaN(z) {
				// 裂缝空带：醒目红色 + 竖纹。
				if (x%3 == 0) || (y%4 == 0) {
					c = color.RGBA{255, 70, 60, 255}
				} else {
					c = color.RGBA{120, 10, 10, 255}
				}
			} else {
				c = shadeAt(z, gx, gy, lx, ly)
			}
			img.SetRGBA(x-crop.Min.X, y-crop.Min.Y, c)
		}
	}
	// 叠加粗级网格线（在粗级区域）。
	lineCol := color.RGBA{20, 16, 10, 150}
	for gx := 0; gx < n; gx += H {
		if gx >= crop.Min.X && gx < crop.Max.X {
			sx := gx - crop.Min.X
			for y := crop.Min.Y; y < minI(crop.Max.Y, seamY); y++ {
				if after || y <= yc {
					setPx(img, sx, y-crop.Min.Y, lineCol)
				}
			}
		}
	}
	for gy := 0; gy < seamY; gy += H {
		if gy >= crop.Min.Y && gy < crop.Max.Y && (after || gy <= yc) {
			drawHLine(img, 0, w-1, gy-crop.Min.Y, lineCol)
		}
	}
	// 接缝参考线：修复后绿色，修复前黄色虚线（名义位置）。
	seamScreenY := seamY - crop.Min.Y
	if seamScreenY >= 0 && seamScreenY < h {
		if after {
			drawHLine(img, 0, w-1, seamScreenY, color.RGBA{40, 255, 120, 255})
		} else {
			for x := 0; x < w; x += 6 {
				drawHLine(img, x, minI(x+3, w-1), seamScreenY, color.RGBA{255, 230, 60, 255})
			}
		}
	}
	return img
}

func minI(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// RenderStitched 生成单个 LOD 层级修复后的完整拼接图（4× 放大，512×512+标注）。
func RenderStitched(ref []float64, n, k int) (*image.RGBA, error) {
	const s = 4
	base := image.NewRGBA(image.Rect(0, 0, n, n))
	zf := sceneHeight(ref, n, k, 63, true)
	g := func(xx, yy int) float64 { return zf(xx, yy) }
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			base.SetRGBA(x, y, shadeAt(zf(x, y), g, g, x, y))
		}
	}
	big := scaleNearest(base, s)
	// 接缝线（不透明亮绿，2 屏幕像素）。
	seamCol := color.RGBA{40, 235, 110, 255}
	for x := 0; x < n*s; x++ {
		big.SetRGBA(x, 63*s, seamCol)
		big.SetRGBA(x, 63*s+1, seamCol)
	}
	// 外框与标题条。
	const bar = 56
	out := image.NewRGBA(image.Rect(0, 0, n*s, n*s+bar))
	drawRect(out, 0, 0, n*s-1, n*s+bar-1, color.RGBA{248, 248, 250, 255})
	copyRGBA(out, big, image.Pt(0, bar))
	drawRectFrame(out, 0, bar, n*s-1, bar+n*s-1, color.RGBA{30, 30, 30, 255})
	drawText(out, 14, 14, lodTitle(k, "STITCHED  w=0px"), 2, color.RGBA{15, 15, 20, 255})
	return out, nil
}

func lodTitle(k int, suffix string) string {
	return "LOD " + string(rune('0'+k)) + "  " + suffix
}

func copyRGBA(dst, src *image.RGBA, at image.Point) {
	for y := 0; y < src.Bounds().Dy(); y++ {
		for x := 0; x < src.Bounds().Dx(); x++ {
			dst.Set(at.X+x, at.Y+y, src.RGBAAt(x, y))
		}
	}
}

// RenderCompare 生成一级的“修复前/修复后 + 几何示意”对照图。
func RenderCompare(ref []float64, n, k int, row LevelRow) (*image.RGBA, error) {
	const scale = 7
	crop := image.Rect(20, 63-26, 108, 63+22) // 88×48
	cw, ch := crop.Dx()*scale, crop.Dy()*scale

	panelB := scaleNearest(renderTerrainCrop(ref, n, k, 63, false, crop), scale)
	panelA := scaleNearest(renderTerrainCrop(ref, n, k, 63, true, crop), scale)

	pad := 24
	titleH := 56
	colW := cw
	schemW := 300
	W := pad + colW + pad + colW + pad + schemW + pad
	H := titleH + ch + pad*2 + 40
	out := image.NewRGBA(image.Rect(0, 0, W, H))
	drawRect(out, 0, 0, W-1, H-1, color.RGBA{250, 250, 252, 255})

	y0 := titleH
	x1 := pad
	x2 := pad + colW + pad
	xs := x2 + colW + pad

	copyRGBAAt(out, panelB, x1, y0)
	copyRGBAAt(out, panelA, x2, y0)
	drawRectFrame(out, x1, y0, x1+cw-1, y0+ch-1, color.RGBA{40, 40, 40, 255})
	drawRectFrame(out, x2, y0, x2+cw-1, y0+ch-1, color.RGBA{40, 40, 40, 255})

	drawText(out, x1, 16, lodTitle(k, "BEFORE NAIVE SEAM"), 2, color.RGBA{150, 20, 20, 255})
	drawText(out, x2, 16, lodTitle(k, "AFTER DIVISIBLE+STITCH"), 2, color.RGBA{10, 110, 50, 255})

	// 面板下方的缝宽读数。
	infoY := y0 + ch + 12
	drawText(out, x1, infoY, formatWidth(row.CrackBeforePx, row.DeltaStride, row.CoefficientC), 2, color.RGBA{120, 20, 20, 255})
	drawText(out, x2, infoY, "w=0.000px  SHARED BOUNDARY", 2, color.RGBA{10, 110, 50, 255})

	// 右侧几何示意图。
	drawSchematic(out, xs, y0, schemW, ch, row)

	// 底部说明。
	drawText(out, pad, H-30, "RED=Crack gap   GREEN=closed seam   CYAN/ORANGE=fine/coarse boundary", 1, color.RGBA{70, 70, 80, 255})
	return out, nil
}

func formatWidth(w float64, delta int, c float64) string {
	return "w=" + f2(w) + "px = C*" + itoa(delta) + "  C=" + f2(c)
}

func f2(v float64) string {
	return bytesTrimFloat(v)
}

// bytesTrimFloat 用固定格式并去掉多余的 0。
func bytesTrimFloat(v float64) string {
	s := strconvFormat(v)
	return s
}

func copyRGBAAt(dst, src *image.RGBA, x, y int) {
	for sy := 0; sy < src.Bounds().Dy(); sy++ {
		for sx := 0; sx < src.Bounds().Dx(); sx++ {
			dst.SetRGBA(x+sx, y+sy, src.RGBAAt(sx, sy))
		}
	}
}

// drawSchematic 在右侧画修复前（上）与修复后（下）的边界折线示意。
func drawSchematic(img *image.RGBA, x0, y0, w, h int, row LevelRow) {
	drawRectFrame(img, x0, y0, x0+w-1, y0+h-1, color.RGBA{180, 180, 190, 255})
	drawText(img, x0+12, y0+10, "GEOMETRY (px)", 1, color.RGBA{60, 60, 70, 255})

	cx := x0 + 70
	xEnd := x0 + w - 24
	scale := 9
	// 上图：before。
	topY := y0 + 70
	fineY := topY
	coarseY := topY - int(row.CrackBeforePx*float64(scale))
	if coarseY < y0+30 {
		coarseY = y0 + 30
	}
	drawHLine(img, cx, xEnd, fineY, color.RGBA{0, 150, 200, 255})
	drawHLine(img, cx, xEnd, coarseY, color.RGBA{230, 130, 0, 255})
	// 顶点。
	for x := cx; x <= xEnd; x += row.FineStride * scale {
		drawRect(img, x-2, fineY-2, x+2, fineY+2, color.RGBA{0, 110, 160, 255})
	}
	for x := cx; x <= xEnd; x += row.CoarseStride * scale {
		drawRect(img, x-3, coarseY-3, x+3, coarseY+3, color.RGBA{200, 100, 0, 255})
	}
	// 缝宽标注。
	drawVLine(img, xEnd+10, coarseY, fineY, color.RGBA{210, 30, 30, 255})
	drawHLine(img, xEnd+5, xEnd+15, coarseY, color.RGBA{210, 30, 30, 255})
	drawHLine(img, xEnd+5, xEnd+15, fineY, color.RGBA{210, 30, 30, 255})
	drawText(img, x0+12, topY+18, "BEFORE", 1, color.RGBA{150, 20, 20, 255})
	drawText(img, xEnd-150, (coarseY+fineY)/2-8, "w="+f1(row.CrackBeforePx), 1, color.RGBA{210, 30, 30, 255})

	// 下图：after，两线重合。
	botY := y0 + h - 70
	drawHLine(img, cx, xEnd, botY, color.RGBA{0, 170, 90, 255})
	drawHLine(img, cx, xEnd, botY+1, color.RGBA{0, 170, 90, 255})
	step := row.FineStride
	if row.AdjFineStride > 0 && row.FineStride == 1 {
		step = 2
	}
	for x := cx; x <= xEnd; x += maxI(step*scale, 6) {
		drawRect(img, x-2, botY-2, x+2, botY+2, color.RGBA{0, 120, 60, 255})
	}
	drawText(img, x0+12, botY+18, "AFTER w=0", 1, color.RGBA{10, 110, 50, 255})
	drawText(img, cx, botY-24, "identical stitched boundary", 1, color.RGBA{10, 110, 50, 255})
}

func f1(v float64) string { return strconvFormatPrec(v, 1) }

func maxI(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// WriteImages 为报告中的每一级写 stitched 与 compare 两张 PNG。
func WriteImages(dir string, ref []float64, n int, rep *Report) error {
	for _, row := range rep.Levels {
		st, err := RenderStitched(ref, n, row.Level)
		if err != nil {
			return err
		}
		if err := savePNG(filepath.Join(dir, filepath.Base(row.StitchedImage)), st); err != nil {
			return err
		}
		cmp, err := RenderCompare(ref, n, row.Level, row)
		if err != nil {
			return err
		}
		if err := savePNG(filepath.Join(dir, filepath.Base(row.CompareImage)), cmp); err != nil {
			return err
		}
	}
	return nil
}
