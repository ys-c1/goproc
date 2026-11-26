package divide

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
)

// EncodeF64 导出高度图：四字节小端 uint32 边长，随后逐行小端 float64 高度。
func EncodeF64(n int, h []float64) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint32(n))
	_ = binary.Write(&buf, binary.LittleEndian, h)
	return buf.Bytes()
}

func clamp8(v float64) uint8 {
	g := int(v*255 + 0.5)
	if g < 0 {
		g = 0
	}
	if g > 255 {
		g = 255
	}
	return uint8(g)
}

// 叠加配色：H（水平格边）用蓝、V（竖直格边）用红，灰度底图上高对比可辨。
var (
	edgeHColor = color.RGBA{R: 0x2a, G: 0x78, B: 0xd6, A: 255} // 蓝
	edgeVColor = color.RGBA{R: 0xe3, G: 0x49, B: 0x48, A: 255} // 红
)

// RenderGray 把高度图渲染为放大 scale 倍的灰度 PNG。
func RenderGray(n int, h []float64, scale int) []byte {
	if scale < 1 {
		scale = 1
	}
	img := image.NewRGBA(image.Rect(0, 0, n*scale, n*scale))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			g := clamp8(h[y*n+x])
			fillCell(img, x, y, scale, color.RGBA{g, g, g, 255})
		}
	}
	return encodePNG(img)
}

// RenderOverlay 在灰度高度图上叠加分水线格边。
// H 边画在第 y*scale 条水平像素线上，V 边画在第 x*scale 条竖直像素线上，线宽 2 像素。
func RenderOverlay(n int, h []float64, horiz, vert []Edge, scale int) []byte {
	if scale < 1 {
		scale = 1
	}
	w := n * scale
	img := image.NewRGBA(image.Rect(0, 0, w, w))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			g := clamp8(h[y*n+x])
			fillCell(img, x, y, scale, color.RGBA{g, g, g, 255})
		}
	}
	for _, e := range horiz {
		px := e.X * scale
		row := e.Y * scale
		for dx := 0; dx <= scale; dx++ {
			setPx(img, px+dx, row, w, edgeHColor)
			setPx(img, px+dx, row+1, w, edgeHColor)
		}
	}
	for _, e := range vert {
		col := e.X * scale
		py := e.Y * scale
		for dy := 0; dy <= scale; dy++ {
			setPx(img, col, py+dy, w, edgeVColor)
			setPx(img, col+1, py+dy, w, edgeVColor)
		}
	}
	return encodePNG(img)
}

func fillCell(img *image.RGBA, x, y, scale int, c color.RGBA) {
	for dy := 0; dy < scale; dy++ {
		for dx := 0; dx < scale; dx++ {
			img.SetRGBA(x*scale+dx, y*scale+dy, c)
		}
	}
}

func setPx(img *image.RGBA, x, y, w int, c color.RGBA) {
	if x < 0 || y < 0 || x >= w || y >= w {
		return
	}
	img.SetRGBA(x, y, c)
}

func encodePNG(img image.Image) []byte {
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}
