// Package viz 把法线场编码为 PNG：标准切线空间法线图与八方位着色图。
package viz

import (
	"image"
	"image/color"
	"image/png"
	"io"
	"math"

	"goproc/internal/normal"
)

// SectorPalette 是八方位分类色（经 dataviz 校验器通过的 8 槽分类调色板，
// 顺序对应箱 0..7，即方位角 [0,45°),[45°,90°),…,[315°,360°)）。
var SectorPalette = [8][3]uint8{
	{0x2a, 0x78, 0xd6}, // 蓝   #2a78d6
	{0xeb, 0x68, 0x34}, // 橙   #eb6834
	{0x1b, 0xaf, 0x7a}, // 青   #1baf7a
	{0xed, 0xa1, 0x00}, // 黄   #eda100
	{0xe8, 0x7b, 0xa4}, // 品红 #e87ba4
	{0x00, 0x83, 0x00}, // 绿   #008300
	{0x62, 0x50, 0xd6}, // 紫   #6250d6
	{0xe3, 0x49, 0x48}, // 红   #e34948
}

func clamp8(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// WriteNormalMap 输出标准切线空间法线图：R=(Nx+1)/2, G=(Ny+1)/2, B=Nz。
// 边界像素（无中心差分）编码为 (128,128,255)，即朝上的默认法线。
func WriteNormalMap(w io.Writer, f *normal.Field) error {
	img := image.NewRGBA(image.Rect(0, 0, f.Size, f.Size))
	for y := 0; y < f.Size; y++ {
		for x := 0; x < f.Size; x++ {
			i := y*f.Size + x
			var r, g, b uint8
			if x == 0 || y == 0 || x == f.Size-1 || y == f.Size-1 {
				r, g, b = 128, 128, 255
			} else {
				r = clamp8((f.Nx[i] + 1) * 127.5)
				g = clamp8((f.Ny[i] + 1) * 127.5)
				b = clamp8(f.Nz[i] * 255)
			}
			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return png.Encode(w, img)
}

// WriteAzimuthMap 输出八方位着色图：按法线方位角选分类色，
// 再以坡度（1-Nz，越大越陡）做亮度调制作为第二编码通道，
// 平顶接近原色、陡坡变暗；边界像素涂灰。
func WriteAzimuthMap(w io.Writer, f *normal.Field) error {
	img := image.NewRGBA(image.Rect(0, 0, f.Size, f.Size))
	for y := 0; y < f.Size; y++ {
		for x := 0; x < f.Size; x++ {
			i := y*f.Size + x
			if x == 0 || y == 0 || x == f.Size-1 || y == f.Size-1 {
				img.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
				continue
			}
			phi := f.Azimuth(i)
			k := int(phi / (2 * math.Pi / 8))
			if k > 7 {
				k = 7
			}
			base := SectorPalette[k]
			// 坡度因子 ∈ [0.55,1]：平顶保留全色，陡坡压暗到 55%。
			shade := 1 - 0.45*(1-f.Nz[i])
			img.SetRGBA(x, y, color.RGBA{
				R: clamp8(float64(base[0]) * shade),
				G: clamp8(float64(base[1]) * shade),
				B: clamp8(float64(base[2]) * shade),
				A: 255,
			})
		}
	}
	return png.Encode(w, img)
}
