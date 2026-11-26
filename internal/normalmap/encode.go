package normalmap

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"os"
)

// EncodePPM 把量化法线码写成二进制 PPM（P6）。
//
// maxval = 2^bits-1，因此文件逐通道保存的就是 b 位码值：
// bits<=8 每样本 1 字节；bits>8 按 PNM 规范每样本 2 字节大端。
// 这是“按原位深存档”的权威文件。
func EncodePPM(width, height, bits int, codes [][3]int) []byte {
	m := MaxCode(bits)
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "P6\n%d %d\n%d\n", width, height, m)
	wide := m > 255
	for _, c := range codes {
		for k := 0; k < 3; k++ {
			v := c[k]
			if !wide {
				buf.WriteByte(byte(v))
			} else {
				buf.WriteByte(byte(v >> 8))
				buf.WriteByte(byte(v & 0xff))
			}
		}
	}
	return buf.Bytes()
}

// EncodeNormalPNG 把量化法线展开成 PNG 预览。
// bits<=8 输出 8 位 NRGBA（码值线性缩放到 0..255）；
// bits>8 输出 16 位 RGBA64，保留完整码值精度。
func EncodeNormalPNG(width, height, bits int, codes [][3]int) ([]byte, error) {
	m := MaxCode(bits)
	var img image.Image
	if bits <= 8 {
		rgba := image.NewNRGBA(image.Rect(0, 0, width, height))
		for i, c := range codes {
			rgba.Pix[i*4+0] = uint8(scaleTo8(c[0], m))
			rgba.Pix[i*4+1] = uint8(scaleTo8(c[1], m))
			rgba.Pix[i*4+2] = uint8(scaleTo8(c[2], m))
			rgba.Pix[i*4+3] = 255
		}
		img = rgba
	} else {
		rgba := image.NewRGBA64(image.Rect(0, 0, width, height))
		for i, c := range codes {
			o := i * 8
			rgba.Pix[o+0] = uint8(scaleTo16(c[0], m) >> 8)
			rgba.Pix[o+1] = uint8(scaleTo16(c[0], m) & 0xff)
			rgba.Pix[o+2] = uint8(scaleTo16(c[1], m) >> 8)
			rgba.Pix[o+3] = uint8(scaleTo16(c[1], m) & 0xff)
			rgba.Pix[o+4] = uint8(scaleTo16(c[2], m) >> 8)
			rgba.Pix[o+5] = uint8(scaleTo16(c[2], m) & 0xff)
			rgba.Pix[o+6] = 0xff
			rgba.Pix[o+7] = 0xff
		}
		img = rgba
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func scaleTo8(code, m int) int {
	v := (code*255 + m/2) / m
	if v > 255 {
		v = 255
	}
	return v
}

func scaleTo16(code, m int) uint16 {
	v := uint64(code) * 65535 / uint64(m)
	return uint16(v)
}

// EncodeGrayPNG 把 [0,1] 灰度场编码成 8 位 PNG。
func EncodeGrayPNG(width, height int, v []float64) ([]byte, error) {
	img := image.NewGray(image.Rect(0, 0, width, height))
	for i, x := range v {
		g := int(x*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		img.Pix[i] = uint8(g)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// EncodeHeightPGM16 把高度场按其自身 min/max 拉伸后写成 16 位 PGM（P5,
// maxval=65535），避免高度图先在入口被压成 8 位而引入额外台阶。
func EncodeHeightPGM16(width, height int, h []float64) []byte {
	lo, hi := h[0], h[0]
	for _, x := range h {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "P5\n%d %d\n65535\n", width, height)
	rng := hi - lo
	for _, x := range h {
		var v uint16
		if rng > 0 {
			f := (x - lo) / rng
			if f < 0 {
				f = 0
			}
			if f > 1 {
				f = 1
			}
			v = uint16(f*65535 + 0.5)
		}
		buf.WriteByte(byte(v >> 8))
		buf.WriteByte(byte(v & 0xff))
	}
	return buf.Bytes()
}

// WriteFileBytes 是把归档字节落盘的小工具。
func WriteFileBytes(path string, b []byte) error {
	return os.WriteFile(path, b, 0o644)
}

// StackProfiles 把若干同宽的一维亮度剖面纵向堆叠，每条剖面重复 rowRepeat
// 行，输出便于肉眼对比台阶的 PNG。
func StackProfiles(width int, profiles [][]float64, rowRepeat int) ([]byte, error) {
	if rowRepeat < 1 {
		rowRepeat = 1
	}
	rows := len(profiles) * rowRepeat
	img := image.NewGray(image.Rect(0, 0, width, rows))
	y := 0
	for _, s := range profiles {
		for r := 0; r < rowRepeat; r++ {
			for x := 0; x < width; x++ {
				g := 255
				if x < len(s) {
					g = int(s[x]*255 + 0.5)
				}
				if g < 0 {
					g = 0
				}
				if g > 255 {
					g = 255
				}
				img.Pix[y*width+x] = uint8(g)
			}
			y++
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
