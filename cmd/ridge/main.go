package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/ridge"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "run" {
		fmt.Fprintln(os.Stderr, "usage: ridge run --sizes 128,256 --seed 1 --octaves 4 --out DIR")
		os.Exit(2)
	}
	opt, err := parseArgs(os.Args[2:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	report, err := run(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Print(report)
}

type options struct {
	sizes   []int
	seed    uint64
	octaves int
	gain    float64
	freq    float64
	outDir  string
}

// parseSizes 解析逗号分隔的画幅列表并校验范围。
func parseSizes(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(part, "%d", &n); err != nil {
			return nil, fmt.Errorf("非法画幅 %q", part)
		}
		if n < 16 || n > 4096 {
			return nil, fmt.Errorf("画幅 %d 越界（16–4096）", n)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("没有画幅")
	}
	return out, nil
}

// writeFile 写文件，失败即退出。
func writeFile(path string, b []byte) {
	if err := os.WriteFile(path, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		os.Exit(1)
	}
}

// maskPGM 输出二值脊线掩码：脊线像素 255，背景 0。
func maskPGM(m ridge.Mask) []byte {
	buf := make([]float64, len(m.Data))
	for i, v := range m.Data {
		if v {
			buf[i] = 1
		}
	}
	return encodePGM(m.Size, buf, 1)
}

// componentsPGM 按连通段着色。像素数前十的段在同一色相内由亮到暗分级，
// 其余脊线像素用低亮度中灰；背景为黑。仅用于目视核对归类，不用于测量。
// 灰度满量程 255，保证每一档都落在可分辨的字节上。
func componentsPGM(f ridge.Frame) []byte {
	buf := make([]float64, len(f.Mask.Data))
	rank := make(map[int32]float64, len(f.Comps))
	for i, c := range f.Comps {
		if i < 10 {
			rank[int32(c.ID)] = 1.0 - float64(i)*(0.55/9)
		} else {
			rank[int32(c.ID)] = 0.22
		}
	}
	for i, v := range f.Mask.Data {
		if v {
			buf[i] = rank[f.Labels[i]]
		}
	}
	return encodePGM(f.Size, buf, 255)
}

// encodePGM 与 hmap.EncodePGM 同格式，但允许显式指定满量程。
func encodePGM(size int, v []float64, max int) []byte {
	head := fmt.Sprintf("P5\n%d %d\n%d\n", size, size, max)
	out := make([]byte, 0, len(head)+len(v))
	out = append(out, head...)
	for _, x := range v {
		g := int(x*float64(max) + 0.5)
		if g < 0 {
			g = 0
		}
		if g > max {
			g = max
		}
		out = append(out, byte(g))
	}
	return out
}

// float64Raw 把小端 float64 高度按行优先原样导出。
// 8 位 PGM 的量化步长约 3.9e-3，已大于曲率均值 3.5e-3，
// 从 PGM 复算会把曲率压成十几个取值并制造大量并列；
// 独立核验必须用这份浮点数据，不能用 height_*.pgm。
func float64Raw(v []float64) []byte {
	out := make([]byte, len(v)*8)
	for i, x := range v {
		bits := math.Float64bits(x)
		for b := 0; b < 8; b++ {
			out[i*8+b] = byte(bits >> (8 * b))
		}
	}
	return out
}

// run 生成各画幅、写图与报告，返回报告正文。
func run(opt options) (string, error) {
	if err := os.MkdirAll(opt.outDir, 0o755); err != nil {
		return "", err
	}
	var frames []ridge.Frame
	for _, n := range opt.sizes {
		h, err := api.Generate(api.Options{
			Size: n, Octaves: opt.octaves, Lacunarity: 2, Gain: opt.gain, Seed: opt.seed, Frequency: opt.freq,
		})
		if err != nil {
			return "", fmt.Errorf("generate: %w", err)
		}
		f := ridge.AnalyzeFrame(n, h.Data)
		frames = append(frames, f)

		writeFile(filepath.Join(opt.outDir, fmt.Sprintf("height_%d.pgm", n)), h.ToPGM())
		writeFile(filepath.Join(opt.outDir, fmt.Sprintf("height_%d.f64", n)), float64Raw(h.Data))
		writeFile(filepath.Join(opt.outDir, fmt.Sprintf("ridge_mask_%d.pgm", n)), maskPGM(f.Mask))
		writeFile(filepath.Join(opt.outDir, fmt.Sprintf("ridge_components_%d.pgm", n)), componentsPGM(f))
	}
	report := renderReport(frames, opt)
	writeFile(filepath.Join(opt.outDir, "report.txt"), []byte(report))
	return report, nil
}
