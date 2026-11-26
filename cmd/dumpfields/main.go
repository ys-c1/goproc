// Command dumpfields 把增益×层数×种子矩阵的高度场以小端 float32 原始数据导出，
// 供 analysis/ 下的频谱分析读取。它复用 api.Generate，保证分析对象就是产品实现。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
)

type condition struct {
	Gain    float64 `json:"gain"`
	Octaves int     `json:"octaves"`
}

type manifest struct {
	Size       int         `json:"size"`
	Frequency  float64     `json:"frequency"`
	Lacunarity float64     `json:"lacunarity"`
	Seeds      []uint64    `json:"seeds"`
	Encoding   string      `json:"encoding"`
	Files      []fileEntry `json:"files"`
}

type fileEntry struct {
	File    string  `json:"file"`
	Gain    float64 `json:"gain"`
	Octaves int     `json:"octaves"`
	Seed    uint64  `json:"seed"`
}

func parseGains(s string) ([]float64, error) {
	var out []float64
	for _, tok := range strings.Split(s, ",") {
		var g float64
		if _, err := fmt.Sscanf(strings.TrimSpace(tok), "%g", &g); err != nil {
			return nil, fmt.Errorf("解析增益 %q: %w", tok, err)
		}
		out = append(out, g)
	}
	return out, nil
}

func main() {
	outDir := flag.String("out", "build/fields", "原始场输出目录")
	size := flag.Int("size", 512, "边长")
	freq := flag.Float64("freq", api.DefaultOptions().Frequency, "基础频率（像素倒数）")
	lac := flag.Float64("lacunarity", 2.0, "频率倍增")
	gainsFlag := flag.String("gains", "0.35,0.50,0.65", "逗号分隔的三档增益")
	octsFlag := flag.String("octaves", "4,8", "逗号分隔的层数档")
	nSeeds := flag.Int("seeds", 16, "连续种子数量（从 seed-base 起）")
	seedBase := flag.Uint64("seed-base", 1000, "种子基址")
	flag.Parse()

	gains, err := parseGains(*gainsFlag)
	if err != nil {
		fatal(err)
	}
	var octs []int
	for _, tok := range strings.Split(*octsFlag, ",") {
		var o int
		if _, err := fmt.Sscanf(strings.TrimSpace(tok), "%d", &o); err != nil {
			fatal(err)
		}
		octs = append(octs, o)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err)
	}

	mf := manifest{
		Size: *size, Frequency: *freq, Lacunarity: *lac,
		Encoding: "raw little-endian float32, row-major, normalized [0,1]",
	}
	for i := 0; i < *nSeeds; i++ {
		mf.Seeds = append(mf.Seeds, *seedBase+uint64(i))
	}

	for _, g := range gains {
		for _, o := range octs {
			for _, seed := range mf.Seeds {
				opt := api.Options{
					Size: *size, Octaves: o, Lacunarity: *lac,
					Gain: g, Seed: seed, Frequency: *freq,
				}
				h, err := api.Generate(opt)
				if err != nil {
					fatal(err)
				}
				name := fmt.Sprintf("g%.2f_o%d_s%d.f32", g, o, seed)
				path := filepath.Join(*outDir, name)
				if err := writeF32(path, h.Data); err != nil {
					fatal(err)
				}
				mf.Files = append(mf.Files, fileEntry{
					File: name, Gain: g, Octaves: o, Seed: seed,
				})
			}
			fmt.Printf("dumped gain=%.2f octaves=%d seeds=%d\n", g, o, *nSeeds)
		}
	}
	b, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		fatal(err)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "manifest.json"), b, 0o644); err != nil {
		fatal(err)
	}
	fmt.Printf("manifest -> %s (%d fields)\n", filepath.Join(*outDir, "manifest.json"), len(mf.Files))
}

func writeF32(path string, v []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		bits := math.Float32bits(float32(x))
		buf[4*i] = byte(bits)
		buf[4*i+1] = byte(bits >> 8)
		buf[4*i+2] = byte(bits >> 16)
		buf[4*i+3] = byte(bits >> 24)
	}
	if _, err := f.Write(buf); err != nil {
		return err
	}
	return nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "dumpfields:", err)
	os.Exit(1)
}
