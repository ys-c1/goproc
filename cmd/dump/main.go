// Command dump 调用真实的 api.Generate，导出归一化后的 float64 原始高度、
// 八位 PGM 地形图以及一份 JSON 元数据，供离线频谱/计数分析使用。
//
// 用法：
//
//	go run ./cmd/dump -size 256 -seed 7 -octaves 4 -freq 0.05 \
//	    -gain 0.5 -lac 2 -prefix artifacts/data/g1
//
// 产出 <prefix>.f64（小端 float64，长度 size*size，行优先）、
// <prefix>.pgm 与 <prefix>.json。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/noise"
)

type meta struct {
	Size       int     `json:"size"`
	Seed       uint64  `json:"seed"`
	Octaves    int     `json:"octaves"`
	Lacunarity float64 `json:"lacunarity"`
	Gain       float64 `json:"gain"`
	Frequency  float64 `json:"frequency"`
	Weights    []float64 `json:"weights"` // 归一化后的各层振幅权重 w_i = gain^i / sum
}

func main() {
	size := flag.Int("size", 256, "边长")
	seed := flag.Uint64("seed", 7, "种子")
	octaves := flag.Int("octaves", 4, "层数")
	lac := flag.Float64("lac", 2.0, "频率倍增")
	gain := flag.Float64("gain", 0.5, "振幅衰减")
	freq := flag.Float64("freq", 0.02, "基础频率")
	prefix := flag.String("prefix", "out", "输出前缀（生成 .f64/.pgm/.json）")
	flag.Parse()

	opt := api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: *lac,
		Gain: *gain, Seed: *seed, Frequency: *freq,
	}
	h, err := api.Generate(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}

	// 额外保留归一化前的原始 fBm（用于验证功率谱的分层可加性与 w_i^2 权重）。
	n := *size
	src := noise.New(opt.Seed)
	raw := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			raw[y*n+x] = fbm.Eval(src, float64(x)*opt.Frequency, float64(y)*opt.Frequency,
				opt.Octaves, opt.Lacunarity, opt.Gain)
		}
	}

	// 归一化权重，和 internal/fbm 的 sum/norm 完全一致。
	weights := make([]float64, *octaves)
	norm, amp := 0.0, 1.0
	for i := 0; i < *octaves; i++ {
		weights[i] = amp
		norm += amp
		amp *= *gain
	}
	for i := range weights {
		weights[i] /= norm
	}

	m := meta{
		Size: *size, Seed: *seed, Octaves: *octaves,
		Lacunarity: *lac, Gain: *gain, Frequency: *freq, Weights: weights,
	}
	jb, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(*prefix+".json", jb, 0o644); err != nil {
		fatal(err)
	}
	if err := os.WriteFile(*prefix+".pgm", h.ToPGM(), 0o644); err != nil {
		fatal(err)
	}

	f, err := os.Create(*prefix + ".f64")
	if err != nil {
		fatal(err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, h.Data); err != nil {
		fatal(err)
	}
	rf, err := os.Create(*prefix + ".raw.f64")
	if err != nil {
		fatal(err)
	}
	if err := binary.Write(rf, binary.LittleEndian, raw); err != nil {
		fatal(err)
	}
	rf.Close()
	fmt.Printf("dumped prefix=%s size=%d freq=%g oct=%d weights=%v\n",
		*prefix, h.Size, *freq, *octaves, weights)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "dump failed:", err)
	os.Exit(1)
}
