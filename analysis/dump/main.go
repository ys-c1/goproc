// Command dump 调用仓库自身的 api 生成高度图并导出原始 float64（小端）。
// 两种模式：
//
//	单图：  go run ./analysis/dump --size 768 --seed 1 --octaves 4 --out x.f64
//	矩阵：  go run ./analysis/dump --matrix artifacts/extrema/data
//
// 矩阵模式生成分析所需的全部受控配置，并写 manifest.json。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
)

type entry struct {
	Tag       string  `json:"tag"`
	File      string  `json:"file"`
	Role      string  `json:"role"`
	Size      int     `json:"size"`
	Seed      uint64  `json:"seed"`
	Octaves   int     `json:"octaves"`
	Gain      float64 `json:"gain"`
	Lacunarity float64 `json:"lacunarity"`
	Frequency float64 `json:"frequency"`
	Product   float64 `json:"gain_times_octaves"`
}

func writeRaw(path string, d []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var buf [8]byte
	for _, v := range d {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		if _, err := f.Write(buf[:]); err != nil {
			return err
		}
	}
	return nil
}

func genOne(dir string, e entry) (entry, error) {
	h, err := api.Generate(api.Options{
		Size: e.Size, Octaves: e.Octaves, Lacunarity: e.Lacunarity,
		Gain: e.Gain, Seed: e.Seed, Frequency: e.Frequency,
	})
	if err != nil {
		return e, err
	}
	e.File = e.Tag + ".f64"
	e.Product = e.Gain * float64(e.Octaves)
	if err := writeRaw(filepath.Join(dir, e.File), h.Data); err != nil {
		return e, err
	}
	return e, nil
}

func matrix(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var es []entry
	seeds6 := []uint64{1, 7, 42, 99, 2024, 31337}
	seeds4 := []uint64{1, 7, 42, 99}
	add := func(role string, size, n int, g, f float64, seeds []uint64, tagf string) {
		for _, s := range seeds {
			tag := fmt.Sprintf(tagf, s)
			es = append(es, entry{Tag: tag, Role: role, Size: size, Seed: s,
				Octaves: n, Gain: g, Lacunarity: 2.0, Frequency: f})
		}
	}
	// 两个主组：默认增益 0.5 与高增益 0.8，层数 1..8
	for n := 1; n <= 8; n++ {
		add("A", 768, n, 0.5, 0.02, seeds6, fmt.Sprintf("A_g05_N%d_s%%d", n))
		add("B", 768, n, 0.8, 0.02, seeds6, fmt.Sprintf("B_g08_N%d_s%%d", n))
	}
	// 等 g*N 乘积 / 固定层数变增益 的判决性配置
	extra := []struct {
		n int
		g float64
	}{
		{5, 0.4}, {3, 2.0 / 3.0}, // gN=2（与 g=.5,N=4 同乘积）
		{4, 0.75}, // gN=3（与 g=.5,N=6 同乘积）
		{5, 0.3}, {5, 0.7}, {5, 0.9}, // 固定 N=5 变增益
	}
	for _, c := range extra {
		add("product", 768, c.n, c.g, 0.02, seeds4,
			fmt.Sprintf("P_g%.4f_N%d_s%%%%d", c.g, c.n))
	}
	// 单倍频频率标定（N=1，扫描基础频率）
	for _, f := range []float64{0.0125, 1.0 / 60.0, 0.02, 0.025, 1.0 / 30.0,
		0.04, 0.05, 0.0625, 0.08, 0.1, 0.16} {
		add("cal", 768, 1, 0.5, f, []uint64{1, 7, 42},
			fmt.Sprintf("cal_f%.6f_s%%%%d", f))
	}
	// 仓库默认尺寸 128 的对照集
	for n := 1; n <= 8; n++ {
		add("small128", 128, n, 0.5, 0.02, []uint64{1, 7},
			fmt.Sprintf("S128_N%d_s%%d", n))
	}

	out := make([]entry, 0, len(es))
	for i, e := range es {
		ne, err := genOne(dir, e)
		if err != nil {
			return fmt.Errorf("entry %s failed: %w", e.Tag, err)
		}
		out = append(out, ne)
		if (i+1)%20 == 0 || i == len(es)-1 {
			fmt.Printf("generated %d/%d\n", i+1, len(es))
		}
	}
	mf, _ := os.Create(filepath.Join(dir, "manifest.json"))
	enc := json.NewEncoder(mf)
	enc.SetIndent("", " ")
	if err := enc.Encode(out); err != nil {
		return err
	}
	mf.Close()
	fmt.Printf("wrote manifest with %d entries to %s\n", len(out), dir)
	return nil
}

func main() {
	size := flag.Int("size", 512, "边长像素")
	seed := flag.Uint64("seed", 1, "种子")
	octaves := flag.Int("octaves", 4, "层数")
	lac := flag.Float64("lacunarity", 2.0, "频率倍增")
	gain := flag.Float64("gain", 0.5, "振幅增益")
	freq := flag.Float64("frequency", 0.02, "基础频率")
	out := flag.String("out", "", "单图输出 .f64 路径")
	matrixDir := flag.String("matrix", "", "若设置，生成完整实验矩阵到该目录")
	flag.Parse()

	if *matrixDir != "" {
		if err := matrix(*matrixDir); err != nil {
			fmt.Fprintln(os.Stderr, "matrix failed:", err)
			os.Exit(1)
		}
		return
	}
	h, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: *lac,
		Gain: *gain, Seed: *seed, Frequency: *freq,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	if *out == "" {
		fmt.Fprintln(os.Stderr, "missing --out or --matrix")
		os.Exit(2)
	}
	if err := writeRaw(*out, h.Data); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s n=%d\n", *out, len(h.Data))
}
