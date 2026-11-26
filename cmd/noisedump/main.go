// Command noisedump 把每个 fbm 分层（未经振幅加权的原始 Value 场）导出为 JSON，
// 供外部分析脚本与 Python 复现逐值对拍。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"goproc/internal/noise"
)

type dump struct {
	Size        int         `json:"size"`
	Seed        uint64      `json:"seed"`
	Frequency   float64     `json:"frequency"`
	Lacunarity  float64     `json:"lacunarity"`
	Octaves     int         `json:"octaves"`
	Layers      [][]float64 `json:"layers"`
	Composite   []float64   `json:"composite"`
}

func main() {
	size := flag.Int("size", 32, "边长")
	seed := flag.Uint64("seed", 7, "种子")
	octaves := flag.Int("octaves", 8, "层数")
	frequency := flag.Float64("frequency", 0.02, "基础频率")
	lacunarity := flag.Float64("lacunarity", 2.0, "频率倍增")
	gain := flag.Float64("gain", 0.5, "振幅衰减")
	flag.Parse()

	src := noise.New(*seed)
	layers := make([][]float64, *octaves)
	comp := make([]float64, (*size)*(*size))
	amp, freq, norm := 1.0, *frequency, 0.0
	for k := 0; k < *octaves; k++ {
		row := make([]float64, (*size)*(*size))
		for y := 0; y < *size; y++ {
			for x := 0; x < *size; x++ {
				v := src.Value(float64(x)*freq, float64(y)*freq)
				row[y**size+x] = v
				comp[y**size+x] += v * amp
			}
		}
		layers[k] = row
		norm += amp
		amp *= *gain
		freq *= *lacunarity
	}
	for i := range comp {
		comp[i] /= norm
	}
	out, err := json.Marshal(dump{
		Size: *size, Seed: *seed, Frequency: *frequency,
		Lacunarity: *lacunarity, Octaves: *octaves,
		Layers: layers, Composite: comp,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Stdout.Write(out)
}
