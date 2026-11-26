// Command energydump 导出 fBm 原图与每个未舍入的尺度分量，供方差能量核对。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/internal/hmap"
	"goproc/internal/noise"
)

type metadata struct {
	Size        int       `json:"size"`
	Octaves     int       `json:"octaves"`
	Seed        uint64    `json:"seed"`
	Lacunarity  float64   `json:"lacunarity"`
	Gain        float64   `json:"gain"`
	Frequency   float64   `json:"frequency"`
	Min         float64   `json:"raw_min"`
	Max         float64   `json:"raw_max"`
	Scale       float64   `json:"normalization_scale"`
	NormAmp     float64   `json:"amplitude_normalizer"`
	Amplitudes  []float64 `json:"amplitudes_coarse_first"`
	Frequencies []float64 `json:"noise_coordinates_coarse_first"`
	Order       string    `json:"binary_order"`
}

func main() {
	size := flag.Int("size", 128, "边长")
	seed := flag.Uint64("seed", 7, "种子")
	octaves := flag.Int("octaves", 4, "fBm 层数")
	lacunarity := flag.Float64("lacunarity", 2.0, "频率倍增")
	gain := flag.Float64("gain", 0.5, "振幅增益")
	frequency := flag.Float64("frequency", 0.02, "基础坐标频率")
	prefix := flag.String("prefix", "artifacts/energy/data/terrain", "输出前缀")
	flag.Parse()

	if *size <= 0 || *octaves <= 0 {
		fmt.Fprintln(os.Stderr, "size and octaves must be positive")
		os.Exit(2)
	}

	n := *size * *size
	src := noise.New(*seed)
	terms := make([][]float64, *octaves)
	amps := make([]float64, *octaves)
	freqs := make([]float64, *octaves)
	ampSum := 0.0
	amp, freq := 1.0, 1.0

	for j := 0; j < *octaves; j++ {
		terms[j] = make([]float64, n)
		amps[j] = amp
		freqs[j] = *frequency * freq
		for y := 0; y < *size; y++ {
			for x := 0; x < *size; x++ {
				terms[j][y**size+x] = src.Value(float64(x)**frequency*freq, float64(y)**frequency*freq) * amp
			}
		}
		ampSum += amp
		amp *= *gain
		freq *= *lacunarity
	}

	raw := make([]float64, n)
	lo, hi := math.Inf(1), math.Inf(-1)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := range terms {
			sum += terms[j][i]
		}
		raw[i] = sum / ampSum
		lo = math.Min(lo, raw[i])
		hi = math.Max(hi, raw[i])
	}
	scale := 1.0 / (hi - lo)

	// Binary layout: normalized original X, then normalized additive components
	// C_j for generator/octave indices j=0..K-1 (coarse to fine).
	bin := make([]float64, 0, (*octaves+1)*n)
	x := make([]float64, n)
	for i := range raw {
		x[i] = (raw[i] - lo) * scale
		bin = append(bin, x[i])
	}
	for j := range terms {
		for i := 0; i < n; i++ {
			bin = append(bin, terms[j][i]/ampSum*scale)
		}
	}

	if err := os.MkdirAll(dirOf(*prefix), 0o755); err != nil {
		fatal(err)
	}
	f, err := os.Create(*prefix + ".f64")
	if err != nil {
		fatal(err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, bin); err != nil {
		fatal(err)
	}

	meta := metadata{
		Size: *size, Octaves: *octaves, Seed: *seed,
		Lacunarity: *lacunarity, Gain: *gain, Frequency: *frequency,
		Min: lo, Max: hi, Scale: scale, NormAmp: ampSum,
		Amplitudes: amps, Frequencies: freqs,
		Order: "X then C_0...C_(K-1), coarse first; row-major float64 JSON array",
	}
	mf, err := os.Create(*prefix + ".json")
	if err != nil {
		fatal(err)
	}
	defer mf.Close()
	enc := json.NewEncoder(mf)
	enc.SetIndent("", "  ")
	if err := enc.Encode(meta); err != nil {
		fatal(err)
	}
	if err := os.WriteFile(*prefix+".pgm", hmap.EncodePGM(*size, x), 0o644); err != nil {
		fatal(err)
	}
	fmt.Printf("wrote %s.f64/json/pgm size=%d octaves=%d\n", *prefix, *size, *octaves)
}

func dirOf(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			return p[:i]
		}
	}
	return "."
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
