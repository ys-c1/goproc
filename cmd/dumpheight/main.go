// Command dumpheight writes normalized float64 height fields for analysis.
// It is intentionally separate from gen so entropy and quantiles are computed
// before eight-bit PGM quantization.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/api"
)

func main() {
	size := flag.Int("size", 128, "边长")
	seed := flag.Uint64("seed", 1, "种子")
	frequency := flag.Float64("frequency", 0.02, "基础频率")
	out := flag.String("out", "", "输出原始 float64 文件")
	octave := flag.Int("octaves", 4, "层数")
	flag.Parse()

	if *out == "" {
		fmt.Fprintln(os.Stderr, "usage: dumpheight --octaves K --out FILE")
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{
		Size:       *size,
		Octaves:    *octave,
		Lacunarity: 2,
		Gain:       0.5,
		Seed:       *seed,
		Frequency:  *frequency,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create failed:", err)
		os.Exit(1)
	}
	defer f.Close()

	var header [16]byte
	binary.LittleEndian.PutUint64(header[0:8], uint64(*octave))
	binary.LittleEndian.PutUint64(header[8:16], uint64(len(h.Data)))
	if _, err := f.Write(header[:]); err != nil {
		panic(err)
	}
	var buf [8]byte
	for _, v := range h.Data {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		if _, err := f.Write(buf[:]); err != nil {
			panic(err)
		}
	}
}
