// Command autocorr dumps rotated goproc height fields as raw float64 rows.
// The Python driver computes directional autocorrelation from these values.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

type angleSpec struct {
	label string
	deg   float64
}

func parseAngles(s string) ([]angleSpec, error) {
	var out []angleSpec
	for _, item := range strings.Split(s, ",") {
		parts := strings.SplitN(item, ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("angle %q must be label:degrees", item)
		}
		deg, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			return nil, fmt.Errorf("angle %q has bad degrees: %w", item, err)
		}
		out = append(out, angleSpec{label: parts[0], deg: deg})
	}
	return out, nil
}

func dump(path string, size int, src *noise.Source, h float64, octaves int, lacunarity, gain, deg float64) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	data := make([]float64, size*size)
	c := (float64(size) - 1) / 2
	theta := deg * math.Pi / 180
	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)

	for row := 0; row < size; row++ {
		dy := float64(row) - c
		for col := 0; col < size; col++ {
			dx := float64(col) - c
			// Rotate the sampling coordinates, not the finished image.
			x := h * (cosTheta*dx - sinTheta*dy)
			y := h * (sinTheta*dx + cosTheta*dy)
			data[row*size+col] = fbm.Eval(src, x, y, octaves, lacunarity, gain)
		}
	}

	bw := bufio.NewWriter(f)
	if err := binary.Write(bw, binary.LittleEndian, data); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return f.Close()
}

func main() {
	outDir := flag.String("out-dir", "artifacts/autocorr/raw", "directory for raw float64 dumps")
	size := flag.Int("size", 2048, "square side length")
	seed := flag.Uint64("seed", 7, "noise seed")
	h := flag.Float64("frequency", 0.5, "noise-coordinate step per pixel (Options.Frequency)")
	octaves := flag.Int("octaves", 1, "FBM octave count; use 1 to isolate the permutation period")
	lacunarity := flag.Float64("lacunarity", 2.0, "FBM lacunarity")
	gain := flag.Float64("gain", 0.5, "FBM gain")
	anglesFlag := flag.String("angles", "theta45:45,theta26p565:26.56505117707799", "comma-separated label:degrees")
	flag.Parse()

	angles, err := parseAngles(*anglesFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if *size <= 0 || *h <= 0 || *octaves <= 0 {
		fmt.Fprintln(os.Stderr, "size, frequency and octaves must be positive")
		os.Exit(2)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	src := noise.New(*seed)
	for _, a := range angles {
		path := filepath.Join(*outDir, a.label+".f64")
		if err := dump(path, *size, src, *h, *octaves, *lacunarity, *gain, a.deg); err != nil {
			fmt.Fprintf(os.Stderr, "dump %s failed: %v\n", a.label, err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s\n", path)
	}
}
