package viz

import (
	"bytes"
	"image/png"
	"testing"

	"goproc/internal/normal"
)

func TestWritePNGs(t *testing.T) {
	size := 16
	h := make([]float64, size*size)
	seed := uint64(1)
	for i := range h {
		seed = seed*6364136223846793005 + 1
		h[i] = float64(seed%1000) / 1000
	}
	f := normal.FromHeightmap(size, h)
	var buf bytes.Buffer
	if err := WriteNormalMap(&buf, f); err != nil {
		t.Fatal(err)
	}
	img, err := png.Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if img.Bounds().Dx() != size || img.Bounds().Dy() != size {
		t.Fatalf("法线图尺寸错误: %v", img.Bounds())
	}
	buf.Reset()
	if err := WriteAzimuthMap(&buf, f); err != nil {
		t.Fatal(err)
	}
	if img, err = png.Decode(&buf); err != nil {
		t.Fatal(err)
	}
	if img.Bounds().Dx() != size {
		t.Fatalf("方位图尺寸错误: %v", img.Bounds())
	}
}
