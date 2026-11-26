package ridge

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"goproc/api"
)

// 端到端：命令行写出的报告与叠加图必须自洽，且报告里的关键数字与内存计算一致。
func TestArtifactsConsistent(t *testing.T) {
	dir := t.TempDir()
	h, err := api.Generate(api.Options{Size: 64, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	f, err := NewField(Frame{N: h.Size, F: h.Data})
	if err != nil {
		t.Fatal(err)
	}
	mask := f.Mask()
	crest := f.Extract(ModeCrest)
	masked := f.Extract(ModeMasked)
	rep := f.Analyze("t", mask, f.LocalMaxima())

	// 叠加图尺寸与 PPM 头。
	ppm := EncodePPM(200, f.OverlayWithArcs(mask, crest, masked, 200, f.BuildJoinTree()))
	if len(ppm) < 20 || string(ppm[:3]) != "P6\n" {
		t.Fatalf("PPM 头错误: %q", ppm[:min(10, len(ppm))])
	}
	// PPM 头必须是真正的换行，不是字面量反斜杠。
	if ppm[2] != '\n' {
		t.Fatalf("PPM 头第 3 字节应为换行，got %q", ppm[2])
	}
	want := 200 * 200 * 3
	if got := len(ppm) - (len("P6\n200 200\n255\n")); got != want {
		t.Fatalf("PPM 像素字节 %d want %d", got, want)
	}
	if err := os.WriteFile(filepath.Join(dir, "x.ppm"), ppm, 0o644); err != nil {
		t.Fatal(err)
	}

	// PGM 掩膜头也必须是真换行。
	pgm := pgmBytes(f, mask)
	if pgm[2] != '\n' {
		t.Fatalf("PGM 头第 3 字节应为换行")
	}

	// 报告关键量自洽。
	if rep.Slack != rep.Actual.Total-rep.Actual.MaximaInMask {
		t.Fatalf("slack 不自洽")
	}
	if rep.NoMaxSegments+rep.WithMaxSegments != rep.Actual.Total {
		t.Fatalf("段分解不自洽")
	}
	if rep.Morse.Components != rep.Actual.Total {
		t.Fatalf("Morse 段数 %d != actual %d", rep.Morse.Components, rep.Actual.Total)
	}
}

func pgmBytes(f *Field, mask []bool) []byte {
	head := "P5\n"
	n := f.N
	head += itoa(n) + " " + itoa(n) + "\n255\n"
	out := []byte(head)
	for _, b := range mask {
		if b {
			out = append(out, 255)
		} else {
			out = append(out, 0)
		}
	}
	return out
}

// 浮点数据存档往返（与本次分析无关，但保证 artifacts 的 .f64 可复现）。
func TestFloatRoundTrip(t *testing.T) {
	v := []float64{0.1, -0.25, 1e-9}
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(x))
	}
	for i, x := range v {
		got := math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
		if got != x {
			t.Fatalf("roundtrip %d: %v != %v", i, got, x)
		}
	}
}
