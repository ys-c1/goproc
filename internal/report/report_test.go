package report

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goproc/api"
)

func TestAnalyzeCalmSetWithinTwoDegrees(t *testing.T) {
	s, err := Analyze("calm", api.Options{Size: 256, Seed: 7, Octaves: 4, Lacunarity: 2, Gain: 0.5, Frequency: 0.012})
	if err != nil {
		t.Fatal(err)
	}
	if s.Measured.Count != 254*254 {
		t.Fatalf("参与统计 %d 格，应为 %d", s.Measured.Count, 254*254)
	}
	if math.Abs(s.Overview-s.Measured.MeanTilt) > 0.5 {
		t.Fatalf("整图预判 %.3f 与实测 %.3f 偏差超过 0.5 度", s.Overview, s.Measured.MeanTilt)
	}
	for _, b := range s.Bands {
		if b.AboveNyquist() {
			t.Fatalf("低频组不应有超 Nyquist 档: %v", b.Frequency)
		}
		if v := b.Verdict(); v != "合格" {
			t.Fatalf("第 %d 档结论 %q，应为合格", b.Index, v)
		}
	}
	if s.WorstAbsDev > TwoDegrees {
		t.Fatalf("最差逐档偏差 %.3f 超过两度", s.WorstAbsDev)
	}
}

func TestAnalyzeRuggedSetFlagsSuperNyquist(t *testing.T) {
	s, err := Analyze("rugged", api.Options{Size: 256, Seed: 7, Octaves: 6, Lacunarity: 2, Gain: 0.62, Frequency: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	last := s.Bands[len(s.Bands)-1]
	if !last.AboveNyquist() {
		t.Fatalf("最高档 f=%.3f 应超过 Nyquist", last.Frequency)
	}
	if math.Abs(last.DeltaTilt) <= TwoDegrees {
		t.Fatalf("超 Nyquist 档偏差 %.3f 应超过两度", last.DeltaTilt)
	}
	if last.SlopeRatio >= 0.8 {
		t.Fatalf("超 Nyquist 档斜率比 %.3f 应明显小于 1", last.SlopeRatio)
	}
	if !strings.Contains(last.Verdict(), "斜率") {
		t.Fatalf("超限结论应提到斜率: %q", last.Verdict())
	}
}

func TestAnalyzeRawHasSpan(t *testing.T) {
	s, err := Analyze("calm", api.Options{Size: 64, Seed: 3, Octaves: 4, Lacunarity: 2, Gain: 0.5, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	if s.Field.Span <= 0 {
		t.Fatalf("原始场极差 %.4f 应为正", s.Field.Span)
	}
	// 归一化后梯度被除以 Span，倾角应按 atan(tan/R) 缩小。
	norm, err := api.Generate(s.Options)
	if err != nil {
		t.Fatal(err)
	}
	if norm.Span != s.Field.Span {
		t.Fatalf("归一化记录极差 %.6f 与原始 %.6f 不一致", norm.Span, s.Field.Span)
	}
}

func TestWriteArtifacts(t *testing.T) {
	s, err := Analyze("calm", api.Options{Size: 48, Seed: 7, Octaves: 3, Lacunarity: 2, Gain: 0.5, Frequency: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := s.Write(dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"bands.csv", "tilt.pgm", "tilt-hist.svg"} {
		st, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("%s 未生成: %v", name, err)
		}
		if st.Size() == 0 {
			t.Fatalf("%s 为空", name)
		}
	}
	data, err := os.ReadFile(filepath.Join(dir, "tilt.pgm"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(data), "P5\n48 48\n255\n") {
		t.Fatalf("PGM 头部不符: %q", data[:15])
	}
	if len(data) != len("P5\n48 48\n255\n")+48*48 {
		t.Fatalf("PGM 长度 %d，应为 %d", len(data), len("P5\n48 48\n255\n")+48*48)
	}
}

func TestMarkdownReportsSlopeError(t *testing.T) {
	s, err := Analyze("rugged", api.Options{Size: 64, Seed: 7, Octaves: 6, Lacunarity: 2, Gain: 0.62, Frequency: 0.05})
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	s.WriteMarkdown(&b)
	out := b.String()
	if !strings.Contains(out, "斜率估计误差") {
		t.Fatal("超限报告未说明斜率估计误差")
	}
	if !strings.Contains(out, "Nyquist") {
		t.Fatal("超限报告未解释 Nyquist")
	}
}
