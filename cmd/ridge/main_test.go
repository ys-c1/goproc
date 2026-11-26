package main

import (
	"bytes"
	"strings"
	"testing"

	"goproc/api"
	"goproc/internal/ridge"
)

// 掩码 PGM 是合法的 P5（真换行分隔的头），像素值恰为 0 或 maxval。
func TestMaskPGM(t *testing.T) {
	f := analyze(t, 128)
	b := maskPGM(f.Mask)
	head := []byte("P5\n128 128\n1\n")
	if !bytes.HasPrefix(b, head) {
		t.Fatalf("头 %q", b[:len(head)])
	}
	body := b[len(head):]
	if len(body) != 128*128 {
		t.Fatalf("体长 %d", len(body))
	}
	on := 0
	for _, v := range body {
		if v > 1 {
			t.Fatalf("像素值 %d 越界", v)
		}
		if v == 1 {
			on++
		}
	}
	if on != f.Mask.Count() {
		t.Fatalf("白像素 %d ≠ 掩码像素 %d", on, f.Mask.Count())
	}
}

// 连通段着色图：背景为零，脊线像素非零。
func TestComponentsPGM(t *testing.T) {
	f := analyze(t, 128)
	b := componentsPGM(f)
	head := []byte("P5\n128 128\n255\n")
	body := b[len(head):]
	for i, v := range body {
		if (v != 0) != f.Mask.Data[i] {
			t.Fatalf("像素 %d 着色 %d 与掩码 %v 不符", i, v, f.Mask.Data[i])
		}
	}
}

// 报告的表格区每个画幅恰占一行，且「偏差超过一成」的说明与逐行数据一致。
func TestReportRows(t *testing.T) {
	f128 := analyze(t, 128)
	f256 := analyze(t, 256)
	opt := options{sizes: []int{128, 256}, seed: 1, octaves: 4, gain: 0.5, freq: 0.02}
	report := renderReport([]ridge.Frame{f128, f256}, opt)

	var rows []string
	for _, line := range strings.Split(report, "\n") {
		if strings.HasPrefix(line, "128x128") || strings.HasPrefix(line, "256x256") {
			rows = append(rows, line)
		}
	}
	if len(rows) != 2 {
		t.Fatalf("表格应有两行，实际 %d:\n%s", len(rows), report)
	}
	for _, r := range rows {
		if len(strings.Fields(r)) != 8 {
			t.Fatalf("行字段数应为 8（画幅+6 个数+分水偏差）：%q", r)
		}
	}
	if !strings.Contains(report, "偏差超过一成的画幅：无") {
		t.Fatalf("默认参数下应报告无超限画幅：\n%s", report)
	}
}

// 若某画幅偏差超过一成，报告必须点名该画幅。
func TestReportFlagsDeviation(t *testing.T) {
	// 构造一个预判严重偏低的帧：曲率脊实测远大于预判。
	f := analyze(t, 128)
	f.KMin.Count = float64(f.KMin.Measured) * 1.5
	opt := options{sizes: []int{128}, seed: 1, octaves: 4, gain: 0.5, freq: 0.02}
	report := renderReport([]ridge.Frame{f}, opt)
	if !strings.Contains(report, "128x128（曲率脊") {
		t.Fatalf("超限画幅应被点名：\n%s", report)
	}
}

// parseSizes 接受逗号分隔并校验范围。
func TestParseSizes(t *testing.T) {
	got, err := parseSizes("128, 256 ,512")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 128 || got[1] != 256 || got[2] != 512 {
		t.Fatalf("got %v", got)
	}
	for _, bad := range []string{"", "abc", "8", "5000"} {
		if _, err := parseSizes(bad); err == nil {
			t.Fatalf("%q 应报错", bad)
		}
	}
}

// parseArgs 未注册的参数在解析前就存在，--sizes 能被识别。
func TestParseArgs(t *testing.T) {
	opt, err := parseArgs([]string{"--sizes", "64,128", "--seed", "3", "--out", "/tmp/x"})
	if err != nil {
		t.Fatal(err)
	}
	if len(opt.sizes) != 2 || opt.seed != 3 || opt.outDir != "/tmp/x" {
		t.Fatalf("got %+v", opt)
	}
	if _, err := parseArgs([]string{"--nope"}); err == nil {
		t.Fatal("未知参数应报错")
	}
}

func analyze(t *testing.T, n int) ridge.Frame {
	t.Helper()
	h, err := api.Generate(api.Options{Size: n, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 1, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	return ridge.AnalyzeFrame(n, h.Data)
}
