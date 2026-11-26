// Package hmap 的测试：归一化、八位量化与 PGM 字节布局。
package hmap

import (
	"bytes"
	"testing"
)

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("归一化结果 %v", v)
	}
	flat := []float64{3, 3, 3}
	Normalize(flat)
	for _, x := range flat {
		if x != 0 {
			t.Fatalf("常量数组应归零，得到 %v", flat)
		}
	}
}

func TestGray8(t *testing.T) {
	cases := []struct {
		in   float64
		want uint8
	}{
		{-1, 0}, {0, 0}, {0.5, 128}, {1, 255}, {2, 255}, {0.4999, 127},
	}
	for _, c := range cases {
		if got := Gray8(c.in); got != c.want {
			t.Fatalf("Gray8(%v)=%d want %d", c.in, got, c.want)
		}
	}
}

// TestEncodePGMHeader PGM 文件头必须是真实换行分隔：既有的字面量 "\\n"
// 会让标准解码器把尺寸读错，属于导入时就存在的缺陷。
func TestEncodePGMHeader(t *testing.T) {
	data := EncodePGM(2, []float64{0, 0.5, 1, 1})
	want := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(data, want) {
		t.Fatalf("文件头 %q，应为 %q", data[:len(want)], want)
	}
	if len(data) != len(want)+4 {
		t.Fatalf("总长度 %d，应为 %d", len(data), len(want)+4)
	}
	if got := data[len(want):]; !bytes.Equal(got, []byte{0, 128, 255, 255}) {
		t.Fatalf("像素 %v", got)
	}
	if bytes.Contains(data, []byte("\\n")) {
		t.Fatalf("文件头不应含字面量反斜杠 n")
	}
}
