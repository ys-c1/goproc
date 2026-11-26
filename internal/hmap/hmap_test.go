package hmap

import (
	"bytes"
	"testing"
)

// PGM 头必须是合法的 P5：魔数、宽高、满量程各自以一个真正的换行字节分隔。
func TestEncodePGMHeader(t *testing.T) {
	v := make([]float64, 4)
	v[0], v[1], v[2], v[3] = 0, 1.0/255, 0.5, 1
	got := EncodePGM(2, v)
	wantHead := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(got, wantHead) {
		t.Fatalf("头字节 %q，应为 %q", got[:len(wantHead)], wantHead)
	}
	if len(got) != len(wantHead)+4 {
		t.Fatalf("总长 %d，应为 %d", len(got), len(wantHead)+4)
	}
	body := got[len(wantHead):]
	for i, w := range []byte{0, 1, 128, 255} {
		if body[i] != w {
			t.Fatalf("像素 %d = %d，应为 %d", i, body[i], w)
		}
	}
	if bytes.Contains(got[:len(wantHead)], []byte(`\n`)) {
		t.Fatal("头部含字面反斜杠 n，不是合法 P5")
	}
}

// 归一化把任意范围拉伸到零到一，常数输入全置零。
func TestNormalize(t *testing.T) {
	v := []float64{-2, 0, 2, 4}
	Normalize(v)
	for i, want := range []float64{0, 1.0 / 3, 2.0 / 3, 1} {
		if diff := v[i] - want; diff > 1e-12 || diff < -1e-12 {
			t.Fatalf("v[%d]=%v，应为 %v", i, v[i], want)
		}
	}
	c := []float64{3, 3, 3}
	Normalize(c)
	for i := range c {
		if c[i] != 0 {
			t.Fatalf("常数输入应全为零，v[%d]=%v", i, c[i])
		}
	}
	Normalize(nil)
}
