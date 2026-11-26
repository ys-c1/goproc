package hmap

import "testing"

// PGM 头必须是真实的换行分隔。此前头部写成字面量 `\n`（两个字符 '\' 'n'），
// 导致任何标准读取器（PIL、netpbm）都拒绝加载。
func TestEncodePGMHeaderIsValid(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.5, 1, 0.25})
	want := []byte("P5\n2 2\n255\n")
	if len(got) < len(want) {
		t.Fatalf("输出过短: %d 字节", len(got))
	}
	if string(got[:len(want)]) != string(want) {
		t.Fatalf("头部不符:\n got %q\nwant %q", got[:len(want)], want)
	}
	if len(got) != len(want)+4 {
		t.Fatalf("像素字节数应为 4，实得 %d", len(got)-len(want))
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("归一化错误: %v", v)
	}
}
