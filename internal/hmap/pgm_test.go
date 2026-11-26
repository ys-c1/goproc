package hmap

import (
	"bytes"
	"testing"
)

func TestPGMRoundTrip(t *testing.T) {
	size := 4
	v := []float64{0, 0.25, 0.5, 1, 0.1, 0.2, 0.3, 0.4, 0, 0, 0, 0, 0.7, 0.8, 0.9, 1}
	raw := EncodePGM(size, v)
	// 头部必须是真实换行，而不是字面反斜杠 n。
	if !bytes.HasPrefix(raw, []byte("P5\n4 4\n255\n")) {
		t.Fatalf("PGM 头部非法: %q", raw[:12])
	}
	gotSize, got, err := DecodePGM(raw)
	if err != nil {
		t.Fatalf("DecodePGM 失败: %v", err)
	}
	if gotSize != size {
		t.Fatalf("尺寸不符: %d", gotSize)
	}
	if len(got) != len(v) {
		t.Fatalf("数据长度不符: %d", len(got))
	}
	// 量化后再归一化，允许 1/255 量级误差。
	for i := range v {
		want := float64(int(v[i]*255+0.5)) / 255
		if abs(got[i]-want) > 1e-9 {
			t.Fatalf("第 %d 个像素 got=%v want=%v", i, got[i], want)
		}
	}
}

func TestDecodePGMRejectsBackslashN(t *testing.T) {
	// 旧版本缺陷：头部写成字面 \n 两字节，必须被拒绝。
	bad := []byte("P5\\n2 2\\n255\\n\x00\x00\x00\x00")
	if _, _, err := DecodePGM(bad); err == nil {
		t.Fatal("字面反斜杠 n 的非法 PGM 应被拒绝")
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
