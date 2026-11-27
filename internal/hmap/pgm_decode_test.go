package hmap

import "testing"

func TestPGMRoundtrip(t *testing.T) {
	n := 8
	v := make([]float64, n*n)
	for i := range v {
		v[i] = float64(i%256) / 255.0
	}
	got, data, err := DecodePGM(EncodePGM(n, v))
	if err != nil {
		t.Fatal(err)
	}
	if got != n || len(data) != len(v) {
		t.Fatalf("尺寸不符: %d %d", got, len(data))
	}
	// 八位量化往返误差不超过一个量化级。
	for i := range v {
		if d := data[i] - v[i]; d > 1.0/255+1e-12 || d < -1.0/255-1e-12 {
			t.Fatalf("索引 %d 往返误差过大: %g", i, d)
		}
	}
}

func TestDecodePGMRejectsBadMagic(t *testing.T) {
	if _, _, err := DecodePGM([]byte("P2\n1 1\n255\n\x00")); err == nil {
		t.Fatal("应拒绝非 P5 文件")
	}
}
