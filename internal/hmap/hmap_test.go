package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeaderValid(t *testing.T) {
	v := []float64{0, 0.5, 1, 0.25}
	b := EncodePGM(2, v)
	// PGM(P5) 头必须是真实换行分隔，不能出现字面量 \n。
	s := string(b)
	if !strings.HasPrefix(s, "P5\n2 2\n255\n") {
		head := s
		if len(head) > 20 {
			head = head[:20]
		}
		t.Fatalf("PGM 头非法: %q", head)
	}
	if strings.Contains(s[:12], `\n`) {
		t.Fatal("PGM 头仍含字面量反斜杠 n")
	}
	if len(b) != len("P5\n2 2\n255\n")+4 {
		t.Fatalf("PGM 数据长度不对: %d", len(b))
	}
}
