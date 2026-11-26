package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeaderUsesRealNewlines(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	prefixLen := 16
	if len(got) < prefixLen {
		prefixLen = len(got)
	}
	if !strings.HasPrefix(string(got), "P5\n2 2\n255\n") {
		t.Fatalf("PGM 头不是合法二进制 P5 头: %q", got[:prefixLen])
	}
	if len(got) != len("P5\n2 2\n255\n")+4 {
		t.Fatalf("PGM 数据长度错误: %d", len(got))
	}
}
