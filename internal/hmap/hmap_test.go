package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeaderHasRealNewlines(t *testing.T) {
	pgm := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	if !strings.HasPrefix(string(pgm), "P5\n2 2\n255\n") {
		t.Fatalf("PGM 头不是合法换行格式: %q", pgm[:16])
	}
	if len(pgm) != len("P5\n2 2\n255\n")+4 {
		t.Fatalf("PGM 数据长度错误: %d", len(pgm))
	}
}
