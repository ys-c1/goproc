package api

import (
	"errors"
	"math"
	"testing"
)

// TestDivisibleIsRemainderZero 把整除判定钉在两数取余上。
func TestDivisibleIsRemainderZero(t *testing.T) {
	cases := []struct {
		a, b int
		want bool
	}{
		{256, 256, true},
		{512, 256, true},
		{1024, 512, true},
		{270, 256, false},
		{300, 256, false},
		{384, 256, false},
		{500, 256, false},
		{0, 256, true},
		{7, 0, false},
	}
	for _, c := range cases {
		if got := Divisible(c.a, c.b); got != c.want {
			t.Errorf("Divisible(%d,%d)=%v want %v", c.a, c.b, got, c.want)
		}
		if c.b != 0 && c.want != (c.a%c.b == 0) {
			t.Errorf("Divisible(%d,%d) 与取余结论不一致", c.a, c.b)
		}
	}
}

// TestPhaseIsFrequencyTimesSize 钉住相位换算：频率 × 边长。
func TestPhaseIsFrequencyTimesSize(t *testing.T) {
	o := DefaultOptions()
	o.TileCells = 256
	o = o.Resolve()
	if got, want := o.Phase(0), 256.0; got != want {
		t.Fatalf("Phase(0)=%v want %v", got, want)
	}
	// lac=2 时相位逐层翻倍
	for i := 0; i < o.Octaves; i++ {
		want := 256.0 * math.Pow(2, float64(i))
		if got := o.Phase(i); got != want {
			t.Errorf("Phase(%d)=%v want %v", i, got, want)
		}
	}
	// 频率与边长乘积的独立写法
	f := 0.02
	o2 := DefaultOptions()
	o2.Frequency = f
	if got, want := o2.Phase(1), f*2*128; math.Abs(got-want) > 1e-12 {
		t.Errorf("Phase(1)=%v want %v", got, want)
	}
}

// TestTileableCondition 验证充要条件：每一层相位都要被置换表周期整除。
func TestTileableCondition(t *testing.T) {
	ok := DefaultOptions()
	ok.TileCells = 512
	ok.Period = 256
	if !ok.Tileable() {
		t.Fatal("512/256 的各层都应整除")
	}
	bad := DefaultOptions()
	bad.TileCells = 300
	bad.Period = 256
	if bad.Tileable() {
		t.Fatal("300 对 256 取余 44，不应判为可平铺")
	}
	if _, err := Generate(bad); !errors.Is(err, ErrNotTileable) {
		t.Fatalf("生成不可平铺参数应报 ErrNotTileable，得到 %v", err)
	}
}

// TestRemainderDrivesTileability 余数非零的层必然不可平铺，且报错带出余数。
func TestRemainderDrivesTileability(t *testing.T) {
	for _, cells := range []int{257, 270, 300, 384, 500} {
		o := DefaultOptions()
		o.TileCells = cells
		o.Period = 256
		r := o.Resolve()
		if rem, ok := r.RemainderCells(0); !ok || rem != cells%256 {
			t.Fatalf("cells=%d RemainderCells=%d,%v want %d", cells, rem, ok, cells%256)
		}
		_, err := Generate(o)
		if err == nil {
			t.Fatalf("cells=%d 应拒绝", cells)
		}
	}
}

// TestTileableIsSeamless 构造上严格整除时，边界外的应然值与首像素逐位相等。
func TestTileableIsSeamless(t *testing.T) {
	opt := DefaultOptions()
	opt.Seed = 7
	opt.TileCells = 512
	opt.Period = 256
	h, err := Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	n := h.Size
	for k := 0; k < n; k++ {
		if d := math.Abs(h.Sample(float64(n), float64(k)) - h.At(0, k)); d > 1e-12 {
			t.Fatalf("x 接缝第 %d 行差 %v", k, d)
		}
		if d := math.Abs(h.Sample(float64(k), float64(n)) - h.At(k, 0)); d > 1e-12 {
			t.Fatalf("y 接缝第 %d 列差 %v", k, d)
		}
	}
}

// TestRepeatEmbeds 拼接图在每个接缝两侧保持连续。
func TestRepeatEmbeds(t *testing.T) {
	opt := DefaultOptions()
	opt.Seed = 7
	opt.TileCells = 512
	h, _ := Generate(opt)
	big := h.Repeat(2)
	side := 2 * h.Size
	// 拼好的图内部相邻差的最大值，应与单张贴图内部基线同量级
	worst := 0.0
	for y := 0; y < side; y++ {
		for x := 1; x < side; x++ {
			d := math.Abs(big[y*side+x] - big[y*side+x-1])
			if d > worst {
				worst = d
			}
		}
	}
	// 拼接不应引入比单图内部更陡的差
	inner := 0.0
	for y := 0; y < h.Size; y++ {
		for x := 1; x < h.Size; x++ {
			d := math.Abs(h.At(x, y) - h.At(x-1, y))
			if d > inner {
				inner = d
			}
		}
	}
	if worst > inner+1e-12 {
		t.Fatalf("拼接后最大相邻差 %v 超出单图内部 %v", worst, inner)
	}
}
