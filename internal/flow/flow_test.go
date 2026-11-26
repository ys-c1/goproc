package flow

import (
	"bytes"
	"testing"
)

// 手工构造高度数组，h[y*size+x]。
func grid(size int, f func(x, y int) float64) []float64 {
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = f(x, y)
		}
	}
	return h
}

func TestDirectCardinal(t *testing.T) {
	// 中心 10，正东 1，其余 9；正东落差最大。
	h := grid(3, func(x, y int) float64 {
		if x == 1 && y == 1 {
			return 10
		}
		if x == 2 && y == 1 {
			return 1
		}
		return 9
	})
	if got := Direct(h, 3, 1, 1); got != E {
		t.Fatalf("got %d want E(0)", got)
	}
}

func TestDirectDiagonalDistanceNormalized(t *testing.T) {
	// 正东落差 1.4（距离 1，坡度 1.4），东南落差 2（距离 sqrt2，
	// 坡度约 1.4142）；不归一化会选错，归一化后选东南。
	h := grid(3, func(x, y int) float64 {
		if x == 1 && y == 1 {
			return 10
		}
		switch {
		case x == 2 && y == 1:
			return 8.6
		case x == 2 && y == 2:
			return 8
		default:
			return 9.9
		}
	})
	if got := Direct(h, 3, 1, 1); got != SE {
		t.Fatalf("got %d want SE(1)", got)
	}
}

func TestDirectTieBreak(t *testing.T) {
	// 正东与正南同落差（对称构造），并列时取编号最小的 E。
	h := grid(3, func(x, y int) float64 {
		if x == 1 && y == 1 {
			return 10
		}
		if (x == 2 && y == 1) || (x == 1 && y == 2) {
			return 5
		}
		return 10
	})
	if got := Direct(h, 3, 1, 1); got != E {
		t.Fatalf("got %d want E(0) on tie", got)
	}
}

func TestDirectTieCardinalBeforeDiagonal(t *testing.T) {
	// 两个斜方向 SE(1) 与 SW(3) 同落差、同距离 => 精确同坡度，
	// 并列时取编号最小的 SE。正/斜方向的坡度因含 sqrt2 在浮点下
	// 无法精确相等，其固定次序由编号约定保证，见 TestDirectTieBreak。
	h := grid(3, func(x, y int) float64 {
		if x == 1 && y == 1 {
			return 10
		}
		if (x == 2 && y == 2) || (x == 0 && y == 2) {
			return 5
		}
		return 10
	})
	if got := Direct(h, 3, 1, 1); got != SE {
		t.Fatalf("got %d want SE(1), lower dir index must win equal-slope tie", got)
	}
}

func TestSinkAndFlat(t *testing.T) {
	// 全图同高：每格最大坡度为零，全部判 Sink。
	h := grid(4, func(x, y int) float64 { return 7 })
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if got := Direct(h, 4, x, y); got != Sink {
				t.Fatalf("flat cell (%d,%d) got %d want Sink", x, y, got)
			}
		}
	}
	if got := len(FlatCells(h, 4)); got != 16 {
		t.Fatalf("FlatCells=%d want 16", got)
	}
	// 局部高点：内陆中心无更低邻格也是 Sink。
	h2 := grid(3, func(x, y int) float64 {
		if x == 1 && y == 1 {
			return 0
		}
		return 5
	})
	if got := Direct(h2, 3, 1, 1); got != Sink {
		t.Fatalf("closed low got %d want Sink", got)
	}
}

func TestBoundaryOutlet(t *testing.T) {
	// 单调向东倾斜：所有追踪应终止于东边界 x=size-1。
	size := 6
	h := grid(size, func(x, y int) float64 { return float64(size - x) })
	m := Build(h, size)
	term := m.ResolveAll()
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			got := term[y*size+x]
			// 首达边界即停：边界格终点是自己；内陆格向东走到同行了东边界。
			wantX, wantY := size-1, y
			if x == 0 || y == 0 || y == size-1 {
				wantX, wantY = x, y
			}
			if got.Kind != TermOutlet || got.X != wantX || got.Y != wantY {
				t.Fatalf("(%d,%d) terminus=%+v want outlet (%d,%d)", x, y, got, wantX, wantY)
			}
		}
	}
	// 东向倾斜下东边界 size 格都被内陆流路命中，加上其余三边自身
	// 也是出口；全部边界格都应出现在出口清单中。
	wantOutlets := size*size - (size-2)*(size-2)
	if got := len(Outlets(term, size)); got != wantOutlets {
		t.Fatalf("outlets=%d want %d", got, wantOutlets)
	}
}

func TestResolveMatchesNaive(t *testing.T) {
	// 非对称地形，保证两种追踪逐格一致。
	size := 12
	h := grid(size, func(x, y int) float64 {
		v := float64((x*37+y*53)%97) * 0.013
		// 中部放一个封闭洼地
		if x >= 4 && x <= 6 && y >= 4 && y <= 6 {
			v -= 5
		}
		return v
	})
	m := Build(h, size)
	term := m.ResolveAll()
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			want := NaiveTrace(h, size, x, y)
			got := term[y*size+x]
			if got != want {
				t.Fatalf("(%d,%d) resolve=%+v naive=%+v", x, y, got, want)
			}
		}
	}
}

func TestNoCycles(t *testing.T) {
	size := 10
	h := grid(size, func(x, y int) float64 {
		return float64((x*7+y*13)%31) + 0.5*float64((x+y)%3)
	})
	m := Build(h, size)
	// 沿方向指针行走，高度必须严格下降；Sink 或边界处停止。
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			cx, cy := x, y
			prev := h[cy*size+cx]
			steps := 0
			for {
				if cx == 0 || cy == 0 || cx == size-1 || cy == size-1 {
					break
				}
				d := m.At(cx, cy)
				if d == Sink {
					break
				}
				nx, ny := Neighbor(cx, cy, d)
				next := h[ny*size+nx]
				if !(next < prev) {
					t.Fatalf("non-descending step at (%d,%d): %v -> %v", cx, cy, prev, next)
				}
				prev = next
				cx, cy = nx, ny
				steps++
				if steps > size*size {
					t.Fatalf("cycle suspected from (%d,%d)", x, y)
				}
			}
		}
	}
}

func TestPGMRoundTrip(t *testing.T) {
	size := 5
	h := grid(size, func(x, y int) float64 {
		return float64((x*11 + y*17) % 23)
	})
	m := Build(h, size)
	raw := m.EncodePGM()
	// 校验 P5 头并解码像素区。
	if !bytes.HasPrefix(raw, []byte("P5\n")) {
		t.Fatal("not P5")
	}
	i := bytes.Index(raw, []byte("255\n"))
	if i < 0 {
		t.Fatal("missing maxval")
	}
	pixels := raw[i+4:]
	if len(pixels) != size*size {
		t.Fatalf("pixel bytes=%d want %d", len(pixels), size*size)
	}
	for id, d := range m.Dirs {
		if got := FromGray(pixels[id]); got != d {
			t.Fatalf("cell %d round-trip got %d want %d", id, got, d)
		}
	}
	// Sink 灰度必须为 0。
	if Sink.Gray() != 0 {
		t.Fatal("sink gray must be 0")
	}
}
