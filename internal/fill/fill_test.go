package fill

import (
	"testing"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

// xorshiftGrid 造确定性伪随机高度场。
func xorshiftGrid(size int, seed uint64) *Grid {
	s := seed | 1
	g := make([]int, size*size)
	for i := range g {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		g[i] = int(s % 256)
	}
	return &Grid{Size: size, G: g}
}

// checkConsistency 核验：优先泛滥与迭代松弛逐格一致；
// 每片湖水位等于出口格最小高度（命题1）；
// 由溢流高度预判的像素数与最大水深和填洼结果一致（命题2）。
func checkConsistency(t *testing.T, g *Grid) {
	t.Helper()
	wPri := PriorityFlood(g)
	wIter := IterativeFill(g)
	for i := range wPri {
		if wPri[i] != wIter[i] {
			t.Fatalf("格(%d,%d) 优先泛滥水位=%d 迭代水位=%d",
				i%g.Size, i/g.Size, wPri[i], wIter[i])
		}
	}
	lakes, err := AnalyzeLakes(g, wPri)
	if err != nil {
		t.Fatal(err)
	}
	for _, lk := range lakes {
		if lk.OutletMin != lk.Level {
			t.Errorf("湖%d 水位=%d 出口最小高度=%d 应相等", lk.Label, lk.Level, lk.OutletMin)
		}
		if len(lk.Outlets) == 0 {
			t.Errorf("湖%d 没有高度等于水位的出口格", lk.Label)
		}
		cells, minGround := PredictLake(g, lk.Cells[0], lk.Level)
		if len(cells) != lk.Pixels {
			t.Errorf("湖%d 预判像素=%d 填洼像素=%d", lk.Label, len(cells), lk.Pixels)
		}
		if lk.Level-minGround != lk.MaxDepth {
			t.Errorf("湖%d 预判最大水深=%d 填洼最大水深=%d",
				lk.Label, lk.Level-minGround, lk.MaxDepth)
		}
		inPred := make([]bool, g.Size*g.Size)
		for _, i := range cells {
			inPred[i] = true
		}
		for _, i := range lk.Cells {
			if !inPred[i] {
				t.Errorf("湖%d 格(%d,%d) 在填洼水面内但不在预判分量内", lk.Label, i%g.Size, i/g.Size)
			}
		}
	}
}

func TestSinglePit(t *testing.T) {
	// 5x5，外圈 10，内 3x3 为 5，中心 1：整片填到 10。
	g := &Grid{Size: 5, G: make([]int, 25)}
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			v := 10
			if x >= 1 && x <= 3 && y >= 1 && y <= 3 {
				v = 5
			}
			g.G[y*5+x] = v
		}
	}
	g.G[2*5+2] = 1
	w := PriorityFlood(g)
	if w[2*5+2] != 10 {
		t.Fatalf("中心水位=%d 应为 10", w[2*5+2])
	}
	lakes, err := AnalyzeLakes(g, w)
	if err != nil {
		t.Fatal(err)
	}
	if len(lakes) != 1 {
		t.Fatalf("湖数=%d 应为 1", len(lakes))
	}
	lk := lakes[0]
	if lk.Level != 10 || lk.Pixels != 9 || lk.MaxDepth != 9 || lk.OutletMin != 10 {
		t.Fatalf("湖属性错: %+v", lk)
	}
	checkConsistency(t, g)
}

func TestDiagonalSpill(t *testing.T) {
	// 洼地经对角走廊泄流：八邻域下溢流高度为 7，不应填到 10。
	// 外圈 10；内 3x3 为 5；右下角沿对角开一条高度 7 的缝。
	g := &Grid{Size: 7, G: make([]int, 49)}
	for y := 0; y < 7; y++ {
		for x := 0; x < 7; x++ {
			v := 10
			if x >= 1 && x <= 3 && y >= 1 && y <= 3 {
				v = 5
			}
			g.G[y*7+x] = v
		}
	}
	g.G[4*7+4] = 7
	g.G[5*7+5] = 7
	g.G[6*7+6] = 7
	w := PriorityFlood(g)
	if w[2*7+2] != 7 {
		t.Fatalf("对角溢流时水位=%d 应为 7", w[2*7+2])
	}
	checkConsistency(t, g)
}

func TestRandomGrids(t *testing.T) {
	for _, size := range []int{8, 17, 32} {
		for seed := uint64(1); seed <= 6; seed++ {
			checkConsistency(t, xorshiftGrid(size, seed*7919))
		}
	}
}

func TestFBMTerrain(t *testing.T) {
	// 真实分形地形上核验两条命题。
	for seed := uint64(1); seed <= 4; seed++ {
		src := noise.New(seed)
		size := 64
		data := make([]float64, size*size)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				data[y*size+x] = fbm.Eval(src, float64(x)*0.02, float64(y)*0.02, 4, 2, 0.5)
			}
		}
		hmap.Normalize(data)
		checkConsistency(t, &Grid{Size: size, G: hmap.ToGray8(data)})
	}
}
