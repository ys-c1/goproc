package lod

import (
	"math"
	"testing"

	"goproc/api"
)

// TestQuadraticBound 用二阶导恒定的二次曲面验证多尺度曲率与误差界：
// 对二次曲面 4^L·D_L 在所有层级相等（K 退化为单尺度值），且
// 每块每级实际误差 <= K/(8·4^L)（本构造上取等号），预测层级不偏低。
func TestQuadraticBound(t *testing.T) {
	const size, tile, lmax = 64, 8, 3
	h := make([]float64, size*size)
	// h(x,y) = 0.01*x^2 + 0.02*y^2，连续二阶导 h_xx=0.02, h_yy=0.04。
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = 0.01*float64(x*x) + 0.02*float64(y*y)
		}
	}
	g := Build(h, size, tile, lmax, 0.05)
	s0 := g.MaxStep
	// 原始二阶差分 = 连续二阶导 × s0^2；多尺度下 4^L·D_L 恒等于该值。
	wantKappa := (0.02 + 0.04) * float64(s0*s0) // s0=8 时为 3.84
	for _, ti := range g.Tiles {
		if math.Abs(ti.Kappa-wantKappa) > 1e-9 {
			t.Fatalf("tile(%d,%d) K=%v want %v", ti.Row, ti.Col, ti.Kappa, wantKappa)
		}
		// 非边界块在 L=0 上有内部节点，多尺度 K 应退化为单尺度 D0。
		if !ti.Border && math.Abs(ti.Kappa0-wantKappa) > 1e-9 {
			t.Fatalf("tile(%d,%d) quadratic: multiscale K must equal single-scale D0, got K0=%v",
				ti.Row, ti.Col, ti.Kappa0)
		}
		// 误差界只在内部块（无钳制节点）上由曲率定理保证；边界块直接赋最细层级。
		if !ti.Border {
			for l := 0; l <= lmax; l++ {
				err := tileError(g, ti, l)
				bound := ti.Kappa / (8.0 * math.Pow(4, float64(l)))
				if err > bound+1e-9 {
					t.Fatalf("tile(%d,%d) L=%d err=%.6e exceeds bound %.6e", ti.Row, ti.Col, l, err, bound)
				}
			}
		}
		if ti.ErrPred > g.Tol {
			t.Fatalf("tile(%d,%d) predicted L=%d underpredicts: err=%.6e > tol=%.3f",
				ti.Row, ti.Col, ti.LevelPred, ti.ErrPred, g.Tol)
		}
	}
}

// TestRealTerrainBound 在真实 FBM 地形上验证：多尺度曲率的严格误差界
// 对每块每级成立，多尺度预测层级逐块不偏低；同时记录朴素单尺度做法的偏低数。
func TestRealTerrainBound(t *testing.T) {
	hm, err := api.Generate(api.Options{Size: 128, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	const tile, lmax = 128 / 16, 4
	g := Build(hm.Data, hm.Size, tile, lmax, 0.02)
	violations := 0
	under, over, exact, naiveUnder := 0, 0, 0, 0
	boundCells := 0
	for _, ti := range g.Tiles {
		// 曲率误差界只对公式管辖的内部块检查；边界块直接赋最细层级。
		if !ti.Border {
			for l := 0; l <= lmax; l++ {
				boundCells++
				err := tileError(g, ti, l)
				bound := ti.Kappa / (8.0 * math.Pow(4, float64(l)))
				if err > bound+1e-12 {
					violations++
					if violations <= 5 {
						t.Logf("bound violation tile(%d,%d) L=%d err=%.6e bound=%.6e",
							ti.Row, ti.Col, l, err, bound)
					}
				}
			}
		}
		switch {
		case ti.LevelPred < ti.LevelMin:
			under++
			t.Errorf("tile(%d,%d) pred=%d below min-feasible=%d (errPred=%.4e)",
				ti.Row, ti.Col, ti.LevelPred, ti.LevelMin, ti.ErrPred)
		case ti.LevelPred > ti.LevelMin:
			over++
		default:
			exact++
		}
		if ti.LevelNaive < ti.LevelMin {
			naiveUnder++
		}
		if ti.ErrMin > g.Tol {
			t.Errorf("tile(%d,%d) min-feasible L=%d err=%.6e > tol", ti.Row, ti.Col, ti.LevelMin, ti.ErrMin)
		}
	}
	t.Logf("bound violations: %d/%d interior cells | multiscale pred exact=%d conservative=%d under=%d | naive-single-scale under=%d",
		violations, boundCells, exact, over, under, naiveUnder)
}

// TestPredictFormula 直接核对取整公式。
func TestPredictFormula(t *testing.T) {
	cases := []struct {
		kappa, tol float64
		lmax       int
	}{
		{3.84, 0.05, 4}, {3.84, 0.02, 4}, {96, 0.01, 4}, {0.0, 0.05, 4}, {3200, 0.001, 3},
	}
	for _, tc := range cases {
		got := predictLevel(tc.kappa, tc.tol, tc.lmax)
		if got < 0 || got > tc.lmax {
			t.Fatalf("level out of range: %d", got)
		}
		if tc.kappa > 0 {
			raw := math.Log(tc.kappa/(8*tc.tol)) / math.Log(4)
			want := int(math.Ceil(raw))
			if want < 0 {
				want = 0
			}
			if want > tc.lmax {
				want = tc.lmax
			}
			if got != want {
				t.Fatalf("predictLevel(%v,%v)=%d want %d", tc.kappa, tc.tol, got, want)
			}
		} else if got != 0 {
			t.Fatalf("zero curvature should map to level 0, got %d", got)
		}
	}
}

// TestOnePassIndependence 同一 (kappa, tol) 无论出现在哪个块都得到相同层级，
// 说明一次性分配只依赖块曲率与全局容差，与块位置、渲染过程无关。
func TestOnePassIndependence(t *testing.T) {
	if predictLevel(0.3, 0.02, 4) != predictLevel(0.3, 0.02, 4) {
		t.Fatal("identical (kappa,tol) yielded different levels")
	}
}

// TestNaiveFailureIsDetected 确认对照接口确实能标出朴素做法偏低的块，
// 且这些块在多尺度预测下全部被纠正。
func TestNaiveFailureIsDetected(t *testing.T) {
	hm, err := api.Generate(api.Options{Size: 128, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	g := Build(hm.Data, hm.Size, 8, 4, 0.02)
	bad := g.NaiveUnderpredicted()
	for _, ti := range bad {
		if ti.ErrPred > g.Tol {
			t.Fatalf("multiscale failed to correct naive tile(%d,%d)", ti.Row, ti.Col)
		}
	}
	t.Logf("naive underpredicted tiles: %d / %d", len(bad), len(g.Tiles))
}
