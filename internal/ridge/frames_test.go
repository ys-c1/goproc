package ridge

import (
	"testing"

	"goproc/api"
)

func frameFromAPI(t *testing.T, seed uint64, size int) *Field {
	t.Helper()
	h, err := api.Generate(api.Options{Size: size, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	f, err := NewField(Frame{N: h.Size, F: h.Data})
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// 两个存档画幅上的全部计数关系：引理 1、Morse 恒等式、上升脊 2P-2、
// 分量含 Δf 极小点、脊线度 ≤2。
func TestArchivedFrames(t *testing.T) {
	for _, seed := range []uint64{7, 42} {
		f := frameFromAPI(t, seed, 128)
		mask := f.Mask()
		lap := f.Laplacian()
		for i := range mask {
			if mask[i] != (lap[i] < 0) {
				t.Fatalf("seed %d: Mask 与 Δf<0 在 %d 不一致", seed, i)
			}
		}

		maxima := f.LocalMaxima()
		rep := f.Analyze("frame", mask, maxima)
		mc := rep.Morse
		if mc.Components != rep.Actual.Total {
			t.Fatalf("seed %d: morse=%d actual=%d", seed, mc.Components, rep.Actual.Total)
		}
		if mc.Births-mc.MergeLoss != mc.Components {
			t.Fatalf("seed %d: 段数分解不闭合", seed)
		}
		if mc.Births != f.NegativeLaplacianMinima() {
			t.Fatalf("seed %d: births=%d negLapMin=%d", seed, mc.Births, f.NegativeLaplacianMinima())
		}
		if rep.Actual.Interior > rep.Prediction.InteriorMinima {
			t.Fatalf("seed %d: 内部段数 %d 超过 Δf 内部极小点 %d",
				seed, rep.Actual.Interior, rep.Prediction.InteriorMinima)
		}
		if rep.Actual.InteriorMaxima > 0 && rep.Actual.MaximaInMask == 0 {
			t.Fatalf("seed %d: 内部极大点全在 Mask 外", seed)
		}

		jt := f.BuildJoinTree()
		if jt.TieCount != 0 {
			t.Fatalf("seed %d: 等高邻接 %d，非通用情形", seed, jt.TieCount)
		}
		if jt.P != len(maxima) {
			t.Fatalf("seed %d: P=%d 极大点=%d", seed, jt.P, len(maxima))
		}
		if len(jt.Saddles) != jt.P-1 {
			t.Fatalf("seed %d: S=%d P-1=%d", seed, len(jt.Saddles), jt.P-1)
		}
		if len(jt.Arcs) != 2*jt.P-2 {
			t.Fatalf("seed %d: 弧=%d 2P-2=%d", seed, len(jt.Arcs), 2*jt.P-2)
		}
		for ar := range jt.Arities {
			if ar != 2 {
				t.Fatalf("seed %d: 非二元合并 arity=%d", seed, ar)
			}
		}

		crest := f.Extract(ModeCrest)
		for v, d := range crest.Degree() {
			if d > 2 {
				t.Fatalf("seed %d: 脊线顶点 %d 度数 %d", seed, v, d)
			}
		}
		// 绝大多数极大点应落在脊线 1 格内。
		near := 0
		for _, m := range maxima {
			if DistToRidge(crest, float64(m.X), float64(m.Y)) <= 1.0 {
				near++
			}
		}
		if frac := float64(near) / float64(len(maxima)); frac < 0.85 {
			t.Fatalf("seed %d: 脊线覆盖极大点比例仅 %.2f", seed, frac)
		}
	}
}
