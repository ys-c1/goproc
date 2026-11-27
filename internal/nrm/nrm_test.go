package nrm

import (
	"math"
	"testing"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

const (
	fieldSize = 128
	fieldFreq = 0.02
	fieldOct  = 4
	fieldLac  = 2.0
	fieldGain = 0.5
)

// field 复刻 api.Generate 的流程：fbm 求值后归一化到零到一，
// 同时按同一线性变换缩放解析梯度，作为法线真值。
type field struct {
	h, gx, gy []float64 // 归一化高度与解析梯度（像素坐标单位）
	n         int
}

func buildField(n int, seed uint64) field {
	src := noise.New(seed)
	f := field{make([]float64, n*n), make([]float64, n*n), make([]float64, n*n), n}
	lo, hi := math.Inf(1), math.Inf(-1)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			v, dx, dy := fbm.EvalGrad(src, float64(i)*fieldFreq, float64(j)*fieldFreq, fieldOct, fieldLac, fieldGain)
			f.h[j*n+i] = v
			// 像素坐标下的梯度：链式法则乘频率。
			f.gx[j*n+i] = dx * fieldFreq
			f.gy[j*n+i] = dy * fieldFreq
			lo = math.Min(lo, v)
			hi = math.Max(hi, v)
		}
	}
	s := 1 / (hi - lo)
	for k := range f.h {
		f.h[k] = (f.h[k] - lo) * s
		f.gx[k] *= s
		f.gy[k] *= s
	}
	return f
}

func unitNormal(gx, gy float64) (float64, float64, float64) {
	n := math.Sqrt(gx*gx + gy*gy + 1)
	return -gx / n, -gy / n, 1 / n
}

func angleDeg(ax, ay, az, bx, by, bz float64) float64 {
	d := math.Max(-1, math.Min(1, ax*bx+ay*by+az*bz))
	return math.Acos(d) * 180 / math.Pi
}

type stats struct {
	mean, rms, max float64
	n              int
}

// eval 以 stride 像素为格距，用核 w 估计梯度，并与解析法线比较夹角。
func eval(f field, w []float64, stride int) stats {
	m := (len(w) - 1) / 2
	lo := m * stride
	hi := f.n - m*stride
	var sum, sq, mx float64
	cnt := 0
	for j := lo; j < hi; j += stride {
		for i := lo; i < hi; i += stride {
			var gx, gy float64
			for t, wt := range w {
				off := (t - m) * stride
				if off == 0 {
					continue
				}
				gx += wt * f.h[j*f.n+i+off]
				gy += wt * f.h[(j+off)*f.n+i]
			}
			gx /= float64(stride)
			gy /= float64(stride)
			ax, ay, az := unitNormal(f.gx[j*f.n+i], f.gy[j*f.n+i])
			bx, by, bz := unitNormal(gx, gy)
			a := angleDeg(ax, ay, az, bx, by, bz)
			sum += a
			sq += a * a
			mx = math.Max(mx, a)
			cnt++
		}
	}
	return stats{sum / float64(cnt), math.Sqrt(sq / float64(cnt)), mx, cnt}
}

// 反解核在核长三时退化为中心差分。
func TestSolveK3IsCentralDifference(t *testing.T) {
	w, err := Solve(1)
	if err != nil {
		t.Fatal(err)
	}
	cd := CentralDifference()
	if len(w) != len(cd) {
		t.Fatalf("核长不同: %d vs %d", len(w), len(cd))
	}
	for i := range w {
		if math.Abs(w[i]-cd[i]) > 1e-12 {
			t.Fatalf("K=3 反解应为中心差分，得到 %v", w)
		}
	}
}

// 核长五：列出每个权重，并检查约束与目标。
func TestSolveK5WeightsAndMoments(t *testing.T) {
	w, err := Solve(2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1.0 / 12, -2.0 / 3, 0, 2.0 / 3, -1.0 / 12}
	for i := range w {
		if math.Abs(w[i]-want[i]) > 1e-12 {
			t.Fatalf("K=5 权重[%d]=%.12f，期望 %.12f", i, w[i], want[i])
		}
	}
	t.Logf("K=5 权重（偏移 -2,-1,0,1,2）= [%.12f, %.12f, %.12f, %.12f, %.12f]",
		w[0], w[1], w[2], w[3], w[4])
	if s := Moment(w, 0); math.Abs(s) > 1e-12 {
		t.Fatalf("权重和应为零，得 %.3e", s)
	}
	for k := 1; k < len(w); k++ {
		if math.Abs(w[k]+w[len(w)-1-k]) > 1e-12 {
			t.Fatalf("核不反对称: w[%d]=%v w[%d]=%v", k, w[k], len(w)-1-k, w[len(w)-1-k])
		}
	}
	if s := Moment(w, 1); math.Abs(s-1) > 1e-12 {
		t.Fatalf("一阶矩应为 1，得 %.3e", s)
	}
	if s := Moment(w, 3); math.Abs(s) > 1e-12 {
		t.Fatalf("三阶矩应为零（目标最优），得 %.3e", s)
	}
}

// 两种格距下反解核的角误差必须小于中心差分。
func TestSolvedBeatsCentralAtTwoSpacings(t *testing.T) {
	f := buildField(fieldSize, 7)
	solved, err := Solve(2)
	if err != nil {
		t.Fatal(err)
	}
	cd := CentralDifference()
	for _, stride := range []int{1, 2} {
		cs := eval(f, cd, stride)
		ss := eval(f, solved, stride)
		t.Logf("格距 d=%d 像素: 中心差分 mean=%.6f° rms=%.6f° max=%.6f° | 反解核 mean=%.6f° rms=%.6f° max=%.6f° | 差值(CD-反解)=%.6f°",
			stride, cs.mean, cs.rms, cs.max, ss.mean, ss.rms, ss.max, cs.mean-ss.mean)
		if !(ss.mean < cs.mean && ss.rms < cs.rms && ss.max < cs.max) {
			t.Fatalf("格距 d=%d 下反解核未全面优于中心差分", stride)
		}
	}
}

// 对照：逐抽头平方式目标的解，在两档格距下都应劣于中心差分。
// 这验证了正确的「平方和」是逐点相干后再求和，而不是逐抽头平方。
func TestPerTapObjectiveLosesToCentral(t *testing.T) {
	f := buildField(fieldSize, 7)
	perp, err := SolvePerTap(2)
	if err != nil {
		t.Fatal(err)
	}
	cd := CentralDifference()
	for _, stride := range []int{1, 2} {
		cs := eval(f, cd, stride)
		ps := eval(f, perp, stride)
		t.Logf("格距 d=%d 像素: 中心差分 mean=%.6f° | 逐抽头解 mean=%.6f° | 逐抽头更差 %.6f°",
			stride, cs.mean, ps.mean, ps.mean-cs.mean)
		if ps.mean <= cs.mean {
			t.Fatalf("格距 d=%d：逐抽头解不应优于中心差分", stride)
		}
	}
}

// 若把「平方和」误读成逐抽头、且把「高度二阶项」按字面取 h_xx（即核权重乘
// k²），解 u1=2/5,u2=1/20，得 w=[-1/20,-2/5,0,2/5,1/20]。它同样在两档格距下
// 都劣于中心差分。三种写法里只有相干 k³ 形式能通过验收。
func TestSecondOrderPerTapLosesToo(t *testing.T) {
	f := buildField(fieldSize, 7)
	perp2 := []float64{-1.0 / 20, -2.0 / 5, 0, 2.0 / 5, 1.0 / 20}
	cd := CentralDifference()
	for _, stride := range []int{1, 2} {
		cs := eval(f, cd, stride)
		ps := eval(f, perp2, stride)
		t.Logf("格距 d=%d 像素: 中心差分 mean=%.6f° | 逐抽头 k² 解 mean=%.6f° | 更差 %.6f°",
			stride, cs.mean, ps.mean, ps.mean-cs.mean)
		if ps.mean <= cs.mean {
			t.Fatalf("格距 d=%d：逐抽头 k² 解不应优于中心差分", stride)
		}
	}
}
