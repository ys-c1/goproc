package dft

import (
	"math"
	"math/rand"
	"testing"
)

// naiveDFT 是 O(N²) 的定义式 DFT，仅用于校验。
func naiveDFT(a []complex128, inverse bool) []complex128 {
	n := len(a)
	out := make([]complex128, n)
	sign := 1.0
	if inverse {
		sign = -1.0
	}
	for k := 0; k < n; k++ {
		var s complex128
		for j := 0; j < n; j++ {
			ang := sign * 2 * math.Pi * float64(k*j) / float64(n)
			s += a[j] * complex(math.Cos(ang), math.Sin(ang))
		}
		if inverse {
			s /= complex(float64(n), 0)
		}
		out[k] = s
	}
	return out
}

func TestFFTMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, n := range []int{2, 4, 8, 16, 64, 128, 256} {
		a := make([]complex128, n)
		for i := range a {
			a[i] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		got := append([]complex128(nil), a...)
		FFT(got, false)
		want := naiveDFT(a, false)
		for i := range got {
			if math.Abs(real(got[i])-real(want[i])) > 1e-9 || math.Abs(imag(got[i])-imag(want[i])) > 1e-9 {
				t.Fatalf("n=%d idx=%d got=%v want=%v", n, i, got[i], want[i])
			}
		}
	}
}

func TestFFTRoundTrip(t *testing.T) {
	n := 64
	rng := rand.New(rand.NewSource(7))
	a := make([]complex128, n)
	for i := range a {
		a[i] = complex(rng.NormFloat64()*3, rng.NormFloat64())
	}
	got := append([]complex128(nil), a...)
	FFT(got, false)
	FFT(got, true)
	for i := range got {
		if math.Abs(real(got[i])-real(a[i])) > 1e-10 || math.Abs(imag(got[i])-imag(a[i])) > 1e-10 {
			t.Fatalf("idx=%d roundtrip=%v orig=%v", i, got[i], a[i])
		}
	}
}

func TestFFT2KnownModes(t *testing.T) {
	abs := func(c complex128) float64 { return math.Hypot(real(c), imag(c)) }
	// 场 h(x,y)=3cos(2π·2x/n)+2sin(2π·3y/n)，对应频点能量应可解析定位。
	n := 32
	real := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			real[y*n+x] = 3*math.Cos(2*math.Pi*2*float64(x)/float64(n)) +
				2*math.Sin(2*math.Pi*3*float64(y)/float64(n))
		}
	}
	spec := FFT2(real, n)
	mag := func(kx, ky int) float64 { return abs(spec[ky*n+kx]) }
	// 二维 FFT 中只随 x 变化的模式，频点系数为一维 DFT 的 n 倍：
	// 余弦在 k=±2 的系数幅为 3n²/2。
	if want := 3 * float64(n) * float64(n) / 2; math.Abs(mag(2, 0)-want) > 1e-9 || math.Abs(mag(n-2, 0)-want) > 1e-9 {
		t.Fatalf("余弦模式能量错误: %v", mag(2, 0))
	}
	// 正弦在 k=±3 的系数幅为 n²（一维 sin 系数 n/2，二维乘 n）。
	if want := float64(n) * float64(n); math.Abs(mag(0, 3)-want) > 1e-9 || math.Abs(mag(0, n-3)-want) > 1e-9 {
		t.Fatalf("正弦模式能量错误: %v", mag(0, 3))
	}
	// 其余非直流非上述频点应接近零。
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			mode := (x == 2 || x == n-2) && y == 0
			mode = mode || ((x == 0) && (y == 3 || y == n-3))
			if !mode && mag(x, y) > 1e-8 {
				t.Fatalf("非预期频点 (%d,%d) 能量 %g", x, y, mag(x, y))
			}
		}
	}
}

func TestFreq(t *testing.T) {
	w := Freq(8)
	want := []float64{0, math.Pi / 4, math.Pi / 2, 3 * math.Pi / 4, math.Pi, -3 * math.Pi / 4, -math.Pi / 2, -math.Pi / 4}
	for i := range w {
		if math.Abs(w[i]-want[i]) > 1e-12 {
			t.Fatalf("idx=%d got=%v want=%v", i, w[i], want[i])
		}
	}
}
