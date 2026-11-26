package spectrum

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"goproc/internal/dft"
)

func freqForTest(n int) []float64 { return dft.Freq(n) }

// inverseReal 对 Hermitian 谱做二维逆变换并取实部。
func inverseReal(spec []complex128, n int) []float64 {
	buf := append([]complex128(nil), spec...)
	row := make([]complex128, n)
	for y := 0; y < n; y++ {
		copy(row, buf[y*n:(y+1)*n])
		dft.FFT(row, true)
		copy(buf[y*n:(y+1)*n], row)
	}
	col := make([]complex128, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = buf[y*n+x]
		}
		dft.FFT(col, true)
		for y := 0; y < n; y++ {
			buf[y*n+x] = col[y]
		}
	}
	out := make([]float64, n*n)
	for i, c := range buf {
		out[i] = real(c)
	}
	return out
}

func TestSectorIsotropicUniform(t *testing.T) {
	// 各向同性协方差：任意扇区概率必须严格相等。
	p := SectorProbabilities(8, 2, 2, 0)
	for _, v := range p {
		if math.Abs(v-1.0/8) > 1e-12 {
			t.Fatalf("各向同性下八方位概率不均: %v", p)
		}
	}
	p16 := SectorProbabilities(16, 3, 3, 0)
	for _, v := range p16 {
		if math.Abs(v-1.0/16) > 1e-12 {
			t.Fatalf("各向同性下 16 箱概率不均: %v", p16)
		}
	}
}

func TestAnalyticMatchesNumeric(t *testing.T) {
	cases := []struct{ mxx, myy, mxy float64 }{
		{1, 1, 0},
		{4, 1, 0},
		{3, 2, 0.7},
		{2.5, 0.6, -0.4},
	}
	for _, c := range cases {
		a := SectorProbabilities(16, c.mxx, c.myy, c.mxy)
		n := SectorProbabilitiesNumeric(16, c.mxx, c.myy, c.mxy)
		var sumA, sumN, maxErr float64
		for k := range a {
			sumA += a[k]
			sumN += n[k]
			if d := math.Abs(a[k] - n[k]); d > maxErr {
				maxErr = d
			}
		}
		if math.Abs(sumA-1) > 1e-12 || math.Abs(sumN-1) > 1e-9 {
			t.Fatalf("概率不归一: %v %v", sumA, sumN)
		}
		if maxErr > 1e-4 {
			t.Fatalf("解析与数值积分偏差过大: %v vs %v", a, n)
		}
	}
}

func TestMonteCarloSectors(t *testing.T) {
	// 生成指定协方差的二元高斯样本，经验扇区占比应对上解析概率。
	rng := rand.New(rand.NewSource(123))
	mxx, myy, mxy := 3.0, 1.0, 0.6
	// Cholesky 分解 Σ=L Lᵀ。
	a := math.Sqrt(mxx)
	b := mxy / a
	c := math.Sqrt(myy - b*b)
	const N = 400000
	nb := 8
	emp := make([]float64, nb)
	width := 2 * math.Pi / float64(nb)
	for i := 0; i < N; i++ {
		z1, z2 := rng.NormFloat64(), rng.NormFloat64()
		p := a * z1
		q := b*z1 + c*z2
		phi := math.Atan2(q, p)
		if phi < 0 {
			phi += 2 * math.Pi
		}
		k := int(phi / width)
		if k >= nb {
			k = nb - 1
		}
		emp[k]++
	}
	want := SectorProbabilities(nb, mxx, myy, mxy)
	var maxErr float64
	for k := range emp {
		emp[k] /= N
		if d := math.Abs(emp[k] - want[k]); d > maxErr {
			maxErr = d
		}
	}
	// 4e5 样本下统计涨落约 0.05%，放到 0.4% 留足余量。
	if maxErr > 0.004 {
		t.Fatalf("Monte Carlo 与解析扇区概率不符: emp=%v want=%v", emp, want)
	}
}

// 对白噪声施加已知传递函数的方向滤波，多次实现平均后，
// 谱矩应收敛到传递函数给出的解析值（单实现周期图涨落很大，必须平均）。
func TestSpectralMomentsMatchEmpirical(t *testing.T) {
	size := 128
	const trials = 40
	var sumX, sumY, sumXY float64
	for trial := 0; trial < trials; trial++ {
		rng := rand.New(rand.NewSource(int64(99 + trial)))
		white := make([]float64, size*size)
		for i := range white {
			white[i] = rng.NormFloat64()
		}
		// h = Dx(white) + 0.3·Dy(white)，Dx/Dy 为周期前向差分，
		// 传递函数模平方 |Tx|²=4sin²(wx/2)。
		h := make([]float64, size*size)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				i := y*size + x
				xr := (x + 1) % size
				yr := (y + 1) % size
				h[i] = (white[y*size+xr] - white[i]) + 0.3*(white[yr*size+x]-white[i])
			}
		}
		m := SpectralMoments(size, h, false)
		sumX += m.Mxx
		sumY += m.Myy
		sumXY += m.Mxy
	}
	gotX, gotY, gotXY := sumX/trials, sumY/trials, sumXY/trials

	// 解析基准：单位方差白噪声的二维 DFT 系数 E|H|²=N²。
	// 场 h=Dx(w)+0.3·Dy(w) 中两个差分共享同一白噪声，传递函数
	// T=(e^{iwx}-1)+0.3(e^{iwy}-1)，|T|² 含非零交叉项，不能只加模平方。
	// 谱矩对应中心差分核 sin(w)，= Σ sin(w)sin(w)·|T|² / N²。
	w := freqForTest(size)
	var expX, expY, expXY float64
	for _, wy := range w {
		for _, wx := range w {
			tx := complex(math.Cos(wx)-1, math.Sin(wx))
			ty := complex(math.Cos(wy)-1, math.Sin(wy))
			t := tx + 0.3*ty
			t2 := real(t * cmplx.Conj(t))
			sx, sy := math.Sin(wx), math.Sin(wy)
			expX += sx * sx * t2
			expY += sy * sy * t2
			expXY += sx * sy * t2
		}
	}
	n2 := float64(size * size)
	expX /= n2
	expY /= n2
	expXY /= n2
	if d := math.Abs(gotX-expX) / expX; d > 0.06 {
		t.Fatalf("Mxx 谱矩均值=%v 解析=%v 相对偏差 %v", gotX, expX, d)
	}
	if d := math.Abs(gotY-expY) / expY; d > 0.06 {
		t.Fatalf("Myy 谱矩均值=%v 解析=%v 相对偏差 %v", gotY, expY, d)
	}
	if math.Abs(gotXY-expXY) > 0.1*math.Max(expX, expY) {
		t.Fatalf("Mxy 谱矩均值=%v 解析=%v", gotXY, expXY)
	}
}

func TestRadialSlopeSyntheticFBM(t *testing.T) {
	// 在谱域直接合成 P(w)∝w^-β 的各向同性场，检验斜率拟合。
	size := 256
	beta := 2.5
	rng := rand.New(rand.NewSource(2026))
	spec := make([]complex128, size*size)
	w := freqForTest(size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r := math.Hypot(w[x], w[y])
			if r < 1e-9 {
				continue
			}
			amp := math.Pow(r, -beta/2) / float64(size)
			re := rng.NormFloat64() * amp
			im := rng.NormFloat64() * amp
			// Hermitian 共轭，保证逆变换为实数。
			xm, ym := (size-x)%size, (size-y)%size
			spec[y*size+x] = complex(re, im)
			spec[ym*size+xm] = complex(re, -im)
		}
	}
	h := inverseReal(spec, size)
	fit := RadialSlope(size, h, 0.35, 2.4)
	if math.Abs(fit.Beta-beta) > 0.15 {
		t.Fatalf("谱斜率拟合 β=%v 期望 %v R²=%v", fit.Beta, beta, fit.RSquared)
	}
	if fit.RSquared < 0.95 {
		t.Fatalf("幂律拟合决定系数过低: %v", fit.RSquared)
	}
}
