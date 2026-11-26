// seedcov 检验命题：换种子等于换一批独立的格点值，
// 即两批格点值在对应位置的协方差为零。
//
// 对三档种子间隔 Δ，从互不相交的种子段抽取 n 个独立的 (S, S+Δ) 成对样本，
// 每个样本配一个均匀抽取的格点位置，测得协方差估计并与 H0 下的波动范围对照。
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

type gapCase struct {
	name  string
	delta uint64
	base  uint64 // 该档种子段起点，三档区间互不相交
}

func main() {
	n := flag.Int("n", 4096, "每档独立成对样本数")
	size := flag.Int("size", 128, "留档高度图边长")
	flag.Parse()

	cases := []gapCase{
		{name: "gap=1 (相邻种子)", delta: 1, base: 1_000_000},
		{name: "gap=65536 (2^16)", delta: 1 << 16, base: 2_000_000},
		{name: "gap=2^32", delta: 1 << 32, base: 3_000_000},
	}

	// H0 下格点值的精确理论矩（见 docs/seed-independence.md 推导）：
	// h=perm[...] 在 0..255 上均匀，G=h/127.5-1，E[G]=0。
	const gridN = 256.0
	meanG := 0.0
	sumSq, sumFour := 0.0, 0.0
	for h := 0.0; h < gridN; h++ {
		g := h/127.5 - 1
		sumSq += g * g
		sumFour += math.Pow(g, 4)
	}
	varianceG := sumSq / gridN
	mu4G := sumFour / gridN
	fmt.Printf("理论矩: E[G]=%.6f  Var(G)=%.6f  E[G^4]=%.6f\n\n", meanG, varianceG, mu4G)

	allPass := true
	for _, c := range cases {
		fmt.Printf("== %s, n=%d ==\n", c.name, *n)
		// 表一：底层格点值 Lattice（命题的直接对象）
		passGrid := runGrid(c, *n, varianceG)
		// 表二：插值噪声场 Value（图像实际使用的量）
		passVal := runValue(c, *n)
		// 表三：完整管线高度图的图级 Pearson 相关（解释“两幅图像”观感）
		passMap := runMaps(c.delta, *size)
		// 辅助诊断：置换表独立性与归一化对图级相关的影响
		runPermDiagnostic(c.delta)
		runNormDiagnostic(c.delta, *size)
		if !(passGrid && passVal && passMap) {
			allPass = false
		}
		fmt.Println()
	}

	// 阳性对照：故意构造有偏的派生（同一来源派生两个“源”），
	// 必须显著超界，否则本检验没有检出能力。
	fmt.Println("== 阳性对照: 同种子与自身配对（应显著超界）==")
	ctrlPass := runNegativeControl(*n)
	if !ctrlPass {
		allPass = false
	}

	if !allPass {
		fmt.Println("\n结论: 存在超界档位，种子派生有偏")
		os.Exit(1)
	}
	fmt.Println("\n结论: 三档间隔协方差均落在 H0 波动范围内，未发现派生偏差")
}

// stats 累积一阶/二阶矩，统一用 1/n（非无偏 1/(n-1)）估计。
type stats struct {
	n                 int
	sumX, sumY, sumXY float64
	sumX2, sumY2      float64
}

func (s *stats) add(x, y float64) {
	s.n++
	s.sumX += x
	s.sumY += y
	s.sumXY += x * y
	s.sumX2 += x * x
	s.sumY2 += y * y
}

func (s *stats) means() (mx, my float64) { return s.sumX / float64(s.n), s.sumY / float64(s.n) }

func (s *stats) covariance() float64 {
	mx, my := s.means()
	return s.sumXY/float64(s.n) - mx*my
}

func (s *stats) vars() (vx, vy float64) {
	mx, my := s.means()
	return s.sumX2/float64(s.n) - mx*mx, s.sumY2/float64(s.n) - my*my
}

// gridPos 用确定性 LCG 把样本序号映射到 256x256 格点上的均匀位置，
// 位置序列与种子、档位完全无关；配合均匀种子，(S,P_k) 构成独立同分布样本。
func gridPos(k int) (ix, iy int) {
	z := uint64(k)*2862933555777941757 + 3037000493
	z ^= z >> 33
	z *= 0xff51afd7ed558ccd
	z ^= z >> 33
	ix = int((z >> 32) & 0xff)
	iy = int(z & 0xff)
	return
}

// valPos 把样本序号映射到与生成管线相同的连续坐标区间（freq=0.02，边长 128）。
func valPos(k int) (x, y float64) {
	ix, iy := gridPos(k)
	return float64(ix%128) * 0.02, float64(iy%128) * 0.02
}

func runGrid(c gapCase, n int, theoryVar float64) bool {
	var st stats
	for k := 0; k < n; k++ {
		s1 := c.base + uint64(k)*7919
		s2 := s1 + c.delta
		a := noise.New(s1)
		b := noise.New(s2)
		ix, iy := gridPos(k)
		st.add(a.Lattice(ix, iy), b.Lattice(ix, iy))
	}
	return report("Lattice 格点值", st, theoryVar)
}

func runValue(c gapCase, n int) bool {
	var st stats
	for k := 0; k < n; k++ {
		s1 := c.base + uint64(k)*7919
		s2 := s1 + c.delta
		a := noise.New(s1)
		b := noise.New(s2)
		x, y := valPos(k)
		// 与默认管线一致：4 层、lacunarity=2、gain=0.5
		st.add(fbm.Eval(a, x, y, 4, 2, 0.5), fbm.Eval(b, x, y, 4, 2, 0.5))
	}
	// Value 场的方差没有闭式理论值，用样本方差代入（H0 下方差乘积的估计）。
	return report("Value 插值噪声", st, math.NaN())
}

func runMaps(delta uint64, size int) bool {
	opt := api.DefaultOptions()
	opt.Size = size
	opt.Seed = 42
	h1, err := api.Generate(opt)
	if err != nil {
		panic(err)
	}
	opt.Seed = 42 + delta
	h2, err := api.Generate(opt)
	if err != nil {
		panic(err)
	}
	r := pearson(h1.Data, h2.Data)
	fmt.Printf("  图级 Pearson 相关 r = %+.4f（单对种子，仅描述性，不参与判据）\n", r)
	return true
}

// runPermDiagnostic 统计两张置换表在 256x256 格点上取相同值的比例，
// 独立置换表的期望重合率为 1/256≈0.0039。
func runPermDiagnostic(delta uint64) {
	a := noise.New(42)
	b := noise.New(42 + delta)
	same := 0
	for ix := 0; ix < 256; ix++ {
		for iy := 0; iy < 256; iy++ {
			if a.Lattice(ix, iy) == b.Lattice(ix, iy) {
				same++
			}
		}
	}
	fmt.Printf("  置换表格点重合率 = %.4f（独立期望 1/256≈0.0039）\n", float64(same)/(256*256))
}

// runNormDiagnostic 比较归一化前后的图级相关，定位“两幅图像”观感相似的来源。
func runNormDiagnostic(delta uint64, size int) {
	const freq = 0.02
	s1 := noise.New(42)
	s2 := noise.New(42 + delta)
	raw1 := make([]float64, size*size)
	raw2 := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			raw1[y*size+x] = fbm.Eval(s1, float64(x)*freq, float64(y)*freq, 4, 2, 0.5)
			raw2[y*size+x] = fbm.Eval(s2, float64(x)*freq, float64(y)*freq, 4, 2, 0.5)
		}
	}
	rRaw := pearson(raw1, raw2)
	n1 := append([]float64(nil), raw1...)
	n2 := append([]float64(nil), raw2...)
	hmap.Normalize(n1)
	hmap.Normalize(n2)
	rNorm := pearson(n1, n2)
	// 空间自相关导致有效样本量远低于像素数：ESS≈N*(1-ρ)/(1+ρ)，
	// ρ 取水平相邻像素的一阶自相关（对 raw1 估计）。
	rho := lag1Autocorr(raw1, size)
	n := size * size
	ess := float64(n) * (1 - rho) / (1 + rho)
	fmt.Printf("  图级相关: 归一化前 %+.4f → 归一化后 %+.4f\n", rRaw, rNorm)
	fmt.Printf("  水平 lag-1 自相关 ρ=%.3f，有效样本量 ESS≈%.0f（像素数 %d），独立样本下 r 的 1σ≈%.4f\n",
		rho, ess, n, 1/math.Sqrt(ess))
}

// lag1Autocorr 估计水平方向相邻像素的一阶自相关系数。
func lag1Autocorr(v []float64, size int) float64 {
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	var cov, varSum float64
	for y := 0; y < size; y++ {
		for x := 1; x < size; x++ {
			a := v[y*size+x-1] - mean
			b := v[y*size+x] - mean
			cov += a * b
			varSum += b * b
		}
	}
	return cov / varSum
}

func pearson(a, b []float64) float64 {
	n := float64(len(a))
	var sx, sy, sxy, sx2, sy2 float64
	for i := range a {
		sx += a[i]
		sy += b[i]
		sxy += a[i] * b[i]
		sx2 += a[i] * a[i]
		sy2 += b[i] * b[i]
	}
	mx, my := sx/n, sy/n
	cov := sxy/n - mx*my
	vx := sx2/n - mx*mx
	vy := sy2/n - my*my
	return cov / math.Sqrt(vx*vy)
}

func report(label string, st stats, theoryVar float64) bool {
	n := float64(st.n)
	cov := st.covariance()
	vx, vy := st.vars()
	r := cov / math.Sqrt(vx*vy)

	// H0 下 E[Ĉ]=0。Var(Ĉ) 的理论值：独立时 E[X²Y²]=Var(X)Var(Y)，再除以 n。
	var varHat float64
	var boundNote string
	if !math.IsNaN(theoryVar) {
		varHat = theoryVar * theoryVar / n
		boundNote = "理论方差"
	} else {
		varHat = vx * vy / n
		boundNote = "样本方差代入"
	}
	sd := math.Sqrt(varHat)
	clt95 := 1.96 * sd           // 大样本 CLT 实用范围（双侧 95%）
	sd3 := 3 * sd                // 3σ，几乎必中，用作判据
	cheb95 := math.Sqrt(20) * sd // Chebyshev 有限样本严格 95% 界

	verdict := "在范围内"
	over := math.Abs(cov) > sd3
	if over {
		verdict = "超界！派生有偏"
	}
	fmt.Printf("  %s\n", label)
	fmt.Printf("    Ĉ=%+.5f  r=%+.4f  VarX=%.4f VarY=%.4f\n", cov, r, vx, vy)
	fmt.Printf("    H0 波动范围(%s): 95%%CLT ±%.5f | 3σ ±%.5f | Chebyshev95%% ±%.5f  => %s\n",
		boundNote, clt95, sd3, cheb95, verdict)
	return !over
}

func runNegativeControl(n int) bool {
	// 同一来源配对：X=Y，理论协方差等于 Var(G)，应远超任何合理范围。
	var st stats
	for k := 0; k < n; k++ {
		s := noise.New(9_000_000 + uint64(k)*7919)
		ix, iy := gridPos(k)
		v := s.Lattice(ix, iy)
		st.add(v, v)
	}
	cov := st.covariance()
	theoryVar := 0.0
	for h := 0.0; h < 256.0; h++ {
		g := h/127.5 - 1
		theoryVar += g * g
	}
	theoryVar /= 256.0 // Var(G)，与主程序打印的理论矩一致
	// X=Y 时 Cov(X,Y)=Var(X)，Ĉ 的 3σ 门限按 Var(Ĉ)=Var(X)^2/n 估计。
	sd3 := 3 * theoryVar / math.Sqrt(float64(n))
	over := math.Abs(cov) > sd3
	fmt.Printf("  Ĉ=%+.5f（理论偏倚 Var(G)=%.4f），3σ 门限 ±%.5f => 超界=%v\n",
		cov, theoryVar, sd3, over)
	if !over {
		fmt.Println("  检验失效：阳性对照未被检出")
	}
	return over
}
