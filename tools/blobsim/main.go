// blobsim 测量分形叠加地形的团块强度 S = 大窗口平均方差 / 小窗口平均方差。
//
// 团块强度按需求：窗口取 32 格（大）与 8 格（小）。主口径用 32×32 与 8×8 的
// 不重叠块（tiling）；另给滑动窗口口径做稳健性对照，因为值噪声在小窗口下
// 有格点相位依赖，单靠一种取法容易把相位差当成团块差。
//
// 用法：
//
//	go run ./tools/blobsim                      # 256 边长、32 个种子
//	go run ./tools/blobsim -size 512 -seeds 16
//
// 输出带 # 标题，可直接重定向到文件。
package main

import (
	"flag"
	"fmt"
	"math"

	"goproc/internal/noise"
)

const (
	winBig = 32 // 大窗口
	winSml = 8  // 小窗口
)

// octField 返回单个倍频程（频率 freq*lac^i）在整幅图上的原始值场。
func octField(src *noise.Source, size int, freq, lac float64, i int) []float64 {
	f := freq * math.Pow(lac, float64(i))
	out := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			out[y*size+x] = src.Value(float64(x)*f, float64(y)*f)
		}
	}
	return out
}

// meanVarTile 返回所有 w×w 不重叠块内方差的平均值。
func meanVarTile(data []float64, size, w int) float64 {
	n := size / w
	total := 0.0
	for by := 0; by < n; by++ {
		for bx := 0; bx < n; bx++ {
			mean := 0.0
			for y := 0; y < w; y++ {
				for x := 0; x < w; x++ {
					mean += data[(by*w+y)*size+bx*w+x]
				}
			}
			mean /= float64(w * w)
			v := 0.0
			for y := 0; y < w; y++ {
				for x := 0; x < w; x++ {
					d := data[(by*w+y)*size+bx*w+x] - mean
					v += d * d
				}
			}
			total += v / float64(w*w)
		}
	}
	return total / float64(n*n)
}

// prefix2D 构造二维前缀和，尺寸 (size+1)²。
func prefix2D(data []float64, size int, square bool) []float64 {
	p := make([]float64, (size+1)*(size+1))
	for y := 0; y < size; y++ {
		row := 0.0
		for x := 0; x < size; x++ {
			v := data[y*size+x]
			if square {
				v *= v
			}
			row += v
			p[(y+1)*(size+1)+(x+1)] = p[y*(size+1)+(x+1)] + row
		}
	}
	return p
}

func rectSum(p []float64, size, x0, y0, w int) float64 {
	s := size + 1
	return p[(y0+w)*s+x0+w] - p[y0*s+x0+w] - p[(y0+w)*s+x0] + p[y0*s+x0]
}

// meanVarSlide 返回所有完全落在图内的 w×w 滑窗方差的平均值。
func meanVarSlide(data []float64, size, w int) float64 {
	ps := prefix2D(data, size, false)
	pq := prefix2D(data, size, true)
	n := float64(w * w)
	cnt := 0
	total := 0.0
	for y := 0; y+w <= size; y++ {
		for x := 0; x+w <= size; x++ {
			s := rectSum(ps, size, x, y, w)
			q := rectSum(pq, size, x, y, w)
			total += q/n - (s/n)*(s/n)
			cnt++
		}
	}
	return total / float64(cnt)
}

type config struct {
	size, seeds int
	freq, lac   float64
	nOct        int
	fields      [][][]float64 // [seed][octave][pixel]
}

func newConfig(size, seeds int, freq, lac float64, nOct int) *config {
	c := &config{size: size, seeds: seeds, freq: freq, lac: lac, nOct: nOct}
	c.fields = make([][][]float64, seeds)
	for s := 0; s < seeds; s++ {
		src := noise.New(uint64(s + 1))
		c.fields[s] = make([][]float64, nOct)
		for i := 0; i < nOct; i++ {
			c.fields[s][i] = octField(src, size, freq, lac, i)
		}
	}
	return c
}

// measure 对给定各层权重，返回种子平均后的
// (S_tile, S_slide, V32_tile, V8_tile) 以及逐种子 S_tile 的标准差。
func (c *config) measure(weights []float64) (s, ss, v32, v8, sd float64) {
	buf := make([]float64, c.size*c.size)
	per := make([]float64, c.seeds)
	for si := range c.fields {
		for p := range buf {
			buf[p] = 0
		}
		for i, w := range weights {
			f := c.fields[si][i]
			for p := range buf {
				buf[p] += w * f[p]
			}
		}
		a := meanVarTile(buf, c.size, winBig)
		b := meanVarTile(buf, c.size, winSml)
		v32 += a
		v8 += b
		per[si] = a / b
		ss += meanVarSlide(buf, c.size, winBig) / meanVarSlide(buf, c.size, winSml)
	}
	n := float64(c.seeds)
	v32 /= n
	v8 /= n
	s = v32 / v8
	ss /= n
	for _, x := range per {
		sd += (x - s) * (x - s)
	}
	sd = math.Sqrt(sd / n)
	return
}

// weights 构造长度为 n 的权重向量，前几个取自 vals。
func weights(n int, vals ...float64) []float64 {
	w := make([]float64, n)
	copy(w, vals)
	return w
}

// pct 返回 x 相对 ref 的百分比偏差。
func pct(x, ref float64) float64 { return (x/ref - 1) * 100 }

func main() {
	size := flag.Int("size", 256, "边长")
	freq := flag.Float64("freq", 0.02, "基础频率")
	lac := flag.Float64("lac", 2.0, "频率倍增")
	seeds := flag.Int("seeds", 32, "平均种子数")
	nOct := flag.Int("oct", 4, "倍频程数")
	flag.Parse()

	c := newConfig(*size, *seeds, *freq, *lac, *nOct)
	fmt.Printf("# config size=%d freq=%v lac=%v seeds=%d octaves=%d windows=%d/%d\n",
		c.size, c.freq, c.lac, c.seeds, c.nOct, winBig, winSml)
	fmt.Printf("# 波长: %s\n", lambdaString(c))

	// ---- 1. 单层基线 ----
	fmt.Println("\n# 1. 单倍频程基线（权重 1，其余 0）")
	fmt.Println("octave,lambda_px,V32_tile,V8_tile,S_tile,S_slide")
	v32o := make([]float64, c.nOct)
	v8o := make([]float64, c.nOct)
	so := make([]float64, c.nOct)
	sso := make([]float64, c.nOct)
	for i := 0; i < c.nOct; i++ {
		w := make([]float64, c.nOct)
		w[i] = 1
		s, ss, a, b, sd := c.measure(w)
		v32o[i], v8o[i], so[i], sso[i] = a, b, s, ss
		fmt.Printf("%d,%.2f,%.6f,%.6f,%.6f,%.6f (sd=%.4f)\n",
			i, 1/(c.freq*math.Pow(c.lac, float64(i))), a, b, s, ss, sd)
	}

	// ---- 2. 两层扫描：低=oct0，高=oct1 ----
	fmt.Println("\n# 2. 两层扫描  低=oct0 权重 1，高=oct1 权重 r=w_high/w_low")
	fmt.Println("r,S_tile,S_slide")
	type pt struct{ r, s, ss float64 }
	var pts []pt
	for i := -20; i <= 40; i++ { // r = 2^(i/4) ∈ [1/32, 1024]
		r := math.Pow(2, float64(i)/4)
		s, ss, _, _, _ := c.measure(weights(c.nOct, 1, r))
		pts = append(pts, pt{r, s, ss})
		fmt.Printf("%.6f,%.6f,%.6f\n", r, s, ss)
	}
	fmt.Println("# 相邻差分的符号变化（tile / slide 两种口径）：")
	reportExtrema := func(name string, get func(pt) float64) {
		found := 0
		for i := 1; i+1 < len(pts); i++ {
			d1 := get(pts[i]) - get(pts[i-1])
			d2 := get(pts[i+1]) - get(pts[i])
			if (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0) {
				found++
				fmt.Printf("#   %s 内部极值 %s 附近 r=%.6f S=%.6f（相对 r=1/32 处 %+.3f%%）\n",
					name, map[bool]string{true: "最大", false: "最小"}[d1 > 0],
					pts[i].r, get(pts[i]), pct(get(pts[i]), get(pts[0])))
			}
		}
		if found == 0 {
			fmt.Printf("#   %s：扫描区间内无内点极值\n", name)
		}
		// 端点回落量：若内部极小后又回升，比较极小与扫描末端。
		minI, maxI := 0, 0
		for i := range pts {
			if get(pts[i]) < get(pts[minI]) {
				minI = i
			}
			if get(pts[i]) > get(pts[maxI]) {
				maxI = i
			}
		}
		fmt.Printf("#   %s 区间内最小 r=%.4f S=%.6f；最大 r=%.4f S=%.6f；末端 r=%.0f S=%.6f\n",
			name, pts[minI].r, get(pts[minI]), pts[maxI].r, get(pts[maxI]), pts[len(pts)-1].r, get(pts[len(pts)-1]))
		fmt.Printf("#   %s 最小→末端 回升 %+.4f%%（%s）\n", name,
			pct(get(pts[len(pts)-1]), get(pts[minI])),
			map[bool]string{true: "存在先降后升", false: "末端仍在下行或持平"}[get(pts[len(pts)-1]) > get(pts[minI])])
	}
	reportExtrema("tile", func(p pt) float64 { return p.s })
	reportExtrema("slide", func(p pt) float64 { return p.ss })

	// ---- 2b. 两层扫描：低=最低层 oct0，高=最高层 oct(top-1) ----
	// fbm 的“高频层”通常指整个栈里最高那一层，与相邻层不同。
	top := c.nOct - 1
	fmt.Printf("\n# 2b. 两层扫描  低=oct0 权重 1，高=oct%d（最高层）权重 r\n", top)
	fmt.Println("r,S_tile,S_slide")
	type pt2 struct{ r, s, ss float64 }
	var ptsT []pt2
	for i := -20; i <= 40; i++ {
		r := math.Pow(2, float64(i)/4)
		w := make([]float64, c.nOct)
		w[0], w[top] = 1, r
		s, ss, _, _, _ := c.measure(w)
		ptsT = append(ptsT, pt2{r, s, ss})
		fmt.Printf("%.6f,%.6f,%.6f\n", r, s, ss)
	}
	fmt.Println("# 相邻差分的符号变化（tile / slide）：")
	for _, which := range []string{"tile", "slide"} {
		get := func(p pt2) float64 {
			if which == "tile" {
				return p.s
			}
			return p.ss
		}
		found := 0
		for i := 1; i+1 < len(ptsT); i++ {
			d1 := get(ptsT[i]) - get(ptsT[i-1])
			d2 := get(ptsT[i+1]) - get(ptsT[i])
			if (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0) {
				found++
				fmt.Printf("#   %s 内部极值 %s r=%.4f S=%.6f\n", which,
					map[bool]string{true: "最大", false: "最小"}[d1 > 0], ptsT[i].r, get(ptsT[i]))
			}
		}
		if found == 0 {
			fmt.Printf("#   %s 无内点极值（单调）\n", which)
		}
	}

	// ---- 3. 解析：S(u) 是 u=r² 的莫比乌斯函数 ----
	fmt.Println("\n# 3. 两层解析")
	a0, b1 := v32o[0], v32o[1]
	c0, d1 := v8o[0], v8o[1]
	fmt.Printf("# 模型 S(r)=(a+b*r^2)/(c+d*r^2)  a=%.6f b=%.6f c=%.6f d=%.6f\n", a0, b1, c0, d1)
	fmt.Printf("# 判别式 a*d-b*c = %+.6e  (<0 => 随 u 单调递减)\n", a0*d1-b1*c0)
	fmt.Printf("# S(0)=%.6f (纯低频)  S(inf)=%.6f (纯高频)  差=%+.2f%%\n",
		a0/c0, b1/d1, pct(b1/d1, a0/c0))
	fmt.Println("# 分子分母都是正系数，且 b*c-a*d<0 => 在 r∈[0,∞) 上单调递减，无内点极值；")
	fmt.Println("# 下确界在 r→∞（低频层权重→0）。")
	_, _, mix32, mix8, _ := c.measure(weights(c.nOct, 1, 1))
	fmt.Printf("# 实测交叉项：2*C32=%.6f  2*C8=%.6f（相对单层和 <1%%）\n",
		mix32-a0-b1, mix8-c0-d1)
	_, _, mix32b, mix8b, _ := c.measure(weights(c.nOct, 1, 8))
	fmt.Printf("# 交叉项在 r=8 时：2*C32=%.6f  2*C8=%.6f\n", mix32b-a0-64*b1, mix8b-c0-64*d1)
	fmt.Println("# 交叉项只改变过渡段形状，不改变两端，也不产生内部极小。")

	// ---- 4. 两档权重取整损失 ----
	sinf := b1 / d1
	fmt.Println("\n# 4. 两档权重取整损失（基准 S_inf = 纯高频层 = 连续下确界）")
	fmt.Printf("# S_inf = %.6f\n", sinf)
	for _, g := range []float64{0.25, 0.5, 0.75} {
		fmt.Printf("\n## 两档 {%.2f, 1.00}\n", g)
		fmt.Println("assignment,w_low,w_high,r,S_tile,S_slide,loss_pct")
		s1, ss1, _, _, _ := c.measure(weights(c.nOct, g, 1))
		s2, ss2, _, _, _ := c.measure(weights(c.nOct, 1, g))
		fmt.Printf("high=1.00,low=%.2f,%.2f,1.00,%.6f,%.6f,%.3f%%\n",
			g, g, s1, ss1, pct(s1, sinf))
		fmt.Printf("high=%.2f,low=1.00,1.00,%.2f,%.6f,%.6f,%.3f%%\n",
			g, g, s2, ss2, pct(s2, sinf))
	}
	fmt.Println("\n## 两档 {0, 1}（只保留或去掉低频层）")
	fmt.Println("assignment,w_low,w_high,r,S_tile,S_slide,loss_pct")
	sA, ssA, _, _, _ := c.measure(weights(c.nOct, 0, 1))
	sB, ssB, _, _, _ := c.measure(weights(c.nOct, 1, 0))
	fmt.Printf("drop-low,0.00,1.00,inf,%.6f,%.6f,%.3f%%\n", sA, ssA, pct(sA, sinf))
	fmt.Printf("drop-high,1.00,0.00,0.00,%.6f,%.6f,%.3f%%\n", sB, ssB, pct(sB, sinf))

	fmt.Println("\n## 对数等比档位 r=2^k 的损失（两档取整若落在这些档上）")
	fmt.Println("r,S_tile,S_slide,loss_pct")
	for _, k := range []int{0, 1, 2, 3, 4, 5, 6, 8} {
		r := math.Pow(2, float64(k))
		s, ss, _, _, _ := c.measure(weights(c.nOct, 1, r))
		fmt.Printf("%.0f,%.6f,%.6f,%.3f%%\n", r, s, ss, pct(s, sinf))
	}
	fmt.Println("## 归一化线性档位 w_low+w_high=1，n 等分")
	fmt.Println("w_low,w_high,r,S_tile,S_slide,loss_pct")
	for _, n := range []int{2, 4, 8} {
		for k := 1; k < n; k++ {
			lw := float64(k) / float64(n)
			hw := 1 - lw
			s, ss, _, _, _ := c.measure(weights(c.nOct, lw, hw))
			fmt.Printf("%.3f,%.3f,%.4f,%.6f,%.6f,%.3f%%\n",
				lw, hw, hw/lw, s, ss, pct(s, sinf))
		}
	}

	// ---- 4b. 低=oct0 / 高=oct(top) 的取整损失 ----
	sinfTop := so[top]
	fmt.Printf("\n# 4b. 低=oct0 / 高=oct%d 取整损失（基准 S_inf = 纯 oct%d = %.6f）\n", top, top, sinfTop)
	fmt.Println("scheme,r,S_tile,S_slide,loss_pct")
	for _, k := range []int{1, 2, 3, 4, 5, 6, 8, 10} {
		r := math.Pow(2, float64(k))
		w := make([]float64, c.nOct)
		w[0], w[top] = 1, r
		s, ss, _, _, _ := c.measure(w)
		fmt.Printf("1:%.0f,%.0f,%.6f,%.6f,%.3f%%\n", r, r, s, ss, pct(s, sinfTop))
	}
	fmt.Println("两档权重分配（两档 {g,1}），每种分法各算一次：")
	fmt.Println("scheme,w_low,w_high,r,S_tile,loss_pct")
	for _, g := range []float64{0.25, 0.5, 0.75} {
		for _, lo := range []float64{g, 1} {
			hiw := 1.0
			if lo == 1 {
				hiw = g
			}
			w := make([]float64, c.nOct)
			w[0], w[top] = lo, hiw
			s, _, _, _, _ := c.measure(w)
			fmt.Printf("low=%.2f,high=%.2f,%.2f,%.2f,%.3f,%.6f,%.3f%%\n",
				lo, hiw, lo, hiw, hiw/lo, s, pct(s, sinfTop))
		}
	}

	// ---- 5. 三档层数（K=2,3,4），gain=0.5，逐档对照 ----
	fmt.Println("\n# 5. 三档层数 K=2,3,4，gain=0.5：w_low:w_high = 1:0.5^(K-1)")
	fmt.Println("# S_pred_full = Σ w_i² V32_i / Σ w_i² V8_i（全部层参与）")
	fmt.Println("# S_pred_2lyr = 只取最低+最高层的同式（检验“两层权重比”能否解释）")
	fmt.Println("K,ratio_low_high,S_measured,S_pred_full,dev_full_pct,S_pred_2lyr,dev_2lyr_pct,flag,sd")
	for k := 2; k <= 4; k++ {
		w := make([]float64, c.nOct)
		for i := 0; i < k; i++ {
			w[i] = math.Pow(0.5, float64(i))
		}
		s, _, _, _, sd := c.measure(w)
		var A, B float64
		for i := 0; i < k; i++ {
			A += w[i] * w[i] * v32o[i]
			B += w[i] * w[i] * v8o[i]
		}
		Aw := w[0] * w[0] * v32o[0]
		Bw := w[0] * w[0] * v8o[0]
		Aw += w[k-1] * w[k-1] * v32o[k-1]
		Bw += w[k-1] * w[k-1] * v8o[k-1]
		devFull := pct(s, A/B)
		dev2 := pct(s, Aw/Bw)
		flag := "OK"
		if math.Abs(devFull) > 10 || math.Abs(dev2) > 10 {
			flag = "超一成"
		}
		fmt.Printf("%d,1:%.4f,%.6f,%.6f,%+.3f%%,%.6f,%+.3f%%,%s,%.4f\n",
			k, math.Pow(0.5, float64(k-1)), s, A/B, devFull, Aw/Bw, dev2, flag, sd)
	}
	fmt.Println("# 各层自身 S（单层，权重无关）：")
	for i := 0; i < c.nOct; i++ {
		fmt.Printf("#  oct%d λ=%.1fpx  S_tile=%.4f  S_slide=%.4f\n",
			i, 1/(c.freq*math.Pow(c.lac, float64(i))), so[i], sso[i])
	}

	// ---- 6. 实际可调范围：K=4 下 gain∈(0,1)，看团块强度能降到哪 ----
	// 核心恒等式：S = Σ (w_i² V8_i) S_i / Σ (w_i² V8_i)，
	// 即 S 是各层 S_i 的加权平均，权重 w_i² V8_i ≥ 0。故任意单个权重变化时
	// S 只会在“该层 S”与“其余层平均 S”之间单调移动，不存在内点极小。
	fmt.Println("\n# 6. K=4，gain 扫描（api 允许 gain∈(0,1)）")
	fmt.Println("gain,ratio_low_high,S_tile,S_slide,loss_vs_inf_pct")
	for _, g := range []float64{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99} {
		w := make([]float64, c.nOct)
		for i := 0; i < c.nOct; i++ {
			w[i] = math.Pow(g, float64(i))
		}
		s, ss, _, _, _ := c.measure(w)
		fmt.Printf("%.2f,1:%.4f,%.6f,%.6f,%.3f%%\n",
			g, math.Pow(g, 3), s, ss, pct(s, sinf))
	}
	fmt.Println("# 恒等式检验：S(r=1) 应等于两层 S 的 V8 加权平均（同一权重下）")
	w1 := v8o[0] / (v8o[0] + v8o[1])
	s11, ss11, _, _, _ := c.measure(weights(c.nOct, 1, 1))
	fmt.Printf("# 加权平均预测 %.6f  实测 %.6f  tile 差 %+.3f%%，slide 差 %+.3f%%\n",
		w1*so[0]+(1-w1)*so[1], s11, pct(s11, w1*so[0]+(1-w1)*so[1]),
		pct(ss11, w1*sso[0]+(1-w1)*sso[1]))
}

func lambdaString(c *config) string {
	s := ""
	for i := 0; i < c.nOct; i++ {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("oct%d=%.1fpx", i, 1/(c.freq*math.Pow(c.lac, float64(i))))
	}
	return s
}
