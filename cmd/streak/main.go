// Command streak 渲染地形镜面高光包络，测量亮斑长宽比，
// 并按各向异性关系 AR = (σ_x/σ_y)·secθ 反解坡度比，与真实梯度对比。
//
// 模型：光与视点关于天顶对称、倾角均为 θ，视点距 D（透视投影）。
// 微面元斜率容差为各向异性高斯（Cox–Munk 型包络）：
//   I(x,y) = exp( -[(p̄-p*)²/(2σ_x²) + (q̄-q*)²/(2σ_y²)] )
// p̄,q̄ 为宏观地形坡度，(p*,q*) 为该像素精确所需的镜面斜率，
// σ_x, σ_y 为微粗糙 RMS 坡度（从高度图梯度实测）。
// 线性化给出 ds*_x = cosθ·dx/(2D)，ds*_y = dy/(2D·cosθ)，
// 故亮斑半轴 a_x = 2Dσ_x·secθ·√c，a_y = 2Dσ_y·cosθ·√c，
// 长宽比 AR = (σ_x/σ_y)·sec²θ，反解 σ_x/σ_y = AR·cos²θ。
package main

import (
	"fmt"
	"math"
	"os"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

const (
	n        = 256   // 边长像素
	dEye     = 1200  // 视点距离（像素）
	aniso    = 1.5   // 微粗糙 x 向采样拉伸比，决定 σ_x/σ_y ≈ aniso
	targetSx = 0.01  // 目标微粗糙 x 向 RMS 坡度
	zMacro   = 0.2   // 宏观起伏幅度（像素高）
)

// buildField 生成零均值分形噪声场，采样坐标 (x*fx*freq, y*fy*freq)。
// fx != fy 时场在统计上各向异性：σ_x/σ_y = fx/fy。
func buildField(seed uint64, octaves int, freq, fx, fy float64) []float64 {
	src := noise.New(seed)
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			out[y*n+x] = fbm.Eval(src, float64(x)*freq*fx, float64(y)*freq*fy, octaves, 2.0, 0.5)
		}
	}
	hmap.Normalize(out)
	mean := 0.0
	for _, v := range out {
		mean += v
	}
	mean /= float64(len(out))
	for i := range out {
		out[i] -= mean
	}
	return out
}

// grad 返回中心差分坡度场 (p, q)，分别为 x（水平）向与 y（竖直）向。
func grad(h []float64) ([]float64, []float64) {
	p := make([]float64, n * n)
	q := make([]float64, n * n)
	at := func(x, y int) float64 {
		if x < 0 {
			x = 0
		}
		if x >= n {
			x = n - 1
		}
		if y < 0 {
			y = 0
		}
		if y >= n {
			y = n - 1
		}
		return h[y*n+x]
	}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			p[y*n+x] = (at(x+1, y) - at(x-1, y)) / 2
			q[y*n+x] = (at(x, y+1) - at(x, y-1)) / 2
		}
	}
	return p, q
}

func rms(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s / float64(len(v)))
}

func rmsOver(v []float64, idx []int) float64 {
	s := 0.0
	for _, j := range idx {
		s += v[j] * v[j]
	}
	return math.Sqrt(s / float64(len(idx)))
}

type vec3 struct{ x, y, z float64 }

func (a vec3) sub(b vec3) vec3 { return vec3{a.x - b.x, a.y - b.y, a.z - b.z} }
func (a vec3) add(b vec3) vec3 { return vec3{a.x + b.x, a.y + b.y, a.z + b.z} }
func (a vec3) norm() vec3 {
	l := math.Sqrt(a.x*a.x + a.y*a.y + a.z*a.z)
	return vec3{a.x / l, a.y / l, a.z / l}
}

// render 计算高光包络强度图。theta 为视线倾角（光与视点关于天顶对称）。
func render(theta float64, pm, qm, h []float64, sx, sy float64) []float64 {
	st, ct := math.Sin(theta), math.Cos(theta)
	L := vec3{-st, 0, ct}
	E := vec3{dEye * st, 0, dEye * ct}
	c := float64(n-1) / 2
	img := make([]float64, n*n)
	for iy := 0; iy < n; iy++ {
		for ix := 0; ix < n; ix++ {
			P := vec3{float64(ix) - c, float64(iy) - c, h[iy*n+ix]}
			V := E.sub(P).norm()
			H := L.add(V).norm()
			// 所需镜面斜率：法线 (−p,−q,1) 与 H 对齐
			ps, qs := -H.x/H.z, -H.y/H.z
			dx := pm[iy*n+ix] - ps
			dy := qm[iy*n+ix] - qs
			img[iy*n+ix] = math.Exp(-(dx*dx/(2*sx*sx) + dy*dy/(2*sy*sy)))
		}
	}
	return img
}

// components 返回亮度超过阈值的 8-连通段（每段为像素索引列表）。
func components(img []float64, thr float64) [][]int {
	vis := make([]bool, n*n)
	var comps [][]int
	for i := 0; i < n*n; i++ {
		if vis[i] || img[i] < thr {
			continue
		}
		vis[i] = true
		queue := []int{i}
		var comp []int
		for len(queue) > 0 {
			j := queue[len(queue)-1]
			queue = queue[:len(queue)-1]
			comp = append(comp, j)
			x, y := j%n, j/n
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx, ny := x+dx, y+dy
					if nx < 0 || nx >= n || ny < 0 || ny >= n {
						continue
					}
					k := ny*n + nx
					if !vis[k] && img[k] >= thr {
						vis[k] = true
						queue = append(queue, k)
					}
				}
			}
		}
		comps = append(comps, comp)
	}
	return comps
}

// aspect 用二阶矩估计连通段的主轴/次轴之比（不用外接框）。
// 返回长宽比、质心、两个特征值与主轴方位角。
func aspect(comp []int) (ar, cx, cy, l1, l2, phi float64) {
	m := float64(len(comp))
	for _, j := range comp {
		cx += float64(j % n)
		cy += float64(j / n)
	}
	cx /= m
	cy /= m
	var sxx, syy, sxy float64
	for _, j := range comp {
		dx := float64(j%n) - cx
		dy := float64(j/n) - cy
		sxx += dx * dx
		syy += dy * dy
		sxy += dx * dy
	}
	sxx /= m
	syy /= m
	sxy /= m
	tr := (sxx + syy) / 2
	det := math.Sqrt((sxx-syy)*(sxx-syy)/4 + sxy*sxy)
	l1, l2 = tr+det, tr-det
	phi = math.Atan2(2*sxy, sxx-syy) / 2 * 180 / math.Pi
	return math.Sqrt(l1 / l2), cx, cy, l1, l2, phi
}

func main() {
	// 宏观地形：低频各向同性；微粗糙：x 向拉伸 aniso 倍 → σ_x/σ_y ≈ aniso。
	macro := buildField(7, 2, 0.006, 1, 1)
	micro := buildField(11, 3, 0.06, aniso, 1)

	// 微粗糙幅度按目标 σ_x 自动标定（确定性）。
	mp, mq := grad(micro)
	sx0, sy0 := rms(mp), rms(mq)
	zMicro := targetSx / sx0
	for i := range micro {
		micro[i] *= zMicro
	}
	sx, sy := sx0*zMicro, sy0*zMicro

	h := make([]float64, n*n)
	for i := range h {
		h[i] = zMacro*macro[i] + micro[i]
	}
	fp, fq := grad(h)   // 全地形梯度（实际坡度）
	gp, gq := grad(macro) // 宏观坡度（包络中心用）
	for i := range gp {
		gp[i] *= zMacro
		gq[i] *= zMacro
	}
	// 微粗糙-only 梯度（诊断用）
	up, uq := grad(micro)

	fmt.Printf("微粗糙 RMS 坡度: σ_x=%.5f σ_y=%.5f  比值 σ_x/σ_y=%.4f (构造值 %.2f)\n",
		sx, sy, sx/sy, aniso)
	fmt.Printf("宏观 RMS 坡度: x=%.5f y=%.5f\n\n", rms(gp), rms(gq))

	ars := make(map[float64]float64)
	for _, deg := range []float64{30, 60} {
		theta := deg * math.Pi / 180
		img := render(theta, gp, gq, h, sx, sy)
		maxI := 0.0
		for _, v := range img {
			if v > maxI {
				maxI = v
			}
		}
		thr := 0.5 * maxI
		comps := components(img, thr)
		// 取最大连通段作为高光斑
		bi := 0
		for i := range comps {
			if len(comps[i]) > len(comps[bi]) {
				bi = i
			}
		}
		comp := comps[bi]
		ar, cx, cy, l1, l2, phi := aspect(comp)
		ars[deg] = ar

		ct := math.Cos(theta)
		rInv := ar * ct * ct // 反解坡度比 = AR·cos²θ
		rAct := rmsOver(fp, comp) / rmsOver(fq, comp) // 实际梯度坡度比
		arPred := (sx / sy) / (ct * ct)
		dev := (rInv - rAct) / rAct * 100

		fmt.Printf("θ=%.0f°  阈值=%.3f  连通段数=%d  主斑像素=%d\n", deg, thr, len(comps), len(comp))
		fmt.Printf("  质心=(%.1f, %.1f)  半轴 2√λ1=%.1f 2√λ2=%.1f  主轴方位=%.1f°\n", cx, cy, 2*math.Sqrt(l1), 2*math.Sqrt(l2), phi)
		fmt.Printf("  长宽比实测 AR=%.4f   公式预测 (σ_x/σ_y)sec²θ=%.4f\n", ar, arPred)
		fmt.Printf("  反解坡度比 AR·cos²θ=%.4f   实际梯度坡度比=%.4f   偏差=%+.2f%%\n", rInv, rAct, dev)
		fmt.Printf("  诊断: 亮斑处微粗糙-only比=%.4f  亮斑处宏观RMS(x,y)=(%.5f,%.5f)  全图全梯度比=%.4f\n\n",
			rmsOver(up, comp)/rmsOver(uq, comp), rmsOver(gp, comp), rmsOver(gq, comp), rms(fp)/rms(fq))

		if err := os.WriteFile(fmt.Sprintf("artifacts/streak_%02d.pgm", int(deg)), hmap.EncodePGM(n, img), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write pgm:", err)
		}
	}
	fmt.Printf("AR(60°)/AR(30°) 实测=%.4f   (sec60/sec30)² 理论=%.4f\n",
		ars[60]/ars[30], math.Pow(math.Cos(30*math.Pi/180)/math.Cos(60*math.Pi/180), 2))
}
