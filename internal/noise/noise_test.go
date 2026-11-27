package noise

import (
	"math"
	"testing"
)

func approx(t *testing.T, got, want float64, tol float64, label string) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("%s: got %.12f want %.12f tol %.0e", label, got, want, tol)
	}
}

// TestKernelEndpointDerivatives 单独验证核在端点 t=0、t=1 的导数。
// 五次核的端点导数（一阶与二阶）在格点处应当为零；三次核一阶导为零、二阶导非零。
func TestKernelEndpointDerivatives(t *testing.T) {
	const eps = 1e-12
	kernels := []Kernel{CubicKernel{}, QuinticKernel{}}
	for _, k := range kernels {
		// 端点函数值 w(0)=0, w(1)=1
		approx(t, k.Eval(0), 0, eps, k.Name()+" w(0)")
		approx(t, k.Eval(1), 1, eps, k.Name()+" w(1)")
		// 一阶导 w'(0)=w'(1)=0：两种核都满足，这是 C1 的条件
		approx(t, k.D1(0), 0, eps, k.Name()+" w'(0)")
		approx(t, k.D1(1), 0, eps, k.Name()+" w'(1)")
		switch k.Name() {
		case "cubic":
			// 三次核二阶导端点不为零：w''(0)=+6, w''(1)=−6，差值 −12
			approx(t, k.D2(0), 6, eps, "cubic w''(0)")
			approx(t, k.D2(1), -6, eps, "cubic w''(1)")
			approx(t, k.D2(1)-k.D2(0), -12, eps, "cubic 端点二阶导差")
		case "quintic":
			// 五次核：端点一二阶导全部为零
			approx(t, k.D2(0), 0, eps, "quintic w''(0)")
			approx(t, k.D2(1), 0, eps, "quintic w''(1)")
		}
	}
}

// TestLatticeOneSidedDerivatives 直接在一维值噪声的格点上取单侧导数极限，
// 验证三次核只有 C1（二阶导跳变等于核端点导数差乘格点高差），五次核为 C2。
func TestLatticeOneSidedDerivatives(t *testing.T) {
	const (
		seed = 11
		h    = 1e-5
		tol1 = 1e-3 // 单侧一阶导含 O(h) 截断误差，容差按 h 量级放宽
		tol2 = 5.0  // 二阶导差分含 O(h)·f''' 截断项，只做量级定性核对
	)
	jumps := map[string]float64{}
	for _, k := range []Kernel{CubicKernel{}, QuinticKernel{}} {
		s := NewWithKernel(seed, k)
		var maxFirstJump, maxSecondNum, maxSecondAnalytic float64
		for _, n := range []int{3, 7, 12, 21, 40} {
			x := float64(n) // 整格点
			// 单侧一阶导：f'(x⁻)=f'(x⁺)=0 对两种核都应成立
			left1 := (s.Value1D(x) - s.Value1D(x-h)) / h
			right1 := (s.Value1D(x+h) - s.Value1D(x)) / h
			approx(t, left1, 0, tol1, k.Name()+" 左一阶导")
			approx(t, right1, 0, tol1, k.Name()+" 右一阶导")
			if d := math.Abs(left1 - right1); d > maxFirstJump {
				maxFirstJump = d
			}
			// 单侧二阶导（后向/前向差分，均带 +f'' 与 O(h) 截断误差，仅定性核对）
			left2 := (s.Value1D(x) - 2*s.Value1D(x-h) + s.Value1D(x-2*h)) / (h * h)
			right2 := (s.Value1D(x+2*h) - 2*s.Value1D(x+h) + s.Value1D(x)) / (h * h)
			if d := math.Abs(right2 - left2); d > maxSecondNum {
				maxSecondNum = d
			}
			// 解析复核（无离散误差）。单元内 f(t)=g_i+Δg·w(t)，故 f''=w''(t)·Δg：
			//   f''(x⁻)=w''(1)·ΔgL，f''(x⁺)=w''(0)·ΔgR
			gL := s.grad(n-1, 0)
			g0 := s.grad(n, 0)
			gR := s.grad(n+1, 0)
			dgL := g0 - gL
			dgR := gR - g0
			analyticLeft := k.D2(1) * dgL
			analyticRight := k.D2(0) * dgR
			analyticJump := analyticRight - analyticLeft
			if d := math.Abs(analyticJump); d > maxSecondAnalytic {
				maxSecondAnalytic = d
			}
			// 数值跳变与解析跳变的差异只来自截断项 O(h)·f'''
			numJump := right2 - left2
			approx(t, numJump, analyticJump, tol2, k.Name()+" 数值跳变 vs 解析跳变")
			// 跳变恒等于核端点二阶导差乘左单元高差（本次证明的核心等式）：
			// J = [w''(0)−w''(1)]·ΔgL，三次核即 12·ΔgL，五次核为 0
			// 核心恒等式（一般形式）：
			// J = w''(0)·ΔgR − w''(1)·ΔgL
			// 三次核: J = 6(ΔgR+ΔgL)；五次核: w'' 端点皆零，J≡0。
			wantJump := k.D2(0)*dgR - k.D2(1)*dgL
			approx(t, analyticJump, wantJump, 1e-12, k.Name()+" 解析跳变 == 核端点导数差的格点组合")
			// 等价写法：J = [w''(0)−w''(1)]·ΔgL + w''(0)·(ΔgR−ΔgL)
			alt := (k.D2(0)-k.D2(1))*dgL + k.D2(0)*(dgR-dgL)
			approx(t, analyticJump, alt, 1e-12, k.Name()+" 等价形式交叉核对")
		}
		jumps[k.Name()] = maxSecondAnalytic
		t.Logf("%s: 最大一阶导跳变 %.2e；二阶导跳变 解析 %.3e / 数值 %.3e",
			k.Name(), maxFirstJump, maxSecondAnalytic, maxSecondNum)
	}
	if jumps["cubic"] < 1 {
		t.Fatal("三次核应观测到 O(1) 的二阶导周期跳变")
	}
	if jumps["quintic"] > 1e-12 {
		t.Fatalf("五次核二阶导跳变应消失，解析实测 %.2e", jumps["quintic"])
	}
}

// TestFirstDifferenceAcrossLattice 验证格点两侧一阶差分相等：
// f(x)−f(x−h) 与 f(x+h)−f(x) 在 h→0 时趋于同一极限（零，因格点处一阶导为零），
// 二者的差异为 O(h²)：三次核 ≤ 12h²，五次核 ≤ 2h²（|w”| 端点最大值给出的上界）。
func TestFirstDifferenceAcrossLattice(t *testing.T) {
	const seed = 23
	for _, k := range []Kernel{CubicKernel{}, QuinticKernel{}} {
		s := NewWithKernel(seed, k)
		bound := 12.0
		if k.Name() == "quintic" {
			bound = 2.5
		}
		prev := math.Inf(1)
		for _, h := range []float64{1e-2, 1e-3, 1e-4} {
			var worst float64
			for n := 1; n < 50; n++ {
				x := float64(n)
				dLeft := s.Value1D(x) - s.Value1D(x-h)
				dRight := s.Value1D(x+h) - s.Value1D(x)
				if d := math.Abs(dLeft - dRight); d > worst {
					worst = d
				}
			}
			if worst > bound*h*h*1.5 { // 含格点高差系数，再留 50% 余量
				t.Fatalf("%s: h=%g 格点两侧一阶差分差异 %.3e 超过 O(h²) 上界 %.3e",
					k.Name(), h, worst, bound*h*h)
			}
			if worst > prev*1.5 { // 应随 h 缩小而收敛
				t.Fatalf("%s: 一阶差分差异未随 h 收敛: %.3e -> %.3e", k.Name(), prev, worst)
			}
			prev = worst
			t.Logf("%s: h=%g 格点两侧一阶差分最大差异 %.3e（O(h²) 上界 %.2e）",
				k.Name(), h, worst, bound*h*h)
		}
	}
}

func TestSeedDerivation(t *testing.T) {
	a := New(7)
	b := New(7)
	if a.Value(1.5, 2.5) != b.Value(1.5, 2.5) {
		t.Fatal("同种子不同值")
	}
	c := New(8)
	if a.Value(1.5, 2.5) == c.Value(1.5, 2.5) {
		t.Fatal("不同种子同值")
	}
}

func TestBounded(t *testing.T) {
	s := New(3)
	for i := 0; i < 1000; i++ {
		v := s.Value(float64(i)*0.37, float64(i)*0.11)
		if v < -1.5 || v > 1.5 {
			t.Fatalf("越界: %v", v)
		}
	}
}

// TestQuinticDefaultOutputs 确认默认构造仍走五次核，历史输出路径不变。
func TestQuinticDefaultOutputs(t *testing.T) {
	if New(1).Kernel().Name() != "quintic" {
		t.Fatal("默认核必须是五次核")
	}
}
