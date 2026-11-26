package erosion

import (
	"math"
	"testing"
)

// pyramid 构造一个单峰金字塔场：中心最高、逐层降低，全程为正，
// 且取整到量子 q 上后仍是定点数，便于做精确守恒核对。
// height 是中心峰值，决定坡度与搬运通量的量级。
func pyramid(size int, height, q float64) []float64 {
	h := make([]float64, size*size)
	center := float64(size-1) / 2.0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dist := math.Abs(float64(x)-center) + math.Abs(float64(y)-center)
			v := height - dist
			h[y*size+x] = q * math.Floor(v/q)
		}
	}
	return h
}

// TestStepConservesExactly 验证定理一：只做搬运时，每条边流入等于
// 流出，全场总高度在一步之后保持不变（定点场下连浮点残差都没有）。
func TestStepConservesExactly(t *testing.T) {
	const size = 8
	c := DefaultConfig()
	h := pyramid(size, 10, c.Quantum)
	before := Total(h)
	if err := Step(h, size, c); err != nil {
		t.Fatal(err)
	}
	after := Total(h)
	if after != before {
		t.Fatalf("单步总高度不守恒: before=%.12f after=%.12f diff=%.3e", before, after, after-before)
	}
}

// TestInflowEqualsOutflow 直接核对通量层面的守恒条件：全场总流入
// 必须逐边等于全场总流出。
func TestInflowEqualsOutflow(t *testing.T) {
	const size = 10
	c := DefaultConfig()
	h := pyramid(size, 10, c.Quantum)
	out, in := fluxes(h, size, c)
	var sumOut, sumIn float64
	for i := range h {
		sumOut += out[i]
		sumIn += in[i]
	}
	// 两边是同一组 f_ij 按不同格点顺序求和，只允许浮点求和残差。
	tol := 1e-14 * math.Max(1, math.Abs(sumOut))
	if math.Abs(sumIn-sumOut) > tol {
		t.Fatalf("流入 %.12f != 流出 %.12f，超出容差 %.1e", sumIn, sumOut, tol)
	}
}

// TestSingleEdgeTransfer 单条下降边上，授格减少量必须等于受格增加量。
func TestSingleEdgeTransfer(t *testing.T) {
	const size = 2
	c := DefaultConfig()
	h := []float64{4, 1, 1, 1}
	before := append([]float64(nil), h...)
	if err := Step(h, size, c); err != nil {
		t.Fatal(err)
	}
	gain := 0.0
	loss := 0.0
	for i := range h {
		d := h[i] - before[i]
		if d > 0 {
			gain += d
		} else {
			loss -= d
		}
	}
	if math.Abs(gain-loss) > 0 {
		t.Fatalf("搬运不配对: gain=%.12f loss=%.12f", gain, loss)
	}
	if Total(h) != Total(before) {
		t.Fatal("单条边搬运后总高度改变")
	}
}

// TestNaiveLosesMassAndBounded 验证定理二：逐格独立向下取整使每步
// 丢失量非负，且不超过 q 乘以发生搬运的格点数（即不超过 Nq）。
func TestNaiveLosesMassAndBounded(t *testing.T) {
	const size = 12
	c := DefaultConfig()
	h := pyramid(size, 10, c.Quantum)
	before := Total(h)
	lost, err := NaiveStep(h, size, c)
	if err != nil {
		t.Fatal(err)
	}
	if lost < 0 {
		t.Fatalf("截断式取整不应产生物质: lost=%.3e", lost)
	}
	got := before - Total(h)
	if math.Abs(got-lost) > 1e-12 {
		t.Fatalf("返回丢失量 %.6e 与实测 %.6e 不符", lost, got)
	}
	if bound := float64(size*size) * c.Quantum; lost > bound {
		t.Fatalf("单步丢失 %.3e 超过上界 Nq=%.3e", lost, bound)
	}
}

// TestNaiveLossIsSlopeDriven 平坦区域（无下降边）不产生任何搬运，
// 量化误差必须为零：丢失量只在坡度 s>0 的地方出现。
func TestNaiveLossIsSlopeDriven(t *testing.T) {
	const size = 6
	c := DefaultConfig()
	h := make([]float64, size*size)
	for i := range h {
		h[i] = c.Quantum * 4 // 全场等高且落在量子格点上，没有边搬运也没有存储残差
	}
	lost, err := NaiveStep(h, size, c)
	if err != nil {
		t.Fatal(err)
	}
	if lost != 0 {
		t.Fatalf("平坦场不应有丢失: %.3e", lost)
	}
}

// TestFixedConservationAtRequiredSteps 三个规定步数下，守恒版的
// 归一化总高差必须全部低于千分之一。
func TestFixedConservationAtRequiredSteps(t *testing.T) {
	const size = 64
	c := DefaultConfig()
	base := pyramid(size, 40, c.Quantum)
	for _, steps := range []int{10, 50, 200} {
		_, res, err := RunFixed(base, size, steps, c)
		if err != nil {
			t.Fatal(err)
		}
		if res.RelDiff >= ConservationThreshold {
			t.Fatalf("fixed steps=%d 归一高差 %.3e 未达阈值 %g", steps, res.RelDiff, ConservationThreshold)
		}
		if !res.Pass {
			t.Fatalf("fixed steps=%d 判定未通过", steps)
		}
	}
}

// TestNaiveFailsAtLongRun 定点版的系统性丢失随步数累积，长程运行
// 必须越过千分之一阈值，否则误差公式缺少实验对照。
func TestNaiveFailsAtLongRun(t *testing.T) {
	const size = 64
	c := DefaultConfig()
	base := pyramid(size, 40, c.Quantum)
	_, res, err := RunNaive(base, size, 200, c)
	if err != nil {
		t.Fatal(err)
	}
	if res.Pass {
		t.Fatalf("定点版 200 步意外达标: 归一高差 %.3e", res.RelDiff)
	}
	if res.FirstDeviationStep != 1 {
		t.Fatalf("截断取整应在第 1 步就出现丢失，实际第 %d 步", res.FirstDeviationStep)
	}
}

// TestStudySnapshotsAndReport 实验编排必须为每个目标步数产出快照，
// 且报告中的步数清单与判定自洽。64 见方、峰高 40 的场在 200 步时
// 定点版归一丢失越过千分之一，可同时核对两个版本的判定。
func TestStudySnapshotsAndReport(t *testing.T) {
	const size = 64
	c := DefaultConfig()
	base := pyramid(size, 40, c.Quantum)
	study, err := RunStudy(base, size, []int{10, 50, 200}, c)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []int{10, 50, 200} {
		f, ok := study.Fixed[s]
		if !ok || len(f.Height) == 0 || len(f.Erosion) == 0 {
			t.Fatalf("fixed 缺少第 %d 步存档", s)
		}
		n, ok := study.Naive[s]
		if !ok || len(n.Height) == 0 || len(n.Erosion) == 0 {
			t.Fatalf("naive 缺少第 %d 步存档", s)
		}
	}
	if !study.Report.AllFixedPass {
		t.Fatal("fixed 应全部达标")
	}
	if study.Report.AllNaivePass {
		t.Fatal("naive 不应全部达标")
	}
}

// TestRejectsBadInputs 把参数与尺寸错误挡在入口。
func TestRejectsBadInputs(t *testing.T) {
	h := pyramid(8, 10, 1e-4)
	if err := Step(h, 8, Config{K: 0, CellSize: 1, Quantum: 1e-4}); err == nil {
		t.Fatal("K 非正应报错")
	}
	if err := Step(h, 7, DefaultConfig()); err == nil {
		t.Fatal("尺寸与数据长度不匹配应报错")
	}
	if _, _, err := RunFixed(h, 8, 0, DefaultConfig()); err == nil {
		t.Fatal("步数为零应报错")
	}
}
