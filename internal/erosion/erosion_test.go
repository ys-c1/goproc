package erosion

import (
	"math"
	"testing"
)

func testGrid() (int, []float64) {
	// 3x3，中心为局部高点，向东侧持续降低。
	return 3, []float64{
		0.80, 0.70, 0.60,
		0.65, 0.90, 0.40,
		0.55, 0.30, 0.10,
	}
}

func TestStepAmountIsKTimesFlowTimesSlope(t *testing.T) {
	size, h := testGrid()
	f, err := BuildFields(size, h, 1)
	if err != nil {
		t.Fatal(err)
	}
	const k = 0.0125
	step, err := StepAmount(f, k)
	if err != nil {
		t.Fatal(err)
	}
	for i := range step {
		want := k * f.Flow[i] * f.Slope[i]
		if step[i] != want {
			t.Fatalf("格 %d 单步侵蚀未按 K*Q*S: got %.17g want %.17g", i, step[i], want)
		}
	}
}

func TestClosedFormMatchesFixedIteration(t *testing.T) {
	size, h := testGrid()
	f, err := BuildFields(size, h, 1)
	if err != nil {
		t.Fatal(err)
	}
	const k = 0.002
	for _, rounds := range []int{1, 2, 16, 64} {
		closed, err := ClosedFormFixed(h, f, k, rounds)
		if err != nil {
			t.Fatal(err)
		}
		iterated, err := IterateFixed(h, f, k, rounds)
		if err != nil {
			t.Fatal(err)
		}
		cmp, err := Compare(size, rounds, closed, iterated, h)
		if err != nil {
			t.Fatal(err)
		}
		if cmp.OverOneBinCount != 0 {
			t.Fatalf("rounds=%d 有 %d 格偏差超过一档灰度", rounds, cmp.OverOneBinCount)
		}
		if cmp.EncodedOverOneLevelCount != 0 {
			t.Fatalf("rounds=%d 有 %d 格灰度码差超过 1", rounds, cmp.EncodedOverOneLevelCount)
		}
		if cmp.MaxAbsHeightError > 1e-12 {
			t.Fatalf("rounds=%d 浮点误差过大: %.3g", rounds, cmp.MaxAbsHeightError)
		}
	}
}

func TestClosedFormInductionOnSyntheticFields(t *testing.T) {
	// 不依赖 D8 构造，直接指定 Q/S，验证 N 轮累加的归纳式。
	f := &Fields{
		Size:     1,
		Receiver: []int{-1},
		Slope:    []float64{0.25},
		Flow:     []float64{40},
	}
	const k = 0.01
	const n = 37
	got, err := ClosedFormFixed([]float64{1}, f, k, n)
	if err != nil {
		t.Fatal(err)
	}
	want := 1 - float64(n)*k*40*0.25
	if math.Abs(got[0]-want) > 1e-15 {
		t.Fatalf("闭式总侵蚀错误: got %.17g want %.17g", got[0], want)
	}
}

func TestRejectsBadCoefficientAndRounds(t *testing.T) {
	f := &Fields{Size: 1, Receiver: []int{-1}, Slope: []float64{0}, Flow: []float64{1}}
	if _, err := StepAmount(f, -1); err == nil {
		t.Fatal("负系数应报错")
	}
	if _, err := ClosedFormFixed([]float64{1}, f, 0.1, -2); err == nil {
		t.Fatal("负轮数应报错")
	}
}
