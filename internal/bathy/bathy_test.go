package bathy

import "testing"

// 整除情形：Q=256, L=16，两映射必须逐点相同。
func TestDivisibleIdentical(t *testing.T) {
	const Q, L = 256, 16
	for q := 0; q < Q; q++ {
		// 取该档中点作为连续水深代表
		d := (float64(q) + 0.5) / Q
		ca := floor(d * L)
		cb := floor(float64(q) * L / Q)
		if ca != cb {
			t.Fatalf("整除仍失配 q=%d d=%.4f A=%d B=%d", q, d, ca, cb)
		}
	}
}

// 不整除情形：Q=256, L=10，构造同一点两映射给出不同色阶。
func TestNonDivisibleCounterexample(t *testing.T) {
	const Q, L = 256, 10
	// q=25 档 [25/256,26/256)=[0.09765625,0.1015625)
	// 取 d=0.100：A=floor(1.0)=1，B=floor(25*10/256)=floor(0.9766)=0
	d := 0.100
	q := int(d * Q)
	ca := floor(d * L)
	cb := floor(float64(q) * L / Q)
	if q != 25 {
		t.Fatalf("前置假设失效 q=%d", q)
	}
	if ca != 1 || cb != 0 || ca-cb != 1 {
		t.Fatalf("反例不成立 A=%d B=%d", ca, cb)
	}
}

// 不整除时色阶差按档数计恒为 1，且 B 恒浅一档；失配测度与公式一致。
func TestDeltaAndMeasure(t *testing.T) {
	const Q, L = 256, 10
	worst := 0
	for q := 0; q < Q; q++ {
		d := (float64(q) + 0.5) / Q
		ca := floor(d * L)
		cb := floor(float64(q) * L / Q)
		if ca < cb {
			t.Fatalf("出现 B 比 A 深 q=%d A=%d B=%d", q, ca, cb)
		}
		if ca-cb > worst {
			worst = ca - cb
		}
	}
	if worst != 1 {
		t.Fatalf("最大色阶差应为 1，实际 %d", worst)
	}
	m := MismatchMeasure(Q, L)
	if m <= 0 || m >= 1 {
		t.Fatalf("失配测度异常 %v", m)
	}
	if MismatchMeasure(Q, 16) != 0 {
		t.Fatal("整除时失配测度应为零")
	}
}

// Compare 的端到端检查：整除零差异，不整除有差异且包围盒合法。
func TestCompareReports(t *testing.T) {
	const size, Q = 8, 256
	// 构造若干水下属点，包含落在失配区间的 d=0.100
	depth := make([]float64, size*size)
	mask := make([]bool, size*size)
	depth[0] = 0.100
	mask[0] = true
	depth[1] = 0.500
	mask[1] = true
	qbins := Quantize(depth, Q)

	div := Compare(size, depth, mask, qbins, Q, 16)
	if div.DiffPixels != 0 {
		t.Fatal("L=16 不应有差异")
	}
	non := Compare(size, depth, mask, qbins, Q, 10)
	if non.DiffPixels != 1 || non.Mismatch[0].X != 0 || non.Mismatch[0].Y != 0 {
		t.Fatalf("L=10 应在 (0,0) 失配，实际 %+v", non.Mismatch)
	}
	if non.BBox == nil || non.BBox.X0 != 0 || non.BBox.Y1 != 0 {
		t.Fatal("包围盒错误")
	}
	if non.MaxAbsDelta != 1 {
		t.Fatalf("色阶差应为 1，实际 %d", non.MaxAbsDelta)
	}
}

func floor(x float64) int {
	i := int(x)
	if float64(i) > x {
		return i - 1
	}
	return i
}
