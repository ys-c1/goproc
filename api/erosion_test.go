package api

import (
	"testing"
)

// TestErodeConservesTotal 对外入口在三个规定步数下的归一化总高差
// 必须全部低于千分之一。
func TestErodeConservesTotal(t *testing.T) {
	h, err := Generate(DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	initial := sumHeight(h.Data)
	for _, steps := range []int{10, 50, 200} {
		fresh, err := Generate(DefaultOptions())
		if err != nil {
			t.Fatal(err)
		}
		if err := fresh.Erode(steps, DefaultErosionConfig()); err != nil {
			t.Fatal(err)
		}
		rel := abs(sumHeight(fresh.Data)-initial) / initial
		if rel >= 1e-3 {
			t.Fatalf("steps=%d 归一高差 %.3e 超过千分之一", steps, rel)
		}
	}
}

// TestConservationStudyArchives 实验入口必须返回自洽报告。
func TestConservationStudyArchives(t *testing.T) {
	h, err := Generate(DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	report, err := h.ConservationStudy([]int{10, 50, 200}, DefaultErosionConfig(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if !report.AllFixedPass {
		t.Fatal("守恒版三个步数应全部达标")
	}
	if report.AllNaivePass {
		t.Fatal("定点量化版应至少有一个步数不达标")
	}
}

func sumHeight(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
