package analyze

import (
	"os"
	"path/filepath"
	"testing"

	"goproc/api"
)

func TestRunWritesArtifacts(t *testing.T) {
	dir := t.TempDir()
	opt := api.DefaultOptions()
	opt.Size = 64
	rep, err := Run(opt, dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Rows8) != 8 || len(rep.Rows16) != 16 {
		t.Fatalf("箱数错误: %d %d", len(rep.Rows8), len(rep.Rows16))
	}
	// 每个表的占比必须归一。
	assertSum := func(rows []Row) {
		var s float64
		for _, r := range rows {
			s += r.Measured
			if r.PredSpectral < 0 || r.PredSpectral > 1 {
				t.Fatalf("预判占比越界: %v", r.PredSpectral)
			}
		}
		if abs(s-1) > 1e-9 {
			t.Fatalf("实测占比和 %v ≠ 1", s)
		}
	}
	assertSum(rep.Rows8)
	assertSum(rep.Rows16)
	// 标注逻辑与偏差数值一致。
	for _, r := range rep.Rows8 {
		if r.FlagSpec != (abs(r.DevSpecPP) > ThresholdPP) {
			t.Fatalf("谱预判标注与偏差不一致: %+v", r)
		}
		if r.FlagIso != (abs(r.DevIsoPP) > ThresholdPP) {
			t.Fatalf("基线标注与偏差不一致: %+v", r)
		}
	}
	// 文件落盘齐全。
	for _, name := range []string{"orientation_table_8bin.csv", "orientation_table_16bin.csv", "analysis_summary.json"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("缺少产物 %s: %v", name, err)
		}
	}
	if rep.Beta.Beta <= 0 || rep.Beta.RSquared <= 0 || rep.Beta.RSquared > 1 {
		t.Fatalf("谱斜率拟合异常: %+v", rep.Beta)
	}
}

func TestRunRejectsBadOptions(t *testing.T) {
	opt := api.DefaultOptions()
	opt.Size = 0
	if _, err := Run(opt, ""); err == nil {
		t.Fatal("非法尺寸应报错")
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
