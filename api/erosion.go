package api

import (
	"goproc/internal/erosion"
)

// ErosionConfig 是侵蚀算子的对外参数，新增参数必须注明默认值。
type ErosionConfig struct {
	K        float64 // 通量系数（长度量纲），默认 0.05
	CellSize float64 // 格距 d，默认 1.0
	Quantum  float64 // 定点存储量化步长 q，仅定点版使用，默认 1e-4
}

// DefaultErosionConfig 返回默认侵蚀参数。
func DefaultErosionConfig() ErosionConfig {
	c := erosion.DefaultConfig()
	return ErosionConfig{K: c.K, CellSize: c.CellSize, Quantum: c.Quantum}
}

func (e ErosionConfig) internal() erosion.Config {
	return erosion.Config{K: e.K, CellSize: e.CellSize, Quantum: e.Quantum}
}

// Erode 在高度图原地执行 steps 步守恒版侵蚀（只搬运、不增删物质）。
func (h *Heightmap) Erode(steps int, cfg ErosionConfig) error {
	return erosion.StepN(h.Data, h.Size, steps, cfg.internal())
}

// ErodeNaive 在高度图原地执行 steps 步定点量化版侵蚀，返回各步
// 因独立取整丢失的高度，仅用于对照实验与文档中的误差公式验证。
func (h *Heightmap) ErodeNaive(steps int, cfg ErosionConfig) ([]float64, error) {
	return erosion.StepNNaive(h.Data, h.Size, steps, cfg.internal())
}

// StudyReport 是守恒实验报告的对外别名。
type StudyReport = erosion.Report

// VariantResultView 是单条轨迹核对结果的对外别名。
type VariantResultView = erosion.VariantResult

// ConservationStudy 在当前高度图上运行守恒实验并把侵蚀图与报告存档到 dir。
// stepList 为要核对的迭代步数清单，例如 10、50、200。
func (h *Heightmap) ConservationStudy(stepList []int, cfg ErosionConfig, dir string) (*erosion.Report, error) {
	study, err := erosion.RunStudy(h.Data, h.Size, stepList, cfg.internal())
	if err != nil {
		return nil, err
	}
	if err := study.Archive(dir, h.Size); err != nil {
		return nil, err
	}
	return &study.Report, nil
}
