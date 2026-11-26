package gully

// 本文件给出本次实验的固定判定阈值与两组侵蚀强度，
// cmd/gully 与单元测试共用，避免“报告参数”和“验证参数”不一致。

// DCore 是侵蚀后“实测核心格”的下切深度检测杠（归一化高程单位）。
// 取 0.005：远大于浮点累计误差，且约为一个 PGM 灰度档 1/255≈0.00392 的 1.3 倍。
const DCore = 0.005

// EpsEroded 是“该格确实被侵蚀过”的深度下限。
const EpsEroded = 1e-12

// CanonicalThresholds 返回双阈值合取判定所用的两个阈值。
// 在 size=128 seed=7 的基准地形上，该组合给出 69 个核心格、25 个八邻域段。
func CanonicalThresholds() Thresholds {
	return Thresholds{Acc: 32, Slope: 0.030} // 判定为 Acc>32 且 Slope>0.030
}

// IntensityGroups 返回弱、强两组侵蚀强度；除 K 与沟岸宽度量外完全同参。
// Rounds 内汇流/坡度场冻结，故轮数只线性放大下切量。
func IntensityGroups() []ErodeParams {
	return []ErodeParams{
		{
			Name: "weak", Label: "弱强度组",
			K: 0.0015, Rounds: 10,
			CapAccRatio: 8, CapSlopeRatio: 8,
			BankPerNeighbor: 0.0006, BankCap: 0.0012,
			Floor: 0,
		},
		{
			Name: "strong", Label: "高强度组",
			K: 0.0045, Rounds: 10,
			CapAccRatio: 8, CapSlopeRatio: 8,
			BankPerNeighbor: 0.0015, BankCap: 0.0030,
			Floor: 0,
		},
	}
}
