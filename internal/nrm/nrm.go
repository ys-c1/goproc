// Package nrm 推导并反解把高度图转成法线时用的差分核，并给出与中心差分的
// 角误差对照。
//
// 记格距 d、高度 h(x,y)。法线 n ∝ (-h_x,-h_y,1)，所以法线角误差由梯度估计误差
// 决定。差分核把梯度估成
//
//	ĝ_x = (1/d) Σ_k w_k h(x + k d, y)
//
// 把 h(x+kd) 在 x 处展开：
//
//	ĝ_x - h_x = (1/d)·[ h Σ_k w_k + d h_x Σ_k k w_k + (d²/2) h_xx Σ_k k² w_k
//	                    + (d³/6) h_xxx Σ_k k³ w_k + O(d⁴) ]
//
// 约束「权重和为零」(Σ_k w_k = 0) 与「反对称」(w_{-k} = -w_k) 中，反对称本身
// 就蕴含和为零，并使全部偶数阶矩消失——包括真正的高度二阶项 h_xx·Σ_k k²w_k。
// 于是残差首个存活项来自 h_xxx：
//
//	ĝ_x - h_x = (d²/6) (Σ_k w_k k³) h_xxx + O(d⁴).          (★)
//
// (★) 是 O(d²) 的残差，每个抽头 k 的贡献是 w_k·(k³ d²/6) h_xxx，所有抽头
// 共享同一个 h_xxx(x)，因此**相干相加**，该点残差正比于 S3 := Σ_k w_k k³。
//
// 单点的法线角误差平方 θ² 正比于 (★) 的平方，即正比于 S3²·h_xxx²；
// 对整幅图所有采样点的 θ² 求和（「平方和」是对采样点求和）：
//
//	E(w) = Σ_点 θ² = S3² · Σ_点 (d⁴/36) h_xxx² / (1+|∇h|²)²
//	              = S3² · C,   C > 0 且与 w 无关。
//
// 所以最小化总平方角误差等价于最小化 (Σ_k w_k k³)²，即反解条件 S3 = 0。
//
// 注意：这里是「逐点先相干求和、再对点平方」，不是「逐抽头平方后相加」。
// 若误用后者 Σ_k (w_k k³)²，得到的核在两档格距下的角误差都**大于**中心差分
// （见 SolvePerTap 与测试），不满足验收；相干形式才给出正确解。
//
// 仅凭「和为零 + 反对称」会把 w=0 也当成最优（平凡解），因此还必须加上
// 「一阶矩为一」(Σ_k k w_k = 1)——这才是核确实近似一阶导数的条件。
// 于是反解就是在
//
//	反 对 称：w_{-k} = -w_k
//	归一化：  Σ_k k w_k = 1
//	反解条件：Σ_k w_k k³ = 0
//
// 下求最小范数解。核长 3（半长 m=1）时自由权重只有一个、约束 u₁=1/2 唯一确定
// w=[-1/2,0,1/2]，即中心差分；此时 S3=1≠0，中心差分是该核长下唯一可行点，
// 因而就是「特例」。核长 5（m=2）时解为经典五点核 [1/12,-2/3,0,2/3,-1/12]，
// 其 S3=0，O(d²) 残差消失（误差降到 O(d⁴)）。
package nrm

import (
	"fmt"
	"math"
)

// Moment 返回差分核的 p 阶矩 Σ_k w_k k^p，k 为相对中心的整数偏移。
func Moment(w []float64, p int) float64 {
	if len(w) == 0 || len(w)%2 == 0 {
		return math.NaN()
	}
	m := len(w) / 2
	s := 0.0
	for i, wi := range w {
		s += wi * math.Pow(float64(i-m), float64(p))
	}
	return s
}

// Objective 是相干反解目标 E(w) = (Σ_k w_k k³)²，即 (★) 中 O(d²) 残差系数的平方。
func Objective(w []float64) float64 {
	s := Moment(w, 3)
	return s * s
}

// CentralDifference 是核长三的中心差分核。
func CentralDifference() []float64 { return []float64{-0.5, 0, 0.5} }

// Solve 反解半长为 m（核长 2m+1）的相干最小核：反对称、归一化、且 S3=0。
//
// 设自由权重 u_k = w_k（k=1..m）。反对称自动满足和为零；约束为
//
//	Σ_k k  u_k = 1/2    （归一化，注意两端各算一次）
//	Σ_k k³ u_k = 0      （反解条件 S3=0）
//
// m≥2 时两行线性无关，最小范数解由拉格朗日乘子给出。m=1 时两行成比例，
// S3=0 不可达，可行集退化为单点 u₁=1/2，即中心差分。
func Solve(m int) ([]float64, error) {
	u, err := minNorm(m, true)
	if err != nil {
		return nil, err
	}
	w := make([]float64, 2*m+1)
	for k := 1; k <= m; k++ {
		w[m+k] = u[k-1]
		w[m-k] = -u[k-1]
	}
	return w, nil
}

// SolvePerTap 反解「逐抽头平方」目标 Σ_k (w_k k³)² 下的最小核，仅用于对照。
// 结果在两种格距下的角误差都大于中心差分，说明相干形式才是正确目标。
func SolvePerTap(m int) ([]float64, error) {
	if m < 1 || m > 8 {
		return nil, fmt.Errorf("nrm: 核半长 %d 越界（支持 1..8）", m)
	}
	// 最小化 Σ (k³ u_k)² s.t. Σ k u_k = 1/2，得 u_k ∝ 1/k⁵。
	var inv4 float64
	for k := 1; k <= m; k++ {
		inv4 += 1 / math.Pow(float64(k), 4)
	}
	u := make([]float64, m)
	for k := 1; k <= m; k++ {
		u[k-1] = 1 / (2 * math.Pow(float64(k), 5) * inv4)
	}
	w := make([]float64, 2*m+1)
	for k := 1; k <= m; k++ {
		w[m+k] = u[k-1]
		w[m-k] = -u[k-1]
	}
	return w, nil
}

// minNorm 在约束 Σ k u_k = 1/2、可选再加 Σ k³ u_k = 0 下求 ‖u‖ 最小的 u。
func minNorm(m int, withS3 bool) ([]float64, error) {
	if m < 1 || m > 8 {
		return nil, fmt.Errorf("nrm: 核半长 %d 越界（支持 1..8）", m)
	}
	k := make([]float64, m)
	for i := 1; i <= m; i++ {
		k[i-1] = float64(i)
	}
	if !withS3 || m == 1 {
		// 仅归一化：u_k ∝ k。
		var dd float64
		for i := 0; i < m; i++ {
			dd += k[i] * k[i]
		}
		u := make([]float64, m)
		for i := 0; i < m; i++ {
			u[i] = k[i] * 0.5 / dd
		}
		return u, nil
	}
	// 两约束最小范数：u = Cᵀλ，其中 C 的行是 [k³; k]，解 C Cᵀ λ = [0; 1/2]。
	var cc, cd, dd float64
	for i := 0; i < m; i++ {
		c3 := math.Pow(k[i], 3)
		cc += c3 * c3
		cd += c3 * k[i]
		dd += k[i] * k[i]
	}
	det := cc*dd - cd*cd
	if math.Abs(det) <= 1e-12*cc*dd {
		return nil, fmt.Errorf("nrm: 约束退化（m=%d）", m)
	}
	lam1 := (-cd) * 0.5 / det
	lam2 := cc * 0.5 / det
	u := make([]float64, m)
	for i := 0; i < m; i++ {
		u[i] = math.Pow(k[i], 3)*lam1 + k[i]*lam2
	}
	return u, nil
}
