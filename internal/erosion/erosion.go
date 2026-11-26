// Package erosion 实现只搬运、不增删物质的热侵蚀（岩屑坡）算子。
//
// 算子在正方形格网上以 Jacobi 方式推进：先用旧高度场算出全部有向
// 下降边的通量，再统一更新。边界闭合，格点不存在的邻域一律跳过，
// 因此没有物质流出计算域。
package erosion

import (
	"errors"
	"fmt"
	"math"
)

// ConservationThreshold 是归一化总高度差的验收阈值：千分之一。
const ConservationThreshold = 1e-3

// Config 控制侵蚀算子。新增参数必须注明默认值。
type Config struct {
	K        float64 // 通量系数，量纲为长度，默认 0.05
	CellSize float64 // 格距 d，默认 1.0
	Quantum  float64 // 定点存储量化步长 q，仅 NaiveStep 使用，默认 1e-4
}

// DefaultConfig 返回默认参数。
func DefaultConfig() Config {
	return Config{K: 0.05, CellSize: 1.0, Quantum: 1e-4}
}

// ErrBadConfig 表示侵蚀参数越界。
var ErrBadConfig = errors.New("侵蚀参数非法")

func (c Config) validate() error {
	if c.K <= 0 {
		return fmt.Errorf("%w: K=%v 必须为正", ErrBadConfig, c.K)
	}
	if c.CellSize <= 0 {
		return fmt.Errorf("%w: CellSize=%v 必须为正", ErrBadConfig, c.CellSize)
	}
	if c.Quantum <= 0 {
		return fmt.Errorf("%w: Quantum=%v 必须为正", ErrBadConfig, c.Quantum)
	}
	return nil
}

// Total 返回全场总高度 H = Σ h_i。格距 d 给出的单元格面积 d² 是
// 所有格点的公因子，归一化时约去，故总质量直接用总高度代表。
func Total(h []float64) float64 {
	sum := 0.0
	for _, v := range h {
		sum += v
	}
	return sum
}

// fluxes 计算每条有向下降边的搬运通量。
//
// 对邻接格点 i、j，坡度 s_ij = (h_i-h_j)/d 仅在 h_i>h_j 时为正，
// 通量取
//
//	f_ij = min(K*s_ij, (h_i-h_j)/2)
//
// 后半段是稳定性上限：单次搬运至多移走落差的一半。四邻域遍历中
// 每条无向边只会产生一个有向通量（低处回看时 dh<=0 被跳过）。
// 返回每格的总流出与总流入，两个切片按有向边共享同一个 f_ij。
func fluxes(h []float64, size int, c Config) (out, in []float64) {
	out = make([]float64, len(h))
	in = make([]float64, len(h))
	dh := [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			hi := h[i]
			for _, off := range dh {
				nx, ny := x+off[0], y+off[1]
				if nx < 0 || nx >= size || ny < 0 || ny >= size {
					continue // 闭合边界：没有边通向域外
				}
				drop := hi - h[ny*size+nx]
				if drop <= 0 {
					continue // 只往低处搬
				}
				f := c.K * (drop / c.CellSize)
				if cap := drop * 0.5; f > cap {
					f = cap
				}
				out[i] += f
				in[ny*size+nx] += f
			}
		}
	}
	return out, in
}

// Step 是守恒版单步推进：同一条边的通量从授格扣除、原样加到受格，
// 流入恒等于流出。剩余误差只来自 float64 四则运算本身的舍入。
func Step(h []float64, size int, c Config) error {
	if err := c.validate(); err != nil {
		return err
	}
	if size <= 0 || len(h) != size*size {
		return fmt.Errorf("%w: 尺寸 %d 与数据长度 %d 不匹配", ErrBadConfig, size, len(h))
	}
	out, in := fluxes(h, size, c)
	for i := range h {
		h[i] += in[i] - out[i]
	}
	return nil
}

// NaiveStep 模拟带定点存储缺陷的实现：净通量按浮点算完后，每格
// 新高度被独立量化到量子 q 上（向下取整）再存回。
//
// 各格独立取整使授格与受格的残差无法配对抵消，本步丢失的高度
// （非负）作为返回值；它只在真正发生搬运的格点上非零。
func NaiveStep(h []float64, size int, c Config) (float64, error) {
	if err := c.validate(); err != nil {
		return 0, err
	}
	if size <= 0 || len(h) != size*size {
		return 0, fmt.Errorf("%w: 尺寸 %d 与数据长度 %d 不匹配", ErrBadConfig, size, len(h))
	}
	out, in := fluxes(h, size, c)
	lost := 0.0
	q := c.Quantum
	for i := range h {
		u := h[i] + in[i] - out[i]
		stored := q * math.Floor(u/q)
		lost += u - stored
		h[i] = stored
	}
	return lost, nil
}

// VariantResult 是一条侵蚀轨迹相对初始场的守恒核对结果。
type VariantResult struct {
	Steps              int     // 迭代步数
	FinalTotal         float64 // 末态总高度
	AbsDiff            float64 // |H_steps - H_0|
	RelDiff            float64 // |H_steps - H_0| / H_0，按总高度归一
	FirstDeviationStep int     // 首次出现非零偏差的迭代步，0 表示全程无偏差
	FirstDeviationAbs  float64 // 首次偏差的绝对量级
	Pass               bool    // 归一化守恒差是否低于千分之一
}

func newResult(steps int, initialTotal, finalTotal float64, firstStep int, firstAbs float64) VariantResult {
	abs := math.Abs(finalTotal - initialTotal)
	rel := 0.0
	if initialTotal != 0 {
		rel = abs / initialTotal
	}
	return VariantResult{
		Steps:              steps,
		FinalTotal:         finalTotal,
		AbsDiff:            abs,
		RelDiff:            rel,
		FirstDeviationStep: firstStep,
		FirstDeviationAbs:  firstAbs,
		Pass:               rel < ConservationThreshold,
	}
}

func cloneField(h []float64) []float64 {
	out := make([]float64, len(h))
	copy(out, h)
	return out
}

// StepN 在场上原地执行 n 步守恒版侵蚀。
func StepN(h []float64, size, n int, c Config) error {
	if n <= 0 {
		return fmt.Errorf("%w: steps=%d 必须为正", ErrBadConfig, n)
	}
	for k := 0; k < n; k++ {
		if err := Step(h, size, c); err != nil {
			return err
		}
	}
	return nil
}

// StepNNaive 在场上原地执行 n 步定点量化版侵蚀，返回每步因独立
// 取整丢失的高度序列。
func StepNNaive(h []float64, size, n int, c Config) ([]float64, error) {
	if n <= 0 {
		return nil, fmt.Errorf("%w: steps=%d 必须为正", ErrBadConfig, n)
	}
	losses := make([]float64, n)
	for k := 0; k < n; k++ {
		lost, err := NaiveStep(h, size, c)
		if err != nil {
			return nil, err
		}
		losses[k] = lost
	}
	return losses, nil
}

// RunFixed 从 base 复制一份场，跑 steps 步守恒版侵蚀，返回末态场与核对结果。
func RunFixed(base []float64, size, steps int, c Config) ([]float64, VariantResult, error) {
	if err := c.validate(); err != nil {
		return nil, VariantResult{}, err
	}
	if steps <= 0 {
		return nil, VariantResult{}, fmt.Errorf("%w: steps=%d 必须为正", ErrBadConfig, steps)
	}
	h := cloneField(base)
	initial := Total(base)
	firstStep, firstAbs := 0, 0.0
	for k := 1; k <= steps; k++ {
		if err := Step(h, size, c); err != nil {
			return nil, VariantResult{}, err
		}
		if firstStep == 0 {
			if d := math.Abs(Total(h) - initial); d != 0 {
				firstStep, firstAbs = k, d
			}
		}
	}
	return h, newResult(steps, initial, Total(h), firstStep, firstAbs), nil
}

// RunNaive 从 base 复制一份场，跑 steps 步定点量化版侵蚀，返回末态场与核对结果。
func RunNaive(base []float64, size, steps int, c Config) ([]float64, VariantResult, error) {
	if err := c.validate(); err != nil {
		return nil, VariantResult{}, err
	}
	if steps <= 0 {
		return nil, VariantResult{}, fmt.Errorf("%w: steps=%d 必须为正", ErrBadConfig, steps)
	}
	h := cloneField(base)
	initial := Total(base)
	firstStep, firstAbs := 0, 0.0
	for k := 1; k <= steps; k++ {
		if _, err := NaiveStep(h, size, c); err != nil {
			return nil, VariantResult{}, err
		}
		if firstStep == 0 {
			if d := math.Abs(Total(h) - initial); d != 0 {
				firstStep, firstAbs = k, d
			}
		}
	}
	return h, newResult(steps, initial, Total(h), firstStep, firstAbs), nil
}
