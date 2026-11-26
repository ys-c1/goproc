// Package stretch 对噪声采样坐标施加带方向的剪切拉伸。
//
// 约定（列向量，角度以正东为零度、逆时针为正）：
//
//	θ 为拉伸方向角，r 为轴比（剪切参数）。在拉伸方向对齐 x 轴的局部
//	坐标系中，拉伸矩阵为 M_r = [[1,r],[0,1]]：拉伸方向上的点保持不变，
//	垂直方向的点被沿拉伸方向位移 r 倍。全局矩阵 M = R(θ) M_r R(-θ)，
//	采样时 H(p) = N(M p)，N 为各向同性噪声。
package stretch

import "math"

// Warp 是一次带方向的拉伸。
type Warp struct {
	Theta float64 // 拉伸方向角（弧度），正东为零，逆时针为正
	Ratio float64 // 轴比 r，即剪切参数；零表示不拉伸
}

// Apply 把物理坐标 (x, y) 映射为拉伸后的噪声采样坐标。
// r=0 时为恒等映射。
func (w Warp) Apply(x, y float64) (float64, float64) {
	if w.Ratio == 0 {
		return x, y
	}
	c, s := math.Cos(w.Theta), math.Sin(w.Theta)
	// 局部坐标：u 沿拉伸方向，v 垂直拉伸方向
	u := c*x + s*y
	v := -s*x + c*y
	// M_r：u' = u + r v, v' = v
	u += w.Ratio * v
	// 转回全局坐标
	return c*u - s*v, s*u + c*v
}
