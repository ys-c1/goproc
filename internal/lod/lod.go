// Package lod 在一张全分辨率高度图上做分块 LOD 层级分配与验证。
//
// 层级编号 L=0 最粗（采样步长 2^Lmax），L=Lmax 最细（采样步长 1，即原图）。
// 每块以双线性插值在对应步长的网格上重建，用最大绝对高度误差衡量重建质量。
package lod

import "math"

// Tile 是一个方块区域的全部输入与计算结果。
type Tile struct {
	Row, Col   int     // 块在分块网格中的行列坐标（从 0 开始）
	X0, Y0     int     // 块在全分辨率图中的左上角像素
	Kappa      float64 // 多尺度曲率 max_L 4^L·D_L（严格误差界所用）
	Kappa0     float64 // 仅最粗层级 L=0 的单尺度曲率 D_0（朴素做法，作对照）
	LevelPred  int     // 由曲率对数公式（多尺度）一次性预测的最低可行层级
	LevelNaive int     // 由单尺度曲率预测的层级（对照，用于暴露高频失效）
	LevelMin   int     // 逐块试渲染得到的最低可行层级（基准）
	ErrPred    float64 // 在预测层级上的实际最大误差
	ErrMin     float64 // 在最低可行层级上的实际最大误差
	Border     bool    // 是否触碰图像边界（此类块直接赋最细层级，不走曲率公式）
}

// Grid 描述一次完整的分块计算。
type Grid struct {
	Size    int     // 高度图边长
	Tile    int     // 每块像素边长
	Lmax    int     // 最细层级（步长 1）
	NTile   int     // 每边块数
	Tiles   []*Tile // 按行优先排列
	Height  []float64
	Tol     float64 // 允许误差
	MaxStep int     // 最粗层级 L=0 的采样步长 2^Lmax
}

func stride(level, lmax int) int { return 1 << (lmax - level) }

// clampNode 把网格节点索引钳制到合法范围 [0, maxNode]。
// 节点 g 对应像素 g*s，最大合法像素为 size-1，故 maxNode=(size-1)/s。
func clampNode(gi, maxNode int) int {
	if gi < 0 {
		return 0
	}
	if gi > maxNode {
		return maxNode
	}
	return gi
}

// nodeAt 取步长 s 网格中节点 (gi,gj) 的高度，节点索引先钳制到合法域。
func nodeAt(h []float64, size, s, gi, gj int) float64 {
	maxNode := (size - 1) / s
	px := clampNode(gi, maxNode) * s
	py := clampNode(gj, maxNode) * s
	return h[py*size+px]
}

// nodeDiff 在步长 s 的网格节点 (gi,gj) 上取 x、y 方向的原始二阶差分。
// 调用方保证该节点为内部节点（两侧邻点都在图内），此处钳制仅作兜底。
func nodeDiff(h []float64, size, s, gi, gj int) (dx, dy float64) {
	dx = nodeAt(h, size, s, gi-1, gj) - 2*nodeAt(h, size, s, gi, gj) + nodeAt(h, size, s, gi+1, gj)
	dy = nodeAt(h, size, s, gi, gj-1) - 2*nodeAt(h, size, s, gi, gj) + nodeAt(h, size, s, gi, gj+1)
	return dx, dy
}

// scaleKappa 计算块在某一层级（节点间距 s）上的离散曲率 D_L：
// 覆盖块插值会用到的内部节点（块内网格点及块外下一个节点，但排除图边界
// 节点——其差分邻点需钳制，不满足误差定理的前提；边界像素本身落在采样
// 节点上，双线性插值误差为零，无需曲率约束）。x、y 方向各自取节点二阶
// 差分绝对值的最大，再相加。
func scaleKappa(h []float64, size, s, x0, y0, tile int) float64 {
	// 覆盖块内全部像素插值所需的节点：从块左上角所在节点，到覆盖块右下角
	// 像素的最小节点索引（ceil 用整数除法表达，不额外 +1，以免重复纳入下一节点）。
	ig0 := x0 / s
	jg0 := y0 / s
	ig1 := (x0 + tile - 1) / s
	jg1 := (y0 + tile - 1) / s
	// 内部节点必须保证两侧邻点像素都在图内：(g+1)*s <= size-1，
	// 故节点上界为 (size-1)/s - 1，下界为 1。边界节点（邻点需钳制）
	// 不满足误差定理前提，一律排除；其所在像素落在采样节点上，插值误差为零。
	lastInterior := (size-1)/s - 1
	if ig0 < 1 {
		ig0 = 1
	}
	if jg0 < 1 {
		jg0 = 1
	}
	if ig1 > lastInterior {
		ig1 = lastInterior
	}
	if jg1 > lastInterior {
		jg1 = lastInterior
	}
	if ig1 < ig0 || jg1 < jg0 {
		return 0 // 该层级块内无内部节点，曲率约束为空
	}
	kxx, kyy := 0.0, 0.0
	for j := jg0; j <= jg1; j++ {
		for i := ig0; i <= ig1; i++ {
			dx, dy := nodeDiff(h, size, s, i, j)
			if a := math.Abs(dx); a > kxx {
				kxx = a
			}
			if a := math.Abs(dy); a > kyy {
				kyy = a
			}
		}
	}
	return kxx + kyy
}

// multiscaleKappa 计算严格的多尺度曲率
//
//	K = max over L of 4^L · D_L,  D_L = scaleKappa at level L
//
// 单尺度曲率 D_0 只在多项式曲面上满足 D_L = D_0/4^L；FBM 的高频分量使
// 细网格二阶差分大于该缩放。取所有层级归一化后的最大值，对任意离散数据
// 都保证 E(L) <= K/(8·4^L)。对二次曲面所有 4^L·D_L 相等，K 退化为 D_0。
func multiscaleKappa(h []float64, size, lmax, x0, y0, tile int) (k, k0 float64) {
	k = 0.0
	for l := 0; l <= lmax; l++ {
		s := stride(l, lmax)
		dl := scaleKappa(h, size, s, x0, y0, tile)
		if l == 0 {
			k0 = dl
		}
		if v := dl * math.Pow(4, float64(l)); v > k {
			k = v
		}
	}
	return k, k0
}

// bilinear 在步长 s 的粗网格上对像素 (px,py) 做双线性采样。
// 网格节点索引越界时由 nodeAt 钳制到合法节点域 [0, (size-1)/s]。
func bilinear(h []float64, size, s, px, py int) float64 {
	gx := float64(px) / float64(s)
	gy := float64(py) / float64(s)
	i0 := int(math.Floor(gx))
	j0 := int(math.Floor(gy))
	fx := gx - float64(i0)
	fy := gy - float64(j0)
	c00 := nodeAt(h, size, s, i0, j0)
	c10 := nodeAt(h, size, s, i0+1, j0)
	c01 := nodeAt(h, size, s, i0, j0+1)
	c11 := nodeAt(h, size, s, i0+1, j0+1)
	top := c00 + (c10-c00)*fx
	bot := c01 + (c11-c01)*fx
	return top + (bot-top)*fy
}

// tileError 计算某块在给定层级上的实际最大绝对重建误差。
func tileError(g *Grid, t *Tile, level int) float64 {
	s := stride(level, g.Lmax)
	worst := 0.0
	for y := t.Y0; y < t.Y0+g.Tile; y++ {
		for x := t.X0; x < t.X0+g.Tile; x++ {
			approx := bilinear(g.Height, g.Size, s, x, y)
			d := math.Abs(approx - g.Height[y*g.Size+x])
			if d > worst {
				worst = d
			}
		}
	}
	return worst
}

// predictLevel 由曲率与允许误差按对数公式预测层级，再向上取整并钳制到 [0,Lmax]。
//
//	误差界 E(L) <= kappa / (8 * 4^L)，令其 <= tol：
//	L >= log4(kappa/(8*tol))
//
// kappa 为多尺度曲率时该界对任意离散数据成立。
func predictLevel(kappa, tol float64, lmax int) int {
	if kappa <= 0 {
		return 0
	}
	raw := math.Log(kappa/(8.0*tol)) / math.Log(4.0)
	l := int(math.Ceil(raw)) // 向上取整：精确运算下保证预测层级不偏低；浮点误差只会偏向更细一级
	if l < 0 {
		l = 0
	}
	if l > lmax {
		l = lmax
	}
	return l
}

// Build 对整图分块，一次性算出每块曲率、预测层级，并用逐块试渲染求基准最低层级。
//
// 预测只读取高度场的二阶差分（曲率）与全局容差，不做任何逐块渲染；
// 逐块试渲染仅作为独立基准用于事后比对。
func Build(height []float64, size, tileSize, lmax int, tol float64) *Grid {
	nTile := size / tileSize
	g := &Grid{
		Size: size, Tile: tileSize, Lmax: lmax, NTile: nTile,
		Height: height, Tol: tol, MaxStep: stride(0, lmax),
	}
	g.Tiles = make([]*Tile, 0, nTile*nTile)
	for r := 0; r < nTile; r++ {
		for c := 0; c < nTile; c++ {
			t := &Tile{Row: r, Col: c, X0: c * tileSize, Y0: r * tileSize}
			// 触碰图像边界的块含有被钳制的采样节点，曲率误差定理不适用，
			// 直接保守赋最细层级 Lmax（其重建误差天然最小，恒为可行）。
			touchesBorder := t.X0 == 0 || t.Y0 == 0 ||
				t.X0+tileSize >= size || t.Y0+tileSize >= size
			t.Border = touchesBorder
			// 一次性曲率计算：多尺度（严格）与单尺度 L=0（对照）。
			t.Kappa, t.Kappa0 = multiscaleKappa(height, size, lmax, t.X0, t.Y0, tileSize)
			if touchesBorder {
				t.LevelPred = lmax
			} else {
				t.LevelPred = predictLevel(t.Kappa, tol, lmax)
			}
			t.LevelNaive = predictLevel(t.Kappa0, tol, lmax)

			// 逐块试渲染：从最细到最粗逐层求误差，找最低可行层级（基准）。
			errs := make([]float64, lmax+1)
			lmin := lmax
			for l := lmax; l >= 0; l-- {
				errs[l] = tileError(g, t, l)
				if errs[l] <= tol {
					lmin = l
				}
			}
			t.LevelMin = lmin
			t.ErrMin = errs[lmin]
			t.ErrPred = errs[t.LevelPred]
			g.Tiles = append(g.Tiles, t)
		}
	}
	return g
}

// Underpredicted 返回多尺度预测层级下，实际误差超过允许误差的块。
// 由误差界保证该集合为空；保留方法以便程序化核验并在退化时报告坐标。
func (g *Grid) Underpredicted() []*Tile {
	var out []*Tile
	for _, t := range g.Tiles {
		if t.ErrPred > g.Tol {
			out = append(out, t)
		}
	}
	return out
}

// NaiveUnderpredicted 返回单尺度（仅 L=0 曲率）预测层级下实际误差超标的块，
// 即朴素公式因高频分量而偏低的块。
func (g *Grid) NaiveUnderpredicted() []*Tile {
	var out []*Tile
	for _, t := range g.Tiles {
		if t.LevelNaive < t.LevelMin {
			out = append(out, t)
		}
	}
	return out
}

// LevelGrid 按行返回层级矩阵（预测值），每行一个 int 切片。
func (g *Grid) LevelGrid(usePred bool) [][]int {
	m := make([][]int, g.NTile)
	for r := 0; r < g.NTile; r++ {
		m[r] = make([]int, g.NTile)
		for c := 0; c < g.NTile; c++ {
			t := g.Tiles[r*g.NTile+c]
			if usePred {
				m[r][c] = t.LevelPred
			} else {
				m[r][c] = t.LevelMin
			}
		}
	}
	return m
}
