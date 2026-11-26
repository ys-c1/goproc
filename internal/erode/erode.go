// Package erode 提供热侵蚀模拟与坡度统计。
//
// 侵蚀算子：每轮对每个格子检查四邻域，凡比本格低且高差超过
// 坡止角 talus 的邻居，按 c*(d-talus) 从本格搬运物质到邻居。
// 搬运先累计到增量缓冲再统一应用，单轮结果与遍历顺序无关，
// 全过程确定且物质守恒。talus=0 时算子退化为下坡扩散。
package erode

// Round 对高度图执行一轮热侵蚀，就地修改。h 长度必须为 size*size。
// talus 为坡止高差（单位与高度相同），c 为搬运系数，需满足 4c <= 1 以保稳定。
func Round(h []float64, size int, talus, c float64) {
	delta := make([]float64, len(h))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			hi := h[i]
			if x > 0 {
				move(h, delta, i, i-1, hi, talus, c)
			}
			if x < size-1 {
				move(h, delta, i, i+1, hi, talus, c)
			}
			if y > 0 {
				move(h, delta, i, i-size, hi, talus, c)
			}
			if y < size-1 {
				move(h, delta, i, i+size, hi, talus, c)
			}
		}
	}
	for i := range h {
		h[i] += delta[i]
	}
}

func move(h, delta []float64, i, j int, hi, talus, c float64) {
	d := hi - h[j]
	if d <= talus {
		return
	}
	m := c * (d - talus)
	delta[i] -= m
	delta[j] += m
}
