package divide

// Point 是高度图上的一个格像素坐标。
type Point struct{ X, Y int }

// Crest 用与逐格流向完全独立的局部判据检测脊线像素，作为人工/对照脊线的算法基准。
//
// 判据（四条过格轴线之一上的严格局部极大，零洪泛、零流向）：格 (x,y) 满足下列任一即为脊像素——
//   - 竖直脊：高于左右邻居 h(x±1,y)，脊沿 y 延伸；
//   - 水平脊：高于上下邻居 h(x,y±1)，脊沿 x 延伸；
//   - 主对角脊：高于 h(x-1,y-1)、h(x+1,y+1)；
//   - 副对角脊：高于 h(x-1,y+1)、h(x+1,y-1)。
//
// 四条线覆盖正交与两个对角走向，能描出沿斜向山脊的连续脊像素，与分水线格边可比。
// 边界格缺少对应方向的两个邻居，不参与判定，避免边界伪脊；
// 等高水平台地不满足“严格高于”，不会产生成片伪脊，与平地处理保持一致。
func Crest(n int, h []float64) []Point {
	var pts []Point
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			c := h[y*n+x]
			vertical := c > h[y*n+x-1] && c > h[y*n+x+1]
			horizontal := c > h[(y-1)*n+x] && c > h[(y+1)*n+x]
			diagMain := c > h[(y-1)*n+x-1] && c > h[(y+1)*n+x+1]
			diagAnti := c > h[(y+1)*n+x-1] && c > h[(y-1)*n+x+1]
			if vertical || horizontal || diagMain || diagAnti {
				pts = append(pts, Point{x, y})
			}
		}
	}
	return pts
}
