# 两层混合的权重泛函与最优解

## 1. 问题设定

低频层 \(L(x)\) 与高频层 \(H(x)\) 在过渡带 \([a,b]\) 内混合，权重 \(w(x)\) 满足端点固定约束

\[
w(a)=0,\qquad w(b)=1,
\]

混合结果

\[
B(x)=(1-w)L+wH = L + w\Delta,\qquad \Delta(x)\equiv H(x)-L.
\]

入口处先把两层对齐：\(\Delta(a)=0\)（实验中每行减去常数偏置实现，不改变任何层的梯度）。

## 2. 混合结果与理想带限结果之差

理想带限结果 \(B^*\) 的定义是"梯度按权重混合、值从 \(L(a)\) 出发"：

\[
(B^*)'=(1-w)L'+wH',\qquad B^*(a)=L(a).
\]

由 \(B=L+w\Delta\) 求导，

\[
B'=L'+w'\Delta+w\Delta'=(1-w)L'+wH'+w'\Delta,
\]

故混合梯度比理想结果多出一项 \(r=w'\Delta\)，这就是肉眼看到的"棱"。

差值本身用积分写出。由 \(\Delta(a)=0\)，

\[
w(x)\Delta(x)=\int_a^x (w\Delta)'ds=\int_a^x w'\Delta\,ds+\int_a^x w\Delta' ds,
\]

而

\[
B^*(x)-L(a)=\int_a^x (B^*)'ds=\int_a^x L' ds+\int_a^x w\Delta' ds,
\]

两式相减（注意 \(B(a)=B^*(a)=L(a)\)）：

\[
\boxed{\,B(x)-B^*(x)=\int_a^x w'(s)\,\Delta(s)\,ds
      =\int_a^x w(s)\,q(s)\,ds\ \text{的互补形式}\,}
\]

其中 \(q=\Delta'=H'-L'\) 是两层梯度差。第二个等号来自分部积分
\(w\Delta=\int w'\Delta+\int w\Delta'\)。**结论：混合误差完全由权重与两层梯度差（等价地，权重导数与层差）的积分决定。**

## 3. 可最小化的泛函

以棱的能量（误差梯度的 \(L^2\) 范数平方）为目标，在固定端点 \(w(a)=0,\,w(b)=1\) 下：

\[
E[w]=\int_a^b \bigl(w'(x)\Delta(x)\bigr)^2\,dx.
\]

这是关于 \(w\) 的凸二次泛函，存在唯一极小。

### Euler–Lagrange 方程

被积函数 \(F=(w'\Delta)^2\) 不显含 \(w\)，故 \(\dfrac{\partial F}{\partial w}=0\)，EL 方程为

\[
\frac{d}{dx}\frac{\partial F}{\partial w'}=0
\quad\Longrightarrow\quad
\boxed{\,\Delta(x)^2\,w'(x)=C\ \text{（常数）}\,}.
\]

### 解析解

由约束 \(\int_a^b w' dx=1\) 定出常数：

\[
C=\left(\int_a^b \frac{dx}{\Delta(x)^2}\right)^{-1},
\qquad
\boxed{\,w^*(x)=\frac{\displaystyle\int_a^x \Delta(s)^{-2}ds}
{\displaystyle\int_a^b \Delta(s)^{-2}ds}\,}.
\]

它不是随便挑的平滑曲线：线性权重 \(w=t\) 对应 \(w'=\text{const}\)，smoothstep 对应 \(w'=6t(1-t)\)，
二者都不满足 \(\Delta^2w'=\text{const}\)；只有上式让"层差大的地方权重走得慢、层差小的地方走得快"，
从而把棱的能量压到最低。

### 理论最小值（Cauchy–Schwarz 下界取等）

\[
1=\int_a^b w' dx=\int_a^b (w'\Delta)\cdot\frac{1}{\Delta}dx
\;\le\;
\underbrace{\int_a^b (w'\Delta)^2 dx}_{E[w]}\;\int_a^b \Delta^{-2}dx,
\]

因此

\[
\boxed{\,E[w]\ge E^*=\left(\int_a^b \Delta^{-2}dx\right)^{-1}\,},
\]

等号恰在 \(w=w^*\) 时成立。

### 奇性与正则化

当 \(\Delta\) 在带内过零时 \(\Delta^{-2}\) 发散，无正则解在过零边退化为阶跃（解析上仍是合法极小，
但数值上不可用）。引入 \(\varepsilon^2\) 正则：

\[
E_\varepsilon[w]=\int_a^b (\Delta^2+\varepsilon^2)(w')^2 dx,
\]

EL 方程变为 \((\Delta^2+\varepsilon^2)w'=C_\varepsilon\)，解与下界分别为

\[
w_\varepsilon^*(x)=
\frac{\int_a^x (\Delta^2+\varepsilon^2)^{-1}ds}
     {\int_a^b (\Delta^2+\varepsilon^2)^{-1}ds},
\qquad
E_\varepsilon^*=
\left(\int_a^b (\Delta^2+\varepsilon^2)^{-1}dx\right)^{-1}.
\]

实验取 \(\varepsilon=0.15\cdot\mathrm{rms}(\Delta)\)。

## 4. 离散形式（代码实际求解的问题）

网格 \(x_k=a+k\,dx\)，边中点层差 \(\Delta_k=\tfrac12(\Delta(x_k)+\Delta(x_{k+1}))\)，\(k=0..m-1\)：

\[
E_\varepsilon=dx\sum_{k=0}^{m-1}(\Delta_k^2+\varepsilon^2)(w'_k)^2,
\qquad w'_k=\frac{w_{k+1}-w_k}{dx},\quad w_0=0,\ w_m=1.
\]

离散 EL：\((\Delta_k^2+\varepsilon^2)(w_{k+1}-w_k)=\lambda\)（与 \(k\) 无关），故

\[
w_k=\frac{\sum_{j<k}(\Delta_j^2+\varepsilon^2)^{-1}}
         {\sum_{j=0}^{m-1}(\Delta_j^2+\varepsilon^2)^{-1}},
\qquad
E_\varepsilon^*=\frac{1}{dx\sum_k(\Delta_k^2+\varepsilon^2)^{-1}}.
\]

对应实现：`internal/blend` 的 `OptimalWeights` 与 `RegularizedBound`，
单测 `TestELStationarity`、`TestRegularizedEquality`、`TestCauchyEquality` 验证驻点条件与下界取等。

## 5. 度量定义

- **接缝跳变** \(J_a,J_b\)：带两端点处内侧与外侧一阶差分（坡度）之差，单位 高度/格；
- **泛函能量** \(E_\varepsilon\)：含正则项，是求解器真正最小化的量；
- **可见棱能量**：只计 \(\sum(w'\Delta)^2\)，对应肉眼可见部分；
- **gap**：\((E_\varepsilon-E_\varepsilon^*)/E_\varepsilon^*\)，最优权重逐行应为机器零；
- **权重 RMS 差**：实际权重与同数据下 \(w_\varepsilon^*\) 的逐点 RMS 距离，最优权重为 0。

## 6. 复现

```bash
go test ./...                                   # 数学恒等式与下界取等测试
go run ./cmd/blendexp --size 256 --seed 7 \
    --widths 8,16,32 --out artifacts/exp        # 生成逐行指标与裁剪数据
python3 cmd/blendexp/render.py \
    --exp artifacts/exp --out artifacts/exp/figs  # 渲染接缝放大图/权重图/对照表
```
