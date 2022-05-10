# ML 2020: Regression

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Notes:** [LeeML-Notes (datawhalechina.github.io)](https://datawhalechina.github.io/leeml-notes/#/)
3. **Extra reading:** [Lecture 16: Projection matrices and least squares (mit.edu)](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/198bde3a6536b62989b6cc09495575c1_MIT18_06SCF11_Ses2.3sum.pdf)
4. **Extra videos:** None

---

简单回顾回归与最小二乘法

## Least Squares Fitting A Line

我们经常需要拟合直线：

![image-20220307160338773](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202203071603956.png)

三个点坐标 (1,1) (2,2) (3,2) 显然不在一条直线上。

可以用 $y=b+wx$ 拟合，因此需要解方程：

$$
\begin{matrix}b
 &+  &w  &=  &1 \\b
 &+  &2w  &=  &2 \\b
 &+  &3w  &=  &2
\end{matrix}
$$

表示成 $\boldsymbol{Ax}=\boldsymbol{b}$ 显然无解：

$$
\begin{bmatrix}
1 &1 \\1
 &2 \\1
 &3
\end{bmatrix}\begin{bmatrix}
b\\w
\end{bmatrix}=\begin{bmatrix}
1\\2
\\2
\end{bmatrix}
$$

于是需要把 $\bm{b}$ 投影到 $A$ 的列空间得到投影向量 $\bm{p}$，解 $\boldsymbol{A}\hat{\boldsymbol{x}}=\boldsymbol{p}$

即 $\boldsymbol{A}^\mathrm{T}\boldsymbol{A}\hat {\boldsymbol{x}}=\boldsymbol{A}^\mathrm{T}\boldsymbol{b}=\boldsymbol{A}^\mathrm{T}\boldsymbol{p}$

得 $\hat{\boldsymbol{x}}=(\boldsymbol{A^{\mathrm{T}}\boldsymbol{A}})^{-1}\boldsymbol{A^{\mathrm{T}}}\boldsymbol{b}=\begin{bmatrix}
\displaystyle \frac{2}{3} & \displaystyle \frac{1}{2} \end{bmatrix}^{\mathrm{T}}$

因此拟合直线：$y=\displaystyle \frac{2}{3}+\displaystyle \frac{1}{2}x$。

## Regression

我们现在做的是拟合

$$
y=f(\bm{x})=b+\bm{w}\bm{x}=b+\displaystyle\sum_{i=1}^{m} w_ix_i
$$

变成了高维的平面. $\bm{x}$ 的维度代表了特征数量, 每个特征用下标表示.

已知每个数据, 用上标表示: $(\bm{x}^{(j)}, \hat{y}^{(j)}), j=1\cdots n$.

大可以求出矩阵 $A=\begin{bmatrix}
    1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_m^{(1)}\\
    1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_m^{(2)}\\
    \vdots & \vdots & \vdots & \vdots & \vdots\\
    1 & x_1^{(n)} & x_2^{(n)} & \cdots & x_m^{(n)}\\
\end{bmatrix} $, 向量 $\bm{b}=\begin{bmatrix}
    y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)}
\end{bmatrix} $

解得 $\hat{\boldsymbol{x}}=(\boldsymbol{A^{\mathrm{T}}\boldsymbol{A}})^{-1}\boldsymbol{A^{\mathrm{T}}}\boldsymbol{b}=\begin{bmatrix}
b \\ w_1 \\ \vdots \\ w_n \end{bmatrix}^{\mathrm{T}}$.

但是机器学习从来不会这么求, 如果矩阵 $A$ 特别大, 求逆什么的岂不是很麻烦? 怎么做呢? 就要把大象放进冰箱里了. 第一步选 function set 已经完事, 接下来需要定义评估函数好坏的指标.

<span class="heimu" title="你知道的太多了">上面的一切看不懂的最小二乘法公式就都可以通通不看了, 只需要知道 regression 就是 y=b+wx 就行哈哈哈.</span>

## Loss

通常来说都是定义函数有多坏. 输入函数 $f$, 输出 loss $L(f)$.

$$
L(f) = L(\bm{w}, b) = \displaystyle\sum_{i=1}^{n} (\hat{y}^{(i)}-(b+\bm{w}\bm{x}^{(n)}))^{2}
$$

要选择让 loss 最小的 $f$, 也就是选择

$$
\bm{w}^*, b^* = \mathop{\arg\min}\limits_{\bm{w}, b}\displaystyle\sum_{i=1}^{n} (\hat{y}^{(i)}-(b+\bm{w}\bm{x}^{(n)}))^{2}
$$

## Gradient Descent

接下来第三步, 选取最好的函数, 也就是找到让 loss 最小的函数, 也就是求 $\bm{w}^*$ 和 $b^*$. 需要用到梯度下降法.

### Approach

- 随机初始化 $\bm{w}=\bm{w}_0, b=b_0$.
- 计算偏导 $\displaystyle \frac{\partial L}{\partial \bm{w}}, \displaystyle\frac{\partial L}{\partial b}$.
- 若为负, 需要增大变量下降. 否则上升.
  - $\Delta \bm{w} = -\alpha \displaystyle\frac{\partial L}{\partial \bm{w}}, \Delta b = -\alpha \displaystyle\frac{\partial L}{\partial b}$
- 更新参数.
  - $\bm{w} := \bm{w} + \Delta \bm{w}, b := b + \Delta b$
- 直到 $\Delta \bm{w}, \Delta b$ 充分小.

用梯度表示 $\nabla L = \begin{bmatrix}
    \displaystyle \frac{\partial L}{\partial \bm{w}} &
    \displaystyle \frac{\partial L}{\partial b}
\end{bmatrix}^{\mathrm{T}} $, 所以 $\begin{bmatrix}
    \bm{w} & b
\end{bmatrix}^{\mathrm{T}} :=  \begin{bmatrix}
    \bm{w} & b
\end{bmatrix}^{\mathrm{T}} - \alpha \nabla L$

总结, 对 $f(\bm{x})$:

$$
\bm{x} := \bm{x} - \alpha \nabla L(f)
$$

### Calculation

$$
\frac{\partial L}{\partial \bm{w}} = 2\displaystyle\sum_{i=0}^{n} (\hat{y}^{(i)}-(b+\bm{w}\bm{x}^{(i)}))(-\bm{x}^{(i)})
$$

$$
\frac{\partial L}{\partial b} = 2\displaystyle\sum_{i=0}^{n} (\hat{y}^{(i)}-(b+\bm{w}\bm{x}^{(i)}))(-1)
$$

### Local Optima?

确实, 梯度下降只能求 local optima. 但是线性回归是 convex 的. 而且 gradient descent 的速度明显比最小二乘法那个公式快多了.

## Evaluation

需要把数据集分成 train 和 test. train 代表模拟题. test 代表考试题. 模拟题答得再好, 考试也可能很烂.

复杂的数学公式可以说明, 用上面的平方误差作为 loss, 进行梯度下降, 可以让 error 最小. error 最小, 正是最小二乘法的目标. 因此这种梯度下降求得结果等价于最小二乘法.

我们就用最小二乘法里的 error 来衡量模型: $E = \displaystyle\sum_{i=1}^{n} e^{(i)}$

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205092219714.png" alt="image-20220509221911950" style="zoom:50%;" />

但我们更关心 generalization, 也就是 test data 上的效果

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205092221787.png" alt="image-20220509222134509" style="zoom:50%;" />

一般来说 test 的表现都会比 train 的要差.

## How Can We Do Better?

- Model Selection: 模型越复杂, 训练集拟合越好, 但是测试集一开始还好, 后来就烂了.
  - 过拟合
    - <img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205092227272.png" alt="image-20220509222713831" style="zoom: 50%;" />
- Hidden Factors:
  - if 分支型
  - 加入其他 features
  - 加入相关 factor 会很好, 但是加入不相关的就又 overfitting
    - <img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205092235650.png" alt="image-20220509223502745" style="zoom:50%;" />
- Regularization
  - $L = \displaystyle\sum_{i=1}^{n} \hat{y}^{(i)}-(b+\bm{w}\bm{x})^{2}+\lambda\left\vert |\bm{w} |\right\vert ^{2} $
  - 这样会让 $w_i$ 越小越好, 这样在 $\bm{x}$ 变化下才不会敏感, 曲线变得平滑, 防止 noise 影响训练.
  - $\lambda$ 越大, train loss 增大, 因为影响模型去拟合 train data. test loss 最开始会变小, 因为去除了噪声. 但是后面又会烂掉, 因为太平滑模型太简单了.
  - 为什么不考虑 bias? 因为 bias 仅仅是偏移量, 对曲线上下平移, 不影响曲线复杂度的.
  - $\lambda$ 的取值?
    - <img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205092244505.png" alt="image-20220509224300659" style="zoom:50%;" />

那么什么时候该用什么招数呢? 我们需要分析 Error 从哪来的.

## Error Analysis

实际上真正的函数 $\hat{f}$ 我们不知道. 我们只能通过收集的数据 $\bm{x}^{(1)}, \cdots , \bm{x}^{(n)}$ 来从模型/函数集 $M$ 中选择最佳的函数集 $f^*$ 作为 $\hat{f}$ 的估计.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/110a9e7a038dad370465188b566dc22509476fb1455c39bbc1cbaa32c7104107.png" />  

这个过程就像打靶, $\hat f$ 就是我们的靶心, $f^*$ 就是我们投掷的结果. 如上图所示, $\hat f$ 与  $f^*$ 之间蓝色部分的差距就是偏差和方差导致的. 什么是偏差和方差?

### Bias & Variance

假设 $x$ 的平均值是  $\mu$, 方差为 $\sigma^2$. 有 $N$ 个样本点：$\{x^1,x^2,···,x^N\}$.

样本点平均值 $m$ 有:

$$
m=\frac{1}{N}\sum_{i=1}^N x^i \neq \mu
$$

$$\operatorname E[m]=\operatorname E[\frac{1}{N}\sum x^i]=\frac{1}{N}\sum_i\operatorname E[x^i]=\mu$$

这个估计是 unbiased.

$$
\operatorname{Var}[m] = \displaystyle\frac{\sigma^{2}}{N}
$$

$N$ 越小, $m$ 分布对于 $\mu$ 的离散程度越大.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8ed343709cd79c11bad037582bec26a2275f5f1d26809264a82ad1aba5f07e83.png" />  

对于样本点方差 $s^{2}$:

$$
s^{2} = \displaystyle\frac{1}{N}\displaystyle\sum_{i=1}^{N} (x^i-m)^{2}
$$

$$
\operatorname{E}[s^{2}] = \displaystyle\frac{N-1}{N} \sigma ^{2} \neq \sigma^{2}
$$

这个估计是 biased.

现在我们来看函数的 mean 和 variance. $x^i=f(\bm{x}^{(i)})$. 在 $N$ 充分大的情况下, 按照下图说明 bias 和 variance 之间的关系.

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/cb8854707981d4a14add8c89487863cb33c0ac44f067cf98acfeb7b43cb00326.png" style="zoom:50%;" />  

为什么会有很多不同的模型? 因为在不同的训练集中找到的 $f^∗$ 就是不一样的.

简单的模型受到不同训练集的影响是比较小的, 所以方差小. 但是偏差比较大, 因为简单的模型函数集的 space 比较小, 所以可能 space 里面就没有包含靶心, 肯定射不中.

复杂的模型函数集的 space 比较大, 可能就包含的靶心, 只是受训练集影响较大, 方差大, 没有办法找到确切的靶心在哪. 但只要样本足够多, 就可能得到真正的 $\hat{f}$.

### Underfitting & Overfitting

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/2122c1cee182fff5611adcd85d0f809746988f386cf9ba08fd689dff1c7893ad.png" style="zoom:50%;" />  

- 对于欠拟合, 真正的 model 可能不在设定的 set 里面, 因此需要让 model 更复杂, 或者设计更好的 features
- 对于过拟合, 增加 data 是最好的. 要么正则化, 但是会伤害 bias, 需要适度 (上面有写)

## Cross Validation

现在在偏差和方差之间就需要一个权衡想选择的模型, 可以平衡偏差和方差产生的错误, 使得总错误最小. 但是下面这件事最好不要做:

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/7a4907405df815c21c873f6ba728e53555a765f24a9dc4e1541bc6bcea14a158.png" style="zoom:50%;" />  

虽然不少人会这么干, 但是其实是作弊. 用训练集训练不同的模型, 然后在测试集上选择错误最小的模型. 这样仍然是在向测试集拟合, 但是测试集并不是真正世界中的集合. 比如在已有的测试集上错误是 0.5, 但有条件收集到更多的测试集后通常得到的错误都是大于 0.5 的.

通常会对 training set 作划分, 得到新的 training 和 validation. 用 validation 的效果来调整 training 的模型 (但 validation 不负责更新模型参数, 而是负责看要不要再换一个模型/加特征). 然后 testing 作为最终效果的展示.

不过大家都<span class="heimu" title="你知道的太多了">知法犯法, </span>喜欢根据 testing 再调, 那就对 testing 分成 public 和 private, 本质上 public testing set 仍然是 validation set 的一部分了... 哦有时 validation set 又叫 development set...

<img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/53bb1c1a75776ea6fb58d1ce1597fd794f86162883d6584bbfc5c9ecb1a0bebf.png" style="zoom:50%;" />  

可能会担心将训练集拆分的时候分的效果比较差怎么办, 可以用 N 折交叉验证, 比如 N=3:

<img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/bec2f72c3395999055ef3d15d0bf643e152d48a1093d03d2462fe649a02c24d2.png" style="zoom:50%;" />  
