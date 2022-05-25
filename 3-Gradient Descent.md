# ML 2020: Gradient Descent

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Notes:** [LeeML-Notes (datawhalechina.github.io)](https://datawhalechina.github.io/leeml-notes/#/)
3. **Extra reading:**
   1. [An overview of gradient descent optimization algorithms (ruder.io)](https://ruder.io/optimizing-gradient-descent/index.html#fn9)
   1. [梯度下降实用技巧I之特征缩放 Gradient Descent in practice I - feature scaling_天泽28的博客-CSDN博客](https://blog.csdn.net/u012328159/article/details/51030366)
   1. [stand.dvi (umd.edu)](http://www.math.umd.edu/~mboyle/courses/131f15/stand.pdf)
   1. [标准化和归一化，请勿混为一谈，透彻理解数据变换_夏洛克江户川的博客-CSDN博客_标准化和归一化](https://blog.csdn.net/weixin_36604953/article/details/102652160)
   1. [鞍点 - 维基百科，自由的百科全书 (wikipedia.org)](https://zh.m.wikipedia.org/zh-hans/鞍點)
   1. [一篇综述带你全面了解课程学习(Curriculum Learning) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/362351969)
4. **Extra videos:** None

---

最近的全角和半角转换比较混乱，因为我个人一直是一律半角，然而写毕业论文不得不改成全角，最近只能用全角写 blogs 了。这一节将会具体讲一些 Gradient Descent 方法，不过我有点没太听懂就是了，尤其是后面的补充内容，sad。

## Review: Gradient Descent

在回归问题的第三步中，需要解决下面的最优化问题：

$$
\boldsymbol{\theta}^∗= \underset{ \boldsymbol{\theta} }{\operatorname{arg\ min}}  L(f(\boldsymbol{\theta}))
$$

需要用到梯度下降法：

$$
\begin{aligned}
    \boldsymbol{\theta} &:= \boldsymbol{\theta} - \alpha \nabla L(f(\boldsymbol{\theta})) \\
    \nabla L(f(\boldsymbol{\theta})) &= \begin{bmatrix} \frac{\partial L(f)}{\partial \theta_1} & \frac{\partial L(f)}{\partial \theta_2} & \cdots & \frac{\partial L(f)}{\partial \theta_p} \\\end{bmatrix}^{\mathrm{T}}
\end{aligned}
$$

梯度代表了 loss 等高线的法线方向。那么 GD 就这样就可以了吗？当然不是！

## Tip 1: Tuning Your Learning Rates

### Be Careful of Your LR

举例：

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525103823.png" alt="20220525103823" style="zoom:50%;" />

Learning Rate 调不好，模型可能都不收敛！如果过大，反复横跳，看黄绿色的线，根本到不了最低点，反而有可能飞出去。如果过小，更新又太慢，看蓝色点，而且可能卡在平台和局部最小值。只有红色的线是坠吼的。

虽然这样的可视化可以很直观观察，但可视化也只是能在参数是一维或者二维的时候进行，更高维的情况已经无法可视化了。

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525104201.png" alt="20220525104201" style="zoom:50%;" />

解决方法就是将参数改变对损失函数的影响进行可视化。比如学习率太小（蓝色的线），损失函数下降的非常慢；学习率太大（绿色的线），损失函数下降很快，但马上就卡住不下降了；学习率特别大（黄色的线），损失函数就飞出去了；红色的就是差不多刚好，可以得到一个好的结果。

还有没有什么改进空间？

### Adaptive Learning Rates

epoch 增加，lr 应该减少。

- 通常刚开始，初始点会距离最低点比较远，所以使用大一点的学习率
- update 好几次参数之后呢，比较靠近最低点了，此时减少学习率

提出一个 1/t decay：$\eta^t =\frac{\eta}{\sqrt{t+1}}$，$t$ 是次数。随着次数的增加，$\eta^t$ 减小

不同的参数需要不同的学习率。

综合以上两点，有人提出了 Adagrad。

### Adagrad

Adagrad 在 1/t decay 上加入了对不同参数所做的不同学习率修正，对第 $i$ 个参数有：

$$
\boldsymbol{w}^{t+1}_i :=  \boldsymbol{w}^t_i -\frac{η^t}{\boldsymbol{\sigma}^t_i}\boldsymbol{g}^t_i
$$

其中 $\boldsymbol{g}^t_i$ 为第 $i$ 个参数在 $t$ 时刻的梯度，主要是 $\boldsymbol{\sigma}^t = \sqrt{G^t_i + \epsilon} = \sqrt{\frac{1}{t+1}\sum_{n=0}^{t}(\boldsymbol{g}^n_i)^2}$ 表示之前参数的所有微分的均方根，对于每个参数都是不一样的。这里 $G^t$ 是一个对角矩阵，$G^t_i$ 代表 $i$ 行 $i$ 列的元素。$\epsilon$ 用来防止分母为零。于是代入 1/t decay 并向量化有：

$$
\boldsymbol{w}^{t+1} :=  \boldsymbol{w}^t -\frac{η}{\sqrt{G^t + \epsilon}}\boldsymbol{g}^t
$$

#### Contradiction?

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/!%5B%5D(reschapter6-6.png).png" alt="reschapter6-6.png" style="zoom:50%;" />

在 Adagrad 中，当梯度越大的时候，步伐应该越大，但下面分母又导致当梯度越大的时候，步伐会越小。

直观的解释是：衡量当前梯度较前面梯度的反差有多大。如图所示：

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525110442.png" alt="20220525110442" style="zoom:50%;" />

第一行是特别大的情况，在没出现 g4 之前，前面的 g 和平方和做比之后并不是很大，但是加入 g4 之后，由于 g4 远远大于前面的 g，因此认为前面的 g 的平方和忽略不计了，因此这一步的步伐趋近于 1，一下子变大了。

第二行是特别小的情况，在没出现 g4 之前，由于大家数量级差不多，所以做比之后也没小太多，然后分子 g4 特别小，你可以认为他是 0 了，而分母的平方和可以忽略 g4，所以很大，因此这一步步伐变得很小。

然后 slide 后面说 Adagrad 模拟了一阶微分和二阶微分的比，而这个值代表了二次函数从任一点到极值的距离 $x + \displaystyle\frac{b}{2a}$。然而这样的话便需要认为前面时刻梯度的平方和开根号可以近似二阶微分，我没有想明白是怎么可以近似的。。。上网也没有查到除了李老师外还有其他人也这么解释的，看原论文公式好多头有点大，就先跳过。

而且想一想直观解释，好像有点问题，步数没那么大还好，大了之后分母的平方和越累积越大，到时候 lr 就会巨小，无论 g 多大可能都顶不住。其实这正是 Adagrad 的问题。<span class="heimu" title="你知道的太多了">所以没看到有人现在用 Adagrad 的 lol</span>

结论1-1：梯度越大，就跟最低点的距离越远。

这个结论在多个参数的时候就不一定成立了。

## Tip 2: Stochastic Gradient Descent

之前的梯度下降：

$$L=\sum_n(\hat y^n-(b+\sum w_ix_i^n))^2 $$
$$\theta^i =\theta^{i-1}- \eta\triangledown L(\theta^{i-1}) $$

而随机梯度下降法更快：

损失函数不需要处理训练集所有的数据，随机选取一个例子 $x^n$

$$L=(\hat y^n-(b+\sum w_ix_i^n))^2 $$
$$\theta^i =\theta^{i-1}- \eta\triangledown L^n(\theta^{i-1}) $$

此时不需要像之前那样对所有的数据进行处理，只需要计算某一个例子的损失函数 Ln，就可以赶紧 update 梯度。

对比：

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525113554.png" alt="20220525113554" style="zoom:50%;" />

常规梯度下降法走一步要处理到所有二十个例子，但随机算法此时已经走了二十步（每处理一个例子就更新）。

## Tip 3: Feature Scaling

要求每个 $x_i$ 的分布要搞成一样（大小要差不多），这样 loss 函数 更 smooth，收敛更快（右图直奔中心，左图要拐大弯，lr 要先快后慢不好调）

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525144640.png" alt="20220525144640" style="zoom:50%;" />

为啥左面 loss 画出来是个椭圆？因为：

$$
\frac{\partial L}{\partial \bm{w}} = 2\displaystyle\sum_{i=0}^{n} (\hat{y}^{(i)}-(b+\bm{w}\bm{x}^{(i)}))(-\bm{x}^{(i)})
$$

和 $x$ 的大小有关，$x$ 越大则偏导越大。左图 $x_1$ 的值小，因此 loss 关于 $w_1$ 的偏导小，对等高线横着切一刀，很平滑 ok 合理。而 $w_2$ 偏导大，纵着切一刀，陡的，代表偏导大，合理。

对每个 $x_i$，求出 $m_i, \sigma_i$，归一化有：

$$
x_i^{(n)} := \displaystyle\frac{x_i^{(n)} - m_i}{\sigma_i}
$$

这玩意概率论绝对讲过，standardized random variable，我最开始以为会是 $\mathcal{N}(0,1)$，这是错的，只有正态分布 standardized 之后才是标准正态分布。

## Theory of Gradient Descent

假设 $\boldsymbol{\theta}$ 只有两个参量 $\theta_1, \theta_2$，则在 $(\theta_1,\theta_2)=(a,b)$ 处一阶泰勒展开（多元）：

$$
\begin{aligned}
 L(\boldsymbol{\theta}) &\thickapprox L(a,b) + \displaystyle \frac{\partial L(a,b)}{\partial \theta_1}(\theta_1 - a) + \displaystyle \frac{\partial L(a,b)}{\partial \theta_2}(\theta_2 - b)\\
 &= s + u(\theta_1 - a) + v(\theta_2 - b)\\
 &= s + u\Delta\theta_1 + v\Delta\theta_2\\
 &= s + (u,v) \cdot (\Delta\theta_1,\Delta\theta_2)
\end{aligned}
$$

上面的近似，需要满足 $\theta_1,\theta_2$ 离 $a,b$ 充分近，此时：

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525151838.png" alt="20220525151838" style="zoom:50%;" />

怎样最小？当两个向量反向时最小！v.w = |v||w|cosθ，180° cosθ=-1。所以：

$$
\begin{bmatrix} \Delta\theta_1 \\ \Delta\theta_2 \\\end{bmatrix} = - \eta \begin{bmatrix} u \\ v \\\end{bmatrix}
$$

所以有：

$$
\begin{bmatrix} \theta_1 \\ \theta_2 \\\end{bmatrix} = \begin{bmatrix} a \\ b \\\end{bmatrix} - \eta \begin{bmatrix} u \\ v \\\end{bmatrix}
$$

**只有 circle 小，Taylor 才精确，所以 learning rate 理论上要无穷小，lr 很重要！**

为什么不考虑二次 Taylor 近似？涉及 Hessian 矩阵，矩阵计算太慢了。

## Limitation

- Local minima
- Saddle point
- Plateau

其实 GD 不仅在 local minima 抓瞎，碰到 saddle point，哪怕 $-x^{2}$ 这种二次函数，你在最大值这里梯度也是 0，梯度下降也是不会下降的，这种就有点像临界稳定。更别提 plateau 了。所以梯度为零不一定是最小/极小，甚至可能什么也不是/最大。

然后举的两个例子，帝国时代这个回答的问题有点搞笑，就有人问“我一看就知道是局部最小值/鞍点/最大值”啊，为什么 GD 他就看不出来呢。帝国时代那个例子可以对这些憨憨进行解答。然后我的世界那个例子很奇怪，他拿一个离散的地图模拟 loss 曲面，肯定是出问题的啊，连续可微函数的梯度怎么也不会出现他那个情况啊，看着乐一乐就好。

接下来来到了看不太懂的环节，~~不懂就对了~~。这堆公式好像就算记住了也没什么用，毕竟我不可能深入到 ML theory 了...

## New Optimiz~~ation~~ers for Deep Learning

[An overview of gradient descent optimization algorithms (ruder.io)](https://ruder.io/optimizing-gradient-descent/index.html#challenges) 这个写的蛮清晰的

TLDR: 只要会调包就好，SGDM 和 Adam 赛高！

接下来介绍一些 Optimization 方法<span class="heimu" title="你知道的太多了">（optimizers for 调包）</span>，他们仅在一些情况下可以 converge。

### SGD Momentum

高中学的动量是 $p=mv$，不过在国外大家会把动量视作惯性，那就当是惯性吧。其实就是在更新的时候，算上以前时刻的梯度

$$
\begin{aligned}
\begin{split}
m_t &= \gamma m_{t-1} + \eta \nabla_\theta J( \theta ) \\
\theta &= \theta - m_t
\end{split}
\end{aligned}
$$

加减号可能会换一下，问题不大。

为啥可以呢？举例：

| <img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/B01F776F26236B84CB9FA29A1B03856A.png" alt="B01F776F26236B84CB9FA29A1B03856A" style="zoom: 25%;" />SGD and GD stuck at plateau | 在这个图里，SGD 或者 GD 会停留在 plateau 或者 local minima，不会继续前进，而加入 momentum 之后因为前面时刻一直下降，所以这里有惯性，还要继续下降。 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |

| ![SGD without momentum](https://ruder.io/content/images/2015/12/without_momentum.gif)SGD without momentum | ![SGD with momentum](https://ruder.io/content/images/2015/12/with_momentum.gif)SGD with momentum |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |

在这个图里，SGD 对这种椭圆状的等高线下降得不太好，会一直摇摆。你当然可以 standardized，不过 momentum 会让梯度下降不那么摇摆。momentum 会尽可能让“小球”按照一个方向下降，而不对方向的变化过于敏感。

### Nesterov Accelerated Gradient (NAG)

加上惯性之后，上面的问题好多了，但是可能导致本来能跑到最小值的，结果搞个惯性又飞出去了，这可太憨憨了，怎么办呢，可以让“小球”预知未来。由于 momentum 梯度项可能很小，$\theta$ 主要会移动到 $\theta - \gamma m_{t-1}$ 上，所以只要预测这个位置的梯度再按照这个梯度更新就会好一些：

$$
\begin{aligned}
\begin{split}
m_t &= \gamma m_{t-1} + \eta \nabla_\theta J( \theta - \gamma m_{t-1} ) \\
\theta &= \theta - m_t
\end{split}
\end{aligned}
$$

### Adagrad

接下来开始有人琢磨能不能 adaptive lr 了。adagrad 就出现了，[前面](#adagrad)提到了，这里总结：

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \cdot g_{t}
$$

这个 Adagrad 的分母前面提到了，很憨批，所以大家都不用。Adadelta 和 RMSprop 同时出现，想法相似，用于解决 Adagrad 的问题。

### RMSprop and Adadelta

先说 RMSprop，这是 Hinton 在 Coursera 课程上提出来的，未正式发表。

$$
\begin{aligned}
\begin{split}
E[g^2]_t &= 0.9 E[g^2]_{t-1} + 0.1 g^2_t \\
\theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t}
\end{split}
\end{aligned}
$$

只是分母和 Adagrad 不一样了，RMSprop 的想法是既然把所有的梯度的平方和加起来开根号很憨，那么我就只记录一部分就好了啊，比如我只累加前 window size $w$ 个平方和，然而这样也很憨，采用移动加权平均更高效一点，于是你看到了上面第一行式子，一般权值取 0.9。

为什么叫 RMS 呢？因为这个分母正是 root mean squared (RMS) error criterion of the gradient。也就是说 $\operatorname{RMS}[g]_t = \sqrt{E[g^2]_t + \epsilon}$。

Adadelta 和 RMSprop 到这里是一样的，不过他后来认为 the units in this update (as well as in SGD, Momentum, or Adagrad) do not match。所以他在分子上去掉 $\eta$，换为参数更新 $\Delta\theta$ 的 RMS，有：

$$
\begin{aligned}
\begin{split}
\Delta \theta_t &= - \dfrac{RMS[\Delta \theta]_{t-1}}{RMS[g]_{t}} g_{t} \\
\theta_{t+1} &= \theta_t + \Delta \theta_t
\end{split}
\end{aligned}
$$

结果好像还没有 RMSprop 出名？

### Adam

Adam 的全名是 Adaptive Moment Estimation，他是 SGDM+RMSprop 的组合。首先他把 momentum 的 $m_t$ 移动平均方式进行了小改变：

$$
\begin{aligned}
\begin{split}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
\end{split}
\end{aligned}
$$

然后分母 RMS 项也要有：

$$
\begin{aligned}
\begin{split}
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
\end{split}
\end{aligned}
$$

接着进行 de-biasing：

$$
\begin{aligned}
\begin{split}
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2} \end{split}
\end{aligned}
$$

最终有：

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

关于 2014 年提出来的 Adam 在 2022 年仍然是 NLP models 中的主要 optimizer，正所谓：

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525174744.png" alt="20220525174744" style="zoom:50%;" />

而 CV 中（比如 YoLo）用的是 SGDM，助教 Chien 认为这俩模型占据了半壁江山。还有的比如上面的链接也说 Adam 永远的神，不过另外提的是 SGD 无 M 和 RMSprop。这个链接的中文翻译版本也真有意思，读起来有股机翻味而且 Adam 之后的模型就没翻译了，这说明什么：Adam yyds ！（

## Comparison between Adam and SGDM

那么我因为数学太烂了，只能姑且相信 Chien 说的了，也就是说 SGDM 和 Adam 占据半壁江山。那么为什么还要 improve 他们呢？Chien 对此进行了分析，反正我也不知道他分析的对不对，只能先这么样了。

| ![image-20220525180046627](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205251800670.png)Training Accuracy | ![image-20220525180112605](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205251801448.png)Validation Accuracy |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |

我们发现训练的时候 Adam 好，但是验证的时候反而是 SGDM 好了。然后助教又贴了几个图，然后说 Adam 训练快，但是不稳定；SGDM 训练慢，但是稳定（？），不过是这样吗？谁知道呢。。。

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525180658.png" alt="20220525180658" style="zoom:50%;" />

说 Adam 会陷入 sharp minimum 中，而 train 和 test 的分布一定不同，因此有个 generalization gap，在 sharp minimum 中体现的很明显。而 SGDM 会进入 flat minimum，就还不错。

然后助教提了一堆针对 Adam 和 SGDM 的优化，都不是很满意的样子，我也没仔细看。总结了几个觉得 make sense 的。

## L2 Regularization or Weight Decay?

在讲 Improvement 之前突然发现漏掉了这一部分，感觉并不算 improve，于是单独拎出来了。

大家直到 2017 年才在 Adam 的实现代码中发现一个问题。在 m 和 v 的计算中，应不应该融入 λ 啊？

大家好像也被问住了，终于意识到这两个是有区别的，也不知道哪个好一点，那干脆都留下来吧，m 和 v 中不融入 λ 的叫 weight decay，融入的叫 l2 regularization。

## Improvement

### Shuffling and Curriculum Learning

训练集一般会排成 meaningful order，这可能会让模型学到一些 bias，因此人们会把数据集打乱。就比如刷题的时候可能如果很规律地按一章一章复习刷题可能刷完这一章上一章就忘了，就白给了。所以在训练的时候打乱顺序，全都刷，这样就能学到整个知识形成知识体系了。

然后有人觉得这搞得像是让高一入学学生刷高考卷，有点太扯了。大部分人应该还是先学 A ，刷 A 的题，再学 B，刷 A+B 的题，以此类推，训练难度越来越大，这样有目的性设计训练集的叫做 Curriculum Learning

<img src="https://pic2.zhimg.com/80/v2-794426d707866d311f83ec3b4b1892e1_720w.jpg" alt="img" style="zoom:50%;" />

可以看作是 clean data -> dirty data 的过程。

### Gradient Noise

添加一个服从 $X \sim \mathcal{N}(0, \sigma^2_t)$ 的分布到每个梯度：

$$
g_{t, i} = g_{t, i} + X
$$

随着训练增加，noise 应该越来越小，叫做 annealing（退火）：

$$
\sigma^2_t = \dfrac{\eta}{(1 + t)^\gamma}
$$

说这样能够避免陷入局部极值。

### Warm up

最开始叫做 warm-up，大踏步有全局观，避免陷入局部最优解。接着 annealing，步子越来越收敛，不然不稳定。最后 fine-tuning（和下面的 fine tuning 在某种意义上一样）。

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525200713.png" alt="20220525200713" style="zoom:50%;" />

One Cycle LR 是针对 SGDM 的，而 Adam 也有 RAdam 的 warm up。比较复杂。

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525201223.png" alt="20220525201223" style="zoom:50%;" />

最开始 RAdam 用了 SGDM，这是因为 RAdam 的近似条件在一开始用不了，只能用 SGDM 代替。

### Early Stopping

> According to Geoff Hinton: "*Early stopping (is) beautiful free lunch*" ([NIPS 2015 Tutorial slides](http://www.iro.umontreal.ca/~bengioy/talks/DL-Tutorial-NIPS2015.pdf), slide 63). You should thus always monitor error on a validation set during training and stop (**with some patience**) if your validation error does not improve enough.（直接引用 *An overview of gradient descent optimization algorithms*）

### Look ahead or Look into Future

look ahead 就是说每走几步就要往出发点的方向回去几次：

<img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/20220525201556.png" alt="20220525201556" style="zoom:50%;" />

look into future 在 [NAG](#nesterov-accelerated-gradient-nag) 里面已经提过了，有人在 Adam 类似地实现了 Nadam，公式太复杂就没有放上去。

### Fine Tuning

站在巨人的肩膀上，用人家训练好的模型（pretrained）再训练。

### Dropout, Normalization, Regularization

- Dropout：去掉一些神经元，会对训练准确率造成一点阻碍。
- (Batch) Normalization：看上去人畜无害（？）
    > Batch normalization reestablishes these normalizations for every mini-batch and changes are back-propagated through the operation as well. By making normalization part of the model architecture, we are able to use higher learning rates and pay less attention to the initialization parameters. Batch normalization additionally acts as a **regularizer**, reducing (and sometimes even eliminating) the need for **Dropout**. （直接引用 *An overview of gradient descent optimization algorithms*）

    当然，它种类好多啊
    <img src="https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205251959336.png" alt="image-20220525195934976" style="zoom:50%;" />
- Regularization：会对训练准确率造成一点阻碍。
  - l2 reg
  - weight decay
