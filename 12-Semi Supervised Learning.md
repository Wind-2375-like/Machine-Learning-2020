# Semi-supervised Learning

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [如何理解 inductive learning 与 transductive learning? - 知乎 (zhihu.com)](https://www.zhihu.com/question/68275921)
    2. [MITPress- SemiSupervised Learning.pdf (acad.bg) page 57-59](http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf)
    3. [【机器学习】EM——期望最大（非常详细） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/78311644)
    4. [Self-Training with Selection-by-Rejection (utdallas.edu)](https://personal.utdallas.edu/~mxk055100/publications/icdm-self.pdf)
    
3. **Extra videos:** None

---

## Training Dataset

supervised learning 所有训练数据集都有标签: $\left \{ (x^r, \hat{y}^r)  \right \} ^{R}_{r=1}$.

semi-supervised learning 中训练数据有的有标签, 有的没有标签: $\left \{ (x^r, \hat{y}^r)  \right \} ^{R}_{r=1} \cup \left \{ x^u \right \} ^{R+U}_{r=R+1}$

其中 $U\gg R$.

semi-supervised learning 分为 inductive 和 transductive.

inductive 意思是测试集的数据在训练不可见, 在这里是指 semi-supervised 中训练集的 unlabelled data 不包含 test 数据, 也就是训练时有标签无标签数据一起训练, 但是测试时的数据不会出现在训练集. 因此这需要模型具有很强的泛化能力, 否则测试效果很烂.

transductive 的意思是测试集的数据会作为训练集出现, 只不过训练的时候没有标签罢了. transductive 的定义代表了其本身一定是 semi-supervised 的, 而 inductive 有可能时 supervised 也可以是 semi-supervised 的. transductive 会让模型提前看到测试集的试题, 因此减少了泛化误差.

## Why we need SEMI-supervised?

首先, 得到数据很容易. 图片我就让 machine 自动拍, 文本放个爬虫自己爬. 但是 label 是个 time-consuming 的工作... 尽可能少 label 能不能减轻人类的负担.

但我们又不能说, 干脆摆烂, 只 label 一点点就让机器学那一点就够了. 剩下的不看了. 这些没标记的数据也很有用, 能够提供 distribution 的信息, 帮助我们画出更正确的 decision boundary, 得到更真实地 distribution, 和增加有限噪声提升模型鲁棒性. 但是注意, 这是需要 assumption 的.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/442b5e6ca558857527c906598060e2f53240db600341405cbc269a73fe6db99c.png" />  

人类学习也是用到 semi-supervised 的. 比如练听力, 第一次听到 monkey, dog, cat, 不知道是什么. 别人告诉我之后, 我知道了. 下次我再听到 monkey, dog, cat, 不需要别人告诉, 我就可以自己知道了. 甚至不同人说 monkey, dog, cat, ... 的声音语调稍微有点不一样, 我也能反应过来, 这样就得到更全面的 distribution 和 decision boundary. 甚至听听印式/日式/法式/中式/...英语之后对 accent (noise) 也 robust 了.

## Semi-supervised Learning for Generative Model

logistic regression 中已经知道了 generative model. 我们通过 semi-supervised learning 的 unlabelled data 可以重新影响 $\mu$ 和 $\Sigma$ 的分布.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/0e9fe64d69c85e6a8bf5a2eb4f19f270fd278ac74eb3e46f860f9e4e98810db0.png" />

步骤如下, 这是一个迭代求解的算法. 我认为 ppt 的 $\mu$ 算错了.

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/203129a4bbad3a91e251fec4a87aa580418f7771df6611324873b134db5e0655.png" />

为什么要迭代求解?

- 因为你知道每个类数据的精确分布 => 才能预测任何一个点属于某类的概率.
- 但是只有事先知道任何一个点的标签是什么 => 才能估计每个类数据的分布.

陷入死锁了. 因此需要先初始化假定各数据点的标签 (用 supervised 学出的模型预测 unlabelled data), 推出分布, 再重新估测标签, 重新推出分布, 直到收敛.

这其实就是 EM 算法. 理论上可以收敛, 但是和初始值选择有关. 这里的目标函数是非 convex 的, EM 算法选择采用迭代法求解.

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/4e750502d1bbc18678b64cd8c2dc9ad8f39c83d8a9f12c6ebc64cf17272cf0b6.png" />  

## Low-density Separation Assumption

他的假设是在交界处的 density 较低, 非黑即白, 交界处不存在数据.

### Self-training

$f^*\to \text{pseudo label} \to \text{choose some of them into labelled dataset} \to \text{re-train to get} f^*$

这种方法无法更新 linear regression 的 loss function. 因为:

$$\mathcal{L}(f)=\frac{1}{2}\sum_{r=1}^R(f^*(x^r)-\hat{y}^r)^2+\frac{1}{2}\sum_{r=R+1}^U(f^*(x^r)-\hat{y}^r)^2$$

pseudo label 必然满足 $f^*(x^r) = \hat{y}^r |_{r=R+1}^U$, 因此 loss function 和 pseudo label 无关. $f^*$ 不会更新.

所以想要用 self-training, 就一定看看 pseudo label 的计算, 再计算一下 loss function 是否会因为 pseudo label 发生改变. 比如, 用 NN 作 classification, 算出各个类的概率. pseudo label 一定得是取整之后的 logits (hard label), 而不能是 probability, 否则没有用.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9b3611e45a9fe09c823c220d217c4298b05fca8f71bfc55c4dd845d9f9573c3f.png" />

这也正符合 low density 的假设. 非黑即白. 70% 被分为 A 类, 那它就是 A 类.

### Entropy-based Regularization

上面的方法可能过于武断, 于是 entropy based 做了些改进. entropy-based regularization 没有强行给数据一个 label, 但是他认为 unlabel data 类别 distribution 一定要集中在一个类别上, 这样才算做 "非黑即白"

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/7cb1d3aef74a3df6e4fb29ce4b19a6e79be7309b61c03a1db4e47be513168c0e.png" style="zoom:33%;" />  

于是他对 unlabel data 设计了一个 entropy, 衡量 unlabel data 的 distribution 集中度 (5 替换成 class number):

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/32f5d726e2c5fc5f4b4c2c6af30115514b4fee01fe64f01ab602da0b33171c24.png" style="zoom:50%;" />

$y^u$ 只能为 1 或者 0 才能让 E 最小.

这样 loss function 变成:

$$
\mathcal{L}(f) = \sum_{x^r}C(y^r, \hat{y}^r) + \lambda\sum_{x^u}E(y^u)
$$

后面的项就像惩罚项一样. 前面是让 label data 尽可能准确, 后面是让 unlabel data 的 distribution 尽可能集中.

### Semi-supervised SVM

SVM 是为了 find boundary, 保证

- 最大 margin
- 最小 error

semi supervised 的方法是, 对 unlabel data 穷举所有可能性, 比如 10000 个点的 2 分类, 理论上就有 $2^10000$ 种可能.

对每种可能用 SVM 保证最大 margin, 再选出 error 最小的一种可能.

显然穷举是不可能的. 近似方法:

- 每次只改一个 label
- objective function 提升了就保留, 否则不保留

这只是一个贪心算法了.

## Smoothness Assumption

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8b1ead77c2caf1ecec3bfbaa11688d4ce05799ec6d87c2a2ffb2cc6c473e9f68.png" />  

近朱者赤, 近墨者黑

这个假设有一定合理性, 因为同种 label 的 data 之间哪怕看起来不太一样, 但是可以有很多中间的过渡形态连接他们. 但是不同种 label 之间就不行.

<img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/c5e7f7c578e5a301e3cac94698a3991e60cd62d1c665048c585561c9036d7829.png" />  

### Cluster and Label

<img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/dbea22098d2ba8f6c76b0c2fc6903706a38886019427dc8bee5f2b205637aa74.png" />  

cluster 本身就是种无监督方法, 不需要数据. 不过这里是 semi supervised, 如何利用上已有的 label 在这里并没有仔细提及.

### Graph-based Approach

graph-based 图怎么建呢?

- 直接建, 有的数据自带图结构. 比如超链接的网页文本, 论文的引用关系等.
- 设计一个算法计算数据的相似度, 建图
  - 硬边
    - KNN
    - e-Neighborhood
    <img alt="图 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/50574acd2494b48b56ea80ee69878563122de149a2642765120be19873c689ce.png" />  
  - 软边: weight => $s(x^i, x^j)$

关于软边, 需要着重介绍. 首先是衡量相似度的 RBM:

$$w_{ij} = s(x^i, x^j) = e^{-\gamma\left\| x^i-x^j \right\|^2}$$

RBM 默认所有边全都连上, 如果距离越远, RBM 越接近 0.

接下来要定义 smoothness, 我们希望这个值越小越好:

$$
S = \frac{1}{2} \sum_{i, j \in \text{all data}}w_{ij}(y^i-y^j)^2
$$

这个式子表示, 如果两个点之间距离较近, 相似度比较大, $w_{ij
}$ 比较大的话, 这两个点之间就是由一条 high density path 连接, 那么他们的 label 应该一致才行, 这样值就小. 否则 S 会很大.

而如果两个点距离比较远导致 $w_{ij}$ 比较小怎么办? 这时无论 $y^i, y^j$ 一不一样对 S 来说都没影响了. 所以在这里**应该去掉不相似的边**, 就是 $w_{ij}$ 比较小的. 我感觉这里应该和 KNN 或者 e-Neighborhood 同时使用.

以下图为例, 我们觉得左面的图更 smooth 一些, 而右面的不行.

<img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/0b4ff266b20300da88d2e3ff51b17620ef2bf5fe270de51fd16ccc1d8382bb63.png" />  

还有一个问题, 这里 all data 还包括 unlabel 吗? 包括. 就是用 label data train 完的模型算出 unlabel data 的标签, 然后利用这个 S 衡量 smoothness 当作 loss function 的一部分. 对了, 这和上面 entropy-based regularization 太像了, 这还是一个 regularization 方法.

这里这个 S 可以进一步表示为 $y^{\mathrm{T}}Ly$, 其中 $y=\begin{bmatrix} \cdots & y^i & \cdots & y^j & \cdots \end{bmatrix}^{\mathrm{T}} $, $L=D-W$, 被称为 Graph-Laplacian.

于是 loss function 变成:

$$
\mathcal{L}(f) = \sum_{x^r}C(y^r, \hat{y}^r) + \lambda S
$$

这里的 smoothness 可以放到 output 的任何地方

<img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/654d3737d7a7589ee2208e72d7d976fbb92e1aeebc4bcdd391e01947e365226a.png" />  

## Better Representation

<img alt="图 11" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/a70d12c0deaa9e92b5213906f815391c54a9f735a6c6a3104851e1554b5d5c07.png" />  

就是我们找到本质的 vector, 没有仔细讲, lecturer 用一个例子抖了个机灵, 不明所以. 可以搜索 "杨过剪胡子".
